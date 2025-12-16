import argparse
import os
import sys
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from config import PipelineConfig
from data.font_dataset import FontDataset
from data.collate_utils import pad_stroke_batch
from models.vae import StrokeVAE
from models.dit import DiT


def collate_with_params(batch, num_points: int, max_strokes: int):
    return pad_stroke_batch(batch, num_points=num_points, max_strokes=max_strokes)


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.data_root = args.data_root
    cfg.device = args.device

    cfg.dims.num_points = args.num_points
    cfg.dims.latent_dim = args.latent_dim
    cfg.dims.model_dim = args.model_dim
    cfg.dims.image_size = args.image_size
    cfg.dims.max_strokes = args.max_strokes

    cfg.vae.hidden_dim = args.vae_hidden
    cfg.vae.dropout = args.dropout
    cfg.vae.num_layers = args.vae_layers
    cfg.vae.latent_dim = args.latent_dim

    cfg.dit.patch_size = args.patch_size
    cfg.dit.depth = args.dit_depth
    cfg.dit.num_heads = args.dit_heads
    cfg.dit.dropout = args.dropout
    cfg.dit.use_pretrained_cnn = args.use_pretrained_cnn

    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.num_workers = args.num_workers
    cfg.train.stage = args.stage
    cfg.train.resume = args.resume

    cfg.dist.ddp = args.ddp
    cfg.dist.min_free_ratio = args.min_free_ratio
    return cfg


def unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def save_checkpoint(vae: nn.Module, dit: nn.Module, optim: torch.optim.Optimizer, epoch: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "vae": unwrap_module(vae).state_dict(),
        "dit": unwrap_module(dit).state_dict(),
        "optim": optim.state_dict(),
    }
    path = os.path.join(save_dir, f"ckp.pt")
    torch.save(state, path)


def build_dataloader(
    cfg: PipelineConfig,
    split: str,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = FontDataset(data_root=cfg.data_root, split=split)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=split == "all")

    collate_fn = partial(collate_with_params, num_points=cfg.dims.num_points, max_strokes=cfg.dims.max_strokes)

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(sampler is None and split == "all"),
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def pack_valid_latents(
    latents_valid: torch.Tensor,
    lengths: torch.Tensor,
    max_strokes: int,
) -> torch.Tensor:
    """Pack flattened per-stroke latents back to padded (B, max_strokes, D).

    NOTE: This is intended for teacher/target tensors (often detached), so we
    keep it simple and allow in-place assignment.
    """

    B = lengths.size(0)
    D = latents_valid.size(-1)
    out = torch.zeros(B, max_strokes, D, device=latents_valid.device, dtype=latents_valid.dtype)
    offset = 0
    for b in range(B):
        n = int(lengths[b].item())
        if n <= 0:
            continue
        n = min(n, max_strokes)
        out[b, :n] = latents_valid[offset : offset + n]
        offset += n
    return out


def eos_labels(lengths: torch.Tensor, max_strokes: int) -> torch.Tensor:
    B = lengths.size(0)
    labels = torch.zeros(B, max_strokes + 1, device=lengths.device)
    for b in range(B):
        idx = min(int(lengths[b].item()), max_strokes)
        labels[b, idx:] = 1.0
    return labels


def stack_valid(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for b in range(tensor.size(0)):
        if lengths[b] == 0:
            continue
        chunks.append(tensor[b, : lengths[b]])
    if len(chunks) == 0:
        return torch.zeros(0, *tensor.shape[2:], device=tensor.device)
    return torch.cat(chunks, dim=0)


def train_one_epoch(
    vae: StrokeVAE,
    dit: DiT,
    optim: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    sampler: Optional[DistributedSampler],
    is_main: bool,
    cfg: PipelineConfig,
) -> Dict[str, float]:
    vae.train()
    dit.train()

    logs: Dict[str, float] = {"loss": 0.0, "vae": 0.0, "kl": 0.0, "latent": 0.0, "eos": 0.0}
    if sampler is not None:
        sampler.set_epoch(epoch)
    from tqdm import tqdm
    process_bar = tqdm(loader, desc=f"Epoch {epoch} Training", disable=not is_main)
    for batch in process_bar:
        images = batch["images"].to(device)
        strokes = batch["strokes"].to(device)
        lengths = batch["stroke_lengths"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        B, S, P, _ = strokes.shape
        max_strokes = cfg.dims.max_strokes
        optim.zero_grad()

        # VAE per stroke
        valid_strokes = stack_valid(strokes, lengths)
        if valid_strokes.numel() == 0:
            continue

        # Stage 1: train VAE with reconstruction/KL only.
        # Stage 2: keep VAE fixed (teacher) and do NOT run it with grad.
        if cfg.train.stage == 1:
            recon, kl, recon_loss, _ = vae(valid_strokes)
        else:
            recon_loss = torch.tensor(0.0, device=device)
            kl = torch.tensor(0.0, device=device)

        # Teacher latents from VAE encoder (detached): DiT should match VAE.
        # Keep detached so gradients never push VAE to chase DiT/Flow.
        with torch.no_grad():
            mu_valid, _ = unwrap_module(vae).encode(valid_strokes)
        latent_targets_main = pack_valid_latents(mu_valid, lengths, max_strokes=max_strokes)

        # DiT prediction (now returns 2 values: latent, eos_logits)
        # Pass target_latents for teacher forcing
        latent_pred, eos_logits = dit(images, latent_targets_main, attn_mask)
        
        # latent_pred is (B, max_strokes, D), aligned with latent_targets_main
        # No need to drop first query or shift, as the model handles autoregressive shift internally

        # Masked latent loss
        mask = torch.arange(max_strokes, device=device)[None, :] < lengths[:, None]
        latent_loss = ((latent_pred - latent_targets_main).pow(2).sum(-1) * mask.float()).sum() / mask.float().clamp(min=1).sum()

        # EOS loss
        # eos_logits is (B, max_strokes)
        # eos_target is (B, max_strokes + 1) -> slice to max_strokes
        eos_target = eos_labels(lengths, max_strokes)[:, :max_strokes]
        eos_loss = nn.functional.binary_cross_entropy_with_logits(eos_logits, eos_target, reduction="none")
        eos_loss = eos_loss.mean()
        # Loss weights (keep simple & explicit)
        target_kl_w = 1e-4  # 建议由 0.1 降低到 1e-4 或 1e-3
        warmup_epochs = 5
        if epoch < warmup_epochs:
            kl_w = target_kl_w * (epoch / warmup_epochs)
        else:
            kl_w = target_kl_w

        if cfg.train.stage == 1:
            total = recon_loss + kl_w * kl + latent_loss + eos_loss
        else:
            # Stage 2: Only train DiT (VAE fixed as teacher) + continuity regularization
            total = eos_loss + latent_loss

        total.backward()
        nn.utils.clip_grad_norm_(list(vae.parameters()) + list(dit.parameters()), 1.0)
        optim.step()

        logs["loss"] += total.item()
        logs["vae"] += recon_loss.item()
        logs["kl"] += kl.item()
        logs["latent"] += latent_loss.item()
        logs["eos"] += eos_loss.item()
        # update loss to progress bar
        process_bar.set_postfix({k: f"{logs[k]/(process_bar.n+1):.4f}" for k in logs})
    process_bar.close()

    n = max(1, len(loader))
    for k in logs:
        logs[k] /= n
    return logs

def save_epoch_visualization(
    vae: StrokeVAE,
    dit: DiT,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    save_dir: str,
    cfg: PipelineConfig
) -> None:
    vae.eval()
    dit.eval()
    
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    B = batch["images"].size(0)
    idx = torch.randint(0, B, (1,)).item()
    
    image = batch["images"][idx].to(device)
    strokes = batch["strokes"][idx].to(device)
    key = batch["key"][idx]
    font_name = batch["font_name"][idx]
    length = int(batch["stroke_lengths"][idx].item())
    
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    gt_strokes = strokes[:length].cpu().numpy()
    
    valid_strokes = strokes[:length]
    
    # VAE encode-decode reconstruction
    if length > 0:
        with torch.no_grad():
            # VAE encode to latent and decode back
            mus = vae.infer(valid_strokes)
            vae_recon = unwrap_module(vae).decode(mus).cpu().numpy()
    else:
        vae_recon = np.zeros((0, cfg.dims.num_points, 2))

    # DiT + Flow predictions
    with torch.no_grad():
        latent_pred, eos_probs, num_strokes = dit.infer(image.unsqueeze(0))
        latent_pred = latent_pred[0]  # [num_strokes, latent_dim]
        eos_probs = eos_probs[0]    # [num_strokes]
        
        # Use the predicted length from infer
        pred_length = min(num_strokes, cfg.dims.max_strokes)

        print(f"Epoch {epoch} Visualization: GT length={length}, Predicted length={pred_length}")
        
        # Use all generated latents
        stroke_latents = latent_pred[:pred_length]
        
        if stroke_latents.size(0) > 0:
            dit_strokes = unwrap_module(vae).decode(stroke_latents).cpu().numpy()
        else:
            dit_strokes = np.zeros((0, cfg.dims.num_points, 2))

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Input Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Input Image, {font_name}, {key}", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    def plot_strokes(ax, strokes_data, title, background_img=None):
        """
        绘制笔画轨迹，可选添加背景图像
        
        Args:
            ax: matplotlib axis
            strokes_data: 笔画数据
            title: 子图标题
            background_img: 背景图像 (可选)
        """
        # 如果提供了背景图像，先显示它
        if background_img is not None:
            # extent 参数将图像映射到 [-1, 1] 的坐标范围
            ax.imshow(background_img, extent=[-1, 1, -1, 1], alpha=0.5)  # alpha=0.5 半透明
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制边框
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', alpha=0.5, linewidth=2)
        
        if len(strokes_data) > 0:
            colors = plt.cm.jet(np.linspace(0, 1, len(strokes_data)))
            for i, s in enumerate(strokes_data):
                # 加粗线条
                ax.plot(s[:, 0], -s[:, 1], color=colors[i], linewidth=3, alpha=0.9)
                # 标记起点
                ax.scatter(s[0, 0], -s[0, 1], color=colors[i], s=50, 
                          marker='o', edgecolors='white', linewidths=2, zorder=5)
                # 标记轨迹点
                ax.scatter(s[:, 0], -s[:, 1], color=colors[i], s=5, zorder=4)
    
    # 2. Ground Truth (带背景)
    plot_strokes(axes[1], gt_strokes, "Ground Truth", background_img=img_np)
    
    # 3. VAE Reconstruction (带背景)
    plot_strokes(axes[2], vae_recon, "VAE Recon\n(encode→decode)", background_img=img_np)

    # 4. DiT + VAE Decoder (带背景)
    plot_strokes(axes[3], dit_strokes, "DiT + VAE Decoder", background_img=img_np)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_epoch_{epoch}.png"), dpi=150, bbox_inches='tight')
    plt.close()

def available_gpus(min_free_ratio: float = 0.7) -> list:
    if not torch.cuda.is_available():
        return []
    free_ids = []
    for idx in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(idx)
            if total == 0:
                continue
            if float(free) / float(total) >= min_free_ratio:
                free_ids.append(idx)
        except Exception:
            continue
    return free_ids


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_worker(rank: int, world_size: int, cfg: PipelineConfig) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    vae = StrokeVAE(cfg.dims, cfg.vae).to(device)
    dit = DiT(cfg.dims, cfg.dit).to(device)

    if cfg.train.resume:
        if rank == 0:
            print(f"Loading checkpoint from {cfg.train.resume}")
        try:
            ckpt = torch.load(cfg.train.resume, map_location=device)
            try:
                vae.load_state_dict(ckpt["vae"])
            except RuntimeError:
                if rank == 0:
                    print("Failed to load VAE from checkpoint.")
            try:
                dit.load_state_dict(ckpt["dit"])
            except RuntimeError:
                if rank == 0:
                    print("Failed to load DiT from checkpoint.")
        except Exception as e:
            if rank == 0:
                print(f"Failed to load checkpoint: {e}")


    if cfg.train.stage == 2:
        params = list(dit.parameters())
    else:
        params = list(vae.parameters()) + list(dit.parameters())

    vae = nn.parallel.DistributedDataParallel(vae, device_ids=[rank] if torch.cuda.is_available() else None, output_device=rank if torch.cuda.is_available() else None, find_unused_parameters=True)
    dit = nn.parallel.DistributedDataParallel(dit, device_ids=[rank] if torch.cuda.is_available() else None, output_device=rank if torch.cuda.is_available() else None, find_unused_parameters=True)

    optim = torch.optim.AdamW(
        params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.train.epochs)

    train_loader = build_dataloader(
        cfg,
        split="all",
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    history = []
    for epoch in range(cfg.train.epochs):
        logs = train_one_epoch(vae, dit, optim, train_loader, device, epoch, train_loader.sampler, rank == 0, cfg=cfg)
        scheduler.step()
        if rank == 0:
            history.append(logs)
            print(
                f"Epoch {epoch}: loss={logs['loss']:.4f} vae={logs['vae']:.4f} latent={logs['latent']:.4f} eos={logs['eos']:.4f}"
            )
            save_checkpoint(vae, dit, optim, epoch, cfg.save_dir)
            save_epoch_visualization(unwrap_module(vae), unwrap_module(dit), train_loader, device, epoch, cfg.save_dir, cfg)
    # if rank == 0 and cfg.vis_out:
    #     save_vis(history, cfg.vis_out)

    cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint training for image-to-stroke pipeline")
    parser.add_argument("--data_root", type=str, default="/data1/jizy/font_out")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_points", type=int, default=24)
    parser.add_argument("--latent_dim", type=int, default=48)
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_strokes", type=int, default=32)

    parser.add_argument("--vae_hidden", type=int, default=768)
    parser.add_argument("--vae_layers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--dit_depth", type=int, default=16)
    parser.add_argument("--dit_heads", type=int, default=12)
    parser.add_argument("--use_pretrained_cnn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--stage", type=int, default=1, help="Training stage: 1 or 2")
    # parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--resume", type=str, default="/data1/jizy/checkpoints/ckp.pt", help="Path to checkpoint to resume from")

    parser.add_argument("--ddp", action="store_true", help="Enable DDP and auto-pick free GPUs")
    parser.add_argument("--min_free_ratio", type=float, default=0.7, help="Minimum free memory ratio to select GPU")
    parser.add_argument("--save_dir", type=str, default="/data1/jizy/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--vis_out", type=str, default="", help="Optional path to save loss curve png")

    args = parser.parse_args()
    cfg = build_config_from_args(args)
    cfg.save_dir = args.save_dir
    cfg.vis_out = args.vis_out

    if cfg.dist.ddp:
        free = available_gpus(min_free_ratio=cfg.dist.min_free_ratio)
        if len(free) < 1:
            print("No free GPUs detected; falling back to single process.")
            cfg.dist.ddp = False
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in free)
            world_size = len(free)
            mp.spawn(run_worker, nprocs=world_size, args=(world_size, cfg))
            return
    
if __name__ == "__main__":
    main()
