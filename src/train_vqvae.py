import argparse
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.stroke_dataset import StrokeDataset
from models.vqvae import StrokeVQVAE, VQVAEConfig


def get_free_gpus(threshold_mb: int = 512):
    try:
        import subprocess

        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used = [int(x) for x in result.strip().split('\n')]
        return [str(i) for i, mem in enumerate(memory_used) if mem < threshold_mb]
    except Exception as e:
        print(f"Could not query GPUs: {e}")
        return []


def get_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE to tokenize strokes")
    parser.add_argument("--data_root", type=str, default="/data1/jizy/font_out", help="Root directory of processed fonts")
    parser.add_argument("--save_dir", type=str, default="checkpoints/vqvae", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_points", type=int, default=24, help="Points per stroke after resampling")
    parser.add_argument("--tokens_per_stroke", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=192)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume a saved VQ-VAE checkpoint")
    parser.add_argument("--ddp", action="store_true", help="Enable multi-GPU DistributedDataParallel training")
    parser.add_argument("--master_port", type=str, default="12365", help="Master port for DDP")
    return parser.parse_args()


def build_config(args) -> VQVAEConfig:
    return VQVAEConfig(
        num_points=args.num_points,
        tokens_per_stroke=args.tokens_per_stroke,
        codebook_size=args.codebook_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        commitment_cost=args.commitment_cost,
        ema_decay=args.ema_decay,
    )


def train_worker(rank: int, world_size: int, args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    cfg = build_config(args)

    if args.resume and os.path.exists(args.resume):
        model = StrokeVQVAE.load(args.resume, map_location=device)
        if rank == 0:
            print(f"Loaded VQ-VAE from {args.resume}")
    else:
        model = StrokeVQVAE(cfg)

    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    train_ds = StrokeDataset(
        data_root=args.data_root,
        split="train",
        val_ratio=args.val_ratio,
        num_points=args.num_points,
    )
    val_ds = StrokeDataset(
        data_root=args.data_root,
        split="val",
        val_ratio=args.val_ratio,
        num_points=args.num_points,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    print(f"Rank {rank}: Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.fp16)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") if rank == 0 else train_loader
        for batch in pbar:
            strokes = batch["stroke"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                out = model(strokes)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if rank == 0:
                pbar.set_postfix({"loss": loss.item(), "pplx": out["perplexity"].item()})

        running_tensor = torch.tensor(running, device=device)
        count_tensor = torch.tensor(len(train_loader), device=device, dtype=torch.float32)
        dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        train_loss = (running_tensor / torch.clamp(count_tensor, min=1.0)).item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                strokes = batch["stroke"].to(device)
                out = model(strokes)
                val_loss += out["loss"].item()
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_count_tensor = torch.tensor(len(val_loader), device=device, dtype=torch.float32)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_tensor / torch.clamp(val_count_tensor, min=1.0)).item()

        if rank == 0:
            ckpt_path = os.path.join(args.save_dir, f"vqvae.pt")
            model.module.save(ckpt_path)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, saved={ckpt_path}"
            )

    dist.destroy_process_group()


def main():
    args = get_args()

    if args.ddp:
        free_gpus = get_free_gpus()
        if not free_gpus:
            print("No free GPUs found for DDP; falling back to single GPU.")
            args.ddp = False
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free_gpus)
            world_size = len(free_gpus)
            print(f"Launching DDP with GPUs: {free_gpus}")
            mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = build_config(args)

    if args.resume and os.path.exists(args.resume):
        model = StrokeVQVAE.load(args.resume, map_location=device)
        print(f"Loaded VQ-VAE from {args.resume}")
    else:
        model = StrokeVQVAE(cfg)

    model.to(device)

    train_ds = StrokeDataset(
        data_root=args.data_root,
        split="train",
        val_ratio=args.val_ratio,
        num_points=args.num_points,
    )
    val_ds = StrokeDataset(
        data_root=args.data_root,
        split="val",
        val_ratio=args.val_ratio,
        num_points=args.num_points,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.fp16)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            strokes = batch["stroke"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                out = model(strokes)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix({"loss": loss.item(), "pplx": out["perplexity"].item()})

        train_loss = running / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                strokes = batch["stroke"].to(device)
                out = model(strokes)
                val_loss += out["loss"].item()
        val_loss = val_loss / max(1, len(val_loader))

        ckpt_path = os.path.join(args.save_dir, f"vqvae.pt")
        model.save(ckpt_path)
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, saved={ckpt_path}"
        )


if __name__ == "__main__":
    main()
