import argparse
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from data.token_dataset import TokenDataset
from models.vla import Pi0VLA


def code_token_str(idx: int) -> str:
    return f"<|code_{idx:04d}|>"


class TokenVLACollate:
    def __init__(self, processor, tokens_per_stroke: int, include_labels: bool = True):
        self.processor = processor
        self.tokens_per_stroke = tokens_per_stroke
        self.include_labels = include_labels

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        keys = [b["key"] for b in batch]
        token_lists: List[List[List[int]]] = [b["tokens"] for b in batch]

        # Flatten actions/centers if available
        actions = []
        centers = []
        for b in batch:
            if b.get("strokes_centered") is not None and len(b.get("strokes_centered")) > 0:
                actions.append(b["strokes_centered"])
            if b.get("centers") is not None and len(b.get("centers")) > 0:
                centers.append(b["centers"])
        actions_tensor = torch.cat(actions, dim=0) if len(actions) > 0 else None
        centers_tensor = torch.cat(centers, dim=0) if len(centers) > 0 else None

        processed_texts = []
        eos = self.processor.tokenizer.eos_token
        # Dedicated end-of-trajectory token; must already be in tokenizer/model (added in Pi0VLA)
        stop_token = "<|stop|>"
        if self.processor.tokenizer.convert_tokens_to_ids(stop_token) == self.processor.tokenizer.unk_token_id:
            raise ValueError("Stop token <|stop|> missing from tokenizer. Ensure Pi0VLA initializes special tokens before building the dataloader.")
        for key, tokens, image in zip(keys, token_lists, images):
            prompt = f"Write the character"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            flat_tokens = [code for stroke in tokens for code in stroke]
            # Append stop_token then eos to teach VLM to terminate
            code_text = "".join([code_token_str(i) for i in flat_tokens]) + stop_token + eos
            processed_texts.append(chat_prompt + code_text)

        inputs = self.processor(text=processed_texts, images=images, padding=True, return_tensors="pt")

        if self.include_labels:
            inputs["labels"] = inputs["input_ids"].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            # mask everything before first code token; keep stop/eos as targets
            for i, tokens in enumerate(token_lists):
                row = inputs["input_ids"][i]
                flat_tokens = [code for stroke in tokens for code in stroke]
                code_token_ids = [self.processor.tokenizer.convert_tokens_to_ids(code_token_str(t)) for t in flat_tokens]
                code_token_ids = list(set(code_token_ids))
                if code_token_ids:
                    mask = torch.zeros_like(row, dtype=torch.bool)
                    for tid in code_token_ids:
                        mask |= row == tid
                    code_positions = mask.nonzero(as_tuple=True)[0]
                    if len(code_positions) > 0:
                        start_idx = code_positions[0]
                        inputs["labels"][i, :start_idx] = -100
                if pad_id is not None:
                    inputs["labels"][i, row == pad_id] = -100
        else:
            inputs.pop("labels", None)

        return inputs, actions_tensor, centers_tensor


def get_args():
    parser = argparse.ArgumentParser(description="Token-stage training for Pi0 VLA")
    parser.add_argument("--data_root", type=str, default="/data1/jizy/font_out")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--tokens_per_stroke", type=int, default=3)
    parser.add_argument("--num_points", type=int, default=24, help="Number of points per stroke (matches VQ-VAE)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="/data1/jizy/checkpoints/token_stage")
    parser.add_argument("--stage", choices=["lm", "action"], default="lm", help="lm: finetune VLM on tokens; action: freeze VLM and train heads")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--inference_steps", type=int, default=10, help="Flow-matching steps used in action-stage visualization")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU training")
    parser.add_argument("--master_port", type=str, default="12367", help="Master port for DDP")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    return parser.parse_args()


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

def validate_lm_stage(model, processor, dataset, args, epoch_idx: int, rank: int = 0):
    """
    Validation function for the LM stage.
    Samples a batch of data, runs a forward pass, and calculates token prediction accuracy.
    """
    # 仅在主进程打印
    if rank != 0:
        return

    # 获取未封装的 model (处理 DDP 情况)
    model_to_eval = model.module if hasattr(model, "module") else model
    model_to_eval.eval()

    # 随机采样一个 batch (使用 args.batch_size 或固定数量，例如 4)
    num_samples = min(args.batch_size, 8)  # 限制最大验证 batch 为 8 以防 OOM
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    batch_samples = [dataset[i] for i in indices]

    # 复用 TokenVLACollate 来处理 chat 模板、token 拼接和 label masking (-100)
    collate = TokenVLACollate(processor, tokens_per_stroke=args.tokens_per_stroke, include_labels=True)

    try:
        # 预处理数据
        inputs, _, _ = collate(batch_samples)
        device = next(model_to_eval.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            # 直接调用内部的 vlm 获取 logits
            outputs = model_to_eval.vlm(**inputs, return_dict=True)
            logits = outputs.logits  # [B, Seq_Len, Vocab_Size]

        if "labels" in inputs:
            labels = inputs["labels"]
            preds = torch.argmax(logits, dim=-1)

            # 计算准确率: 忽略 label 为 -100 的部分 (prompt 和 padding)
            mask = labels != -100
            # print labels and preds for debugging
            print(f"Labels: {labels[0]}")
            print(f"Preds: {preds[0]}")
            if mask.any():
                correct = (preds[mask] == labels[mask]).sum()
                total = mask.sum()
                acc = (correct.float() / total.float()).item()
                print(f"*** [Epoch {epoch_idx + 1}] LM Stage Validation (n={num_samples}) ***")
                print(f"    Token Accuracy: {acc:.4f}")
                
                # 可选：打印一段解码结果作为直观参考
                # decode_ids = preds[0][mask[0]]
                # print(f"    Sample Pred: {processor.tokenizer.decode(decode_ids, skip_special_tokens=False)}")
            else:
                print(f"[Epoch {epoch_idx + 1}] LM Valid: No valid labels found for accuracy calculation.")
    
    except Exception as e:
        print(f"LM Validation failed: {e}")

    # 恢复训练模式
    model_to_eval.train()

def visualize_action_stage(model, processor, dataset, args, epoch_idx: int, rank: int = 0):
    """Runs an inference-style visualization for the action stage using autoregressive generation."""
    if len(dataset) == 0:
        return

    model_to_use = model.module if hasattr(model, "module") else model
    model_to_use.eval()

    sample_idx = epoch_idx % len(dataset)
    sample = dataset[sample_idx]

    # Compute token accuracy in a teacher-forced pass (masked like training)
    acc_token = float("nan")
    collate_labels = TokenVLACollate(processor, tokens_per_stroke=args.tokens_per_stroke, include_labels=True)
    try:
        inputs_gt, _, _ = collate_labels([sample])
        inputs_gt = {k: v.to(next(model_to_use.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in inputs_gt.items()}
        with torch.no_grad():
            outputs_gt = model_to_use.vlm(**inputs_gt, return_dict=True)
        if "labels" in inputs_gt:
            labels = inputs_gt["labels"]
            print(f"GT Labels: {labels}")
            logits = outputs_gt.logits
            with torch.no_grad():
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                preds = torch.argmax(shift_logits, dim=-1)
                print(f"Preds: {preds}")
                mask = shift_labels != -100

                if mask.any():
                    acc_token = (preds[mask] == shift_labels[mask]).float().mean().item()
                else:
                    acc_token = 0.0
    except Exception as e:
        if rank == 0:
            print(f"Token accuracy computation failed: {e}")

    device = next(model_to_use.parameters()).device
    image = sample["image"]
    key = sample.get("key", "?")

    prompt = f"Write the character"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    max_strokes = 32

    with torch.no_grad():
        generated = model_to_use.generate_action(inputs, max_strokes=max_strokes, steps=args.inference_steps)

    if not generated:
        model_to_use.train()
        return

    pred_actions = torch.cat(generated, dim=0)
    pred_strokes = pred_actions.view(-1, args.num_points, 2).cpu().numpy()

    gt_strokes = []
    gt_actions = sample.get("strokes_centered")
    gt_centers = sample.get("centers")
    if isinstance(gt_actions, torch.Tensor) and gt_actions.numel() > 0:
        gt_actions_view = gt_actions.view(-1, args.num_points, 2)
        if isinstance(gt_centers, torch.Tensor) and gt_centers.numel() > 0:
            gt_centers_view = gt_centers.view(-1, 1, 2)
            gt_strokes = (gt_actions_view + gt_centers_view).cpu().numpy()
        else:
            gt_strokes = gt_actions_view.cpu().numpy()

    image_np = image.permute(1, 2, 0).cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)

    def _plot_strokes(ax, strokes, title):
        if len(strokes) == 0:
            ax.set_title(f"{title} (empty)")
            ax.axis("off")
            return
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k--", alpha=0.3)
        colors = plt.cm.jet(np.linspace(0, 1, max(1, len(strokes))))
        for idx, stroke in enumerate(strokes):
            ax.plot(stroke[:, 0], -stroke[:, 1], color=colors[idx], label=f"Stroke {idx + 1}")
            ax.scatter(stroke[:, 0], -stroke[:, 1], color=colors[idx], s=6)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(image_np)
    axes[0].set_title(f"Input: {key}")
    axes[0].axis("off")
    _plot_strokes(axes[1], gt_strokes, "Ground Truth")
    _plot_strokes(axes[2], pred_strokes, "Predicted")

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(loc="upper right", fontsize=6)
    plt.tight_layout()

    vis_dir = os.path.join(args.save_dir, "vis_action_stage")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, f"epoch_{epoch_idx + 1}.png")
    plt.savefig(vis_path)
    plt.close(fig)

    if rank == 0:
        print(f"Saved action-stage visualization to {vis_path}; token accuracy={acc_token:.4f}")

    model_to_use.train()


def train_worker(rank: int, world_size: int, args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dataset = TokenDataset(
        data_root=args.data_root,
        tokens_per_stroke=args.tokens_per_stroke,
        split="train",
    )
    if not (0 < args.train_ratio <= 1.0):
        raise ValueError("train_ratio must be in (0, 1].")
    if args.train_ratio < 1.0:
        total = len(dataset)
        subset = int(total * args.train_ratio)
        indices = torch.randperm(total)[:subset]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    model = Pi0VLA(
        model_id=args.model_id,
        action_dim=args.num_points * 2,
        num_action_tokens=args.tokens_per_stroke,
        codebook_size=args.codebook_size,
        tokens_per_stroke=args.tokens_per_stroke,
        freeze_vlm=False,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        torch_dtype=dtype,
        device_map={"": rank},
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    model.to(device)

    if args.stage == "action":
        lm_ckpt = os.path.join(args.save_dir, "token_stage_lm.pt")
        if os.path.exists(lm_ckpt):
            model.load_checkpoint(lm_ckpt)
            if rank == 0:
                print(f"Loaded LM-stage checkpoint from {lm_ckpt}")
        elif rank == 0:
            print(f"LM-stage checkpoint not found at {lm_ckpt}; training action stage from scratch.")

    # Stage-specific trainables
    if args.stage == "lm":
        for p in model.action_head.parameters():
            p.requires_grad = False
        for p in model.center_head.parameters():
            p.requires_grad = False
    else:  # action stage: keep VLM trainable to allow fine-tuning
        # If using LoRA, PEFT handles requires_grad (only adapters trainable).
        # If NOT using LoRA, we unfreeze the full VLM.
        if not args.use_lora:
            for p in model.vlm.parameters():
                p.requires_grad = True

    # Some batches may lack action tokens; allow unused params safely
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    processor = model.module.processor
    if processor is None:
        raise RuntimeError("Processor failed to load inside Pi0VLA; cannot proceed with tokenization.")

    # Always include language-model labels so the VLM can continue to receive loss in action stage
    collate = TokenVLACollate(processor, tokens_per_stroke=args.tokens_per_stroke, include_labels=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}") if rank == 0 else loader
        total_loss = 0.0
        total_loss_vlm = 0.0
        total_loss_action = 0.0
        total_loss_center = 0.0

        for step, (inputs, actions, centers) in enumerate(progress):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if args.stage == "lm":
                actions = None
                centers = None
            else:
                if actions is not None:
                    actions = actions.to(device)
                    if actions.numel() == 0:
                        continue  # skip empty batch
                if centers is not None and centers.numel() > 0:
                    centers = centers.to(device)
                else:
                    centers = None

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=dtype, enabled=args.fp16 or args.bf16):
                loss, loss_dict = model(inputs, actions=actions, centers=centers)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_scalar = loss.item() * args.gradient_accumulation_steps
            total_loss += loss_scalar
            total_loss_vlm += loss_dict.get("loss_vlm", 0.0) * args.gradient_accumulation_steps
            total_loss_action += loss_dict.get("loss_action", 0.0) * args.gradient_accumulation_steps
            total_loss_center += loss_dict.get("loss_center", 0.0) * args.gradient_accumulation_steps
            if rank == 0:
                progress.set_postfix({"loss": loss_scalar, "loss_vlm": loss_dict.get("loss_vlm", 0.0), "loss_action": loss_dict.get("loss_action", 0.0), "loss_center": loss_dict.get("loss_center", 0.0)})

        total_loss_tensor = torch.tensor(total_loss, device=device)
        count_tensor = torch.tensor(len(loader), device=device, dtype=torch.float32)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / torch.clamp(count_tensor, min=1.0)).item()
        avg_loss_vlm = torch.tensor(total_loss_vlm, device=device)
        avg_loss_action = torch.tensor(total_loss_action, device=device)
        avg_loss_center = torch.tensor(total_loss_center, device=device)
        dist.all_reduce(avg_loss_vlm, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_loss_action, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_loss_center, op=dist.ReduceOp.SUM)
        avg_loss_vlm = (avg_loss_vlm / torch.clamp(count_tensor, min=1.0)).item()
        avg_loss_action = (avg_loss_action / torch.clamp(count_tensor, min=1.0)).item()
        avg_loss_center = (avg_loss_center / torch.clamp(count_tensor, min=1.0)).item()

        if rank == 0:
            ckpt = os.path.join(args.save_dir, f"token_stage_{args.stage}.pt")
            model.module.save_checkpoint(ckpt)
            print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f}, avg_loss_vlm={avg_loss_vlm:.4f}, avg_loss_action={avg_loss_action:.4f}, avg_loss_center={avg_loss_center:.4f}, saved {ckpt}")
            if args.stage == "lm":
                validate_lm_stage(model, processor, dataset, args, epoch, rank)
            if args.stage == "action":
                visualize_action_stage(model, processor, dataset, args, epoch, rank)

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

if __name__ == "__main__":
    main()
