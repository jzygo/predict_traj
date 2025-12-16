import argparse
import os
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from config import PipelineConfig
from models.vae import StrokeVAE
from models.dit import DiT


def build_cfg(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
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
    return cfg



def load_checkpoint(vae: torch.nn.Module, dit: torch.nn.Module, path: str, device: torch.device) -> None:
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    vae.load_state_dict(state["vae"])
    dit.load_state_dict(state["dit"])
    print(f"Loaded checkpoint from {path}")


def pick_lengths(eos_probs: torch.Tensor, max_strokes: int) -> torch.Tensor:
    # eos_probs: (B, max_strokes+1)
    lengths = torch.argmax(eos_probs, dim=1)
    return lengths.clamp(max=max_strokes)


def visualize(image: Image.Image, trajs: List[torch.Tensor], output_path: str) -> None:
    w, h = image.size

    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # Use a colormap to distinguish strokes
    colors = plt.cm.jet(np.linspace(0, 1, len(trajs)))

    for i, traj in enumerate(trajs):
        # traj: (num_points, 2) in [-1, 1]
        t = traj.numpy()
        # Map to image coordinates
        x = (t[:, 0] + 1) / 2 * w
        y = (t[:, 1] + 1) / 2 * h

        plt.plot(x, y, color=colors[i], linewidth=2)
        # Mark start point
        plt.scatter(x[0], y[0], color=colors[i], s=30, marker='o', edgecolors='white')

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")


def pad_and_resize(image: Image.Image, target_size: int) -> Image.Image:
    w, h = image.size
    if w == h:
        return image.resize((target_size, target_size))
    
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    new_img.paste(image, (0, 0))
    # save new_img for debug
    new_img.save("/home/jizy/project/video_tokenizer/debug.png")
    return new_img.resize((target_size, target_size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference script: image -> stroke trajectories")
    parser.add_argument("--image_path", type=str, default="/home/jizy/project/video_tokenizer/data/ding.png", help="Path to input image")
    parser.add_argument("--ckpt", type=str, default="/data1/jizy/checkpoints/ckp.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="/home/jizy/project/video_tokenizer/output.pt", help="Optional path to save trajectories (.pt)")

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
    args = parser.parse_args()

    cfg = build_cfg(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image_path).convert("RGB")
    image = pad_and_resize(image, cfg.dims.image_size)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    vae = StrokeVAE(cfg.dims, cfg.vae).to(device).eval()
    dit = DiT(cfg.dims, cfg.dit).to(device).eval()

    load_checkpoint(vae, dit, args.ckpt, device)

    with torch.no_grad():
        latents, eos_probs, _ = dit.infer(image_tensor, max_len=cfg.dims.max_strokes)
        latents = latents[0]
        trajs = vae.decode(latents).cpu()  # (N, P, 2)

    if args.output:
        vis_path = os.path.splitext(args.output)[0] + ".png"
        if len(trajs) > 0:
            visualize(image, trajs, vis_path)
    else:
        for idx, trajs in enumerate(trajs):
            print(f"Sample {idx}: {len(trajs)} strokes")
            for t_id, traj in enumerate(trajs):
                print(f"  Stroke {t_id}: shape {tuple(traj.shape)}")


if __name__ == "__main__":
     main()
