import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List
import concurrent.futures
import multiprocessing

import torch
from tqdm import tqdm

# Ensure project src/ is on sys.path for relative imports when run as a script
CURRENT_DIR = Path(__file__).resolve()
SRC_ROOT = CURRENT_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.vqvae import StrokeVQVAE
from data.traj_utils import resample_stroke


def load_strokes(font_dir: str, font_name: str):
    stroke_path = os.path.join(font_dir, f"{font_name}_strokes.pkl")
    if not os.path.exists(stroke_path):
        return {}
    with open(stroke_path, "rb") as f:
        return pickle.load(f)


def _batch_encode(strokes: List, model: StrokeVQVAE, device: torch.device, num_points: int, batch_size: int = 8192):
    tokens_per_char = []
    centers_per_char = []
    recon_inputs = []

    # Precompute centers and resampled strokes on CPU, then batch to GPU once per chunk
    centers = []
    resampled_list = []
    for stroke in strokes:
        stroke_tensor = torch.tensor(stroke, dtype=torch.float32)
        center = stroke_tensor.mean(dim=0) if stroke_tensor.numel() > 0 else torch.zeros(2)
        centered = stroke_tensor - center
        resampled = resample_stroke(centered, num_points=num_points)
        centers.append(center)
        resampled_list.append(resampled)

    total = len(resampled_list)
    if total == 0:
        return tokens_per_char, centers_per_char, recon_inputs

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = torch.stack(resampled_list[start:end]).to(device)
        with torch.no_grad():
            indices = model.encode(batch).cpu()
        for i in range(end - start):
            tokens_per_char.append(indices[i].tolist())
            centers_per_char.append(centers[start + i].tolist())
            recon_inputs.append(resampled_list[start + i].tolist())

    return tokens_per_char, centers_per_char, recon_inputs


def process_font(font_dir: str, model: StrokeVQVAE, device: torch.device, save_dir: str):
    font_name = os.path.basename(font_dir)
    strokes_data = load_strokes(font_dir, font_name)
    if not strokes_data:
        return

    output: Dict[str, Dict[str, List]] = {}
    max_workers = max(1, multiprocessing.cpu_count())
    # Batch over each character to fully utilize GPU; tqdm over keys for visibility
    for key, strokes in tqdm(strokes_data.items(), desc=f"{font_name}"):
        tokens_per_char, centers_per_char, recon_inputs = _batch_encode(
            strokes, model, device, model.cfg.num_points, batch_size=8192
        )
        output[key] = {
            "tokens": tokens_per_char,
            "centers": centers_per_char,
            "strokes_centered": recon_inputs,
        }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, f"{font_name}_tokens.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved tokens to {save_path} ({len(output)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Precompute stroke tokens per font")
    parser.add_argument("--data_root", type=str, default="/data1/jizy/font_out", help="Root of processed fonts")
    parser.add_argument("--vqvae_ckpt", type=str, default="/home/jizy/project/video_tokenizer/srcnew/checkpoints/vqvae/vqvae.pt", help="Path to trained VQ-VAE checkpoint")
    parser.add_argument(
        "--save_root", type=str, default=None, help="Directory to store token pickles (default: font dir)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrokeVQVAE.load(args.vqvae_ckpt, map_location=device)
    model.to(device)
    model.eval()

    font_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    for font in font_dirs:
        font_dir = os.path.join(args.data_root, font)
        save_dir = font_dir if args.save_root is None else os.path.join(args.save_root, font)
        process_font(font_dir, model, device, save_dir)


if __name__ == "__main__":
    main()
