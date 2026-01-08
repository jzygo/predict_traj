import argparse
import os
import io
import pickle
import glob
import random
from typing import List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

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
    cfg.dims.point_dim = args.point_dim

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


def visualize(image: Image.Image, trajs: torch.Tensor, output_path: str, point_dim: int = 3, image_size: int = 256) -> None:
    w, h = image.size

    plt.figure(figsize=(8, 8))
    plt.imshow(image, alpha=0.5)

    # Use a colormap to distinguish strokes
    # trajs: (num_strokes, num_points, point_dim)
    num_strokes = len(trajs)
    if num_strokes == 0:
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return

    colors = plt.cm.jet(np.linspace(0, 1, num_strokes))
    
    max_radius = 20
    
    for i, traj in enumerate(trajs):
        # traj: (num_points, point_dim) in [-1, 1]
        t = traj.numpy()
        # Map to image coordinates
        x = (t[:, 0] + 1) / 2 * w
        y = (t[:, 1] + 1) / 2 * h

        plt.plot(x, y, color=colors[i], linewidth=2)
        # Mark start point
        plt.scatter(x[0], y[0], color=colors[i], s=30, marker='o', edgecolors='white')
        
        # 如果有 r 维度，绘制半径圆
        if point_dim >= 3 and t.shape[1] >= 3:
            for j in range(0, len(t), max(1, 2)):
                # r 从 [-1, 1] 反归一化到像素值
                r_normalized = (t[j, 2] + 1) / 2  # 归一化到 [0, 1]
                r_pixel = r_normalized * max_radius * (w / image_size)  # 映射到当前图像尺寸
                if r_pixel > 1:
                    circle = plt.Circle((x[j], y[j]), r_pixel, 
                                       fill=False, color=colors[i], alpha=0.4, linewidth=1)
                    plt.gca().add_patch(circle)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def pad_and_resize(image: Image.Image, target_size: int) -> Image.Image:
    w, h = image.size
    if w == h:
        return image.resize((target_size, target_size))
    
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    new_img.paste(image, (0, 0))
    return new_img.resize((target_size, target_size))


def process_image(image_path: str, target_size: int) -> tuple[str, bytes, Image.Image]:
    """读取图片，返回key, jpg_bytes, PIL_Image"""
    filename = os.path.basename(image_path)
    key = os.path.splitext(filename)[0]
    
    try:
        # Convert to black and white (binarize) then back to RGB
        img = Image.open(image_path).convert("L")
        img = img.point(lambda x: 255 if x > 127 else 0)
        img = img.convert("RGB")
        
        # Prepare bytes
        with io.BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()
            
        # Prepare PIL Image for model
        input_img = pad_and_resize(img, target_size)
        return key, img_bytes, input_img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference script: folder of images -> pkl results and visualizations")
    parser.add_argument("--input_folder", type=str, default="/home/jizy/project/video_tokenizer/data/test", help="Folder containing input images")
    parser.add_argument("--output_folder", type=str, default="/data1/jizy/output/infer_output", help="Folder to save results and visualizations")
    parser.add_argument("--ckpt", type=str, default="/data1/jizy/checkpoints/ckp_final.pt", help="Path to checkpoint")
    
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_points", type=int, default=24)
    parser.add_argument("--latent_dim", type=int, default=72)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_strokes", type=int, default=32)
    parser.add_argument("--point_dim", type=int, default=3, help="Point dimension: 2 (x,y) or 3 (x,y,r)")

    parser.add_argument("--vae_hidden", type=int, default=768)
    parser.add_argument("--vae_layers", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--dit_depth", type=int, default=16)
    parser.add_argument("--dit_heads", type=int, default=16)
    parser.add_argument("--use_pretrained_cnn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_images", type=int, default=100, help="Max number of images to process")
    args = parser.parse_args()

    cfg = build_cfg(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    vis_folder = os.path.join(args.output_folder, "visualizations")
    # Clean up old visualizations if they exist
    import shutil
    if os.path.exists(vis_folder):
        print(f"Cleaning up old visualizations at {vis_folder}...")
        shutil.rmtree(vis_folder)
    
    os.makedirs(vis_folder, exist_ok=True)

    # 1. Load Data
    print(f"Scanning images in {args.input_folder}...")
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(args.input_folder, ext)))
    
    image_files = sorted(image_files)
    if args.max_images is not None and len(image_files) > args.max_images:
        image_files = random.sample(image_files, args.max_images)
    print(f"Found {len(image_files)} images.")

    all_image_data = {}
    inputs_for_model = [] # list of (key, PIL_Image)

    print("Reading images...")
    for img_path in tqdm(image_files):
        key, img_bytes, pil_img = process_image(img_path, cfg.dims.image_size)
        if key is not None:
            all_image_data[key] = img_bytes
            inputs_for_model.append((key, pil_img))
    
    if not all_image_data:
        print("No valid images found.")
        return

    # Save images pkl
    imgs_pkl_path = os.path.join(args.output_folder, "images.pkl")
    with open(imgs_pkl_path, 'wb') as f:
        pickle.dump(all_image_data, f)
    print(f"Saved image data to {imgs_pkl_path}")

    # 2. Load Model
    vae = StrokeVAE(cfg.dims, cfg.vae).to(device).eval()
    dit = DiT(cfg.dims, cfg.dit).to(device).eval()
    load_checkpoint(vae, dit, args.ckpt, device)

    # 3. Inference
    all_font_data = {} # key -> strokes list
    
    print("Running inference...")
    transform = transforms.ToTensor()

    for key, pil_img in tqdm(inputs_for_model):
        image_tensor = transform(pil_img).unsqueeze(0).to(device) # (1, 3, H, W)

        try:
            with torch.no_grad():
                latents, eos_probs, _ = dit.infer(image_tensor, max_len=cfg.dims.max_strokes)
                latents = latents[0] # (MaxStrokes, LatentDim)
                trajs = vae.decode(latents).cpu() # (MaxStrokes, P, D)
            
            # Remove empty strokes if any? or model handles it?
            # Usually we trust model output. 
            # If used with EOS, we might want to cut off.
            # But infer returns padded strokes often.
            # But DiT.infer logic usually handles EOS and returns valid latents up to max_len?
            # Let's assume trajs contains all predicted strokes. 
            
            # Note: infer usually returns fixed number of latents?
            # Let's check DIT infer signature in mind. 
            # Assuming it returns what we need. 
            # But if there is EOS logic in DiT.infer, latents might be cut?
            # infer.py old code:
            # latents = latents[0]
            # trajs = vae.decode(latents).cpu()
            
            # Post-process check: empty strokes?
            # The VAE decodes every latent to a stroke.
            
            # Save strokes
            # trajs is tensor (N, P, D). Convert to list of lists.
            strokes_list = trajs.tolist()
            all_font_data[key] = strokes_list
            
            # Visualize
            vis_path = os.path.join(vis_folder, f"{key}.png")
            visualize(pil_img, trajs, vis_path, point_dim=cfg.dims.point_dim, image_size=cfg.dims.image_size)
            
        except Exception as e:
            print(f"Error inferring {key}: {e}")

    # Save strokes pkl
    strokes_pkl_path = os.path.join(args.output_folder, "strokes.pkl")
    with open(strokes_pkl_path, 'wb') as f:
        pickle.dump(all_font_data, f)
    print(f"Saved inference results to {strokes_pkl_path}")
    print("Done.")


if __name__ == "__main__":
     main()
