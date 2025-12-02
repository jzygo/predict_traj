import os
# pick_gpu_and_run.py
import GPUtil

if __name__ == "__main__":
    # 尝试找出“最空闲”的两个 GPU（基于内存和 load）
    target_gpu_count = 2
    avail = GPUtil.getAvailable(order='memory', limit=target_gpu_count, maxLoad=0.5, maxMemory=0.5, includeNan=False)
    
    if len(avail) < target_gpu_count:
        # 没找到满足条件的足够GPU，可改策略：取内存最大的那些
        gpus = GPUtil.getGPUs()
        # 按空闲内存从大到小排序
        sorted_indices = sorted(range(len(gpus)), key=lambda i: gpus[i].memoryFree, reverse=True)
        gpu_ids = sorted_indices[:target_gpu_count]
    else:
        gpu_ids = avail

    gpu_ids_str = ",".join(map(str, gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str   # 必须在 import torch 之前
    print("Using GPUs", gpu_ids_str)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import yaml
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import random
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

from src.models.tokenizer import VisualActionTokenizer
from src.losses.composite_loss import CompositeLoss
from src.data.dataset import CalligraphyVideoDataset

def save_video_sample(video_tensor, save_path):
    """
    video_tensor: (C, T, H, W) normalized to [-1, 1]
    """
    with torch.no_grad():
        # Denormalize
        video_tensor = (video_tensor + 1.0) / 2.0
        video_tensor = video_tensor.clamp(0, 1)
        video_tensor = (video_tensor * 255).byte()
        
        # (C, T, H, W) -> (T, H, W, C)
        video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
        
    T, H, W, C = video_np.shape
    
    # Save as mp4 using cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (W, H))
    
    for t in range(T):
        frame = video_np[t]
        if C == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config):
    setup(rank, world_size)
    
    # 1. Prepare Data
    dataset = CalligraphyVideoDataset(
        root_dir=config['data_path'], 
        frames=config['frames'], 
        img_size=config['img_size'], 
        channels=config.get('in_chans', 3),
        data_percentage=config.get('data_percentage', 1.0)
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
    
    # 2. Prepare Model
    model = VisualActionTokenizer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        frames=config['frames'],
        in_chans=config.get('in_chans', 3),
        embed_dim=config['embed_dim'],
        num_heads=config.get('num_heads', 12),
        num_queries=config['num_queries'],
        use_checkpoint=config.get('use_checkpoint', True)
    ).to(rank)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # 4. Loss
    criterion = CompositeLoss(
        lambda_rec=config['lambda_rec'],
        lambda_flow=config['lambda_flow'],
        lambda_perceptual=config['lambda_perceptual'],
        device=rank
    )

    scaler = GradScaler()

    # Diffusion Scheduler (Linear)
    num_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=rank)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # 5. Training Loop
    num_epochs = config['epochs']
    accumulation_steps = config.get('accumulation_steps', 1)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        # Initialize accumulators for average loss
        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        num_batches = 0
        
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
            
        optimizer.zero_grad()
        
        for batch_idx, video in enumerate(pbar):
            video = video.to(rank)
            
            # Diffusion Training Logic
            # 1. Add Noise to Video (Diffusion Process)
            t = torch.randint(0, num_timesteps, (video.shape[0],), device=rank).long()
            noise = torch.randn_like(video)
            
            # Proper scheduler
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t]).view(-1, 1, 1, 1, 1)
            noisy_video = video * sqrt_alphas_cumprod_t + noise * sqrt_one_minus_alphas_cumprod_t
            
            with autocast():
                # 2. Predict (Denoise)
                # Uses the internal denoise method which handles encoding and decoding
                # IMPORTANT: Call model(...) directly to ensure DDP hooks are triggered!
                pred_video = model(video, t=t, noisy_video=noisy_video)
                
                # 4. Calculate Loss
                # We compare predicted x0 with real x0 (video)
                loss, loss_dict = criterion(video, pred_video)
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Accumulate metrics (multiply back to get real loss for logging)
            epoch_loss += loss.item() * accumulation_steps
            epoch_flow_loss += loss_dict['flow_loss'].item()
            num_batches += 1
            
            if rank == 0:
                pbar.set_postfix(loss=loss.item() * accumulation_steps, flow=loss_dict['flow_loss'].item())
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            
            # Calculate and print average loss
            avg_loss = epoch_loss / num_batches
            avg_flow = epoch_flow_loss / num_batches
            print(f"Epoch {epoch} Summary: Avg Loss: {avg_loss:.4f}, Avg Flow Loss: {avg_flow:.4f}, LR: {current_lr:.6f}")
            
            if (epoch + 1) % 10 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                # Save checkpoint
                torch.save(model.module.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")
                
                # Save random sample video with full inference
                os.makedirs("debug", exist_ok=True)
                
                # Pick a random video from the last batch to use as condition (for action latents)
                idx = random.randint(0, video.shape[0] - 1)
                sample_input = video[idx].unsqueeze(0) # (1, C, T, H, W)
                
                model.eval()
                with torch.no_grad():
                    # 1. Encode
                    z_action = model.module.encoder(sample_input)
                    
                    # 2. Diffusion Sampling
                    # Start from noise
                    img = torch.randn_like(sample_input)
                    
                    for i in reversed(range(0, num_timesteps)):
                        t_tensor = torch.full((1,), i, device=rank, dtype=torch.long)
                        
                        # Predict x0
                        pred_x0 = model.module.decoder(img, t_tensor, z_action)
                        
                        # Calculate posterior mean and variance
                        alpha = alphas[i]
                        alpha_cumprod = alphas_cumprod[i]
                        if i > 0:
                            alpha_cumprod_prev = alphas_cumprod[i-1]
                        else:
                            alpha_cumprod_prev = torch.tensor(1.0, device=rank)
                            
                        beta = betas[i]
                        
                        # Posterior Mean (x0 parameterization)
                        posterior_mean = (
                            torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * pred_x0 +
                            torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * img
                        )
                        
                        if i > 0:
                            posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                            noise = torch.randn_like(img)
                            img = posterior_mean + torch.sqrt(posterior_variance) * noise
                        else:
                            img = pred_x0
                            
                    sample_video = img[0] # (C, T, H, W)
                    save_video_sample(sample_video, f"debug/epoch_{epoch+1}_inference.mp4")
                    
                    # Also save the ground truth for comparison
                    save_video_sample(sample_input[0], f"debug/epoch_{epoch+1}_gt.mp4")
                
                model.train()
            
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set a free port for DDP
    if 'MASTER_PORT' not in os.environ:
        port = find_free_port()
        os.environ['MASTER_PORT'] = str(port)
        print(f"Using MASTER_PORT: {port}")

    # 由于是多人共用服务器，开头已经选择了显存占用最低的GPU
    # 根据 CUDA_VISIBLE_DEVICES 确定 world_size
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    else:
        world_size = 1
        
    print(f"Starting training with {world_size} GPUs...")
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
