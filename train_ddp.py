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
import yaml
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import random

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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config):
    setup(rank, world_size)
    
    # 1. Prepare Data
    dataset = CalligraphyVideoDataset(root_dir=config['data_path'], frames=config['frames'], img_size=config['img_size'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
    
    # 2. Prepare Model
    model = VisualActionTokenizer(
        img_size=config['img_size'],
        frames=config['frames'],
        embed_dim=config['embed_dim'],
        num_queries=config['num_queries']
    ).to(rank)
    
    model = DDP(model, device_ids=[rank])
    
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
    
    # 5. Training Loop
    num_epochs = config['epochs']
    
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
            
        for batch_idx, video in enumerate(pbar):
            video = video.to(rank)
            
            # Diffusion Training Logic
            # 1. Encode clean video to get Action Latents
            # Note: In a real scenario, we might want to stop gradients flowing back into Encoder 
            # if we are training them separately, but here we train end-to-end.
            z_action = model.module.encoder(video)
            
            # 2. Add Noise to Video (Diffusion Process)
            t = torch.randint(0, 1000, (video.shape[0],), device=rank).long()
            noise = torch.randn_like(video)
            
            # Simple linear noise scheduler for demo (replace with proper scheduler)
            alpha = 1 - (t / 1000.0).view(-1, 1, 1, 1, 1)
            noisy_video = video * torch.sqrt(alpha) + noise * torch.sqrt(1 - alpha)
            
            # 3. Predict (Denoise)
            # The decoder tries to predict the original video (x0) or the noise (epsilon)
            # Here we predict x0 for simplicity with the reconstruction loss
            pred_video = model.module.decoder(noisy_video, t, z_action)
            
            # 4. Calculate Loss
            # We compare predicted x0 with real x0 (video)
            loss, loss_dict = criterion(video, pred_video)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_flow_loss += loss_dict['flow_loss'].item()
            num_batches += 1
            
            if rank == 0:
                pbar.set_postfix(loss=loss.item(), flow=loss_dict['flow_loss'].item())
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # Calculate and print average loss
        avg_loss = epoch_loss / num_batches
        avg_flow = epoch_flow_loss / num_batches
        print(f"Epoch {epoch} Summary: Avg Loss: {avg_loss:.4f}, Avg Flow Loss: {avg_flow:.4f}, LR: {current_lr:.6f}")

        if rank == 0 and batch_idx % 10 == 0:
            
            os.makedirs("checkpoints", exist_ok=True)
            # Save checkpoint
            torch.save(model.module.state_dict(), f"checkpoints/model.pth")
            
            # Save random sample video
            if 'pred_video' in locals():
                os.makedirs("debug", exist_ok=True)
                idx = random.randint(0, pred_video.shape[0] - 1)
                sample_video = pred_video[idx] # (C, T, H, W)
                save_video_sample(sample_video, f"debug/epoch_{epoch}_sample.mp4")
            
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # 由于是多人共用服务器，开头已经选择了显存占用最低的GPU
    # 根据 CUDA_VISIBLE_DEVICES 确定 world_size
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    else:
        world_size = 1
        
    print(f"Starting training with {world_size} GPUs...")
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
