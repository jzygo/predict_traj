import sys
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'Cosmos-Tokenizer'))
sys.path.append(os.path.join(project_root, 'src'))

from config import Config, get_available_gpus
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
from c3.models import C3Config, C3VideoAutoencoder, ImageToLatentMapper
from c3.dataset import CalligraphyDataset, collate_fn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # Different port than Step A just in case
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Configuration
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # 1. Initialize Tokenizers
    if rank == 0:
        print(f"Initializing Tokenizers on rank {rank}...")
    try:
        video_tokenizer = CausalVideoTokenizer(
            checkpoint_enc=Config.VIDEO_ENC_CKPT,
            checkpoint_dec=Config.VIDEO_DEC_CKPT,
            device=str(device)
        )
        image_tokenizer = ImageTokenizer(checkpoint_enc=Config.IMAGE_ENC_CKPT, device=str(device))
    except Exception as e:
        raise RuntimeError(f"Failed to load Cosmos Tokenizer: {e}")

    # 2. Load Trained C3 Autoencoder (Step A)
    config = C3Config()
    c3_autoencoder = C3VideoAutoencoder(config).to(device)
    if os.path.exists(Config.C3_STEP_A_CKPT):
        # Map location is important for DDP loading
        c3_autoencoder.load_state_dict(torch.load(Config.C3_STEP_A_CKPT, map_location=device))
        if rank == 0:
            print("Loaded C3 Autoencoder checkpoint.")
    else:
        if rank == 0:
            print("Warning: C3 Autoencoder checkpoint not found. Training from scratch (not recommended for Step B).")
    
    c3_autoencoder.eval() # Freeze Step A model
    for param in c3_autoencoder.parameters():
        param.requires_grad = False
        
    # 3. Initialize Image-to-Latent Mapper
    mapper = ImageToLatentMapper(config, image_encoder_dim=Config.IMAGE_FEATURE_DIM).to(device)
    mapper = DDP(mapper, device_ids=[rank], broadcast_buffers=False)
    
    # 4. Data Loader
    dataset = CalligraphyDataset(
        root_dir=Config.DATA_ROOT,
        video_resolution=Config.VIDEO_RESOLUTION,
        image_resolution=Config.IMAGE_RESOLUTION,
        frame_skip=Config.VIDEO_FRAME_SKIP,
        cache_dir=Config.get_cache_dir() if hasattr(Config, 'USE_CACHE') and Config.USE_CACHE else None
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, sampler=sampler)
    
    optimizer = optim.AdamW(mapper.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss() # Regression to Z_c3
    
    # Initialize Scaler
    use_amp = getattr(Config, 'USE_AMP', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if rank == 0:
        print(f"Starting Step B Training... (AMP: {use_amp}, Checkpointing: {config.use_checkpointing})")
        
    for epoch in range(Config.NUM_EPOCHS):
        sampler.set_epoch(epoch)
        mapper.train()
        total_loss = 0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}") if rank == 0 else dataloader
        
        for videos, masks, images in iterator:
            videos = videos.to(device)
            masks = masks.to(device)
            images = images.to(device)
            
            # 1. Get Target Z_c3 from Video
            if videos.ndim == 5: # Raw
                with torch.no_grad():
                    indices, _ = video_tokenizer.encode(videos)
                B, t, h, w = indices.shape
                temp_downsample = videos.shape[2] // t if t > 0 else 1
                token_mask_temporal = masks[:, ::temp_downsample][:, :t]
            else: # Cached
                indices = videos
                token_mask_temporal = masks
                B, t, h, w = indices.shape
            
            with torch.no_grad():
                video_tokens = indices.view(B, -1).long()
                
                # Masking (Same as Step A)
                token_mask = token_mask_temporal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w).reshape(B, -1)
                video_tokens = video_tokens.masked_fill(~token_mask, config.pad_token_id)
                
                src_tokens = video_tokens.transpose(0, 1)
                src_key_padding_mask = ~token_mask
                
                # Get Z_c3
                with torch.cuda.amp.autocast(enabled=use_amp):
                    target_z = c3_autoencoder.encode(src_tokens, src_key_padding_mask) # (N, B, E)
            
            # 2. Get Image Features
            if images.shape[1] == 3: # Raw Image (B, 3, H, W)
                with torch.no_grad():
                    # Assume continuous latent: (B, 16, h, w)
                    img_latent = image_tokenizer.encode(images)
                    if isinstance(img_latent, tuple):
                        img_latent = img_latent[0] # Handle if it returns tuple
            else: # Cached Latent (B, 16, H, W)
                img_latent = images
                
            with torch.no_grad():
                # Flatten spatial dims to sequence
                # (B, C, H, W) -> (B, C, S) -> (S, B, C)
                B, C, H, W = img_latent.shape
                img_features = img_latent.view(B, C, -1).permute(2, 0, 1) # (S_img, B, 16)
                
            # 3. Predict Z_c3
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_z = mapper(img_features) # (N, B, E)
                
                # 4. Loss
                loss = criterion(pred_z, target_z)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        if rank == 0:
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
        
            if (epoch + 1) % 10 == 0:
                torch.save(mapper.module.state_dict(), Config.C3_STEP_B_CKPT)
    
    cleanup()

def main():
    # Auto-run preprocessing if cache is enabled
    if hasattr(Config, 'USE_CACHE') and Config.USE_CACHE:
        print("Verifying cache integrity...")
        script_path = os.path.join(current_dir, 'preprocess_data.py')
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Preprocessing failed ({e}). Training might be slow or fail if cache is missing.")

    # Select GPUs
    selected_gpus = get_available_gpus(max_gpus=Config.MAX_GPUS)
    world_size = len(selected_gpus)
    
    if world_size == 0:
        print("No GPUs available. Exiting.")
        return
        
    print(f"Starting training on {world_size} GPUs: {selected_gpus}")
    
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

