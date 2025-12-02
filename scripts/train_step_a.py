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
from c3.models import C3Config, C3VideoAutoencoder
from c3.dataset import CalligraphyDataset, collate_fn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Configuration
    # In DDP with CUDA_VISIBLE_DEVICES set, cuda:rank maps to the correct device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 2. Initialize C3 Model
    config = C3Config()
    model = C3VideoAutoencoder(config).to(device)
    # Disable buffer broadcasting to avoid syncing large constant buffers (like Positional Encoding)
    # which can cause memory aliasing errors and performance issues.
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    
    # 3. Data Loader
    dataset = CalligraphyDataset(
        root_dir=Config.DATA_ROOT,
        video_resolution=Config.VIDEO_RESOLUTION,
        image_resolution=Config.IMAGE_RESOLUTION,
        frame_skip=Config.VIDEO_FRAME_SKIP,
        cache_dir=Config.get_cache_dir() if hasattr(Config, 'USE_CACHE') and Config.USE_CACHE else None
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, sampler=sampler)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=config.pad_token_id) # Moved inside model for memory efficiency
    
    # Initialize Scaler for AMP
    use_amp = getattr(Config, 'USE_AMP', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if rank == 0:
        print(f"Starting Step A Training... (AMP: {use_amp}, Checkpointing: {config.use_checkpointing})")
        
    for epoch in range(Config.NUM_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        # Only show progress bar on rank 0
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}") if rank == 0 else dataloader
        
        for videos, masks, _ in iterator:
            videos = videos.to(device)
            masks = masks.to(device)
            # masks: (B, T_pixel) if raw, (B, T_token) if cached
            
            # 1. Get Cosmos Tokens
            # Cached Tokens (B, T, H, W)
            indices = videos
            token_mask_temporal = masks
            B, t, h, w = indices.shape
                
            with torch.no_grad():
                # Flatten to sequence: (B, S)
                video_tokens = indices.view(B, -1).long() # (B, t*h*w)
                
                # Expand to spatial
                # (B, t, 1, 1) -> (B, t, h, w)
                token_mask = token_mask_temporal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                token_mask = token_mask.reshape(B, -1) # (B, S)
                
                # Apply PAD token to padded areas in video_tokens
                # Note: video_tokens is indices.
                # We should set padded positions to PAD_TOKEN_ID
                video_tokens = video_tokens.masked_fill(~token_mask, config.pad_token_id)

            # 2. Prepare Inputs for C3
            # Input to Encoder: video_tokens
            # Input to Decoder: video_tokens (shifted)
            
            # Transpose for Transformer: (S, B)
            src_tokens = video_tokens.transpose(0, 1)
            
            # Create padding mask for Transformer (True means ignore)
            src_key_padding_mask = ~token_mask # (B, S)
            
            # Target for Decoder
            # We want to reconstruct src_tokens.
            # AR Training: Input is SOS + tokens[:-1], Target is tokens.
            
            # Add SOS token
            sos_token = torch.full((1, B), config.sos_token_id, device=device, dtype=torch.long)
            dec_input = torch.cat([sos_token, src_tokens[:-1, :]], dim=0)
            
            # Target
            target = src_tokens
            
            # Create Causal Mask
            S_tgt = dec_input.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S_tgt).to(device)
            
            # Create Padding Mask for Decoder
            # dec_input: (S, B). We need (B, S) boolean mask where True is padding.
            tgt_key_padding_mask = (dec_input == config.pad_token_id).transpose(0, 1)

            # Forward
            # We need to pass src_key_padding_mask to encode
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(src_tokens, 
                               src_key_padding_mask=src_key_padding_mask, 
                               tgt_tokens=dec_input,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               target=target)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            if use_amp:
                scaler.unscale_(optimizer)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        if rank == 0:
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
        
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                # Save the underlying model (module)
                torch.save(model.module.state_dict(), Config.C3_STEP_A_CKPT)
    
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
    
    # Set visible devices so that inside spawn, rank 0 -> selected_gpus[0], etc.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

