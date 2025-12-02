import sys
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'Cosmos-Tokenizer'))
sys.path.append(os.path.join(project_root, 'src'))

from config import Config, get_available_gpus
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
from c3.dataset import CalligraphyDataset

def preprocess_worker(rank, world_size, selected_gpus):
    gpu_id = selected_gpus[rank]
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}")
    
    print(f"[Rank {rank}] Initializing on {device}...")

    # 1. Initialize Tokenizers
    try:
        video_tokenizer = CausalVideoTokenizer(checkpoint_enc=Config.VIDEO_ENC_CKPT, device=str(device))
        image_tokenizer = ImageTokenizer(checkpoint_enc=Config.IMAGE_ENC_CKPT, device=str(device))
    except Exception as e:
        print(f"[Rank {rank}] Error loading tokenizers: {e}")
        return

    # 2. Initialize Dataset (Raw)
    dataset = CalligraphyDataset(
        root_dir=Config.DATA_ROOT,
        video_resolution=Config.VIDEO_RESOLUTION,
        image_resolution=Config.IMAGE_RESOLUTION,
        frame_skip=Config.VIDEO_FRAME_SKIP,
        cache_dir=None # Force raw load
    )
    
    # 3. Prepare Cache Directory
    cache_dir = Config.get_cache_dir()
    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache directory: {cache_dir}")
        print(f"Processing {len(dataset)} samples on {world_size} devices...")
    
    # 4. Process subset
    # Simple stride splitting
    indices = list(range(rank, len(dataset), world_size))
    
    # Use position to avoid overlapping bars
    iterator = tqdm(indices, desc=f"Rank {rank}", position=rank, leave=False)
    
    for idx in iterator:
        # Check if file already exists BEFORE loading data
        video_path, _ = dataset.samples[idx]
        rel_path = os.path.relpath(video_path, dataset.video_dir)
        cache_path = os.path.join(cache_dir, os.path.splitext(rel_path)[0] + '.pt')
        
        if os.path.exists(cache_path):
            continue

        try:
            # Ensure directory exists for this file
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            sample = dataset[idx] # {'type': 'raw', 'video': ..., 'image': ...}
            video = sample['video'].unsqueeze(0).to(device) # (1, C, T, H, W)
            image = sample['image'].unsqueeze(0).to(device) # (1, C, H, W)
            
            with torch.no_grad():
                # Encode Video
                indices_enc, _ = video_tokenizer.encode(video)
                video_tokens = indices_enc.squeeze(0).cpu() # (t, h, w)
                
                # Encode Image
                img_latent = image_tokenizer.encode(image)
                if isinstance(img_latent, tuple):
                    img_latent = img_latent[0]
                image_latent = img_latent.squeeze(0).cpu() # (C, H, W)
                
                # Save
                data = {
                    'video_tokens': video_tokens,
                    'image_latent': image_latent
                }
                torch.save(data, cache_path)
        except Exception as e:
            print(f"[Rank {rank}] Error processing index {idx}: {e}")
            continue

    print(f"[Rank {rank}] Finished.")

def main():
    # Select GPUs
    selected_gpus = get_available_gpus(max_gpus=Config.MAX_GPUS)
    world_size = len(selected_gpus)
    
    if world_size == 0:
        print("No GPUs available. Using CPU.")
        preprocess_worker(0, 1, ["cpu"])
        return

    print(f"Starting preprocessing on {world_size} GPUs: {selected_gpus}")
    
    mp.spawn(preprocess_worker, args=(world_size, selected_gpus), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
