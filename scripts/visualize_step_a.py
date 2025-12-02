import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'Cosmos-Tokenizer'))
sys.path.append(os.path.join(project_root, 'src'))

from config import Config, get_available_gpus
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.utils import tensor2numpy, write_video
from c3.models import C3Config, C3VideoAutoencoder
from c3.dataset import CalligraphyDataset, collate_fn

def main():
    # Select GPU
    gpus = get_available_gpus(max_gpus=1)
    device = torch.device(f"cuda:{gpus[0]}" if gpus else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    print("Initializing Cosmos Tokenizer...")
    try:
        tokenizer = CausalVideoTokenizer(
            checkpoint_enc=Config.VIDEO_ENC_CKPT,
            checkpoint_dec=Config.VIDEO_DEC_CKPT,
            device=str(device)
        )
    except Exception as e:
        print(f"Failed to load Cosmos Tokenizer: {e}")
        return

    # 2. Initialize C3 Model
    print("Initializing C3 Model...")
    config = C3Config()
    model = C3VideoAutoencoder(config).to(device)
    
    # Load Checkpoint
    ckpt_path = Config.C3_STEP_A_CKPT
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        print("Please run train_step_a.py first.")
        return
        
    print(f"Loading checkpoint from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Data Loader
    print("Loading Dataset...")
    dataset = CalligraphyDataset(
        root_dir=Config.DATA_ROOT,
        video_resolution=Config.VIDEO_RESOLUTION,
        image_resolution=Config.IMAGE_RESOLUTION,
        frame_skip=Config.VIDEO_FRAME_SKIP,
        cache_dir=Config.get_cache_dir() if hasattr(Config, 'USE_CACHE') and Config.USE_CACHE else None
    )
    
    # Get a sample
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    
    # Create output directory
    output_dir = os.path.join(project_root, 'vis_results_step_a')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    
    with torch.no_grad():
        for i, (videos, masks, _) in enumerate(dataloader):
            if i >= 5: # Visualize 5 samples
                break
            
            print(f"Processing sample {i+1}...")
            videos = videos.to(device)
            
            # 1. Get Cosmos Tokens (Ground Truth)
            if videos.ndim == 5: # Raw Video (B, C, T, H, W)
                # FIX: Cosmos Tokenizer expects input in range [-1, 1]
                # If dataset produces [0, 1], we need to rescale.
                if videos.min() >= 0 and videos.max() <= 1.0:
                    # print("  Rescaling video from [0, 1] to [-1, 1] for Cosmos Tokenizer...")
                    videos_input = 2.0 * videos - 1.0
                else:
                    videos_input = videos

                indices, _ = tokenizer.encode(videos_input)
                # indices: (B, t, h, w)
                
                # For visualization, we also want the original video reconstruction from tokenizer
                # to see the upper bound of quality
                gt_recon = tokenizer.decode(indices)
            else: # Cached Tokens (B, T, H, W)
                indices = videos
                # Decode to get "Ground Truth" (Tokenizer reconstruction)
                gt_recon = tokenizer.decode(indices)

            B, t, h, w = indices.shape
            print(f"  Token shape: {indices.shape}")
            
            # Prepare input for C3
            video_tokens = indices.view(B, -1).long() # (B, S)
            
            # Encode to Z_c3
            # Transpose to (S, B) for Transformer
            src_tokens = video_tokens.transpose(0, 1)
            z_c3 = model.encode(src_tokens) # (N, B, E)
            
            # Generate
            seq_len = t * h * w
            print(f"  Generating {seq_len} tokens...")
            
            # Use model.generate which is autoregressive
            # Note: This might be slow for long sequences
            generated_indices = model.generate(z_c3, max_len=seq_len) # (S+1, B) including SOS
            
            # Remove SOS
            generated_indices = generated_indices[1:, :] # (S, B)
            
            # Reshape to (B, t, h, w)
            generated_indices = generated_indices.transpose(0, 1).view(B, t, h, w)
            
            # Decode generated tokens
            pred_recon = tokenizer.decode(generated_indices)
            
            # Save videos
            gt_video = tensor2numpy(gt_recon) # (B, T, H, W, 3)
            pred_video = tensor2numpy(pred_recon) # (B, T, H, W, 3)
            
            # Concatenate along width
            # gt_video[0]: (T, H, W, 3)
            combined = np.concatenate([gt_video[0], pred_video[0]], axis=2)
            
            save_path = os.path.join(output_dir, f"sample_{i+1}.mp4")
            write_video(save_path, combined, fps=24)
            print(f"  Saved to {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
