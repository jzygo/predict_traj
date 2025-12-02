import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
import mediapy as media

# Ensure Cosmos-Tokenizer is in python path
project_root = os.getcwd()
cosmos_path = os.path.join(project_root, 'Cosmos-Tokenizer')
if cosmos_path not in sys.path:
    sys.path.append(cosmos_path)

# Ensure src is in python path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from config import Config, get_available_gpus
try:
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    from cosmos_tokenizer.utils import read_video, write_video
    from config import Config
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    sys.exit(1)

def main():
    # Select GPU
    gpus = get_available_gpus(max_gpus=1)
    device = torch.device(f"cuda:{gpus[0]}" if gpus else "cpu")
    print(f"Using device: {device}")
    # Configuration
    DATA_DIR = os.path.join(project_root, 'data')
    OUTPUT_DIR = os.path.join(project_root, 'results')
    # You can change this to other models like Cosmos-1.0-Tokenizer-DV8x16x16
    MODEL_NAME = "Cosmos-0.1-Tokenizer-DV8x8x8" 
    CHECKPOINT_DIR = os.path.join(project_root, 'pretrained_ckpts', MODEL_NAME)
    
    # Checkpoints
    enc_path = os.path.join(CHECKPOINT_DIR, 'encoder.jit')
    dec_path = os.path.join(CHECKPOINT_DIR, 'decoder.jit')
    
    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        print(f"Checkpoints not found at {CHECKPOINT_DIR}")
        print(f"Please download {MODEL_NAME} checkpoints to {CHECKPOINT_DIR}")
        print("You can use the script in Cosmos-Tokenizer/README.md to download them.")
        # We continue, but it will fail if files are missing
    print(f"Using device: {device}")

    # Initialize Tokenizer
    try:
        # Note: We use JIT checkpoints as per the CLI example
        tokenizer = CausalVideoTokenizer(
            checkpoint_enc=enc_path,
            checkpoint_dec=dec_path,
            device=device
        )
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        return

    # Find videos
    # Searching recursively for mp4 files in data directory
    video_pattern = os.path.join(DATA_DIR, '**', '*.mp4')
    video_files = glob.glob(video_pattern, recursive=True)
    
    if not video_files:
        print(f"No video files found in {DATA_DIR}")
        return

    print(f"Found {len(video_files)} videos.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for video_path in video_files:
        print(f"Processing {video_path}...")
        try:
            # Read video
            # Returns TxHxWxC, range [0..255], uint8
            video = read_video(video_path) 
            
            # Resize video
            if Config.VIDEO_RESOLUTION:
                video = media.resize_video(video, shape=Config.VIDEO_RESOLUTION)
                print("  Resized to:", Config.VIDEO_RESOLUTION)

            # Frame skipping
            if Config.VIDEO_FRAME_SKIP > 1:
                video = video[::Config.VIDEO_FRAME_SKIP]
                print("  Applied frame skip:", Config.VIDEO_FRAME_SKIP)
            
            # Add batch dimension: BxTxHxWxC
            batch_video = video[np.newaxis, ...]
            
            # Encode and Decode (Compression and Reconstruction)
            # The forward method handles sliding window, padding, encoding and decoding
            # It returns BxTxHxWx3
            reconstructed_video = tokenizer(batch_video)
            
            # Remove batch dimension
            output_video = reconstructed_video[0]
            
            # Save result
            rel_path = os.path.relpath(video_path, DATA_DIR)
            output_path = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write video
            # Assuming 24 fps, or we could try to read fps from original video if needed
            write_video(output_path, output_video, fps=24) 
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
