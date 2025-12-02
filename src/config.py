import os
import torch

def get_available_gpus(max_gpus=6):
    """
    Selects up to max_gpus available GPUs based on free memory.
    Returns a list of device indices (e.g., [0, 1, 3]).
    """
    if not torch.cuda.is_available():
        return []
    
    try:
        num_devices = torch.cuda.device_count()
        gpu_stats = []
        
        for i in range(num_devices):
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                gpu_stats.append((i, free_mem))
            except Exception:
                continue
        
        # Sort by free memory (descending)
        gpu_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N GPUs
        selected_gpus = [x[0] for x in gpu_stats[:max_gpus]]
        
        print(f"Selected GPUs: {selected_gpus} (Available: {num_devices})")
        return selected_gpus
    except Exception as e:
        print(f"Error selecting GPUs: {e}. Defaulting to [0]")
        return [0]

def get_best_device():
    gpus = get_available_gpus(max_gpus=1)
    if gpus:
        return f"cuda:{gpus[0]}"
    return "cpu"

class Config:
    # Project Root
    # Assuming this file is in src/config.py, so project root is one level up
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Device
    # For DDP, we will determine device in the script based on rank
    # This is just a fallback or for single-process scripts
    DEVICE = get_best_device()
    
    # Multi-GPU
    MAX_GPUS = 8
    
    # Training
    BATCH_SIZE = 1 # Per GPU batch size

    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    
    # Data
    DATA_ROOT = PROJECT_ROOT 
    VIDEO_RESOLUTION = (256, 256)
    IMAGE_RESOLUTION = (256, 256)
    VIDEO_FRAME_SKIP = 30
    
    # Cache
    CACHE_ROOT = os.path.join(DATA_ROOT, 'cache')
    USE_CACHE = True
    
    @staticmethod
    def get_cache_dir():
        # Create a unique cache directory based on config
        dir_name = f"res{Config.VIDEO_RESOLUTION[0]}x{Config.VIDEO_RESOLUTION[1]}_skip{Config.VIDEO_FRAME_SKIP}"
        return os.path.join(Config.CACHE_ROOT, dir_name)
    
    # Cosmos Tokenizer Paths
    PRETRAINED_CKPTS_DIR = os.path.join(PROJECT_ROOT, 'pretrained_ckpts')
    
    # Video Tokenizer
    VIDEO_MODEL_NAME = "Cosmos-0.1-Tokenizer-DV8x8x8"
    VIDEO_CHECKPOINT_DIR = os.path.join(PRETRAINED_CKPTS_DIR, VIDEO_MODEL_NAME)
    VIDEO_ENC_CKPT = os.path.join(VIDEO_CHECKPOINT_DIR, 'encoder.jit')
    VIDEO_DEC_CKPT = os.path.join(VIDEO_CHECKPOINT_DIR, 'decoder.jit')
    
    # Image Tokenizer
    IMAGE_MODEL_NAME = "Cosmos-0.1-Tokenizer-DI16x16" 
    # Note: In original code, it pointed to VIDEO_MODEL_NAME for image dir. 
    # Assuming standard structure where each model has its own dir.
    IMAGE_CHECKPOINT_DIR = os.path.join(PRETRAINED_CKPTS_DIR, IMAGE_MODEL_NAME)
    IMAGE_ENC_CKPT = os.path.join(IMAGE_CHECKPOINT_DIR, 'encoder.jit')
    IMAGE_DEC_CKPT = os.path.join(IMAGE_CHECKPOINT_DIR, 'decoder.jit')
    
    # C3 Model Checkpoints
    C3_STEP_A_CKPT = os.path.join(PROJECT_ROOT, "c3_step_a.pth")
    C3_STEP_B_CKPT = os.path.join(PROJECT_ROOT, "c3_step_b.pth")
    
    # C3 Model Architecture
    VOCAB_SIZE = 64000 + 10
    EMBED_DIM = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    NUM_QUERIES = 512
    DROPOUT = 0.1
    MAX_LEN = 20000
    
    PAD_TOKEN_ID = 64000
    SOS_TOKEN_ID = 64001
    EOS_TOKEN_ID = 64002
    
    # Step B specific
    IMAGE_FEATURE_DIM = 16

    # Optimization
    USE_AMP = True
    USE_CHECKPOINTING = True
