import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.example.com"
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoProcessor
import subprocess

from data.font_dataset import FontDataset
from data.traj_utils import normalize_strokes
from models.vla import Pi0VLA

def get_args():
    parser = argparse.ArgumentParser(description="Train Pi0 VLA Model")
    parser.add_argument("--data_root", type=str, default=r"/data1/jizy/font_out", help="Path to data root")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_points", type=int, default=128, help="Number of points in trajectory")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

class VLACollate:
    def __init__(self, processor, num_points=128, prompt_template="Write the character: {}"):
        self.processor = processor
        self.num_points = num_points
        self.prompt_template = prompt_template

    def __call__(self, batch):
        # batch is a list of dicts from FontDataset
        # {'key': str, 'image': PIL.Image, 'strokes': list of tensors}
        
        images = [item['image'] for item in batch]
        keys = [item['key'] for item in batch]
        raw_strokes = [item['strokes'] for item in batch]
        
        # 1. Prepare Text Prompts
        texts = [self.prompt_template.format(k) for k in keys]
        
        # 2. Prepare Actions (Trajectories)
        actions = []
        for s in raw_strokes:
            # normalize_strokes returns [num_points * 3]
            act = normalize_strokes(s, num_points=self.num_points)
            actions.append(act)
        actions = torch.stack(actions)
        
        # 3. Process Inputs for VLM (Qwen-VL)
        # Qwen-VL processor expects a specific format for chat-like inputs usually
        # Or we can use the lower-level call.
        # For Qwen2-VL/Qwen3-VL, the input is often a list of messages.
        
        messages_list = []
        for text, img in zip(texts, images):
            # Construct message for Qwen-VL
            # Note: The exact format depends on the processor version, 
            # but generally it supports a list of dicts.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            messages_list.append(messages)
            
        # Use processor to pad and create tensors
        # text=messages_list is supported by Qwen2VLProcessor's apply_chat_template logic internally 
        # or we use apply_chat_template explicitly.
        # Let's try the standard call which handles images + text.
        
        # Qwen-VL processors expect chat-formatted text strings + images separately.
        chat_texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_list
        ]

        inputs = self.processor(
            text=chat_texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        return inputs, actions

def get_free_gpus(threshold_mb=1000):
    try:
        # Query GPU memory usage
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # Parse output
        memory_used = [int(x) for x in result.strip().split('\n')]
        # Find GPUs with memory usage below threshold
        free_gpus = [str(i) for i, mem in enumerate(memory_used) if mem < threshold_mb]
        return free_gpus
    except Exception as e:
        print(f"Error checking GPUs: {e}")
        return []

def identity_transform(x):
    return x

def train_worker(rank, world_size, args):
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    # Note: We set CUDA_VISIBLE_DEVICES in main, so the visible devices are 0, 1, ... world_size-1
    # rank corresponds to the local device index
    device_id = rank
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    if rank == 0:
        print(f"Using {world_size} GPUs. Worker {rank} running on device {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Determine dtype
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 1. Initialize Model
    if rank == 0:
        print(f"Initializing model: {args.model_id} with dtype {dtype}")
        
    action_dim = args.num_points * 3
    
    # Pass device_map to ensure model loads on the correct GPU
    model = Pi0VLA(
        model_id=args.model_id, 
        action_dim=action_dim, 
        freeze_vlm=True,
        device_map={'': device_id},
        torch_dtype=dtype
    )
    
    # Move action head to device
    model.action_head.to(device)

    if args.resume_from:
        if rank == 0:
            print(f"Loading checkpoint from {args.resume_from}")
        state_dict = torch.load(args.resume_from, map_location=device)
        model.action_head.load_state_dict(state_dict)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[device_id])
    
    # 2. Initialize Processor
    if model.module.processor is None:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    else:
        processor = model.module.processor
        
    # 3. Dataset & DataLoader
    dataset = FontDataset(
        data_root=args.data_root, 
        split='all',
        transform=identity_transform 
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    collate_fn = VLACollate(processor, num_points=args.num_points)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=1,
        sampler=sampler,
        pin_memory=False
    )
    
    # 4. Optimizer
    optimizer = AdamW(model.module.action_head.parameters(), lr=args.lr)
    
    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 5. Training Loop
    if rank == 0:
        print("Starting training...")
        
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            progress_bar = dataloader
            
        for batch_idx, (inputs, actions) in enumerate(progress_bar):
            actions = actions.to(device)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype, enabled=(args.bf16 or args.fp16)):
                loss = model(inputs, actions)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if rank == 0:
                progress_bar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        
        if rank == 0:
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            checkpoint_path = os.path.join(args.save_dir, f"vla.pt")
            torch.save(model.module.action_head.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    dist.destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    
    free_gpus = get_free_gpus()
    if not free_gpus:
        print("No free GPUs found!")
        exit(1)
        
    print(f"Found free GPUs: {free_gpus}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free_gpus)
    world_size = len(free_gpus)
    
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size)
