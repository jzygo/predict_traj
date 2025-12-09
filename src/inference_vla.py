import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor

from models.vla import Pi0VLA

def get_args():
    parser = argparse.ArgumentParser(description="Inference Pi0 VLA Model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/vla.pt", help="Path to trained checkpoint (vla.pt)")
    parser.add_argument("--image_path", type=str, default="/home/jizy/project/video_tokenizer/data/ding.png", help="Path to input image")
    parser.add_argument("--text", type=str, default="Write the character", help="Text prompt")
    parser.add_argument("--output", type=str, default="inference_result.png", help="Output image path")
    parser.add_argument("--num_points", type=int, default=128, help="Number of points in trajectory")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--num_action_tokens", type=int, default=8, help="Number of special action tokens to append")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="HuggingFace Model ID")

    return parser.parse_args()

def main():
    args = get_args()
    
    # Determine dtype
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    print(f"Using device: {args.device}, dtype: {dtype}")

    # 1. Initialize Model
    action_dim = args.num_points * 2
    
    print(f"Initializing model...")
    # Note: Pi0VLA hardcodes the VLM path.
    # We use device_map to place the model on the correct device.
    # If using CPU, device_map might need to be adjusted or removed if it causes issues, 
    # but Pi0VLA uses 'auto' by default which is usually fine, or we can pass specific map.
    
    device_map = "auto"
    if args.device == "cpu":
        device_map = "cpu"
    elif args.device.startswith("cuda"):
        # Force VLM to the specific device to ensure inputs and model are on the same device
        device_map = args.device
        
    model = Pi0VLA(
        model_id=args.model_id,
        action_dim=action_dim,
        num_action_tokens=args.num_action_tokens,
        freeze_vlm=True,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device_map=device_map,
        torch_dtype=dtype
    )
    
    # 2. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    if os.path.exists(args.checkpoint):
        # Use the new load_checkpoint method
        model.load_checkpoint(args.checkpoint)
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    model.to(args.device)
    model.eval()
    
    # 3. Prepare Input
    if not os.path.exists(args.image_path):
        print(f"Error: Image {args.image_path} not found.")
        return
        
    image = Image.open(args.image_path).convert("RGB")
    
    # Processor
    if model.processor is None:
        # Fallback if model didn't load processor
        local_dir = "/data1/jizy/qwen/qwen3-vl-4b-instruct-local"
        print(f"Model processor is None, trying to load from {local_dir}...")
        try:
            processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            print(f"Failed to load processor from local dir: {e}")
            print("Trying to load from HuggingFace hub...")
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
    else:
        processor = model.processor
        
    # Prepare chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    
    # 4. Inference
    print(f"Running inference with {args.steps} steps...")
    with torch.no_grad():
        # generate_action returns list of tensors [num_points * 2]
        actions = model.generate_action(inputs, steps=args.steps)
        
    # 5. Visualize
    print("Visualizing...")
    
    plt.figure(figsize=(12, 6))
    
    # Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Trajectory
    plt.subplot(1, 2, 2)
    # Draw border
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', alpha=0.5)
    
    colors = plt.cm.jet(np.linspace(0, 1, len(actions)))
    for i, action in enumerate(actions):
        stroke = action.float().cpu().numpy().reshape(-1, 2)
        # Invert Y for visualization if needed, assuming -1 is top
        # Based on visualize_sample in font_dataset.py
        plt.plot(stroke[:, 0], -stroke[:, 1], label=f'Stroke {i+1}', color=colors[i])
        plt.scatter(stroke[0, 0], -stroke[0, 1], color=colors[i], s=20, marker='o')
        plt.scatter(stroke[:, 0], -stroke[:, 1], color=colors[i], s=2)
        
    plt.title("Predicted Trajectory")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(args.output)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
