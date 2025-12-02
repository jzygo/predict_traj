import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'Cosmos-Tokenizer'))
sys.path.append(os.path.join(project_root, 'src'))

from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
from c3.models import C3Config, C3VideoAutoencoder, ImageToLatentMapper

def run_inference(image_path, output_video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    video_enc_ckpt = "pretrained_models/cosmos_tokenizer/video_encoder.jit"
    video_dec_ckpt = "pretrained_models/cosmos_tokenizer/video_decoder.jit"
    image_enc_ckpt = "pretrained_models/cosmos_tokenizer/image_encoder.jit"
    
    c3_step_a_ckpt = "c3_step_a_epoch_100.pth"
    c3_step_b_ckpt = "c3_step_b_epoch_100.pth"
    
    # 1. Load Models
    print("Loading models...")
    try:
        video_tokenizer = CausalVideoTokenizer(checkpoint_enc=video_enc_ckpt, checkpoint_dec=video_dec_ckpt, device=str(device))
        image_tokenizer = ImageTokenizer(checkpoint_enc=image_enc_ckpt, device=str(device))
    except:
        print("Error loading Cosmos Tokenizers. Ensure checkpoints are present.")
        return

    config = C3Config()
    
    # Step A Model (Decoder needed)
    c3_autoencoder = C3VideoAutoencoder(config).to(device)
    c3_autoencoder.load_state_dict(torch.load(c3_step_a_ckpt, map_location=device))
    c3_autoencoder.eval()
    
    # Step B Model
    mapper = ImageToLatentMapper(config, image_encoder_dim=16).to(device)
    mapper.load_state_dict(torch.load(c3_step_b_ckpt, map_location=device))
    mapper.eval()
    
    # 2. Process Image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    # Resize/Transform as needed
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Example size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
    ])
    img_tensor = transform(image).unsqueeze(0).to(device) # (1, 3, H, W)
    
    with torch.no_grad():
        # 3. Image -> Image Features
        img_latent = image_tokenizer.encode(img_tensor)
        if isinstance(img_latent, tuple): img_latent = img_latent[0]
        
        B, C, H, W = img_latent.shape
        img_features = img_latent.view(B, C, -1).permute(2, 0, 1) # (S, B, C)
        
        # 4. Image Features -> Z_c3 (Latent)
        z_c3 = mapper(img_features) # (N, B, E)
        
        # 5. Z_c3 -> Video Tokens (Autoregressive Generation)
        # This generates indices
        generated_indices = c3_autoencoder.generate(z_c3, max_len=1000) # (S_video, B)
        
        # Remove SOS
        generated_indices = generated_indices[1:, :]
        
        # Reshape to (B, t, h, w)
        # We need to know h, w. Assuming fixed for now or inferred.
        # If we flattened t*h*w, we need to unflatten.
        # This is the tricky part of AR generation for video: knowing when to stop and how to reshape.
        # For simplicity, let's assume we generate a fixed number of tokens or until EOS, 
        # and we know the spatial resolution (e.g. 16x16 tokens).
        
        h_token, w_token = 16, 16 # Example
        tokens_per_frame = h_token * w_token
        
        seq_len = generated_indices.size(0)
        num_frames = seq_len // tokens_per_frame
        
        # Truncate to full frames
        valid_len = num_frames * tokens_per_frame
        generated_indices = generated_indices[:valid_len, :]
        
        indices = generated_indices.view(num_frames, h_token, w_token).unsqueeze(0) # (1, t, h, w)
        
        # 6. Video Tokens -> Video
        # Cosmos Tokenizer decode expects indices or latent
        # If indices, we need to pass them.
        # Note: CausalVideoTokenizer.decode expects (B, t, h, w) for discrete?
        # Let's check video_lib.py again.
        # "input_latent: ... or the discrete indices Bxtxhxw for DV."
        
        video = video_tokenizer.decode(indices)
        
        # Save Video
        # video is (B, 3, T, H, W) in [-1, 1]
        video = (video.clamp(-1, 1) + 1) / 2.0 # [0, 1]
        video = (video * 255).byte().cpu().numpy()
        
        # Save using mediapy or cv2
        # video: (1, 3, T, H, W) -> (T, H, W, 3)
        video_np = video[0].transpose(1, 2, 3, 0)
        
        import cv2
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (video_np.shape[2], video_np.shape[1]))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved video to {output_video_path}")

if __name__ == "__main__":
    # Example usage
    run_inference("data/figure/test.jpg", "output.mp4")
