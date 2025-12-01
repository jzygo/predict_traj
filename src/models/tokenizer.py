import torch
import torch.nn as nn
from .encoder import VisualActionEncoder
from .decoder import ActionReconstructionDecoder

class VisualActionTokenizer(nn.Module):
    """
    The main VLA Tokenizer model.
    Combines the Encoder (Compressor) and Decoder (Reconstructor).
    """
    def __init__(self, 
                 img_size=128, 
                 frames=16,
                 embed_dim=768,
                 num_queries=32):
        super().__init__()
        
        self.encoder = VisualActionEncoder(
            img_size=img_size,
            frames=frames,
            embed_dim=embed_dim,
            num_queries=num_queries
        )
        
        # Decoder is a Diffusion Model
        # Note: In a real training loop, we train the diffusion process.
        # The forward pass here might return the loss or the predicted noise.
        self.decoder = ActionReconstructionDecoder(
            input_size=num_queries,
            in_channels=3, # Pixel space for simplicity in this demo
            hidden_size=embed_dim,
            patch_size=16,
            img_size=img_size,
            frames=frames
        )
        
    def forward(self, video, t=None):
        """
        video: (B, C, T, H, W)
        t: (B,) timesteps for diffusion training. If None, we might be in inference mode.
        """
        # 1. Encode
        z_action = self.encoder(video) # (B, N, D)
        
        # 2. Decode (Training Step)
        # In diffusion training, we sample noise and try to predict it (or x0).
        # Here we assume the caller handles the noise addition and passes 'video' as the noisy input?
        # Actually, usually the model forward takes x_noisy and t.
        # But here we need to extract z_action from the CLEAN video (or a separate view),
        # and then use it to denoise a NOISY version of the video.
        
        # For the purpose of this class, let's just return z_action and the decoder instance
        # so the training loop can handle the diffusion logic (noise sampling).
        
        return z_action

    def get_action_latents(self, video):
        return self.encoder(video)
