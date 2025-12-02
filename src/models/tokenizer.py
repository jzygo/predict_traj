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
                 patch_size=16,
                 frames=16,
                 in_chans=3,
                 embed_dim=768,
                 num_heads=12,
                 num_queries=32,
                 use_checkpoint=False):
        super().__init__()
        
        self.encoder = VisualActionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            frames=frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            use_checkpoint=use_checkpoint
        )
        
        # Decoder is a Diffusion Model
        # Note: In a real training loop, we train the diffusion process.
        # The forward pass here might return the loss or the predicted noise.
        self.decoder = ActionReconstructionDecoder(
            input_size=num_queries,
            in_channels=in_chans, # Pixel space for simplicity in this demo
            hidden_size=embed_dim,
            num_heads=num_heads,
            patch_size=patch_size,
            img_size=img_size,
            frames=frames,
            use_checkpoint=use_checkpoint
        )
        
    def forward(self, video, t=None, noisy_video=None):
        """
        video: (B, C, T, H, W)
        t: (B,) timesteps for diffusion training.
        noisy_video: (B, C, T, H, W) - Optional, for training pass.
        """
        # If training arguments are provided, perform the full denoising pass
        if noisy_video is not None and t is not None:
            return self.denoise(noisy_video, t, video)

        # 1. Encode
        z_action = self.encoder(video) # (B, N, D)
        
        # If t is provided, we could perform a denoising step here, 
        # but typically the training loop handles noise addition and target generation.
        # So we just return the action latents.
        
        return z_action

    def denoise(self, noisy_video, t, context_video):
        """
        Perform one step of denoising.
        noisy_video: (B, C, T, H, W)
        t: (B,)
        context_video: (B, C, T, H, W) - The clean video used to extract action latents
        """
        z_action = self.encoder(context_video)
        pred_video = self.decoder(noisy_video, t, z_action)
        return pred_video

    def get_action_latents(self, video):
        return self.encoder(video)
