import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_position_embeddings(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal position embeddings."""
    half_dim = dim // 2
    emb = torch.arange(half_dim, device=device).float()
    emb = torch.exp(-math.log(10000.0) * emb / float(half_dim - 1))
    positions = torch.arange(length, device=device).float().unsqueeze(1)
    sinusoid = positions * emb.unsqueeze(0)
    pos_emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)
    if dim % 2 == 1:
        pos_emb = torch.cat([pos_emb, torch.zeros(length, 1, device=device)], dim=1)
    return pos_emb


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for diffusion-style conditioning."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, device: torch.device) -> torch.Tensor:
    """Generate 2D sinusoidal position embeddings for image patches."""
    grid_h = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_w = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, -1).T  # (H*W, 2)
    
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos pos embed"
    omega = torch.arange(embed_dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))
    
    out_h = grid[:, 0:1] * omega  # (H*W, D/4)
    out_w = grid[:, 1:2] * omega  # (H*W, D/4)
    
    pos_embed = torch.cat([
        torch.sin(out_h), torch.cos(out_h),
        torch.sin(out_w), torch.cos(out_w)
    ], dim=1)  # (H*W, D)
    
    return pos_embed


class LatentInputEmbedding(nn.Module):
    """Embeds latent vectors for autoregressive input.
    
    Converts VAE latent vectors into transformer-compatible embeddings,
    with special handling for the start-of-sequence token.
    """
    
    def __init__(self, latent_dim: int, model_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        
        # Project latent to model dimension
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        
        # Special start-of-sequence (SOS) token
        self.sos_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        
        # Position encoding for stroke sequence
        self.max_len = 256  # Maximum stroke count
        pos_enc = sinusoidal_position_embeddings(self.max_len, model_dim, torch.device('cpu'))
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, latents: Optional[torch.Tensor], seq_len: int) -> torch.Tensor:
        """
        Args:
            latents: (B, S, latent_dim) previous latents, or None for just SOS
            seq_len: desired sequence length
            
        Returns:
            embeddings: (B, S+1, model_dim) with SOS prepended
        """
        B = latents.size(0) if latents is not None else 1
        device = latents.device if latents is not None else self.sos_token.device
        
        # Start with SOS token
        sos = self.sos_token.expand(B, -1, -1)  # (B, 1, D)
        
        if latents is None or latents.size(1) == 0:
            embeddings = sos
        else:
            # Project latents
            latent_emb = self.latent_proj(latents)  # (B, S, D)
            embeddings = torch.cat([sos, latent_emb], dim=1)  # (B, S+1, D)
        
        # Add position encoding
        pos = self.pos_enc[:embeddings.size(1)].unsqueeze(0).to(device)
        embeddings = embeddings + pos
        
        return embeddings
