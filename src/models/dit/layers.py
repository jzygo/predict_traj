import torch
import torch.nn as nn
from typing import Tuple

class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization (AdaLN) for conditional generation.
    
    AdaLN modulates the normalized features using learned scale and shift
    parameters derived from conditioning information. This allows the model
    to adapt its behavior based on global context (e.g., image features,
    step index, or previous stroke information).
    """
    
    def __init__(self, model_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, model_dim * 2),
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input features
            cond: (B, D) or (B, N, D) conditioning features
        """
        x = self.norm(x)
        
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, D)
            
        params = self.cond_proj(cond)  # (B, 1, 2D) or (B, N, 2D)
        scale, shift = params.chunk(2, dim=-1)
        
        return x * (1 + scale) + shift


class LatentPredictionHead(nn.Module):
    """Head for predicting latent vectors and EOS from transformer output.
    
    Includes multi-layer refinement for better latent prediction and
    separate EOS prediction with uncertainty estimation.
    """
    
    def __init__(self, model_dim: int, latent_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        # Latent prediction with refinement
        self.latent_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, latent_dim * 2),  # Mean and log_var for sampling
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(model_dim, latent_dim)
        
        # EOS prediction head
        self.eos_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.LayerNorm(model_dim // 2),
            nn.Linear(model_dim // 2, 1),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) transformer output
            
        Returns:
            latent: (B, N, latent_dim) predicted latent vectors
            eos_logits: (B, N) EOS logits
        """
        latent = self.latent_head(x) + self.residual_proj(x)
        eos_logits = self.eos_head(x).squeeze(-1)
        return latent, eos_logits
