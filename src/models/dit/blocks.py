import torch
import torch.nn as nn
from typing import Optional

from .attention import CausalSelfAttention, ImageCrossAttention
from .layers import AdaptiveLayerNorm

class AutoregressiveTransformerBlock(nn.Module):
    """A single transformer block for autoregressive latent generation.
    
    Architecture:
    1. Causal self-attention over stroke sequence (with AdaLN)
    2. Cross-attention to image features (with AdaLN)
    3. Feed-forward network (with AdaLN)
    
    Uses gated residual connections for stable training.
    """
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        # Causal self-attention
        self.self_attn = CausalSelfAttention(model_dim, num_heads, dropout)
        self.adaln1 = AdaptiveLayerNorm(model_dim, model_dim)
        
        # Cross-attention to image
        self.cross_attn = ImageCrossAttention(model_dim, num_heads, dropout)
        self.adaln2 = AdaptiveLayerNorm(model_dim, model_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )
        self.adaln3 = AdaptiveLayerNorm(model_dim, model_dim)
        
        # Gated residual connections - start near identity
        self.gate_sa = nn.Parameter(torch.tensor(0.1))
        self.gate_ca = nn.Parameter(torch.tensor(0.1))
        self.gate_ff = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        global_cond: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) stroke sequence features
            image_features: (B, M, D) image features from encoder
            global_cond: (B, D) global conditioning (image summary)
            causal_mask: (N, N) causal attention mask
        """
        # Causal self-attention
        x_norm = self.adaln1(x, global_cond)
        sa_out = self.self_attn(x_norm, causal_mask)
        x = x + torch.tanh(self.gate_sa) * self.dropout(sa_out)
        
        # Cross-attention to image
        x_norm = self.adaln2(x, global_cond)
        ca_out = self.cross_attn(x_norm, image_features)
        x = x + torch.tanh(self.gate_ca) * self.dropout(ca_out)
        
        # Feed-forward
        x_norm = self.adaln3(x, global_cond)
        ff_out = self.ff(x_norm)
        x = x + torch.tanh(self.gate_ff) * ff_out
        
        return x
