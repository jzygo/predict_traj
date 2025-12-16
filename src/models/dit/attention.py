import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CausalSelfAttention(nn.Module):
    """Causal self-attention for autoregressive generation.
    
    Implements masked self-attention where each position can only attend
    to previous positions, ensuring proper autoregressive behavior.
    """
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        if causal_mask is not None:
            attn = attn + causal_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class ImageCrossAttention(nn.Module):
    """Cross-attention from stroke sequence to image features.
    
    Each position in the stroke sequence attends to all image feature positions,
    allowing the model to gather relevant visual information for predicting
    the next stroke.
    """
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.kv_proj = nn.Linear(model_dim, model_dim * 2)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) stroke sequence features
            context: (B, M, D) image features
        """
        B, N, D = x.shape
        M = context.size(1)
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
