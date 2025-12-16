import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple

from config import DiTConfig, DimConfig
from .embeddings import get_2d_sincos_pos_embed

class PatchEncoder(nn.Module):
    """Enhanced patch encoder with overlapping patches and multi-scale features."""
    
    def __init__(self, dims: DimConfig, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = dims.image_size // cfg.patch_size
        self.out_dim = dims.model_dim

        # Use overlapping convolution for better feature extraction
        self.proj = nn.Sequential(
            nn.Conv2d(3, dims.model_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dims.model_dim // 2),
            nn.GELU(),
            nn.Conv2d(dims.model_dim // 2, dims.model_dim, kernel_size=cfg.patch_size // 2, 
                      stride=cfg.patch_size // 2, padding=0),
            nn.BatchNorm2d(dims.model_dim),
        )
        
        self.cls = nn.Parameter(torch.randn(1, 1, dims.model_dim) * 0.02)
        # Use 2D sinusoidal position embedding
        self.register_buffer(
            'pos_embed_2d', 
            get_2d_sincos_pos_embed(dims.model_dim, self.grid, torch.device('cpu'))
        )
        self.pos_proj = nn.Linear(dims.model_dim, dims.model_dim)
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dims.model_dim))
        
        self.norm = nn.LayerNorm(dims.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.proj(x)  # (B, C, H', W')
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, C)
        
        # Apply 2D position embedding
        pos = self.pos_proj(self.pos_embed_2d.to(x.device))
        tokens = tokens + pos
        
        cls = self.cls.expand(B, -1, -1) + self.cls_pos
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.norm(tokens)
        return tokens


class MultiScaleImageEncoder(nn.Module):
    """Multi-scale image encoder that preserves spatial details at multiple resolutions.
    
    This encoder extracts hierarchical features from the input image and fuses them
    into a unified representation. The multi-scale approach helps capture both
    fine-grained stroke details and global character structure.
    """
    
    def __init__(self, dims: DimConfig, cfg: DiTConfig) -> None:
        super().__init__()
        self.model_dim = dims.model_dim
        
        # Load pretrained ResNet and extract multi-scale features
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 64 channels, 1/4 resolution
        self.layer2 = backbone.layer2  # 128 channels, 1/8 resolution
        self.layer3 = backbone.layer3  # 256 channels, 1/16 resolution
        self.layer4 = backbone.layer4  # 512 channels, 1/32 resolution
        
        # Project multi-scale features to model_dim
        self.proj1 = nn.Conv2d(64, dims.model_dim, 1)
        self.proj2 = nn.Conv2d(128, dims.model_dim, 1)
        self.proj3 = nn.Conv2d(256, dims.model_dim, 1)
        self.proj4 = nn.Conv2d(512, dims.model_dim, 1)
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Position encoding for flattened features
        self.pos_proj = nn.Linear(dims.model_dim, dims.model_dim)
        self.norm = nn.LayerNorm(dims.model_dim)
        
        # Global context token - critical for autoregressive generation
        self.global_token = nn.Parameter(torch.randn(1, 1, dims.model_dim) * 0.02)
        
        # Additional global pooling for conditioning
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims.model_dim, dims.model_dim),
            nn.LayerNorm(dims.model_dim),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image
            
        Returns:
            tokens: (B, N+1, D) image feature tokens with global token prepended
            global_feat: (B, D) global image feature for conditioning
        """
        B = x.size(0)
        
        # Extract multi-scale features
        x = self.stem(x)
        f1 = self.layer1(x)   # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)
        
        # Project to same channel dimension
        p1 = self.proj1(f1)  # (B, D, H/4, W/4)
        p2 = self.proj2(f2)  # (B, D, H/8, W/8)
        p3 = self.proj3(f3)  # (B, D, H/16, W/16)
        p4 = self.proj4(f4)  # (B, D, H/32, W/32)
        
        # Upsample all to same resolution (1/8) and combine
        target_size = p2.shape[-2:]
        p1_up = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
        p3_down = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_down = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        
        # Weighted combination
        weights = F.softmax(self.scale_weights, dim=0)
        combined = weights[0] * p1_up + weights[1] * p2 + weights[2] * p3_down + weights[3] * p4_down
        
        # Extract global feature before flattening
        global_feat = self.global_pool(combined)  # (B, D)
        
        # Flatten to sequence
        tokens = combined.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add 2D position encoding
        H, W = target_size
        pos_2d = get_2d_sincos_pos_embed(self.model_dim, H, x.device)
        if pos_2d.shape[0] != tokens.shape[1]:
            pos_2d = pos_2d[:tokens.shape[1]]
        tokens = tokens + self.pos_proj(pos_2d)
        
        # Add global context token
        global_token = self.global_token.expand(B, -1, -1)
        tokens = torch.cat([global_token, tokens], dim=1)
        
        return self.norm(tokens), global_feat
