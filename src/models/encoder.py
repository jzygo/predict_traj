import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat

class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, frames=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.frames = frames
        self.tubelet_size = tubelet_size
        self.num_patches = (frames // tubelet_size) * (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x

class StrokeCompressor(nn.Module):
    """
    C3-inspired Query-based Compression.
    Extracts 'Action Latents' from visual features.
    """
    def __init__(self, input_dim, num_queries=32, query_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        
        # Learnable Stroke Queries: The "Anchors of Action"
        self.stroke_queries = nn.Parameter(torch.randn(1, num_queries, query_dim))
        
        # Cross Attention Mechanism
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim)
        )
        self.norm_ffn = nn.LayerNorm(query_dim)

    def forward(self, visual_features):
        # visual_features: (B, L_visual, D)
        B = visual_features.shape[0]
        
        # Expand queries for batch
        queries = repeat(self.stroke_queries, '1 n d -> b n d', b=B)
        
        # Cross Attention: Queries attend to Visual Features
        # Key/Value = Visual Features, Query = Stroke Queries
        attn_out, _ = self.cross_attn(query=queries, key=visual_features, value=visual_features)
        
        x = self.norm(queries + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        z_action = self.norm_ffn(x + ffn_out)
        
        return z_action # (B, N=32, D)

class VisualActionEncoder(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 frames=16, 
                 tubelet_size=2, 
                 in_chans=3,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 num_queries=32,
                 use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        # 1. Input Representation: 3D Patch Embedding
        self.patch_embed = PatchEmbed3D(img_size, patch_size, frames, tubelet_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        
        # 2. Backbone: Standard ViT Encoder Blocks (simplified)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. Compression: Stroke Compressor
        self.compressor = StrokeCompressor(input_dim=embed_dim, num_queries=num_queries, query_dim=embed_dim, num_heads=num_heads)
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Extract Spatio-temporal features
        if self.use_checkpoint:
            visual_features = x
            for layer in self.backbone.layers:
                visual_features = torch.utils.checkpoint.checkpoint(layer, visual_features, use_reentrant=False)
        else:
            visual_features = self.backbone(x) # (B, L, D)
        
        # Compress to Action Latents
        z_action = self.compressor(visual_features) # (B, N=32, D)
        
        return z_action
