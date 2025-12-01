import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with Adaptive Layer Norm (AdaLN) and Cross-Attention.
    Conditioning comes from Action Latents via Cross-Attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        # Cross-Attention for Action Tokens
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # AdaLN-Zero: Regress shift/scale parameters from timestep embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb, context):
        # x: (B, L, D) - Noisy Patches
        # t_emb: (B, D) - Timestep Embedding
        # context: (B, N, D) - Action Latents
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self-Attention
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross-Attention (Action Tokens)
        # We don't use AdaLN modulation for Cross-Attn norm usually, or we can reuse.
        # Let's keep it simple: Standard LayerNorm for Cross-Attn
        x_norm_cross = self.norm_cross(x)
        cross_out, _ = self.cross_attn(query=x_norm_cross, key=context, value=context)
        x = x + cross_out
        
        # MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionReconstructionDecoder(nn.Module):
    """
    DiT-based Decoder.
    Reconstructs video V from Action Latents Z_action.
    """
    def __init__(self, 
                 input_size=32, # N=32 action tokens
                 patch_size=2, 
                 in_channels=4, # Latent channels if using VAE, or 3 if pixel space
                 hidden_size=768, 
                 depth=12, 
                 num_heads=12, 
                 learn_sigma=False,
                 img_size=128,
                 frames=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # Input embedding for Noisy Latents (Video Patches)
        # Assuming we are working in a Latent Space (e.g. VAE encoded video) or Pixel Space
        # For simplicity, let's assume Pixel Space with Tubelet embedding similar to Encoder
        # But DiT usually works on VAE latents. Let's assume input is (B, T, C, H, W)
        
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Condition Embedder: Process Z_action (B, N, D) -> (B, D)
        # We can pool the action tokens or use cross-attention in blocks.
        # For standard DiT (AdaLN), we need a global vector.
        # Let's use a simple attention pooling or mean pooling for the global condition,
        # AND potentially Cross-Attention inside blocks if we want fine-grained control.
        # For this implementation, we'll stick to AdaLN with a global context from Z_action.
        self.action_pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh() # Simple pooling/projection
        )
        
        # Positional Embedding
        # num_patches needs to be calculated based on input resolution
        self.num_patches = frames * (img_size // patch_size) ** 2
        print(f"DEBUG: Initializing ActionReconstructionDecoder with num_patches={self.num_patches}, img_size={img_size}, frames={frames}")
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, T, H, W):
        """
        x: (N, T*H*W, patch_size*patch_size*C)
        imgs: (N, C, T, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = H // p
        w = W // p
        
        x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
        x = torch.einsum('nthwpqc->ncthpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, T, h * p, w * p))
        return imgs

    def forward(self, x, t, z_action):
        """
        x: (B, C, T, H, W) video tensor (noisy)
        t: (B,) timestep
        z_action: (B, N, D) action latents
        """
        # 1. Patchify Input
        # x: (B, C, T, H, W) -> (B, T*H*W/P^2, P*P*C)
        B, C, T, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, T, H // p, p, W // p, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
        x = x.view(B, -1, p * p * C)
        
        # 2. Embeddings
        x = self.x_embedder(x) + self.pos_embed[:, :x.shape[1], :]
        t_emb = self.t_embedder(t)
        
        # 3. Condition Processing
        # In this updated version, we use t_emb for AdaLN and z_action for Cross-Attention
        c = t_emb 
        
        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x, c, z_action)
            
        # 5. Final Layer
        x = self.final_layer(x, c)
        
        # 6. Unpatchify
        x = self.unpatchify(x, T, H, W)
        
        return x
