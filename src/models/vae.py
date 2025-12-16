from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DimConfig, VAEConfig


def _auto_num_heads(model_dim: int) -> int:
    for h in (12, 8, 4, 2):
        if model_dim % h == 0:
            return h
    return 1


def _make_transformer_encoder_layer(model_dim: int, num_heads: int, ff_dim: int, dropout: float):
    """Create a TransformerEncoderLayer compatible with older PyTorch versions."""

    try:
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        return layer, True
    except TypeError:
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=F.gelu,
        )
        return layer, False

class CrossAttentionDecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention to latent."""
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(attn_out)
        
        # Cross-attention to latent
        x2 = self.norm2(x)
        attn_out, _ = self.cross_attn(x2, memory, memory)
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

def sinusoidal_position_embeddings(length: int, dim: int) -> torch.Tensor:
    """Generate fixed sinusoidal position embeddings."""
    half_dim = dim // 2
    emb = torch.arange(half_dim).float()
    import math
    emb = torch.exp(-math.log(10000.0) * emb / float(half_dim - 1))
    positions = torch.arange(length).float().unsqueeze(1)
    sinusoid = positions * emb.unsqueeze(0)
    pos_emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)
    if dim % 2 == 1:
        pos_emb = torch.cat([pos_emb, torch.zeros(length, 1)], dim=1)
    return pos_emb

class StrokeVAE(nn.Module):
    """VAE with query-based decoder using cross-attention.
    
    This is the RECOMMENDED approach - it uses learnable query tokens
    that attend to the latent via cross-attention.
    """
    
    def __init__(self, dims: DimConfig, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dims = dims
        model_dim = cfg.hidden_dim
        latent_dim = dims.latent_dim
        num_heads = getattr(cfg, "num_heads", None) or _auto_num_heads(model_dim)
        ff_mult = getattr(cfg, "ff_mult", 4)
        ff_dim = int(model_dim * ff_mult)

        # ========== Encoder (保持不变) ==========
        self.point_embed = nn.Linear(2, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, dims.num_points + 1, model_dim))
        self.enc_drop = nn.Dropout(cfg.dropout)

        enc_layer, self._enc_batch_first = _make_transformer_encoder_layer(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=cfg.dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.enc_norm = nn.LayerNorm(model_dim)

        self.mu = nn.Linear(model_dim, latent_dim)
        self.logvar = nn.Linear(model_dim, latent_dim)

        # ========== Improved Decoder ==========
        # 1. 可学习的query tokens (每个点一个独立的query)
        self.query_tokens = nn.Parameter(torch.randn(1, dims.num_points, model_dim) * 0.02)
        
        # 2. 增强的位置编码 (使用固定sinusoidal + 可学习)
        fixed_pos = sinusoidal_position_embeddings(dims.num_points, model_dim)
        self.register_buffer('fixed_pos_embed', fixed_pos)
        self.learnable_pos_embed = nn.Parameter(torch.zeros(1, dims.num_points, model_dim))
        
        # 3. Latent投影到memory space
        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim)
        )
        
        # 4. Cross-attention decoder layers
        self.decoder_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(model_dim, num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.dec_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, 2)

        # 初始化
        nn.init.trunc_normal_(self.enc_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.learnable_pos_embed, std=0.1)  # 更大的初始值

    def encode(self, stroke: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = stroke.shape
        x = self.point_embed(stroke)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.enc_pos_embed[:, : P + 1]
        x = self.enc_drop(x)

        if self._enc_batch_first:
            h = self.encoder(x)
        else:
            h = self.encoder(x.transpose(0, 1)).transpose(0, 1)

        h = self.enc_norm(h)
        pooled = h[:, 0]
        mu = self.mu(pooled)
        logvar = self.logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Improved decoder with query-based cross-attention."""
        B = z.size(0)
        P = self.dims.num_points
        
        # 1. 将latent转换为memory (B, 1, D)
        memory = self.latent_to_memory(z).unsqueeze(1)
        
        # 2. 初始化queries (每个点有独立的query)
        queries = self.query_tokens.expand(B, -1, -1)
        
        # 3. 添加强位置编码 (固定 + 可学习)
        pos_embed = self.fixed_pos_embed.unsqueeze(0) + self.learnable_pos_embed
        queries = queries + pos_embed
        
        # 4. Cross-attention layers
        x = queries
        for layer in self.decoder_layers:
            x = layer(x, memory)
        
        x = self.dec_norm(x)
        out = self.out_proj(x)
        return out

    def forward(self, stroke: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(stroke)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        B, T, C = stroke.shape
        eps = 1e-6

        # --- 基础 pointwise 重构误差（按点 L1） ---
        pointwise_l1 = F.l1_loss(recon, stroke, reduction='none').sum(dim=-1)  # [B, T]

        # --- velocity / acc (二阶差分) ---
        vel_recon = recon[:, 1:] - recon[:, :-1]        # [B, T-1, 2]
        vel_target = stroke[:, 1:] - stroke[:, :-1]     # [B, T-1, 2]
        vel_loss = F.smooth_l1_loss(vel_recon, vel_target)

        a_recon = vel_recon[:, 1:] - vel_recon[:, :-1]  # [B, T-2, 2]
        a_target = vel_target[:, 1:] - vel_target[:, :-1]
        acc_loss = F.smooth_l1_loss(a_recon, a_target)

        # --- 曲率近似 (用于加权) ---
        # curvature ≈ ||a|| / (||v|| + eps), 对应中心索引 1..T-2
        v_center = vel_target[:, :-1]  # [B, T-2, 2] 对应 a_i 的速度
        a_center = a_target             # [B, T-2, 2]
        v_norm = torch.norm(v_center, dim=-1) + eps
        a_norm = torch.norm(a_center, dim=-1)
        curv = a_norm / v_norm          # [B, T-2]

        # 归一化曲率到 [0,1]（按 sample 内最大值）
        curv_max = torch.max(curv, dim=1, keepdim=True).values  # [B, 1]
        curv_norm = curv / (curv_max + eps)                      # [B, T-2]
        curv_norm = torch.clamp(curv_norm, 0, 1)  # 确保在 [0, 1] 范围内

        # weight per point (index 1..T-2 in original sequence)
        weights = torch.ones((B, T), device=stroke.device)
        weights[:, 1:-1] = 1.0 + 2.0 * curv_norm  # 降低权重幅度，避免过度强调

        # apply weights to pointwise L1
        recon_loss_weighted = (pointwise_l1 * weights).mean()

        # --- 组合总的重构相关损失 ---
        recon_loss = recon_loss_weighted
        recon_loss = recon_loss + 2.0 * vel_loss + 2.0 * acc_loss

        # KL散度
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon, kl, recon_loss, z

    @torch.no_grad()
    def infer(self, stroke: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(stroke)
        return mu

