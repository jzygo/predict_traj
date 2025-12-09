import json
from dataclasses import asdict, dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    num_points: int = 24
    tokens_per_stroke: int = 3
    codebook_size: int = 512
    embedding_dim: int = 128
    hidden_dim: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "VQVAEConfig":
        return VQVAEConfig(**json.loads(s))


class VectorQuantizerEMA(nn.Module):
    """EMA Vector Quantizer used in VQ-VAE.

    Inputs are expected to be of shape (B, T, D). Returns quantized outputs,
    commitment loss, and perplexity along with token indices.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, T, D)
        z_flat = z.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.embedding.t())
            + torch.sum(self.embedding**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(z.dtype)

        quantized = torch.matmul(encodings, self.embedding).view(z.shape)

        if self.training:
            # EMA updates done in no_grad to avoid graph retention
            with torch.no_grad():
                cluster_size = torch.sum(encodings, dim=0)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                dw = torch.matmul(encodings.t(), z_flat.detach())
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = torch.sum(self.ema_cluster_size)
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps)
                    * n
                )

                embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.copy_(embed_normalized)

        # Losses follow the common VQ-VAE pattern to avoid double-backward issues
        # encoder update: pull z toward the selected embedding (no grad to codebook)
        codebook_loss = F.mse_loss(quantized.detach(), z)
        # commitment: encourage encoder outputs to stay close to embeddings (no extra grad to encoder twice)
        commitment_loss = self.commitment_cost * F.mse_loss(quantized, z.detach())
        vq_loss = commitment_loss + codebook_loss

        # Straight-through estimator: gradients pass to encoder, embeddings updated via EMA only
        quantized = z + (quantized - z).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity, encoding_indices.view(z.shape[0], z.shape[1])


class StrokeEncoder(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        in_dim = cfg.num_points * 2
        out_dim = cfg.tokens_per_stroke * cfg.embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        return self.mlp(x)


class StrokeDecoder(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        in_dim = cfg.tokens_per_stroke * cfg.embedding_dim
        out_dim = cfg.num_points * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, tokens_per_stroke, D)
        z_flat = z.flatten(1)
        return self.mlp(z_flat)


class StrokeVQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StrokeEncoder(cfg)
        self.decoder = StrokeDecoder(cfg)
        self.codebook = VectorQuantizerEMA(
            num_embeddings=cfg.codebook_size,
            embedding_dim=cfg.embedding_dim,
            commitment_cost=cfg.commitment_cost,
            decay=cfg.ema_decay,
        )

    def forward(self, strokes: torch.Tensor):
        # strokes: (B, num_points * 2) already centered and normalized
        h = self.encoder(strokes)  # (B, T * D)
        h = h.view(-1, self.cfg.tokens_per_stroke, self.cfg.embedding_dim)

        quantized, vq_loss, perplexity, indices = self.codebook(h)
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, strokes)

        loss = recon_loss + vq_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "indices": indices,
            "recon": recon,
        }

    @torch.no_grad()
    def encode(self, strokes: torch.Tensor) -> torch.Tensor:
        h = self.encoder(strokes)
        h = h.view(-1, self.cfg.tokens_per_stroke, self.cfg.embedding_dim)
        _, _, _, indices = self.codebook(h)
        return indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T)
        embed = self.codebook.embedding
        z = embed[indices]  # (B, T, D)
        recon = self.decoder(z)
        return recon

    def save(self, path: str):
        payload = {
            "config": self.cfg.to_json(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location=None) -> "StrokeVQVAE":
        payload = torch.load(path, map_location=map_location)
        cfg = VQVAEConfig.from_json(payload["config"])
        model = StrokeVQVAE(cfg)
        model.load_state_dict(payload["state_dict"])
        return model
