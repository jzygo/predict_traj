import torch
import torch.nn as nn
from typing import Tuple, Optional

from config import DiTConfig, DimConfig
from .image_encoder import MultiScaleImageEncoder
from .embeddings import LatentInputEmbedding
from .blocks import AutoregressiveTransformerBlock
from .layers import LatentPredictionHead

class AutoregressiveLatentTransformer(nn.Module):
    """Autoregressive Transformer for stroke latent generation.
    
    This model generates strokes autoregressively, where each stroke prediction
    is conditioned on:
    1. The input image (via cross-attention to multi-scale features)
    2. All previously generated strokes (via causal self-attention)
    3. Global image context (via AdaLN modulation)
    
    This architecture solves the key issues of parallel prediction:
    - Mean regression: Each stroke is conditioned on previous strokes, breaking symmetry
    - Unable to stop: Explicit EOS prediction at each step with clear conditioning
    - Parallel collapse: Causal attention ensures proper sequence dependencies
    
    Training uses teacher forcing (feeding ground truth previous strokes).
    Inference generates strokes one by one until EOS or max length.
    """
    
    def __init__(self, dims: DimConfig, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dims = dims
        
        # Image encoder - always use multi-scale for rich spatial features
        self.image_encoder = MultiScaleImageEncoder(dims, cfg)
        
        # Latent input embedding (for autoregressive input)
        self.latent_embed = LatentInputEmbedding(dims.latent_dim, dims.model_dim)
        
        # Transformer blocks with causal attention
        self.blocks = nn.ModuleList([
            AutoregressiveTransformerBlock(dims.model_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(dims.model_dim)
        
        # Prediction heads
        self.pred_head = LatentPredictionHead(dims.model_dim, dims.latent_dim, cfg.dropout)
        
        # Cache for causal masks of different sizes
        self._causal_mask_cache = {}
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask."""
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len].to(device)
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image into features and global conditioning.
        
        Returns:
            image_features: (B, M, D) spatial features for cross-attention
            global_cond: (B, D) global conditioning for AdaLN
        """
        return self.image_encoder(image)
    
    def forward(
        self,
        image: torch.Tensor,
        target_latents: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with teacher forcing for training.
        
        Args:
            image: (B, 3, H, W) input image
            target_latents: (B, S, latent_dim) ground truth latents for teacher forcing
            attn_mask: (B, S+1) bool mask, True for padding positions
            
        Returns:
            latent_pred: (B, S+1, latent_dim) predicted latents (shifted by 1)
            eos_logits: (B, S+1) EOS prediction logits
            
        Note:
            The model predicts latent[i] given latent[0:i-1] and image.
            Position 0 predicts the first stroke given only image (SOS input).
            Position i predicts stroke i given strokes 0..i-1.
        """
        B, S, _ = target_latents.shape
        device = image.device
        
        # Encode image
        image_features, global_cond = self.encode_image(image)
        
        # Prepare input: shift target latents right and prepend SOS
        # Input at position i is the latent of stroke i-1 (for predicting stroke i)
        input_latents = target_latents[:, :-1]  # Remove last (B, S-1, D)
        x = self.latent_embed(input_latents, S)  # (B, S, model_dim) with SOS prepended
        
        # Get causal mask
        causal_mask = self._get_causal_mask(x.size(1), device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, image_features, global_cond, causal_mask)
        
        x = self.norm(x)
        
        # Predict latents and EOS
        latent_pred, eos_logits = self.pred_head(x)
        
        return latent_pred, eos_logits
    
    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        max_len: Optional[int] = None,
        eos_threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Autoregressive inference - generate strokes one by one.
        
        Args:
            image: (B, 3, H, W) input image
            max_len: maximum number of strokes to generate
            eos_threshold: probability threshold for stopping
            temperature: temperature for EOS probability (higher = more strokes)
            
        Returns:
            latents: (B, num_strokes, latent_dim) generated latent vectors
            eos_probs: (B, num_strokes) EOS probabilities
            num_strokes: actual number of strokes generated
        """
        self.eval()
        B = image.size(0)
        device = image.device
        max_len = max_len or self.dims.max_strokes
        
        # Encode image once
        image_features, global_cond = self.encode_image(image)
        
        # Start with just SOS token
        generated_latents = []
        eos_probs = []
        
        # Track which samples have finished
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_len):
            # Prepare input sequence
            if step == 0:
                x = self.latent_embed(None, 1)  # Just SOS
                x = x.expand(B, -1, -1)
            else:
                prev_latents = torch.stack(generated_latents, dim=1)  # (B, step, D)
                x = self.latent_embed(prev_latents, step + 1)  # (B, step+1, D)
            
            # Get causal mask
            causal_mask = self._get_causal_mask(x.size(1), device)
            
            # Forward through transformer
            h = x
            for block in self.blocks:
                h = block(h, image_features, global_cond, causal_mask)
            
            h = self.norm(h)
            
            # Get prediction for current step (last position)
            latent_step, eos_logits_step = self.pred_head(h[:, -1:])
            latent_step = latent_step.squeeze(1)  # (B, latent_dim)
            eos_prob_step = torch.sigmoid(eos_logits_step.squeeze(-1) / temperature)  # (B,)
            
            # Check if all samples have finished
            finished = finished | (eos_prob_step > eos_threshold)
            if finished.all():
                break
            generated_latents.append(latent_step)
            eos_probs.append(eos_prob_step)
            
        
        # Stack results
        latents = torch.stack(generated_latents, dim=1)  # (B, num_steps, latent_dim)
        probs = torch.stack(eos_probs, dim=1)  # (B, num_steps)
        
        return latents, probs, latents.size(1)
    
    @torch.no_grad()
    def infer_with_kv_cache(
        self,
        image: torch.Tensor,
        max_len: Optional[int] = None,
        eos_threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Optimized inference with KV caching (faster for long sequences).
        
        This version caches the key-value pairs from previous steps to avoid
        recomputing them. Currently a placeholder - full implementation would
        require modifying the attention layers.
        """
        # For now, fall back to regular inference
        # TODO: Implement proper KV caching for faster inference
        return self.infer(image, max_len, eos_threshold, temperature)
