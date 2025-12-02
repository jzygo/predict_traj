import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
try:
    from config import Config
except ImportError:
    # Fallback or assume config is available in path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import Config

class C3Config:
    def __init__(self):
        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.nhead = Config.NHEAD
        self.num_encoder_layers = Config.NUM_ENCODER_LAYERS
        self.num_decoder_layers = Config.NUM_DECODER_LAYERS
        self.dim_feedforward = Config.DIM_FEEDFORWARD
        self.num_queries = Config.NUM_QUERIES
        self.dropout = Config.DROPOUT
        self.pad_token_id = Config.PAD_TOKEN_ID
        self.sos_token_id = Config.SOS_TOKEN_ID
        self.eos_token_id = Config.EOS_TOKEN_ID
        self.max_len = Config.MAX_LEN
        self.use_checkpointing = getattr(Config, 'USE_CHECKPOINTING', False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class C3VideoAutoencoder(nn.Module):
    def __init__(self, config: C3Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoder = PositionalEncoding(config.embed_dim, config.dropout, max_len=config.max_len)
        
        # Learnable Context Queries (The Bottleneck)
        self.query_embed = nn.Parameter(torch.randn(config.num_queries, 1, config.embed_dim))
        
        # Encoder (Compressor)
        # We use a Transformer Encoder. 
        # Input will be [Queries; Video_Tokens]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            dropout=config.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        # Decoder (Reconstructor)
        # Standard Autoregressive Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            dropout=config.dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        
        # Output Head
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings and queries with small std
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        
        # Initialize output head
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.output_head.bias, 0.0)
        
        # Initialize Transformer layers (optional, PyTorch defaults are usually okay but Xavier is better for deep models)
        # Here we stick to PyTorch defaults for Transformer layers as they are robust enough,
        # but ensuring embeddings are small is critical.

    def encode(self, video_tokens, src_key_padding_mask=None):
        """
        video_tokens: (Seq_Len, Batch) - Discrete indices
        src_key_padding_mask: (Batch, Seq_Len) - True for padding
        """
        B = video_tokens.size(1)
        
        # 1. Embed Video Tokens
        src = self.token_embedding(video_tokens) # (S, B, E)
        src = self.pos_encoder(src)
        
        # 2. Prepare Queries
        # Expand queries for batch: (N, B, E)
        queries = self.query_embed.repeat(1, B, 1)
        
        # 3. Concatenate [Queries; Video]
        # Note: Queries don't have padding mask (they are always valid)
        # We need to adjust padding mask
        combined_src = torch.cat([queries, src], dim=0) # (N+S, B, E)
        
        if src_key_padding_mask is not None:
            # Create mask for queries (False means not padded)
            query_mask = torch.zeros((B, self.config.num_queries), dtype=torch.bool, device=src.device)
            combined_mask = torch.cat([query_mask, src_key_padding_mask], dim=1) # (B, N+S)
        else:
            combined_mask = None
            
        # 4. Pass through Transformer Encoder
        # We rely on self-attention to let Queries gather info from Video
        if self.config.use_checkpointing:
            memory = combined_src
            for layer in self.transformer_encoder.layers:
                def custom_forward(module, x, mask, key_padding_mask):
                    return module(x, src_mask=mask, src_key_padding_mask=key_padding_mask)
                
                memory = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    layer,
                    memory,
                    None,
                    combined_mask,
                    use_reentrant=False
                )
            if self.transformer_encoder.norm is not None:
                memory = self.transformer_encoder.norm(memory)
        else:
            memory = self.transformer_encoder(combined_src, src_key_padding_mask=combined_mask)
        
        # 5. Extract Latent Z_c3 (The first N tokens)
        z_c3 = memory[:self.config.num_queries] # (N, B, E)
        
        return z_c3

    def decode(self, z_c3, tgt_tokens, tgt_mask=None, tgt_key_padding_mask=None, return_features=False):
        """
        z_c3: (N, B, E) - The fixed length latent
        tgt_tokens: (Tgt_Len, B) - Target tokens for AR training (input to decoder)
        """
        # Embed Target
        tgt = self.token_embedding(tgt_tokens)
        tgt = self.pos_encoder(tgt)
        
        # Pass through Decoder
        # Memory is z_c3
        if self.config.use_checkpointing:
            output = tgt
            for layer in self.transformer_decoder.layers:
                def custom_forward(module, x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
                    return module(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
                
                output = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    layer,
                    output,
                    z_c3,
                    tgt_mask,
                    None, # memory_mask
                    tgt_key_padding_mask,
                    None, # memory_key_padding_mask
                    use_reentrant=False
                )
            if self.transformer_decoder.norm is not None:
                output = self.transformer_decoder.norm(output)
        else:
            output = self.transformer_decoder(
                tgt, 
                z_c3, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        
        if return_features:
            return output

        logits = self.output_head(output)
        return logits

    def forward(self, video_tokens, src_key_padding_mask=None, tgt_tokens=None, tgt_mask=None, tgt_key_padding_mask=None, target=None):
        # Encode
        z_c3 = self.encode(video_tokens, src_key_padding_mask)
        
        # Decode
        if tgt_tokens is None:
            # Inference mode (not implemented in forward, use generate)
            return z_c3
        
        if target is not None:
            # Memory efficient loss computation
            output = self.decode(z_c3, tgt_tokens, tgt_mask, tgt_key_padding_mask, return_features=True)
            
            # Chunked loss computation
            # output: (S, B, E)
            # target: (S, B)
            S, B, E = output.shape
            flat_output = output.view(-1, E)
            flat_target = target.view(-1)
            
            chunk_size = 4096
            total_loss = 0
            total_count = 0
            
            for i in range(0, flat_output.size(0), chunk_size):
                chunk_out = flat_output[i:i+chunk_size]
                chunk_tgt = flat_target[i:i+chunk_size]
                
                chunk_logits = self.output_head(chunk_out)
                loss_chunk = F.cross_entropy(chunk_logits, chunk_tgt, ignore_index=self.config.pad_token_id, reduction='sum')
                
                total_loss += loss_chunk
                total_count += (chunk_tgt != self.config.pad_token_id).sum()
            
            if total_count > 0:
                return total_loss / total_count
            return total_loss

        logits = self.decode(z_c3, tgt_tokens, tgt_mask, tgt_key_padding_mask)
        return logits

    def generate(self, z_c3, max_len=100):
        """
        Autoregressive generation from z_c3
        """
        B = z_c3.size(1)
        device = z_c3.device
        
        # Start with SOS
        generated = torch.full((1, B), self.config.sos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(0)).to(device)
            
            logits = self.decode(z_c3, generated, tgt_mask=tgt_mask)
            next_token_logits = logits[-1, :, :] # (B, Vocab)
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(0) # (1, B)
            
            generated = torch.cat([generated, next_token], dim=0)
            
            # Check EOS (simplified, assumes all finish or fixed length)
            # In practice, handle per-batch EOS
            
        return generated

class ImageToLatentMapper(nn.Module):
    def __init__(self, config: C3Config, image_encoder_dim=1024):
        super().__init__()
        self.config = config
        
        # Assuming we get a sequence of image features from a ViT or CNN
        # e.g., (S_img, B, D_img)
        self.image_feature_proj = nn.Linear(image_encoder_dim, config.embed_dim)
        
        # Learnable Queries (Same concept as in Autoencoder, but trained to map Image -> Z_c3)
        # We can initialize this with the trained queries from Step A, or learn from scratch.
        self.query_embed = nn.Parameter(torch.randn(config.num_queries, 1, config.embed_dim))
        
        # Transformer Decoder-like architecture
        # Queries attend to Image Features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.mapper = nn.TransformerDecoder(decoder_layer, num_layers=4) # Lighter than full model
        
        # Final projection to match Z_c3 space exactly if needed, 
        # but if d_model is same, we can output directly.
        
    def forward(self, image_features):
        """
        image_features: (S_img, B, D_img)
        """
        B = image_features.size(1)
        
        # Project image features
        memory = self.image_feature_proj(image_features) # (S_img, B, E)
        
        # Prepare Queries
        tgt = self.query_embed.repeat(1, B, 1) # (N, B, E)
        
        # Apply Cross Attention (Decoder style)
        # tgt attends to memory (image features)
        if self.config.use_checkpointing:
            output = tgt
            for layer in self.mapper.layers:
                def custom_forward(module, x, memory):
                    return module(x, memory)
                
                output = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    layer,
                    output,
                    memory,
                    use_reentrant=False
                )
            if self.mapper.norm is not None:
                output = self.mapper.norm(output)
            z_pred = output
        else:
            z_pred = self.mapper(tgt, memory)
        
        return z_pred
