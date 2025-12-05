import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowMatchingHead(nn.Module):
    """
    Simple MLP-based Flow Matching Policy Head.
    Mimics the action expert in pi0.
    """
    def __init__(self, action_dim, hidden_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.model = nn.Sequential(
            nn.Linear(action_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, t, cond):
        # x: [B, action_dim]
        # t: [B]
        # cond: [B, cond_dim]
        
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        
        inp = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.model(inp)

class Pi0VLA(nn.Module):
    def __init__(self, 
                 model_id="Qwen/Qwen3-VL-4B-Instruct", 
                 action_dim=256, 
                 freeze_vlm=True,
                 device_map="auto",
                 torch_dtype=torch.bfloat16,
                 use_gradient_checkpointing=False):
        """
        Args:
            model_id: HuggingFace model ID for the VLM.
            action_dim: Dimension of the action space (e.g., flattened trajectory).
            freeze_vlm: Whether to freeze the VLM backbone.
            device_map: Device map for the VLM.
            torch_dtype: Data type for the VLM.
            use_gradient_checkpointing: Whether to use gradient checkpointing for VLM.
        """
        super().__init__()
        
        print(f"Loading VLM: {model_id}...")
        local_dir = "/data1/jizy/qwen/qwen3-vl-4b-instruct-local"
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            local_dir, 
            dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        
        if use_gradient_checkpointing and not freeze_vlm:
            self.vlm.gradient_checkpointing_enable()
            
        # We don't strictly need the processor here if data is pre-processed, 
        # but good to have for reference or inference methods.
        try:
            self.processor = AutoProcessor.from_pretrained(
                local_dir, 
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            print(f"Warning: Could not load processor: {e}")
            self.processor = None
        
        self.freeze_vlm = freeze_vlm
        if freeze_vlm:
            print("Freezing VLM parameters...")
            for param in self.vlm.parameters():
                param.requires_grad = False
            self.vlm.eval() # Set to eval mode
        
        # Get hidden size
        vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        
        # Action Head (Flow Matching)
        self.action_head = FlowMatchingHead(
            action_dim=action_dim,
            hidden_dim=1024,
            cond_dim=vlm_hidden_dim
        )
        
        self.action_dim = action_dim

    def get_vlm_embedding(self, inputs, target_device=None, target_dtype=torch.float32):
        """
        Extracts embeddings from the VLM to condition the policy.
        """
        # Forward pass through VLM
        # We pass all inputs directly (input_ids, pixel_values, attention_mask, image_grid_thw, etc.)
        
        # Use no_grad if VLM is frozen to save memory
        context_manager = torch.no_grad() if self.freeze_vlm else torch.enable_grad()
        
        with context_manager:
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # We use the last hidden state.
            # Strategy: Take the embedding of the last token.
            # This assumes the input is formatted as [Image] [Instruction] -> [Action Start]
            last_hidden_state = outputs.hidden_states[-1] # [B, Seq, Dim]
            
            # Gather the last token for each batch item.
            # If attention_mask is provided, we should use it to find the last non-pad token.
            if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
                # attention_mask: [B, Seq]
                # Note: Qwen-VL might use 4D attention mask or other forms, but usually 2D for padding.
                # We assume standard 2D mask for padding identification here.
                # If it's not 2D, we might need to be careful. 
                # For generation tasks, usually the last token is the one we want.
                mask = inputs['attention_mask']
                if mask.dim() == 2:
                    last_indices = mask.sum(dim=1) - 1
                    cond = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_indices]
                else:
                    # Fallback if mask is complex (e.g. flash attn)
                    cond = last_hidden_state[:, -1, :]
            else:
                cond = last_hidden_state[:, -1, :]
            
        cond = cond.to(dtype=target_dtype)
        if target_device is not None:
            cond = cond.to(target_device)
        return cond

    def forward(self, inputs, actions):
        """
        Training step.
        Args:
            inputs: Dict containing input_ids, pixel_values, etc.
            actions: [B, action_dim] - Ground truth actions (x1)
        Returns:
            loss: Flow matching loss
        """
        # 1. Get Condition from VLM
        cond = self.get_vlm_embedding(
            inputs,
            target_device=actions.device,
            target_dtype=actions.dtype,
        )
        
        # 2. Flow Matching Setup
        B = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        
        # Sample t ~ Uniform[0, 1]
        t = torch.rand(B, device=device, dtype=dtype)
        
        # Sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(actions)
        x1 = actions
        
        # Interpolate state: x_t = t * x1 + (1 - t) * x0
        # Reshape t for broadcasting
        t_view = t.view(B, *([1] * (len(actions.shape) - 1)))
        xt = t_view * x1 + (1 - t_view) * x0
        
        # Target velocity: v_target = x1 - x0 (for Optimal Transport path)
        v_target = x1 - x0
        
        # 3. Predict velocity
        v_pred = self.action_head(xt, t, cond)
        
        # 4. Loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def generate_action(self, inputs, steps=10):
        """
        Inference using Euler integration.
        """
        policy_param = next(self.action_head.parameters())
        policy_device = policy_param.device
        policy_dtype = policy_param.dtype
        
        cond = self.get_vlm_embedding(
            inputs,
            target_device=policy_device,
            target_dtype=policy_dtype,
        )
        
        B = cond.shape[0]
        device = policy_device
        
        # Start from noise
        x = torch.randn(B, self.action_dim, device=device, dtype=policy_dtype)
        
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.ones(B, device=device, dtype=policy_dtype) * t_val
            
            v = self.action_head(x, t, cond)
            x = x + v * dt
            
        return x
