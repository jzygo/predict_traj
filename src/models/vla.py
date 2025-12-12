import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("Warning: peft not installed. LoRA will not be available.")

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

class DiTFlowMatchingHead(nn.Module):
    """
    Diffusion Transformer (DiT) based Flow Matching Policy Head.
    Treats the trajectory as a sequence of points.
    """
    def __init__(self, action_dim, hidden_dim, cond_dim, num_layers=6, nhead=8):
        super().__init__()
        self.action_dim = action_dim
        self.point_dim = 2 
        self.num_points = action_dim // self.point_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(self.point_dim, hidden_dim)
        
        # Learnable Positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_points, hidden_dim) * 0.02)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.point_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t, cond):
        # x: [B, action_dim]
        B = x.shape[0]
        x = x.view(B, self.num_points, self.point_dim)
        
        # Embeddings
        x_emb = self.input_proj(x) + self.pos_emb # [B, N, D]
        
        # Time & Condition
        t_emb = self.time_mlp(t).unsqueeze(1) # [B, 1, D]
        
        # cond: [B, num_action_tokens, cond_dim] or [B, cond_dim]
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
            
        c_emb = self.cond_proj(cond) # [B, num_action_tokens, D]
        
        # Prepend tokens: [Time, Cond, Points...]
        tokens = torch.cat([t_emb, c_emb, x_emb], dim=1) # [B, 1 + num_action_tokens + N, D]
        
        # Transformer
        out = self.transformer(tokens)
        
        # Extract points (skip Time and Cond)
        # The number of prefix tokens is 1 (time) + c_emb.shape[1]
        prefix_len = 1 + c_emb.shape[1]
        out_points = out[:, prefix_len:, :] # [B, N, D]
        
        # Project to velocity
        out_vel = self.output_proj(out_points) # [B, N, 2]
        
        return out_vel.reshape(B, self.action_dim)

class CenterHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim  * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class Pi0VLA(nn.Module):
    def __init__(self, 
                 model_id="Qwen/Qwen3-VL-4B-Instruct", 
                 action_dim=128, 
                 num_action_tokens=1,
                 codebook_size=0,
                 tokens_per_stroke=1,
                 freeze_vlm=False,
                 use_lora=False,
                 lora_rank=16,
                 lora_alpha=32,
                 lora_dropout=0.05,
                 device_map="auto",
                 torch_dtype=torch.bfloat16,
                 use_gradient_checkpointing=False):
        """
        Args:
            model_id: HuggingFace model ID for the VLM.
            action_dim: Dimension of the action space (e.g., flattened trajectory).
            num_action_tokens: Number of special action tokens to append.
            codebook_size: Size of VQ-VAE codebook (number of discrete tokens). If >0, adds code tokens.
            tokens_per_stroke: Number of code tokens that represent a stroke (e.g., 3 for VQ-VAE compression).
            freeze_vlm: Whether to freeze the VLM backbone.
            use_lora: Whether to use LoRA for VLM fine-tuning.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            device_map: Device map for the VLM.
            torch_dtype: Data type for the VLM.
            use_gradient_checkpointing: Whether to use gradient checkpointing for VLM.
        """
        super().__init__()
        self.num_action_tokens = num_action_tokens
        self.tokens_per_stroke = max(1, tokens_per_stroke)
        self.codebook_size = max(0, codebook_size)
        
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
            self.vlm.enable_input_require_grads()
            
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
        
        # Add special action tokens
        self.stroke_token = "<|stroke|>"
        self.stop_token = "<|stop|>"
        self.code_token_prefix = "<|code_"  # tokens like <|code_0000|>
        self.tokens_added = False
        self.code_token_ids = []
        self.stop_token_id = None
        if self.processor is not None:
            special_tokens = [self.stroke_token, self.stop_token]
            if self.codebook_size > 0:
                code_tokens = [f"{self.code_token_prefix}{i:04d}|>" for i in range(self.codebook_size)]
                special_tokens.extend(code_tokens)
            num_added = self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
            if num_added > 0:
                self.vlm.resize_token_embeddings(len(self.processor.tokenizer))
                self.tokens_added = True
                print(f"Added {num_added} special tokens: {special_tokens}")
            if self.codebook_size > 0:
                self.code_token_ids = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in code_tokens]
            self.stop_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.stop_token)

        self.freeze_vlm = freeze_vlm
        
        if use_lora:
            print(f"Applying LoRA to VLM (rank={lora_rank}, alpha={lora_alpha})...")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["embed_tokens", "lm_head"]
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            self.vlm.print_trainable_parameters()
            # When using LoRA, we are training the adapters, so we shouldn't treat VLM as fully frozen in logic
            self.freeze_vlm = False 
            self.use_lora_flag = True
            
        elif freeze_vlm:
            print("Freezing VLM parameters...")
            for param in self.vlm.parameters():
                param.requires_grad = False
            self.vlm.eval() # Set to eval mode
        
        if self.tokens_added:
            print("Unfreezing input embeddings for special tokens...")
            # We unfreeze the whole embedding layer to ensure the new tokens are trained.
            # Note: This might train existing embeddings too if optimizer includes them.
            # In train_vla.py, we filter params by requires_grad, so this works.
            self.vlm.get_input_embeddings().requires_grad_(True)

        # Get hidden size
        vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        
        # Action Head (Flow Matching)
        self.action_head = DiTFlowMatchingHead(
            action_dim=action_dim,
            hidden_dim=512,
            cond_dim=vlm_hidden_dim,
            num_layers=6,
            nhead=8
        )
        
        self.center_head = CenterHead(vlm_hidden_dim)
        
        self.action_dim = action_dim

    def train(self, mode=True):
        """
        Override train to ensure VLM stays in eval mode if frozen or using LoRA.
        """
        super().train(mode)
        
        # If we are freezing the VLM or using LoRA (where base model should be frozen),
        # we explicitly set VLM to eval mode.
        # Note: For LoRA, the adapters will be set to train by super().train(mode) 
        # because they are Modules, but we want the base model to be eval (for BatchNorm/Dropout).
        # However, peft handles adapter training state. We just need to ensure base model is eval.
        if self.freeze_vlm or getattr(self, "use_lora_flag", False):
            self.vlm.eval()

    def save_checkpoint(self, path):
        """
        Saves the model checkpoint. 
        If LoRA is used, saves adapters and action head.
        """
        save_dict = {
            'action_head': self.action_head.state_dict(),
            'center_head': self.center_head.state_dict()
        }
        save_dict['meta'] = {
            'codebook_size': self.codebook_size,
            'tokens_per_stroke': self.tokens_per_stroke,
            'stroke_token': self.stroke_token,
            'stop_token': self.stop_token,
        }
        if self.processor is not None:
            save_dict['tokenizer_added_vocab'] = self.processor.tokenizer.get_added_vocab()
        
        # Check if VLM is a PeftModel
        # We can check for 'peft_config' attribute or import PeftModel
        if hasattr(self.vlm, 'peft_config'):
            from peft import get_peft_model_state_dict
            save_dict['lora'] = get_peft_model_state_dict(self.vlm)
        else:
            # When training the full VLM (no LoRA), persist its weights to avoid losing fine-tuning.
            save_dict['vlm'] = self.vlm.state_dict()
            
        torch.save(save_dict, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        if 'action_head' in checkpoint:
            self.action_head.load_state_dict(checkpoint['action_head'])
        else:
            # Legacy support if checkpoint only had action head
            self.action_head.load_state_dict(checkpoint)
            
        if 'center_head' in checkpoint:
            self.center_head.load_state_dict(checkpoint['center_head'])
        
        if 'lora' in checkpoint and hasattr(self.vlm, 'peft_config'):
            from peft import set_peft_model_state_dict
            try:
                set_peft_model_state_dict(self.vlm, checkpoint['lora'])
                print("Successfully loaded LoRA adapters.")
            except Exception as e:
                print(f"Error loading LoRA adapters: {e}")
                print("Attempting to load with strict=False or ignoring mismatch if possible...")
                # In some versions set_peft_model_state_dict doesn't have strict arg, 
                # but we can try to debug keys.
                model_keys = set(self.vlm.state_dict().keys())
                ckpt_keys = set(checkpoint['lora'].keys())
                print(f"Model keys (sample): {list(model_keys)[:5]}")
                print(f"Ckpt keys (sample): {list(ckpt_keys)[:5]}")
                raise e
        elif 'vlm' in checkpoint:
            # Full-model fine-tuning path (no LoRA). Map to current device to keep memory contiguous.
            self.vlm.load_state_dict(checkpoint['vlm'], strict=False)
            print("Successfully loaded VLM weights from checkpoint.")

        # if 'tokenizer_added_vocab' in checkpoint and self.processor is not None:
        #     added = checkpoint['tokenizer_added_vocab']
        #     # This call is idempotent; tokenizer will not duplicate tokens
        #     self.processor.tokenizer.add_tokens(list(added.keys()), special_tokens=True)
        #     self.vlm.resize_token_embeddings(len(self.processor.tokenizer))
        #     if self.codebook_size > 0:
        #         code_tokens = [f"{self.code_token_prefix}{i:04d}|>" for i in range(self.codebook_size)]
        #         self.code_token_ids = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in code_tokens]

    def forward(self, inputs, actions=None, centers=None):
        """
        Training step.
        Args:
            inputs: Dict containing input_ids, pixel_values, etc.
            actions: [Total_Strokes, action_dim] - Ground truth actions
            centers: [Total_Strokes, 2] - Ground truth centers
        Returns:
            loss: Total loss
        """
        # 1. VLM Forward
        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        loss_vlm = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.vlm.device)
        
        if actions is None:
            return loss_vlm, {"loss_vlm": loss_vlm.item(), "loss_action": 0.0, "loss_center": 0.0}
            
        # 2. Extract embeddings for <|stroke|> tokens
        input_ids = inputs['input_ids']
        last_hidden_state = outputs.hidden_states[-1]

        # Build mask for either code tokens or legacy stroke token
        if self.code_token_ids:
            token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in self.code_token_ids:
                token_mask |= input_ids == tid
            tokens_per_stroke = self.tokens_per_stroke
        else:
            stroke_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.stroke_token)
            token_mask = input_ids == stroke_token_id
            tokens_per_stroke = 1

        token_embeddings = last_hidden_state[token_mask]

        if token_embeddings.shape[0] == 0:
            return loss_vlm, {"loss_vlm": loss_vlm.item()}

        expected_tokens = actions.shape[0] * tokens_per_stroke
        if token_embeddings.shape[0] < expected_tokens:
            # Trim actions to match available tokens
            max_strokes = token_embeddings.shape[0] // tokens_per_stroke
            actions = actions[:max_strokes]
            if centers is not None:
                centers = centers[:max_strokes]
            token_embeddings = token_embeddings[: max_strokes * tokens_per_stroke]
        elif token_embeddings.shape[0] > expected_tokens:
            token_embeddings = token_embeddings[:expected_tokens]

        if tokens_per_stroke > 1:
            token_embeddings = token_embeddings.view(-1, tokens_per_stroke, token_embeddings.shape[-1]).mean(dim=1)

        # Align dtypes to action/center heads to avoid mixed-precision matmul errors
        head_dtype = next(self.action_head.parameters()).dtype
        token_embeddings = token_embeddings.to(head_dtype)
        actions = actions.to(head_dtype)
        if centers is not None:
            centers = centers.to(head_dtype)

        # 3. Flow Matching Loss
        loss_action = self.compute_flow_matching_loss(token_embeddings, actions) * 3.0
        
        loss_center = 0.0
        if centers is not None:
            pred_centers = self.center_head(token_embeddings)
            loss_center = F.mse_loss(pred_centers, centers)
        
        return loss_vlm + loss_action + loss_center, {"loss_vlm": loss_vlm.item(), "loss_action": loss_action.item(), "loss_center": loss_center.item()}

    def compute_flow_matching_loss(self, cond, actions):
        B = actions.shape[0]
        device = actions.device
        dtype = actions.dtype
        
        t = torch.rand(B, device=device, dtype=dtype)
        x0 = torch.randn_like(actions)
        x1 = actions
        
        t_view = t.view(B, *([1] * (len(actions.shape) - 1)))
        xt = t_view * x1 + (1 - t_view) * x0
        v_target = x1 - x0
        
        v_pred = self.action_head(xt, t, cond)
        
        # 1. Velocity MSE Loss
        loss_mse = F.mse_loss(v_pred, v_target)
        
        # 2. Cosine Similarity Loss (Direction)
        loss_cos = 1.0 - F.cosine_similarity(v_pred, v_target, dim=-1, eps=1e-6).mean()
        
        # 3. Trajectory Integration Loss (Few-step lookahead)
        # Integrate for a few steps and compare with ground truth interpolation
        n_steps = 3
        dt = 0.1
        
        # Only apply to samples where integration stays within [0, 1]
        mask = (t + n_steps * dt) <= 1.0
        
        loss_traj = torch.tensor(0.0, device=device, dtype=dtype)
        
        if mask.any():
            # Filter samples
            x_curr = xt[mask]
            t_curr = t[mask]
            cond_curr = cond[mask] if cond.shape[0] == B else cond
            
            x1_curr = x1[mask]
            x0_curr = x0[mask]
            
            # Target at t + n_steps * dt
            t_final = t_curr + n_steps * dt
            t_final_view = t_final.view(-1, *([1] * (len(actions.shape) - 1)))
            x_target_final = t_final_view * x1_curr + (1 - t_final_view) * x0_curr
            
            # Euler Integration
            for _ in range(n_steps):
                v_step = self.action_head(x_curr, t_curr, cond_curr)
                x_curr = x_curr + v_step * dt
                t_curr = t_curr + dt
                
            loss_traj = F.mse_loss(x_curr, x_target_final)
            loss_traj_smooth_l1 = F.smooth_l1_loss(x_curr, x_target_final)
            
        # Weighted sum
        return loss_mse + 0.1 * loss_cos + 0.1 * loss_traj + 0.1 * loss_traj_smooth_l1

    @torch.no_grad()
    def generate_action(self, inputs, max_strokes=20, steps=10):
        """
        Autoregressive generation of strokes from predicted code tokens.
        """
        device = self.vlm.device
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        pixel_values = inputs.get('pixel_values', None)
        image_grid_thw = inputs.get('image_grid_thw', None)
        
        generated_strokes = []
        eos_token_id = self.processor.tokenizer.eos_token_id
        stop_token_id = self.stop_token_id if self.stop_token_id is not None else -1
        code_token_ids = self.code_token_ids if self.code_token_ids else [self.processor.tokenizer.convert_tokens_to_ids(self.stroke_token)]
        tokens_per_stroke = self.tokens_per_stroke if self.code_token_ids else 1

        stop_ids = {eos_token_id}
        if stop_token_id >= 0:
            stop_ids.add(stop_token_id)

        print(f"Stop token IDs: {stop_ids}")
        
        # 1. Pre-fill
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        curr_input_ids = next_token_id.unsqueeze(1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=1)

        token_buffer = []
        max_tokens = max_strokes * tokens_per_stroke

        token_id_list = []

        for _ in range(max_tokens):
            token_id = curr_input_ids[0, 0].item()
            token_id_list.append(token_id)
            if token_id in stop_ids:
                break

            outputs = self.vlm(
                input_ids=curr_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )

            past_key_values = outputs.past_key_values

            if token_id in code_token_ids:
                last_hidden_state = outputs.hidden_states[-1]  # [B, 1, Dim]
                token_buffer.append(last_hidden_state[:, -1, :])

                if len(token_buffer) == tokens_per_stroke:
                    cond = torch.stack(token_buffer, dim=1).mean(dim=1)
                    cond = cond.to(next(self.action_head.parameters()).dtype)
                    token_buffer = []

                    stroke = self.sample_flow_matching(cond, steps=steps)
                    center = self.center_head(cond)

                    B_size, dim = stroke.shape
                    stroke_reshaped = stroke.view(B_size, -1, 2)
                    center_reshaped = center.view(B_size, 1, 2)
                    stroke_restored = stroke_reshaped + center_reshaped
                    stroke = stroke_restored.view(B_size, dim)

                    generated_strokes.append(stroke.cpu())

                    if len(generated_strokes) >= max_strokes:
                        break

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            curr_input_ids = next_token_id.unsqueeze(1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=1)
        
        print(f"Predicted token IDs: {token_id_list}")

        return generated_strokes

    def sample_flow_matching(self, cond, steps=10):
        policy_param = next(self.action_head.parameters())
        device = policy_param.device
        dtype = policy_param.dtype
        
        B = cond.shape[0]
        x = torch.randn(B, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t = torch.ones(B, device=device, dtype=dtype) * t_val
            v = self.action_head(x, t, cond.to(dtype))
            x = x + v * dt
            
        return x
