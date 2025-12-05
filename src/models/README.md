# VLA Model (Pi0 Mimic)

This directory contains the implementation of a Vision-Language-Action (VLA) model mimicking the $\pi_0$ architecture.

## Structure

- **VLM Backbone**: Uses `Qwen/Qwen3-VL-4B-Instruct` (or compatible) to encode visual and textual inputs.
- **Action Head**: Uses a Flow Matching policy (Conditional Flow Matching) to generate continuous actions (trajectories).

## Usage

```python
import torch
from src.models.vla import Pi0VLA

# Initialize model
# Note: This requires downloading the Qwen model (approx 8GB+).
model = Pi0VLA(model_id="Qwen/Qwen3-VL-4B-Instruct", action_dim=256)

# Dummy Data
B = 2
input_ids = torch.randint(0, 1000, (B, 128))
pixel_values = torch.randn(B, 3, 256, 256) # Adjust shape based on Qwen processor
actions = torch.randn(B, 256) # Ground truth actions (flattened trajectory)

# Training Forward Pass
loss = model(input_ids, pixel_values, actions)
loss.backward()

# Inference
generated_actions = model.generate_action(input_ids, pixel_values, steps=20)
```

## Notes

- The `action_dim` should match the flattened size of your trajectory representation (e.g., `num_points * 2` or `num_points * 3`).
- You may need to adjust `pixel_values` and `image_grid_thw` handling depending on the exact version of `transformers` and the Qwen model.
