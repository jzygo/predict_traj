# Visual Action Tokenizer (VAT) for Calligraphy

This project implements a Visual Action Tokenizer to compress calligraphy video sequences into compact "Action Tokens" (Z_action) and reconstruct them using a Diffusion Transformer (DiT).

## Project Structure

*   `src/models/`: Contains the model architecture.
    *   `encoder.py`: Video Encoder with C3-inspired Stroke Compressor.
    *   `decoder.py`: DiT-based Decoder with Cross-Attention conditioning.
    *   `tokenizer.py`: Main wrapper class.
*   `src/losses/`: Loss functions.
    *   `composite_loss.py`: Reconstruction + Optical Flow + Perceptual Loss.
*   `src/data/`: Data loading.
    *   `dataset.py`: Calligraphy video dataset loader.
*   `scripts/`: Training scripts.
    *   `train_ddp.py`: Distributed Data Parallel training script.
*   `configs/`: Configuration files.
    *   `train_config.yaml`: Hyperparameters.

## Requirements

*   Python 3.8+
*   PyTorch 1.10+
*   torchvision
*   einops
*   pyyaml
*   tqdm
*   lpips (optional, for perceptual loss)

## Usage

1.  **Prepare Data**: Place your calligraphy videos in `data/videos` (or update config).
2.  **Configure**: Edit `configs/train_config.yaml` to set batch size, learning rate, etc.
3.  **Train**:
    Run the DDP training script. Adjust `nproc_per_node` based on your GPU count (e.g., 8).

    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 scripts/train_ddp.py --config configs/train_config.yaml
    ```
    
    Or simply run the script directly if using `mp.spawn` inside (as implemented):
    
    ```bash
    python scripts/train_ddp.py --config configs/train_config.yaml
    ```

## Key Features

*   **Trajectory-Free Learning**: Learns dynamics purely from pixels.
*   **Action Compression**: Compresses video into 32 learnable tokens.
*   **Optical Flow Consistency**: Enforces motion dynamics in the latent space.
