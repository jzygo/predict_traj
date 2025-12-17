from dataclasses import dataclass, field


@dataclass
class DimConfig:
    num_points: int = 64
    latent_dim: int = 128
    model_dim: int = 512
    image_size: int = 256
    max_strokes: int = 32
    point_dim: int = 3  # x, y, r (r: 最大纯黑圆半径)


@dataclass
class VAEConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    num_layers: int = 3
    latent_dim: int = 128
    # Transformer-style VAE enhancements (keep defaults for backward compat)
    num_heads: int = 8
    ff_mult: int = 4


@dataclass
class DiTConfig:
    patch_size: int = 16
    depth: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    use_pretrained_cnn: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    stage: int = 1
    resume: str = ""
    use_amp: bool = False  # 启用混合精度训练 (Automatic Mixed Precision)


@dataclass
class DistConfig:
    ddp: bool = False
    min_free_ratio: float = 0.7


@dataclass
class PipelineConfig:
    data_root: str = ""
    device: str = "cuda"
    dims: DimConfig = field(default_factory=DimConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dist: DistConfig = field(default_factory=DistConfig)
