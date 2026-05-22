"""
All hyperparameters and constants for the InfiniteRace world model.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 2
    latent_channels: int = 4
    image_size: int = 256
    latent_size: int = 32       # image_size // 8
    vae_checkpoint: str = "stabilityai/sd-vae-ft-mse"
    unet_checkpoint: str = "runwayml/stable-diffusion-v1-5"


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""
    model: ModelConfig = None       # type: ignore[assignment]
    batch_size: int = 16
    lr: float = 1e-4
    steps: int = 50_000
    warmup_steps: int = 1_000
    data_dir: str = "training_data/"
    checkpoint_dir: str = "checkpoints/"
    log_every: int = 100
    save_every: int = 5_000
    grad_clip: float = 1.0
    mixed_precision: bool = True
    val_every: int = 1_000          # run validation every N steps (0 = disabled)
    early_stop_patience: int = 0    # stop if val LPIPS doesn't improve for N steps (0 = disabled)

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()


@dataclass
class DistillConfig:
    """LCM consistency distillation hyperparameters."""
    checkpoint: str = "checkpoints/level3_final.pt"
    lr_distill: float = 5e-6
    steps: int = 20_000
    skip_steps: int = 500           # t1 - t2
    num_train_timesteps: int = 1000
    ema_decay: float = 0.999
    batch_size: int = 8
    data_dir: str = "training_data/"
    checkpoint_dir: str = "checkpoints/"
    mixed_precision: bool = True


@dataclass
class InferenceConfig:
    """Real-time inference settings."""
    checkpoint: str = "checkpoints/lcm_final.pt"
    steps: int = 2
    queue_maxsize: int = 4
    jitter_maxlen: int = 4
    device: str = "cuda"
    dtype: str = "float16"      # fp16 at inference
