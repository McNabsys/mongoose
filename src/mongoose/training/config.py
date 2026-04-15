"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration for the T2D U-Net training loop."""

    # Data
    data_dir: Path = Path("data")
    num_workers: int = 0

    # Model
    in_channels: int = 1
    conditioning_dim: int = 6

    # Training
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    # Loss
    lambda_bp: float = 1.0
    lambda_vel: float = 1.0
    lambda_count: float = 1.0
    warmup_epochs: int = 5
    warmstart_epochs: int = 5
    warmstart_fade_epochs: int = 2
    softdtw_gamma: float = 0.1
    peakiness_window: int = 20
    nms_threshold: float = 0.3

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 5  # epochs

    # Synthetic data (for testing)
    use_synthetic: bool = False
    synthetic_num_molecules: int = 100
    synthetic_min_length: int = 1000
    synthetic_max_length: int = 4000
