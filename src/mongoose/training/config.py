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
    # Real-data cache directories (list of preprocessed cache dirs). When
    # use_synthetic=False, at least one cache_dir is required.
    cache_dirs: list[Path] | None = None
    # If set, cap the combined dataset to this many molecules after
    # concatenation (useful for CPU smoke training on a subsample).
    max_molecules: int | None = None
    # Fraction of the dataset used for validation (80/20 default).
    val_fraction: float = 0.2
    # Random seed for the train/val split.
    split_seed: int = 42
    # Whether to apply data augmentations to the training split.
    augment_train: bool = False

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
    min_blend: float = 0.0
    scale_probe: float = 1.0
    scale_bp: float = 1.0
    scale_vel: float = 1.0
    scale_count: float = 1.0

    # V3 spike: physics-informed loss switch. When True the trainer uses
    # ``L511Loss`` (L_511 + L_smooth + L_length) in place of CombinedLoss.
    use_l511: bool = False
    lambda_511: float = 1.0
    lambda_smooth: float = 0.001
    lambda_length: float = 0.5

    # Option A (T2D-hybrid): when True, the model's velocity head output is
    # reinterpreted as a tanh-bounded residual modulating a physics-informed
    # v_T2D baseline. Requires caches enriched by precompute_t2d_params.py.
    # Composes cleanly with use_l511 (both supervise the composed velocity).
    use_t2d_hybrid: bool = False

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 5  # epochs
    # Optional warm-start: load ONLY model weights from this path, then
    # train from epoch 0 with a fresh optimizer and scheduler. Distinct
    # from auto-resume (which loads all state from checkpoint_dir). Use
    # this to continue a run with a different loss / epoch count / schedule
    # while keeping the pre-trained features.
    init_from: Path | None = None

    # Synthetic data (for testing)
    use_synthetic: bool = False
    synthetic_num_molecules: int = 100
    synthetic_min_length: int = 1000
    synthetic_max_length: int = 4000
