"""Argument parsing and TrainConfig assembly for the train CLI.

Split from scripts/train.py so it can be unit-tested without invoking a
subprocess. scripts/train.py is a thin wrapper around this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mongoose.training.config import TrainConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the train CLI.

    Kept separate from config construction so tests can parse argv without
    triggering SystemExit from validation.
    """
    parser = argparse.ArgumentParser(description="Train T2D U-Net")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (mutually exclusive with --cache-dir)",
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dirs",
        action="append",
        default=None,
        type=Path,
        help=(
            "Path to a preprocessed cache directory (repeat the flag to "
            "train on multiple runs). Mutually exclusive with --synthetic."
        ),
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help=(
            "Cap the dataset to this many molecules (random subsample). "
            "Useful for CPU smoke training."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of dataset used for validation split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/val split and subsampling.",
    )
    parser.add_argument("--augment-train", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--warmstart-epochs",
        type=int,
        default=None,
        help="Override warmstart_epochs (use 0 to skip warmstart)",
    )
    parser.add_argument(
        "--warmstart-fade-epochs",
        type=int,
        default=None,
        help="Override warmstart_fade_epochs",
    )
    parser.add_argument(
        "--synthetic-num-molecules",
        type=int,
        default=None,
        help="Override number of synthetic molecules (only with --synthetic)",
    )
    parser.add_argument(
        "--min-blend",
        type=float,
        default=None,
        help="Floor for the focal-vs-peakiness blend; 0.1 keeps focal supervision on forever.",
    )
    parser.add_argument(
        "--scale-bp",
        type=float,
        default=None,
        help="Divisor applied to raw bp_loss for per-component gradient balance.",
    )
    parser.add_argument("--scale-vel", type=float, default=None)
    parser.add_argument("--scale-count", type=float, default=None)
    parser.add_argument("--scale-probe", type=float, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a validated TrainConfig from parsed CLI args.

    Raises ``SystemExit`` (via ``argparse.ArgumentParser.error``-style
    behavior) on misconfigurations so the CLI prints a friendly error.
    """
    use_synthetic: bool = bool(args.synthetic)
    cache_dirs: list[Path] | None = args.cache_dirs

    if use_synthetic and cache_dirs:
        raise SystemExit(
            "error: --synthetic and --cache-dir are mutually exclusive"
        )
    if not use_synthetic and not cache_dirs:
        raise SystemExit(
            "error: must pass either --synthetic or at least one --cache-dir"
        )

    if cache_dirs:
        for cd in cache_dirs:
            if not cd.exists():
                raise SystemExit(
                    f"error: cache dir does not exist: {cd}"
                )
            if not cd.is_dir():
                raise SystemExit(f"error: cache dir is not a directory: {cd}")

    config = TrainConfig(
        use_synthetic=use_synthetic,
        cache_dirs=cache_dirs,
        max_molecules=args.max_molecules,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        augment_train=args.augment_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        checkpoint_dir=Path(args.checkpoint_dir),
        save_every=args.save_every,
    )
    if args.warmstart_epochs is not None:
        config.warmstart_epochs = args.warmstart_epochs
    if args.warmstart_fade_epochs is not None:
        config.warmstart_fade_epochs = args.warmstart_fade_epochs
    if args.synthetic_num_molecules is not None:
        config.synthetic_num_molecules = args.synthetic_num_molecules
    if args.min_blend is not None:
        config.min_blend = args.min_blend
    if args.scale_bp is not None:
        config.scale_bp = args.scale_bp
    if args.scale_vel is not None:
        config.scale_vel = args.scale_vel
    if args.scale_count is not None:
        config.scale_count = args.scale_count
    if args.scale_probe is not None:
        config.scale_probe = args.scale_probe

    return config


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Parses argv, builds config, runs training."""
    from mongoose.training.trainer import Trainer  # local import to keep CLI imports light

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    trainer = Trainer(config)
    trainer.fit()
