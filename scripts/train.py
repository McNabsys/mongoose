"""Train the T2D U-Net model."""

from __future__ import annotations

import argparse
from pathlib import Path

from mongoose.training.config import TrainConfig
from mongoose.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train T2D U-Net")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--warmstart-epochs",
        type=int,
        default=None,
        help="Override warmstart_epochs in config (use 0 to skip warmstart)",
    )
    parser.add_argument(
        "--warmstart-fade-epochs",
        type=int,
        default=None,
        help="Override warmstart_fade_epochs in config",
    )
    parser.add_argument(
        "--synthetic-num-molecules",
        type=int,
        default=None,
        help="Override number of synthetic molecules (only used with --synthetic)",
    )
    args = parser.parse_args()

    config = TrainConfig(
        use_synthetic=args.synthetic,
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

    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
