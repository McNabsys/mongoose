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

    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
