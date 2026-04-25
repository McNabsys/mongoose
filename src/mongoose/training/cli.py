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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker-process count. Default from TrainConfig (0). "
            "Setting >0 moves data prep off the main process so the GPU "
            "doesn't wait on IO; try 4-8 on multi-core cloud VMs."
        ),
    )
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
    parser.add_argument(
        "--scale-vel",
        type=float,
        default=None,
        help="Divisor applied to raw vel_loss for per-component gradient balance.",
    )
    parser.add_argument(
        "--scale-count",
        type=float,
        default=None,
        help="Divisor applied to raw count_loss for per-component gradient balance.",
    )
    parser.add_argument(
        "--scale-probe",
        type=float,
        default=None,
        help="Divisor applied to raw probe_loss for per-component gradient balance.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Override the cosine scheduler's minimum LR (default: 1e-6 from TrainConfig).",
    )
    parser.add_argument(
        "--use-l511",
        action="store_true",
        help=(
            "V3 spike: replace CombinedLoss (soft-DTW + teacher-forced L_vel) "
            "with L511Loss (physics-informed L_511 + L_smooth + L_length). "
            "Probe head still warmstarts from wfmproc; velocity head is now "
            "supervised purely by the 511-bp integral constraint."
        ),
    )
    parser.add_argument(
        "--lambda-511",
        type=float,
        default=None,
        help="Weight on the L_511 term (only meaningful with --use-l511).",
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=None,
        help="Weight on L_smooth velocity-TV regularizer (only with --use-l511).",
    )
    parser.add_argument(
        "--lambda-length",
        type=float,
        default=None,
        help="Weight on L_length span anchor (only with --use-l511).",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Path to a checkpoint to warm-start from. Loads only "
            "model_state_dict; optimizer and scheduler start fresh, and "
            "training begins at epoch 0. Ignored if --checkpoint-dir already "
            "contains checkpoints (auto-resume wins). Use to extend a spike "
            "run with a new cosine schedule, or to start V3 from V1 features."
        ),
    )
    parser.add_argument(
        "--use-t2d-hybrid",
        action="store_true",
        help=(
            "Option A (T2D-hybrid): velocity head output is reinterpreted as "
            "a tanh-bounded residual modulating a physics-informed v_T2D "
            "baseline. Requires each --cache-dir to have a t2d_params.npy "
            "produced by scripts/precompute_t2d_params.py."
        ),
    )
    parser.add_argument(
        "--probe-aware-velocity",
        action="store_true",
        help=(
            "Architecture-level: feed the (sigmoided, detached) probe "
            "heatmap as an additional input channel into the velocity "
            "head. Helps it learn probe-state-conditioned corrections. "
            "Old V1 / spike checkpoints can't warm-start when this flips."
        ),
    )
    parser.add_argument(
        "--lambda-align",
        type=float,
        default=None,
        help=(
            "Mixed-supervision alignment loss weight. When > 0 and the "
            "molecule's match ratio >= --align-min-confidence, an L1 loss "
            "is added on cum_bp at probe centers vs reference_bp_positions. "
            "0 disables (default)."
        ),
    )
    parser.add_argument(
        "--align-min-confidence",
        type=float,
        default=None,
        help=(
            "Min match ratio (matched/total ref probes) for a molecule to "
            "receive alignment supervision. Default 0.7."
        ),
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help=(
            "Stream per-epoch metrics, hyperparameters, git commit hash, and "
            "GPU info to a wandb.ai dashboard. Requires the WANDB_API_KEY env "
            "var (get one at https://wandb.ai/authorize). Off by default — "
            "existing offline runs are unchanged when this flag is omitted."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="wandb project name (default: 'mongoose-v3'). Only used with --use-wandb.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=(
            "Name for this wandb run. If omitted, wandb auto-generates one. "
            "Only used with --use-wandb."
        ),
    )
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
    if args.num_workers is not None:
        config.num_workers = args.num_workers
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
    if args.min_lr is not None:
        config.min_lr = args.min_lr
    if args.use_l511:
        config.use_l511 = True
    if args.lambda_511 is not None:
        config.lambda_511 = args.lambda_511
    if args.lambda_smooth is not None:
        config.lambda_smooth = args.lambda_smooth
    if args.lambda_length is not None:
        config.lambda_length = args.lambda_length
    if args.init_from is not None:
        if not args.init_from.exists():
            raise SystemExit(f"error: --init-from path does not exist: {args.init_from}")
        config.init_from = Path(args.init_from)
    if args.use_t2d_hybrid:
        config.use_t2d_hybrid = True
        # Verify each cache dir has a t2d_params.npy — fail fast rather than
        # silently collapsing to non-hybrid at batch time.
        if cache_dirs is not None:
            for cd in cache_dirs:
                if not (cd / "t2d_params.npy").exists():
                    raise SystemExit(
                        f"error: --use-t2d-hybrid requires t2d_params.npy in "
                        f"each cache dir; missing at {cd}. Run "
                        f"scripts/precompute_t2d_params.py first."
                    )
    if args.probe_aware_velocity:
        config.probe_aware_velocity = True
    if args.lambda_align is not None:
        config.lambda_align = args.lambda_align
    if args.align_min_confidence is not None:
        config.align_min_confidence = args.align_min_confidence
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name

    return config


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Parses argv, builds config, runs training."""
    from mongoose.training.trainer import Trainer  # local import to keep CLI imports light

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    trainer = Trainer(config)
    trainer.fit()
