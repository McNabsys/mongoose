"""Train the per-probe residual MLP (Direction C).

Standalone training entry point. Sibling of ``scripts/train.py``; uses
its own minimal training loop (no Trainer class) because the per-probe
MLP path is structurally different from the V1/V3/V4-A waveform U-Net
trainer (no waveforms, no soft-DTW, no warmstart schedule).

Modes (mutually exclusive):
    --remap-dir + --run-id : build the residual table on the fly from
                             a Remapped/AllCh/ directory.
    --residual-parquet     : load a pre-built parquet (one or more,
                             concatenated). Faster after the first run.

Either mode can be repeated to train across multiple runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from mongoose.data.residual_dataset import (
    ResidualDataset,
    add_molecule_aggregates,
    make_split,
)
from mongoose.etl.reads_maps_table import build_residual_table
from mongoose.model.residual_mlp import ResidualMLP


LOG = logging.getLogger("train_residual")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train per-probe residual MLP.")
    parser.add_argument(
        "--remap-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Path to a Remapped/AllCh/ directory. Repeat the flag for "
            "multiple runs. Each --remap-dir requires a matching --run-id "
            "in the same order."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        action="append",
        default=None,
        help="Run identifier (file prefix). Pair with --remap-dir.",
    )
    parser.add_argument(
        "--residual-parquet",
        type=Path,
        action="append",
        default=None,
        help=(
            "Pre-built residual-table parquet (from "
            "build_residual_table). Repeat for multiple runs."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap dataset size after concatenation.")
    parser.add_argument("--include-rejected", action="store_true",
                        help="Train on all probes, not just accepted.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--target-mode",
        choices=["production_residual", "ref_anchored_residual"],
        default="production_residual",
        help=(
            "Which column to use as the regression target. "
            "'production_residual' (default) -> target = post - pre, "
            "model mimics production, ceiling = production. "
            "'ref_anchored_residual' -> target = (ref - pre) anchored "
            "at the first matched probe, model trains against the genome "
            "directly, no a-priori ceiling (could beat production)."
        ),
    )
    parser.add_argument(
        "--holdout-run-id",
        type=str,
        action="append",
        default=None,
        help=(
            "Run identifier to EXCLUDE from training (held-out test set). "
            "Repeat the flag to hold out multiple runs. The model is "
            "trained on all other --run-id runs, then evaluated on the "
            "held-out runs at the end of training. Provides a clean "
            "out-of-distribution comparison vs production."
        ),
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--loss-type",
        choices=["mse", "huber"],
        default="mse",
        help=(
            "Regression loss. 'mse' is sensitive to outliers in the target "
            "distribution. 'huber' is L2 below --huber-delta and L1 above, "
            "more robust when the target has heavy tails (e.g. ref-anchored "
            "residuals can range to +/- 50 kbp)."
        ),
    )
    parser.add_argument("--huber-delta", type=float, default=500.0,
                        help="Threshold for Huber loss (only with --loss-type huber).")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])

    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("residual_checkpoints"))
    parser.add_argument("--save-every", type=int, default=2,
                        help="Save a checkpoint every N epochs.")

    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log a per-batch line every N batches.")
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_residual_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.residual_parquet:
        frames = [pd.read_parquet(p) for p in args.residual_parquet]
        df = pd.concat(frames, ignore_index=True)
        LOG.info("loaded %d rows from %d parquets",
                 len(df), len(args.residual_parquet))
        return df

    if not args.remap_dir or not args.run_id:
        raise SystemExit(
            "error: must pass either --residual-parquet, or matched "
            "--remap-dir/--run-id pairs."
        )
    if len(args.remap_dir) != len(args.run_id):
        raise SystemExit(
            f"error: --remap-dir count ({len(args.remap_dir)}) does not match "
            f"--run-id count ({len(args.run_id)})."
        )

    frames = []
    for remap_dir, run_id in zip(args.remap_dir, args.run_id):
        LOG.info("building residual table for run_id=%s", run_id)
        frames.append(build_residual_table(remap_dir, run_id))
    df = pd.concat(frames, ignore_index=True)
    LOG.info("built residual table with %d rows across %d runs",
             len(df), len(frames))
    return df


def _per_decile_residual_mae(
    pred: np.ndarray, target: np.ndarray, n_deciles: int = 10
) -> dict[int, float]:
    """Compute per-decile MAE stratified by ``|target|`` magnitude.

    Helps surface whether the model is wrong about big shifts more than
    small ones (the head-dive / TVC asymptote regime).
    """
    if len(pred) == 0:
        return {}
    abs_target = np.abs(target)
    cuts = np.quantile(abs_target, np.linspace(0, 1, n_deciles + 1))
    out: dict[int, float] = {}
    for d in range(n_deciles):
        lo = cuts[d]
        hi = cuts[d + 1] if d < n_deciles - 1 else np.inf
        mask = (abs_target >= lo) & (abs_target <= hi if d == n_deciles - 1 else abs_target < hi)
        if mask.sum() == 0:
            continue
        mae = float(np.abs(pred[mask] - target[mask]).mean())
        out[d] = mae
    return out


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    LOG.info("device=%s", device)

    df = _load_residual_table(args)
    if args.max_rows is not None and args.max_rows < len(df):
        rng = np.random.default_rng(args.split_seed)
        idx = rng.choice(len(df), size=args.max_rows, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        LOG.info("subsampled to %d rows", len(df))

    df = add_molecule_aggregates(df)

    # Held-out per-run split: separate the holdout runs entirely so they
    # never influence the trained model. We retain them in a
    # ``held_out_df`` for end-of-training evaluation.
    holdout_set = set(args.holdout_run_id or [])
    if holdout_set:
        held_out_df = df[df["run_id"].isin(holdout_set)].reset_index(drop=True)
        df = df[~df["run_id"].isin(holdout_set)].reset_index(drop=True)
        LOG.info(
            "holdout split: %d rows in %d holdout runs, %d rows for training",
            len(held_out_df), len(holdout_set), len(df),
        )
        held_out_runs_seen = set(held_out_df["run_id"].unique())
        missing = holdout_set - held_out_runs_seen
        if missing:
            LOG.warning("holdout run_ids not present in data: %s", sorted(missing))
    else:
        held_out_df = None

    target_column = (
        "ref_anchored_residual_bp"
        if args.target_mode == "ref_anchored_residual"
        else "residual_bp"
    )
    require_reference = args.target_mode == "ref_anchored_residual"
    LOG.info("target_mode=%s -> column=%s, require_reference=%s",
             args.target_mode, target_column, require_reference)

    accepted_only = not args.include_rejected
    full_ds = ResidualDataset(
        df,
        accepted_only=accepted_only,
        target_column=target_column,
        require_reference=require_reference,
    )
    LOG.info("dataset size after filter: %d probes (accepted_only=%s)",
             len(full_ds), accepted_only)

    if len(full_ds) < 2:
        raise SystemExit("error: dataset has fewer than 2 samples; nothing to train.")

    split = make_split(len(full_ds), val_fraction=args.val_fraction,
                       seed=args.split_seed)
    train_ds = Subset(full_ds, split.train_indices.tolist())
    val_ds = Subset(full_ds, split.val_indices.tolist())
    LOG.info("train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # Set the head bias to the empirical training mean so the model
    # starts at a calibrated baseline (instead of the hard-coded +2200).
    # This is the simplest variance-reduction trick available before
    # the model learns anything.
    train_targets = full_ds.target[torch.from_numpy(split.train_indices)]
    target_mean = float(train_targets.mean().item())
    target_std = float(train_targets.std().item())
    LOG.info("training target stats: mean=%+.1f bp, std=%.1f bp",
             target_mean, target_std)

    model = ResidualMLP(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)
    torch.nn.init.constant_(model.head.bias, target_mean)
    LOG.info("model parameters: %d", model.num_parameters)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        n_batches = 0
        for i, (feats, target) in enumerate(train_loader):
            feats = feats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(feats)
            if args.loss_type == "huber":
                loss = torch.nn.functional.huber_loss(
                    pred, target, delta=args.huber_delta
                )
            else:
                loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
            if (i + 1) % args.log_interval == 0:
                LOG.info(
                    "epoch %d batch %d/%d  train_mse=%.1f  rmse=%.1f bp",
                    epoch, i + 1, len(train_loader),
                    total / n_batches, math.sqrt(total / n_batches),
                )
        train_mse = total / max(n_batches, 1)
        train_rmse = math.sqrt(train_mse)

        # Validation
        model.eval()
        val_preds: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for feats, target in val_loader:
                feats = feats.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(feats)
                val_total += float(torch.nn.functional.mse_loss(pred, target).item())
                val_batches += 1
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(target.detach().cpu().numpy())
        val_mse = val_total / max(val_batches, 1)
        val_rmse = math.sqrt(val_mse)
        val_pred_arr = np.concatenate(val_preds) if val_preds else np.array([])
        val_target_arr = np.concatenate(val_targets) if val_targets else np.array([])
        val_mae = float(np.abs(val_pred_arr - val_target_arr).mean()) if len(val_pred_arr) else float("nan")

        # Naive baseline: always predict the training mean.
        if len(val_target_arr):
            baseline_mae = float(np.abs(val_target_arr - target_mean).mean())
        else:
            baseline_mae = float("nan")
        per_decile = _per_decile_residual_mae(val_pred_arr, val_target_arr)

        scheduler.step()
        elapsed = time.time() - t0
        LOG.info(
            "[epoch %d/%d done in %.1fs] train_rmse=%.1f bp  val_rmse=%.1f bp  "
            "val_mae=%.1f bp  baseline_mae=%.1f bp  ratio=%.3f",
            epoch + 1, args.epochs, elapsed, train_rmse, val_rmse,
            val_mae, baseline_mae,
            val_mae / baseline_mae if baseline_mae > 0 else float("nan"),
        )

        epoch_record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "baseline_mae": baseline_mae,
            "per_decile_mae": per_decile,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ckpt_path = args.checkpoint_dir / f"residual_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": epoch_record,
                    "args": vars(args),
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                ckpt_path,
            )
            LOG.info("saved checkpoint: %s", ckpt_path)

    history_path = args.checkpoint_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    LOG.info("training complete; history at %s", history_path)

    # End-of-training held-out evaluation.
    if held_out_df is not None and len(held_out_df) > 0:
        LOG.info("=== HELD-OUT EVALUATION ===")
        held_out_ds = ResidualDataset(
            held_out_df,
            accepted_only=accepted_only,
            target_column=target_column,
            require_reference=require_reference,
            compute_aggregates=False,  # already computed
        )
        LOG.info("held-out dataset size: %d probes", len(held_out_ds))
        ho_loader = DataLoader(
            held_out_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        model.eval()
        ho_preds: list[np.ndarray] = []
        ho_targets: list[np.ndarray] = []
        with torch.no_grad():
            for feats, target in ho_loader:
                feats = feats.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(feats)
                ho_preds.append(pred.detach().cpu().numpy())
                ho_targets.append(target.detach().cpu().numpy())
        ho_pred = np.concatenate(ho_preds) if ho_preds else np.array([])
        ho_target = np.concatenate(ho_targets) if ho_targets else np.array([])

        ho_mae = float(np.abs(ho_pred - ho_target).mean()) if len(ho_pred) else float("nan")
        ho_rmse = float(np.sqrt(((ho_pred - ho_target) ** 2).mean())) if len(ho_pred) else float("nan")
        ho_baseline_mae = (
            float(np.abs(ho_target - target_mean).mean()) if len(ho_target) else float("nan")
        )
        ho_per_decile = _per_decile_residual_mae(ho_pred, ho_target)

        # Per-run breakdown over the held-out set.
        held_out_df_filtered = held_out_df.copy()
        if accepted_only:
            held_out_df_filtered = held_out_df_filtered[held_out_df_filtered["accepted"]]
        if require_reference:
            held_out_df_filtered = held_out_df_filtered[held_out_df_filtered["has_reference"]]
        held_out_df_filtered = held_out_df_filtered.reset_index(drop=True)
        per_holdout_run = {}
        if len(held_out_df_filtered) == len(ho_pred):
            held_out_df_filtered["pred_residual"] = ho_pred
            for run_id, group in held_out_df_filtered.groupby("run_id", sort=False):
                idx = group.index.to_numpy()
                preds = ho_pred[idx]
                tgts = ho_target[idx]
                per_holdout_run[run_id] = {
                    "n_probes": int(len(idx)),
                    "mae_bp": float(np.abs(preds - tgts).mean()),
                    "rmse_bp": float(np.sqrt(((preds - tgts) ** 2).mean())),
                }
        LOG.info(
            "[HOLDOUT FINAL] mae=%.1f bp  rmse=%.1f bp  baseline_mae=%.1f bp  ratio=%.3f",
            ho_mae, ho_rmse, ho_baseline_mae,
            ho_mae / ho_baseline_mae if ho_baseline_mae > 0 else float("nan"),
        )
        for run_id, stats in per_holdout_run.items():
            LOG.info("  %s: n=%d mae=%.1f bp rmse=%.1f bp",
                     run_id, stats["n_probes"], stats["mae_bp"], stats["rmse_bp"])

        held_out_record = {
            "n_probes": int(len(ho_pred)),
            "mae_bp": ho_mae,
            "rmse_bp": ho_rmse,
            "baseline_mae_bp": ho_baseline_mae,
            "ratio": (ho_mae / ho_baseline_mae) if ho_baseline_mae > 0 else float("nan"),
            "per_decile_mae": ho_per_decile,
            "per_run": per_holdout_run,
        }
        held_out_path = args.checkpoint_dir / "held_out_eval.json"
        with open(held_out_path, "w") as f:
            json.dump(held_out_record, f, indent=2, default=str)
        LOG.info("wrote held-out eval: %s", held_out_path)
    else:
        LOG.info("no holdout runs specified; skipping held-out eval")


if __name__ == "__main__":
    main()
