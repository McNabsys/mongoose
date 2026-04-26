"""V5 Phase 2: train the per-molecule sequence model.

Sibling of ``scripts/train_residual.py`` adapted for the sequence
model. Each batch is a set of variable-length molecule sequences
collated to a common max-K with a padding mask. Loss is masked so
padded positions don't contribute to gradients.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mongoose.data.residual_dataset import add_molecule_aggregates, make_split
from mongoose.data.sequence_dataset import (
    MoleculeSequenceDataset,
    collate_molecules,
)
from mongoose.etl.reads_maps_table import build_residual_table
from mongoose.model.sequence_residual import (
    DEFAULT_MAX_SEQ_LEN,
    SequenceResidualModel,
)


LOG = logging.getLogger("train_sequence")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train per-molecule sequence model.")
    p.add_argument("--remap-dir", type=Path, action="append", default=None)
    p.add_argument("--run-id", type=str, action="append", default=None)
    p.add_argument("--residual-parquet", type=Path, action="append", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--include-rejected", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--target-mode",
                   choices=["production_residual", "ref_anchored_residual"],
                   default="ref_anchored_residual")
    p.add_argument("--holdout-run-id", type=str, action="append", default=None)
    p.add_argument("--min-probes", type=int, default=2)
    p.add_argument("--max-probes", type=int, default=DEFAULT_MAX_SEQ_LEN,
                   help="Drop molecules with > this many probes.")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Number of MOLECULES per batch (not probes).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--loss-type", choices=["mse", "huber"], default="huber")
    p.add_argument("--huber-delta", type=float, default=500.0)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("sequence_checkpoints"))
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--log-interval", type=int, default=100)
    return p


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_residual_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.residual_parquet:
        frames = [pd.read_parquet(p) for p in args.residual_parquet]
        df = pd.concat(frames, ignore_index=True)
        LOG.info("loaded %d rows from %d parquets", len(df), len(args.residual_parquet))
        return df
    if not args.remap_dir or not args.run_id:
        raise SystemExit(
            "must pass --residual-parquet or matched --remap-dir/--run-id pairs")
    if len(args.remap_dir) != len(args.run_id):
        raise SystemExit("--remap-dir count must match --run-id count")
    frames = []
    for remap_dir, run_id in zip(args.remap_dir, args.run_id):
        LOG.info("building residual table for run_id=%s", run_id)
        frames.append(build_residual_table(remap_dir, run_id))
    df = pd.concat(frames, ignore_index=True)
    LOG.info("built residual table: %d rows across %d runs", len(df), len(frames))
    return df


def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``t`` over positions where ``mask`` is True (i.e. real probes)."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    masked = t * mask.to(t.dtype)
    return masked.sum() / mask.sum().clamp(min=1).to(t.dtype)


def _compute_loss(
    pred: torch.Tensor, target: torch.Tensor,
    real_mask: torch.Tensor, args: argparse.Namespace,
) -> torch.Tensor:
    """Loss over real probes only (real_mask True at real positions)."""
    if args.loss_type == "huber":
        per_probe = F.huber_loss(pred, target, delta=args.huber_delta, reduction="none")
    else:
        per_probe = (pred - target) ** 2
    return _masked_mean(per_probe, real_mask)


def _compute_residual_mae(
    pred: np.ndarray, target: np.ndarray, real_mask: np.ndarray,
) -> float:
    """MAE over real probes only."""
    diffs = np.abs(pred - target)
    masked = diffs[real_mask]
    return float(masked.mean()) if masked.size > 0 else float("nan")


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

    holdout_set = set(args.holdout_run_id or [])
    if holdout_set:
        held_out_df = df[df["run_id"].isin(holdout_set)].reset_index(drop=True)
        df = df[~df["run_id"].isin(holdout_set)].reset_index(drop=True)
        LOG.info("holdout split: %d held-out rows, %d for training",
                 len(held_out_df), len(df))
    else:
        held_out_df = None

    target_column = (
        "ref_anchored_residual_bp"
        if args.target_mode == "ref_anchored_residual"
        else "residual_bp"
    )
    require_reference = args.target_mode == "ref_anchored_residual"
    accepted_only = not args.include_rejected
    LOG.info("target=%s require_reference=%s accepted_only=%s",
             target_column, require_reference, accepted_only)

    full_ds = MoleculeSequenceDataset(
        df,
        accepted_only=accepted_only,
        require_reference=require_reference,
        target_column=target_column,
        compute_aggregates=False,
        min_probes=args.min_probes,
        max_probes=args.max_probes,
    )
    LOG.info("molecules in training set: %d", len(full_ds))
    if len(full_ds) < 2:
        raise SystemExit("dataset has fewer than 2 molecules; nothing to train.")

    split = make_split(len(full_ds), val_fraction=args.val_fraction,
                       seed=args.split_seed)
    train_ds = Subset(full_ds, split.train_indices.tolist())
    val_ds = Subset(full_ds, split.val_indices.tolist())
    LOG.info("train molecules=%d val molecules=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=collate_molecules,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_molecules,
    )

    # Compute training-target mean over REAL probes (excluding 0-padded)
    # for head bias init.
    all_targets: list[float] = []
    for idx in split.train_indices.tolist():
        rec = full_ds.molecules[idx]
        all_targets.extend(rec.target.numpy().tolist())
    target_mean = float(np.mean(all_targets)) if all_targets else 0.0
    target_std = float(np.std(all_targets)) if all_targets else 1.0
    LOG.info("training target stats: mean=%+.1f bp std=%.1f bp",
             target_mean, target_std)

    model = SequenceResidualModel(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=args.max_probes,
        head_bias_init=target_mean,
    ).to(device)
    LOG.info("model parameters: %d", model.num_parameters)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(train_loader):
            feats = batch["features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pad_mask = batch["padding_mask"].to(device, non_blocking=True)
            real_mask = ~pad_mask

            optimizer.zero_grad()
            pred = model(feats, pad_mask)
            loss = _compute_loss(pred, target, real_mask, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            if (i + 1) % args.log_interval == 0:
                LOG.info("epoch %d batch %d/%d  loss=%.1f",
                         epoch, i + 1, len(train_loader), total_loss / n_batches)
        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_preds: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        val_masks: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                pad_mask = batch["padding_mask"].to(device, non_blocking=True)
                pred = model(feats, pad_mask)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(target.cpu().numpy())
                val_masks.append((~pad_mask).cpu().numpy())
        # Concatenate ragged batches into flat arrays of real-probe values.
        val_pred_flat = np.concatenate([p[m] for p, m in zip(val_preds, val_masks)])
        val_target_flat = np.concatenate([t[m] for t, m in zip(val_targets, val_masks)])
        val_mae = _compute_residual_mae(val_pred_flat, val_target_flat,
                                         np.ones_like(val_pred_flat, dtype=bool))
        val_rmse = float(np.sqrt(((val_pred_flat - val_target_flat) ** 2).mean()))
        baseline_mae = float(np.abs(val_target_flat - target_mean).mean())

        scheduler.step()
        elapsed = time.time() - t0
        ratio = val_mae / baseline_mae if baseline_mae > 0 else float("nan")
        LOG.info(
            "[epoch %d/%d done in %.1fs] train_loss=%.1f val_rmse=%.1f bp "
            "val_mae=%.1f bp baseline_mae=%.1f bp ratio=%.3f",
            epoch + 1, args.epochs, elapsed, train_loss, val_rmse, val_mae,
            baseline_mae, ratio,
        )

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "baseline_mae": baseline_mae,
            "ratio": ratio,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(rec)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ckpt = args.checkpoint_dir / f"sequence_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": rec,
                "args": vars(args),
                "target_mean": target_mean,
                "target_std": target_std,
            }, ckpt)
            LOG.info("saved checkpoint: %s", ckpt)

    history_path = args.checkpoint_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    LOG.info("training complete; history at %s", history_path)


if __name__ == "__main__":
    main()
