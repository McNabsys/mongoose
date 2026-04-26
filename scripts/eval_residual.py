"""Evaluation harness for Direction C: bp-interval rel-error vs reference.

Loads a trained ResidualMLP checkpoint and computes:

1. **bp-interval rel-error** -- the metric V3/T2D's 16.2% baseline comes
   from. For each molecule's accepted probes with a reference match:

       pred_position = pre_position_bp + model(features)
       pred_intervals  = abs(diff(pred_position))
       prod_intervals  = abs(diff(post_position_bp))   # production's prediction
       ref_intervals   = abs(diff(reference_position_bp))  # true genomic
       model_rel_err = |pred_iv - ref_iv| / ref_iv
       prod_rel_err  = |prod_iv - ref_iv| / ref_iv

   Reports median + p95 per holdout cache + overall.

2. **Per-decile residual-prediction MAE** stratified by |residual_bp|.
   Diagnoses whether the model handles small shifts well but fails on
   the +5kb-17kb tail (the head-dive territory).

3. **Per-run breakdown** -- median bp-interval rel-error per cache.
   Diagnoses whether the model overfits a few runs and underperforms
   on others (which would point at missing per-detector / per-condition
   features).

Outputs a structured JSON with all metrics for downstream writeup.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mongoose.data.residual_dataset import (
    FEATURE_DIM,
    ResidualDataset,
    add_molecule_aggregates,
    extract_features,
)
from mongoose.etl.reads_maps_table import build_residual_table
from mongoose.model.residual_mlp import ResidualMLP


LOG = logging.getLogger("eval_residual")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _per_decile_mae(
    pred_residual: np.ndarray, target_residual: np.ndarray, n_deciles: int = 10
) -> dict:
    """MAE of model residual prediction stratified by |target_residual| decile."""
    if pred_residual.size == 0:
        return {}
    abs_target = np.abs(target_residual)
    quantile_edges = np.quantile(abs_target, np.linspace(0, 1, n_deciles + 1))
    out: dict = {}
    for d in range(n_deciles):
        lo = quantile_edges[d]
        hi = quantile_edges[d + 1]
        mask = (abs_target >= lo) & (
            (abs_target < hi) if d < n_deciles - 1 else (abs_target <= hi)
        )
        if mask.sum() == 0:
            continue
        out[d] = {
            "decile_lo_bp": float(lo),
            "decile_hi_bp": float(hi),
            "n_probes": int(mask.sum()),
            "mae_bp": float(np.abs(pred_residual[mask] - target_residual[mask]).mean()),
            "rmse_bp": float(np.sqrt(((pred_residual[mask] - target_residual[mask]) ** 2).mean())),
            "mean_target_bp": float(target_residual[mask].mean()),
            "mean_abs_target_bp": float(abs_target[mask].mean()),
        }
    return out


def _interval_rel_errors(
    df: pd.DataFrame, pred_residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each molecule's matched probes, compute model + production
    bp-interval rel-error against reference.

    Returns:
        (model_rel_err, prod_rel_err) -- two flat arrays of per-interval
        relative errors. ``len(arrays) = sum over molecules of (K - 1)``
        for K matched probes.
    """
    df_local = df.copy()
    df_local["pred_residual"] = pred_residual
    df_local["pred_position_bp"] = (
        df_local["pre_position_bp"].astype(np.int64) + pred_residual.astype(np.int64)
    )

    model_rel: list[float] = []
    prod_rel: list[float] = []
    for _, group in df_local.groupby("uid", sort=False):
        # Only use accepted probes with a reference match. Sort by
        # probe_idx (temporal order in molecule).
        sub = group[group["accepted"] & group["has_reference"]].sort_values("probe_idx")
        if len(sub) < 2:
            continue
        pre = sub["pre_position_bp"].to_numpy(dtype=np.float64)
        post = sub["post_position_bp"].to_numpy(dtype=np.float64)
        pred = sub["pred_position_bp"].to_numpy(dtype=np.float64)
        ref = sub["reference_position_bp"].to_numpy(dtype=np.float64)

        ref_iv = np.abs(np.diff(ref))
        prod_iv = np.abs(np.diff(post))
        pred_iv = np.abs(np.diff(pred))

        # Drop intervals where ref_iv is zero (degenerate: same ref position twice).
        m = ref_iv > 1.0
        if not np.any(m):
            continue
        model_rel.extend((np.abs(pred_iv[m] - ref_iv[m]) / ref_iv[m]).tolist())
        prod_rel.extend((np.abs(prod_iv[m] - ref_iv[m]) / ref_iv[m]).tolist())

    return np.array(model_rel, dtype=np.float64), np.array(prod_rel, dtype=np.float64)


def _summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
    }


def evaluate_run(
    model: ResidualMLP,
    remap_dir: Path,
    run_id: str,
    device: torch.device,
    *,
    accepted_only: bool = True,
) -> dict:
    """Evaluate the model on one run's residual table."""
    df = build_residual_table(remap_dir, run_id)
    df = add_molecule_aggregates(df)
    if accepted_only:
        df = df[df["accepted"]].reset_index(drop=True)

    feats = extract_features(df)
    target = df["residual_bp"].to_numpy(dtype=np.float32)

    # Inference in batches to avoid GPU OOM.
    model.eval()
    preds: list[np.ndarray] = []
    batch = 16384
    feats_t = torch.from_numpy(feats)
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch):
            chunk = feats_t[i : i + batch].to(device, non_blocking=True)
            preds.append(model(chunk).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)

    # 1) bp-interval rel-error
    model_rel, prod_rel = _interval_rel_errors(df, pred)

    # 2) per-decile residual-prediction MAE
    per_decile = _per_decile_mae(pred, target)

    # Overall residual-prediction MAE
    overall_mae = float(np.abs(pred - target).mean()) if pred.size else float("nan")

    return {
        "run_id": run_id,
        "n_probes": int(len(df)),
        "n_intervals": int(model_rel.size),
        "residual_mae_bp": overall_mae,
        "model_interval_rel_err": _summarize(model_rel),
        "prod_interval_rel_err": _summarize(prod_rel),
        "per_decile_mae": per_decile,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--remap-dir", type=Path, action="append", required=True)
    parser.add_argument("--run-id", type=str, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    LOG.info("device=%s", device)

    if len(args.remap_dir) != len(args.run_id):
        raise SystemExit("--remap-dir count must match --run-id count")

    # Load checkpoint
    LOG.info("loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    LOG.info("checkpoint metadata: epoch=%s, target_mean=%s",
             ckpt.get("epoch"), ckpt.get("target_mean"))

    model = ResidualMLP(
        hidden_dim=train_args.get("hidden_dim", 256),
        n_blocks=train_args.get("n_blocks", 4),
        dropout=train_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    per_run: list[dict] = []
    for remap_dir, run_id in zip(args.remap_dir, args.run_id, strict=True):
        LOG.info("evaluating run %s", run_id)
        t0 = time.time()
        try:
            result = evaluate_run(model, remap_dir, run_id, device)
        except Exception as exc:
            LOG.exception("failed on %s: %s", run_id, exc)
            continue
        per_run.append(result)
        m = result["model_interval_rel_err"]
        p = result["prod_interval_rel_err"]
        LOG.info(
            "  %s: n=%d intervals  model median=%.4f p95=%.4f  prod median=%.4f p95=%.4f  "
            "(eval %.1fs)",
            run_id, m["n"], m["median"], m["p95"], p["median"], p["p95"],
            time.time() - t0,
        )

    # Aggregate across runs
    all_model_rel = np.concatenate(
        [
            np.array([])  # placeholder, real arrays below
        ]
    )
    # Walk per_run again to pull aggregates (simpler than threading arrays through).
    LOG.info("=== AGGREGATE ===")
    overall_median_models = [r["model_interval_rel_err"]["median"] for r in per_run if r["model_interval_rel_err"]["n"] > 0]
    overall_median_prods = [r["prod_interval_rel_err"]["median"] for r in per_run if r["prod_interval_rel_err"]["n"] > 0]
    overall_n = sum(r["model_interval_rel_err"]["n"] for r in per_run)
    LOG.info("aggregated %d runs, %d intervals total", len(per_run), overall_n)
    LOG.info("median(per-run-medians): model=%.4f  prod=%.4f",
             float(np.median(overall_median_models)) if overall_median_models else float("nan"),
             float(np.median(overall_median_prods)) if overall_median_prods else float("nan"))

    out = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "T2D_baseline_overall_median": 0.162,
        "n_runs": len(per_run),
        "median_per_run_medians_model": (
            float(np.median(overall_median_models)) if overall_median_models else float("nan")
        ),
        "median_per_run_medians_prod": (
            float(np.median(overall_median_prods)) if overall_median_prods else float("nan")
        ),
        "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    LOG.info("wrote: %s", args.output)


if __name__ == "__main__":
    main()
