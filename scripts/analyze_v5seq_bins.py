"""Per-bin analysis for V5-Sequence vs production vs uncorrected prior.

Produces breakdowns matching the format of the algorithm-team
overnight report: per reference-interval-size bin and per pre-
interval-size bin (the bp-domain analog of the temporal-gap axis
since we don't have time-domain probe centers in this pipeline).

For each bin reports:
    n_intervals
    prior_mae_bp        ← uncorrected (pre_position) intervals vs ref
    production_mae_bp   ← _reads_maps.bin (post-correction) vs ref
    model_mae_bp        ← V5-Sequence (pre + predicted_shift) vs ref
    model_reduction_vs_prior_%
    model_reduction_vs_prod_%
    intervals_improved_vs_prior_%
    intervals_improved_vs_prod_%
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mongoose.data.residual_dataset import add_molecule_aggregates
from mongoose.data.sequence_dataset import (
    MoleculeSequenceDataset,
    collate_molecules,
)
from mongoose.etl.reads_maps_table import build_residual_table
from mongoose.model.sequence_residual import SequenceResidualModel


# Same bin edges as the algorithm-team report so results can be compared
# axis-for-axis.
REF_INTERVAL_BINS = [
    (0, 500), (500, 1000), (1000, 2500), (2500, 5000),
    (5000, 10000), (10000, 20000), (20000, 50000), (50000, float("inf")),
]
# Bp-domain analog of "temporal gap": the pre-correction interval size
# between adjacent probes.
PRE_INTERVAL_BINS = [
    (0, 500), (500, 1000), (1000, 2500), (2500, 5000),
    (5000, 10000), (10000, 20000), (20000, float("inf")),
]


def _bin_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f"({lo:g}, inf]"
    return f"({lo:g}, {hi:g}]"


def _bin_stats(
    bins: list[tuple[float, float]],
    bin_values: np.ndarray,        # value used for binning (e.g. ref_iv)
    prior_iv: np.ndarray,
    model_iv: np.ndarray,
    prod_iv: np.ndarray,
    ref_iv: np.ndarray,
) -> list[dict]:
    out: list[dict] = []
    prior_err = np.abs(prior_iv - ref_iv)
    model_err = np.abs(model_iv - ref_iv)
    prod_err = np.abs(prod_iv - ref_iv)
    for lo, hi in bins:
        mask = (bin_values > lo) & (bin_values <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        prior_mae = float(prior_err[mask].mean())
        model_mae = float(model_err[mask].mean())
        prod_mae = float(prod_err[mask].mean())
        out.append({
            "bin": _bin_label(lo, hi),
            "n_intervals": n,
            "prior_mae_bp": prior_mae,
            "production_mae_bp": prod_mae,
            "model_mae_bp": model_mae,
            "model_reduction_vs_prior_pct": (
                100.0 * (prior_mae - model_mae) / prior_mae if prior_mae > 0 else float("nan")
            ),
            "model_reduction_vs_production_pct": (
                100.0 * (prod_mae - model_mae) / prod_mae if prod_mae > 0 else float("nan")
            ),
            "intervals_improved_vs_prior_pct": float(
                100.0 * (model_err[mask] < prior_err[mask]).mean()
            ),
            "intervals_improved_vs_production_pct": float(
                100.0 * (model_err[mask] < prod_err[mask]).mean()
            ),
        })
    return out


def evaluate_run(
    model: SequenceResidualModel,
    remap_dir: Path,
    run_id: str,
    device: torch.device,
    *,
    max_probes: int = 80,
    batch_size: int = 32,
) -> dict:
    df = build_residual_table(remap_dir, run_id)
    df = add_molecule_aggregates(df)
    ds = MoleculeSequenceDataset(
        df, accepted_only=True, require_reference=True,
        target_column="ref_anchored_residual_bp",
        compute_aggregates=False, min_probes=2, max_probes=max_probes,
    )
    if len(ds) == 0:
        return {"run_id": run_id, "n_intervals": 0}

    df_sorted = (
        df[df["accepted"] & df["has_reference"]]
        .sort_values(["run_id", "uid", "probe_idx"])
        .reset_index(drop=True)
    )
    aux = []
    for (rid, uid), grp in df_sorted.groupby(["run_id", "uid"], sort=False):
        if len(grp) < 2 or len(grp) > max_probes:
            continue
        aux.append({
            "pre": grp["pre_position_bp"].to_numpy(dtype=np.float64),
            "post": grp["post_position_bp"].to_numpy(dtype=np.float64),
            "ref": grp["reference_position_bp"].to_numpy(dtype=np.float64),
        })
    assert len(aux) == len(ds)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_molecules)
    pred_per_mol = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device, non_blocking=True)
            pad_mask = batch["padding_mask"].to(device, non_blocking=True)
            lengths = batch["lengths"].cpu().numpy()
            pred = model(feats, pad_mask).cpu().numpy()
            for i, k in enumerate(lengths):
                pred_per_mol.append(pred[i, : int(k)])

    # Concatenate per-interval arrays across all molecules in this run.
    prior_iv: list[np.ndarray] = []
    model_iv: list[np.ndarray] = []
    prod_iv: list[np.ndarray] = []
    ref_iv: list[np.ndarray] = []
    pre_iv_for_binning: list[np.ndarray] = []
    for shift, a in zip(pred_per_mol, aux):
        pre = a["pre"]
        post = a["post"]
        ref = a["ref"]
        pred_pos = pre + shift.astype(np.float64)
        prior_iv.append(np.abs(np.diff(pre)))
        model_iv.append(np.abs(np.diff(pred_pos)))
        prod_iv.append(np.abs(np.diff(post)))
        ref_iv.append(np.abs(np.diff(ref)))
        pre_iv_for_binning.append(np.abs(np.diff(pre)))

    prior_iv_a = np.concatenate(prior_iv)
    model_iv_a = np.concatenate(model_iv)
    prod_iv_a = np.concatenate(prod_iv)
    ref_iv_a = np.concatenate(ref_iv)
    pre_iv_a = np.concatenate(pre_iv_for_binning)
    keep = ref_iv_a > 1.0

    return {
        "run_id": run_id,
        "n_intervals": int(keep.sum()),
        "n_molecules": len(aux),
        "overall": {
            "prior_mae_bp": float(np.abs(prior_iv_a[keep] - ref_iv_a[keep]).mean()),
            "production_mae_bp": float(np.abs(prod_iv_a[keep] - ref_iv_a[keep]).mean()),
            "model_mae_bp": float(np.abs(model_iv_a[keep] - ref_iv_a[keep]).mean()),
            "model_median_rel_err": float(
                np.median(np.abs(model_iv_a[keep] - ref_iv_a[keep]) / ref_iv_a[keep])
            ),
            "production_median_rel_err": float(
                np.median(np.abs(prod_iv_a[keep] - ref_iv_a[keep]) / ref_iv_a[keep])
            ),
            "prior_median_rel_err": float(
                np.median(np.abs(prior_iv_a[keep] - ref_iv_a[keep]) / ref_iv_a[keep])
            ),
        },
        "by_ref_interval": _bin_stats(
            REF_INTERVAL_BINS, ref_iv_a[keep],
            prior_iv_a[keep], model_iv_a[keep], prod_iv_a[keep], ref_iv_a[keep],
        ),
        "by_pre_interval": _bin_stats(
            PRE_INTERVAL_BINS, pre_iv_a[keep],
            prior_iv_a[keep], model_iv_a[keep], prod_iv_a[keep], ref_iv_a[keep],
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--remap-dir", type=Path, action="append", required=True)
    p.add_argument("--run-id", type=str, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    model = SequenceResidualModel(
        hidden_dim=train_args.get("hidden_dim", 128),
        n_layers=train_args.get("n_layers", 4),
        n_heads=train_args.get("n_heads", 4),
        dropout=train_args.get("dropout", 0.1),
        max_seq_len=train_args.get("max_probes", 80),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    per_run: list[dict] = []
    for remap_dir, run_id in zip(args.remap_dir, args.run_id, strict=True):
        print(f"evaluating {run_id}")
        t0 = time.time()
        per_run.append(evaluate_run(model, remap_dir, run_id, device,
                                     max_probes=train_args.get("max_probes", 80)))
        print(f"  done in {time.time() - t0:.1f}s")

    # Aggregate across runs by re-binning the concatenated per-interval data.
    out = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "n_runs": len(per_run),
        "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
