"""V5 Phase 2: bp-interval rel-error eval for the sequence model.

Loads a trained SequenceResidualModel checkpoint and computes
bp-interval rel-err vs reference genome positions, mirroring
``scripts/eval_residual.py`` but with the sequence-model data path.
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
from torch.utils.data import DataLoader

from mongoose.data.residual_dataset import add_molecule_aggregates
from mongoose.data.sequence_dataset import (
    MoleculeSequenceDataset,
    collate_molecules,
)
from mongoose.etl.reads_maps_table import build_residual_table
from mongoose.model.sequence_residual import SequenceResidualModel


LOG = logging.getLogger("eval_sequence")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


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
    model: SequenceResidualModel,
    remap_dir: Path,
    run_id: str,
    device: torch.device,
    *,
    accepted_only: bool = True,
    max_probes: int = 80,
    batch_size: int = 32,
) -> dict:
    """Run sequence-model inference on one run and compute bp-iv rel-err.

    Strategy: build the residual DataFrame, then for each molecule in
    the test set:
      1. Run sequence model to get predicted_shift per probe
      2. pred_position = pre_position + predicted_shift
      3. ref_intervals from reference_position_bp diffs
      4. pred_intervals from pred_position diffs
      5. prod_intervals from post_position_bp diffs
      6. Compute |pred_iv - ref_iv| / ref_iv per interval

    Production's predicted intervals come from the same df (post -
    pre) so we get a direct apples-to-apples comparison.
    """
    df = build_residual_table(remap_dir, run_id)
    df = add_molecule_aggregates(df)

    # Build a sequence dataset on the FULL run (no train/val split here;
    # all probes are evaluation probes).
    ds = MoleculeSequenceDataset(
        df,
        accepted_only=accepted_only,
        require_reference=True,  # only matched probes -- need ref positions
        target_column="ref_anchored_residual_bp",
        compute_aggregates=False,
        min_probes=2,
        max_probes=max_probes,
    )
    if len(ds) == 0:
        return {
            "run_id": run_id,
            "n_molecules": 0,
            "n_intervals": 0,
            "model_interval_rel_err": _summarize(np.array([])),
            "prod_interval_rel_err": _summarize(np.array([])),
        }

    # We also need per-probe (pre_position_bp, post_position_bp,
    # reference_position_bp) to compute intervals; the sequence dataset
    # only stores features + target. Re-derive them from df by joining
    # on (run_id, uid, probe_idx). The MoleculeSequenceDataset sorts by
    # (run_id, uid, probe_idx), so molecule order matches.
    df_sorted = (
        df[
            df["accepted"]
            & df["has_reference"]
        ]
        .sort_values(["run_id", "uid", "probe_idx"])
        .reset_index(drop=True)
    )
    grouped = df_sorted.groupby(["run_id", "uid"], sort=False)
    aux_per_molecule: list[dict] = []
    for (rid, uid), grp in grouped:
        if len(grp) < 2 or len(grp) > max_probes:
            continue
        aux_per_molecule.append({
            "uid": int(uid),
            "pre": grp["pre_position_bp"].to_numpy(dtype=np.float64),
            "post": grp["post_position_bp"].to_numpy(dtype=np.float64),
            "ref": grp["reference_position_bp"].to_numpy(dtype=np.float64),
        })
    if len(aux_per_molecule) != len(ds):
        raise RuntimeError(
            f"aux/dataset mismatch: aux={len(aux_per_molecule)} ds={len(ds)}"
        )

    # Run inference. We accumulate predictions per molecule.
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules,
    )
    pred_per_molecule: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device, non_blocking=True)
            pad_mask = batch["padding_mask"].to(device, non_blocking=True)
            lengths = batch["lengths"].cpu().numpy()
            pred = model(feats, pad_mask).cpu().numpy()
            for i, k in enumerate(lengths):
                pred_per_molecule.append(pred[i, : int(k)])

    if len(pred_per_molecule) != len(aux_per_molecule):
        raise RuntimeError(
            f"prediction/aux count mismatch: pred={len(pred_per_molecule)} "
            f"aux={len(aux_per_molecule)}"
        )

    model_rel: list[float] = []
    prod_rel: list[float] = []
    for pred_shift, aux in zip(pred_per_molecule, aux_per_molecule):
        if len(pred_shift) != len(aux["pre"]):
            continue  # defensive; shouldn't happen
        pre = aux["pre"]
        post = aux["post"]
        ref = aux["ref"]
        pred_pos = pre + pred_shift.astype(np.float64)

        ref_iv = np.abs(np.diff(ref))
        prod_iv = np.abs(np.diff(post))
        pred_iv = np.abs(np.diff(pred_pos))

        keep = ref_iv > 1.0
        if not np.any(keep):
            continue
        model_rel.extend((np.abs(pred_iv[keep] - ref_iv[keep]) / ref_iv[keep]).tolist())
        prod_rel.extend((np.abs(prod_iv[keep] - ref_iv[keep]) / ref_iv[keep]).tolist())

    return {
        "run_id": run_id,
        "n_molecules": len(ds),
        "n_intervals": int(len(model_rel)),
        "model_interval_rel_err": _summarize(np.array(model_rel, dtype=np.float64)),
        "prod_interval_rel_err": _summarize(np.array(prod_rel, dtype=np.float64)),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--remap-dir", type=Path, action="append", required=True)
    p.add_argument("--run-id", type=str, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)

    if len(args.remap_dir) != len(args.run_id):
        raise SystemExit("--remap-dir count must match --run-id count")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    LOG.info("device=%s", device)

    LOG.info("loading checkpoint: %s", args.checkpoint)
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
        LOG.info("evaluating run %s", run_id)
        t0 = time.time()
        result = evaluate_run(
            model, remap_dir, run_id, device,
            max_probes=train_args.get("max_probes", 80),
        )
        per_run.append(result)
        m = result["model_interval_rel_err"]
        prod = result["prod_interval_rel_err"]
        LOG.info(
            "  %s: n_intervals=%d  model median=%.4f  prod median=%.4f  (%.1fs)",
            run_id, m["n"], m["median"], prod["median"], time.time() - t0,
        )

    medians = [r["model_interval_rel_err"]["median"]
               for r in per_run if r["model_interval_rel_err"]["n"] > 0]
    prod_medians = [r["prod_interval_rel_err"]["median"]
                    for r in per_run if r["prod_interval_rel_err"]["n"] > 0]
    LOG.info("=== AGGREGATE ===")
    LOG.info("median(per-run-medians): model=%.4f  prod=%.4f",
             float(np.median(medians)) if medians else float("nan"),
             float(np.median(prod_medians)) if prod_medians else float("nan"))

    out = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "T2D_baseline_overall_median": 0.162,
        "n_runs": len(per_run),
        "median_per_run_medians_model": float(np.median(medians)) if medians else float("nan"),
        "median_per_run_medians_prod": float(np.median(prod_medians)) if prod_medians else float("nan"),
        "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    LOG.info("wrote: %s", args.output)


if __name__ == "__main__":
    main()
