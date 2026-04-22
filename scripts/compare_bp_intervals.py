"""Unified bp-interval comparison harness for V3 decision-making.

Evaluates a list of model checkpoints AND the legacy T2D baseline on a
shared set of holdout caches, producing a single side-by-side comparison
table. This is the "one command → decision" artifact used to judge the
V3 spike and its extensions.

Usage:
    python scripts/compare_bp_intervals.py \\
        --checkpoint l511_spike_checkpoints/checkpoint_epoch_000.pt \\
        --checkpoint l511_spike_checkpoints/checkpoint_epoch_001.pt \\
        --checkpoint l511_spike_checkpoints/checkpoint_epoch_002.pt \\
        --cache-dir "E. coli/cache/STB03-063B..." \\
        --transform-file ".../STB03-063B..._transForm.txt" \\
        --cache-dir "E. coli/cache/STB03-064D..." \\
        --transform-file ".../STB03-064D..._transForm.txt" \\
        --cache-dir "E. coli/cache/STB03-065H..." \\
        --transform-file ".../STB03-065H..._transForm.txt" \\
        --output phase6_holdout_eval/comparison.json

The --cache-dir / --transform-file pairs must match 1:1 and in the same
order. Checkpoints are labeled by their filename stem (e.g. checkpoint_epoch_002).

Output: JSON summary + printed markdown table.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.inference.legacy_t2d import legacy_t2d_bp_positions
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.transform import load_transforms
from mongoose.model.unet import T2DUNet


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0, "median": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _per_cache_eval_v3(
    checkpoint_path: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run a V3 checkpoint against a single cache; return interval metrics."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = CachedMoleculeDataset([cache_dir], augment=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_molecules,
    )

    abs_err: list[float] = []
    rel_err: list[float] = []
    all_ref: list[float] = []
    all_pred: list[float] = []

    for batch in loader:
        waveform = batch["waveform"].to(device)
        conditioning = batch["conditioning"].to(device)
        mask = batch["mask"].to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _probe, cum_bp, _vel, _logits = model(waveform, conditioning, mask)
            cum_bp = cum_bp.float()

        b = waveform.shape[0]
        for i in range(b):
            centers = batch["warmstart_probe_centers_samples"][i]
            ref_bp = batch["reference_bp_positions"][i]
            if centers is None:
                continue
            c = centers.detach().cpu().numpy().astype(np.int64)
            r = ref_bp.detach().cpu().numpy().astype(np.int64)
            if c.size < 2 or r.size < 2 or c.size != r.size:
                continue
            T = cum_bp.shape[-1]
            if c.max() >= T or c.min() < 0:
                continue
            cum_at = cum_bp[i, c].detach().cpu().numpy().astype(np.float64)
            pred_iv = np.abs(np.diff(cum_at))
            ref_iv = np.abs(np.diff(r)).astype(np.float64)
            m = ref_iv > 0
            if not np.any(m):
                continue
            pred_iv = pred_iv[m]
            ref_iv = ref_iv[m]
            abs_e = np.abs(pred_iv - ref_iv)
            rel_e = abs_e / ref_iv
            abs_err.extend(abs_e.tolist())
            rel_err.extend(rel_e.tolist())
            all_ref.extend(ref_iv.tolist())
            all_pred.extend(pred_iv.tolist())

    abs_a = np.asarray(abs_err, dtype=np.float64)
    rel_a = np.asarray(rel_err, dtype=np.float64)
    corr = (
        float(np.corrcoef(np.asarray(all_ref), np.asarray(all_pred))[0, 1])
        if len(all_ref) >= 2
        else 0.0
    )
    return {
        "n_intervals": int(abs_a.size),
        "rel_err": _percentiles(rel_a),
        "abs_err_bp": _percentiles(abs_a),
        "corr": corr,
    }


def _per_cache_eval_t2d(
    cache_dir: Path, transform_file: Path
) -> dict[str, Any]:
    """Run legacy T2D on a single cache; return interval metrics."""
    transforms = load_transforms(transform_file)
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    with open(cache_dir / "molecules.pkl", "rb") as f:
        gt_list: list[dict] = pickle.load(f)
    pbin = next(transform_file.parent.glob("*_probes.bin"), None)
    if pbin is None:
        raise SystemExit(
            f"error: no *_probes.bin found in {transform_file.parent}"
        )
    pf = load_probes_bin(pbin)
    mols_by_uid = {int(m.uid): m for m in pf.molecules}

    abs_err: list[float] = []
    rel_err: list[float] = []
    all_ref: list[float] = []
    all_pred: list[float] = []

    for mol_info, gt in zip(manifest["molecules"], gt_list):
        centers = gt.get("warmstart_probe_centers_samples")
        ref_bp = gt.get("reference_bp_positions")
        if centers is None or ref_bp is None:
            continue
        c = np.asarray(centers, dtype=np.int64)
        r = np.asarray(ref_bp, dtype=np.int64)
        if c.size < 2 or c.size != r.size:
            continue
        mol = mols_by_uid.get(int(mol_info["uid"]))
        t = transforms.get(f"Ch{int(mol_info['channel']):03d}")
        if mol is None or t is None:
            continue
        bp = legacy_t2d_bp_positions(
            c, mol=mol, mult_const=t.mult_const, addit_const=t.addit_const, alpha=t.alpha
        )
        pred_iv = np.abs(np.diff(bp))
        ref_iv = np.abs(np.diff(r)).astype(np.float64)
        m = ref_iv > 0
        if not np.any(m):
            continue
        pred_iv = pred_iv[m]
        ref_iv = ref_iv[m]
        abs_e = np.abs(pred_iv - ref_iv)
        rel_e = abs_e / ref_iv
        abs_err.extend(abs_e.tolist())
        rel_err.extend(rel_e.tolist())
        all_ref.extend(ref_iv.tolist())
        all_pred.extend(pred_iv.tolist())

    abs_a = np.asarray(abs_err, dtype=np.float64)
    rel_a = np.asarray(rel_err, dtype=np.float64)
    corr = (
        float(np.corrcoef(np.asarray(all_ref), np.asarray(all_pred))[0, 1])
        if len(all_ref) >= 2
        else 0.0
    )
    return {
        "n_intervals": int(abs_a.size),
        "rel_err": _percentiles(rel_a),
        "abs_err_bp": _percentiles(abs_a),
        "corr": corr,
    }


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _print_table(results: dict[str, dict[str, dict[str, Any]]], cache_names: list[str]) -> None:
    # Header row
    print(
        "\n| Method          | "
        + " | ".join(f"{c[:10]:10s} med / p95" for c in cache_names)
        + " | Overall med / p95 / corr |"
    )
    print(
        "|-----------------|"
        + "|".join(["-" * 24 for _ in cache_names])
        + "|---------------------------|"
    )
    for method, per_cache in results.items():
        row = f"| {method:15s} |"
        all_rel: list[float] = []
        all_corrs: list[float] = []
        for c in cache_names:
            stats = per_cache.get(c, {})
            if not stats:
                row += "         -         |"
                continue
            med = stats["rel_err"]["median"]
            p95 = stats["rel_err"]["p95"]
            row += f" {_fmt_pct(med):>7s} / {_fmt_pct(p95):>7s} |"
            all_rel.append(med)
            all_corrs.append(stats["corr"])
        if all_rel:
            mean_med = np.mean(all_rel)
            mean_p95 = np.mean([per_cache[c]["rel_err"]["p95"] for c in cache_names if c in per_cache])
            mean_corr = np.mean(all_corrs)
            row += f" {_fmt_pct(mean_med):>7s} / {_fmt_pct(mean_p95):>7s} / {mean_corr:.3f} |"
        else:
            row += "              -              |"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", default=[], type=Path,
        help="Repeatable. Model checkpoints to evaluate.",
    )
    parser.add_argument(
        "--cache-dir", action="append", required=True, type=Path,
        help="Repeatable. Holdout cache directory.",
    )
    parser.add_argument(
        "--transform-file", action="append", required=True, type=Path,
        help="Repeatable. _transForm.txt, 1:1 with --cache-dir.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-t2d", action="store_true",
        help="Skip the legacy-T2D baseline (e.g., if already computed).",
    )
    args = parser.parse_args()

    if len(args.cache_dir) != len(args.transform_file):
        raise SystemExit("error: --cache-dir and --transform-file must match 1:1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cache name = final path component (run_id).
    cache_names = [c.name for c in args.cache_dir]

    results: dict[str, dict[str, dict[str, Any]]] = {}

    # T2D baseline first (no GPU cost).
    if not args.skip_t2d:
        print("Evaluating T2D baseline ...")
        t2d_per_cache: dict[str, dict[str, Any]] = {}
        for cdir, tfile in zip(args.cache_dir, args.transform_file):
            t2d_per_cache[cdir.name] = _per_cache_eval_t2d(cdir, tfile)
        results["T2D"] = t2d_per_cache

    # V3 checkpoints.
    for ckpt in args.checkpoint:
        label = ckpt.stem  # e.g., "checkpoint_epoch_002"
        print(f"Evaluating {label} ...")
        per_cache: dict[str, dict[str, Any]] = {}
        for cdir in args.cache_dir:
            per_cache[cdir.name] = _per_cache_eval_v3(
                ckpt, cdir, device, batch_size=args.batch_size
            )
        results[label] = per_cache

    # Save + print.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    _print_table(results, cache_names)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
