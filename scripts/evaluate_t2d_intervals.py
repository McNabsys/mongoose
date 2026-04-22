"""Evaluate legacy T2D (per-channel power-law) bp-interval accuracy on a cache.

This is the production-pipeline baseline. The Oliver 2023 derivation
(``support/T2D.pdf``) gives the physics:

    L_TE = C' * t^(2/3)

— pure power-law in time, working backwards from the molecule's trailing
edge. The empirical Nabsys implementation generalizes this slightly:

    L(t) = mult_const * t_from_tail_ms ^ alpha + addit_const

where:
  * ``t_from_tail_ms`` is the time elapsed before the trailing edge
    (MILLISECONDS, not samples)
  * ``alpha`` is empirically ~0.55 (close to but slightly below the
    physics-pure 2/3)
  * ``addit_const`` is a bp-space offset to the result (calibration
    correction, NOT a time-space offset to the input)

The per-channel constants come from a ``_transForm.txt`` file produced by
Nabsys tooling alongside remapping.

Output format matches ``evaluate_bp_intervals.py`` so T2D and V3 numbers
can be compared apples-to-apples on the same holdout caches.

NOTE: ``src/mongoose/inference/legacy_t2d.py`` has the formula wrong (uses
samples for t, treats addit_const as a time offset inside the parens). It
was never validated end-to-end. This script is the correct implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mongoose.inference.legacy_t2d import legacy_t2d_bp_positions
from mongoose.io.transform import load_transforms


def _run_id_for_cache(cache_dir: Path) -> str:
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    return str(manifest["run_id"])


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _aggregate(abs_errors: np.ndarray, rel_errors: np.ndarray) -> dict[str, Any]:
    return {
        "abs_err_bp": _percentiles(abs_errors),
        "rel_err": _percentiles(rel_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        action="append",
        required=True,
        type=Path,
        help="Repeatable; cache directory whose wfmproc centers to score.",
    )
    parser.add_argument(
        "--transform-file",
        action="append",
        required=True,
        type=Path,
        help=(
            "Path to the _transForm.txt for each cache. Repeat once per "
            "--cache-dir, in the same order. Each file supplies per-channel "
            "(mult_const, addit_const, alpha)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-probes",
        type=int,
        default=2,
        help="Skip molecules with fewer than this many warmstart probes.",
    )
    args = parser.parse_args()

    if len(args.cache_dir) != len(args.transform_file):
        raise SystemExit(
            f"error: {len(args.cache_dir)} --cache-dir but "
            f"{len(args.transform_file)} --transform-file; must match 1:1"
        )

    import pickle
    from mongoose.io.probes_bin import load_probes_bin

    all_abs_err: list[float] = []
    all_rel_err: list[float] = []
    all_ref: list[float] = []
    all_pred: list[float] = []
    per_run: dict[str, dict[str, Any]] = {}

    SAMPLE_PERIOD_MS = 1000.0 / 32000  # 32 kHz TDB sample rate.

    for cache_dir, tf_path in zip(args.cache_dir, args.transform_file):
        run_id = _run_id_for_cache(cache_dir)
        transforms = load_transforms(tf_path)

        with open(cache_dir / "manifest.json") as f:
            manifest = json.load(f)
        with open(cache_dir / "molecules.pkl", "rb") as f:
            gt_list: list[dict] = pickle.load(f)

        # Load the original probes.bin for fall_t50 and start_within_tdb_ms,
        # which the cache doesn't preserve. The probes.bin lives alongside
        # the _transForm.txt file (same Remapped/AllCh directory).
        pbin_path = next(tf_path.parent.glob("*_probes.bin"), None)
        if pbin_path is None:
            raise SystemExit(
                f"error: no *_probes.bin found in {tf_path.parent} (needed for fall_t50)"
            )
        pf = load_probes_bin(pbin_path)
        mols_by_uid = {int(m.uid): m for m in pf.molecules}

        run_abs: list[float] = []
        run_rel: list[float] = []
        run_ref: list[float] = []
        run_pred: list[float] = []
        run_skipped = 0
        run_n_molecules = 0

        for mol_info, gt in zip(manifest["molecules"], gt_list):
            centers = gt.get("warmstart_probe_centers_samples")
            ref_bp = gt.get("reference_bp_positions")
            if centers is None or ref_bp is None:
                run_skipped += 1
                continue

            centers_np = np.asarray(centers, dtype=np.int64)
            ref_np = np.asarray(ref_bp, dtype=np.int64)

            if centers_np.size < args.min_probes or ref_np.size < args.min_probes:
                run_skipped += 1
                continue
            if centers_np.size != ref_np.size:
                run_skipped += 1
                continue

            channel_key = f"Ch{int(mol_info['channel']):03d}"
            transform = transforms.get(channel_key)
            if transform is None:
                run_skipped += 1
                continue

            mol = mols_by_uid.get(int(mol_info["uid"]))
            if mol is None:
                run_skipped += 1
                continue

            bp_positions = legacy_t2d_bp_positions(
                centers_np,
                mol=mol,
                mult_const=transform.mult_const,
                addit_const=transform.addit_const,
                alpha=transform.alpha,
            )

            pred_intervals = np.abs(np.diff(bp_positions))
            ref_intervals = np.abs(np.diff(ref_np)).astype(np.float64)

            mask = ref_intervals > 0
            if not np.any(mask):
                run_skipped += 1
                continue
            pred_intervals = pred_intervals[mask]
            ref_intervals = ref_intervals[mask]

            abs_err = np.abs(pred_intervals - ref_intervals)
            rel_err = abs_err / ref_intervals

            all_abs_err.extend(abs_err.tolist())
            all_rel_err.extend(rel_err.tolist())
            all_ref.extend(ref_intervals.tolist())
            all_pred.extend(pred_intervals.tolist())

            run_abs.extend(abs_err.tolist())
            run_rel.extend(rel_err.tolist())
            run_ref.extend(ref_intervals.tolist())
            run_pred.extend(pred_intervals.tolist())

            run_n_molecules += 1

        if run_skipped > 0:
            print(
                f"  WARNING: skipped {run_skipped} molecules in run {run_id}",
                file=sys.stderr,
            )

        per_run[run_id] = {
            "n_molecules": run_n_molecules,
            "n_intervals": len(run_abs),
            "n_skipped": run_skipped,
            **_aggregate(
                np.asarray(run_abs, dtype=np.float64),
                np.asarray(run_rel, dtype=np.float64),
            ),
            "correlation_ref_pred": (
                float(np.corrcoef(np.asarray(run_ref), np.asarray(run_pred))[0, 1])
                if len(run_ref) >= 2
                else 0.0
            ),
        }

    overall_abs = np.asarray(all_abs_err, dtype=np.float64)
    overall_rel = np.asarray(all_rel_err, dtype=np.float64)
    overall = {
        "n_molecules": sum(r["n_molecules"] for r in per_run.values()),
        "n_intervals": int(overall_abs.size),
        "n_skipped": sum(r["n_skipped"] for r in per_run.values()),
        **_aggregate(overall_abs, overall_rel),
        "correlation_ref_pred": (
            float(np.corrcoef(np.asarray(all_ref), np.asarray(all_pred))[0, 1])
            if len(all_ref) >= 2
            else 0.0
        ),
    }

    summary = {
        "method": "legacy-T2D-power-law",
        "overall": overall,
        "per_run": per_run,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Legacy T2D BP-Interval Accuracy ===")
    o = overall
    print(
        f"  Overall: n_intervals={o['n_intervals']}"
        f"  median_rel={o['rel_err']['median']:.4f}"
        f"  p95_rel={o['rel_err']['p95']:.4f}"
        f"  p99_rel={o['rel_err']['p99']:.4f}"
    )
    print(
        f"  Overall abs: median={o['abs_err_bp']['median']:.1f} bp"
        f"  p95={o['abs_err_bp']['p95']:.1f} bp"
        f"  p99={o['abs_err_bp']['p99']:.1f} bp"
        f"  corr(ref,pred)={o['correlation_ref_pred']:.4f}"
    )
    for run_id, stats in per_run.items():
        print(
            f"  Run {run_id}: "
            f"n_int={stats['n_intervals']}  "
            f"median_rel={stats['rel_err']['median']:.4f}  "
            f"p95_rel={stats['rel_err']['p95']:.4f}  "
            f"corr={stats['correlation_ref_pred']:.4f}"
        )
    print(f"\n  Wrote: {args.output}")


if __name__ == "__main__":
    main()
