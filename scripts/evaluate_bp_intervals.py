"""Evaluate a trained checkpoint by inter-probe bp-interval accuracy.

This is the bp-domain eval (as opposed to the time-domain peak-match F1
produced by ``evaluate_peak_match.py``). It measures how well the
velocity-head integration (``cumulative_bp``) reconstructs inter-probe
distances in basepairs.

Method (anchored-to-wfmproc-centers):

    Probe locations are taken from the wfmproc warmstart centers
    (sample-index array paired 1:1 with ``reference_bp_positions``).
    The predicted inter-probe bp interval between probe i and probe i+1
    is then just::

        pred_interval = abs(cum_bp[centers[i+1]] - cum_bp[centers[i]])
        ref_interval  = abs(ref_bp[i+1] - ref_bp[i])

    This isolates the velocity-head quality from the peak-detector
    (which is measured separately in ``evaluate_peak_match.py``).

Usage:
    python scripts/evaluate_bp_intervals.py \\
        --checkpoint phase6_checkpoints/best_model.pt \\
        --cache-dir "E. coli/cache/STB03-063B-02L58270w05-433B23b" \\
        --output bp_intervals.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.model.unet import T2DUNet


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple caches.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Cap molecules processed per cache (exact count).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--min-probes",
        type=int,
        default=2,
        help="Skip molecules with fewer than this many warmstart probes (need at least 2 for an interval).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False required because our checkpoints include a TrainConfig
    # dataclass, not just raw tensors.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Per-interval records feed every aggregate; per-molecule summaries
    # feed per-run + overall.
    all_abs_err: list[float] = []
    all_rel_err: list[float] = []
    all_ref_intervals: list[float] = []
    all_pred_intervals: list[float] = []

    per_run: dict[str, dict[str, Any]] = {}
    per_molecule: list[dict[str, Any]] = []

    for cache_dir in args.cache_dir:
        run_id = _run_id_for_cache(cache_dir)
        dataset = CachedMoleculeDataset([cache_dir], augment=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_molecules,
        )

        run_abs_err: list[float] = []
        run_rel_err: list[float] = []
        run_ref_intervals: list[float] = []
        run_pred_intervals: list[float] = []
        run_skipped = 0
        run_n_molecules = 0

        for batch_idx, batch in enumerate(loader):
            waveform = batch["waveform"].to(device)
            conditioning = batch["conditioning"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    _probe, cum_bp, _vel, _logits = model(
                        waveform, conditioning, mask
                    )
                cum_bp = cum_bp.float()

            b = waveform.shape[0]
            for i in range(b):
                global_idx = batch_idx * args.batch_size + i
                if args.max_molecules is not None and global_idx >= args.max_molecules:
                    break

                centers_tensor = batch["warmstart_probe_centers_samples"][i]
                ref_bp_tensor = batch["reference_bp_positions"][i]
                if centers_tensor is None:
                    run_skipped += 1
                    continue

                centers = centers_tensor.detach().cpu().numpy().astype(np.int64)
                ref_bp = ref_bp_tensor.detach().cpu().numpy().astype(np.int64)

                if centers.size < args.min_probes or ref_bp.size < args.min_probes:
                    run_skipped += 1
                    continue

                # Pair must be same length. The GT builder guarantees this
                # (warmstart centers are built from the same unique_matched
                # list as reference_bp_positions), but double-check.
                if centers.size != ref_bp.size:
                    run_skipped += 1
                    continue

                # Clip centers to valid sample range for safety. They should
                # always be in-range by construction.
                T = cum_bp.shape[-1]
                if centers.max() >= T or centers.min() < 0:
                    run_skipped += 1
                    continue

                cum_bp_at_centers = (
                    cum_bp[i, centers].detach().cpu().numpy().astype(np.float64)
                )

                # Inter-probe intervals: N-1 per molecule.
                pred_intervals = np.abs(np.diff(cum_bp_at_centers))
                ref_intervals = np.abs(np.diff(ref_bp)).astype(np.float64)

                # Guard against zero-length reference intervals (would blow up
                # rel_err). Shouldn't happen post-dedup but be safe.
                mask_valid = ref_intervals > 0
                if not np.any(mask_valid):
                    run_skipped += 1
                    continue
                pred_intervals = pred_intervals[mask_valid]
                ref_intervals = ref_intervals[mask_valid]

                abs_err = np.abs(pred_intervals - ref_intervals)
                rel_err = abs_err / ref_intervals

                all_abs_err.extend(abs_err.tolist())
                all_rel_err.extend(rel_err.tolist())
                all_ref_intervals.extend(ref_intervals.tolist())
                all_pred_intervals.extend(pred_intervals.tolist())

                run_abs_err.extend(abs_err.tolist())
                run_rel_err.extend(rel_err.tolist())
                run_ref_intervals.extend(ref_intervals.tolist())
                run_pred_intervals.extend(pred_intervals.tolist())

                run_n_molecules += 1
                per_molecule.append(
                    {
                        "run_id": run_id,
                        "molecule_uid": int(batch["molecule_uid"][i]),
                        "n_intervals": int(abs_err.size),
                        "median_abs_err_bp": float(np.median(abs_err)),
                        "median_rel_err": float(np.median(rel_err)),
                        "p95_rel_err": float(np.percentile(rel_err, 95)),
                    }
                )

        if run_skipped > 0:
            print(
                f"  WARNING: skipped {run_skipped} molecules in run {run_id}"
                " (no warmstart GT, too few probes, or length mismatch)",
                file=sys.stderr,
            )

        per_run[run_id] = {
            "n_molecules": run_n_molecules,
            "n_intervals": len(run_abs_err),
            "n_skipped": run_skipped,
            **_aggregate(
                np.asarray(run_abs_err, dtype=np.float64),
                np.asarray(run_rel_err, dtype=np.float64),
            ),
            "correlation_ref_pred": (
                float(
                    np.corrcoef(
                        np.asarray(run_ref_intervals), np.asarray(run_pred_intervals)
                    )[0, 1]
                )
                if len(run_ref_intervals) >= 2
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
            float(
                np.corrcoef(
                    np.asarray(all_ref_intervals), np.asarray(all_pred_intervals)
                )[0, 1]
            )
            if len(all_ref_intervals) >= 2
            else 0.0
        ),
    }

    summary = {
        "method": "anchored-to-wfmproc-centers",
        "overall": overall,
        "per_run": per_run,
        "checkpoint": str(args.checkpoint),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"\n=== BP-Interval Accuracy (anchored to wfmproc centers) ===")
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
