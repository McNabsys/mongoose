"""Evaluate a trained checkpoint by peak-match F1 in REFERENCE BP SPACE.

Companion to ``evaluate_peak_match.py`` (which matches in sample space
against wfmproc centers). This script instead:

  1. Runs the model forward to get ``probe_heatmap`` and ``cumulative_bp``.
  2. Extracts predicted peaks via the same NMS used at training time.
  3. Reads ``cumulative_bp`` at each predicted peak sample.
  4. Zero-anchors predicted bp deltas at the first predicted peak, matching
     the training L_bp convention in ``combined.py:246-247``.
  5. Zero-anchors reference bp deltas at ``reference_bp_positions[0]`` with
     ``abs()`` to handle reverse molecules.
  6. Matches via Hungarian assignment with tolerance in BASEPAIRS.

Also computes the sample-space wfmproc F1 alongside so both numbers land
in the same JSON for side-by-side comparison.

Usage:
    python scripts/evaluate_reference_bp.py \\
        --checkpoint micro_phase6_checkpoints/best_model.pt \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --output micro_phase6_ref_bp_eval.json \\
        --tolerance-bp 500 --tolerance-samples 50
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
from mongoose.inference.bp_match import evaluate_molecule_bp
from mongoose.inference.peak_match import (
    aggregate_per_molecule_metrics,
    compute_metrics,
    match_peaks,
)
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def _run_id_for_cache(cache_dir: Path) -> str:
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    return str(manifest["run_id"])


def _aggregate_bp_mae(per_molecule: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate bp_mae_matched across molecules where it is defined."""
    maes = [m["bp_mae_matched"] for m in per_molecule if not np.isnan(m["bp_mae_matched"])]
    if not maes:
        return {"bp_mae_mean": float("nan"), "bp_mae_median": float("nan"),
                "bp_mae_p95": float("nan"), "n_with_matches": 0}
    arr = np.array(maes, dtype=np.float64)
    return {
        "bp_mae_mean": float(np.mean(arr)),
        "bp_mae_median": float(np.median(arr)),
        "bp_mae_p95": float(np.percentile(arr, 95)),
        "n_with_matches": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, action="append", required=True,
        help="Repeat for multiple caches.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-molecules", type=int, default=None,
        help="Cap molecules processed per cache (exact count).",
    )
    parser.add_argument(
        "--tolerance-bp", type=float, default=500.0,
        help="Matching tolerance in basepairs for the reference-bp metric.",
    )
    parser.add_argument(
        "--tolerance-samples", type=float, default=50.0,
        help="Matching tolerance in samples for the parallel wfmproc metric.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Peak-extraction confidence threshold.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: checkpoints carry a TrainConfig dataclass, not just tensors.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bp_per_molecule: list[dict[str, Any]] = []
    wfm_per_molecule: list[dict[str, Any]] = []
    bp_overall = {"tp": 0, "fp": 0, "fn": 0}
    wfm_overall = {"tp": 0, "fp": 0, "fn": 0}
    per_run: dict[str, dict[str, Any]] = {}

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

        run_bp_per_mol: list[dict[str, Any]] = []
        run_wfm_per_mol: list[dict[str, Any]] = []
        run_bp = {"tp": 0, "fp": 0, "fn": 0}
        run_wfm = {"tp": 0, "fp": 0, "fn": 0}
        run_skipped_bp = 0
        run_skipped_wfm = 0

        for batch_idx, batch in enumerate(loader):
            waveform = batch["waveform"].to(device)
            conditioning = batch["conditioning"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    probe_heatmap, cumulative_bp, raw_velocity, _logits = model(
                        waveform, conditioning, mask
                    )
                probe_heatmap = probe_heatmap.float()
                cumulative_bp = cumulative_bp.float()
                raw_velocity = raw_velocity.float()

            b = waveform.shape[0]
            for i in range(b):
                global_idx = batch_idx * args.batch_size + i
                if args.max_molecules is not None and global_idx >= args.max_molecules:
                    break

                h = probe_heatmap[i]
                v = raw_velocity[i]
                m = mask[i]
                cum = cumulative_bp[i]
                h_masked = h * m.to(h.dtype)
                v_masked = torch.where(m, v, torch.zeros_like(v))
                pred_idx = extract_peak_indices(
                    h_masked, v_masked, threshold=args.threshold
                )
                pred_np = pred_idx.detach().cpu().numpy().astype(np.int64)
                cum_np = cum.detach().cpu().numpy().astype(np.float64)

                molecule_uid = int(batch["molecule_uid"][i])

                # Reference-bp matching (primary metric).
                ref_bp_tensor = batch["reference_bp_positions"][i]
                if ref_bp_tensor is None or ref_bp_tensor.numel() == 0:
                    run_skipped_bp += 1
                else:
                    ref_bp_np = ref_bp_tensor.detach().cpu().numpy().astype(np.int64)
                    bp_res = evaluate_molecule_bp(
                        pred_peak_samples=pred_np,
                        pred_cumulative_bp=cum_np,
                        reference_bp_positions=ref_bp_np,
                        tolerance_bp=args.tolerance_bp,
                    )
                    bp_res["run_id"] = run_id
                    bp_res["molecule_uid"] = molecule_uid
                    run_bp_per_mol.append(bp_res)
                    bp_per_molecule.append(bp_res)
                    run_bp["tp"] += bp_res["tp"]
                    run_bp["fp"] += bp_res["fp"]
                    run_bp["fn"] += bp_res["fn"]

                # Parallel sample-space wfmproc matching (for side-by-side).
                centers_tensor = batch["warmstart_probe_centers_samples"][i]
                if centers_tensor is None:
                    run_skipped_wfm += 1
                else:
                    ref_np = centers_tensor.detach().cpu().numpy().astype(np.int64)
                    matches, fps, fns = match_peaks(
                        pred_np, ref_np, tolerance=args.tolerance_samples
                    )
                    tp, fp, fn = len(matches), len(fps), len(fns)
                    row = {
                        "run_id": run_id,
                        "molecule_uid": molecule_uid,
                        "n_pred": int(pred_np.size),
                        "n_ref": int(ref_np.size),
                        "tp": tp, "fp": fp, "fn": fn,
                        **compute_metrics(tp=tp, fp=fp, fn=fn),
                    }
                    run_wfm_per_mol.append(row)
                    wfm_per_molecule.append(row)
                    run_wfm["tp"] += tp
                    run_wfm["fp"] += fp
                    run_wfm["fn"] += fn

        if run_skipped_bp > 0:
            print(
                f"  WARNING: skipped {run_skipped_bp} molecules in {run_id}"
                " (no reference_bp_positions)",
                file=sys.stderr,
            )
        if run_skipped_wfm > 0:
            print(
                f"  WARNING: skipped {run_skipped_wfm} molecules in {run_id}"
                " (no wfmproc ground truth)",
                file=sys.stderr,
            )

        for k in ("tp", "fp", "fn"):
            bp_overall[k] += run_bp[k]
            wfm_overall[k] += run_wfm[k]

        per_run[run_id] = {
            "reference_bp": {
                **run_bp,
                **compute_metrics(tp=run_bp["tp"], fp=run_bp["fp"], fn=run_bp["fn"]),
                "per_molecule_mean": aggregate_per_molecule_metrics(run_bp_per_mol),
                "bp_mae": _aggregate_bp_mae(run_bp_per_mol),
                "n_skipped": run_skipped_bp,
            },
            "wfmproc_samples": {
                **run_wfm,
                **compute_metrics(tp=run_wfm["tp"], fp=run_wfm["fp"], fn=run_wfm["fn"]),
                "per_molecule_mean": aggregate_per_molecule_metrics(run_wfm_per_mol),
                "n_skipped": run_skipped_wfm,
            },
        }

    summary = {
        "overall": {
            "reference_bp": {
                **bp_overall,
                **compute_metrics(tp=bp_overall["tp"], fp=bp_overall["fp"], fn=bp_overall["fn"]),
                "per_molecule_mean": aggregate_per_molecule_metrics(bp_per_molecule),
                "bp_mae": _aggregate_bp_mae(bp_per_molecule),
            },
            "wfmproc_samples": {
                **wfm_overall,
                **compute_metrics(tp=wfm_overall["tp"], fp=wfm_overall["fp"], fn=wfm_overall["fn"]),
                "per_molecule_mean": aggregate_per_molecule_metrics(wfm_per_molecule),
            },
        },
        "per_run": per_run,
        "tolerance_bp": args.tolerance_bp,
        "tolerance_samples": args.tolerance_samples,
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=float))

    # Stdout summary: the comparison is the point of this script.
    o_bp = summary["overall"]["reference_bp"]
    o_wfm = summary["overall"]["wfmproc_samples"]
    mae = o_bp["bp_mae"]
    print(f"\n=== Reference-bp F1 @ tolerance={args.tolerance_bp} bp ===")
    print(
        f"  Sum-of-counts:     P={o_bp['precision']:.3f}  "
        f"R={o_bp['recall']:.3f}  F1={o_bp['f1']:.3f}"
    )
    m = o_bp["per_molecule_mean"]
    print(
        f"  Per-molecule mean: P={m['precision']:.3f}  "
        f"R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['n_molecules']})"
    )
    print(
        f"  bp MAE at matched peaks: "
        f"mean={mae['bp_mae_mean']:.1f}  median={mae['bp_mae_median']:.1f}  "
        f"p95={mae['bp_mae_p95']:.1f}  (n_with_matches={mae['n_with_matches']})"
    )
    print(f"\n=== Wfmproc sample-space F1 @ tolerance={args.tolerance_samples} samples (comparison) ===")
    print(
        f"  Sum-of-counts:     P={o_wfm['precision']:.3f}  "
        f"R={o_wfm['recall']:.3f}  F1={o_wfm['f1']:.3f}"
    )
    m = o_wfm["per_molecule_mean"]
    print(
        f"  Per-molecule mean: P={m['precision']:.3f}  "
        f"R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['n_molecules']})"
    )

    print(f"\n  Wrote: {args.output}")


if __name__ == "__main__":
    main()
