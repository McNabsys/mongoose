"""Evaluate a trained checkpoint by peak-match F1 against wfmproc probe centers.

Usage:
    python scripts/evaluate_peak_match.py \\
        --checkpoint overnight_training/best_model.pt \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --output peak_match.json

Output: JSON with overall + per-run + per-molecule metrics, plus a stdout
summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.inference.peak_match import (
    aggregate_per_molecule_metrics,
    compute_metrics,
    match_peaks,
)
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def _run_id_for_cache(cache_dir: Path) -> str:
    import json as _json
    with open(cache_dir / "manifest.json") as f:
        manifest = _json.load(f)
    return str(manifest["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True,
                        help="Repeat for multiple caches.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=None,
                        help="Cap per-cache molecule count (debug).")
    parser.add_argument("--tolerance", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Peak-extraction confidence threshold.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
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

        run_tp = run_fp = run_fn = 0
        run_per_mol: list[dict[str, Any]] = []

        for batch_idx, batch in enumerate(loader):
            if args.max_molecules is not None:
                if batch_idx * args.batch_size >= args.max_molecules:
                    break

            waveform = batch["waveform"].to(device)
            conditioning = batch["conditioning"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    probe_heatmap, _, raw_velocity = model(waveform, conditioning, mask)
                probe_heatmap = probe_heatmap.float()
                raw_velocity = raw_velocity.float()

            b = waveform.shape[0]
            for i in range(b):
                h = probe_heatmap[i]
                v = raw_velocity[i]
                m = mask[i]
                h_masked = h * m.to(h.dtype)
                v_masked = torch.where(m, v, torch.zeros_like(v))
                pred_idx = extract_peak_indices(
                    h_masked, v_masked, threshold=args.threshold
                )
                pred_np = pred_idx.detach().cpu().numpy().astype(np.int64)

                # Reference peaks: wfmproc centers stored in the cached
                # gt dict, accessed via the underlying dataset.
                global_idx = batch_idx * args.batch_size + i
                if global_idx >= len(dataset):
                    break
                dir_idx, mol_idx = dataset.entries[global_idx]
                gt = dataset.gt_lists[dir_idx][mol_idx]
                centers = gt.get("warmstart_probe_centers_samples")
                if centers is None:
                    # Skip molecules with no wfmproc ground truth.
                    continue
                ref_np = np.asarray(centers, dtype=np.int64)

                matches, fps, fns = match_peaks(
                    pred_np, ref_np, tolerance=args.tolerance
                )
                tp, fp, fn = len(matches), len(fps), len(fns)
                run_tp += tp; run_fp += fp; run_fn += fn
                metrics = compute_metrics(tp=tp, fp=fp, fn=fn)
                row = {
                    "run_id": run_id,
                    "molecule_uid": int(batch["molecule_uid"][i]),
                    "n_pred": int(pred_np.size),
                    "n_ref": int(ref_np.size),
                    "tp": tp, "fp": fp, "fn": fn,
                    **metrics,
                }
                run_per_mol.append(row)
                per_molecule.append(row)

        overall_tp += run_tp; overall_fp += run_fp; overall_fn += run_fn
        per_run[run_id] = {
            "tp": run_tp, "fp": run_fp, "fn": run_fn,
            **compute_metrics(tp=run_tp, fp=run_fp, fn=run_fn),
            "per_molecule_mean": aggregate_per_molecule_metrics(run_per_mol),
        }

    summary = {
        "overall": {
            "tp": overall_tp, "fp": overall_fp, "fn": overall_fn,
            **compute_metrics(tp=overall_tp, fp=overall_fp, fn=overall_fn),
            "per_molecule_mean": aggregate_per_molecule_metrics(per_molecule),
        },
        "per_run": per_run,
        "tolerance": args.tolerance,
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    # Print a short summary.
    o = summary["overall"]
    print(f"\n=== Peak-Match F1 @ tolerance={args.tolerance} samples ===")
    print(f"  Overall (sum-of-counts): P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}")
    m = o["per_molecule_mean"]
    print(f"  Per-molecule mean:       P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['n_molecules']})")
    for run_id, stats in per_run.items():
        print(f"  Run {run_id}: F1_sum={stats['f1']:.3f}  F1_mol={stats['per_molecule_mean']['f1']:.3f}  (n={stats['per_molecule_mean']['n_molecules']})")
    print(f"\n  Wrote: {args.output}")


if __name__ == "__main__":
    main()
