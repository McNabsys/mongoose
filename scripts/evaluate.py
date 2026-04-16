"""Evaluate trained T2D U-Net against legacy model on held-out data.

.. deprecated:: 2026-04-16

    This script is out of date and will not run as-is. The V1 rearchitecture
    (R4, 2026-04-15) removed fields that this script still references:

    * ``MoleculeGT.inter_probe_deltas_bp`` -- removed; now derived at training
      time from detected peaks (see ``src/mongoose/data/ground_truth.py``).
    * ``MoleculeGT.reference_probe_bp`` -- replaced by
      ``reference_bp_positions``.

    The conditioning vector built by ``_build_conditioning`` also no longer
    matches what the trainer writes. Training-time preprocess writes
    ``[mean_lvl1_from_tdb, log(duration_samples), log(inter_event_ms+1),
    time_in_run_frac, 0.0, 0.0]`` (see ``src/mongoose/data/preprocess.py``),
    but this script still produces the pre-rearchitecture probes.bin-derived
    vector.

    Finally, this script uses a dummy all-zero waveform at lines ~185-188
    rather than loading real TDB data, so even if the other mismatches were
    fixed the inference results would be meaningless.

    A rewrite should:

    1. Consume a preprocessed cache directory via
       ``mongoose.data.cached_dataset.CachedMoleculeDataset`` rather than
       scanning probes.bin and building a dummy waveform.
    2. Emit the same conditioning layout the trainer wrote (read it straight
       from the cache; don't rebuild it).
    3. Update the legacy-T2D comparison to use the new ``reference_bp_positions``
       plus an on-the-fly ``diff()`` when inter-probe deltas are needed.

    Until that rewrite lands, this script remains in-tree only as a design
    reference for what end-to-end evaluation should look like. Do not invoke.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mongoose.data.ground_truth import MoleculeGT, build_molecule_gt
from mongoose.inference.evaluate import (
    EvalMetrics,
    PeakCountStats,
    evaluate_intervals,
    evaluate_peak_counts,
)
from mongoose.inference.legacy_t2d import legacy_t2d_intervals
from mongoose.inference.pipeline import InferredMolecule, run_inference
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import Molecule, load_probes_bin
from mongoose.io.reference_map import load_reference_map
from mongoose.io.transform import load_transforms
from mongoose.model.unet import T2DUNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _format_metrics(name: str, metrics: EvalMetrics) -> str:
    """Format metrics as a table row."""
    return (
        f"  {name:<20s} "
        f"MAE={metrics.mae_bp:>8.1f} bp  "
        f"MedAE={metrics.median_ae_bp:>8.1f} bp  "
        f"StdAE={metrics.std_ae_bp:>8.1f} bp  "
        f"mols={metrics.num_molecules}  "
        f"intervals={metrics.num_intervals}"
    )


def _build_conditioning(mol: Molecule) -> torch.Tensor:
    """Build the 6-element conditioning vector for a molecule.

    .. deprecated:: 2026-04-16

        This layout is WRONG for the current trainer. Training-time
        preprocess (``src/mongoose/data/preprocess.py``) writes
        ``[mean_lvl1_from_tdb, log(duration_samples), log(inter_event_ms+1),
        time_in_run_frac, 0.0, 0.0]``. The vector below is the
        pre-rearchitecture probes.bin-derived shape and would cause the
        FiLM head to condition on totally different features than those the
        trainer saw -- results would be silently meaningless.

        The right replacement is to read the conditioning row straight out
        of the preprocessed cache (``conditioning.npy``), not rebuild it.
    """
    return torch.tensor(
        [
            [
                mol.transloc_time_ms,
                mol.rise_t50,
                mol.fall_t50,
                mol.mean_lvl1,
                float(mol.num_probes),
                mol.use_partial_time_ms,
            ]
        ],
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained T2D U-Net against legacy model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--probes-bin", type=str, required=True, help="Path to probes.bin file"
    )
    parser.add_argument(
        "--assigns", type=str, required=True, help="Path to .assigns file"
    )
    parser.add_argument(
        "--reference-map", type=str, required=True, help="Path to _referenceMap.txt"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        help="Path to _transForm.txt for legacy baseline comparison",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Probe detection confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu or cuda)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write structured JSON evaluation results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier to embed in JSON metadata",
    )
    args = parser.parse_args()

    # --- Load data ---
    logger.info("Loading data...")
    probes_file = load_probes_bin(args.probes_bin, max_molecules=args.max_molecules)
    assignments = load_assigns(args.assigns)
    ref_map = load_reference_map(Path(args.reference_map))

    # Build UID -> molecule lookup
    mol_by_uid: dict[int, Molecule] = {m.uid: m for m in probes_file.molecules}

    # Build UID -> assignment lookup
    assign_by_uid = {a.fragment_uid: a for a in assignments}

    # Load transforms if provided
    transforms = None
    if args.transform:
        transforms = load_transforms(args.transform)

    # --- Load model ---
    logger.info("Loading model from %s", args.checkpoint)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = T2DUNet()
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # --- Build ground truth and evaluate ---
    logger.info("Building ground truth and running inference...")
    unet_pred_intervals: list[np.ndarray] = []
    gt_intervals: list[np.ndarray] = []
    legacy_pred_intervals: list[np.ndarray] = []
    model_peak_counts: list[int] = []
    reference_matched_counts: list[int] = []

    evaluated = 0
    skipped = 0

    for mol in probes_file.molecules:
        assign = assign_by_uid.get(mol.uid)
        if assign is None:
            skipped += 1
            continue

        gt = build_molecule_gt(mol, assign, ref_map)
        if gt is None:
            skipped += 1
            continue

        if len(gt.inter_probe_deltas_bp) == 0:
            skipped += 1
            continue

        gt_intervals.append(gt.inter_probe_deltas_bp)

        # --- U-Net inference ---
        # Build a simple waveform placeholder -- in real usage this would come
        # from a TDB file. For now we need the waveform data to run inference.
        # This script assumes a checkpoint was trained, so the waveform must
        # be loadable. For a minimal pipeline, we create a dummy waveform
        # of the right length and run inference. This is a placeholder that
        # should be replaced with actual waveform loading.
        num_samples = int(mol.transloc_time_ms / 0.025)  # 40kHz sample rate
        waveform = torch.zeros(1, 1, max(num_samples, 32), dtype=torch.float32)
        conditioning = _build_conditioning(mol)
        mask = torch.ones(1, max(num_samples, 32), dtype=torch.bool)

        result = run_inference(
            model, waveform, conditioning, mask, threshold=args.threshold, device=device
        )

        # Extract predicted intervals
        if len(result.intervals_bp) > 0:
            unet_pred = np.array(result.intervals_bp, dtype=np.float64)
        else:
            unet_pred = np.array([], dtype=np.float64)

        # Match lengths: use min of predicted and GT interval count
        n_gt = len(gt.inter_probe_deltas_bp)
        n_pred = len(unet_pred)
        n_common = min(n_gt, n_pred)

        if n_common > 0:
            unet_pred_intervals.append(unet_pred[:n_common])
            # Replace gt_intervals[-1] with trimmed version
            gt_intervals[-1] = gt.inter_probe_deltas_bp[:n_common]
        else:
            # No matching intervals -- remove the GT entry we just added
            gt_intervals.pop()
            skipped += 1
            continue

        # Record peak-count discrepancy vs wfmproc-matched probe count.
        # Matched count = number of wfmproc probes that aligned to the reference.
        # Recorded only after we commit to keeping this molecule so the list
        # length matches the number of evaluated molecules.
        model_peak_counts.append(len(result.probes))
        reference_matched_counts.append(len(gt.reference_probe_bp))

        # --- Legacy T2D baseline ---
        if transforms is not None:
            ch_key = f"Ch{mol.channel:03d}"
            ch_transform = transforms.get(ch_key)
            if ch_transform is not None:
                legacy_ivl = legacy_t2d_intervals(
                    mol, gt, ch_transform.mult_const, ch_transform.addit_const, ch_transform.alpha
                )
                legacy_pred_intervals.append(legacy_ivl[:n_common])
            else:
                # No transform for this channel, use NaN placeholder
                legacy_pred_intervals.append(
                    np.full(n_common, np.nan, dtype=np.float64)
                )
        evaluated += 1

    logger.info("Evaluated %d molecules, skipped %d", evaluated, skipped)

    if evaluated == 0:
        logger.error("No molecules could be evaluated.")
        sys.exit(1)

    # --- Compute metrics ---
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    unet_metrics = evaluate_intervals(unet_pred_intervals, gt_intervals)
    print(_format_metrics("U-Net", unet_metrics))

    legacy_metrics: EvalMetrics | None = None
    if transforms is not None and len(legacy_pred_intervals) > 0:
        # Filter out molecules with NaN legacy predictions
        valid_legacy = []
        valid_gt = []
        for lp, gp in zip(legacy_pred_intervals, gt_intervals):
            if not np.any(np.isnan(lp)):
                valid_legacy.append(lp)
                valid_gt.append(gp)

        if len(valid_legacy) > 0:
            legacy_metrics = evaluate_intervals(valid_legacy, valid_gt)
            print(_format_metrics("Legacy T2D", legacy_metrics))

            # Improvement summary
            if legacy_metrics.mae_bp > 0:
                improvement = (
                    (legacy_metrics.mae_bp - unet_metrics.mae_bp)
                    / legacy_metrics.mae_bp
                    * 100
                )
                print(f"\n  MAE improvement: {improvement:+.1f}%")
        else:
            print("  Legacy T2D: no valid predictions")

    # Peak-count discrepancy vs wfmproc-matched probe counts
    peak_stats: PeakCountStats | None = None
    if len(model_peak_counts) > 0:
        peak_stats = evaluate_peak_counts(model_peak_counts, reference_matched_counts)
        print(
            f"\n  Peak-count discrepancy (model - wfmproc): "
            f"mean={peak_stats.mean_discrepancy:+.2f}  "
            f"median={peak_stats.median_discrepancy:+.2f}  "
            f"std={peak_stats.std_discrepancy:.2f}  "
            f"more={peak_stats.fraction_more_detections*100:.0f}%  "
            f"fewer={peak_stats.fraction_fewer_detections*100:.0f}%  "
            f"equal={peak_stats.fraction_equal_detections*100:.0f}%"
        )

    print("=" * 80)

    # --- Optional JSON output ---
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict: dict = {
            "metadata": {
                "checkpoint": str(args.checkpoint),
                "run_id": args.run_id,
                "num_molecules_evaluated": unet_metrics.num_molecules,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "dl_model": {
                "mae_bp": unet_metrics.mae_bp,
                "median_ae_bp": unet_metrics.median_ae_bp,
                "std_ae_bp": unet_metrics.std_ae_bp,
                "num_intervals": unet_metrics.num_intervals,
            },
        }

        if legacy_metrics is not None:
            result_dict["legacy_t2d"] = {
                "mae_bp": legacy_metrics.mae_bp,
                "median_ae_bp": legacy_metrics.median_ae_bp,
                "std_ae_bp": legacy_metrics.std_ae_bp,
                "num_intervals": legacy_metrics.num_intervals,
            }

        if peak_stats is not None:
            result_dict["peak_count"] = {
                "mean_discrepancy": peak_stats.mean_discrepancy,
                "median_discrepancy": peak_stats.median_discrepancy,
                "std_discrepancy": peak_stats.std_discrepancy,
                "fraction_more_detections": peak_stats.fraction_more_detections,
                "fraction_fewer_detections": peak_stats.fraction_fewer_detections,
                "fraction_equal_detections": peak_stats.fraction_equal_detections,
                "num_molecules": peak_stats.num_molecules,
            }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result_dict, fh, indent=2)
        logger.info("Wrote JSON results to %s", output_path)


if __name__ == "__main__":
    main()
