"""Reference-bp-space peak matching.

Companion to ``peak_match.py``. The sample-space matcher there measures
how well a model replicates wfmproc's peak sample indices. That's not
the same as measuring whether the model predicts accurate basepair
positions on the reference genome.

This module does the sample-space -> reference-bp conversion using the
model's ``cumulative_bp`` output, zero-anchors both sequences with the
same convention as the training loss (``combined.py`` lines 246-247),
and matches via the existing Hungarian solver in ``peak_match.py``.

Zero-anchor convention (matches training L_bp):

    pred_bp_deltas[k] = pred_cumulative_bp[pred_peak_samples[k]]
                      - pred_cumulative_bp[pred_peak_samples[0]]
    ref_bp_deltas[k]  = abs(reference_bp_positions[k] - reference_bp_positions[0])

The ``.abs()`` on the reference side handles reverse molecules (descending
bp in temporal order) without changing the sign convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mongoose.inference.peak_match import compute_metrics, match_peaks


def compute_reference_bp_deltas(
    pred_peak_samples: np.ndarray,
    pred_cumulative_bp: np.ndarray,
    reference_bp_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-anchor predicted and reference bp sequences at their firsts.

    Args:
        pred_peak_samples: int64 [K] array of predicted peak sample indices.
        pred_cumulative_bp: float [T] array of model-predicted cumulative bp.
        reference_bp_positions: int64 [N] array of reference bp coordinates
            (from ``reference_map.bin`` via ``assigns.bin``).

    Returns:
        ``(pred_bp_deltas [K], ref_bp_deltas [N])``. When ``K == 0``,
        ``pred_bp_deltas`` has shape ``(0,)`` and ``ref_bp_deltas`` is still
        returned zero-anchored. When ``N == 0``, ``ref_bp_deltas`` is empty.
    """
    ref = np.asarray(reference_bp_positions, dtype=np.float64)
    if ref.size == 0:
        ref_deltas = np.zeros((0,), dtype=np.float64)
    else:
        ref_deltas = np.abs(ref - ref[0])

    pred_samples = np.asarray(pred_peak_samples, dtype=np.int64)
    if pred_samples.size == 0:
        pred_deltas = np.zeros((0,), dtype=np.float64)
    else:
        cum = np.asarray(pred_cumulative_bp, dtype=np.float64)
        pred_bp_at_peaks = cum[pred_samples]
        pred_deltas = pred_bp_at_peaks - pred_bp_at_peaks[0]

    return pred_deltas, ref_deltas


def evaluate_molecule_bp(
    *,
    pred_peak_samples: np.ndarray,
    pred_cumulative_bp: np.ndarray,
    reference_bp_positions: np.ndarray,
    tolerance_bp: float,
) -> dict[str, Any]:
    """Match predicted peaks to reference probes in bp space.

    Uses the Hungarian assignment from ``peak_match.match_peaks`` on
    zero-anchored bp deltas, with the tolerance in bp.

    Args:
        pred_peak_samples: int64 [K] array of predicted peak sample indices.
        pred_cumulative_bp: float [T] array of cumulative bp from the model.
        reference_bp_positions: int64 [N] array of reference bp coordinates.
        tolerance_bp: matching tolerance in basepairs (distance metric is
            absolute bp-delta error on matched pairs).

    Returns:
        Dict with keys:
            tp, fp, fn (ints), precision, recall, f1 (floats),
            n_pred, n_ref (ints), bp_mae_matched (float or NaN when no
            matches).
    """
    pred_deltas, ref_deltas = compute_reference_bp_deltas(
        pred_peak_samples, pred_cumulative_bp, reference_bp_positions
    )
    n_pred = int(pred_deltas.shape[0])
    n_ref = int(ref_deltas.shape[0])

    if n_pred == 0 or n_ref == 0:
        tp = 0
        fp = n_pred
        fn = n_ref
        bp_mae = float("nan")
    else:
        matches, unmatched_preds, unmatched_refs = match_peaks(
            pred_deltas, ref_deltas, tolerance=float(tolerance_bp)
        )
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_refs)
        if tp > 0:
            errors = [abs(pred_deltas[p] - ref_deltas[r]) for p, r in matches]
            bp_mae = float(np.mean(errors))
        else:
            bp_mae = float("nan")

    metrics = compute_metrics(tp=tp, fp=fp, fn=fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_pred": n_pred,
        "n_ref": n_ref,
        "bp_mae_matched": bp_mae,
    }
