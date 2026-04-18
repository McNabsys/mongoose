"""Optimal 1:1 peak matching between predicted and reference peak positions.

Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) over
the pairwise absolute-distance matrix, with out-of-tolerance pairs blocked
by a sentinel of `np.inf` so the solver cannot select them.

Note on matching semantics: this is optimal 1:1 assignment, not greedy-by-
confidence as in COCO/VOC F1. Good for internal model comparisons; not
directly benchmark-comparable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

def match_peaks(
    pred_positions: np.ndarray,
    ref_positions: np.ndarray,
    tolerance: float = 50.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted peaks to reference peaks optimally at ``tolerance``.

    Args:
        pred_positions: 1D integer array of predicted peak positions (sample indices).
        ref_positions: 1D integer array of reference peak positions.
        tolerance: Absolute-distance tolerance for a valid match.

    Returns:
        ``(matches, unmatched_preds, unmatched_refs)`` where ``matches`` is a
        list of ``(pred_idx, ref_idx)`` index pairs and the two lists hold
        indices into ``pred_positions`` / ``ref_positions`` for unmatched
        peaks.
    """
    n_pred = int(pred_positions.shape[0])
    n_ref = int(ref_positions.shape[0])

    if n_pred == 0 and n_ref == 0:
        return [], [], []
    if n_pred == 0:
        return [], [], list(range(n_ref))
    if n_ref == 0:
        return [], list(range(n_pred)), []

    # Pairwise absolute distances. Shape: (n_pred, n_ref).
    pred_f = pred_positions.astype(np.float64)
    ref_f = ref_positions.astype(np.float64)
    dist = np.abs(pred_f[:, None] - ref_f[None, :])

    # Block out-of-tolerance pairs with np.inf so the solver cannot select
    # them.  Because linear_sum_assignment requires a complete assignment on
    # square matrices, we pad with dummy rows/columns at (tolerance + 1) so
    # blocked peaks can route to a dummy "unmatched" slot.  Real in-tolerance
    # matches (cost ≤ tolerance) are always cheaper than the dummy cost, so
    # the solver prefers them; dummy-to-dummy slots are set to 0.
    n_p, n_r = dist.shape
    cost = np.where(dist <= tolerance, dist, np.inf)
    dummy_cost = tolerance + 1.0
    padded = np.full((n_p + n_r, n_p + n_r), dummy_cost)
    padded[:n_p, :n_r] = cost
    padded[n_p:, n_r:] = 0.0

    row_ind, col_ind = linear_sum_assignment(padded)

    matches: list[tuple[int, int]] = []
    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n_p and c < n_r and dist[r, c] <= tolerance:
            matches.append((int(r), int(c)))
            matched_pred.add(int(r))
            matched_ref.add(int(c))

    unmatched_preds = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_refs = [i for i in range(n_ref) if i not in matched_ref]
    return matches, unmatched_preds, unmatched_refs


def compute_metrics(*, tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision / recall / F1 from TP / FP / FN counts.

    Returns zeros for undefined cases (tp=fp=0 or tp=fn=0) instead of raising.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_per_molecule_metrics(
    per_molecule: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate mean precision / recall / F1 across a list of per-molecule metric dicts.

    Each input dict must contain ``"precision"``, ``"recall"``, ``"f1"``.
    Returns a dict with the same keys whose values are the arithmetic means.
    """
    if not per_molecule:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_molecules": 0}
    p = float(np.mean([m["precision"] for m in per_molecule]))
    r = float(np.mean([m["recall"] for m in per_molecule]))
    f = float(np.mean([m["f1"] for m in per_molecule]))
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "n_molecules": len(per_molecule),
    }
