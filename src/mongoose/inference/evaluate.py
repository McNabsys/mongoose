"""Evaluation metrics for predicted vs ground truth inter-probe intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics across molecules."""

    mae_bp: float  # mean absolute error on inter-probe intervals
    median_ae_bp: float  # median absolute error
    std_ae_bp: float  # std of absolute errors
    num_molecules: int  # how many molecules evaluated
    num_intervals: int  # total intervals compared
    per_molecule_mae: list[float]  # per-molecule MAE for distribution analysis


def evaluate_intervals(
    predicted_intervals: list[np.ndarray],
    gt_intervals: list[np.ndarray],
) -> EvalMetrics:
    """Compare predicted vs ground truth inter-probe intervals.

    Both lists have the same length (one entry per molecule).
    Each entry is a 1D array of intervals. They must have the same length
    per molecule (same number of matched probes).

    Args:
        predicted_intervals: Per-molecule predicted inter-probe bp.
        gt_intervals: Per-molecule ground truth inter-probe bp.

    Returns:
        EvalMetrics with aggregated statistics.

    Raises:
        ValueError: If list lengths or per-molecule array lengths don't match.
    """
    if len(predicted_intervals) != len(gt_intervals):
        raise ValueError(
            f"Molecule count mismatch: {len(predicted_intervals)} predicted "
            f"vs {len(gt_intervals)} ground truth"
        )

    all_errors: list[float] = []
    per_molecule_mae: list[float] = []

    for i, (pred, gt) in enumerate(zip(predicted_intervals, gt_intervals)):
        if len(pred) != len(gt):
            raise ValueError(
                f"Molecule {i}: interval count mismatch: "
                f"{len(pred)} predicted vs {len(gt)} ground truth"
            )
        errors = np.abs(pred - gt)
        all_errors.extend(errors.tolist())
        if len(errors) > 0:
            per_molecule_mae.append(float(np.mean(errors)))
        else:
            per_molecule_mae.append(0.0)

    all_errors_arr = np.array(all_errors, dtype=np.float64)

    return EvalMetrics(
        mae_bp=float(np.mean(all_errors_arr)) if len(all_errors_arr) > 0 else 0.0,
        median_ae_bp=float(np.median(all_errors_arr)) if len(all_errors_arr) > 0 else 0.0,
        std_ae_bp=float(np.std(all_errors_arr)) if len(all_errors_arr) > 0 else 0.0,
        num_molecules=len(predicted_intervals),
        num_intervals=len(all_errors),
        per_molecule_mae=per_molecule_mae,
    )
