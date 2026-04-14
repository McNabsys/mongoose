"""Tests for evaluation metrics computation."""

import numpy as np
import pytest

from mongoose.inference.evaluate import (
    EvalMetrics,
    PeakCountStats,
    evaluate_intervals,
    evaluate_peak_counts,
)


def test_evaluate_perfect_prediction():
    gt = [np.array([1000.0, 2000.0, 3000.0])]
    pred = [np.array([1000.0, 2000.0, 3000.0])]
    metrics = evaluate_intervals(pred, gt)
    assert metrics.mae_bp < 0.01
    assert metrics.num_molecules == 1
    assert metrics.num_intervals == 3


def test_evaluate_with_errors():
    gt = [np.array([1000.0, 2000.0])]
    pred = [np.array([1100.0, 1900.0])]  # +100, -100 error
    metrics = evaluate_intervals(pred, gt)
    assert abs(metrics.mae_bp - 100.0) < 0.01
    assert metrics.median_ae_bp == 100.0


def test_evaluate_multiple_molecules():
    gt = [np.array([1000.0]), np.array([2000.0, 3000.0])]
    pred = [np.array([1050.0]), np.array([2100.0, 2900.0])]
    metrics = evaluate_intervals(pred, gt)
    assert metrics.num_molecules == 2
    assert metrics.num_intervals == 3


def test_evaluate_std_ae():
    """Std should be zero when all errors are identical."""
    gt = [np.array([1000.0, 2000.0, 3000.0])]
    pred = [np.array([1050.0, 2050.0, 3050.0])]
    metrics = evaluate_intervals(pred, gt)
    assert abs(metrics.mae_bp - 50.0) < 0.01
    assert abs(metrics.std_ae_bp) < 0.01  # all errors are 50, std = 0


def test_evaluate_per_molecule_mae():
    gt = [np.array([1000.0, 2000.0]), np.array([3000.0])]
    pred = [np.array([1100.0, 2200.0]), np.array([3050.0])]
    metrics = evaluate_intervals(pred, gt)
    assert len(metrics.per_molecule_mae) == 2
    assert abs(metrics.per_molecule_mae[0] - 150.0) < 0.01  # (100+200)/2
    assert abs(metrics.per_molecule_mae[1] - 50.0) < 0.01


def test_evaluate_mismatched_lengths_raises():
    gt = [np.array([1000.0])]
    pred = [np.array([1000.0, 2000.0])]
    with pytest.raises(ValueError):
        evaluate_intervals(pred, gt)


def test_evaluate_mismatched_molecule_count_raises():
    gt = [np.array([1000.0])]
    pred = [np.array([1000.0]), np.array([2000.0])]
    with pytest.raises(ValueError):
        evaluate_intervals(pred, gt)


# --- Peak-count discrepancy tests ---


def test_peak_count_equal():
    stats = evaluate_peak_counts([3, 4, 5], [3, 4, 5])
    assert stats.mean_discrepancy == 0.0
    assert stats.median_discrepancy == 0.0
    assert stats.std_discrepancy == 0.0
    assert stats.fraction_equal_detections == 1.0
    assert stats.fraction_more_detections == 0.0
    assert stats.fraction_fewer_detections == 0.0
    assert stats.num_molecules == 3


def test_peak_count_model_finds_more():
    # Discrepancies: +2, +2, +2 -> mean = 2.0
    stats = evaluate_peak_counts([5, 6, 5], [3, 4, 3])
    assert stats.mean_discrepancy == 2.0
    assert stats.fraction_more_detections == 1.0
    assert stats.fraction_equal_detections == 0.0
    assert stats.fraction_fewer_detections == 0.0


def test_peak_count_model_finds_fewer():
    stats = evaluate_peak_counts([2, 3], [4, 5])
    assert stats.mean_discrepancy == -2.0
    assert stats.fraction_fewer_detections == 1.0
    assert stats.fraction_more_detections == 0.0
    assert stats.fraction_equal_detections == 0.0


def test_peak_count_mixed():
    stats = evaluate_peak_counts([3, 5, 2], [3, 4, 3])
    # Discrepancies: 0, +1, -1
    assert abs(stats.mean_discrepancy - 0.0) < 1e-6
    assert abs(stats.fraction_more_detections - 1 / 3) < 1e-6
    assert abs(stats.fraction_fewer_detections - 1 / 3) < 1e-6
    assert abs(stats.fraction_equal_detections - 1 / 3) < 1e-6
    assert stats.num_molecules == 3


def test_peak_count_median_and_std():
    # Discrepancies: -2, 0, +2, +4 -> median=1.0
    stats = evaluate_peak_counts([3, 5, 7, 9], [5, 5, 5, 5])
    assert abs(stats.median_discrepancy - 1.0) < 1e-6
    assert stats.std_discrepancy > 0.0


def test_peak_count_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        evaluate_peak_counts([1, 2, 3], [1, 2])


def test_peak_count_empty_raises():
    with pytest.raises(ValueError):
        evaluate_peak_counts([], [])
