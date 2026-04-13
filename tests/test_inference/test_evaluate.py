"""Tests for evaluation metrics computation."""

import numpy as np
import pytest

from mongoose.inference.evaluate import EvalMetrics, evaluate_intervals


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
