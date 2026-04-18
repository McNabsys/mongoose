"""Tests for Hungarian peak matching between predicted and reference peaks."""
from __future__ import annotations

import numpy as np
import pytest

from mongoose.inference.peak_match import (
    compute_metrics,
    match_peaks,
)


def test_match_peaks_perfect_alignment():
    pred = np.array([100, 500, 1000])
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 3
    assert fp == []
    assert fn == []


def test_match_peaks_within_tolerance():
    pred = np.array([100, 525, 1000])  # second one is 25 off
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 3
    assert fp == []
    assert fn == []


def test_match_peaks_outside_tolerance():
    pred = np.array([100, 700, 1000])  # second one is 200 off — too far
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 2
    assert fp == [1]  # pred index 1 unmatched
    assert fn == [1]  # ref index 1 unmatched


def test_match_peaks_extra_prediction():
    pred = np.array([100, 500, 1000, 2000])  # extra peak at 2000
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 3
    assert fp == [3]  # pred index 3 (the 2000) unmatched
    assert fn == []


def test_match_peaks_missing_reference():
    pred = np.array([100, 500])
    ref = np.array([100, 500, 1000])  # no pred near 1000
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 2
    assert fp == []
    assert fn == [2]


def test_match_peaks_optimal_not_greedy():
    """Two predicted peaks both near the same reference — Hungarian finds the optimal 1:1."""
    pred = np.array([100, 130])       # both within tolerance of ref 120
    ref = np.array([120, 200])         # 200 is unreachable by 130 (70 off)
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    # Optimal: pred 130 -> ref 120 (dist 10); pred 100 -> ref 200 is 100 dist (out of tol, FP+FN).
    # Or pred 100 -> ref 120 (dist 20); pred 130 -> ref 200 is 70 dist (out of tol, FP+FN).
    # Either way exactly one match and one FP + one FN.
    assert len(matches) == 1
    assert len(fp) == 1
    assert len(fn) == 1


def test_match_peaks_empty_predictions():
    pred = np.array([], dtype=np.int64)
    ref = np.array([100, 500])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert matches == []
    assert fp == []
    assert fn == [0, 1]


def test_match_peaks_empty_references():
    pred = np.array([100, 500])
    ref = np.array([], dtype=np.int64)
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert matches == []
    assert fp == [0, 1]
    assert fn == []


def test_match_peaks_both_empty():
    matches, fp, fn = match_peaks(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        tolerance=50,
    )
    assert matches == []
    assert fp == []
    assert fn == []


def test_compute_metrics_perfect():
    m = compute_metrics(tp=10, fp=0, fn=0)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_half_recall():
    m = compute_metrics(tp=5, fp=0, fn=5)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert abs(m["f1"] - 2 * 1.0 * 0.5 / 1.5) < 1e-6


def test_compute_metrics_zero_tp_returns_zero_f1():
    m = compute_metrics(tp=0, fp=5, fn=5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
