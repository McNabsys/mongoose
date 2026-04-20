"""Tests for reference-bp-space peak matching.

Motivation: existing ``evaluate_peak_match.py`` compares predicted peak
sample indices against wfmproc ``warmstart_probe_centers_samples``. That
measures wfmproc-replication fidelity, not reference-genome bp accuracy.
These tests pin down the pure-function helpers that let us match peaks in
*reference bp space* instead.
"""

from __future__ import annotations

import numpy as np

from mongoose.inference.bp_match import (
    compute_reference_bp_deltas,
    evaluate_molecule_bp,
)


def test_deltas_forward_direction_match_training_convention():
    """Pred deltas subtract pred_bp at first peak; ref deltas are |ref - ref[0]|.

    Matches ``combined.py`` lines 246-247 so evaluation and training use
    identical zero-anchoring.
    """
    # Cumulative bp ramps 0 -> 1000 linearly over 1001 samples.
    pred_cumulative_bp = np.linspace(0.0, 1000.0, 1001, dtype=np.float32)
    pred_peak_samples = np.array([100, 400, 900], dtype=np.int64)
    # Ascending reference bp (forward molecule).
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    pred_deltas, ref_deltas = compute_reference_bp_deltas(
        pred_peak_samples, pred_cumulative_bp, reference_bp_positions
    )

    # pred_cumulative_bp at [100,400,900] is [100,400,900]; minus [0]=100 → [0,300,800]
    np.testing.assert_allclose(pred_deltas, [0.0, 300.0, 800.0], rtol=1e-5)
    # ref: |[5000,5300,5800] - 5000| = [0,300,800]
    np.testing.assert_allclose(ref_deltas, [0.0, 300.0, 800.0], rtol=1e-5)


def test_deltas_reverse_direction_reference_is_absolute():
    """Reverse molecules have descending ref bp; ref deltas take abs()."""
    pred_cumulative_bp = np.linspace(0.0, 1000.0, 1001, dtype=np.float32)
    pred_peak_samples = np.array([100, 400, 900], dtype=np.int64)
    # Descending reference bp (reverse molecule).
    reference_bp_positions = np.array([5800, 5300, 5000], dtype=np.int64)

    _, ref_deltas = compute_reference_bp_deltas(
        pred_peak_samples, pred_cumulative_bp, reference_bp_positions
    )

    # |[5800,5300,5000] - 5800| = [0, 500, 800]
    np.testing.assert_allclose(ref_deltas, [0.0, 500.0, 800.0], rtol=1e-5)


def test_deltas_empty_predictions_returns_empty_arrays():
    """Zero predicted peaks → empty pred_deltas and empty ref_deltas."""
    pred_cumulative_bp = np.linspace(0.0, 1000.0, 1001, dtype=np.float32)
    pred_peak_samples = np.array([], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    pred_deltas, ref_deltas = compute_reference_bp_deltas(
        pred_peak_samples, pred_cumulative_bp, reference_bp_positions
    )
    assert pred_deltas.shape == (0,)
    # Ref deltas should always be produced regardless of pred count.
    assert ref_deltas.shape == (3,)


def test_deltas_single_peak_gives_zero_delta():
    """One predicted peak → pred_deltas is [0]."""
    pred_cumulative_bp = np.linspace(0.0, 1000.0, 1001, dtype=np.float32)
    pred_peak_samples = np.array([500], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    pred_deltas, _ = compute_reference_bp_deltas(
        pred_peak_samples, pred_cumulative_bp, reference_bp_positions
    )
    np.testing.assert_allclose(pred_deltas, [0.0], rtol=1e-5)


def test_evaluate_molecule_bp_perfect_prediction_gives_f1_one():
    """Predictions land exactly at reference bp deltas → precision=recall=F1=1."""
    # Cumulative bp ramps 0 -> 10000 linearly over 10001 samples.
    pred_cumulative_bp = np.linspace(0.0, 10000.0, 10001, dtype=np.float32)
    # Reference probes at bp 5000, 5300, 5800 → expected deltas [0, 300, 800].
    # Pick predicted peak samples so cumulative_bp deltas match exactly.
    # At samples [s0, s0+300, s0+800] the deltas are [0, 300, 800] for any s0.
    pred_peak_samples = np.array([2000, 2300, 2800], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    result = evaluate_molecule_bp(
        pred_peak_samples=pred_peak_samples,
        pred_cumulative_bp=pred_cumulative_bp,
        reference_bp_positions=reference_bp_positions,
        tolerance_bp=50.0,
    )

    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["bp_mae_matched"] == 0.0
    assert result["n_pred"] == 3
    assert result["n_ref"] == 3


def test_evaluate_molecule_bp_missed_probe_counts_as_fn():
    """Missing the middle reference probe → 2 TP, 0 FP, 1 FN."""
    pred_cumulative_bp = np.linspace(0.0, 10000.0, 10001, dtype=np.float32)
    # Only 2 predicted peaks; they align to ref[0] and ref[2] (bp-delta 0 and 800).
    pred_peak_samples = np.array([2000, 2800], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    result = evaluate_molecule_bp(
        pred_peak_samples=pred_peak_samples,
        pred_cumulative_bp=pred_cumulative_bp,
        reference_bp_positions=reference_bp_positions,
        tolerance_bp=50.0,
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 1


def test_evaluate_molecule_bp_spurious_peak_counts_as_fp():
    """Extra predicted peak outside any reference → 3 TP, 1 FP, 0 FN."""
    pred_cumulative_bp = np.linspace(0.0, 10000.0, 10001, dtype=np.float32)
    # Four predicted peaks; three match references, one is far off (5000 bp delta).
    pred_peak_samples = np.array([2000, 2300, 2800, 7000], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    result = evaluate_molecule_bp(
        pred_peak_samples=pred_peak_samples,
        pred_cumulative_bp=pred_cumulative_bp,
        reference_bp_positions=reference_bp_positions,
        tolerance_bp=50.0,
    )

    assert result["tp"] == 3
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_evaluate_molecule_bp_reports_bp_mae_at_matched_peaks():
    """bp_mae_matched is the mean absolute bp error over matched pairs only."""
    pred_cumulative_bp = np.linspace(0.0, 10000.0, 10001, dtype=np.float32)
    # Peaks off by +10, +20, +30 bp in cumulative space.
    pred_peak_samples = np.array([2000, 2310, 2830], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    result = evaluate_molecule_bp(
        pred_peak_samples=pred_peak_samples,
        pred_cumulative_bp=pred_cumulative_bp,
        reference_bp_positions=reference_bp_positions,
        tolerance_bp=50.0,
    )

    # Pred deltas (anchored at pred_peak_samples[0]=2000, cumulative_bp[2000]=2000):
    #   [0, 310, 830]
    # Ref deltas: [0, 300, 800]
    # Errors: |0-0|, |310-300|, |830-800| = [0, 10, 30]
    # Mean: 40/3 ≈ 13.333
    assert result["tp"] == 3
    np.testing.assert_allclose(result["bp_mae_matched"], (0 + 10 + 30) / 3, rtol=1e-5)


def test_evaluate_molecule_bp_empty_pred_is_all_fn():
    """Zero predicted peaks with N reference probes → 0 TP, 0 FP, N FN."""
    pred_cumulative_bp = np.linspace(0.0, 10000.0, 10001, dtype=np.float32)
    pred_peak_samples = np.array([], dtype=np.int64)
    reference_bp_positions = np.array([5000, 5300, 5800], dtype=np.int64)

    result = evaluate_molecule_bp(
        pred_peak_samples=pred_peak_samples,
        pred_cumulative_bp=pred_cumulative_bp,
        reference_bp_positions=reference_bp_positions,
        tolerance_bp=50.0,
    )

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 3
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    # MAE undefined when no matches; reported as None or NaN (we'll use NaN).
    assert np.isnan(result["bp_mae_matched"])


def test_evaluate_molecule_bp_empty_ref_is_all_fp():
    """N predictions with zero reference probes → 0 TP, N FP, 0 FN."""
    pred_cumulative_bp = np.linspace(0.0, 10000.0, 10001, dtype=np.float32)
    pred_peak_samples = np.array([2000, 2300, 2800], dtype=np.int64)
    reference_bp_positions = np.array([], dtype=np.int64)

    result = evaluate_molecule_bp(
        pred_peak_samples=pred_peak_samples,
        pred_cumulative_bp=pred_cumulative_bp,
        reference_bp_positions=reference_bp_positions,
        tolerance_bp=50.0,
    )

    assert result["tp"] == 0
    assert result["fp"] == 3
    assert result["fn"] == 0
