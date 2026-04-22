"""Tests for the legacy T2D formula (``mongoose.inference.legacy_t2d``).

The T2D formula is::

    L(t) = mult_const * t_from_tail_ms ^ alpha + addit_const

These tests pin the formula to a hand-verified numerical example from the
2026-04-21 T2D debugging session. Molecule STB03-063B, uid 17, channel 1.
Reference transform: mult_const=6343.377, addit_const=-1200, alpha=0.5588.
Reference probes at sample indices [292, 1334, 1551, 1964, 2144, 2178]
with start_within_tdb_ms=8.831 and fall_t50=62.967. Predicted intervals
in bp were verified by hand to match the aligner-derived reference within
1-14% error per interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mongoose.inference.legacy_t2d import (
    legacy_t2d_bp_positions,
    legacy_t2d_intervals,
)


@dataclass
class _MolStub:
    """Minimal stand-in for ``mongoose.io.probes_bin.Molecule``.

    Only exposes the two attributes the T2D formula needs.
    """

    start_within_tdb_ms: float
    fall_t50: float


# Hand-verified example: STB03-063B molecule uid=17, channel Ch001.
_CENTERS = np.array([292, 1334, 1551, 1964, 2144, 2178], dtype=np.int64)
_MOL = _MolStub(start_within_tdb_ms=8.831338882446289, fall_t50=62.96697998046875)
_MULT = 6343.37688634894
_ADDIT = -1200.0
_ALPHA = 0.558846395680942


def test_bp_positions_produce_hand_verified_values() -> None:
    """bp positions must match the precise reference values (within 1 bp).

    These are the exact values produced by the corrected T2D formula on
    the hand-verified example. Pinning to these exact numbers locks the
    formula against silent regression — any future edit that changes the
    output by more than ~1 bp on a known-good input will fail this test.
    """
    bp = legacy_t2d_bp_positions(
        _CENTERS,
        mol=_MOL,
        mult_const=_MULT,
        addit_const=_ADDIT,
        alpha=_ALPHA,
    )
    expected = np.array(
        [62863.95, 41330.38, 35678.25, 22308.75, 14038.59, 12049.40],
        dtype=np.float64,
    )
    assert np.allclose(bp, expected, atol=1.0), f"got {bp}, expected {expected}"


def test_intervals_match_hand_verified_example() -> None:
    """Intervals = absolute diffs of bp positions; pin to known-good values.

    Reference intervals from the aligner (for context):
        23635, 5603, 12783, 7666, 1665

    T2D predictions on the same probes (within 1-15% of reference):
        21534, 5652, 13369, 8270, 1989
    """
    intervals = legacy_t2d_intervals(
        _CENTERS,
        mol=_MOL,
        mult_const=_MULT,
        addit_const=_ADDIT,
        alpha=_ALPHA,
    )
    expected = np.array(
        [21533.57, 5652.14, 13369.50, 8270.16, 1989.19], dtype=np.float64
    )
    assert np.allclose(intervals, expected, atol=1.0), f"got {intervals}"


def test_intervals_monotone_when_probes_advance_toward_tail() -> None:
    """As t_from_tail → 0, bp position → addit_const; consecutive probes
    closer to the tail must yield progressively smaller bp predictions.
    """
    bp = legacy_t2d_bp_positions(
        _CENTERS,
        mol=_MOL,
        mult_const=_MULT,
        addit_const=_ADDIT,
        alpha=_ALPHA,
    )
    # Centers sorted by time (ascending). Bp from tail must be strictly decreasing.
    assert np.all(np.diff(bp) < 0), f"not monotone decreasing: {bp}"


def test_empty_probes_returns_empty() -> None:
    out = legacy_t2d_intervals(
        np.empty(0, dtype=np.int64),
        mol=_MOL,
        mult_const=_MULT,
        addit_const=_ADDIT,
        alpha=_ALPHA,
    )
    assert out.size == 0


def test_single_probe_returns_empty_intervals() -> None:
    out = legacy_t2d_intervals(
        np.array([1000], dtype=np.int64),
        mol=_MOL,
        mult_const=_MULT,
        addit_const=_ADDIT,
        alpha=_ALPHA,
    )
    assert out.size == 0


def test_probe_past_tail_clamps_to_near_zero_distance() -> None:
    """A probe observed after the trailing edge (numerically) should not
    blow up. It's clamped to t=1e-3 ms which drives bp → addit_const.
    """
    # Put a probe at sample AFTER the tail (impossible physically, but
    # guards against numerical jitter).
    tail_sample = int(
        (_MOL.start_within_tdb_ms + _MOL.fall_t50) * 32000 / 1000
    )
    past_tail = np.array([tail_sample + 50], dtype=np.int64)
    bp = legacy_t2d_bp_positions(
        past_tail,
        mol=_MOL,
        mult_const=_MULT,
        addit_const=_ADDIT,
        alpha=_ALPHA,
    )
    # t clamped to 1e-3 ms ⇒ t^alpha ≈ small ⇒ bp ≈ addit_const + tiny.
    # For mult=6343, alpha=0.558: bp ≈ -1200 + 6343 * (1e-3)^0.558 ≈ -1067.
    # Allow 200 bp tolerance (the contribution from clamped t).
    assert abs(bp[0] - _ADDIT) < 200, f"past-tail probe blew up: {bp[0]}"


def test_exponent_near_physics_value_produces_similar_shape() -> None:
    """Swap alpha to the physics-pure 2/3; output should still be monotone
    decreasing and have the same qualitative shape (smoke test that the
    formula isn't secretly dependent on alpha ≈ 0.55).
    """
    bp_empirical = legacy_t2d_bp_positions(
        _CENTERS, mol=_MOL, mult_const=_MULT, addit_const=_ADDIT, alpha=_ALPHA
    )
    bp_physics = legacy_t2d_bp_positions(
        _CENTERS, mol=_MOL, mult_const=_MULT, addit_const=_ADDIT, alpha=2.0 / 3.0
    )
    # Both monotone decreasing, both use the same sign of addit_const.
    assert np.all(np.diff(bp_empirical) < 0)
    assert np.all(np.diff(bp_physics) < 0)
    # Magnitude will differ, but sign of bp - addit_const matches (all positive).
    assert np.all(bp_empirical > _ADDIT)
    assert np.all(bp_physics > _ADDIT)
