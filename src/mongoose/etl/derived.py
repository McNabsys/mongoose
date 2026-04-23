"""Derived-column computations for the Phase 0 probe-table ETL.

Each function here is a pure transformation — no IO. They consume the
already-parsed inputs (probes.bin molecule/probe records, assigns,
referenceMap, settings) and return numpy / scalar values that the
probe_table builder drops into DataFrame columns.

A couple of derived fields are deliberately left unresolved — see
:mod:`mongoose.etl.probe_widths` and the schema comments for context.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Iterable

import numpy as np

from mongoose.etl.schema import ATTRIBUTE_BIT_FIELDS
from mongoose.inference.legacy_t2d import TDB_SAMPLE_RATE_HZ, legacy_t2d_bp_positions
from mongoose.io.probes_bin import Molecule


def unpack_attribute_bitfield(attribute: int) -> dict[str, bool]:
    """Expand a probe.attribute uint32 to named booleans.

    Uses the bit positions listed in the spec (§74-76):
    bit 0 = clean_region, 1 = folded_end, 2 = folded_start,
    3 = in_structure, 4 = excl_amp_high, 5 = excl_width_sp,
    6 = excl_width_remap, 7 = accepted, 8 = excl_outside_partial.
    """
    return {name: bool(attribute & (1 << bit)) for name, bit in ATTRIBUTE_BIT_FIELDS}


def compute_probe_gaps(center_ms_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-probe gaps to the previous/next probe in the same molecule.

    Args:
        center_ms_sorted: 1-D float array of probe centers sorted
            ascending by time within a single molecule.

    Returns:
        (prev_gap, next_gap) both length-N float32 with NaN at the
        molecule boundaries (first probe has no prev, last has no next).
    """
    n = center_ms_sorted.size
    prev_gap = np.full(n, np.nan, dtype=np.float32)
    next_gap = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        diffs = np.diff(center_ms_sorted).astype(np.float32)
        next_gap[:-1] = diffs
        prev_gap[1:] = diffs
    return prev_gap, next_gap


def compute_local_density(
    center_ms_sorted: np.ndarray, *, window_ms: float
) -> np.ndarray:
    """Count of probes within ±window_ms of each probe's center.

    Each probe counts itself, so a probe with no other probes nearby has
    density = 1.
    """
    n = center_ms_sorted.size
    density = np.empty(n, dtype=np.uint16)
    for i, t in enumerate(center_ms_sorted):
        lo = bisect_left(center_ms_sorted, t - window_ms)
        hi = bisect_right(center_ms_sorted, t + window_ms)
        density[i] = max(0, hi - lo)
    return density


def compute_molecule_velocity_bp_per_ms(
    molecule_bp_length: int | None, translocation_time_ms: float
) -> float | None:
    """bp per ms of translocation. None when length is null or time is non-positive."""
    if molecule_bp_length is None or translocation_time_ms <= 0:
        return None
    return float(molecule_bp_length) / float(translocation_time_ms)


def compute_t2d_predicted_bp(
    probe_centers_ms: Iterable[float],
    *,
    mol: Molecule,
    mult_const: float,
    addit_const: float,
    alpha: float,
    sample_rate_hz: int = TDB_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Apply production T2D to each probe's center_ms, return bp positions.

    ``probe_centers_ms`` is molecule-local (probes.bin convention). We
    convert to cached-waveform samples using the same formula the cache
    builder uses in ``data/ground_truth.py`` — the ``start_within_tdb_ms``
    offset algebraically cancels inside ``legacy_t2d_bp_positions`` but
    we pass the canonically-converted samples for consistency with
    any future changes to the T2D implementation.
    """
    centers = np.fromiter(probe_centers_ms, dtype=np.float64)
    if centers.size == 0:
        return np.empty(0, dtype=np.float64)
    sample_period_ms = 1000.0 / sample_rate_hz
    samples = np.round(
        (float(mol.start_within_tdb_ms) + centers) / sample_period_ms
    ).astype(np.int64)
    return legacy_t2d_bp_positions(
        samples,
        mol=mol,
        mult_const=mult_const,
        addit_const=addit_const,
        alpha=alpha,
        sample_rate_hz=sample_rate_hz,
    )


def compute_molecule_bp_length(
    probe_assigned_bp: Iterable[int | None],
) -> int | None:
    """Span of aligned probes in bp. None when fewer than 2 probes aligned."""
    positions = [int(p) for p in probe_assigned_bp if p is not None]
    if len(positions) < 2:
        return None
    return int(max(positions) - min(positions))
