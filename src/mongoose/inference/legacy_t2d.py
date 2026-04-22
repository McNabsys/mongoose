"""Legacy T2D conversion: power-law model for time-to-distance mapping.

Reference: ``support/T2D.pdf`` (Oliver, 2023). The physics derivation
yields, for a RecA-coated DNA filament under translocation through a
nanochannel detector::

    L_TE = C' * t^(2/3)

where L_TE is the length of molecule that has translocated past the
trailing edge at time t. The empirical Nabsys implementation generalizes
this with a fitted exponent and a bp-space calibration offset:

    L(t) = mult_const * t_from_tail_ms ^ alpha + addit_const

with units:

  * ``t_from_tail_ms`` — time before the trailing edge, in MILLISECONDS.
  * ``mult_const`` — bp / (ms ^ alpha).
  * ``alpha`` — empirically ~0.55 for the Nabsys system (close to but
    slightly below the physics-pure 2/3).
  * ``addit_const`` — additive offset to the result, in bp.

Per-channel ``(mult_const, addit_const, alpha)`` triples are produced by
Nabsys tooling alongside remapping and live in ``_transForm.txt``.

History note: an earlier version of this module had the formula wrong —
it used samples for ``t`` and treated ``addit_const`` as a time-space
offset inside the power. That version produced bp magnitudes ~5x off
from reference. Fixed 2026-04-21 after the T2D.pdf derivation was
located. The deprecated ``scripts/evaluate.py`` had no end-to-end test
of the formula, which is how the bug went unnoticed.
"""

from __future__ import annotations

import numpy as np

from mongoose.io.probes_bin import Molecule

# Nabsys TDB sample rate (Hz). Used to convert sample-domain probe centers
# (as stored in cached training data) to milliseconds for the T2D formula.
TDB_SAMPLE_RATE_HZ = 32000


def legacy_t2d_bp_positions(
    probe_centers_samples: np.ndarray,
    *,
    mol: Molecule,
    mult_const: float,
    addit_const: float,
    alpha: float,
    sample_rate_hz: int = TDB_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Compute per-probe bp positions (distance from molecule trailing edge).

    Args:
        probe_centers_samples: 1D int64 array of probe center sample
            indices in CACHED-WAVEFORM coordinates (i.e., they include
            the molecule's ``start_within_tdb_ms`` offset, matching the
            convention used by ``MoleculeGT.warmstart_probe_centers_samples``).
        mol: ``Molecule`` from probes.bin. Used for ``start_within_tdb_ms``
            (cache-coordinate offset) and ``fall_t50`` (trailing-edge time
            within the molecule). Keyword-only.
        mult_const: Per-channel multiplicative constant from
            ``_transForm.txt``. Keyword-only.
        addit_const: Per-channel additive offset to the bp result, from
            ``_transForm.txt``. Keyword-only.
        alpha: Per-channel exponent from ``_transForm.txt``. Keyword-only.
        sample_rate_hz: TDB sample rate for samples-to-ms conversion.
            Default 32000.

    Returns:
        1D float64 array of bp positions, one per probe. The position is
        the distance from the molecule's trailing edge backwards into
        the molecule (i.e., L_TE in the physics derivation).
    """
    sample_period_ms = 1000.0 / sample_rate_hz
    centers_ms = probe_centers_samples.astype(np.float64) * sample_period_ms

    # Tail (= falling-edge time) in cached-waveform milliseconds.
    tail_ms = float(mol.start_within_tdb_ms) + float(mol.fall_t50)

    # Probes physically must precede the trailing edge; clamp at a tiny
    # positive value to keep the power well-defined for any near-tail
    # probe that crept past due to numerical jitter.
    t_from_tail_ms = np.maximum(tail_ms - centers_ms, 1e-3)

    return mult_const * np.power(t_from_tail_ms, alpha) + addit_const


def legacy_t2d_intervals(
    probe_centers_samples: np.ndarray,
    *,
    mol: Molecule,
    mult_const: float,
    addit_const: float,
    alpha: float,
    sample_rate_hz: int = TDB_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Compute inter-probe bp intervals using the legacy T2D formula.

    Convenience wrapper around :func:`legacy_t2d_bp_positions`. Returns
    the absolute differences between consecutive predicted bp positions —
    the per-molecule output that's actually compared against assembler-
    bound interval lists.

    Args:
        probe_centers_samples: See :func:`legacy_t2d_bp_positions`.
        mol: See :func:`legacy_t2d_bp_positions`.
        mult_const: See :func:`legacy_t2d_bp_positions`.
        addit_const: See :func:`legacy_t2d_bp_positions`.
        alpha: See :func:`legacy_t2d_bp_positions`.
        sample_rate_hz: See :func:`legacy_t2d_bp_positions`.

    Returns:
        1D float64 array of length ``len(probe_centers_samples) - 1``
        containing absolute inter-probe bp distances. Empty array when
        there are fewer than 2 probes.
    """
    if probe_centers_samples.size < 2:
        return np.empty(0, dtype=np.float64)

    bp = legacy_t2d_bp_positions(
        probe_centers_samples,
        mol=mol,
        mult_const=mult_const,
        addit_const=addit_const,
        alpha=alpha,
        sample_rate_hz=sample_rate_hz,
    )
    return np.abs(np.diff(bp))
