"""Legacy T2D conversion: power-law model for time-to-distance mapping.

The legacy transform converts sample-domain positions to base-pair positions
using the formula:
    L(t) = mult_const * (t_from_tail + addit_const) ^ alpha

where t_from_tail is the number of samples from the molecule's trailing edge
back to the probe position (positive for probes before the tail).
"""

from __future__ import annotations

import numpy as np

from mongoose.data.ground_truth import MoleculeGT, SAMPLE_PERIOD_MS
from mongoose.io.probes_bin import Molecule


def legacy_t2d_intervals(
    mol: Molecule,
    gt: MoleculeGT,
    mult_const: float,
    addit_const: float,
    alpha: float,
) -> np.ndarray:
    """Compute inter-probe intervals using the legacy T2D power-law model.

    For each matched probe, compute its bp position from:
        t_from_tail = tail_sample - probe_sample  (in samples, positive)
        L(t) = mult_const * (t_from_tail + addit_const) ^ alpha

    Then intervals = abs(diff(L_at_probes)).

    Args:
        mol: Molecule from probes.bin (used for fall_t50 trailing edge).
        gt: MoleculeGT containing probe_sample_indices (sorted temporally).
        mult_const: Multiplicative constant from _transForm.txt.
        addit_const: Additive constant from _transForm.txt (can be negative).
        alpha: Exponent from _transForm.txt.

    Returns:
        1D array of inter-probe intervals in bp (length = num_probes - 1).
    """
    # Trailing edge in sample index
    tail_sample = mol.fall_t50 / SAMPLE_PERIOD_MS

    probe_samples = gt.probe_sample_indices  # sorted ascending temporally

    # Time from trailing edge in samples (positive for probes before the tail)
    t_from_tail = tail_sample - probe_samples.astype(np.float64)

    # Apply the power-law transform: L(t) = C * (t + offset)^alpha
    # The argument (t + addit_const) must be positive for the power to work.
    # Clamp to a small positive value to avoid domain errors.
    arg = t_from_tail + addit_const
    arg = np.maximum(arg, 1.0)  # avoid zero/negative in power

    bp_positions = mult_const * np.power(arg, alpha)

    # Inter-probe intervals: absolute difference of consecutive positions
    intervals = np.abs(np.diff(bp_positions))

    return intervals
