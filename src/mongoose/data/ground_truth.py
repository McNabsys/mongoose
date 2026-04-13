"""Ground truth builder: maps detected probes to reference genome positions.

Produces shift-invariant inter-probe deltas for training the U-Net model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mongoose.io.assigns import MoleculeAssignment
from mongoose.io.probes_bin import Molecule
from mongoose.io.reference_map import ReferenceMap

TAG_WIDTH_BP = 511
SAMPLE_RATE_HZ = 40_000
SAMPLE_PERIOD_MS = 1000.0 / SAMPLE_RATE_HZ  # 0.025 ms


@dataclass
class MoleculeGT:
    """Ground truth for a single molecule."""

    probe_sample_indices: np.ndarray  # int64, temporal sample index of each matched probe
    inter_probe_deltas_bp: np.ndarray  # float64, abs(diff(ref_bp)), always positive
    velocity_targets_bp_per_ms: np.ndarray  # float64, TAG_WIDTH_BP / duration per matched probe
    reference_probe_bp: np.ndarray  # int64, absolute ref bp for each matched probe
    direction: int  # 1 = forward, -1 = reverse


def build_molecule_gt(
    mol: Molecule,
    assign: MoleculeAssignment,
    ref: ReferenceMap,
    min_matched_probes: int = 4,
) -> MoleculeGT | None:
    """Build ground truth for a single molecule from its assignment.

    Args:
        mol: Molecule from probes.bin with detected probe events.
        assign: Assignment mapping this molecule's probes to reference probes.
        ref: Reference map with known probe positions on the genome.
        min_matched_probes: Minimum number of matched probes required.

    Returns:
        MoleculeGT if enough probes matched, None otherwise.
    """
    if assign.ref_index < 0:
        return None

    sample_indices: list[int] = []
    ref_bps: list[int] = []
    velocities: list[float] = []

    for i, ref_probe_idx in enumerate(assign.probe_indices):
        # 0 means unmatched probe
        if ref_probe_idx == 0:
            continue

        # Guard against out-of-range molecule probe index
        if i >= len(mol.probes):
            continue

        probe = mol.probes[i]

        # Skip probes with invalid duration
        if probe.duration_ms <= 0:
            continue

        # Convert 1-based reference probe index to 0-based and look up bp position
        bp = ref.probe_positions[ref_probe_idx - 1]

        sample_idx = int(round(probe.center_ms / SAMPLE_PERIOD_MS))
        velocity = TAG_WIDTH_BP / probe.duration_ms

        sample_indices.append(sample_idx)
        ref_bps.append(bp)
        velocities.append(velocity)

    if len(sample_indices) < min_matched_probes:
        return None

    # Sort all matched data by temporal sample index
    sort_order = np.argsort(sample_indices)
    sample_indices_arr = np.array(sample_indices, dtype=np.int64)[sort_order]
    ref_bps_arr = np.array(ref_bps, dtype=np.int64)[sort_order]
    velocities_arr = np.array(velocities, dtype=np.float64)[sort_order]

    # Shift-invariant inter-probe deltas: abs(diff()) handles both orientations
    deltas = np.abs(np.diff(ref_bps_arr)).astype(np.float64)

    return MoleculeGT(
        probe_sample_indices=sample_indices_arr,
        inter_probe_deltas_bp=deltas,
        velocity_targets_bp_per_ms=velocities_arr,
        reference_probe_bp=ref_bps_arr,
        direction=assign.direction,
    )
