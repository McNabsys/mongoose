"""Ground truth builder: maps matched probes to reference genome positions.

V1 rearchitecture (R4):

The primary ground truth is now ``reference_bp_positions`` -- the ordered
list of reference basepair coordinates derived from the aligner's
``probe_indices``. This drops the wfmproc-derived ``probe_sample_indices``,
``inter_probe_deltas_bp`` and ``velocity_targets_bp_per_ms`` as primary
training signal; those quantities are now derived during training from
detected peaks and heatmap widths instead of at preprocessing time.

The wfmproc probe centers and durations are still retained, optionally,
as ``warmstart_probe_*`` arrays. They are consumed during the first 3-5
epochs of training to warmstart ``L_probe`` while the model bootstraps
peak detection.

Direction convention used by this module:

    direction == 1  (forward): temporal order corresponds to ASCENDING bp.
    direction == -1 (reverse): temporal order corresponds to DESCENDING bp
                               (the molecule entered trailing-end first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mongoose.io.assigns import MoleculeAssignment
from mongoose.io.probes_bin import Molecule
from mongoose.io.reference_map import ReferenceMap

TAG_WIDTH_BP = 511


@dataclass
class MoleculeGT:
    """Ground truth for a single molecule (V1 rearchitecture schema).

    Attributes:
        reference_bp_positions: int64 array of reference bp coordinates,
            sorted in temporal order given ``direction``. Primary GT.
        n_ref_probes: len(reference_bp_positions); used as target for the
            L_count regression head.
        direction: 1 = forward (ascending bp in temporal order),
                   -1 = reverse (descending bp in temporal order).
        warmstart_probe_centers_samples: int64 array of wfmproc probe
            center sample indices, one per matched probe (aligned to
            ``reference_bp_positions``). Optional: None when warmstart
            labels are not requested or unavailable.
        warmstart_probe_durations_samples: float32 array of wfmproc probe
            durations in samples. Optional, same pairing as centers.
    """

    reference_bp_positions: np.ndarray
    n_ref_probes: int
    direction: int
    warmstart_probe_centers_samples: np.ndarray | None = None
    warmstart_probe_durations_samples: np.ndarray | None = None


def build_molecule_gt(
    mol: Molecule,
    assign: MoleculeAssignment,
    ref: ReferenceMap,
    *,
    sample_rate_hz: int,
    min_matched_probes: int = 4,
    include_warmstart: bool = True,
) -> MoleculeGT | None:
    """Build V1 ground truth for a single molecule from its assignment.

    Args:
        mol: Molecule from probes.bin with detected probe events.
        assign: Assignment mapping this molecule's probes to reference probes.
        ref: Reference map with known probe positions on the genome.
        sample_rate_hz: TDB sample rate in Hz (read from ``TdbHeader.sample_rate``).
            Used to convert ``probe.center_ms`` to sample indices. Required and
            keyword-only — hardcoding this previously caused a silent label
            mis-alignment bug.
        min_matched_probes: Minimum number of matched probes required.
            Keyword-only.
        include_warmstart: If True, populate the warmstart_* fields using
            wfmproc probe centers and durations (only for probes with
            duration > 0). Keyword-only.

    Returns:
        MoleculeGT if enough probes matched, None otherwise.
    """
    if assign.ref_index < 0:
        return None

    # Collect (molecule_probe_idx, reference_bp) pairs for all matched probes.
    # Preserve encounter order so we can deduplicate consecutive duplicates
    # while keeping temporal pairing for the warmstart arrays.
    matched: list[tuple[int, int]] = []  # (molecule probe index, ref bp)
    for i, ref_probe_idx in enumerate(assign.probe_indices):
        # 0 indicates an unmatched probe slot in the aligner output.
        if ref_probe_idx == 0:
            continue

        # Convert 1-based reference probe index to 0-based ref-map lookup.
        bp = int(ref.probe_positions[ref_probe_idx - 1])
        matched.append((i, bp))

    if not matched:
        return None

    # Deduplicate reference bp positions while preserving order. For each
    # unique bp we keep the first occurrence (and its associated molecule
    # probe index) so the warmstart arrays remain paired 1:1 with the
    # reference bp positions.
    seen_bps: set[int] = set()
    unique_matched: list[tuple[int, int]] = []
    for mol_probe_idx, bp in matched:
        if bp in seen_bps:
            continue
        seen_bps.add(bp)
        unique_matched.append((mol_probe_idx, bp))

    if len(unique_matched) < min_matched_probes:
        return None

    # Sort by reference bp -- direction controls ascending vs descending.
    # Forward (direction == 1): temporal order == ascending bp.
    # Reverse (direction == -1): temporal order == descending bp.
    reverse = assign.direction != 1
    unique_matched.sort(key=lambda pair: pair[1], reverse=reverse)

    reference_bp_positions = np.array(
        [bp for _, bp in unique_matched], dtype=np.int64
    )

    warmstart_centers: np.ndarray | None = None
    warmstart_durations: np.ndarray | None = None

    if include_warmstart:
        # probe.center_ms is measured from the molecule's translocation start
        # (probes.bin stores ``start_within_tdb_ms`` as the molecule-start
        # offset within the TDB block's waveform). The sample index in the
        # cached waveform is therefore:
        #     (start_within_tdb_ms + probe.center_ms) * sample_rate_hz / 1000
        #
        # Schema invariant (V4): the centers/durations arrays are emitted
        # PAIRED 1:1 with ``reference_bp_positions``. Probes with
        # ``duration_ms <= 0`` (or where the .assigns slot is out of range)
        # contribute SENTINELS instead of being dropped:
        #   * center_sample = -1
        #   * duration_samples = 0.0
        # Consumers (build_probe_heatmap, NoiseModelLoss) must filter the
        # sentinels before use. The pre-V4 schema dropped invalid probes,
        # which gave warmstart_centers a SUBSET of reference_bp_positions
        # and broke 1:1 pairing in the noise-model NLL.
        sample_period_ms = 1000.0 / sample_rate_hz
        mol_start_ms = float(mol.start_within_tdb_ms)
        centers: list[int] = []
        durations: list[float] = []
        for mol_probe_idx, _bp in unique_matched:
            if mol_probe_idx >= len(mol.probes):
                centers.append(-1)
                durations.append(0.0)
                continue
            probe = mol.probes[mol_probe_idx]
            if probe.duration_ms <= 0:
                centers.append(-1)
                durations.append(0.0)
                continue
            centers.append(
                int(round((mol_start_ms + probe.center_ms) / sample_period_ms))
            )
            durations.append(float(probe.duration_ms / sample_period_ms))

        # The arrays now have the same length as reference_bp_positions
        # by construction; emit them whenever any centers were generated
        # (even if all are sentinels -- that's a "no warmstart" molecule
        # but the schema still holds).
        warmstart_centers = np.array(centers, dtype=np.int64)
        warmstart_durations = np.array(durations, dtype=np.float32)

    return MoleculeGT(
        reference_bp_positions=reference_bp_positions,
        n_ref_probes=len(reference_bp_positions),
        direction=int(assign.direction),
        warmstart_probe_centers_samples=warmstart_centers,
        warmstart_probe_durations_samples=warmstart_durations,
    )
