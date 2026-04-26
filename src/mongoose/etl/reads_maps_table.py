"""ETL: assemble a per-probe residual table from a Nabsys ``Remapped/AllCh/`` dir.

Direction C trains on the per-probe difference between production's
pre-correction probe positions (``_uncorrected_reads_maps.bin``) and its
post-correction predictions (``_reads_maps.bin``), with the reference
genome positions joined in as ground truth. This module walks one run's
output directory and produces a long-format ``pandas.DataFrame`` with
one row per probe.

Schema (see :func:`build_residual_table`):

* Per-probe features (model inputs): pre_position_bp, width_bp, attribute,
  bit-field accessors, probe-ordering features.
* Molecule features: uid, channel, molecule_length_bp, length_group_bin,
  num_probes, num_probes_accepted, is_clean.
* Reference matching: ref_probe_idx_1based, reference_position_bp
  (genomic, not molecule-relative -- aligning these to molecule-space
  requires stretch_factor / stretch_offset / direction from .assigns
  and is left to consumers), has_reference.
* Targets: post_position_bp (production's prediction in molecule-space),
  residual_bp = post - pre (the production correction signal).

Joining .assigns to maps.bin probes follows the canonical Phase-0
convention (commit ``4799900``): the k-th *accepted* probe in the
maps.bin sequence corresponds to ``assigns.probe_indices[k]``. Excluded
probes (bit 7 unset) have no reference assignment. ``probe_indices[k] == 0``
in .assigns indicates the accepted probe did not match any reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mongoose.io.assigns import MoleculeAssignment, load_assigns
from mongoose.io.reads_maps_bin import (
    MapMolecule,
    PROBE_ATTR_ACCEPTED,
    PROBE_ATTR_IN_CLEAN_REGION,
    PROBE_ATTR_IN_FOLDED_END,
    PROBE_ATTR_IN_FOLDED_START,
    PROBE_ATTR_IN_STRUCTURE,
    PROBE_ATTR_EXCLUDED_AMPLITUDE,
    PROBE_ATTR_EXCLUDED_WIDTH_RM,
    PROBE_ATTR_EXCLUDED_WIDTH_SP,
    PROBE_ATTR_OUTSIDE_PARTIAL,
    load_reads_maps_bin,
)
from mongoose.io.reference_map import ReferenceMap, load_reference_map


# Length-group binning per the head-dive Method 1 paper: 10 kb bins
# starting at 75 kb. Bin index 0 is [75, 85) kb, bin 15 is [225, 235) kb.
# Molecules outside [75, 235) kb get bin -1 (out-of-range).
LENGTH_GROUP_BIN_BP = 10_000
LENGTH_GROUP_MIN_BP = 75_000
LENGTH_GROUP_MAX_BIN = 15  # 16 bins total (0..15)


@dataclass(frozen=True)
class RemapPaths:
    """Paths to the four files we ingest from a ``Remapped/AllCh/`` dir."""

    pre_reads_maps: Path  # _uncorrected_reads_maps.bin
    post_reads_maps: Path  # _reads_maps.bin
    assigns: Path  # _probes.txt_probeassignment.assigns
    reference_map: Path  # _probes.txt_referenceMap.txt


def resolve_remap_paths(remap_dir: Path, run_id: str) -> RemapPaths:
    """Resolve the four standard files in a ``Remapped/AllCh/`` directory.

    Args:
        remap_dir: Path to a ``.../Remapped/AllCh/`` directory.
        run_id: Run identifier (the bit before ``_reads_maps.bin`` etc.).

    Returns:
        Populated :class:`RemapPaths`. No I/O performed; existence is
        validated lazily by :func:`build_residual_table`.
    """
    return RemapPaths(
        pre_reads_maps=remap_dir / f"{run_id}_uncorrected_reads_maps.bin",
        post_reads_maps=remap_dir / f"{run_id}_reads_maps.bin",
        assigns=remap_dir / f"{run_id}_probes.txt_probeassignment.assigns",
        reference_map=remap_dir / f"{run_id}_probes.txt_referenceMap.txt",
    )


def length_group_bin(molecule_length_bp: int) -> int:
    """Map a molecule length to the head-dive Method 1 length-group bin.

    Returns -1 for molecules outside [75 kb, 235 kb).
    """
    if molecule_length_bp < LENGTH_GROUP_MIN_BP:
        return -1
    bin_idx = (molecule_length_bp - LENGTH_GROUP_MIN_BP) // LENGTH_GROUP_BIN_BP
    if bin_idx > LENGTH_GROUP_MAX_BIN:
        return -1
    return int(bin_idx)


def _build_molecule_rows(
    pre_mol: MapMolecule,
    post_mol: MapMolecule,
    assigns_entry: MoleculeAssignment | None,
    refmap: ReferenceMap,
    run_id: str,
) -> list[dict]:
    """Produce one row per probe for a single molecule."""
    if pre_mol.uid != post_mol.uid:
        raise ValueError(
            f"pre/post uid mismatch in {run_id}: pre={pre_mol.uid} post={post_mol.uid}"
        )
    if pre_mol.num_probes != post_mol.num_probes:
        raise ValueError(
            f"pre/post probe-count mismatch for uid={pre_mol.uid} in {run_id}: "
            f"pre={pre_mol.num_probes} post={post_mol.num_probes}"
        )

    n_probes = pre_mol.num_probes
    if n_probes == 0:
        return []

    pre_positions = np.array([p.position_bp for p in pre_mol.probes], dtype=np.int64)
    post_positions = np.array([p.position_bp for p in post_mol.probes], dtype=np.int64)
    widths = np.array([p.width_bp for p in pre_mol.probes], dtype=np.int64)
    attributes = np.array([p.attribute for p in pre_mol.probes], dtype=np.uint32)
    accepted_mask = (attributes & PROBE_ATTR_ACCEPTED) != 0

    # Pre-position adjacency intervals (for context features). For probe
    # 0 we use -1 to indicate "no previous"; for the last probe -1 for
    # "no next". Both are post-clamped to int64.
    prev_interval = np.full(n_probes, -1, dtype=np.int64)
    next_interval = np.full(n_probes, -1, dtype=np.int64)
    if n_probes >= 2:
        prev_interval[1:] = pre_positions[1:] - pre_positions[:-1]
        next_interval[:-1] = pre_positions[1:] - pre_positions[:-1]

    n_accepted = int(accepted_mask.sum())
    bin_idx = length_group_bin(pre_mol.molecule_length_bp)

    direction = int(getattr(assigns_entry, "direction", 0)) if assigns_entry else 0

    # Pre-pass: identify the FIRST accepted probe with a reference match.
    # ``ref_anchored_residual_bp`` is the bp shift that, applied to
    # ``pre_position_bp``, would put the probe at the genome's known
    # position relative to the molecule's first matched probe. We anchor
    # both pre and ref at this first matched probe, so the per-probe
    # target encodes only the SHAPE of the correction along the molecule
    # (not a per-molecule constant offset which production might be
    # arbitrary about). For probes BEFORE the first matched probe (or
    # with no reference at all), ref_anchored_residual_bp = 0.
    first_matched_pre_bp = None
    first_matched_ref_bp = None
    if assigns_entry is not None:
        running_accepted = 0
        for kk in range(n_probes):
            if not bool(accepted_mask[kk]):
                continue
            if running_accepted < len(assigns_entry.probe_indices):
                ref_idx_kk = int(assigns_entry.probe_indices[running_accepted])
                if ref_idx_kk > 0:
                    try:
                        pos_kk, _ = refmap.lookup(ref_idx_kk)
                        first_matched_pre_bp = int(pre_positions[kk])
                        first_matched_ref_bp = int(pos_kk)
                        break
                    except IndexError:
                        pass
            running_accepted += 1

    rows: list[dict] = []
    accepted_idx = 0  # increments only for accepted probes
    for k in range(n_probes):
        attr = int(attributes[k])
        is_accepted = bool(accepted_mask[k])

        ref_probe_idx_1based = 0
        ref_position_bp = -1
        has_reference = False
        if (
            is_accepted
            and assigns_entry is not None
            and accepted_idx < len(assigns_entry.probe_indices)
        ):
            ref_idx = int(assigns_entry.probe_indices[accepted_idx])
            if ref_idx > 0:
                ref_probe_idx_1based = ref_idx
                try:
                    pos, _strand = refmap.lookup(ref_idx)
                    ref_position_bp = int(pos)
                    has_reference = True
                except IndexError:
                    pass
            accepted_idx += 1

        residual_bp = int(post_positions[k] - pre_positions[k])

        # Anchored-to-first-matched ideal-correction target. Equals 0 for
        # the first matched probe (it IS the anchor) and for unmatched
        # probes (no ref to anchor against).
        if (
            has_reference
            and first_matched_pre_bp is not None
            and first_matched_ref_bp is not None
        ):
            pre_anchored = int(pre_positions[k]) - first_matched_pre_bp
            ref_anchored = abs(int(ref_position_bp) - first_matched_ref_bp)
            ref_anchored_residual_bp = int(ref_anchored - pre_anchored)
        else:
            ref_anchored_residual_bp = 0

        frac_pos = float(k) / max(n_probes - 1, 1)
        bp_pos_frac = (
            float(pre_positions[k]) / pre_mol.molecule_length_bp
            if pre_mol.molecule_length_bp > 0
            else 0.0
        )

        rows.append(
            {
                # Identity
                "run_id": run_id,
                "uid": int(pre_mol.uid),
                "channel": int(pre_mol.channel),
                "molecule_id": int(pre_mol.molecule_id),
                "probe_idx": int(k),
                # Probe features (input)
                "pre_position_bp": int(pre_positions[k]),
                "width_bp": int(widths[k]),
                "attribute": attr,
                "accepted": is_accepted,
                "in_clean_region": bool(attr & PROBE_ATTR_IN_CLEAN_REGION),
                "in_structure": bool(attr & PROBE_ATTR_IN_STRUCTURE),
                "in_folded_start": bool(attr & PROBE_ATTR_IN_FOLDED_START),
                "in_folded_end": bool(attr & PROBE_ATTR_IN_FOLDED_END),
                "excluded_amplitude": bool(attr & PROBE_ATTR_EXCLUDED_AMPLITUDE),
                "excluded_width_sp": bool(attr & PROBE_ATTR_EXCLUDED_WIDTH_SP),
                "excluded_width_rm": bool(attr & PROBE_ATTR_EXCLUDED_WIDTH_RM),
                "outside_partial_region": bool(attr & PROBE_ATTR_OUTSIDE_PARTIAL),
                # Position context
                "prev_interval_bp": int(prev_interval[k]),
                "next_interval_bp": int(next_interval[k]),
                "frac_position_in_molecule": frac_pos,
                "bp_position_frac": bp_pos_frac,
                # Molecule features
                "molecule_length_bp": int(pre_mol.molecule_length_bp),
                "num_probes": int(n_probes),
                "num_probes_accepted": int(n_accepted),
                "length_group_bin": bin_idx,
                "is_clean_molecule": bool(pre_mol.is_clean),
                "direction": direction,
                # Reference matching
                "ref_probe_idx_1based": ref_probe_idx_1based,
                "reference_position_bp": ref_position_bp,
                "has_reference": has_reference,
                # Targets
                "post_position_bp": int(post_positions[k]),
                "residual_bp": residual_bp,
                # V4-C-v2: target the GENOME directly (not production's
                # output). Anchoring at first matched probe removes per-
                # molecule constant offsets; the model learns only the
                # SHAPE of the correction along the molecule. Beating
                # production becomes possible if the model captures
                # signal production's hand-tuned curves miss.
                "ref_anchored_residual_bp": ref_anchored_residual_bp,
            }
        )

    return rows


def build_residual_table(remap_dir: Path, run_id: str) -> pd.DataFrame:
    """Assemble a per-probe residual table for one run.

    Args:
        remap_dir: A ``Remapped/AllCh/`` directory containing the four
            standard files (``*_uncorrected_reads_maps.bin``,
            ``*_reads_maps.bin``, ``*_probes.txt_probeassignment.assigns``,
            ``*_probes.txt_referenceMap.txt``).
        run_id: Run identifier prefix used to locate the files.

    Returns:
        A long-format DataFrame: one row per probe across all molecules
        in the run. The schema is documented in the module docstring.
        Empty DataFrame if the run has no molecules.
    """
    paths = resolve_remap_paths(remap_dir, run_id)
    for p in (paths.pre_reads_maps, paths.post_reads_maps, paths.assigns, paths.reference_map):
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    pre_file = load_reads_maps_bin(paths.pre_reads_maps)
    post_file = load_reads_maps_bin(paths.post_reads_maps)
    assigns_list = load_assigns(paths.assigns)
    refmap = load_reference_map(paths.reference_map)

    if pre_file.num_maps != post_file.num_maps:
        raise ValueError(
            f"pre/post num_maps mismatch in {run_id}: "
            f"pre={pre_file.num_maps} post={post_file.num_maps}"
        )

    # Build (fragment_uid -> assignment) lookup. .assigns may include rows
    # for molecules not in maps.bin (e.g., filtered by MF); maps.bin
    # molecules without an .assigns entry get reference_position_bp = -1.
    assigns_by_uid: dict[int, MoleculeAssignment] = {
        a.fragment_uid: a for a in assigns_list
    }

    all_rows: list[dict] = []
    for pre_mol, post_mol in zip(pre_file.molecules, post_file.molecules, strict=True):
        assigns_entry = assigns_by_uid.get(int(pre_mol.uid))
        all_rows.extend(
            _build_molecule_rows(pre_mol, post_mol, assigns_entry, refmap, run_id)
        )

    return pd.DataFrame(all_rows)
