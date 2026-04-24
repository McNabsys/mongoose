"""Per-run ETL: raw probes.bin / assigns / referenceMap / settings → DataFrame.

Outputs a pandas DataFrame with one row per detected probe. The builder
does NOT write parquet — that's the orchestrator's job (see build.py) so
a single run can be sanity-checked in isolation without touching disk.

The schema produced here matches :mod:`mongoose.etl.schema.CORE_SCHEMA`
for the core columns, with per-run Excel metadata attached as extra
columns at orchestrator time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mongoose.etl.derived import (
    compute_local_density,
    compute_molecule_bp_length,
    compute_molecule_velocity_bp_per_ms,
    compute_probe_gaps,
    compute_t2d_predicted_bp,
    unpack_attribute_bitfield,
)
from mongoose.etl.probe_widths import ProbeWidths, load_probe_widths, wide_peak_bin
from mongoose.etl.remap_settings import RemapSettings, load_remap_settings
from mongoose.etl.version_file import VersionInfo, load_version_file
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import Molecule, load_probes_bin
from mongoose.io.reference_map import ReferenceMap, load_reference_map
from mongoose.io.transform import ChannelTransform, load_transforms


DEFAULT_LOCAL_DENSITY_WINDOW_MS: float = 50.0


@dataclass(frozen=True)
class RunAuxPaths:
    """Optional sidecar files resolved alongside the core probes/assigns/refmap trio.

    All are nullable — the ETL records which were found in the per-run
    manifest entry and skips the derived columns that depend on a missing
    sidecar (e.g., t2d_predicted_bp_pos is null if _transForm.txt is
    missing).
    """

    transform_path: Path | None = None
    remap_settings_path: Path | None = None
    version_path: Path | None = None
    probe_widths_path: Path | None = None


def resolve_aux_paths(remap_allch_dir: Path, run_id: str) -> RunAuxPaths:
    """Locate optional sidecars within the Remapped/AllCh/ directory."""
    remap_allch_dir = Path(remap_allch_dir)
    candidates = {
        "transform_path": remap_allch_dir / f"{run_id}_transForm.txt",
        "remap_settings_path": remap_allch_dir / f"{run_id}.txt_remapSettings.txt",
        "version_path": remap_allch_dir / f"{run_id}.txt_version.txt",
        "probe_widths_path": remap_allch_dir / "M1_probeWidths.txt",
    }
    kwargs = {name: (p if p.exists() else None) for name, p in candidates.items()}
    return RunAuxPaths(**kwargs)


def build_run_probe_table(
    *,
    run_id: str,
    probes_bin_path: Path,
    assigns_path: Path,
    reference_map_path: Path,
    aux: RunAuxPaths,
    compute_t2d: bool = True,
    local_density_window_ms: float = DEFAULT_LOCAL_DENSITY_WINDOW_MS,
) -> pd.DataFrame:
    """Build the probe-table DataFrame for a single run.

    Args:
        run_id: Run identifier (used for ``probe_uid`` construction).
        probes_bin_path: Raw ``*_probes.bin`` path.
        assigns_path: ``*_probeassignment.assigns`` path. Canonical
            variant — not ``.subset.assigns`` or ``.tvcsubset.assigns``.
        reference_map_path: ``*_referenceMap.txt`` path.
        aux: Paths to optional sidecars (transform, settings, version,
            probe widths). Null when absent — see :func:`resolve_aux_paths`.
        compute_t2d: When True (default) and the transform sidecar is
            present, compute ``t2d_predicted_bp_pos`` per probe. When
            False (or transform missing), leave null.
        local_density_window_ms: Half-window for ``probe_local_density``;
            default 50 ms per spec §143.

    Returns:
        DataFrame with one row per detected probe, core columns only.
        No per-run Excel columns (orchestrator attaches those).
    """
    pbin = load_probes_bin(probes_bin_path)
    assigns = load_assigns(assigns_path)
    refmap = load_reference_map(reference_map_path)

    transforms: dict[str, ChannelTransform] = {}
    if compute_t2d and aux.transform_path is not None:
        transforms = load_transforms(aux.transform_path)

    settings: RemapSettings | None = (
        load_remap_settings(aux.remap_settings_path)
        if aux.remap_settings_path is not None
        else None
    )
    version: VersionInfo | None = (
        load_version_file(aux.version_path) if aux.version_path is not None else None
    )
    probe_widths: ProbeWidths | None = (
        load_probe_widths(aux.probe_widths_path)
        if aux.probe_widths_path is not None
        else None
    )

    # Pre-compute wide-peak bin per velocity group for O(1) probe-time lookup.
    wide_peak_by_group: dict[int, int] = (
        {g.group: wide_peak_bin(g.counts) for g in probe_widths.groups}
        if probe_widths is not None
        else {}
    )

    assign_by_uid = {a.fragment_uid: a for a in assigns}

    settings_broadcast = _run_settings_fields(settings)
    version_broadcast = _run_version_fields(version)

    rows: list[dict[str, Any]] = []
    for mol in pbin.molecules:
        _extend_rows_for_molecule(
            rows=rows,
            mol=mol,
            run_id=run_id,
            assignment=assign_by_uid.get(mol.uid),
            refmap=refmap,
            transforms=transforms,
            compute_t2d=compute_t2d,
            wide_peak_by_group=wide_peak_by_group,
            probe_widths=probe_widths,
            local_density_window_ms=local_density_window_ms,
            settings_broadcast=settings_broadcast,
            version_broadcast=version_broadcast,
        )

    return pd.DataFrame(rows)


def _extend_rows_for_molecule(
    *,
    rows: list[dict[str, Any]],
    mol: Molecule,
    run_id: str,
    assignment,
    refmap: ReferenceMap,
    transforms: dict[str, ChannelTransform],
    compute_t2d: bool,
    wide_peak_by_group: dict[int, int],
    probe_widths: ProbeWidths | None,
    local_density_window_ms: float,
    settings_broadcast: dict[str, Any],
    version_broadcast: dict[str, Any],
) -> None:
    n_probes = len(mol.probes)
    if n_probes == 0:
        return

    centers_ms = np.array([p.center_ms for p in mol.probes], dtype=np.float64)

    # probes.bin records every detected probe (all attribute bits preserved),
    # but the Nabsys aligner only consumes *accepted* probes (attribute bit
    # 7 set). So the k-th entry in assigns.probe_indices maps to the k-th
    # accepted probe in probes.bin detection order — not the k-th probe
    # overall. Empirically this interpretation matches 76% of aligned
    # molecules exactly; the remaining 24% have len(probe_indices) strictly
    # shorter than the accepted count, consistent with an additional
    # downstream filter (TVC / width gate) that drops a tail of accepted
    # probes before they reach the aligner. Confirmed with Jon 2026-04-23.
    probe_indices_from_assigns: tuple[int, ...] = (
        assignment.probe_indices if assignment is not None else ()
    )
    molecule_aligned = assignment is not None and assignment.ref_index >= 0

    accepted_positions = [
        i for i, p in enumerate(mol.probes) if (p.attribute >> 7) & 1
    ]

    # --- Labels per-probe (assigned bp + strand) ---
    per_probe_ref_idx: list[int | None] = [None] * n_probes
    per_probe_bp: list[int | None] = [None] * n_probes
    per_probe_strand: list[int | None] = [None] * n_probes

    if molecule_aligned and probe_indices_from_assigns:
        n_assigns = len(probe_indices_from_assigns)
        n_accepted = len(accepted_positions)
        # Under the confirmed interpretation, n_assigns <= n_accepted.
        # n_assigns > n_accepted would violate the rule; surface as an
        # error rather than silently truncating.
        if n_assigns > n_accepted:
            raise ValueError(
                f"molecule uid={mol.uid}: assigns has {n_assigns} ProbeK "
                f"values but only {n_accepted} accepted probes in "
                f"probes.bin. Join rule (k-th ProbeK -> k-th accepted "
                f"probe) requires n_assigns <= n_accepted."
            )
        for k, ref_idx_1_based in enumerate(probe_indices_from_assigns):
            probe_pos = accepted_positions[k]
            if ref_idx_1_based > 0:
                bp, strand = refmap.lookup(ref_idx_1_based)
                per_probe_ref_idx[probe_pos] = ref_idx_1_based
                per_probe_bp[probe_pos] = bp
                per_probe_strand[probe_pos] = strand
            else:
                # ref_idx=0 -- accepted probe, detector-found but not
                # assigned to any reference site (real unmatched case).
                per_probe_ref_idx[probe_pos] = 0
        # Accepted probes beyond n_assigns and all non-accepted probes
        # keep per_probe_ref_idx[i] = None (left unset above).

    # --- Derived: molecule-level ---
    molecule_bp_length = compute_molecule_bp_length(per_probe_bp)
    molecule_velocity_bp_per_ms = compute_molecule_velocity_bp_per_ms(
        molecule_bp_length, mol.transloc_time_ms
    )
    local_velocity_group: int | None = None
    expected_width_bin: int | None = None
    # local_velocity_group: resolution blocked on M1 velocity-unit
    # calibration (see probe_widths.py). Left null.

    # --- Derived: per-probe gaps + density ---
    # probes.bin stores probes in detection order; compute gaps against
    # this order directly. No re-sort needed — spec §275 asks us to
    # verify this ordering matches assigns, and that check is the sanity
    # report's job.
    prev_gap, next_gap = compute_probe_gaps(centers_ms)
    density = compute_local_density(centers_ms, window_ms=local_density_window_ms)

    # --- Derived: T2D prediction (per-channel transform lookup) ---
    if compute_t2d:
        channel_key = f"Ch{mol.channel:03d}"
        t2d_params = transforms.get(channel_key)
        if t2d_params is not None:
            t2d_bp = compute_t2d_predicted_bp(
                centers_ms,
                mol=mol,
                mult_const=t2d_params.mult_const,
                addit_const=t2d_params.addit_const,
                alpha=t2d_params.alpha,
            )
        else:
            t2d_bp = np.full(n_probes, np.nan, dtype=np.float64)
    else:
        t2d_bp = np.full(n_probes, np.nan, dtype=np.float64)

    # --- Labels: molecule-level ---
    if assignment is None:
        mol_aligned = False
        mol_refidx = -1
        mol_align_score = 0
        mol_second_best = 0
        mol_stretch_factor = np.float32(np.nan)
        mol_stretch_offset = np.float32(np.nan)
        mol_direction = 0
        mol_weight = np.float32(np.nan)
    else:
        mol_aligned = assignment.ref_index >= 0
        mol_refidx = assignment.ref_index
        mol_align_score = assignment.alignment_score
        mol_second_best = assignment.second_best_score
        mol_stretch_factor = np.float32(assignment.stretch_factor)
        mol_stretch_offset = np.float32(assignment.stretch_offset)
        mol_direction = assignment.direction
        mol_weight = np.float32(assignment.weight)

    mol_broadcast = {
        "run_id": run_id,
        "molecule_uid": np.uint32(mol.uid),
        "molecule_id": np.uint32(mol.molecule_id),
        "detector_channel": np.int32(mol.channel),
        "file_name_index": np.uint32(mol.file_name_index),
        "molecule_start_ms": float(mol.start_ms),
        "translocation_time_ms": np.float32(mol.transloc_time_ms),
        "use_partial_time_ms": np.float32(mol.use_partial_time_ms),
        "mean_lvl1_mv": np.float32(mol.mean_lvl1),
        "rise_time_t10_ms": np.float32(mol.rise_t10),
        "rise_time_t50_ms": np.float32(mol.rise_t50),
        "rise_time_t90_ms": np.float32(mol.rise_t90),
        "fall_time_t90_ms": np.float32(mol.fall_t90),
        "fall_time_t50_ms": np.float32(mol.fall_t50),
        "fall_time_t10_ms": np.float32(mol.fall_t10),
        "folded_start_end_ms": np.float32(mol.folded_start_end),
        "folded_end_start_ms": np.float32(mol.folded_end_start),
        "why_structured_bitfield": np.uint32(mol.why_structured),
        "num_probes": np.uint32(mol.num_probes),
        "num_structures": np.uint32(mol.num_structures),
        "num_recovered_structures": np.uint32(mol.num_recovered_structures),
        "molecule_structured": bool(mol.structured),
        "molecule_use_partial": bool(mol.use_partial),
        "molecule_folded_start": bool(mol.folded_start),
        "molecule_folded_end": bool(mol.folded_end),
        "molecule_do_not_use": bool(mol.do_not_use),
        "molecule_start_within_tdb_ms": np.float32(mol.start_within_tdb_ms),
        "molecule_aligned": bool(mol_aligned),
        "molecule_refindex": np.int32(mol_refidx),
        "molecule_align_score": np.int32(mol_align_score),
        "molecule_second_best_score": np.int32(mol_second_best),
        "molecule_stretch_factor": mol_stretch_factor,
        "molecule_stretch_offset": mol_stretch_offset,
        "molecule_direction": np.int8(mol_direction),
        "molecule_weight": mol_weight,
        "molecule_bp_length": (
            np.int64(molecule_bp_length) if molecule_bp_length is not None else pd.NA
        ),
        "molecule_velocity_bp_per_ms": (
            np.float32(molecule_velocity_bp_per_ms)
            if molecule_velocity_bp_per_ms is not None
            else np.float32(np.nan)
        ),
        "local_velocity_group": (
            np.int16(local_velocity_group) if local_velocity_group is not None else pd.NA
        ),
        "expected_width_at_velocity_bin": (
            np.int16(expected_width_bin) if expected_width_bin is not None else pd.NA
        ),
        **settings_broadcast,
        **version_broadcast,
    }

    for i, probe in enumerate(mol.probes):
        bits = unpack_attribute_bitfield(probe.attribute)
        ref_idx = per_probe_ref_idx[i]
        bp_pos = per_probe_bp[i]
        strand = per_probe_strand[i]
        row: dict[str, Any] = {
            "probe_uid": f"{run_id}:{mol.uid}:{i}",
            "probe_idx_in_molecule": np.uint16(i),
            "start_ms": np.float32(probe.start_ms),
            "duration_ms": np.float32(probe.duration_ms),
            "center_ms": np.float32(probe.center_ms),
            "area_samples_uv": np.float32(probe.area),
            "max_amp_uv": np.float32(probe.max_amplitude),
            "attr_bitfield_raw": np.uint32(probe.attribute),
            **bits,
            # labels
            "ref_idx": np.int32(ref_idx) if ref_idx is not None else pd.NA,
            "ref_genomic_pos_bp": np.int64(bp_pos) if bp_pos is not None else pd.NA,
            "ref_strand": np.int8(strand) if strand is not None else pd.NA,
            "is_assigned": bool(ref_idx is not None and ref_idx > 0),
            # derived per-probe
            "prev_probe_gap_ms": np.float32(prev_gap[i]),
            "next_probe_gap_ms": np.float32(next_gap[i]),
            "probe_local_density": np.uint16(density[i]),
            "width_ratio": np.float32(np.nan),  # needs bin→ms calibration
            "t2d_predicted_bp_pos": float(t2d_bp[i]) if not np.isnan(t2d_bp[i]) else pd.NA,
            **mol_broadcast,
        }
        rows.append(row)


def _run_settings_fields(settings: RemapSettings | None) -> dict[str, Any]:
    """Canonical broadcast columns from _remapSettings.txt. Null when missing."""
    if settings is None:
        return {
            "tag_size_bp": pd.NA,
            "tvc_algorithm": pd.NA,
            "tag_velocity_multiplier": np.float32(np.nan),
            "settings_alpha": np.float32(np.nan),
            "settings_mult_const": np.float32(np.nan),
            "settings_addit_const": np.float32(np.nan),
            "use_probe_width_filter": pd.NA,
            "expected_min_probe_width_factor": np.float32(np.nan),
            "reject_width_too_low_probes": pd.NA,
            "false_neg_assumed": np.float32(np.nan),
            "false_pos_assumed": np.float32(np.nan),
            "align_score_threshold": np.float32(np.nan),
        }
    return {
        "tag_size_bp": _opt_int32(settings.get_int("TagSize")),
        "tvc_algorithm": _opt_int8(settings.get_int("ttnorm_stretch_algorithm")),
        "tag_velocity_multiplier": _opt_float32(settings.get_float("TagVelocityMultiplier")),
        "settings_alpha": _opt_float32(settings.get_float("ALPHA")),
        "settings_mult_const": _opt_float32(settings.get_float("MultConst")),
        "settings_addit_const": _opt_float32(settings.get_float("additConst")),
        "use_probe_width_filter": _opt_bool(settings.get_bool("use_probe_width_filter")),
        "expected_min_probe_width_factor": _opt_float32(
            settings.get_float("expected_min_probe_width_factor")
        ),
        "reject_width_too_low_probes": _opt_bool(
            settings.get_bool("reject_width_too_low_probes")
        ),
        "false_neg_assumed": _opt_float32(settings.get_float("falseNeg")),
        "false_pos_assumed": _opt_float32(settings.get_float("falsePos")),
        "align_score_threshold": _opt_float32(settings.get_float("alignScore")),
    }


def _run_version_fields(version: VersionInfo | None) -> dict[str, Any]:
    if version is None:
        return {"program_version": pd.NA, "picker_version": pd.NA}
    return {
        "program_version": version.program_version or pd.NA,
        "picker_version": version.picker or pd.NA,
    }


def _opt_float32(v: float | None) -> np.float32:
    return np.float32(v) if v is not None else np.float32(np.nan)


def _opt_int32(v: int | None):
    return np.int32(v) if v is not None else pd.NA


def _opt_int8(v: int | None):
    return np.int8(v) if v is not None else pd.NA


def _opt_bool(v: bool | None):
    return bool(v) if v is not None else pd.NA
