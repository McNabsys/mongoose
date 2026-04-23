"""Pyarrow schema for the unified probe table.

Single source of truth for column names and types — referenced by
:mod:`mongoose.etl.probe_table` (builder) and the integration tests.

Columns fall into four groups:

1. **Identity** — per-probe unique key + join keys.
2. **Probe features** — from probes.bin Table 4 (per-probe block).
3. **Molecule features** — from probes.bin Table 3, broadcast to each probe.
4. **Labels** — from .assigns + referenceMap; nullable where molecule
   is unaligned or probe is unassigned.
5. **Derived** — computed at ETL time (bitfield unpack, width_ratio,
   T2D prediction, etc.).
6. **Remap settings** — selected keys from _remapSettings.txt, broadcast
   per-run.
7. **Version info** — _version.txt, broadcast per-run.

Per-run Excel metadata (`concentration_group`, `biochem_flagged_good`,
and the pass-through Excel columns) are **not** declared in this schema.
They are attached at build time with pandas-inferred dtypes and merged
into the output parquet; the core schema here is the enforced subset.
"""

from __future__ import annotations

import pyarrow as pa


# Attribute bits (per spec §74-76); order matches the order listed in the spec.
ATTRIBUTE_BIT_FIELDS: tuple[tuple[str, int], ...] = (
    ("attr_clean_region", 0),
    ("attr_folded_end", 1),
    ("attr_folded_start", 2),
    ("attr_in_structure", 3),
    ("attr_excl_amp_high", 4),
    ("attr_excl_width_sp", 5),
    ("attr_excl_width_remap", 6),
    ("attr_accepted", 7),
    ("attr_excl_outside_partial", 8),
)


CORE_SCHEMA = pa.schema([
    # -- Identity --
    pa.field("probe_uid", pa.string()),  # "{run_id}:{molecule_uid}:{probe_idx_in_molecule}"
    pa.field("run_id", pa.string()),     # categorical at write time
    pa.field("molecule_uid", pa.uint32()),
    pa.field("probe_idx_in_molecule", pa.uint16()),

    # -- Probe features (probes.bin Table 4) --
    pa.field("start_ms", pa.float32()),
    pa.field("duration_ms", pa.float32()),
    pa.field("center_ms", pa.float32()),
    pa.field("area_samples_uv", pa.float32()),
    pa.field("max_amp_uv", pa.float32()),
    pa.field("attr_bitfield_raw", pa.uint32()),
    pa.field("attr_clean_region", pa.bool_()),
    pa.field("attr_folded_end", pa.bool_()),
    pa.field("attr_folded_start", pa.bool_()),
    pa.field("attr_in_structure", pa.bool_()),
    pa.field("attr_excl_amp_high", pa.bool_()),
    pa.field("attr_excl_width_sp", pa.bool_()),
    pa.field("attr_excl_width_remap", pa.bool_()),
    pa.field("attr_accepted", pa.bool_()),
    pa.field("attr_excl_outside_partial", pa.bool_()),

    # -- Molecule features (probes.bin Table 3, broadcast) --
    pa.field("molecule_id", pa.uint32()),
    pa.field("detector_channel", pa.int32()),
    pa.field("file_name_index", pa.uint32()),
    pa.field("molecule_start_ms", pa.float64()),
    pa.field("translocation_time_ms", pa.float32()),
    pa.field("use_partial_time_ms", pa.float32()),
    pa.field("mean_lvl1_mv", pa.float32()),
    pa.field("rise_time_t10_ms", pa.float32()),
    pa.field("rise_time_t50_ms", pa.float32()),
    pa.field("rise_time_t90_ms", pa.float32()),
    pa.field("fall_time_t90_ms", pa.float32()),
    pa.field("fall_time_t50_ms", pa.float32()),
    pa.field("fall_time_t10_ms", pa.float32()),
    pa.field("folded_start_end_ms", pa.float32()),
    pa.field("folded_end_start_ms", pa.float32()),
    pa.field("why_structured_bitfield", pa.uint32()),
    pa.field("num_probes", pa.uint32()),
    pa.field("num_structures", pa.uint32()),
    pa.field("num_recovered_structures", pa.uint32()),
    pa.field("molecule_structured", pa.bool_()),
    pa.field("molecule_use_partial", pa.bool_()),
    pa.field("molecule_folded_start", pa.bool_()),
    pa.field("molecule_folded_end", pa.bool_()),
    pa.field("molecule_do_not_use", pa.bool_()),
    # Needed for T2D math and downstream coordinate conversions.
    pa.field("molecule_start_within_tdb_ms", pa.float32()),

    # -- Labels (assigns + referenceMap; nullable where unaligned/unassigned) --
    pa.field("ref_idx", pa.int32()),
    pa.field("ref_genomic_pos_bp", pa.int64()),
    pa.field("ref_strand", pa.int8()),
    pa.field("is_assigned", pa.bool_()),
    pa.field("molecule_aligned", pa.bool_()),
    pa.field("molecule_refindex", pa.int32()),
    pa.field("molecule_align_score", pa.int32()),
    pa.field("molecule_second_best_score", pa.int32()),
    pa.field("molecule_stretch_factor", pa.float32()),
    pa.field("molecule_stretch_offset", pa.float32()),
    pa.field("molecule_direction", pa.int8()),
    pa.field("molecule_weight", pa.float32()),
    pa.field("molecule_bp_length", pa.int64()),

    # -- Derived --
    pa.field("local_velocity_group", pa.int16()),
    pa.field("expected_width_at_velocity_bin", pa.int16()),
    pa.field("width_ratio", pa.float32()),
    pa.field("probe_local_density", pa.uint16()),
    pa.field("prev_probe_gap_ms", pa.float32()),
    pa.field("next_probe_gap_ms", pa.float32()),
    pa.field("t2d_predicted_bp_pos", pa.float64()),

    # -- Per-run remap settings (broadcast). All nullable — older runs
    #    may lack a given key, and the settings parser returns None in
    #    that case rather than raising. --
    pa.field("tag_size_bp", pa.int32()),
    pa.field("tvc_algorithm", pa.int8()),
    pa.field("tag_velocity_multiplier", pa.float32()),
    pa.field("settings_alpha", pa.float32()),
    pa.field("settings_mult_const", pa.float32()),
    pa.field("settings_addit_const", pa.float32()),
    pa.field("use_probe_width_filter", pa.bool_()),
    pa.field("expected_min_probe_width_factor", pa.float32()),
    pa.field("reject_width_too_low_probes", pa.bool_()),
    pa.field("false_neg_assumed", pa.float32()),
    pa.field("false_pos_assumed", pa.float32()),
    pa.field("align_score_threshold", pa.float32()),

    # -- Version --
    pa.field("program_version", pa.string()),
    pa.field("picker_version", pa.string()),

    # -- Per-run Excel (small canonical subset; every other Excel column
    #    is attached dynamically at write time by probe_table). --
    pa.field("concentration_group", pa.string()),
    pa.field("conc_raw", pa.string()),
    pa.field("biochem_flagged_good", pa.bool_()),
])


CORE_COLUMN_NAMES: tuple[str, ...] = tuple(CORE_SCHEMA.names)


SCHEMA_VERSION = "1"  # bump when the column set or dtypes change in a breaking way
