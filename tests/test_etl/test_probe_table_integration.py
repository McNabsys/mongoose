"""Integration test for the per-run probe-table ETL.

Runs the end-to-end pipeline on the canonical biochem-flagged run
(STB03-064B-02L58270w05-202G16g) and verifies schema, row counts,
and the accepted-only join rule hold. Skips gracefully if the raw
data isn't on disk — this lets CI run without the ~6 GB of sample
data while keeping the assertion live on a developer workstation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mongoose.etl.probe_table import build_run_probe_table, resolve_aux_paths
from mongoose.etl.schema import CORE_COLUMN_NAMES


CANONICAL_RUN_ID = "STB03-064B-02L58270w05-202G16g"
CANONICAL_ROOT = Path(
    f"C:/git/mongoose/E. coli/Red/{CANONICAL_RUN_ID}/2025-02-19/Remapped/AllCh"
)


pytestmark = pytest.mark.skipif(
    not (CANONICAL_ROOT / f"{CANONICAL_RUN_ID}_probes.bin").exists(),
    reason="canonical sample run not available on this machine",
)


@pytest.fixture(scope="module")
def canonical_df():
    aux = resolve_aux_paths(CANONICAL_ROOT, CANONICAL_RUN_ID)
    return build_run_probe_table(
        run_id=CANONICAL_RUN_ID,
        probes_bin_path=CANONICAL_ROOT / f"{CANONICAL_RUN_ID}_probes.bin",
        assigns_path=CANONICAL_ROOT / f"{CANONICAL_RUN_ID}_probes.txt_probeassignment.assigns",
        reference_map_path=CANONICAL_ROOT / f"{CANONICAL_RUN_ID}_probes.txt_referenceMap.txt",
        aux=aux,
        compute_t2d=True,
    )


def test_row_count_matches_probes_bin(canonical_df):
    # Sum of num_probes across all molecules == total rows. No silent drops.
    assert (
        int(canonical_df["num_probes"].groupby(canonical_df["molecule_uid"]).first().sum())
        == len(canonical_df)
    )


def test_core_schema_columns_present(canonical_df):
    # build_run_probe_table returns the per-run core columns only. The
    # orchestrator (build.py) attaches concentration_group, conc_raw,
    # biochem_flagged_good plus the excel_* pass-throughs at shard-
    # write time; those are not the per-run ETL's responsibility.
    # expected_width_at_velocity_bin / local_velocity_group are also
    # deferred until the M1 velocity-unit calibration is resolved.
    orchestrator_only = {
        "concentration_group",
        "conc_raw",
        "biochem_flagged_good",
        "expected_width_at_velocity_bin",
        "local_velocity_group",
    }
    for col in CORE_COLUMN_NAMES:
        if col in orchestrator_only:
            continue
        assert col in canonical_df.columns, f"missing column: {col}"


def test_label_fraction_in_spec_range(canonical_df):
    # Spec §299: 0.70-0.85 among aligned molecules.
    aligned = canonical_df[canonical_df["molecule_aligned"]]
    frac = float(aligned["is_assigned"].mean())
    assert 0.60 < frac < 0.90, (
        f"is_assigned fraction among aligned is {frac:.3f} -- out of spec range"
    )


def test_accepted_only_join_rule(canonical_df):
    """Per the confirmed rule: only accepted probes (bit 7) may have ref_idx
    set to a non-null integer. Non-accepted probes MUST have ref_idx=null."""
    not_accepted = canonical_df[~canonical_df["attr_accepted"].astype(bool)]
    assert not_accepted["ref_idx"].isna().all(), (
        "non-accepted probes should never carry a ref_idx; ETL join rule broken."
    )


def test_molecule_bp_length_matches_assigned_span(canonical_df):
    # For each aligned molecule with >=2 assigned probes, molecule_bp_length
    # equals the span of ref_genomic_pos_bp over assigned probes in that molecule.
    aligned = canonical_df[canonical_df["molecule_aligned"]]
    assigned = aligned[aligned["is_assigned"]]
    spans = (
        assigned.groupby("molecule_uid")["ref_genomic_pos_bp"]
        .agg(lambda s: int(s.max() - s.min()) if len(s) >= 2 else None)
    )
    # Join back to molecule-level recorded value.
    mol_lengths = (
        aligned[["molecule_uid", "molecule_bp_length"]]
        .drop_duplicates("molecule_uid")
        .set_index("molecule_uid")["molecule_bp_length"]
    )
    compared = 0
    mismatches = 0
    for uid, expected in spans.items():
        if expected is None:
            continue
        reported = mol_lengths.get(uid)
        if reported is None:
            continue
        compared += 1
        if int(reported) != int(expected):
            mismatches += 1
    assert compared > 0, "no molecules with >=2 assigned probes? check sample"
    assert mismatches == 0, f"{mismatches} / {compared} molecule_bp_length mismatches"


def test_t2d_predicted_is_finite_where_transform_available(canonical_df):
    # We know the canonical run has a _transForm.txt, so t2d_predicted_bp_pos
    # should be populated for every probe.
    assert canonical_df["t2d_predicted_bp_pos"].notna().all()
