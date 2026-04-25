"""Tests for the per-probe residual ETL (Direction C, phase C.2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mongoose.etl.reads_maps_table import (
    LENGTH_GROUP_BIN_BP,
    LENGTH_GROUP_MAX_BIN,
    LENGTH_GROUP_MIN_BP,
    build_residual_table,
    length_group_bin,
    resolve_remap_paths,
)


# Optional fixture: gitignored under support/. When it's not present
# (e.g., on CI) the data-dependent tests skip.
EXAMPLE_DIR = Path("support/example-remap/STB03-064B-02L58270w05-202G16g/AllCh")
EXAMPLE_RUN_ID = "STB03-064B-02L58270w05-202G16g"
EXAMPLE_PRE = EXAMPLE_DIR / f"{EXAMPLE_RUN_ID}_uncorrected_reads_maps.bin"
EXAMPLE_ASSIGNS = EXAMPLE_DIR / f"{EXAMPLE_RUN_ID}_probes.txt_probeassignment.assigns"
EXAMPLE_REFMAP = EXAMPLE_DIR / f"{EXAMPLE_RUN_ID}_probes.txt_referenceMap.txt"
EXAMPLE_AVAILABLE = (
    EXAMPLE_PRE.exists() and EXAMPLE_ASSIGNS.exists() and EXAMPLE_REFMAP.exists()
)


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------

def test_resolve_remap_paths_constructs_expected_filenames(tmp_path):
    paths = resolve_remap_paths(tmp_path, "MYRUN")
    assert paths.pre_reads_maps == tmp_path / "MYRUN_uncorrected_reads_maps.bin"
    assert paths.post_reads_maps == tmp_path / "MYRUN_reads_maps.bin"
    assert paths.assigns == tmp_path / "MYRUN_probes.txt_probeassignment.assigns"
    assert paths.reference_map == tmp_path / "MYRUN_probes.txt_referenceMap.txt"


def test_build_table_raises_when_inputs_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required input missing"):
        build_residual_table(tmp_path, "NOPE")


# --------------------------------------------------------------------------
# Length-group binning
# --------------------------------------------------------------------------

def test_length_group_bin_below_min_returns_minus_one():
    assert length_group_bin(50_000) == -1
    assert length_group_bin(LENGTH_GROUP_MIN_BP - 1) == -1


def test_length_group_bin_at_min_returns_zero():
    assert length_group_bin(LENGTH_GROUP_MIN_BP) == 0


def test_length_group_bin_in_first_bin():
    assert length_group_bin(LENGTH_GROUP_MIN_BP + LENGTH_GROUP_BIN_BP - 1) == 0


def test_length_group_bin_at_second_boundary():
    assert length_group_bin(LENGTH_GROUP_MIN_BP + LENGTH_GROUP_BIN_BP) == 1


def test_length_group_bin_at_max_bin():
    val = LENGTH_GROUP_MIN_BP + LENGTH_GROUP_MAX_BIN * LENGTH_GROUP_BIN_BP
    assert length_group_bin(val) == LENGTH_GROUP_MAX_BIN


def test_length_group_bin_above_max_returns_minus_one():
    val = LENGTH_GROUP_MIN_BP + (LENGTH_GROUP_MAX_BIN + 1) * LENGTH_GROUP_BIN_BP
    assert length_group_bin(val) == -1


# --------------------------------------------------------------------------
# Real-data ETL (skipped when fixtures absent)
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_returns_one_row_per_probe():
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    # Sanity: each molecule contributes num_probes rows; sum across all
    # molecules equals total rows.
    rows_per_mol = df.groupby("uid")["probe_idx"].count()
    nominal = df.groupby("uid")["num_probes"].first()
    assert (rows_per_mol == nominal).all(), (
        "row count per uid must equal num_probes recorded for that molecule"
    )


@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_residual_is_post_minus_pre():
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    # residual_bp must equal post_position_bp - pre_position_bp.
    diff = df["post_position_bp"] - df["pre_position_bp"]
    assert (df["residual_bp"] == diff).all()


@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_residual_is_predominantly_positive():
    """Production's TVC + head-dive corrections add bp to interval lengths,
    so the per-probe shift is non-negative for nearly every probe."""
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    nonneg = (df["residual_bp"] >= 0).mean()
    assert nonneg > 0.99, f"only {nonneg*100:.1f}% of residuals are non-negative"


@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_accepted_probes_dominate_reference_matches():
    """Reference matches only happen on accepted probes (bit 7 set) by
    construction in the ETL. Unaccepted probes always have has_reference=False."""
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    accepted = df[df["accepted"]]
    unaccepted = df[~df["accepted"]]
    assert (~unaccepted["has_reference"]).all(), (
        "unaccepted probes must have has_reference=False"
    )
    # Most accepted probes should match a reference; >70% is a safe lower bound.
    assert accepted["has_reference"].mean() > 0.7


@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_first_and_last_intervals_are_sentinels():
    """Per-probe context: probe_idx==0 has prev_interval_bp=-1, probe_idx==N-1
    has next_interval_bp=-1."""
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    first = df[df["probe_idx"] == 0]
    assert (first["prev_interval_bp"] == -1).all()
    # Last probe per molecule
    last = df.groupby("uid").tail(1)
    assert (last["next_interval_bp"] == -1).all()


@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_pre_positions_are_within_molecule():
    """Probe positions in maps.bin are relative to molecule start, so
    0 <= pre_position_bp <= molecule_length_bp for every probe."""
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    assert (df["pre_position_bp"] >= 0).all()
    assert (df["pre_position_bp"] <= df["molecule_length_bp"]).all()


@pytest.mark.skipif(
    not EXAMPLE_AVAILABLE,
    reason="example-remap fixtures not present (gitignored).",
)
def test_build_table_length_group_bin_matches_molecule_length():
    df = build_residual_table(EXAMPLE_DIR, EXAMPLE_RUN_ID)
    # For each molecule, length_group_bin should equal length_group_bin(mol_length).
    sample = df.groupby("uid").first().reset_index()
    for _, row in sample.head(50).iterrows():
        expected = length_group_bin(int(row["molecule_length_bp"]))
        assert row["length_group_bin"] == expected
