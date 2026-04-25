"""Tests for the V5 ``_reads_maps.bin`` parser."""
from __future__ import annotations

from pathlib import Path

import pytest

from mongoose.io.reads_maps_bin import (
    PROBE_ATTR_ACCEPTED,
    PROBE_ATTR_IN_CLEAN_REGION,
    PROBE_ATTR_IN_STRUCTURE,
    load_reads_maps_bin,
)


# Optional fixture: gitignored under support/. When it's not present
# (e.g., on CI) the data-dependent tests skip rather than fail.
EXAMPLE_DIR = Path(
    "support/example-remap/STB03-064B-02L58270w05-202G16g/AllCh"
)
EXAMPLE_PRE = EXAMPLE_DIR / "STB03-064B-02L58270w05-202G16g_uncorrected_reads_maps.bin"
EXAMPLE_POST = EXAMPLE_DIR / "STB03-064B-02L58270w05-202G16g_reads_maps.bin"


# --------------------------------------------------------------------------
# Header validation (no fixtures needed -- crafted bytes)
# --------------------------------------------------------------------------

def _build_header(magic=0x5342414E, file_type=0x00010001, version=5, num_maps=0):
    """Construct a 20-byte header with custom fields for negative tests."""
    import struct
    return struct.pack("<iIiii", 20, magic, file_type, version, num_maps)


def test_load_rejects_bad_magic(tmp_path):
    bad = tmp_path / "bad_magic.bin"
    bad.write_bytes(_build_header(magic=0xDEADBEEF))
    with pytest.raises(ValueError, match="bad magic"):
        load_reads_maps_bin(bad)


def test_load_rejects_wrong_file_type(tmp_path):
    bad = tmp_path / "wrong_type.bin"
    bad.write_bytes(_build_header(file_type=0x99))
    with pytest.raises(ValueError, match="unexpected file_type"):
        load_reads_maps_bin(bad)


def test_load_rejects_unsupported_version(tmp_path):
    bad = tmp_path / "old_version.bin"
    bad.write_bytes(_build_header(version=4))
    with pytest.raises(ValueError, match="unsupported file_version"):
        load_reads_maps_bin(bad)


def test_load_rejects_short_header(tmp_path):
    bad = tmp_path / "short.bin"
    bad.write_bytes(b"\x00" * 8)
    with pytest.raises(ValueError, match="shorter than V5 header"):
        load_reads_maps_bin(bad)


def test_load_empty_file_zero_maps(tmp_path):
    """Header with num_maps=0 should parse cleanly to an empty file."""
    empty = tmp_path / "empty.bin"
    empty.write_bytes(_build_header(num_maps=0))
    f = load_reads_maps_bin(empty)
    assert f.file_version == 5
    assert f.num_maps == 0
    assert f.molecules == []


# --------------------------------------------------------------------------
# Real-data parse (skipped when example fixture is not on disk)
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not EXAMPLE_POST.exists(),
    reason="example_reads_maps.bin fixture not present (gitignored).",
)
def test_load_real_post_corrected_returns_populated_molecules():
    f = load_reads_maps_bin(EXAMPLE_POST, max_molecules=5)
    assert f.file_version == 5
    assert f.num_maps > 0
    assert len(f.molecules) == 5

    m = f.molecules[0]
    assert m.uid >= 0
    assert m.channel >= 1  # 1-based per spec
    assert m.molecule_length_bp > 0
    assert len(m.probes) == m.num_probes
    assert len(m.structures) == m.num_structures


@pytest.mark.skipif(
    not EXAMPLE_POST.exists(),
    reason="example_reads_maps.bin fixture not present (gitignored).",
)
def test_real_data_probes_are_widely_accepted():
    """On a clean run, the vast majority of probes should have bit 7 set."""
    f = load_reads_maps_bin(EXAMPLE_POST, max_molecules=20)
    total = sum(len(m.probes) for m in f.molecules)
    accepted = sum(p.accepted for m in f.molecules for p in m.probes)
    # Exact rate varies by run; >70% is a safe lower bound.
    assert total > 0
    assert accepted / total > 0.7, (
        f"only {accepted}/{total} probes accepted -- check parser bit-mapping"
    )


@pytest.mark.skipif(
    not (EXAMPLE_PRE.exists() and EXAMPLE_POST.exists()),
    reason="pre/post pair not present (gitignored).",
)
def test_pre_post_pair_has_same_uid_and_probe_count_per_molecule():
    """The TVC/head-dive correction modifies positions only; molecule
    structure (uid, channel, num_probes) is unchanged."""
    pre = load_reads_maps_bin(EXAMPLE_PRE, max_molecules=10)
    post = load_reads_maps_bin(EXAMPLE_POST, max_molecules=10)
    assert pre.num_maps == post.num_maps
    assert len(pre.molecules) == len(post.molecules)
    for pm, om in zip(pre.molecules, post.molecules, strict=True):
        assert pm.uid == om.uid
        assert pm.channel == om.channel
        assert pm.molecule_id == om.molecule_id
        assert pm.num_probes == om.num_probes
        assert pm.num_structures == om.num_structures


@pytest.mark.skipif(
    not (EXAMPLE_PRE.exists() and EXAMPLE_POST.exists()),
    reason="pre/post pair not present (gitignored).",
)
def test_pre_post_pair_shows_nonzero_position_delta():
    """The whole point of Direction C: production's correction shifts
    probe positions. If the delta is identically zero, either we are
    looking at the wrong files or the corrections were never applied."""
    pre = load_reads_maps_bin(EXAMPLE_PRE, max_molecules=5)
    post = load_reads_maps_bin(EXAMPLE_POST, max_molecules=5)

    n_nonzero_molecules = 0
    for pm, om in zip(pre.molecules, post.molecules, strict=True):
        deltas = [op.position_bp - pp.position_bp for pp, op in zip(pm.probes, om.probes)]
        if any(d != 0 for d in deltas):
            n_nonzero_molecules += 1

    assert n_nonzero_molecules == len(pre.molecules), (
        f"expected ALL molecules to have non-zero position deltas; "
        f"got {n_nonzero_molecules}/{len(pre.molecules)}"
    )


@pytest.mark.skipif(
    not (EXAMPLE_PRE.exists() and EXAMPLE_POST.exists()),
    reason="pre/post pair not present (gitignored).",
)
def test_pre_post_pair_shows_zero_width_delta():
    """The TVC + head-dive corrections are position-only; widths stay fixed."""
    pre = load_reads_maps_bin(EXAMPLE_PRE, max_molecules=5)
    post = load_reads_maps_bin(EXAMPLE_POST, max_molecules=5)

    for pm, om in zip(pre.molecules, post.molecules, strict=True):
        for pp, op in zip(pm.probes, om.probes, strict=True):
            assert pp.width_bp == op.width_bp, (
                f"width mismatch in mol uid={pm.uid}: pre={pp.width_bp} post={op.width_bp}"
            )


# --------------------------------------------------------------------------
# Bit-field property accessors
# --------------------------------------------------------------------------

def test_map_probe_accepted_property():
    from mongoose.io.reads_maps_bin import MapProbe
    p = MapProbe(position_bp=1000, attribute=PROBE_ATTR_ACCEPTED, width_bp=831)
    assert p.accepted
    p2 = MapProbe(position_bp=1000, attribute=PROBE_ATTR_IN_CLEAN_REGION, width_bp=831)
    assert not p2.accepted


def test_map_probe_clean_and_structure_disjoint():
    from mongoose.io.reads_maps_bin import MapProbe
    clean = MapProbe(
        position_bp=1000,
        attribute=PROBE_ATTR_IN_CLEAN_REGION | PROBE_ATTR_ACCEPTED,
        width_bp=831,
    )
    assert clean.in_clean_region
    assert not clean.in_structure
    in_struct = MapProbe(
        position_bp=2000,
        attribute=PROBE_ATTR_IN_STRUCTURE | PROBE_ATTR_ACCEPTED,
        width_bp=831,
    )
    assert not in_struct.in_clean_region
    assert in_struct.in_structure
