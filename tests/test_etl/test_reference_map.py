"""Unit tests for the extended reference_map parser.

These use synthetic fixtures so they don't depend on worktree ``support/``.
"""

from __future__ import annotations

import pytest

from mongoose.io.reference_map import load_reference_map


def _write_fixture(path, positions, strands=None, enzymes=None, genome_length=10000):
    strands = strands if strands is not None else [0] * len(positions)
    enzymes = enzymes if enzymes is not None else [0] * len(positions)
    assert len(positions) == len(strands) == len(enzymes)
    lines = [
        "//File Type:Reference Chromosome Maps",
        "//Line 1:Probe Location",
        "//Line 2:Nicked Strand",
        "//Line 3:Enzyme Index",
        "//DNA Sequence:synthetic",
        f"//Total Basepair Length:{genome_length}",
        "\t".join(str(p) for p in positions),
        "\t".join(str(s) for s in strands),
        "\t".join(str(e) for e in enzymes),
        "//Completed:1/1/2026",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def test_parses_three_data_rows(tmp_path):
    p = tmp_path / "ref.txt"
    _write_fixture(p, positions=[100, 200, 300], strands=[0, 1, 0], enzymes=[0, 0, 0])
    rm = load_reference_map(p)
    assert list(rm.probe_positions) == [100, 200, 300]
    assert list(rm.strands) == [0, 1, 0]
    assert list(rm.enzyme_indices) == [0, 0, 0]
    assert rm.genome_length == 10000


def test_lookup_is_one_based(tmp_path):
    p = tmp_path / "ref.txt"
    _write_fixture(p, positions=[100, 200, 300], strands=[1, 0, 1])
    rm = load_reference_map(p)
    assert rm.lookup(1) == (100, 1)
    assert rm.lookup(2) == (200, 0)
    assert rm.lookup(3) == (300, 1)


def test_lookup_rejects_out_of_range(tmp_path):
    p = tmp_path / "ref.txt"
    _write_fixture(p, positions=[100, 200])
    rm = load_reference_map(p)
    with pytest.raises(IndexError):
        rm.lookup(0)
    with pytest.raises(IndexError):
        rm.lookup(3)


def test_unsorted_positions_fail_loudly(tmp_path):
    """Spec-critical: source file MUST be strictly ascending, no silent sort."""
    p = tmp_path / "ref.txt"
    _write_fixture(p, positions=[300, 100, 200])
    with pytest.raises(ValueError, match="not strictly ascending"):
        load_reference_map(p)


def test_duplicated_positions_also_fail(tmp_path):
    p = tmp_path / "ref.txt"
    _write_fixture(p, positions=[100, 100, 200])
    with pytest.raises(ValueError, match="not strictly ascending"):
        load_reference_map(p)


def test_row_length_mismatch_errors(tmp_path):
    p = tmp_path / "ref.txt"
    # Write strand row with wrong length.
    p.write_text(
        "//File Type:Reference Chromosome Maps\n"
        "//DNA Sequence:synthetic\n"
        "//Total Basepair Length:10000\n"
        "100\t200\t300\n"
        "0\t1\n"
        "0\t0\t0\n",
        encoding="latin-1",
    )
    with pytest.raises(ValueError, match="row lengths disagree"):
        load_reference_map(p)
