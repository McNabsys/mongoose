"""Tests for probes.bin.files companion parser."""

from pathlib import Path

import pytest


def test_parse_single_tdb_files(tmp_path):
    from mongoose.io.probes_bin_files import parse_probes_bin_files

    p = tmp_path / "run_probes.bin.files"
    p.write_bytes(
        b"000000D:\\SharedData\\Samples\\run\\2025-02-19\\run-OhmX202-202_20250219163656.tdb\r\n"
    )

    names = parse_probes_bin_files(p)
    assert names == ["run-OhmX202-202_20250219163656.tdb"]


def test_parse_multiple_tdb_files(tmp_path):
    from mongoose.io.probes_bin_files import parse_probes_bin_files

    p = tmp_path / "run_probes.bin.files"
    content = (
        b"000000D:\\SharedData\\Samples\\run\\date\\first.tdb\r\n"
        b"000001D:\\SharedData\\Samples\\run\\date\\second.tdb\r\n"
        b"000002D:\\SharedData\\Samples\\run\\date\\third.tdb\r\n"
    )
    p.write_bytes(content)

    names = parse_probes_bin_files(p)
    assert names == ["first.tdb", "second.tdb", "third.tdb"]


def test_parse_respects_explicit_index_order(tmp_path):
    """If indices are non-contiguous or out of order, the returned list is indexed correctly."""
    from mongoose.io.probes_bin_files import parse_probes_bin_files

    p = tmp_path / "run_probes.bin.files"
    # Written out-of-order: index 2 then 0 then 1
    content = (
        b"000002D:\\path\\third.tdb\r\n"
        b"000000D:\\path\\first.tdb\r\n"
        b"000001D:\\path\\second.tdb\r\n"
    )
    p.write_bytes(content)

    names = parse_probes_bin_files(p)
    # Returned list must be indexed by the 6-digit index, not file order
    assert names == ["first.tdb", "second.tdb", "third.tdb"]


def test_parse_ignores_blank_and_short_lines(tmp_path):
    from mongoose.io.probes_bin_files import parse_probes_bin_files

    p = tmp_path / "run_probes.bin.files"
    # Trailing blank line after CRLF is expected by some writers
    content = b"000000D:\\path\\only.tdb\r\n\r\n"
    p.write_bytes(content)

    names = parse_probes_bin_files(p)
    assert names == ["only.tdb"]


def test_parse_raises_on_duplicate_index(tmp_path):
    from mongoose.io.probes_bin_files import parse_probes_bin_files

    p = tmp_path / "run_probes.bin.files"
    p.write_bytes(
        b"000000D:\\a\\first.tdb\r\n"
        b"000000D:\\b\\second.tdb\r\n"
    )

    with pytest.raises(ValueError, match="duplicate index"):
        parse_probes_bin_files(p)


def test_parse_raises_on_missing_index(tmp_path):
    """Gap in indices (0 then 2, skipping 1) is an error -- we can't silently
    produce a list with an unknown gap."""
    from mongoose.io.probes_bin_files import parse_probes_bin_files

    p = tmp_path / "run_probes.bin.files"
    p.write_bytes(
        b"000000D:\\a\\first.tdb\r\n"
        b"000002D:\\c\\third.tdb\r\n"
    )

    with pytest.raises(ValueError, match="missing index 1"):
        parse_probes_bin_files(p)
