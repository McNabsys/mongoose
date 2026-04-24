"""Unit tests for _version.txt parser."""

from __future__ import annotations

from mongoose.etl.version_file import load_version_file


def test_parses_canonical(tmp_path):
    p = tmp_path / "version.txt"
    p.write_text("PROGRAM_VERSION=1.9.7309\nPICKER=pp-705\n", encoding="latin-1")
    v = load_version_file(p)
    assert v.program_version == "1.9.7309"
    assert v.picker == "pp-705"
    assert v.extra == {}


def test_missing_keys_yield_none(tmp_path):
    p = tmp_path / "version.txt"
    p.write_text("OTHER=foo\n", encoding="latin-1")
    v = load_version_file(p)
    assert v.program_version is None
    assert v.picker is None
    assert v.extra == {"OTHER": "foo"}
