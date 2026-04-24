"""Unit tests for _remapSettings.txt parser."""

from __future__ import annotations

from pathlib import Path

from mongoose.etl.remap_settings import load_remap_settings


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="latin-1")


def test_parses_basic_key_value(tmp_path):
    p = tmp_path / "settings.txt"
    _write(
        p,
        "// a comment\n"
        "TagSize=511\n"
        "ALPHA=0.56\n"
        "use_probe_width_filter=True\n"
        "\n"
        "additConst=-1200\n",
    )
    s = load_remap_settings(p)
    assert s.get_int("TagSize") == 511
    assert abs(s.get_float("ALPHA") - 0.56) < 1e-9
    assert s.get_bool("use_probe_width_filter") is True
    assert s.get_int("additConst") == -1200


def test_get_float_coerces_int_strings(tmp_path):
    p = tmp_path / "settings.txt"
    _write(p, "TagSize=511\n")
    s = load_remap_settings(p)
    # stored as "511" — get_float should still coerce cleanly.
    assert s.get_float("TagSize") == 511.0


def test_missing_key_returns_none(tmp_path):
    p = tmp_path / "settings.txt"
    _write(p, "TagSize=511\n")
    s = load_remap_settings(p)
    assert s.get("MISSING") is None
    assert s.get_float("MISSING") is None
    assert s.get_int("MISSING") is None
    assert s.get_bool("MISSING") is None


def test_missing_key_default_respected(tmp_path):
    p = tmp_path / "settings.txt"
    _write(p, "")
    s = load_remap_settings(p)
    assert s.get("k", "fallback") == "fallback"
    assert s.get_float("k", default=9.5) == 9.5
    assert s.get_int("k", default=7) == 7
    assert s.get_bool("k", default=True) is True


def test_comment_and_blank_lines_ignored(tmp_path):
    p = tmp_path / "settings.txt"
    _write(
        p,
        "//translocation velocity compensation\n"
        "\n"
        "\n"
        "ttnorm_stretch_algorithm=2\n",
    )
    s = load_remap_settings(p)
    assert s.get_int("ttnorm_stretch_algorithm") == 2
    assert len(s.values) == 1


def test_float_valued_int(tmp_path):
    # Some Nabsys keys write int values as floats ("2.0"). get_int
    # should still yield 2, not crash, not return None.
    p = tmp_path / "settings.txt"
    _write(p, "k=2.0\n")
    s = load_remap_settings(p)
    assert s.get_int("k") == 2
