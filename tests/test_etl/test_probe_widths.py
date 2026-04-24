"""Unit tests for M1_probeWidths.txt parser + wide_peak_bin helper."""

from __future__ import annotations

import numpy as np

from mongoose.etl.probe_widths import load_probe_widths, wide_peak_bin


def _write_fixture(path, groups_data):
    """groups_data: list of (group_id, min_v, max_v, width, counts_list)."""
    lines = ["//M1 Probe Widths"]
    for gid, lo, hi, w, counts in groups_data:
        lines.append(
            f"Group: {gid} bins: {len(counts)} min: {lo} max: {hi} width: {w}"
        )
        lines.append(",".join(str(c) for c in counts))
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def test_load_two_groups(tmp_path):
    p = tmp_path / "M1.txt"
    _write_fixture(
        p,
        [
            (0, 75000, 85000, 10000, [1.0, 2.0, 3.0]),
            (1, 85000, 95000, 10000, [4, 5, 6, 7]),
        ],
    )
    pw = load_probe_widths(p)
    assert len(pw.groups) == 2
    assert pw.groups[0].group == 0
    assert pw.groups[0].n_bins == 3
    assert pw.groups[0].velocity_min == 75000
    assert pw.groups[0].velocity_max == 85000
    assert np.array_equal(pw.groups[0].counts, np.array([1.0, 2.0, 3.0]))
    assert pw.groups[1].n_bins == 4


def test_bin_count_mismatch_raises(tmp_path):
    p = tmp_path / "M1.txt"
    p.write_text(
        "//M1 Probe Widths\n"
        "Group: 0 bins: 5 min: 75000 max: 85000 width: 10000\n"
        "1,2,3\n",
        encoding="latin-1",
    )
    try:
        load_probe_widths(p)
    except ValueError as exc:
        assert "5 bins" in str(exc) and "3 values" in str(exc)
    else:
        raise AssertionError("expected ValueError on bin-count mismatch")


def test_group_for_velocity(tmp_path):
    p = tmp_path / "M1.txt"
    _write_fixture(
        p,
        [
            (0, 75000, 85000, 10000, [1, 1, 1]),
            (1, 85000, 95000, 10000, [1, 1, 1]),
        ],
    )
    pw = load_probe_widths(p)
    assert pw.group_for_velocity(75000) == 0
    assert pw.group_for_velocity(84999.9) == 0
    assert pw.group_for_velocity(85000) == 1
    assert pw.group_for_velocity(94999) == 1
    assert pw.group_for_velocity(95000) is None
    assert pw.group_for_velocity(50000) is None


def test_wide_peak_bin_bimodal():
    # Fake bimodal: narrow peak at bin 5, wide peak at bin 22.
    counts = np.zeros(30, dtype=np.float64)
    counts[5] = 100.0
    counts[22] = 80.0
    # Our heuristic splits at midpoint (15) and returns argmax of upper half.
    assert wide_peak_bin(counts) == 22


def test_wide_peak_bin_unimodal_upper():
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    # Split at 2, upper half is indices 2..4, argmax = 4.
    assert wide_peak_bin(counts) == 4
