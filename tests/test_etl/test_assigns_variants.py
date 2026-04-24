"""Unit tests for the extended assigns parser.

Covers the two row shapes: aligned (long rows with ProbeK trailing
columns) and unaligned (short rows, ref_index=-1, no ProbeK values).
"""

from __future__ import annotations

from mongoose.io.assigns import load_assigns


def test_aligned_row_captures_all_fixed_fields(tmp_path):
    p = tmp_path / "a.assigns"
    # Header line starts with "RefIndex" per spec -- parser skips it.
    p.write_text(
        "RefIndex\tFragmentUID\tDirection\tAlignmentScore\tSecondBest\t"
        "StretchFactor\tStretchOffset\tWeight\tProbe0\tProbe1\n"
        "0\t42\t1\t100000\t20000\t1.05\t-0.5\t0.9\t5\t7\n",
        encoding="latin-1",
    )
    [row] = load_assigns(p)
    assert row.ref_index == 0
    assert row.fragment_uid == 42
    assert row.direction == 1
    assert row.alignment_score == 100000
    assert row.second_best_score == 20000
    assert abs(row.stretch_factor - 1.05) < 1e-9
    assert abs(row.stretch_offset - (-0.5)) < 1e-9
    assert abs(row.weight - 0.9) < 1e-9
    assert row.probe_indices == (5, 7)


def test_unaligned_row_has_no_probe_indices(tmp_path):
    # ref=-1 rows have exactly 8 columns (no trailing probe indices).
    p = tmp_path / "a.assigns"
    p.write_text(
        "-1\t143566\t0\t0\t0\t1.0\t0\t1.0\n",
        encoding="latin-1",
    )
    [row] = load_assigns(p)
    assert row.ref_index == -1
    assert row.fragment_uid == 143566
    assert row.probe_indices == ()


def test_aligned_with_zero_probe_index_means_unassigned(tmp_path):
    # Spec §96: ProbeK=0 means "detected, not matched". Parser passes
    # it through as 0; the downstream ETL distinguishes 0 from missing.
    p = tmp_path / "a.assigns"
    p.write_text("0\t1\t1\t100\t50\t1.0\t0.0\t1.0\t0\t0\t3\n", encoding="latin-1")
    [row] = load_assigns(p)
    assert row.probe_indices == (0, 0, 3)


def test_skips_comment_and_header_lines(tmp_path):
    p = tmp_path / "a.assigns"
    p.write_text(
        "// comment\n"
        "RefIndex\tFragmentUID\tDirection\tAlignmentScore\tSecondBest\t"
        "StretchFactor\tStretchOffset\tWeight\tProbe0\n"
        "\n"
        "0\t1\t1\t10\t5\t1.0\t0.0\t0.5\t2\n",
        encoding="latin-1",
    )
    rows = load_assigns(p)
    assert len(rows) == 1
    assert rows[0].probe_indices == (2,)


def test_variable_length_trailing(tmp_path):
    """Spec §106 warns rows may have fewer trailing probe columns
    than the header advertises."""
    p = tmp_path / "a.assigns"
    p.write_text(
        "RefIndex\tFragmentUID\tDirection\tAlignmentScore\tSecondBest\t"
        "StretchFactor\tStretchOffset\tWeight\tProbe0\tProbe1\tProbe2\n"
        "0\t1\t1\t10\t5\t1.0\t0.0\t0.5\t4\t5\n",  # only 2 probe columns
        encoding="latin-1",
    )
    [row] = load_assigns(p)
    assert row.probe_indices == (4, 5)
