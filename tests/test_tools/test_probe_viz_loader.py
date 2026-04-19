"""Loader tests for the probe-viz tool.

These tests use the real sample data in ``E. coli/``. They skip when that
data is not present, so CI without the dataset stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_DIR = Path("E. coli/Black/STB03-060A-02L58270w05-202G16j/2025-02-19")


def _require_sample_data():
    if not SAMPLE_DIR.exists():
        pytest.skip(f"Sample data missing: {SAMPLE_DIR}")


def test_discover_probes_bin_finds_remapped_allch():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import discover_probes_bin

    probes_bin, probes_bin_files = discover_probes_bin(SAMPLE_DIR)
    assert probes_bin.exists()
    assert probes_bin.name.endswith("_probes.bin")
    assert probes_bin_files.exists()
    assert probes_bin_files.name.endswith("_probes.bin.files")
    assert "AllCh" in str(probes_bin)


def test_discover_probes_bin_missing_raises(tmp_path):
    from mongoose.tools.probe_viz.loader import discover_probes_bin

    with pytest.raises(FileNotFoundError, match="_probes.bin"):
        discover_probes_bin(tmp_path)


def test_loader_filters_do_not_use_by_default():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    assert loader.total > 0
    assert loader.current_index == 0


def test_loader_include_do_not_use_expands_total():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    default = ProbeVizLoader(SAMPLE_DIR)
    all_mols = ProbeVizLoader(SAMPLE_DIR, include_do_not_use=True)
    assert all_mols.total > default.total


def test_loader_advance_clamps():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    loader.advance(-5)
    assert loader.current_index == 0
    loader.advance(loader.total + 10)
    assert loader.current_index == loader.total - 1
    loader.advance(-1)
    assert loader.current_index == loader.total - 2


def test_current_view_returns_waveform_and_scale():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader, ViewData

    loader = ProbeVizLoader(SAMPLE_DIR)
    view = loader.current_view()
    assert isinstance(view, ViewData)
    assert view.tdb_molecule.waveform.size > 0
    assert view.sample_rate > 0
    assert view.scale_uv_per_lsb > 0.0
    assert view.tdb_basename.endswith(".tdb")
    assert view.probe_molecule.channel == view.tdb_molecule.channel_source
    assert view.iter_index == 0
    assert view.iter_total == loader.total


def test_current_view_changes_after_advance():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    uid_a = loader.current_view().probe_molecule.uid
    loader.advance(1)
    uid_b = loader.current_view().probe_molecule.uid
    assert uid_a != uid_b


def test_goto_uid_jumps_to_matching_molecule():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    # Pick a UID that we know is not at index 0.
    loader.advance(5)
    target_uid = loader.current_view().probe_molecule.uid
    loader.advance(-100)  # go to 0
    assert loader.current_index == 0

    assert loader.goto_uid(target_uid) is True
    assert loader.current_view().probe_molecule.uid == target_uid


def test_goto_uid_returns_false_when_not_in_list():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    # Use a UID far above any plausible value.
    assert loader.goto_uid(99_999_999) is False


def test_goto_channel_mid_jumps():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    loader.advance(3)
    ch = loader.current_view().probe_molecule.channel
    mid = loader.current_view().probe_molecule.molecule_id
    loader.advance(-100)
    assert loader.goto_channel_mid(ch, mid) is True
    view = loader.current_view()
    assert view.probe_molecule.channel == ch
    assert view.probe_molecule.molecule_id == mid


def test_toggle_do_not_use_keeps_nearest_by_uid():
    _require_sample_data()
    from mongoose.tools.probe_viz.loader import ProbeVizLoader

    loader = ProbeVizLoader(SAMPLE_DIR)
    loader.advance(3)
    uid_before = loader.current_view().probe_molecule.uid

    loader.toggle_do_not_use()  # now includes do_not_use molecules

    # After toggle, total grew and we landed on the same or nearest UID.
    uid_after = loader.current_view().probe_molecule.uid
    # Same molecule is still present (toggling expands, doesn't remove).
    assert uid_after == uid_before
