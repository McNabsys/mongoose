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
