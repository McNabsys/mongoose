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
