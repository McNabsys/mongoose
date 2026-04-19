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
