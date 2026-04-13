"""Tests for the preprocessing pipeline.

Since preprocess_run requires matching TDB + probes.bin + assigns data,
full integration testing requires real data. These tests focus on:
1. PreprocessStats dataclass behavior
2. Output file format validation on mocked data
3. The preprocess_run function with a fully mocked IO layer
"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mongoose.data.preprocess import PreprocessStats, preprocess_run


def test_preprocess_stats_dataclass():
    """Verify PreprocessStats stores counts correctly."""
    stats = PreprocessStats(
        run_id="run_001",
        total_molecules=1000,
        clean_molecules=300,
        remapped_molecules=200,
        cached_molecules=150,
        total_waveform_bytes=600_000,
    )
    assert stats.run_id == "run_001"
    assert stats.total_molecules == 1000
    assert stats.cached_molecules == 150
    assert stats.total_waveform_bytes == 600_000


def _make_mock_molecule(
    uid, channel, num_probes, transloc_time_ms, mean_lvl1,
    structured=False, folded_start=False, folded_end=False, do_not_use=False,
    start_ms=100.0, probes=None,
):
    """Create a mock Molecule object."""
    mol = MagicMock()
    mol.uid = uid
    mol.channel = channel
    mol.num_probes = num_probes
    mol.transloc_time_ms = transloc_time_ms
    mol.mean_lvl1 = mean_lvl1
    mol.structured = structured
    mol.folded_start = folded_start
    mol.folded_end = folded_end
    mol.do_not_use = do_not_use
    mol.start_ms = start_ms
    mol.probes = probes or []
    return mol


def _make_mock_gt():
    """Create a mock MoleculeGT object."""
    gt = MagicMock()
    gt.probe_sample_indices = np.array([100, 200, 300, 400], dtype=np.int64)
    gt.inter_probe_deltas_bp = np.array([1000.0, 1500.0, 2000.0], dtype=np.float64)
    gt.velocity_targets_bp_per_ms = np.array([400.0, 350.0, 300.0, 280.0], dtype=np.float64)
    gt.reference_probe_bp = np.array([1000, 2000, 3500, 5500], dtype=np.int64)
    gt.direction = 1
    return gt


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_creates_output_files(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """With mocked IO, verify preprocess_run creates all expected output files."""
    # Mock TDB header
    header = MagicMock()
    header.channel_ids = [2, 3]
    header.amplitude_scale_factors = [1.0, 1.5]
    mock_tdb_header.return_value = header

    # Mock probes.bin with 3 molecules: 1 structured (filtered), 2 clean
    mol_structured = _make_mock_molecule(
        uid=0, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5, structured=True,
    )
    mol_clean1 = _make_mock_molecule(
        uid=1, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5,
    )
    mol_clean2 = _make_mock_molecule(
        uid=2, channel=3, num_probes=12, transloc_time_ms=80.0,
        mean_lvl1=0.6,
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 3
    probes_file.molecules = [mol_structured, mol_clean1, mol_clean2]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    # Mock assigns: uid 0 unmapped, uid 1 mapped, uid 2 mapped
    assign_unmapped = MagicMock()
    assign_unmapped.ref_index = -1
    assign_mapped1 = MagicMock()
    assign_mapped1.ref_index = 0
    assign_mapped2 = MagicMock()
    assign_mapped2.ref_index = 0
    mock_assigns.return_value = [assign_unmapped, assign_mapped1, assign_mapped2]

    # Mock reference map
    mock_ref_map.return_value = MagicMock()

    # Mock ground truth
    gt = _make_mock_gt()
    mock_build_gt.return_value = gt

    # Mock TDB molecule with waveform
    tdb_mol = MagicMock()
    tdb_mol.waveform = np.ones(500, dtype=np.int16) * 400
    mock_load_tdb_mol.return_value = tdb_mol

    stats = preprocess_run(
        run_id="test_run",
        tdb_path=Path("fake.tdb"),
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    run_dir = tmp_path / "test_run"
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "waveforms.bin").exists()
    assert (run_dir / "offsets.npy").exists()
    assert (run_dir / "conditioning.npy").exists()
    assert (run_dir / "molecules.pkl").exists()


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_stats(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """Verify stats counts: structured mol filtered, clean mols cached."""
    header = MagicMock()
    header.channel_ids = [2]
    header.amplitude_scale_factors = [1.0]
    mock_tdb_header.return_value = header

    mol_structured = _make_mock_molecule(
        uid=0, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5, structured=True,
    )
    mol_clean = _make_mock_molecule(
        uid=1, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5,
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 2
    probes_file.molecules = [mol_structured, mol_clean]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    assign_mapped = MagicMock()
    assign_mapped.ref_index = 0
    mock_assigns.return_value = [assign_mapped, assign_mapped]

    mock_ref_map.return_value = MagicMock()

    gt = _make_mock_gt()
    mock_build_gt.return_value = gt

    tdb_mol = MagicMock()
    tdb_mol.waveform = np.ones(300, dtype=np.int16) * 400
    mock_load_tdb_mol.return_value = tdb_mol

    stats = preprocess_run(
        run_id="stats_test",
        tdb_path=Path("fake.tdb"),
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    assert stats.total_molecules == 2
    assert stats.clean_molecules == 1  # structured one filtered
    assert stats.remapped_molecules == 1
    assert stats.cached_molecules == 1
    assert stats.total_waveform_bytes == 300 * 2  # 300 int16 samples


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_manifest_content(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """Verify manifest.json has expected structure and values."""
    header = MagicMock()
    header.channel_ids = [2]
    header.amplitude_scale_factors = [1.0]
    mock_tdb_header.return_value = header

    mol = _make_mock_molecule(
        uid=5, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5,
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 1
    probes_file.molecules = [mol]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    assign = MagicMock()
    assign.ref_index = 0
    # Need enough assigns entries for uid=5
    mock_assigns.return_value = [assign] * 6

    mock_ref_map.return_value = MagicMock()

    gt = _make_mock_gt()
    mock_build_gt.return_value = gt

    tdb_mol = MagicMock()
    tdb_mol.waveform = np.ones(400, dtype=np.int16) * 500
    mock_load_tdb_mol.return_value = tdb_mol

    preprocess_run(
        run_id="manifest_test",
        tdb_path=Path("fake.tdb"),
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    with open(tmp_path / "manifest_test" / "manifest.json") as f:
        manifest = json.load(f)

    assert manifest["run_id"] == "manifest_test"
    assert "stats" in manifest
    assert "molecules" in manifest
    assert len(manifest["molecules"]) == 1

    mol_entry = manifest["molecules"][0]
    assert mol_entry["uid"] == 5
    assert mol_entry["channel"] == 2
    assert mol_entry["num_samples"] == 400
    assert mol_entry["direction"] == 1


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_waveform_data(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """Verify waveforms.bin contains correct concatenated int16 data."""
    header = MagicMock()
    header.channel_ids = [2]
    header.amplitude_scale_factors = [1.0]
    mock_tdb_header.return_value = header

    mol = _make_mock_molecule(
        uid=0, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5,
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 1
    probes_file.molecules = [mol]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    assign = MagicMock()
    assign.ref_index = 0
    mock_assigns.return_value = [assign]

    mock_ref_map.return_value = MagicMock()
    mock_build_gt.return_value = _make_mock_gt()

    waveform_data = np.arange(200, dtype=np.int16)
    tdb_mol = MagicMock()
    tdb_mol.waveform = waveform_data
    mock_load_tdb_mol.return_value = tdb_mol

    preprocess_run(
        run_id="wfm_test",
        tdb_path=Path("fake.tdb"),
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    # Read back waveform data
    run_dir = tmp_path / "wfm_test"
    stored = np.fromfile(run_dir / "waveforms.bin", dtype=np.int16)
    np.testing.assert_array_equal(stored, waveform_data)

    # Verify offsets
    offsets = np.load(run_dir / "offsets.npy")
    assert offsets.shape == (1, 2)
    assert offsets[0, 0] == 0  # byte offset
    assert offsets[0, 1] == 200  # num samples


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_filters_low_probe_count(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """Molecules with too few probes should be filtered out."""
    header = MagicMock()
    header.channel_ids = [2]
    header.amplitude_scale_factors = [1.0]
    mock_tdb_header.return_value = header

    # 3 probes, but min_probes=8 by default
    mol = _make_mock_molecule(
        uid=0, channel=2, num_probes=3, transloc_time_ms=50.0,
        mean_lvl1=0.5,
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 1
    probes_file.molecules = [mol]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    mock_assigns.return_value = [MagicMock(ref_index=0)]
    mock_ref_map.return_value = MagicMock()
    mock_build_gt.return_value = _make_mock_gt()

    tdb_mol = MagicMock()
    tdb_mol.waveform = np.ones(100, dtype=np.int16)
    mock_load_tdb_mol.return_value = tdb_mol

    stats = preprocess_run(
        run_id="filter_test",
        tdb_path=Path("fake.tdb"),
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
        min_probes=8,
    )

    assert stats.clean_molecules == 0
    assert stats.cached_molecules == 0


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_filters_short_transloc(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """Molecules with transloc time below threshold should be filtered out."""
    header = MagicMock()
    header.channel_ids = [2]
    header.amplitude_scale_factors = [1.0]
    mock_tdb_header.return_value = header

    mol = _make_mock_molecule(
        uid=0, channel=2, num_probes=10, transloc_time_ms=10.0,  # below 30ms default
        mean_lvl1=0.5,
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 1
    probes_file.molecules = [mol]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    mock_assigns.return_value = [MagicMock(ref_index=0)]
    mock_ref_map.return_value = MagicMock()
    mock_build_gt.return_value = _make_mock_gt()

    tdb_mol = MagicMock()
    tdb_mol.waveform = np.ones(100, dtype=np.int16)
    mock_load_tdb_mol.return_value = tdb_mol

    stats = preprocess_run(
        run_id="transloc_test",
        tdb_path=Path("fake.tdb"),
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    assert stats.clean_molecules == 0
    assert stats.cached_molecules == 0
