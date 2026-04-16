"""Tests for the preprocessing pipeline (V1 rearchitecture, R4).

Since preprocess_run requires matching TDB + probes.bin + assigns data,
full integration testing requires real data. These tests focus on:
1. PreprocessStats dataclass behavior.
2. Output file format validation on mocked data.
3. The preprocess_run function with a fully mocked IO layer.
4. New V1 GT schema (reference_bp_positions, n_ref_probes, direction,
   warmstart_* fields) in the cached molecules.pkl.
5. The TDB-derived mean_lvl1_from_tdb field in the manifest and
   conditioning vector (replacing the wfmproc mean_lvl1).
"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

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
    start_ms=100.0, probes=None, molecule_id=0, file_name_index=0,
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
    mol.molecule_id = molecule_id
    mol.file_name_index = file_name_index
    return mol


def _make_mock_gt(
    reference_bp_positions=None,
    direction=1,
    warmstart_centers=None,
    warmstart_durations=None,
):
    """Create a mock MoleculeGT object with the new V1 schema."""
    if reference_bp_positions is None:
        reference_bp_positions = np.array([1000, 2000, 3500, 5500], dtype=np.int64)
    if warmstart_centers is None:
        warmstart_centers = np.array([100, 200, 300, 400], dtype=np.int64)
    if warmstart_durations is None:
        warmstart_durations = np.array([40.0, 42.0, 41.0, 39.0], dtype=np.float32)

    gt = MagicMock()
    gt.reference_bp_positions = reference_bp_positions
    gt.n_ref_probes = len(reference_bp_positions)
    gt.direction = direction
    gt.warmstart_probe_centers_samples = warmstart_centers
    gt.warmstart_probe_durations_samples = warmstart_durations
    return gt


def _make_mock_tdb_molecule(waveform, rise_end=10, fall_min=None):
    """Create a mock TdbMolecule with the fields preprocess_run needs."""
    if fall_min is None:
        fall_min = max(rise_end + 10, len(waveform) - 10)
    tdb_mol = MagicMock()
    tdb_mol.waveform = waveform
    tdb_mol.rise_conv_end_index = rise_end
    tdb_mol.fall_conv_min_index = fall_min
    return tdb_mol


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_creates_output_files(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
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
    mock_build_gt.return_value = _make_mock_gt()

    # Mock TDB index: channels 2 and 3, molecule_id 0
    mock_load_index.return_value = {(2, 0): 0, (3, 0): 0}

    # Mock TDB molecule with waveform
    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(
        np.ones(500, dtype=np.int16) * 400
    )

    preprocess_run(
        run_id="test_run",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
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
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_stats(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
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
    mock_build_gt.return_value = _make_mock_gt()

    mock_load_index.return_value = {(2, 0): 0}

    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(
        np.ones(300, dtype=np.int16) * 400
    )

    stats = preprocess_run(
        run_id="stats_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
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
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_manifest_content(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """Verify manifest.json has expected structure and values (new V1 schema)."""
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
    mock_build_gt.return_value = _make_mock_gt()

    mock_load_index.return_value = {(2, 0): 0}

    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(
        np.ones(400, dtype=np.int16) * 500
    )

    preprocess_run(
        run_id="manifest_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
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
    # New V1 fields.
    assert "n_ref_probes" in mol_entry
    assert mol_entry["n_ref_probes"] == 4  # from default _make_mock_gt
    assert "mean_lvl1_from_tdb" in mol_entry
    # The old wfmproc mean_lvl1 field is no longer in the manifest.
    assert "mean_lvl1" not in mol_entry


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_run_waveform_data(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
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

    mock_load_index.return_value = {(2, 0): 0}

    waveform_data = np.arange(200, dtype=np.int16)
    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(waveform_data)

    preprocess_run(
        run_id="wfm_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
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
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_filters_low_probe_count(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
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

    mock_load_index.return_value = {(2, 0): 0}

    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(
        np.ones(100, dtype=np.int16)
    )

    stats = preprocess_run(
        run_id="filter_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
        min_probes=8,
    )

    assert stats.clean_molecules == 0
    assert stats.cached_molecules == 0


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_filters_short_transloc(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
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

    mock_load_index.return_value = {(2, 0): 0}

    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(
        np.ones(100, dtype=np.int16)
    )

    stats = preprocess_run(
        run_id="transloc_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    assert stats.clean_molecules == 0
    assert stats.cached_molecules == 0


@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_molecules_pkl_new_schema(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
    mock_load_tdb_mol,
    mock_build_gt,
    tmp_path,
):
    """molecules.pkl stores the V1 GT schema (no deprecated fields)."""
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

    mock_assigns.return_value = [MagicMock(ref_index=0)]
    mock_ref_map.return_value = MagicMock()

    ref_bp = np.array([100, 500, 900, 1500, 2100], dtype=np.int64)
    centers = np.array([20, 40, 60, 80, 100], dtype=np.int64)
    durations = np.array([10.0, 11.0, 9.0, 12.0, 10.5], dtype=np.float32)
    mock_build_gt.return_value = _make_mock_gt(
        reference_bp_positions=ref_bp,
        direction=-1,
        warmstart_centers=centers,
        warmstart_durations=durations,
    )

    mock_load_index.return_value = {(2, 0): 0}

    mock_load_tdb_mol.return_value = _make_mock_tdb_molecule(
        np.ones(300, dtype=np.int16)
    )

    preprocess_run(
        run_id="schema_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    with open(tmp_path / "schema_test" / "molecules.pkl", "rb") as f:
        gt_list = pickle.load(f)

    assert len(gt_list) == 1
    entry = gt_list[0]
    # New schema keys are present.
    assert "reference_bp_positions" in entry
    assert "n_ref_probes" in entry
    assert "direction" in entry
    assert "warmstart_probe_centers_samples" in entry
    assert "warmstart_probe_durations_samples" in entry
    # Deprecated keys are NOT present.
    assert "probe_sample_indices" not in entry
    assert "inter_probe_deltas_bp" not in entry
    assert "velocity_targets_bp_per_ms" not in entry
    assert "reference_probe_bp" not in entry

    np.testing.assert_array_equal(entry["reference_bp_positions"], ref_bp)
    assert entry["n_ref_probes"] == len(ref_bp)
    assert entry["direction"] == -1
    np.testing.assert_array_equal(entry["warmstart_probe_centers_samples"], centers)
    np.testing.assert_array_equal(entry["warmstart_probe_durations_samples"], durations)


@patch("mongoose.data.preprocess.estimate_level1")
@patch("mongoose.data.preprocess.build_molecule_gt")
@patch("mongoose.data.preprocess.load_tdb_molecule_at_offset")
@patch("mongoose.data.preprocess.load_tdb_index")
@patch("mongoose.data.preprocess.load_reference_map")
@patch("mongoose.data.preprocess.load_assigns")
@patch("mongoose.data.preprocess.load_probes_bin")
@patch("mongoose.data.preprocess.load_tdb_header")
def test_preprocess_computes_mean_lvl1_from_tdb(
    mock_tdb_header,
    mock_probes_bin,
    mock_assigns,
    mock_ref_map,
    mock_load_index,
    mock_load_tdb_mol,
    mock_build_gt,
    mock_estimate_level1,
    tmp_path,
):
    """estimate_level1 is invoked per molecule, and its value is stored.

    The manifest's ``mean_lvl1_from_tdb`` and the conditioning vector's
    position 0 must both come from ``estimate_level1`` (TDB-derived), not
    from the wfmproc ``mol.mean_lvl1``.
    """
    header = MagicMock()
    header.channel_ids = [2]
    header.amplitude_scale_factors = [1.0]
    mock_tdb_header.return_value = header

    mol = _make_mock_molecule(
        uid=0, channel=2, num_probes=10, transloc_time_ms=50.0,
        mean_lvl1=0.5,  # wfmproc value, should NOT be stored
    )

    probes_file = MagicMock()
    probes_file.num_molecules = 1
    probes_file.molecules = [mol]
    probes_file.last_sample_time = 100.0
    mock_probes_bin.return_value = probes_file

    mock_assigns.return_value = [MagicMock(ref_index=0)]
    mock_ref_map.return_value = MagicMock()
    mock_build_gt.return_value = _make_mock_gt()

    mock_load_index.return_value = {(2, 0): 0}

    tdb_mol = _make_mock_tdb_molecule(
        np.ones(300, dtype=np.int16) * 400,
        rise_end=5,
        fall_min=290,
    )
    mock_load_tdb_mol.return_value = tdb_mol

    tdb_lvl1 = 1234.5
    mock_estimate_level1.return_value = tdb_lvl1

    preprocess_run(
        run_id="lvl1_test",
        tdb_paths=[Path("fake.tdb")],
        tdb_index_paths=[Path("fake.tdb_index")],
        probes_bin_path=Path("fake_probes.bin"),
        assigns_path=Path("fake.assigns"),
        reference_map_path=Path("fake_ref.txt"),
        output_dir=tmp_path,
    )

    # estimate_level1 called once (one cached molecule) with TDB indices.
    assert mock_estimate_level1.call_count == 1
    _args, kwargs = mock_estimate_level1.call_args
    assert kwargs["rise_end_idx"] == 5
    assert kwargs["fall_min_idx"] == 290

    # Manifest carries the TDB-derived level-1.
    with open(tmp_path / "lvl1_test" / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["molecules"][0]["mean_lvl1_from_tdb"] == tdb_lvl1

    # Conditioning vector position 0 matches.
    cond = np.load(tmp_path / "lvl1_test" / "conditioning.npy")
    assert cond.shape == (1, 6)
    assert cond[0, 0] == np.float32(tdb_lvl1)


def test_preprocess_waveform_identity_when_probes_bin_skips_tdb_molecule(tmp_path):
    """When probes.bin omits TDB molecule N, downstream probes.bin record M
    (M > N) must still load the waveform with the correct (channel, MID).

    This is the bug that made the old positional-indexing code produce
    mis-paired waveforms the instant probes.bin was a strict subset of TDB.
    """
    tdb_header = MagicMock()
    tdb_header.channel_ids = [2]
    tdb_header.amplitude_scale_factors = [1.0]

    # Waveforms distinguishable by their first value:
    # TDB (ch=2,mid=0) = all zeros
    # TDB (ch=2,mid=1) = all ones    (absent from probes.bin)
    # TDB (ch=2,mid=2) = all twos
    mol_at_offset_1000 = _make_mock_tdb_molecule(np.zeros(100, dtype=np.int16))
    mol_at_offset_3000 = _make_mock_tdb_molecule(np.full(100, 2, dtype=np.int16))

    def fake_index(path):
        # One TDB with all three molecules present in the index
        return {(2, 0): 1000, (2, 1): 2000, (2, 2): 3000}

    def fake_load_at_offset(path, offset):
        # Explicit dispatch -- if offset 2000 (mid=1) ever shows up, KeyError
        # will make the test fail loudly, proving a subset-skip bug
        return {1000: mol_at_offset_1000, 3000: mol_at_offset_3000}[offset]

    # Two probes.bin molecules: mid=0 and mid=2. mid=1 missing.
    probes_file = MagicMock()
    probes_file.num_molecules = 2
    probes_file.last_sample_time = 100.0
    probes_file.molecules = [
        _make_mock_molecule(uid=0, channel=2, num_probes=10, transloc_time_ms=50.0,
                             mean_lvl1=0.5, molecule_id=0, file_name_index=0),
        _make_mock_molecule(uid=1, channel=2, num_probes=10, transloc_time_ms=50.0,
                             mean_lvl1=0.5, molecule_id=2, file_name_index=0),
    ]

    assign = MagicMock()
    assign.ref_index = 0

    with patch("mongoose.data.preprocess.load_tdb_header", return_value=tdb_header), \
         patch("mongoose.data.preprocess.load_tdb_index", side_effect=fake_index), \
         patch("mongoose.data.preprocess.load_tdb_molecule_at_offset", side_effect=fake_load_at_offset), \
         patch("mongoose.data.preprocess.load_probes_bin", return_value=probes_file), \
         patch("mongoose.data.preprocess.load_assigns", return_value=[assign, assign]), \
         patch("mongoose.data.preprocess.load_reference_map", return_value=MagicMock()), \
         patch("mongoose.data.preprocess.build_molecule_gt", return_value=_make_mock_gt()), \
         patch("mongoose.data.preprocess.estimate_level1", return_value=0.5):

        stats = preprocess_run(
            run_id="subset",
            tdb_paths=[Path("fake.tdb")],
            tdb_index_paths=[Path("fake.tdb_index")],
            probes_bin_path=Path("fake_probes.bin"),
            assigns_path=Path("fake.assigns"),
            reference_map_path=Path("fake_ref.txt"),
            output_dir=tmp_path,
        )

    assert stats.cached_molecules == 2

    # Verify waveforms pair by identity.
    waveforms_path = tmp_path / "subset" / "waveforms.bin"
    data = np.fromfile(waveforms_path, dtype=np.int16)
    assert data.size == 200, f"expected 200 int16 samples, got {data.size}"
    assert data[0:100].tolist() == [0] * 100, "first molecule should be all zeros (ch=2, mid=0)"
    assert data[100:200].tolist() == [2] * 100, "second molecule should be all twos (ch=2, mid=2), NOT mid=1"
