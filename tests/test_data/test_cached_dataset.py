"""Tests for CachedMoleculeDataset that reads from preprocessed cache directories."""

import json
import pickle

import numpy as np
import pytest
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset


@pytest.fixture
def fake_cache_dir(tmp_path):
    """Create a fake cache directory with 2 molecules for testing."""
    cache_dir = tmp_path / "test_run"
    cache_dir.mkdir()

    # 2 molecules: 100 and 200 samples respectively
    wfm1 = np.ones(100, dtype=np.int16) * 500
    wfm2 = np.ones(200, dtype=np.int16) * 600
    with open(cache_dir / "waveforms.bin", "wb") as f:
        f.write(wfm1.tobytes())
        f.write(wfm2.tobytes())

    # offsets: (byte_offset_into_waveforms_bin, num_samples)
    offsets = np.array([[0, 100], [200, 200]], dtype=np.int64)
    np.save(cache_dir / "offsets.npy", offsets)

    conditioning = np.array(
        [[0.5, 4.6, 0.0, 0.1, 0.0, 0.0], [0.6, 5.3, 0.0, 0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.save(cache_dir / "conditioning.npy", conditioning)

    gt_list = [
        {
            "probe_sample_indices": np.array([20, 50, 80], dtype=np.int64),
            "inter_probe_deltas_bp": np.array([1000.0, 2000.0], dtype=np.float64),
            "velocity_targets_bp_per_ms": np.array(
                [400.0, 350.0, 300.0], dtype=np.float64
            ),
            "reference_probe_bp": np.array([1000, 2000, 4000], dtype=np.int64),
            "direction": 1,
        },
        {
            "probe_sample_indices": np.array([30, 100, 170], dtype=np.int64),
            "inter_probe_deltas_bp": np.array([1500.0, 2500.0], dtype=np.float64),
            "velocity_targets_bp_per_ms": np.array(
                [380.0, 340.0, 290.0], dtype=np.float64
            ),
            "reference_probe_bp": np.array([500, 2000, 4500], dtype=np.int64),
            "direction": -1,
        },
    ]
    with open(cache_dir / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    manifest = {
        "run_id": "test_run",
        "stats": {"cached_molecules": 2},
        "molecules": [
            {
                "uid": 0,
                "channel": 2,
                "num_samples": 100,
                "num_probes": 3,
                "num_matched_probes": 3,
                "transloc_time_ms": 2.5,
                "mean_lvl1": 0.5,
                "direction": 1,
                "amplitude_scale": 1.0,
            },
            {
                "uid": 1,
                "channel": 3,
                "num_samples": 200,
                "num_probes": 3,
                "num_matched_probes": 3,
                "transloc_time_ms": 5.0,
                "mean_lvl1": 0.6,
                "direction": -1,
                "amplitude_scale": 1.0,
            },
        ],
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return cache_dir


def test_cached_dataset_loads(fake_cache_dir):
    """Verify the dataset loads all molecules from cache."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    assert len(ds) == 2


def test_cached_dataset_item_shapes(fake_cache_dir):
    """Verify shapes of tensors returned by __getitem__."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    assert item["waveform"].shape == (1, 100)
    assert item["probe_heatmap"].shape == (100,)
    assert item["mask"].shape == (100,)
    assert item["conditioning"].shape == (6,)
    assert len(item["probe_sample_indices"]) == 3
    assert len(item["gt_deltas_bp"]) == 2
    assert len(item["velocity_targets"]) == 3


def test_cached_dataset_second_molecule(fake_cache_dir):
    """Verify the second molecule has correct shape (different length)."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[1]

    assert item["waveform"].shape == (1, 200)
    assert item["probe_heatmap"].shape == (200,)
    assert item["mask"].shape == (200,)


def test_cached_dataset_waveform_values(fake_cache_dir):
    """Verify waveform is normalized by level-1 amplitude."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    # wfm1 = 500 (int16), amplitude_scale=1.0, mean_lvl1=0.5 mV = 500 uV
    # normalized = (500 * 1.0) / 500.0 = 1.0 for all samples
    expected = 1.0
    np.testing.assert_allclose(item["waveform"].numpy(), expected, rtol=1e-5)


def test_cached_dataset_molecule_uid(fake_cache_dir):
    """Verify molecule_uid is populated correctly."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    assert ds[0]["molecule_uid"] == 0
    assert ds[1]["molecule_uid"] == 1


def test_cached_dataset_tensor_types(fake_cache_dir):
    """Verify all outputs are tensors of expected dtypes."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    assert isinstance(item["waveform"], torch.Tensor)
    assert item["waveform"].dtype == torch.float32
    assert isinstance(item["conditioning"], torch.Tensor)
    assert item["conditioning"].dtype == torch.float32
    assert isinstance(item["probe_heatmap"], torch.Tensor)
    assert item["probe_heatmap"].dtype == torch.float32
    assert isinstance(item["mask"], torch.Tensor)
    assert item["mask"].dtype == torch.bool


def test_cached_dataset_multiple_dirs(fake_cache_dir, tmp_path):
    """Verify dataset can merge multiple cache directories."""
    # Create a second cache dir with 1 molecule
    cache_dir2 = tmp_path / "test_run_2"
    cache_dir2.mkdir()

    wfm = np.ones(150, dtype=np.int16) * 400
    with open(cache_dir2 / "waveforms.bin", "wb") as f:
        f.write(wfm.tobytes())

    offsets = np.array([[0, 150]], dtype=np.int64)
    np.save(cache_dir2 / "offsets.npy", offsets)

    conditioning = np.zeros((1, 6), dtype=np.float32)
    np.save(cache_dir2 / "conditioning.npy", conditioning)

    gt_list = [
        {
            "probe_sample_indices": np.array([30, 75, 120], dtype=np.int64),
            "inter_probe_deltas_bp": np.array([1200.0, 1800.0], dtype=np.float64),
            "velocity_targets_bp_per_ms": np.array(
                [350.0, 320.0, 280.0], dtype=np.float64
            ),
            "reference_probe_bp": np.array([800, 2000, 3800], dtype=np.int64),
            "direction": 1,
        },
    ]
    with open(cache_dir2 / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    manifest = {
        "run_id": "test_run_2",
        "stats": {"cached_molecules": 1},
        "molecules": [
            {
                "uid": 10,
                "channel": 1,
                "num_samples": 150,
                "num_probes": 3,
                "num_matched_probes": 3,
                "transloc_time_ms": 3.75,
                "mean_lvl1": 0.4,
                "direction": 1,
                "amplitude_scale": 1.0,
            },
        ],
    }
    with open(cache_dir2 / "manifest.json", "w") as f:
        json.dump(manifest, f)

    ds = CachedMoleculeDataset([fake_cache_dir, cache_dir2])
    assert len(ds) == 3
    # Third item should come from second dir
    item = ds[2]
    assert item["waveform"].shape == (1, 150)
    assert item["molecule_uid"] == 10


def test_cached_dataset_velocity_units(fake_cache_dir):
    """Verify velocity targets are converted from bp/ms to bp/sample."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    # Original: [400, 350, 300] bp/ms
    # Convert: * 0.025 ms/sample = [10.0, 8.75, 7.5] bp/sample
    expected = np.array([400.0, 350.0, 300.0]) * 0.025
    np.testing.assert_allclose(
        item["velocity_targets"].numpy(), expected, rtol=1e-5
    )


def test_cached_dataset_mask_all_true(fake_cache_dir):
    """Mask should be all True (no padding in individual items)."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]
    assert item["mask"].all()


def test_cached_dataset_empty_dir(tmp_path):
    """Dataset with an empty cache dir (0 molecules) should have length 0."""
    cache_dir = tmp_path / "empty_run"
    cache_dir.mkdir()

    with open(cache_dir / "waveforms.bin", "wb") as f:
        pass  # empty file

    offsets = np.empty((0, 2), dtype=np.int64)
    np.save(cache_dir / "offsets.npy", offsets)

    conditioning = np.empty((0, 6), dtype=np.float32)
    np.save(cache_dir / "conditioning.npy", conditioning)

    with open(cache_dir / "molecules.pkl", "wb") as f:
        pickle.dump([], f)

    manifest = {
        "run_id": "empty_run",
        "stats": {"cached_molecules": 0},
        "molecules": [],
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    ds = CachedMoleculeDataset([cache_dir])
    assert len(ds) == 0
