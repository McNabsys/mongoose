"""Tests for CachedMoleculeDataset that reads from preprocessed cache directories.

V1 rearchitecture (R6): molecules.pkl now uses the new schema
(reference_bp_positions, n_ref_probes, direction, optional warmstart_*).
"""

import json
import pickle

import numpy as np
import pytest
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset


def _write_manifest(cache_dir, molecules):
    manifest = {
        "run_id": cache_dir.name,
        "stats": {"cached_molecules": len(molecules)},
        "molecules": molecules,
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


@pytest.fixture
def fake_cache_dir(tmp_path):
    """Create a fake cache directory with 2 molecules in the new GT schema."""
    cache_dir = tmp_path / "test_run"
    cache_dir.mkdir()

    # 2 molecules: 100 and 200 samples respectively
    wfm1 = np.ones(100, dtype=np.int16) * 500
    wfm2 = np.ones(200, dtype=np.int16) * 600
    with open(cache_dir / "waveforms.bin", "wb") as f:
        f.write(wfm1.tobytes())
        f.write(wfm2.tobytes())

    offsets = np.array([[0, 100], [200, 200]], dtype=np.int64)
    np.save(cache_dir / "offsets.npy", offsets)

    conditioning = np.array(
        [[0.5, 4.6, 0.0, 0.1, 0.0, 0.0], [0.6, 5.3, 0.0, 0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.save(cache_dir / "conditioning.npy", conditioning)

    # Molecule 0: has warmstart labels. Molecule 1: no warmstart (None).
    gt_list = [
        {
            "reference_bp_positions": np.array([1000, 2000, 4000], dtype=np.int64),
            "n_ref_probes": 3,
            "direction": 1,
            "warmstart_probe_centers_samples": np.array(
                [20, 50, 80], dtype=np.int64
            ),
            "warmstart_probe_durations_samples": np.array(
                [10.0, 12.0, 14.0], dtype=np.float32
            ),
        },
        {
            "reference_bp_positions": np.array([4500, 2000, 500], dtype=np.int64),
            "n_ref_probes": 3,
            "direction": -1,
            "warmstart_probe_centers_samples": None,
            "warmstart_probe_durations_samples": None,
        },
    ]
    with open(cache_dir / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    _write_manifest(
        cache_dir,
        [
            {
                "uid": 0,
                "channel": 2,
                "num_samples": 100,
                "num_probes": 3,
                "n_ref_probes": 3,
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
                "n_ref_probes": 3,
                "num_matched_probes": 0,
                "transloc_time_ms": 5.0,
                "mean_lvl1": 0.6,
                "direction": -1,
                "amplitude_scale": 1.0,
            },
        ],
    )

    return cache_dir


def test_cached_dataset_loads(fake_cache_dir):
    ds = CachedMoleculeDataset([fake_cache_dir])
    assert len(ds) == 2


def test_cached_dataset_item_shapes(fake_cache_dir):
    """Verify shapes of tensors returned by __getitem__ (new schema)."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    assert item["waveform"].shape == (1, 100)
    assert item["mask"].shape == (100,)
    assert item["conditioning"].shape == (6,)
    assert item["reference_bp_positions"].shape == (3,)
    assert item["reference_bp_positions"].dtype == torch.long
    assert item["n_ref_probes"].dtype == torch.long
    assert int(item["n_ref_probes"].item()) == 3


def test_cached_dataset_second_molecule(fake_cache_dir):
    """Second molecule has no warmstart; schema fields reflect that."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[1]

    assert item["waveform"].shape == (1, 200)
    assert item["mask"].shape == (200,)
    assert item["warmstart_heatmap"] is None
    assert not bool(item["warmstart_valid"].item())


def test_cached_dataset_warmstart_heatmap_built(fake_cache_dir):
    """When warmstart arrays are cached, a [T] float heatmap is built."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    assert item["warmstart_heatmap"] is not None
    assert item["warmstart_heatmap"].shape == (100,)
    assert item["warmstart_heatmap"].dtype == torch.float32
    # Peaks should be in [0, 1] since build_probe_heatmap uses max-blend.
    assert float(item["warmstart_heatmap"].max().item()) <= 1.0 + 1e-6
    assert float(item["warmstart_heatmap"].max().item()) > 0.5  # has a real peak
    assert bool(item["warmstart_valid"].item())


def test_cached_dataset_waveform_values(fake_cache_dir):
    """Verify waveform is normalized by level-1 amplitude."""
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    # wfm1 = 500 (int16), amplitude_scale=1.0, mean_lvl1=0.5 mV = 500 uV
    # normalized = (500 * 1.0) / 500.0 = 1.0 for all samples
    np.testing.assert_allclose(item["waveform"].numpy(), 1.0, rtol=1e-5)


def test_cached_dataset_molecule_uid(fake_cache_dir):
    ds = CachedMoleculeDataset([fake_cache_dir])
    assert ds[0]["molecule_uid"] == 0
    assert ds[1]["molecule_uid"] == 1


def test_cached_dataset_tensor_types(fake_cache_dir):
    ds = CachedMoleculeDataset([fake_cache_dir])
    item = ds[0]

    assert isinstance(item["waveform"], torch.Tensor)
    assert item["waveform"].dtype == torch.float32
    assert isinstance(item["conditioning"], torch.Tensor)
    assert item["conditioning"].dtype == torch.float32
    assert isinstance(item["mask"], torch.Tensor)
    assert item["mask"].dtype == torch.bool
    assert isinstance(item["reference_bp_positions"], torch.Tensor)
    assert isinstance(item["warmstart_valid"], torch.Tensor)
    assert item["warmstart_valid"].dtype == torch.bool


def test_cached_dataset_multiple_dirs(fake_cache_dir, tmp_path):
    """Verify dataset can merge multiple cache directories."""
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
            "reference_bp_positions": np.array([800, 2000, 3800], dtype=np.int64),
            "n_ref_probes": 3,
            "direction": 1,
            "warmstart_probe_centers_samples": np.array(
                [30, 75, 120], dtype=np.int64
            ),
            "warmstart_probe_durations_samples": np.array(
                [11.0, 13.0, 15.0], dtype=np.float32
            ),
        },
    ]
    with open(cache_dir2 / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    _write_manifest(
        cache_dir2,
        [
            {
                "uid": 10,
                "channel": 1,
                "num_samples": 150,
                "num_probes": 3,
                "n_ref_probes": 3,
                "num_matched_probes": 3,
                "transloc_time_ms": 3.75,
                "mean_lvl1": 0.4,
                "direction": 1,
                "amplitude_scale": 1.0,
            },
        ],
    )

    ds = CachedMoleculeDataset([fake_cache_dir, cache_dir2])
    assert len(ds) == 3
    # Third item should come from second dir
    item = ds[2]
    assert item["waveform"].shape == (1, 150)
    assert item["molecule_uid"] == 10
    assert int(item["n_ref_probes"].item()) == 3


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

    _write_manifest(cache_dir, [])

    ds = CachedMoleculeDataset([cache_dir])
    assert len(ds) == 0
