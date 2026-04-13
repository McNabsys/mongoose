"""Tests for dataset and collation."""

import torch

from mongoose.data.collate import collate_molecules
from mongoose.data.dataset import SyntheticMoleculeDataset


def test_collate_pads_to_multiple_of_32():
    items = [
        {
            "waveform": torch.randn(1, 100),
            "probe_heatmap": torch.zeros(100),
            "mask": torch.ones(100, dtype=torch.bool),
            "conditioning": torch.randn(6),
            "probe_sample_indices": torch.tensor([10, 50, 90]),
            "gt_deltas_bp": torch.tensor([1000.0, 2000.0]),
            "velocity_targets": torch.tensor([0.5, 0.3, 0.2]),
            "molecule_uid": 0,
        },
        {
            "waveform": torch.randn(1, 200),
            "probe_heatmap": torch.zeros(200),
            "mask": torch.ones(200, dtype=torch.bool),
            "conditioning": torch.randn(6),
            "probe_sample_indices": torch.tensor([20, 100, 180]),
            "gt_deltas_bp": torch.tensor([1500.0, 2500.0]),
            "velocity_targets": torch.tensor([0.4, 0.3, 0.2]),
            "molecule_uid": 1,
        },
    ]
    batch = collate_molecules(items)
    # Padded to 224 (next multiple of 32 above 200)
    assert batch["waveform"].shape == (2, 1, 224)
    assert batch["mask"].shape == (2, 224)
    assert batch["probe_heatmap"].shape == (2, 224)
    # Shorter molecule is masked in padding region
    assert batch["mask"][0, 100:].sum() == 0
    # Variable-length fields are lists
    assert isinstance(batch["probe_sample_indices"], list)
    assert len(batch["probe_sample_indices"]) == 2


def test_collate_conditioning_stacked():
    items = [
        {
            "waveform": torch.randn(1, 64),
            "probe_heatmap": torch.zeros(64),
            "mask": torch.ones(64, dtype=torch.bool),
            "conditioning": torch.randn(6),
            "probe_sample_indices": torch.tensor([10, 30, 50]),
            "gt_deltas_bp": torch.tensor([100.0, 200.0]),
            "velocity_targets": torch.tensor([0.5, 0.3, 0.2]),
            "molecule_uid": 0,
        },
    ]
    batch = collate_molecules(items)
    # Conditioning should be stacked into a [B, 6] tensor
    assert batch["conditioning"].shape == (1, 6)
    # molecule_uid should be a list
    assert batch["molecule_uid"] == [0]


def test_synthetic_dataset():
    ds = SyntheticMoleculeDataset(
        num_molecules=10, min_length=500, max_length=2000
    )
    item = ds[0]
    assert "waveform" in item
    assert "probe_heatmap" in item
    assert "mask" in item
    assert "conditioning" in item
    assert item["waveform"].shape[0] == 1  # [1, T]
    assert item["mask"].all()  # no padding within individual items


def test_synthetic_dataset_has_probes():
    ds = SyntheticMoleculeDataset(
        num_molecules=5, min_length=1000, max_length=2000, seed=123
    )
    item = ds[0]
    assert "probe_sample_indices" in item
    assert "gt_deltas_bp" in item
    assert "velocity_targets" in item
    n_probes = item["probe_sample_indices"].shape[0]
    assert n_probes >= 2
    assert item["gt_deltas_bp"].shape[0] == n_probes - 1
    assert item["velocity_targets"].shape[0] == n_probes


def test_synthetic_dataset_length():
    ds = SyntheticMoleculeDataset(num_molecules=7)
    assert len(ds) == 7


def test_collate_already_multiple_of_32():
    """When max length is already a multiple of 32, no extra padding needed."""
    items = [
        {
            "waveform": torch.randn(1, 128),
            "probe_heatmap": torch.zeros(128),
            "mask": torch.ones(128, dtype=torch.bool),
            "conditioning": torch.randn(6),
            "probe_sample_indices": torch.tensor([10, 64, 120]),
            "gt_deltas_bp": torch.tensor([500.0, 700.0]),
            "velocity_targets": torch.tensor([0.4, 0.3, 0.2]),
            "molecule_uid": 5,
        },
    ]
    batch = collate_molecules(items)
    assert batch["waveform"].shape == (1, 1, 128)
