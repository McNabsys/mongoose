"""Tests for dataset and collation (V1 rearchitecture schema)."""

import torch

from mongoose.data.collate import collate_molecules
from mongoose.data.dataset import SyntheticMoleculeDataset


def _make_item(
    length: int,
    n_probes: int = 4,
    molecule_uid: int = 0,
    with_warmstart: bool = True,
):
    """Construct a minimal item in the new schema for collate tests."""
    reference_bp_positions = torch.arange(
        0, n_probes * 1000, 1000, dtype=torch.long
    )[:n_probes]
    item = {
        "waveform": torch.randn(1, length),
        "mask": torch.ones(length, dtype=torch.bool),
        "conditioning": torch.randn(6),
        "reference_bp_positions": reference_bp_positions,
        "n_ref_probes": torch.tensor(n_probes, dtype=torch.long),
        "warmstart_heatmap": torch.zeros(length) if with_warmstart else None,
        "warmstart_valid": torch.tensor(with_warmstart, dtype=torch.bool),
        "warmstart_probe_centers_samples": (
            torch.arange(n_probes, dtype=torch.long) * (length // max(n_probes, 1))
            if with_warmstart else None
        ),
        "molecule_uid": molecule_uid,
    }
    return item


def test_collate_pads_to_multiple_of_32():
    items = [_make_item(100, molecule_uid=0), _make_item(200, molecule_uid=1)]
    batch = collate_molecules(items)
    # Padded to 224 (next multiple of 32 above 200)
    assert batch["waveform"].shape == (2, 1, 224)
    assert batch["mask"].shape == (2, 224)
    assert batch["warmstart_heatmap"].shape == (2, 224)
    # Shorter molecule is masked in padding region
    assert batch["mask"][0, 100:].sum() == 0
    # reference_bp_positions is a list of variable-length tensors
    assert isinstance(batch["reference_bp_positions"], list)
    assert len(batch["reference_bp_positions"]) == 2


def test_collate_conditioning_stacked():
    items = [_make_item(64, molecule_uid=0)]
    batch = collate_molecules(items)
    assert batch["conditioning"].shape == (1, 6)
    assert batch["molecule_uid"] == [0]
    assert batch["n_ref_probes"].shape == (1,)
    assert batch["warmstart_valid"].shape == (1,)


def test_collate_mixed_warmstart_batch_collapses_to_none():
    """If any item lacks warmstart, whole batch goes None (all-or-nothing)."""
    items = [
        _make_item(128, molecule_uid=0, with_warmstart=True),
        _make_item(128, molecule_uid=1, with_warmstart=False),
    ]
    batch = collate_molecules(items)
    assert batch["warmstart_heatmap"] is None
    # warmstart_valid is forced all-False when heatmap is None.
    assert batch["warmstart_valid"].dtype == torch.bool
    assert not batch["warmstart_valid"].any()


def test_collate_all_warmstart_batch_stacks():
    items = [
        _make_item(64, molecule_uid=0, with_warmstart=True),
        _make_item(64, molecule_uid=1, with_warmstart=True),
    ]
    batch = collate_molecules(items)
    assert batch["warmstart_heatmap"] is not None
    assert batch["warmstart_heatmap"].shape == (2, 64)
    assert batch["warmstart_valid"].all()


def test_collate_no_warmstart_batch_all_none():
    items = [
        _make_item(64, molecule_uid=0, with_warmstart=False),
        _make_item(64, molecule_uid=1, with_warmstart=False),
    ]
    batch = collate_molecules(items)
    assert batch["warmstart_heatmap"] is None
    assert not batch["warmstart_valid"].any()


def test_synthetic_dataset():
    ds = SyntheticMoleculeDataset(
        num_molecules=10, min_length=500, max_length=2000
    )
    item = ds[0]
    assert "waveform" in item
    assert "mask" in item
    assert "conditioning" in item
    assert "reference_bp_positions" in item
    assert "n_ref_probes" in item
    assert "warmstart_valid" in item
    assert item["waveform"].shape[0] == 1  # [1, T]
    assert item["mask"].all()  # no padding within individual items


def test_synthetic_dataset_has_reference_probes():
    ds = SyntheticMoleculeDataset(
        num_molecules=5, min_length=1000, max_length=2000, seed=123
    )
    item = ds[0]
    n_probes = int(item["n_ref_probes"].item())
    assert n_probes >= 2
    assert item["reference_bp_positions"].shape == (n_probes,)
    assert item["reference_bp_positions"].dtype == torch.long
    # Ascending order (direction=1 semantics).
    diffs = torch.diff(item["reference_bp_positions"])
    assert (diffs > 0).all()


def test_synthetic_dataset_warmstart_mix():
    """With probability=0.5 we expect both warmstart and non-warmstart items."""
    ds = SyntheticMoleculeDataset(
        num_molecules=32,
        min_length=500,
        max_length=800,
        warmstart_probability=0.5,
        seed=7,
    )
    with_ws = 0
    without_ws = 0
    for i in range(len(ds)):
        item = ds[i]
        if item["warmstart_heatmap"] is None:
            assert not bool(item["warmstart_valid"].item())
            without_ws += 1
        else:
            assert item["warmstart_heatmap"].shape == (item["waveform"].shape[-1],)
            assert bool(item["warmstart_valid"].item())
            with_ws += 1
    assert with_ws > 0
    assert without_ws > 0


def test_synthetic_dataset_length():
    ds = SyntheticMoleculeDataset(num_molecules=7)
    assert len(ds) == 7


def test_collate_already_multiple_of_32():
    """When max length is already a multiple of 32, no extra padding needed."""
    items = [_make_item(128, molecule_uid=5, with_warmstart=True)]
    batch = collate_molecules(items)
    assert batch["waveform"].shape == (1, 1, 128)
    assert batch["warmstart_heatmap"].shape == (1, 128)
