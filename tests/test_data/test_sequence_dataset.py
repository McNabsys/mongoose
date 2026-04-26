"""Tests for the per-molecule sequence dataset (V5 Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mongoose.data.residual_dataset import FEATURE_DIM
from mongoose.data.sequence_dataset import (
    MoleculeSequenceDataset,
    collate_molecules,
)


def _make_synthetic_table(
    n_molecules: int = 4,
    probes_per_mol: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Tiny synthetic residual table that satisfies the schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_molecules):
        mol_len = 100_000 + u * 20_000
        n = probes_per_mol
        spacing = mol_len // (n + 1)
        for k in range(n):
            pos = (k + 1) * spacing
            prev_iv = -1 if k == 0 else spacing
            next_iv = -1 if k == n - 1 else spacing
            rows.append(
                {
                    "run_id": f"RUN-{u % 2}",
                    "uid": int(u),
                    "channel": int(u % 4 + 1),
                    "molecule_id": int(u),
                    "probe_idx": int(k),
                    "pre_position_bp": int(pos),
                    "width_bp": int(800 + rng.integers(-100, 100)),
                    "attribute": 0x81,
                    "accepted": True,
                    "in_clean_region": True,
                    "in_structure": False,
                    "in_folded_start": False,
                    "in_folded_end": False,
                    "excluded_amplitude": False,
                    "excluded_width_sp": False,
                    "excluded_width_rm": False,
                    "outside_partial_region": False,
                    "prev_interval_bp": int(prev_iv),
                    "next_interval_bp": int(next_iv),
                    "frac_position_in_molecule": float(k) / max(n - 1, 1),
                    "bp_position_frac": float(pos) / mol_len,
                    "molecule_length_bp": int(mol_len),
                    "num_probes": int(n),
                    "num_probes_accepted": int(n),
                    "length_group_bin": int(min(15, max(-1, (mol_len - 75_000) // 10_000))),
                    "is_clean_molecule": True,
                    "direction": 1,
                    "ref_probe_idx_1based": int(k + 1),
                    "reference_position_bp": int(pos + 100 * (k + 1)),
                    "has_reference": True,
                    "post_position_bp": int(pos + 100 * (k + 1)),
                    "residual_bp": int(100 * (k + 1)),
                    "ref_anchored_residual_bp": int(50 * k),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# MoleculeSequenceDataset
# --------------------------------------------------------------------------

def test_dataset_one_item_per_molecule():
    df = _make_synthetic_table(n_molecules=4, probes_per_mol=5)
    ds = MoleculeSequenceDataset(df)
    assert len(ds) == 4


def test_dataset_item_shapes_match_probe_count():
    df = _make_synthetic_table(n_molecules=3, probes_per_mol=7)
    ds = MoleculeSequenceDataset(df)
    item = ds[0]
    assert item["features"].shape == (7, FEATURE_DIM)
    assert item["target"].shape == (7,)
    assert item["length"] == 7


def test_dataset_skips_below_min_probes():
    df = _make_synthetic_table(n_molecules=3, probes_per_mol=1)
    ds = MoleculeSequenceDataset(df, min_probes=2)
    assert len(ds) == 0


def test_dataset_drops_above_max_probes():
    df = _make_synthetic_table(n_molecules=3, probes_per_mol=20)
    ds = MoleculeSequenceDataset(df, max_probes=10)
    assert len(ds) == 0


def test_dataset_target_column_must_exist():
    df = _make_synthetic_table()
    with pytest.raises(ValueError, match="not in df"):
        MoleculeSequenceDataset(df, target_column="bogus_column")


def test_dataset_per_probe_ordering_is_temporal():
    """Probes within a molecule should be returned in probe_idx order."""
    df = _make_synthetic_table(n_molecules=1, probes_per_mol=5)
    # Shuffle the rows to ensure the dataset re-sorts.
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ds = MoleculeSequenceDataset(df)
    item = ds[0]
    # Targets in ref_anchored_residual_bp are 0, 50, 100, 150, 200 by
    # construction; if order is preserved we should see them ascending.
    target = item["target"].numpy()
    assert (np.diff(target) >= 0).all()


def test_dataset_uid_and_run_id_index_populated():
    df = _make_synthetic_table(n_molecules=4)
    ds = MoleculeSequenceDataset(df)
    uids = sorted([ds[i]["uid"] for i in range(len(ds))])
    assert uids == [0, 1, 2, 3]
    rids = sorted({ds[i]["run_id_index"] for i in range(len(ds))})
    # Two distinct run ids in the synthetic table -> codes {0, 1}.
    assert rids == [0, 1]


# --------------------------------------------------------------------------
# collate_molecules
# --------------------------------------------------------------------------

def test_collate_pads_to_max_K_in_batch():
    df = _make_synthetic_table(n_molecules=4, probes_per_mol=3)
    # Modify one molecule to have 7 probes by adding extra rows
    df_extra = pd.DataFrame([
        {**df.iloc[0].to_dict(), "probe_idx": idx}
        for idx in range(3, 7)
    ])
    df = pd.concat([df, df_extra], ignore_index=True)
    ds = MoleculeSequenceDataset(df)
    batch = [ds[i] for i in range(len(ds))]
    out = collate_molecules(batch)
    K_max = max(item["length"] for item in batch)
    assert out["features"].shape == (len(batch), K_max, FEATURE_DIM)
    assert out["target"].shape == (len(batch), K_max)
    assert out["padding_mask"].shape == (len(batch), K_max)


def test_collate_padding_mask_marks_real_probes():
    df = _make_synthetic_table(n_molecules=2, probes_per_mol=5)
    ds = MoleculeSequenceDataset(df)
    batch = [ds[0], ds[1]]
    out = collate_molecules(batch)
    # Both have 5 probes; entire mask should be False (no padding).
    assert (~out["padding_mask"]).all()


def test_collate_zero_pads_features_and_targets():
    df_a = _make_synthetic_table(n_molecules=1, probes_per_mol=3)
    df_b = _make_synthetic_table(n_molecules=1, probes_per_mol=6, seed=1)
    df_b["uid"] = df_b["uid"] + 100  # avoid uid collision
    df = pd.concat([df_a, df_b], ignore_index=True)
    ds = MoleculeSequenceDataset(df)
    short, long = (ds[0], ds[1]) if ds[0]["length"] == 3 else (ds[1], ds[0])
    out = collate_molecules([short, long])
    short_idx = 0 if ds[0]["length"] == 3 else 1
    # Short molecule has 3 real probes; positions 3..5 should be zero
    # and padding_mask=True there.
    assert (out["features"][short_idx, 3:] == 0).all()
    assert (out["target"][short_idx, 3:] == 0).all()
    assert out["padding_mask"][short_idx, 3:].all()
    assert (~out["padding_mask"][short_idx, :3]).all()


def test_collate_lengths_match_input():
    df = _make_synthetic_table(n_molecules=3, probes_per_mol=4)
    ds = MoleculeSequenceDataset(df)
    batch = [ds[i] for i in range(len(ds))]
    out = collate_molecules(batch)
    assert (out["lengths"].tolist() == [item["length"] for item in batch])


def test_collate_compatible_with_dataloader():
    df = _make_synthetic_table(n_molecules=8, probes_per_mol=5)
    ds = MoleculeSequenceDataset(df)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, collate_fn=collate_molecules,
    )
    batch = next(iter(loader))
    assert batch["features"].shape[0] == 4
    assert batch["target"].shape[0] == 4
    assert batch["padding_mask"].shape[0] == 4


def test_collate_empty_batch_raises():
    with pytest.raises(ValueError, match="empty batch"):
        collate_molecules([])
