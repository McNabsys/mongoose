"""Tests for the per-probe residual dataset (Direction C, phase C.3)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from mongoose.data.residual_dataset import (
    FEATURE_DIM,
    PROBE_WIDTH_NORM_BP,
    ResidualDataset,
    add_molecule_aggregates,
    extract_features,
    make_split,
)


def _make_synthetic_table(
    n_molecules: int = 3,
    probes_per_mol: int = 4,
    accepted_rate: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Construct a tiny residual table that satisfies the schema contract."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_molecules):
        mol_len = 100_000 + u * 20_000
        n = probes_per_mol
        spacing = mol_len // (n + 1)
        n_accepted = int(round(n * accepted_rate))
        for k in range(n):
            pos = (k + 1) * spacing
            prev_iv = -1 if k == 0 else spacing
            next_iv = -1 if k == n - 1 else spacing
            accepted = k < n_accepted
            rows.append(
                {
                    "run_id": f"RUN-{u % 2}",  # two distinct run ids
                    "uid": int(u),
                    "channel": int(u % 4 + 1),
                    "molecule_id": int(u),
                    "probe_idx": int(k),
                    "pre_position_bp": int(pos),
                    "width_bp": int(800 + rng.integers(-100, 100)),
                    "attribute": 0x81 if accepted else 0x01,
                    "accepted": bool(accepted),
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
                    "num_probes_accepted": int(n_accepted),
                    "length_group_bin": int(min(15, max(-1, (mol_len - 75_000) // 10_000))),
                    "is_clean_molecule": True,
                    "direction": 1,
                    "ref_probe_idx_1based": 0,
                    "reference_position_bp": -1,
                    "has_reference": False,
                    "post_position_bp": int(pos + (k + 1) * 100),  # production added
                    "residual_bp": int((k + 1) * 100),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# add_molecule_aggregates
# --------------------------------------------------------------------------

def test_add_molecule_aggregates_adds_two_columns():
    df = _make_synthetic_table()
    out = add_molecule_aggregates(df)
    assert "mean_probe_width_bp" in out.columns
    assert "median_probe_width_bp" in out.columns
    # Original df is not mutated.
    assert "mean_probe_width_bp" not in df.columns


def test_add_molecule_aggregates_constant_per_molecule():
    df = _make_synthetic_table()
    out = add_molecule_aggregates(df)
    for uid, group in out.groupby("uid"):
        assert group["mean_probe_width_bp"].nunique() == 1
        assert group["median_probe_width_bp"].nunique() == 1


def test_add_molecule_aggregates_handles_empty_df():
    df = pd.DataFrame(
        columns=[
            "uid",
            "width_bp",
            "residual_bp",
            "accepted",
        ]
    )
    out = add_molecule_aggregates(df)
    assert "mean_probe_width_bp" in out.columns
    assert len(out) == 0


# --------------------------------------------------------------------------
# extract_features
# --------------------------------------------------------------------------

def test_extract_features_shape_and_dtype():
    df = add_molecule_aggregates(_make_synthetic_table())
    feats = extract_features(df)
    assert feats.shape == (len(df), FEATURE_DIM)
    assert feats.dtype == np.float32


def test_extract_features_normalizes_width_by_831():
    df = _make_synthetic_table(n_molecules=1, probes_per_mol=1)
    df = add_molecule_aggregates(df)
    df.loc[0, "width_bp"] = int(PROBE_WIDTH_NORM_BP)
    feats = extract_features(df)
    # Feature index 4 = normalized_width.
    assert feats[0, 4] == pytest.approx(1.0, abs=1e-6)


def test_extract_features_first_probe_no_prev_interval():
    df = add_molecule_aggregates(_make_synthetic_table())
    feats = extract_features(df)
    first_probe = df["probe_idx"] == 0
    # Feature index 5 = has_prev_interval.
    assert (feats[first_probe.to_numpy(), 5] == 0.0).all()


def test_extract_features_direction_one_hot():
    df = _make_synthetic_table(n_molecules=2)  # even row count by construction
    df = add_molecule_aggregates(df)
    assert len(df) % 2 == 0, "test fixture must have even row count"
    df["direction"] = [1, -1] * (len(df) // 2)
    feats = extract_features(df)
    forward = df["direction"].to_numpy() == 1
    reverse = df["direction"].to_numpy() == -1
    # Feature 18 = forward, 19 = reverse, 20 = unknown.
    assert (feats[forward, 18] == 1.0).all()
    assert (feats[forward, 19] == 0.0).all()
    assert (feats[reverse, 18] == 0.0).all()
    assert (feats[reverse, 19] == 1.0).all()
    assert (feats[:, 20] == 0.0).all()


def test_extract_features_finite():
    df = add_molecule_aggregates(_make_synthetic_table(n_molecules=5, probes_per_mol=10))
    feats = extract_features(df)
    assert np.isfinite(feats).all()


# --------------------------------------------------------------------------
# ResidualDataset
# --------------------------------------------------------------------------

def test_dataset_filters_accepted_only_by_default():
    df = _make_synthetic_table(n_molecules=2, probes_per_mol=4, accepted_rate=0.5)
    ds = ResidualDataset(df)
    # Two molecules x 4 probes x 50% accepted = 4 samples.
    assert len(ds) == 4


def test_dataset_includes_rejected_when_flag_off():
    df = _make_synthetic_table(n_molecules=2, probes_per_mol=4, accepted_rate=0.5)
    ds = ResidualDataset(df, accepted_only=False)
    assert len(ds) == 8


def test_dataset_getitem_shapes_and_dtypes():
    df = _make_synthetic_table()
    ds = ResidualDataset(df)
    feats, target = ds[0]
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == torch.float32
    assert target.shape == ()
    assert target.dtype == torch.float32


def test_dataset_getitem_target_matches_residual_bp():
    df = _make_synthetic_table()
    ds = ResidualDataset(df)
    df_acc = df[df["accepted"]].reset_index(drop=True)
    for i in range(min(5, len(ds))):
        _, target = ds[i]
        assert float(target.item()) == float(df_acc.loc[i, "residual_bp"])


def test_dataset_run_id_index_factorizes():
    df = _make_synthetic_table(n_molecules=4)
    ds = ResidualDataset(df)
    # Two distinct run ids -> values in {0, 1}.
    assert set(ds.run_id_index.unique().tolist()).issubset({0, 1})


def test_dataset_compatible_with_dataloader():
    df = _make_synthetic_table(n_molecules=3, probes_per_mol=5)
    ds = ResidualDataset(df)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    feats, target = next(iter(loader))
    assert feats.shape == (4, FEATURE_DIM)
    assert target.shape == (4,)


def test_dataset_raises_when_aggregates_missing_and_compute_off():
    df = _make_synthetic_table()
    with pytest.raises(ValueError, match="missing aggregate columns"):
        ResidualDataset(df, compute_aggregates=False)


# --------------------------------------------------------------------------
# make_split
# --------------------------------------------------------------------------

def test_make_split_indices_partition():
    split = make_split(100, val_fraction=0.2, seed=0)
    combined = np.sort(np.concatenate([split.train_indices, split.val_indices]))
    assert (combined == np.arange(100)).all()


def test_make_split_train_count_and_val_count():
    split = make_split(100, val_fraction=0.2, seed=0)
    assert len(split.train_indices) == 80
    assert len(split.val_indices) == 20


def test_make_split_deterministic_with_same_seed():
    s1 = make_split(50, val_fraction=0.3, seed=42)
    s2 = make_split(50, val_fraction=0.3, seed=42)
    assert (s1.train_indices == s2.train_indices).all()
    assert (s1.val_indices == s2.val_indices).all()


def test_make_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        make_split(10, val_fraction=1.0)
    with pytest.raises(ValueError, match="val_fraction"):
        make_split(10, val_fraction=-0.1)
