"""Per-molecule sequence dataset for V5 Phase 2 (sequence model).

Where :class:`mongoose.data.residual_dataset.ResidualDataset` yields one
sample per probe, this dataset yields one sample per *molecule* — each
sample is a variable-length sequence of probes from the same molecule.
The downstream sequence model (transformer encoder) processes the
entire molecule's probe sequence with self-attention, allowing each
probe's prediction to depend on all other probes in the molecule.

Schema per item:

* ``features``: ``[K, FEATURE_DIM]`` float tensor (K = molecule's accepted+matched probe count)
* ``target``: ``[K]`` float tensor (per-probe regression target)
* ``length``: int (= K, used by collate to build padding mask)
* ``uid``: int (molecule UID, for stratified eval)
* ``run_id_index``: int (factorized run id)

The :func:`collate_molecules` function pads variable-length sequences
to ``max_K`` within each batch and emits a key-padding mask the
transformer's ``src_key_padding_mask`` argument expects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mongoose.data.residual_dataset import (
    FEATURE_DIM,
    add_molecule_aggregates,
    extract_features,
)


@dataclass
class _MoleculeRecord:
    """Internal per-molecule storage."""

    features: torch.Tensor  # [K, FEATURE_DIM]
    target: torch.Tensor  # [K]
    uid: int
    run_id_index: int


class MoleculeSequenceDataset(Dataset):
    """Per-molecule sequence dataset.

    Each ``__getitem__`` returns a dict containing the molecule's full
    probe sequence (variable K). Use :func:`collate_molecules` as the
    DataLoader collate function to pad sequences within a batch.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        accepted_only: bool = True,
        require_reference: bool = True,
        target_column: str = "ref_anchored_residual_bp",
        compute_aggregates: bool = True,
        min_probes: int = 2,
        max_probes: int | None = None,
    ) -> None:
        """
        Args:
            df: Residual table from
                :func:`mongoose.etl.reads_maps_table.build_residual_table`.
            accepted_only: When True (default), drop probes with
                ``accepted == False``.
            require_reference: When True (default), drop probes with
                ``has_reference == False``. Required when
                ``target_column == "ref_anchored_residual_bp"`` (target
                undefined for unmatched probes).
            target_column: Name of the regression-target column.
            compute_aggregates: When True (default), call
                :func:`add_molecule_aggregates` to populate the
                per-molecule width-distribution columns the feature
                extractor uses. Pass False if df already has them.
            min_probes: Skip molecules with fewer than this many
                probes after filtering. The transformer needs at
                least 2 probes to form an interval.
            max_probes: When set, drop molecules with more than this
                many probes (helps cap memory in pathological cases).
                None means no cap.
        """
        if compute_aggregates and "mean_probe_width_bp" not in df.columns:
            df = add_molecule_aggregates(df)
        elif "mean_probe_width_bp" not in df.columns:
            raise ValueError(
                "df missing aggregate columns; call add_molecule_aggregates "
                "or pass compute_aggregates=True."
            )

        if accepted_only:
            df = df[df["accepted"]]
        if require_reference:
            df = df[df["has_reference"]]

        if target_column not in df.columns:
            raise ValueError(
                f"target_column={target_column!r} not in df. "
                f"Available columns: {sorted(df.columns)}"
            )

        # Sort by (run_id, uid, probe_idx) so each molecule's probes are
        # in temporal order in the per-molecule slice.
        df = df.sort_values(["run_id", "uid", "probe_idx"]).reset_index(drop=True)

        # Build per-molecule records. Use vectorized feature extraction
        # over the full df, then slice per molecule via group offsets.
        feats = extract_features(df)
        target = df[target_column].to_numpy(dtype=np.float32)
        uids = df["uid"].to_numpy(dtype=np.int64)

        if "run_id" in df.columns:
            codes, _ = pd.factorize(df["run_id"])
            run_id_index = codes.astype(np.int64)
        else:
            run_id_index = np.zeros(len(df), dtype=np.int64)

        # Group offsets via groupby.size().cumsum() trick.
        groups = df.groupby(["run_id", "uid"], sort=False).size().to_numpy()
        offsets = np.concatenate([[0], np.cumsum(groups)])
        starts = offsets[:-1]
        ends = offsets[1:]

        molecules: list[_MoleculeRecord] = []
        feats_t = torch.from_numpy(feats)
        target_t = torch.from_numpy(target)
        for s, e in zip(starts, ends):
            k = int(e - s)
            if k < min_probes:
                continue
            if max_probes is not None and k > max_probes:
                continue
            molecules.append(_MoleculeRecord(
                features=feats_t[s:e].clone(),
                target=target_t[s:e].clone(),
                uid=int(uids[s]),
                run_id_index=int(run_id_index[s]),
            ))

        self.molecules = molecules

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> dict:
        rec = self.molecules[idx]
        return {
            "features": rec.features,  # [K, FEATURE_DIM]
            "target": rec.target,  # [K]
            "length": int(rec.features.shape[0]),
            "uid": rec.uid,
            "run_id_index": rec.run_id_index,
        }


def collate_molecules(batch: list[dict]) -> dict:
    """Pad variable-length molecules within a batch.

    Returns a dict with:
        features: [B, K_max, FEATURE_DIM] float32
        target: [B, K_max] float32 (zero-padded; mask must be applied
                in loss computation)
        padding_mask: [B, K_max] bool (True where padded -- the
                convention ``nn.TransformerEncoder``'s
                ``src_key_padding_mask`` expects)
        lengths: [B] int64 (real probe count per molecule)
        uids: [B] int64
        run_id_indices: [B] int64
    """
    if not batch:
        raise ValueError("empty batch")

    B = len(batch)
    K_max = max(item["length"] for item in batch)
    D = batch[0]["features"].shape[1]
    dtype = batch[0]["features"].dtype

    features = torch.zeros((B, K_max, D), dtype=dtype)
    target = torch.zeros((B, K_max), dtype=batch[0]["target"].dtype)
    padding_mask = torch.ones((B, K_max), dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.int64)
    uids = torch.zeros(B, dtype=torch.int64)
    run_id_indices = torch.zeros(B, dtype=torch.int64)

    for i, item in enumerate(batch):
        K = item["length"]
        features[i, :K] = item["features"]
        target[i, :K] = item["target"]
        padding_mask[i, :K] = False  # real probes -> not padding
        lengths[i] = K
        uids[i] = item["uid"]
        run_id_indices[i] = item["run_id_index"]

    return {
        "features": features,
        "target": target,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "uids": uids,
        "run_id_indices": run_id_indices,
    }
