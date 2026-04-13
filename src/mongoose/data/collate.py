"""Batch collation for variable-length molecule data."""

from __future__ import annotations

import math

import torch


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the next multiple."""
    return int(math.ceil(value / multiple)) * multiple


def collate_molecules(items: list[dict]) -> dict:
    """Pad all tensors to max length in batch, rounded up to multiple of 32.

    Waveform, probe_heatmap, and mask are padded to the same target length.
    probe_sample_indices, gt_deltas_bp, and velocity_targets are stored as
    lists (variable length per molecule, not padded).
    conditioning is stacked into a [B, 6] tensor.
    molecule_uid is collected into a list.

    Args:
        items: List of dicts from MoleculeDataset or SyntheticMoleculeDataset.

    Returns:
        Batched dict with padded tensors and list fields.
    """
    batch_size = len(items)

    # Find max temporal length in batch
    max_len = max(item["waveform"].shape[-1] for item in items)
    padded_len = _round_up_to_multiple(max_len, 32)

    # Pre-allocate padded tensors
    waveforms = torch.zeros(batch_size, 1, padded_len)
    probe_heatmaps = torch.zeros(batch_size, padded_len)
    masks = torch.zeros(batch_size, padded_len, dtype=torch.bool)

    for i, item in enumerate(items):
        t = item["waveform"].shape[-1]
        waveforms[i, :, :t] = item["waveform"]
        probe_heatmaps[i, :t] = item["probe_heatmap"]
        masks[i, :t] = item["mask"]

    # Stack conditioning [B, 6]
    conditioning = torch.stack([item["conditioning"] for item in items])

    return {
        "waveform": waveforms,
        "conditioning": conditioning,
        "probe_heatmap": probe_heatmaps,
        "probe_sample_indices": [item["probe_sample_indices"] for item in items],
        "gt_deltas_bp": [item["gt_deltas_bp"] for item in items],
        "velocity_targets": [item["velocity_targets"] for item in items],
        "mask": masks,
        "molecule_uid": [item["molecule_uid"] for item in items],
    }
