"""Batch collation for variable-length molecule data.

V1 rearchitecture (R6): the item schema now carries reference bp
positions, a reference probe count, and an optional pre-built warmstart
heatmap. The collator pads temporal tensors to a multiple of 32 and
applies an all-or-nothing policy for warmstart -- if any item in the
batch lacks a warmstart heatmap, the entire batch's
``warmstart_heatmap`` is ``None`` and ``warmstart_valid`` is all-False.
This keeps the loss path simple (post-warmstart mode) and avoids
per-molecule gating inside the batch loop.
"""

from __future__ import annotations

import math

import torch


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the next multiple."""
    return int(math.ceil(value / multiple)) * multiple


def collate_molecules(items: list[dict]) -> dict:
    """Pad temporal tensors to max length rounded up to a multiple of 32.

    Temporal tensors padded: ``waveform``, ``mask``, and ``warmstart_heatmap``
    (when present on all items). ``conditioning`` is stacked to ``[B, 6]``,
    ``reference_bp_positions`` is kept as a list of variable-length tensors,
    and ``n_ref_probes`` / ``warmstart_valid`` are stacked to ``[B]``.

    Mixed-batch warmstart policy: if any item has ``warmstart_heatmap ==
    None``, the batch's ``warmstart_heatmap`` is ``None`` and
    ``warmstart_valid`` is all-False. This forces the whole batch into
    post-warmstart mode rather than trying to gate per-molecule.

    Args:
        items: List of dicts from a mongoose dataset (new V1 schema).

    Returns:
        Batched dict with padded tensors and list / scalar batch fields.
    """
    batch_size = len(items)

    # Find max temporal length in batch and round up to multiple of 32.
    max_len = max(item["waveform"].shape[-1] for item in items)
    padded_len = _round_up_to_multiple(max_len, 32)

    waveforms = torch.zeros(batch_size, 1, padded_len)
    masks = torch.zeros(batch_size, padded_len, dtype=torch.bool)

    # All-or-nothing warmstart: only build a batch heatmap when every
    # item has one. Otherwise the batch is treated as post-warmstart.
    all_have_warmstart = all(
        item.get("warmstart_heatmap") is not None for item in items
    )
    if all_have_warmstart:
        warmstart_heatmaps: torch.Tensor | None = torch.zeros(batch_size, padded_len)
    else:
        warmstart_heatmaps = None

    for i, item in enumerate(items):
        t = item["waveform"].shape[-1]
        waveforms[i, :, :t] = item["waveform"]
        masks[i, :t] = item["mask"]
        if all_have_warmstart:
            warmstart_heatmaps[i, :t] = item["warmstart_heatmap"]  # type: ignore[index]

    conditioning = torch.stack([item["conditioning"] for item in items])

    # Stack scalar batch fields.
    n_ref_probes = torch.stack(
        [torch.as_tensor(item["n_ref_probes"], dtype=torch.long) for item in items]
    )

    if all_have_warmstart:
        warmstart_valid = torch.stack(
            [
                torch.as_tensor(item["warmstart_valid"], dtype=torch.bool)
                for item in items
            ]
        )
    else:
        # Force all-False when the batch is heterogeneous.
        warmstart_valid = torch.zeros(batch_size, dtype=torch.bool)

    reference_bp_positions = [
        item["reference_bp_positions"] for item in items
    ]

    return {
        "waveform": waveforms,
        "conditioning": conditioning,
        "mask": masks,
        "reference_bp_positions": reference_bp_positions,
        "n_ref_probes": n_ref_probes,
        "warmstart_heatmap": warmstart_heatmaps,
        "warmstart_valid": warmstart_valid,
        "molecule_uid": [item["molecule_uid"] for item in items],
    }
