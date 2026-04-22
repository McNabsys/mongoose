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
        # Per-molecule z-score on the valid-mask region only. Padding stays at 0.
        # Rationale: post-preprocess waveform amplitudes come out at ~1e-4
        # (unit scaling in preprocess is suspect), which is at the BF16
        # precision floor and incompatible with Kaiming/Xavier N(0,1) init
        # assumptions. Normalizing here is cheaper than rebuilding 1.25M
        # cached molecules and is robust to whatever units the raw waveform
        # ends up in.
        valid = waveforms[i, 0, :t]
        std = valid.std().clamp(min=1e-8)
        waveforms[i, 0, :t] = (valid - valid.mean()) / std

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

    # Per-molecule center lists have variable lengths, so keep as a list
    # (same pattern as reference_bp_positions). Each element is a
    # LongTensor or None.
    warmstart_probe_centers_samples = [
        item["warmstart_probe_centers_samples"] for item in items
    ]

    # Per-molecule T2D params (Option A hybrid). All-or-nothing: if any item
    # in the batch lacks t2d_params, the batch-level ``t2d_params`` is None
    # so the trainer can keep the whole batch in standard (non-hybrid) mode.
    all_have_t2d = all(item.get("t2d_params") is not None for item in items)
    t2d_params: torch.Tensor | None
    if all_have_t2d:
        t2d_params = torch.stack([item["t2d_params"] for item in items])  # [B, 3]
    else:
        t2d_params = None

    return {
        "waveform": waveforms,
        "conditioning": conditioning,
        "mask": masks,
        "reference_bp_positions": reference_bp_positions,
        "n_ref_probes": n_ref_probes,
        "warmstart_heatmap": warmstart_heatmaps,
        "warmstart_valid": warmstart_valid,
        "warmstart_probe_centers_samples": warmstart_probe_centers_samples,
        "molecule_uid": [item["molecule_uid"] for item in items],
        "t2d_params": t2d_params,  # [B, 3] or None
    }
