"""Sparse Huber loss on inter-probe base-pair deltas."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_huber_delta_loss(
    pred_cumulative_at_probes: torch.Tensor,
    gt_deltas: torch.Tensor,
    delta: float = 500.0,
) -> torch.Tensor:
    """Huber loss between predicted and ground-truth inter-probe bp deltas.

    Args:
        pred_cumulative_at_probes: Predicted cumulative bp sampled at probe
            indices [N_probes]. Consecutive differences give predicted deltas.
        gt_deltas: Ground-truth inter-probe deltas [N_probes - 1] in base
            pairs.
        delta: Huber loss transition threshold in base pairs (default 500).

    Returns:
        Scalar loss normalized by mean ground-truth delta size.
    """
    pred_deltas = torch.diff(pred_cumulative_at_probes)
    raw_loss = F.huber_loss(pred_deltas, gt_deltas, delta=delta, reduction="mean")
    mean_gt = gt_deltas.mean().clamp(min=1.0)
    return raw_loss / mean_gt
