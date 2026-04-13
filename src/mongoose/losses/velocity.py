"""Sparse L2 loss on velocity at probe positions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_velocity_loss(
    pred_velocity_at_probes: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between predicted and target velocity at probe positions.

    Args:
        pred_velocity_at_probes: Predicted velocity (bp/sample) at probe
            indices [N_probes].
        target_velocity: Target velocity (bp/sample) at probe indices
            [N_probes]. Computed as 12.775 / duration_ms for each tag.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred_velocity_at_probes, target_velocity)
