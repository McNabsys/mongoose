"""Focal loss for probe detection heatmap with extreme class imbalance."""

from __future__ import annotations

import torch


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    mask: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute focal loss for binary classification with soft targets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred: Predicted probabilities [*] (after sigmoid, values in 0-1).
        target: Target heatmap [*] (soft Gaussian targets, values in 0-1).
        gamma: Focusing parameter (default 2.0).
        alpha: Balancing weight for positive class (default 0.25).
        mask: Optional boolean mask [*]; only masked-in positions contribute.
        eps: Small constant for numerical stability.

    Returns:
        Scalar mean focal loss over valid positions.
    """
    pred = pred.clamp(eps, 1.0 - eps)

    # Binary cross entropy components
    bce_pos = -target * torch.log(pred)
    bce_neg = -(1.0 - target) * torch.log(1.0 - pred)

    # p_t: probability of the "correct" class
    p_t = target * pred + (1.0 - target) * (1.0 - pred)
    focal_weight = (1.0 - p_t) ** gamma

    # Alpha weighting: alpha for positives, (1-alpha) for negatives
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)

    loss = alpha_t * focal_weight * (bce_pos + bce_neg)

    if mask is not None:
        mask_f = mask.float()
        loss = loss * mask_f
        return loss.sum() / mask_f.sum().clamp(min=1.0)

    return loss.mean()
