"""Count loss and peakiness regularizer for self-supervised probe detection.

Used in V1 rearchitecture after the warmstart phase ends. L_probe no longer
receives wfmproc Gaussian supervision; these regularizers provide weak
constraints that guide the heatmap toward producing the right number of
localized peaks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def count_loss(
    heatmap: torch.Tensor,
    target_count: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth L1 loss between sum(heatmap) and target_count.

    Args:
        heatmap: [T] tensor of per-sample probabilities in [0, 1].
        target_count: expected number of peaks in the molecule.
        mask: optional [T] boolean mask; only sum over masked-in positions.

    Returns:
        Scalar tensor: smooth L1 loss normalized by target_count.
    """
    if mask is not None:
        heatmap = heatmap * mask.to(heatmap.dtype)
    predicted = heatmap.sum()
    target = torch.tensor(float(target_count), device=heatmap.device, dtype=heatmap.dtype)
    raw_loss = F.smooth_l1_loss(predicted, target, reduction="mean")
    denominator = max(float(target_count), 1.0)
    return raw_loss / denominator


def peakiness_regularizer(heatmap: torch.Tensor, window: int = 20) -> torch.Tensor:
    """L1 penalty on (1 - max_over_local_window(heatmap)).

    Encourages at least one sharp peak within each sliding window of length `window`.
    Penalty is zero if every window contains a value at or near 1.0.

    Args:
        heatmap: [T] tensor in [0, 1].
        window: sliding-window size (in samples) over which to take the max.

    Returns:
        Scalar tensor: mean L1 penalty over all windows.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    # Pad on both sides with replicate to keep length T
    half = window // 2
    padded = F.pad(heatmap.unsqueeze(0).unsqueeze(0), (half, window - half - 1), mode="replicate")
    max_pooled = F.max_pool1d(padded, kernel_size=window, stride=1).squeeze(0).squeeze(0)
    # Trim to original length (padding may cause 1-off depending on even/odd window)
    max_pooled = max_pooled[: heatmap.shape[0]]
    return (1.0 - max_pooled).mean()
