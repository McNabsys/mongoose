"""Soft-DTW loss (Cuturi & Blondel 2017).

Differentiable dynamic time warping for aligning two 1D sequences.
Used in V1 rearchitecture to match model-detected peak positions (in cumulative bp)
against E. coli reference probe positions, handling FN/FP detections natively.

Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series",
ICML 2017.
"""
from __future__ import annotations

import torch


def _soft_min(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Smooth minimum of three scalars via log-sum-exp.

    soft_min(a,b,c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
    """
    stacked = torch.stack([a, b, c])
    return -gamma * torch.logsumexp(-stacked / gamma, dim=0)


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Soft-DTW distance between two 1D sequences.

    Args:
        x: [N] tensor of first sequence values.
        y: [M] tensor of second sequence values.
        gamma: smoothing parameter. Smaller gamma -> closer to hard DTW (less smooth gradient).
               Typical range: 0.01 - 1.0.

    Returns:
        Scalar tensor: the soft-DTW distance. Non-negative when x and y are numerical.

    Raises:
        ValueError: if either sequence is empty.
    """
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError(f"soft_dtw expects 1D tensors; got shapes {tuple(x.shape)} and {tuple(y.shape)}")
    n = x.shape[0]
    m = y.shape[0]
    if n == 0 or m == 0:
        raise ValueError("soft_dtw does not accept empty sequences")

    device = x.device
    dtype = x.dtype if x.dtype.is_floating_point else torch.float32

    # Pairwise squared-distance cost matrix
    cost = (x.unsqueeze(1) - y.unsqueeze(0)) ** 2  # [N, M]

    # Build DP table column by column to let gradients flow.
    # R has shape [N+1, M+1]; R[0,0] = 0, borders = +inf.
    # We construct it without in-place assignment to preserve autograd.
    # Approach: maintain rows as lists of tensors, then stack at the end.
    inf_tensor = torch.tensor(float("inf"), device=device, dtype=dtype)
    zero_tensor = torch.tensor(0.0, device=device, dtype=dtype)

    # Row 0: [0, inf, inf, ..., inf]
    prev_row: list[torch.Tensor] = [zero_tensor] + [inf_tensor] * m

    for i in range(1, n + 1):
        # Start-of-row boundary: +inf
        curr_row: list[torch.Tensor] = [inf_tensor]
        for j in range(1, m + 1):
            value = cost[i - 1, j - 1] + _soft_min(
                prev_row[j],     # R[i-1, j]    -> up
                prev_row[j - 1], # R[i-1, j-1]  -> diagonal
                curr_row[j - 1], # R[i, j-1]    -> left
                gamma,
            )
            curr_row.append(value)
        prev_row = curr_row

    return prev_row[m]
