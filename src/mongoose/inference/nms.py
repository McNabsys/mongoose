"""Velocity-adaptive non-maximum suppression and sub-sample peak interpolation."""

from __future__ import annotations

import bisect

import numpy as np
import torch


def velocity_adaptive_nms(
    heatmap: torch.Tensor,
    velocity: torch.Tensor,
    threshold: float = 0.3,
    tag_width_bp: float = 511.0,
) -> torch.Tensor:
    """Non-maximum suppression with velocity-adaptive minimum separation.

    min_separation at each position = max(8, tag_width_bp / velocity * 0.5)
    This ensures we don't double-count a single tag while still resolving
    closely spaced clusters at the slow leading edge.

    Args:
        heatmap: 1D tensor [T] of probabilities (0-1).
        velocity: 1D tensor [T] of bp/sample (positive).
        threshold: Minimum confidence to consider a peak.
        tag_width_bp: Tag width in base pairs for separation calculation.

    Returns:
        1D tensor of peak sample indices (sorted ascending).
    """
    # Work in numpy: the greedy NMS does tight per-candidate scalar access
    # (heatmap[idx], velocity[idx], suppression check) that hits per-element
    # overhead hard when routed through torch tensors.
    hm_np = heatmap.detach().cpu().numpy() if heatmap.is_cuda else heatmap.detach().numpy()
    vel_np = velocity.detach().cpu().numpy() if velocity.is_cuda else velocity.detach().numpy()

    above = np.nonzero(hm_np >= threshold)[0]
    if above.size == 0:
        return torch.tensor([], dtype=torch.long)

    # Descending sort by heatmap value for greedy acceptance order.
    order = np.argsort(-hm_np[above], kind="stable")
    candidates = above[order]

    # Greedy NMS. `accepted_sorted` is kept sorted by index so we can use
    # bisect to test only the immediate left/right neighbors instead of
    # scanning every accepted peak (O(N log K) vs. O(N*K)).
    accepted_sorted: list[int] = []
    for idx_np in candidates:
        idx = int(idx_np)
        vel = float(vel_np[idx])
        if vel <= 0:
            vel = 1e-10
        min_sep = max(8.0, tag_width_bp / vel * 0.5)

        pos = bisect.bisect_left(accepted_sorted, idx)
        suppressed = False
        if pos > 0 and idx - accepted_sorted[pos - 1] < min_sep:
            suppressed = True
        elif pos < len(accepted_sorted) and accepted_sorted[pos] - idx < min_sep:
            suppressed = True

        if not suppressed:
            accepted_sorted.insert(pos, idx)

    if not accepted_sorted:
        return torch.tensor([], dtype=torch.long)

    return torch.tensor(accepted_sorted, dtype=torch.long)


def subsample_peak_position(heatmap: torch.Tensor, peak_idx: int) -> float:
    """Parabolic interpolation for sub-sample peak localization.

    Fits a parabola to heatmap[peak_idx-1], heatmap[peak_idx], heatmap[peak_idx+1].
    Returns fractional index.

    Args:
        heatmap: 1D tensor of heatmap values.
        peak_idx: Integer index of the peak.

    Returns:
        Fractional index after parabolic interpolation.
    """
    if peak_idx <= 0 or peak_idx >= len(heatmap) - 1:
        return float(peak_idx)
    p_prev = heatmap[peak_idx - 1].item()
    p_curr = heatmap[peak_idx].item()
    p_next = heatmap[peak_idx + 1].item()
    denom = 2.0 * (p_prev - 2 * p_curr + p_next)
    if abs(denom) < 1e-10:
        return float(peak_idx)
    offset = (p_prev - p_next) / denom
    return float(peak_idx) + max(-0.5, min(0.5, offset))  # clamp to +/-0.5
