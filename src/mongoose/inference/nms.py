"""Velocity-adaptive non-maximum suppression and sub-sample peak interpolation."""

from __future__ import annotations

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
    # Step 1: Find all indices where heatmap >= threshold
    candidates = torch.nonzero(heatmap >= threshold, as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    # Step 2: Sort candidates by heatmap value (descending, greedy NMS)
    candidate_values = heatmap[candidates]
    sorted_order = torch.argsort(candidate_values, descending=True)
    candidates = candidates[sorted_order]

    # Step 3: Greedy NMS with velocity-adaptive separation
    accepted: list[int] = []
    for i in range(len(candidates)):
        idx = candidates[i].item()
        # Compute local min_separation
        vel = velocity[idx].item()
        if vel <= 0:
            vel = 1e-10  # avoid division by zero
        min_sep = max(8.0, tag_width_bp / vel * 0.5)

        # Check if any already-accepted peak is within min_separation
        suppressed = False
        for accepted_idx in accepted:
            if abs(idx - accepted_idx) < min_sep:
                suppressed = True
                break

        if not suppressed:
            accepted.append(idx)

    if not accepted:
        return torch.tensor([], dtype=torch.long)

    # Step 4: Sort accepted peaks by index (ascending temporal order)
    accepted.sort()
    return torch.tensor(accepted, dtype=torch.long)


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
