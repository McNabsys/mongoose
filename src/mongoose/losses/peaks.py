"""Peak-extraction helpers used by CombinedLoss.

These helpers run in ``torch.no_grad()`` and return tensors with
``requires_grad=False``. The returned peak indices and widths are discrete
values used to index differentiable model outputs; gradients flow through
the indexing operation, not through the extraction itself.
"""
from __future__ import annotations

import numpy as np
import torch

from mongoose.inference.nms import velocity_adaptive_nms


@torch.no_grad()
def extract_peak_indices(
    heatmap: torch.Tensor,
    velocity: torch.Tensor,
    threshold: float = 0.3,
    tag_width_bp: float = 511.0,
) -> torch.Tensor:
    """Run velocity-adaptive NMS on a single-molecule heatmap and return peak indices.

    Args:
        heatmap: [T] tensor of probabilities in [0, 1].
        velocity: [T] tensor of bp/sample (positive).
        threshold: minimum confidence to consider a peak.
        tag_width_bp: tag width in base pairs (feeds the NMS separation rule).

    Returns:
        LongTensor of peak sample indices, sorted ascending. Empty if no peaks.
    """
    heatmap_det = heatmap.detach()
    velocity_det = velocity.detach()
    if heatmap_det.is_cuda:
        heatmap_det = heatmap_det.cpu()
        velocity_det = velocity_det.cpu()
    peaks = velocity_adaptive_nms(
        heatmap_det, velocity_det, threshold=threshold, tag_width_bp=tag_width_bp
    )
    return peaks.to(dtype=torch.long, device=heatmap.device)


@torch.no_grad()
def measure_peak_widths_samples(
    heatmap: torch.Tensor,
    peak_indices: torch.Tensor,
    threshold_frac: float = 0.5,
) -> torch.Tensor:
    """Measure per-peak FWHM-style widths in samples by walking outward.

    For each peak index, walk left and right until the heatmap value drops
    below ``peak_value * threshold_frac``. Width is ``right_idx - left_idx + 1``
    and is clipped to a minimum of 1 sample.

    Args:
        heatmap: [T] tensor of heatmap values.
        peak_indices: LongTensor of peak sample indices.
        threshold_frac: fraction of peak height at which to cut off the walk.

    Returns:
        FloatTensor of widths in samples, one per peak.
    """
    k = int(peak_indices.numel())
    if k == 0:
        return torch.zeros(0, dtype=torch.float32, device=heatmap.device)

    # Numpy per-peak with a vectorized boundary search: walks on early-training
    # heatmaps can span thousands of samples per peak, so a Python while-loop
    # (even on numpy scalars) is far too slow. Finding the first below-cutoff
    # sample via argmax on a boolean slice is a single C-level scan per side.
    hm = heatmap.detach().cpu().numpy()
    pk = peak_indices.detach().cpu().numpy()
    n = hm.shape[0]

    widths_np = np.empty(k, dtype=np.float32)
    for i in range(k):
        p = int(pk[i])
        if p < 0 or p >= n:
            widths_np[i] = 1.0
            continue
        cutoff = float(hm[p]) * float(threshold_frac)

        # Left walk: count contiguous samples in hm[0:p] that are >= cutoff,
        # starting from p-1 and going downward.
        if p == 0:
            walk_left = 0
        else:
            left_slice = hm[:p]
            # Reverse so argmax finds the first below-cutoff sample walking
            # outward from the peak.
            below_reversed = left_slice[::-1] < cutoff
            if below_reversed.any():
                walk_left = int(below_reversed.argmax())
            else:
                walk_left = p

        # Right walk: count contiguous samples in hm[p+1:n] that are >= cutoff.
        if p >= n - 1:
            walk_right = 0
        else:
            right_slice = hm[p + 1:]
            below = right_slice < cutoff
            if below.any():
                walk_right = int(below.argmax())
            else:
                walk_right = n - p - 1

        width = float(walk_left + walk_right + 1)
        if width < 1.0:
            width = 1.0
        widths_np[i] = width

    return torch.from_numpy(widths_np).to(heatmap.device)
