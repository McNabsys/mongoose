"""Peak-extraction helpers used by CombinedLoss.

These helpers run in ``torch.no_grad()`` and return tensors with
``requires_grad=False``. The returned peak indices and widths are discrete
values used to index differentiable model outputs; gradients flow through
the indexing operation, not through the extraction itself.
"""
from __future__ import annotations

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
    heatmap_det = heatmap.detach()
    n = int(heatmap_det.shape[0])
    k = int(peak_indices.numel())
    if k == 0:
        return torch.zeros(0, dtype=torch.float32, device=heatmap.device)

    widths = torch.empty(k, dtype=torch.float32, device=heatmap.device)
    for i in range(k):
        p = int(peak_indices[i].item())
        if p < 0 or p >= n:
            widths[i] = 1.0
            continue
        peak_value = float(heatmap_det[p].item())
        cutoff = peak_value * float(threshold_frac)

        # Walk left
        left = p
        while left - 1 >= 0 and float(heatmap_det[left - 1].item()) >= cutoff:
            left -= 1

        # Walk right
        right = p
        while right + 1 < n and float(heatmap_det[right + 1].item()) >= cutoff:
            right += 1

        width = float(right - left + 1)
        if width < 1.0:
            width = 1.0
        widths[i] = width

    return widths
