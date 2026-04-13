"""Probe heatmap target construction for training."""

from __future__ import annotations

import numpy as np


def build_probe_heatmap(
    length: int,
    probe_centers_samples: np.ndarray,
    probe_durations_samples: np.ndarray,
) -> np.ndarray:
    """Build soft Gaussian heatmap target.

    Each probe generates a Gaussian peak centered at its sample index.
    sigma_i = max(1.5, duration_samples_i / 6). Overlapping Gaussians use
    element-wise max (not sum) to keep peak values at 1.0.

    Args:
        length: Total length of the output heatmap array.
        probe_centers_samples: Sample indices of probe centers (int or float).
        probe_durations_samples: Duration of each probe in samples.

    Returns:
        Float32 array of shape (length,) with values in [0, 1].
    """
    heatmap = np.zeros(length, dtype=np.float32)
    for center, duration in zip(probe_centers_samples, probe_durations_samples):
        sigma = max(1.5, float(duration) / 6.0)
        lo = max(0, int(center - 4 * sigma))
        hi = min(length, int(center + 4 * sigma) + 1)
        if lo >= hi:
            continue
        x = np.arange(lo, hi, dtype=np.float32)
        gaussian = np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)
        heatmap[lo:hi] = np.maximum(heatmap[lo:hi], gaussian)
    return heatmap
