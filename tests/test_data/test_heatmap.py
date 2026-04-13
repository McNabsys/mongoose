"""Tests for probe heatmap target construction."""

import numpy as np

from mongoose.data.heatmap import build_probe_heatmap


def test_heatmap_single_probe():
    heatmap = build_probe_heatmap(100, np.array([50]), np.array([20.0]))
    assert heatmap[50] > 0.99  # peak at center
    assert heatmap[0] < 0.01  # far from center


def test_heatmap_sigma_scales_with_duration():
    # Short duration = tight Gaussian
    hm_short = build_probe_heatmap(200, np.array([100]), np.array([6.0]))
    # Long duration = wide Gaussian
    hm_long = build_probe_heatmap(200, np.array([100]), np.array([60.0]))
    # The long-duration heatmap should be wider (more non-zero values)
    assert (hm_long > 0.5).sum() > (hm_short > 0.5).sum()


def test_heatmap_minimum_sigma():
    heatmap = build_probe_heatmap(100, np.array([50]), np.array([1.0]))
    # sigma = max(1.5, 1/6) = 1.5 (minimum enforced)
    # Should still produce a visible peak
    assert heatmap[50] > 0.99


def test_heatmap_overlapping_probes_use_max():
    # Two probes close together
    hm = build_probe_heatmap(100, np.array([50, 55]), np.array([12.0, 12.0]))
    # Peak values should be ~1.0, not >1.0 (max, not sum)
    assert hm.max() <= 1.0 + 1e-6
