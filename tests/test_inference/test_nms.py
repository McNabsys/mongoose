"""Tests for velocity-adaptive NMS and sub-sample peak interpolation."""

import torch

from mongoose.inference.nms import subsample_peak_position, velocity_adaptive_nms


def test_nms_single_peak():
    heatmap = torch.zeros(200)
    heatmap[100] = 0.9
    velocity = torch.ones(200) * 0.5
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 1
    assert peaks[0].item() == 100


def test_nms_suppresses_close_duplicates():
    heatmap = torch.zeros(200)
    heatmap[100] = 0.9
    heatmap[103] = 0.7  # too close
    velocity = torch.ones(200) * 0.5  # min_sep = max(8, 511/0.5*0.5) = 511 samples
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 1
    assert peaks[0].item() == 100


def test_nms_keeps_distant_peaks():
    heatmap = torch.zeros(2000)
    heatmap[200] = 0.9
    heatmap[1800] = 0.8
    velocity = torch.ones(2000) * 0.5
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 2


def test_nms_velocity_adaptive():
    """At high velocity, min_sep is smaller, so close peaks are resolved."""
    heatmap = torch.zeros(200)
    heatmap[50] = 0.9
    heatmap[70] = 0.8  # 20 samples apart
    # High velocity: min_sep = max(8, 511/50*0.5) = max(8, 5.1) = 8
    velocity = torch.ones(200) * 50.0
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 2  # resolved because min_sep = 8 < 20


def test_nms_below_threshold_ignored():
    heatmap = torch.zeros(200)
    heatmap[100] = 0.1  # below threshold
    velocity = torch.ones(200) * 0.5
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 0


def test_subsample_symmetric_peak():
    heatmap = torch.tensor([0.5, 1.0, 0.5])
    pos = subsample_peak_position(heatmap, 1)
    assert abs(pos - 1.0) < 0.01  # symmetric -> no offset


def test_subsample_asymmetric_peak():
    heatmap = torch.tensor([0.3, 1.0, 0.8])
    pos = subsample_peak_position(heatmap, 1)
    assert pos > 1.0  # shifted toward the higher neighbor
    assert pos < 1.5  # but not too far


def test_subsample_edge_peak():
    heatmap = torch.tensor([1.0, 0.5])
    pos = subsample_peak_position(heatmap, 0)
    assert pos == 0.0  # edge case, no interpolation
