"""Tests for peak-extraction helpers used by CombinedLoss."""
from __future__ import annotations

import torch

from mongoose.losses.peaks import extract_peak_indices, measure_peak_widths_samples


def _gaussian_bump(length: int, center: int, sigma: float, amplitude: float = 1.0) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    return amplitude * torch.exp(-0.5 * ((t - center) / sigma) ** 2)


def test_extract_peak_indices_finds_three_peaks():
    heatmap = torch.zeros(200)
    heatmap += _gaussian_bump(200, 30, 2.0)
    heatmap += _gaussian_bump(200, 100, 2.0)
    heatmap += _gaussian_bump(200, 170, 2.0)
    velocity = torch.full((200,), 5.0)

    peaks = extract_peak_indices(heatmap, velocity, threshold=0.3, tag_width_bp=511.0)
    assert peaks.dtype == torch.long
    assert peaks.numel() == 3
    assert torch.all(peaks == torch.tensor([30, 100, 170]))


def test_extract_peak_indices_returns_empty_on_flat_input():
    heatmap = torch.full((100,), 0.1)
    velocity = torch.full((100,), 1.0)
    peaks = extract_peak_indices(heatmap, velocity, threshold=0.3)
    assert peaks.numel() == 0
    assert peaks.dtype == torch.long


def test_extract_peak_indices_detaches_from_graph():
    """Peaks must not propagate gradient; they're discrete indices."""
    heatmap = _gaussian_bump(100, 50, 2.0).clone().requires_grad_(True)
    velocity = torch.full((100,), 1.0, requires_grad=True)
    peaks = extract_peak_indices(heatmap, velocity, threshold=0.3)
    # Returned tensor is LongTensor; cannot carry gradient information by type.
    assert not peaks.is_floating_point()
    assert not peaks.requires_grad


def test_measure_peak_widths_samples_single_peak():
    heatmap = _gaussian_bump(100, 50, 3.0)
    peak_indices = torch.tensor([50], dtype=torch.long)
    widths = measure_peak_widths_samples(heatmap, peak_indices, threshold_frac=0.5)
    # FWHM of a Gaussian with sigma=3 is 2*sqrt(2*ln(2))*sigma ~= 7.06 samples
    assert widths.dtype == torch.float32
    assert widths.numel() == 1
    assert 5.0 <= widths.item() <= 10.0


def test_measure_peak_widths_samples_clipped_minimum_one():
    heatmap = torch.zeros(10)
    heatmap[5] = 1.0  # isolated spike, drops to zero immediately
    peak_indices = torch.tensor([5], dtype=torch.long)
    widths = measure_peak_widths_samples(heatmap, peak_indices, threshold_frac=0.5)
    assert widths.item() >= 1.0


def test_measure_peak_widths_samples_multiple_peaks():
    heatmap = torch.zeros(300)
    heatmap += _gaussian_bump(300, 50, 2.0)
    heatmap += _gaussian_bump(300, 150, 5.0)
    heatmap += _gaussian_bump(300, 250, 10.0)
    peak_indices = torch.tensor([50, 150, 250], dtype=torch.long)
    widths = measure_peak_widths_samples(heatmap, peak_indices, threshold_frac=0.5)
    assert widths.numel() == 3
    # Wider Gaussian -> wider FWHM
    assert widths[0].item() < widths[1].item() < widths[2].item()


def test_measure_peak_widths_samples_edge_peak():
    """Peak at the edge (index 0 or T-1) should not crash and should return >= 1."""
    heatmap = torch.zeros(20)
    heatmap[0] = 1.0
    heatmap[1] = 0.8
    heatmap[2] = 0.3
    peak_indices = torch.tensor([0], dtype=torch.long)
    widths = measure_peak_widths_samples(heatmap, peak_indices, threshold_frac=0.5)
    assert widths.numel() == 1
    assert widths.item() >= 1.0
    assert torch.isfinite(widths).all()


def test_measure_peak_widths_samples_empty_peak_list():
    heatmap = torch.zeros(20)
    peak_indices = torch.tensor([], dtype=torch.long)
    widths = measure_peak_widths_samples(heatmap, peak_indices, threshold_frac=0.5)
    assert widths.numel() == 0
    assert widths.dtype == torch.float32
