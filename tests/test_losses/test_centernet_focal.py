"""Tests for CenterNet-style focal loss used on sparse 1-D probe heatmaps."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mongoose.losses.centernet_focal import centernet_focal_loss


def _gaussian_target(length: int, centers: list[int], sigma: float = 2.0) -> torch.Tensor:
    x = np.arange(length, dtype=np.float32)
    hm = np.zeros(length, dtype=np.float32)
    for c in centers:
        g = np.exp(-0.5 * ((x - float(c)) / sigma) ** 2)
        hm = np.maximum(hm, g)
    return torch.from_numpy(hm)


def test_centernet_focal_loss_perfect_prediction():
    target = _gaussian_target(length=100, centers=[20, 50, 80])
    # Predict logits that sigmoid to (approximately) the target distribution.
    # For target=1, sigmoid=1 means logit=+inf; approximate with large positive.
    logits = torch.where(target > 0.99, torch.tensor(8.0), torch.logit(target.clamp(1e-4, 0.9999)))
    mask = torch.ones(100, dtype=torch.bool)

    loss = centernet_focal_loss(logits, target, mask)
    assert loss.item() < 0.1, f"expected low loss for near-perfect prediction, got {loss.item()}"


def test_centernet_focal_loss_flat_zero_prediction():
    target = _gaussian_target(length=100, centers=[20, 50, 80])
    # Model outputs near-zero probability everywhere: logit << 0
    logits = torch.full((100,), -5.0)
    mask = torch.ones(100, dtype=torch.bool)

    loss = centernet_focal_loss(logits, target, mask)
    # Three missed peaks should produce a substantial per-peak loss (>> perfect case)
    assert loss.item() > 2.0, f"expected high loss for flat-zero prediction, got {loss.item()}"


def test_centernet_focal_loss_length_invariant_with_same_num_positives():
    """Two molecules with same #peaks, same per-peak quality, different lengths
    should produce similar losses (modulo halo effect)."""
    target_short = _gaussian_target(length=100, centers=[20, 50, 80])
    target_long  = _gaussian_target(length=300, centers=[20, 50, 80])
    logits_short = torch.full((100,), -5.0)
    logits_long  = torch.full((300,), -5.0)
    mask_short = torch.ones(100, dtype=torch.bool)
    mask_long  = torch.ones(300, dtype=torch.bool)

    loss_short = centernet_focal_loss(logits_short, target_short, mask_short)
    loss_long  = centernet_focal_loss(logits_long,  target_long,  mask_long)

    # Losses should be within 25% of each other — not 3x as would happen
    # with a seq-length denominator.
    ratio = loss_long.item() / loss_short.item()
    assert 0.75 < ratio < 1.25, (
        f"length dilution detected: short={loss_short.item()}, long={loss_long.item()}, "
        f"ratio={ratio}"
    )


def test_centernet_focal_loss_masked_region_has_no_effect():
    target = _gaussian_target(length=100, centers=[20, 50, 80])
    # Predict peaks at 20 and 50 well, but miss the peak at 80.
    # Non-uniform logits make the three positive terms contribute differently,
    # so masking out peak 80 produces a measurably different per-peak average.
    logits = torch.full((100,), -5.0)
    logits[20] = 8.0  # near-perfect for peak at 20
    logits[50] = 8.0  # near-perfect for peak at 50
    # index 80 stays at -5.0 (missed peak)
    mask_full = torch.ones(100, dtype=torch.bool)
    mask_half = torch.ones(100, dtype=torch.bool)
    mask_half[60:] = False  # drop the peak at 80 and everything after

    loss_full = centernet_focal_loss(logits, target, mask_full)
    loss_half = centernet_focal_loss(logits, target, mask_half)

    # Masking-out the region containing 1 of 3 peaks should change the loss.
    assert abs(loss_full.item() - loss_half.item()) > 1e-3, (
        f"mask had no effect: full={loss_full.item()}, half={loss_half.item()}"
    )
