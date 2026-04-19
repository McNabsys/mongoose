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
    assert loss.item() < 0.5, f"expected low loss for near-perfect prediction, got {loss.item()}"
