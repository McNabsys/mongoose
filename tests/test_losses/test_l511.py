"""Unit tests for the L_511 physics-informed loss module.

Covers three invariants:
  * ``l511_per_probe`` hits zero when integrated velocity == 511 bp.
  * ``l511_per_probe`` produces a nonzero residual when it's off, and
    the gradient flows to ``raw_velocity`` but NOT to the boundaries
    (round-6 slack-variable-trap fix).
  * ``l_length_span`` compares integrated velocity between first/last
    probe to the absolute reference bp span.
"""
from __future__ import annotations

import torch

from mongoose.losses.l511 import (
    l511_per_probe,
    l_length_span,
    l_smooth_velocity,
)


def test_l511_zero_when_integral_equals_511() -> None:
    """If velocity is constant such that integral == 511 over each probe, loss == 0."""
    T = 200
    # Probe 1: center=50, width=10 samples. Sum of 11 samples each = 511/11.
    # Probe 2: center=150, width=20 samples. Sum of 21 samples each = 511/21.
    raw_v = torch.zeros(T)
    raw_v[45:56] = 511.0 / 11.0
    raw_v[140:161] = 511.0 / 21.0

    centers = torch.tensor([50, 150], dtype=torch.long)
    widths = torch.tensor([10.0, 20.0])

    loss = l511_per_probe(raw_v, centers, widths)
    assert loss.item() < 1e-6, f"expected ~0, got {loss.item()}"


def test_l511_nonzero_off_target() -> None:
    """If integral != 511, loss is positive and of the right magnitude."""
    T = 200
    raw_v = torch.ones(T)  # integral over width=10 = 11 bp, way off
    centers = torch.tensor([100], dtype=torch.long)
    widths = torch.tensor([10.0])

    loss = l511_per_probe(raw_v, centers, widths)
    # residual = 11 - 511 = -500. sq = 250000.
    assert abs(loss.item() - 250000.0) < 1.0


def test_l511_gradient_flows_to_velocity_not_boundaries() -> None:
    """Gradient must flow to raw_velocity but NOT to centers/widths."""
    T = 200
    raw_v = torch.ones(T, requires_grad=True)
    centers = torch.tensor([100], dtype=torch.long)
    widths = torch.tensor([10.0], requires_grad=True)

    loss = l511_per_probe(raw_v, centers, widths)
    loss.backward()

    # Velocity grads should be nonzero inside the probe interval.
    assert raw_v.grad is not None
    assert raw_v.grad[95:106].abs().sum().item() > 0, (
        "expected velocity grads inside probe interval"
    )
    # Velocity grads outside the interval should be zero.
    assert raw_v.grad[:80].abs().sum().item() == 0
    assert raw_v.grad[120:].abs().sum().item() == 0

    # Widths grad must remain None (detached) — this is the slack-variable-trap guard.
    assert widths.grad is None, "widths must be detached from the graph"


def test_l511_empty_probes_returns_zero() -> None:
    raw_v = torch.ones(200)
    centers = torch.empty(0, dtype=torch.long)
    widths = torch.empty(0)
    loss = l511_per_probe(raw_v, centers, widths)
    assert loss.item() == 0.0


def test_l_length_span_matches_reference() -> None:
    """When integrated bp == ref span, loss == 0."""
    T = 500
    raw_v = torch.ones(T)  # integral over [10:200] = 191 bp
    centers = torch.tensor([10, 200], dtype=torch.long)
    # ref_bp span = 191 matches integrated.
    ref_bp = torch.tensor([1000, 1191], dtype=torch.long)

    loss = l_length_span(raw_v, centers, ref_bp)
    assert abs(loss.item()) < 1e-4


def test_l_length_span_descending_ref() -> None:
    """Reverse-direction molecule (ref_bp descending) still works via abs()."""
    T = 500
    raw_v = torch.ones(T)
    centers = torch.tensor([10, 200], dtype=torch.long)
    # Descending: ref_bp[-1] < ref_bp[0]. abs(diff) = 191.
    ref_bp = torch.tensor([1191, 1000], dtype=torch.long)

    loss = l_length_span(raw_v, centers, ref_bp)
    assert abs(loss.item()) < 1e-4


def test_l_smooth_velocity_zero_for_constant() -> None:
    raw_v = torch.ones(100)
    mask = torch.ones(100, dtype=torch.bool)
    loss = l_smooth_velocity(raw_v, mask)
    assert loss.item() == 0.0


def test_l_smooth_velocity_positive_for_jitter() -> None:
    raw_v = torch.zeros(100)
    raw_v[::2] = 1.0  # alternating 0, 1
    mask = torch.ones(100, dtype=torch.bool)
    loss = l_smooth_velocity(raw_v, mask)
    # Every adjacent pair differs by 1. Mean sq = 1.0.
    assert abs(loss.item() - 1.0) < 1e-4
