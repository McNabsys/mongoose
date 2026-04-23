"""Tests for Option A: T2D-hybrid velocity interpretation in T2DUNet.

Key invariants (from design):

  * ``v_final(t) = v_T2D(t) * (1 + residual(t))`` where
    ``residual = 0.5 * tanh(velocity_head_output) ∈ [-0.5, +0.5]``
  * Residual = 0 (zero logits) ⇒ ``raw_velocity == v_T2D``. Graceful-
    degradation property: if the neural net learns nothing useful, output
    collapses to pure T2D.
  * Residual saturated at +0.5 (very positive logits) ⇒
    ``raw_velocity == 1.5 * v_T2D``; saturated at -0.5 ⇒ ``0.5 * v_T2D``.
  * ``compute_v_t2d`` produces finite positive velocities even at the
    trailing edge (clamped by ``T2D_MIN_T_FROM_TAIL_MS``).
  * Backward compat: ``forward(x, cond, mask)`` (no t2d_params) produces
    identical shapes and positive velocities — V1 and L_511 inference
    paths are untouched.
"""
from __future__ import annotations

import torch

from mongoose.model.unet import (
    T2D_MIN_T_FROM_TAIL_MS,
    T2D_RESIDUAL_BOUND,
    T2DUNet,
    compute_v_t2d,
)


# -----------------------------------------------------------------------
# compute_v_t2d — the per-sample v_T2D formula
# -----------------------------------------------------------------------
def test_compute_v_t2d_produces_positive_finite_velocities() -> None:
    mult_const = torch.tensor([[6343.0]])
    alpha = torch.tensor([[0.558]])
    tail_ms = torch.tensor([[70.0]])  # molecule tail at 70 ms
    v = compute_v_t2d(mult_const, alpha, tail_ms, T=2240)  # ~70ms at 32 kHz
    assert v.shape == (1, 2240)
    assert torch.all(v > 0)
    assert torch.all(torch.isfinite(v))


def test_compute_v_t2d_monotone_decreasing_with_sample_index() -> None:
    """v_T2D should decrease as we move away from the leading edge
    (smaller t_from_tail → larger power for alpha<1 with negative exponent)."""
    # Actually for alpha < 1, dL/dt = mult * alpha * t^(alpha-1) where
    # alpha-1 < 0. So dL/dt DECREASES as t grows. Translation: the
    # leading edge (large t_from_tail, early in time) has LOW velocity,
    # and velocity accelerates toward the trailing edge (small t_from_tail).
    # This is the expected physics: drag decreases as molecule empties.
    mult_const = torch.tensor([[6343.0]])
    alpha = torch.tensor([[0.558]])
    tail_ms = torch.tensor([[70.0]])
    v = compute_v_t2d(mult_const, alpha, tail_ms, T=2240)
    # Expect v[0] (start, far from tail) < v[-1] (end, near tail).
    assert v[0, 0].item() < v[0, -1].item(), (
        f"expected monotone acceleration toward tail, got v[0]={v[0, 0]:.2f} "
        f"v[-1]={v[0, -1]:.2f}"
    )


def test_compute_v_t2d_clamps_near_tail() -> None:
    """At the trailing edge, t_from_tail_ms → 0 would give divergent v_T2D.
    The clamp keeps things finite."""
    mult_const = torch.tensor([[6343.0]])
    alpha = torch.tensor([[0.558]])
    tail_ms = torch.tensor([[1.0]])  # very short molecule, sample 32 is past tail
    v = compute_v_t2d(mult_const, alpha, tail_ms, T=128)
    assert torch.all(torch.isfinite(v))
    # Max velocity is bounded by the clamp (t_from_tail=2 ms):
    #   v_max = mult * alpha * 2^(alpha-1) * sample_period_ms
    sample_period_ms = 1000.0 / 32000
    v_max = 6343.0 * 0.558 * (T2D_MIN_T_FROM_TAIL_MS ** (0.558 - 1)) * sample_period_ms
    assert torch.all(v <= v_max + 1e-4), f"v exceeds clamp bound, max={v.max()} bound={v_max}"


def test_compute_v_t2d_batches_independently() -> None:
    """Two molecules with different params should get different v_T2D."""
    mult_const = torch.tensor([[6343.0], [5000.0]])
    alpha = torch.tensor([[0.558], [0.6]])
    tail_ms = torch.tensor([[70.0], [50.0]])
    v = compute_v_t2d(mult_const, alpha, tail_ms, T=1000)
    assert v.shape == (2, 1000)
    # Different rows should not be identical.
    assert not torch.allclose(v[0], v[1])


# -----------------------------------------------------------------------
# T2DUNet forward — the hybrid path
# -----------------------------------------------------------------------
def _make_model_with_zeroed_velocity_head() -> T2DUNet:
    """Produce a T2DUNet whose velocity head output is identically zero.

    With residual = 0.5 * tanh(0) = 0, the hybrid path should yield
    raw_velocity == v_T2D exactly.
    """
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    model.eval()
    # Zero both the weight and bias of the final Conv1d in velocity_head.
    # velocity_head is a Sequential; find the last Conv1d.
    for mod in reversed(list(model.velocity_head.modules())):
        if isinstance(mod, torch.nn.Conv1d):
            with torch.no_grad():
                mod.weight.zero_()
                if mod.bias is not None:
                    mod.bias.zero_()
            break
    return model


def test_hybrid_mode_with_zero_residual_equals_pure_t2d() -> None:
    """Graceful-degradation invariant: residual=0 ⇒ output is pure T2D."""
    torch.manual_seed(0)
    model = _make_model_with_zeroed_velocity_head()
    x = torch.randn(2, 1, 128)
    cond = torch.randn(2, 6)
    mask = torch.ones(2, 128, dtype=torch.bool)
    t2d_params = torch.tensor([[6343.0, 0.558, 4.0], [6500.0, 0.55, 4.0]])

    with torch.no_grad():
        _probe, _cum_bp, raw_vel, _logits = model(x, cond, mask, t2d_params=t2d_params)

    # Expected: raw_vel == v_T2D (since residual is zero).
    expected = compute_v_t2d(
        t2d_params[:, 0:1], t2d_params[:, 1:2], t2d_params[:, 2:3], T=128
    )
    assert torch.allclose(raw_vel, expected, atol=1e-4), (
        f"residual=0 did not collapse to pure T2D. "
        f"max abs diff = {(raw_vel - expected).abs().max().item()}"
    )


def test_hybrid_mode_residual_bounded_within_50pct() -> None:
    """Residual is tanh-bounded → output can only be within [0.5, 1.5] × v_T2D."""
    torch.manual_seed(0)
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    model.eval()
    x = torch.randn(4, 1, 256)
    cond = torch.randn(4, 6)
    mask = torch.ones(4, 256, dtype=torch.bool)
    # Use realistic params across the batch.
    t2d_params = torch.tensor([
        [6343.0, 0.558, 60.0],
        [6500.0, 0.55, 70.0],
        [6200.0, 0.56, 50.0],
        [6400.0, 0.555, 80.0],
    ])
    with torch.no_grad():
        _, _, raw_vel, _ = model(x, cond, mask, t2d_params=t2d_params)

    v_t2d = compute_v_t2d(
        t2d_params[:, 0:1], t2d_params[:, 1:2], t2d_params[:, 2:3], T=256
    )
    # raw_vel should be in [(1 - bound) * v_T2D, (1 + bound) * v_T2D].
    lower = (1.0 - T2D_RESIDUAL_BOUND) * v_t2d
    upper = (1.0 + T2D_RESIDUAL_BOUND) * v_t2d
    # Small slack for floating-point.
    assert torch.all(raw_vel >= lower - 1e-4), "raw_vel fell below bound"
    assert torch.all(raw_vel <= upper + 1e-4), "raw_vel exceeded bound"


def test_default_mode_unchanged_without_t2d_params() -> None:
    """Backward compat: forward(x, cond, mask) still works and returns
    positive softplus velocities."""
    torch.manual_seed(0)
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    model.eval()
    x = torch.randn(2, 1, 128)
    cond = torch.randn(2, 6)
    mask = torch.ones(2, 128, dtype=torch.bool)

    with torch.no_grad():
        probe, cum_bp, raw_vel, logits = model(x, cond, mask)

    # All outputs have the right shape + positive velocity.
    assert probe.shape == (2, 128)
    assert cum_bp.shape == (2, 128)
    assert raw_vel.shape == (2, 128)
    assert logits.shape == (2, 128)
    assert torch.all(raw_vel >= 0)
    assert torch.all(torch.isfinite(raw_vel))


def test_hybrid_mode_gradient_flows_to_residual() -> None:
    """L_511-like gradient on raw_velocity must flow back through the
    velocity head (residual) when in hybrid mode."""
    torch.manual_seed(0)
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    model.train()
    x = torch.randn(1, 1, 128)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 128, dtype=torch.bool)
    t2d_params = torch.tensor([[6343.0, 0.558, 4.0]])

    _, _, raw_vel, _ = model(x, cond, mask, t2d_params=t2d_params)
    loss = raw_vel.sum()  # trivial differentiable objective
    loss.backward()

    # At least one velocity_head param should have a nonzero grad.
    grads = [
        p.grad for p in model.velocity_head.parameters() if p.grad is not None
    ]
    assert grads, "velocity_head has no grads — residual disconnected from graph"
    total_grad = sum(g.abs().sum().item() for g in grads)
    assert total_grad > 0, f"velocity_head grads are zero (sum={total_grad})"


# -----------------------------------------------------------------------
# probe_aware_velocity flag
# -----------------------------------------------------------------------
def test_probe_aware_velocity_extends_input_channel() -> None:
    """With probe_aware_velocity=True, vel head accepts an extra channel."""
    torch.manual_seed(0)
    model = T2DUNet(in_channels=1, conditioning_dim=6, probe_aware_velocity=True)
    # First module of velocity_head should be the channel-merging Conv1d
    # taking 33 channels in (32 backbone + 1 probe) and emitting 32.
    first = model.velocity_head[0]
    assert isinstance(first, torch.nn.Conv1d)
    assert first.in_channels == 33, f"expected 33 in_channels, got {first.in_channels}"
    assert first.out_channels == 32


def test_probe_aware_forward_runs_and_is_positive() -> None:
    """End-to-end forward with probe-aware vel + hybrid path."""
    torch.manual_seed(0)
    model = T2DUNet(in_channels=1, conditioning_dim=6, probe_aware_velocity=True)
    model.eval()
    x = torch.randn(2, 1, 128)
    cond = torch.randn(2, 6)
    mask = torch.ones(2, 128, dtype=torch.bool)
    t2d_params = torch.tensor([[6343.0, 0.558, 4.0], [6500.0, 0.55, 4.0]])
    with torch.no_grad():
        probe, cum_bp, raw_vel, _ = model(x, cond, mask, t2d_params=t2d_params)
    assert raw_vel.shape == (2, 128)
    assert torch.all(raw_vel >= 0)
    assert torch.all(torch.isfinite(raw_vel))


def test_probe_aware_does_not_backprop_to_probe_head() -> None:
    """probe.detach() before concat means vel-head loss must NOT update
    probe_head weights (only L_probe should drive probe_head training)."""
    torch.manual_seed(0)
    model = T2DUNet(in_channels=1, conditioning_dim=6, probe_aware_velocity=True)
    model.train()
    x = torch.randn(1, 1, 128)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 128, dtype=torch.bool)
    t2d_params = torch.tensor([[6343.0, 0.558, 4.0]])

    _, _, raw_vel, _ = model(x, cond, mask, t2d_params=t2d_params)
    loss = raw_vel.sum()
    loss.backward()

    # probe_head's grads should be ZERO since the only path to probe_head
    # was through the detached concat. Backbone grads are fine.
    probe_grads = [
        p.grad for p in model.probe_head.parameters() if p.grad is not None
    ]
    if probe_grads:  # if grads were even allocated
        total = sum(g.abs().sum().item() for g in probe_grads)
        assert total == 0.0, (
            f"probe_head got velocity-loss gradient (sum={total}); "
            "detach() in probe-aware path is broken"
        )
