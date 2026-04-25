"""Tests for the per-probe residual MLP (Direction C, phase C.4)."""
from __future__ import annotations

import math

import pytest
import torch

from mongoose.data.residual_dataset import FEATURE_DIM
from mongoose.model.residual_mlp import ResidualBlock, ResidualMLP


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_constructor_default_dimensions():
    model = ResidualMLP()
    assert model.input_dim == FEATURE_DIM
    assert model.hidden_dim == 256
    assert model.n_blocks == 4


def test_constructor_rejects_input_dim_mismatch():
    with pytest.raises(ValueError, match="does not match FEATURE_DIM"):
        ResidualMLP(input_dim=FEATURE_DIM + 7)


def test_head_initialized_to_residual_prior():
    """Head bias ~ empirical residual mean; head weight zero -> initial
    prediction is the bias regardless of input."""
    model = ResidualMLP()
    feats = torch.randn(8, FEATURE_DIM)
    out = model(feats)
    # All outputs should be identical at init (just the bias).
    assert torch.allclose(out, out[0].expand_as(out), atol=1e-5)
    # And close to the +2200 bp prior.
    assert math.isclose(out[0].item(), 2200.0, abs_tol=1.0)


# --------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------

def test_forward_shape():
    model = ResidualMLP(hidden_dim=32, n_blocks=2)
    feats = torch.randn(5, FEATURE_DIM)
    out = model(feats)
    assert out.shape == (5,)


def test_forward_finite_with_random_features():
    model = ResidualMLP(hidden_dim=64, n_blocks=2)
    feats = torch.randn(10, FEATURE_DIM)
    out = model(feats)
    assert torch.isfinite(out).all()


def test_forward_rejects_wrong_shape():
    model = ResidualMLP()
    bad = torch.randn(4, FEATURE_DIM + 3)
    with pytest.raises(ValueError, match="expected features shape"):
        model(bad)
    bad_dim = torch.randn(FEATURE_DIM)  # 1-D
    with pytest.raises(ValueError, match="expected features shape"):
        model(bad_dim)


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------

def test_backward_flows_to_all_parameters():
    model = ResidualMLP(hidden_dim=32, n_blocks=2)
    feats = torch.randn(8, FEATURE_DIM)
    target = torch.randn(8) * 1000 + 2000
    pred = model(feats)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and "head.weight" not in name:
            assert p.grad is not None, f"{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"
    # head.weight is zero-init and skip-connected through residual blocks;
    # gradient should still be finite even if it could be tiny on first step.
    assert torch.isfinite(model.head.weight.grad).all()


def test_overfit_tiny_batch():
    """Sanity check: a small batch should be overfittable in a few steps."""
    torch.manual_seed(0)
    model = ResidualMLP(hidden_dim=64, n_blocks=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    feats = torch.randn(16, FEATURE_DIM)
    target = torch.randn(16) * 500 + 2200  # spread around prior

    initial_loss = None
    for step in range(200):
        optimizer.zero_grad()
        pred = model(feats)
        loss = (pred - target).pow(2).mean()
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
    final = float(loss.item())
    assert final < 0.1 * initial_loss, (
        f"expected >10x loss reduction; initial={initial_loss:.1f}, final={final:.1f}"
    )


# --------------------------------------------------------------------------
# Parameter count sanity
# --------------------------------------------------------------------------

def test_num_parameters_in_expected_range():
    """Default config: 22 in -> 256 hidden, 4 residual blocks -> ~1M params.

    Expected count breakdown (rough):
        stem:     22 * 256 + 256 = 5,888 weights + biases
        layernorm: 2 * 256 = 512
        blocks:   4 * (2 * (256 * 256 + 256) + 4 * 256) ~= 530K
        head:     256 + 1 = 257
    Total around 540K parameters.
    """
    model = ResidualMLP()
    n = model.num_parameters
    # Loose sanity envelope -- hard exact count is brittle.
    assert 100_000 < n < 5_000_000, f"unexpected parameter count: {n}"


# --------------------------------------------------------------------------
# ResidualBlock isolated
# --------------------------------------------------------------------------

def test_residual_block_preserves_shape():
    block = ResidualBlock(hidden_dim=64, dropout=0.0)
    x = torch.randn(4, 64)
    out = block(x)
    assert out.shape == x.shape


def test_residual_block_skip_at_zero_init():
    """A freshly-init block applied to x with all internal zeroed weights
    should approximately return x (skip dominates)."""
    block = ResidualBlock(hidden_dim=32, dropout=0.0)
    for p in block.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(3, 32)
    out = block(x)
    assert torch.allclose(out, x, atol=1e-5)
