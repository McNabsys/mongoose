"""Tests for the V5 Phase 2 sequence model."""
from __future__ import annotations

import math

import pytest
import torch

from mongoose.data.residual_dataset import FEATURE_DIM
from mongoose.model.sequence_residual import (
    DEFAULT_MAX_SEQ_LEN,
    SequenceResidualModel,
)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_constructor_default_dimensions():
    m = SequenceResidualModel()
    assert m.input_dim == FEATURE_DIM
    assert m.hidden_dim == 128
    assert m.n_layers == 4
    assert m.n_heads == 4
    assert m.max_seq_len == DEFAULT_MAX_SEQ_LEN


def test_constructor_rejects_input_dim_mismatch():
    with pytest.raises(ValueError, match="does not match FEATURE_DIM"):
        SequenceResidualModel(input_dim=FEATURE_DIM + 7)


def test_constructor_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="must be divisible"):
        SequenceResidualModel(hidden_dim=130, n_heads=4)


def test_head_initialized_to_bias():
    m = SequenceResidualModel(head_bias_init=1234.0)
    feats = torch.randn(2, 5, FEATURE_DIM)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = m(feats, mask)
    # head.weight zero-init -> output is just the bias everywhere
    assert torch.allclose(out, torch.full_like(out, 1234.0), atol=1e-3)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def test_forward_shape():
    m = SequenceResidualModel(hidden_dim=64, n_layers=2, n_heads=2)
    feats = torch.randn(3, 7, FEATURE_DIM)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    out = m(feats, mask)
    assert out.shape == (3, 7)


def test_forward_finite_with_random_input():
    m = SequenceResidualModel(hidden_dim=64, n_layers=2)
    feats = torch.randn(4, 10, FEATURE_DIM)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    out = m(feats, mask)
    assert torch.isfinite(out).all()


def test_forward_handles_padding_mask():
    """With some positions masked, the output for unmasked positions
    should not depend on the padded positions' content."""
    m = SequenceResidualModel(hidden_dim=64, n_layers=2, n_heads=2)
    m.eval()

    feats_a = torch.randn(1, 5, FEATURE_DIM)
    mask = torch.tensor([[False, False, False, True, True]])
    feats_b = feats_a.clone()
    feats_b[0, 3:, :] = 999.0  # garbage in padded positions

    with torch.no_grad():
        out_a = m(feats_a, mask)
        out_b = m(feats_b, mask)

    # Real positions (0..2) should be identical regardless of padded content.
    assert torch.allclose(out_a[0, :3], out_b[0, :3], atol=1e-4)


def test_forward_rejects_too_long_sequence():
    m = SequenceResidualModel(max_seq_len=20)
    feats = torch.randn(1, 25, FEATURE_DIM)
    mask = torch.zeros(1, 25, dtype=torch.bool)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        m(feats, mask)


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------

def test_backward_flows_to_all_layers():
    m = SequenceResidualModel(hidden_dim=64, n_layers=2)
    feats = torch.randn(2, 5, FEATURE_DIM, requires_grad=True)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    target = torch.randn(2, 5)
    pred = m(feats, mask)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad and p.grad is None:
            pytest.fail(f"{name} has no grad")
        if p.grad is not None and not torch.isfinite(p.grad).all():
            pytest.fail(f"{name} grad has non-finite values")


def test_overfit_tiny_batch():
    """A small batch should be overfittable in a few hundred steps."""
    torch.manual_seed(0)
    m = SequenceResidualModel(hidden_dim=64, n_layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    feats = torch.randn(4, 6, FEATURE_DIM)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    target = torch.randn(4, 6) * 200 + 1500

    initial_loss = None
    for step in range(300):
        optimizer.zero_grad()
        pred = m(feats, mask)
        loss = (pred - target).pow(2).mean()
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
    final = float(loss.item())
    assert final < 0.2 * initial_loss, (
        f"expected >5x loss reduction; initial={initial_loss:.1f}, final={final:.1f}"
    )


# --------------------------------------------------------------------------
# Parameter count
# --------------------------------------------------------------------------

def test_num_parameters_in_expected_range():
    """Default config (~30 in, 128 hidden, 4 layers, 4 heads) should
    produce ~1-2M params -- a reasonable Phase 2 size."""
    m = SequenceResidualModel()
    n = m.num_parameters
    assert 200_000 < n < 5_000_000, f"unexpected param count: {n}"
