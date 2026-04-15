import pytest
import torch

from mongoose.losses.softdtw import soft_dtw


def test_soft_dtw_identical_sequences():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = soft_dtw(x, y, gamma=0.1)
    assert loss.item() < 0.01


def test_soft_dtw_shifted_smaller_than_different():
    """A time-shifted copy should have smaller DTW than a totally different sequence."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_shifted = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    y_different = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    loss_shifted = soft_dtw(x, y_shifted, gamma=0.1)
    loss_different = soft_dtw(x, y_different, gamma=0.1)
    assert loss_shifted < loss_different


def test_soft_dtw_different_lengths():
    """Handles sequences of different lengths natively."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 2.5, 3.0])
    loss = soft_dtw(x, y, gamma=0.1)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_soft_dtw_differentiable():
    """Gradients flow through the soft-DTW computation."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([1.5, 2.5, 3.5])
    loss = soft_dtw(x, y, gamma=0.1)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_soft_dtw_gamma_effect():
    """Smaller gamma -> closer to hard DTW. Non-identical sequences should have larger loss at smaller gamma."""
    x = torch.tensor([1.0, 3.0, 2.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    loss_small_gamma = soft_dtw(x, y, gamma=0.01)
    loss_large_gamma = soft_dtw(x, y, gamma=10.0)
    # Soft DTW with very large gamma approaches the soft-min of all paths,
    # which is typically smaller than the hard-DTW minimum.
    assert loss_small_gamma >= loss_large_gamma - 1e-4


def test_soft_dtw_symmetry():
    """DTW distance should be symmetric: DTW(x, y) == DTW(y, x)."""
    x = torch.tensor([1.0, 2.0, 3.0, 2.0])
    y = torch.tensor([1.0, 3.0, 2.0])
    loss_xy = soft_dtw(x, y, gamma=0.1)
    loss_yx = soft_dtw(y, x, gamma=0.1)
    assert torch.allclose(loss_xy, loss_yx, atol=1e-4)


def test_soft_dtw_single_element_sequences():
    """Edge case: sequences with one element each."""
    x = torch.tensor([3.0])
    y = torch.tensor([5.0])
    loss = soft_dtw(x, y, gamma=0.1)
    # Should equal the point-wise squared distance: (3-5)^2 = 4
    assert abs(loss.item() - 4.0) < 0.1


def test_soft_dtw_empty_sequence_raises():
    """Empty sequences should raise an error."""
    x = torch.tensor([])
    y = torch.tensor([1.0, 2.0])
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        soft_dtw(x, y, gamma=0.1)
