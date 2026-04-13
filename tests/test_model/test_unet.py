import torch
from mongoose.model.unet import T2DUNet


def test_forward_shapes():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    B, T = 2, 4096
    x = torch.randn(B, 1, T)
    cond = torch.randn(B, 6)
    mask = torch.ones(B, T, dtype=torch.bool)
    probe_heatmap, cumulative_bp, raw_velocity = model(x, cond, mask)
    assert probe_heatmap.shape == (B, T)
    assert cumulative_bp.shape == (B, T)
    assert raw_velocity.shape == (B, T)


def test_cumulative_bp_monotonic():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 2048)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 2048, dtype=torch.bool)
    _, cumulative_bp, _ = model(x, cond, mask)
    diffs = torch.diff(cumulative_bp, dim=-1)
    assert (diffs >= 0).all()


def test_probe_heatmap_range():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 2048)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 2048, dtype=torch.bool)
    probe_heatmap, _, _ = model(x, cond, mask)
    assert (probe_heatmap >= 0).all()
    assert (probe_heatmap <= 1).all()


def test_velocity_strictly_positive():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 2048)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 2048, dtype=torch.bool)
    _, _, raw_velocity = model(x, cond, mask)
    assert (raw_velocity > 0).all()


def test_padding_mask_zeroes_velocity():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 2048)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 2048, dtype=torch.bool)
    mask[0, 1536:] = False
    _, cumulative_bp, _ = model(x, cond, mask)
    # In padded region, cumulative BP should be flat
    masked_vals = cumulative_bp[0, 1536:]
    assert torch.allclose(masked_vals, masked_vals[0:1].expand_as(masked_vals), atol=1e-6)


def test_variable_lengths():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    for T in [1024, 2048, 4000, 8192]:
        x = torch.randn(1, 1, T)
        cond = torch.randn(1, 6)
        mask = torch.ones(1, T, dtype=torch.bool)
        probe_heatmap, cumulative_bp, raw_velocity = model(x, cond, mask)
        assert probe_heatmap.shape == (1, T)
        assert cumulative_bp.shape == (1, T)


def test_gradient_flows():
    """Verify gradients flow through both heads."""
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 1024, requires_grad=True)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 1024, dtype=torch.bool)
    probe_heatmap, cumulative_bp, _ = model(x, cond, mask)
    loss = probe_heatmap.sum() + cumulative_bp.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
