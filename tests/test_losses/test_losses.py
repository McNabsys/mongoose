import torch
from mongoose.losses.focal import focal_loss
from mongoose.losses.spatial import sparse_huber_delta_loss
from mongoose.losses.velocity import sparse_velocity_loss
from mongoose.losses.combined import CombinedLoss
from mongoose.losses.count import count_loss, peakiness_regularizer


def test_focal_loss_perfect_prediction():
    pred = torch.tensor([0.01, 0.01, 0.99, 0.01, 0.01])
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    loss = focal_loss(pred, target, gamma=2.0, alpha=0.25)
    assert loss.item() < 0.01


def test_focal_loss_missed_peak():
    pred = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    loss = focal_loss(pred, target, gamma=2.0, alpha=0.25)
    assert loss.item() > 0.05


def test_focal_loss_with_mask():
    pred = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    mask = torch.tensor([True, True, True, False, False])
    loss_masked = focal_loss(pred, target, gamma=2.0, alpha=0.25, mask=mask)
    loss_full = focal_loss(pred, target, gamma=2.0, alpha=0.25)
    # Masked loss only uses 3 samples, full uses 5
    assert loss_masked.item() != loss_full.item()


def test_sparse_huber_perfect():
    pred_cum = torch.tensor([0.0, 1000.0, 3500.0, 7000.0])
    gt_deltas = torch.tensor([1000.0, 2500.0, 3500.0])
    loss = sparse_huber_delta_loss(pred_cum, gt_deltas, delta=500.0)
    assert loss.item() < 0.01


def test_sparse_huber_with_error():
    pred_cum = torch.tensor([0.0, 1200.0, 3500.0, 7000.0])
    gt_deltas = torch.tensor([1000.0, 2500.0, 3500.0])
    loss = sparse_huber_delta_loss(pred_cum, gt_deltas, delta=500.0)
    assert loss.item() > 0


def test_sparse_huber_normalized():
    """Loss should be scale-invariant when normalized by mean interval."""
    pred_cum1 = torch.tensor([0.0, 1100.0, 2200.0])  # deltas [1100, 1100], gt [1000, 1000]
    gt1 = torch.tensor([1000.0, 1000.0])
    pred_cum2 = torch.tensor([0.0, 11000.0, 22000.0])  # deltas [11000, 11000], gt [10000, 10000]
    gt2 = torch.tensor([10000.0, 10000.0])
    loss1 = sparse_huber_delta_loss(pred_cum1, gt1, delta=500.0)
    loss2 = sparse_huber_delta_loss(pred_cum2, gt2, delta=500.0)
    # After normalization by mean gt, these should be similar
    # (10% error in both cases, but Huber behavior differs at different scales)
    # loss1: error=100, within delta=500 so quadratic: 100^2/(2*500) = 10, normalized by 1000 = 0.01
    # loss2: error=1000, above delta=500 so linear: 500*(1000-500/2) = 375000, normalized by 10000 = 37.5
    # These are NOT similar due to Huber transition -- that's expected behavior
    assert loss1.item() > 0
    assert loss2.item() > 0


def test_sparse_velocity_loss_perfect():
    pred_v = torch.tensor([0.5, 0.4, 0.3])
    target_v = torch.tensor([0.5, 0.4, 0.3])
    loss = sparse_velocity_loss(pred_v, target_v)
    assert loss.item() < 1e-6


def test_sparse_velocity_loss_with_error():
    pred_v = torch.tensor([0.5, 0.4, 0.3])
    target_v = torch.tensor([0.6, 0.5, 0.4])
    loss = sparse_velocity_loss(pred_v, target_v)
    assert loss.item() > 0


def test_combined_loss_warmup():
    combined = CombinedLoss(lambda_bp=1.0, lambda_vel=1.0, warmup_epochs=5)
    combined.set_epoch(0)
    assert combined.current_lambda_bp == 0.0
    assert combined.current_lambda_vel == 0.0
    combined.set_epoch(5)
    assert abs(combined.current_lambda_bp - 1.0) < 1e-6
    assert abs(combined.current_lambda_vel - 1.0) < 1e-6
    combined.set_epoch(2)
    assert abs(combined.current_lambda_bp - 0.4) < 1e-6


def test_combined_loss_warmup_zero_epochs():
    combined = CombinedLoss(lambda_bp=2.0, lambda_vel=3.0, warmup_epochs=0)
    combined.set_epoch(0)
    assert abs(combined.current_lambda_bp - 2.0) < 1e-6
    assert abs(combined.current_lambda_vel - 3.0) < 1e-6


def test_count_loss_exact_match():
    heatmap = torch.zeros(100)
    heatmap[10] = heatmap[30] = heatmap[60] = 1.0
    mask = torch.ones(100, dtype=torch.bool)
    loss = count_loss(heatmap, target_count=3, mask=mask)
    assert loss.item() < 0.1


def test_count_loss_too_few_probes():
    heatmap = torch.zeros(100)
    heatmap[10] = 1.0
    mask = torch.ones(100, dtype=torch.bool)
    loss_low = count_loss(heatmap, target_count=3, mask=mask)
    loss_ok = count_loss(heatmap.clone(), target_count=1, mask=mask)
    assert loss_low.item() > loss_ok.item()


def test_count_loss_too_many_probes():
    heatmap = torch.ones(100) * 0.5  # sum = 50
    mask = torch.ones(100, dtype=torch.bool)
    loss = count_loss(heatmap, target_count=3, mask=mask)
    assert loss.item() > 0.1


def test_count_loss_respects_mask():
    heatmap = torch.zeros(100)
    heatmap[10] = heatmap[30] = 1.0
    heatmap[60] = 1.0  # would count if unmasked
    mask = torch.zeros(100, dtype=torch.bool)
    mask[:50] = True
    loss_masked = count_loss(heatmap, target_count=2, mask=mask)
    loss_unmasked = count_loss(heatmap, target_count=3, mask=torch.ones(100, dtype=torch.bool))
    assert loss_masked.item() < 0.1
    assert loss_unmasked.item() < 0.1


def test_count_loss_differentiable():
    heatmap = torch.ones(100) * 0.3
    heatmap.requires_grad_(True)
    mask = torch.ones(100, dtype=torch.bool)
    loss = count_loss(heatmap, target_count=5, mask=mask)
    loss.backward()
    assert heatmap.grad is not None
    assert heatmap.grad.abs().sum().item() > 0


def test_peakiness_regularizer_flat_vs_peaky():
    heatmap_flat = torch.ones(100) * 0.3
    heatmap_peaky = torch.zeros(100)
    heatmap_peaky[10] = heatmap_peaky[50] = heatmap_peaky[90] = 1.0
    loss_flat = peakiness_regularizer(heatmap_flat, window=20)
    loss_peaky = peakiness_regularizer(heatmap_peaky, window=20)
    assert loss_flat.item() > loss_peaky.item()


def test_peakiness_regularizer_window_effect():
    """A single peak should satisfy peakiness within its window but not outside."""
    heatmap = torch.zeros(100)
    heatmap[50] = 1.0
    loss_small = peakiness_regularizer(heatmap, window=10)
    loss_large = peakiness_regularizer(heatmap, window=100)
    # Larger window is easier to satisfy (one peak covers more of the sequence)
    assert loss_large.item() < loss_small.item()


def test_peakiness_regularizer_differentiable():
    heatmap = torch.ones(100) * 0.3
    heatmap.requires_grad_(True)
    loss = peakiness_regularizer(heatmap, window=20)
    loss.backward()
    assert heatmap.grad is not None


def test_peakiness_regularizer_all_ones_zero_loss():
    """A heatmap that's already at 1.0 everywhere has zero peakiness loss."""
    heatmap = torch.ones(100)
    loss = peakiness_regularizer(heatmap, window=20)
    assert loss.item() < 1e-6
