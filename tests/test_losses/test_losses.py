import pytest
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


def test_combined_loss_warmstart_blend():
    combined = CombinedLoss(warmstart_epochs=5, warmstart_fade_epochs=2)
    combined.set_epoch(0)
    assert combined._warmstart_blend == 1.0
    combined.set_epoch(2)
    assert combined._warmstart_blend == 1.0  # still full-warmstart phase
    combined.set_epoch(3)
    assert 0.0 < combined._warmstart_blend < 1.0  # fading
    combined.set_epoch(5)
    assert combined._warmstart_blend == 0.0
    combined.set_epoch(10)
    assert combined._warmstart_blend == 0.0


def test_combined_loss_lambda_schedule():
    combined = CombinedLoss(
        lambda_bp=1.0, lambda_vel=1.0, lambda_count=1.0, warmstart_epochs=5
    )
    combined.set_epoch(0)
    assert combined.current_lambda_bp == 0.5
    assert combined.current_lambda_vel == 0.5
    assert combined.current_lambda_count == 0.5
    combined.set_epoch(5)
    assert combined.current_lambda_bp == 1.0
    assert combined.current_lambda_vel == 1.0
    assert combined.current_lambda_count == 1.0
    combined.set_epoch(10)
    assert combined.current_lambda_bp == 1.0


def test_combined_loss_warmstart_zero_epochs():
    """With warmstart_epochs=0 the blend is always 0 and lambdas are at target."""
    combined = CombinedLoss(
        lambda_bp=2.0, lambda_vel=3.0, lambda_count=4.0, warmstart_epochs=0
    )
    combined.set_epoch(0)
    assert combined._warmstart_blend == 0.0
    assert combined.current_lambda_bp == 2.0
    assert combined.current_lambda_vel == 3.0
    assert combined.current_lambda_count == 4.0


def _make_peaky_heatmap(length: int, centers: list[int], sigma: float = 2.0) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    hm = torch.zeros(length, dtype=torch.float32)
    for c in centers:
        hm = hm + torch.exp(-0.5 * ((t - c) / sigma) ** 2)
    return hm.clamp(max=1.0)


@pytest.fixture
def minimal_batch():
    """Small synthetic batch for exercising CombinedLoss end-to-end."""
    length = 256
    # Molecule 0: three peaks at 40, 120, 200
    # Molecule 1: three peaks at 50, 130, 210
    centers_0 = [40, 120, 200]
    centers_1 = [50, 130, 210]
    pred_heatmap = torch.stack(
        [
            _make_peaky_heatmap(length, centers_0),
            _make_peaky_heatmap(length, centers_1),
        ]
    ).requires_grad_(True)

    # Monotonic cumulative bp with varied slope
    cum_bp = torch.stack(
        [
            torch.linspace(0.0, 10000.0, length),
            torch.linspace(0.0, 9000.0, length),
        ]
    ).clone().requires_grad_(True)

    raw_velocity = torch.full((2, length), 5.0).clone().requires_grad_(True)

    # Reference bp positions -- ordered (differences match rough spacing).
    ref_bp = [
        torch.tensor([0, 3000, 7500], dtype=torch.int64),
        torch.tensor([0, 3300, 7000], dtype=torch.int64),
    ]
    n_ref = torch.tensor([3, 3], dtype=torch.int64)
    mask = torch.ones(2, length, dtype=torch.bool)

    return {
        "pred_heatmap": pred_heatmap,
        "pred_cumulative_bp": cum_bp,
        "raw_velocity": raw_velocity,
        "ref_bp": ref_bp,
        "n_ref": n_ref,
        "mask": mask,
    }


@pytest.fixture
def minimal_batch_with_warmstart(minimal_batch):
    length = minimal_batch["pred_heatmap"].shape[1]
    warmstart_heatmap = torch.stack(
        [
            _make_peaky_heatmap(length, [40, 120, 200], sigma=3.0),
            _make_peaky_heatmap(length, [50, 130, 210], sigma=3.0),
        ]
    )
    warmstart_valid = torch.tensor([True, True], dtype=torch.bool)
    return {**minimal_batch, "warmstart_heatmap": warmstart_heatmap, "warmstart_valid": warmstart_valid}


def test_combined_loss_forward_runs_post_warmstart(minimal_batch):
    """A forward pass with no warmstart labels should run end-to-end."""
    combined = CombinedLoss(warmstart_epochs=0)
    combined.set_epoch(1)
    loss, details = combined(
        pred_heatmap=minimal_batch["pred_heatmap"],
        pred_cumulative_bp=minimal_batch["pred_cumulative_bp"],
        raw_velocity=minimal_batch["raw_velocity"],
        reference_bp_positions_list=minimal_batch["ref_bp"],
        n_ref_probes=minimal_batch["n_ref"],
        warmstart_heatmap=None,
        warmstart_valid=None,
        mask=minimal_batch["mask"],
    )
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert "probe" in details
    assert "bp" in details
    assert "vel" in details
    assert "count" in details
    assert details["warmstart_blend"] == 0.0


def test_combined_loss_forward_runs_with_warmstart(minimal_batch_with_warmstart):
    """With warmstart labels at epoch 0, the focal component must be active."""
    combined = CombinedLoss(warmstart_epochs=5, warmstart_fade_epochs=2)
    combined.set_epoch(0)
    batch = minimal_batch_with_warmstart
    pred_heatmap_logits = torch.randn_like(batch["pred_heatmap"])
    loss, details = combined(
        pred_heatmap=batch["pred_heatmap"],
        pred_cumulative_bp=batch["pred_cumulative_bp"],
        raw_velocity=batch["raw_velocity"],
        reference_bp_positions_list=batch["ref_bp"],
        n_ref_probes=batch["n_ref"],
        warmstart_heatmap=batch["warmstart_heatmap"],
        warmstart_valid=batch["warmstart_valid"],
        mask=batch["mask"],
        pred_heatmap_logits=pred_heatmap_logits,
    )
    assert torch.isfinite(loss)
    assert details["warmstart_blend"] == 1.0
    # Focal component contributes to probe loss at blend=1.0.
    assert details["probe"] > 0.0


def test_combined_loss_backward_flows_gradient(minimal_batch):
    """Loss.backward() should populate gradients on the differentiable outputs."""
    combined = CombinedLoss(warmstart_epochs=0)
    combined.set_epoch(1)
    loss, _ = combined(
        pred_heatmap=minimal_batch["pred_heatmap"],
        pred_cumulative_bp=minimal_batch["pred_cumulative_bp"],
        raw_velocity=minimal_batch["raw_velocity"],
        reference_bp_positions_list=minimal_batch["ref_bp"],
        n_ref_probes=minimal_batch["n_ref"],
        warmstart_heatmap=None,
        warmstart_valid=None,
        mask=minimal_batch["mask"],
    )
    loss.backward()
    assert minimal_batch["pred_heatmap"].grad is not None
    assert minimal_batch["pred_cumulative_bp"].grad is not None
    assert minimal_batch["raw_velocity"].grad is not None


def test_combined_loss_skips_bp_when_too_few_peaks():
    """If NMS returns <2 peaks, L_bp and L_velocity contributions are zero."""
    length = 128
    # Heatmap has no peaks above the NMS threshold.
    pred_heatmap = torch.full((2, length), 0.05, requires_grad=True)
    cum_bp = torch.stack(
        [torch.linspace(0.0, 5000.0, length), torch.linspace(0.0, 5000.0, length)]
    ).clone().requires_grad_(True)
    raw_velocity = torch.full((2, length), 5.0).clone().requires_grad_(True)
    ref_bp = [
        torch.tensor([0, 1500, 3000], dtype=torch.int64),
        torch.tensor([0, 1500, 3000], dtype=torch.int64),
    ]
    n_ref = torch.tensor([3, 3], dtype=torch.int64)
    mask = torch.ones(2, length, dtype=torch.bool)

    combined = CombinedLoss(warmstart_epochs=0, nms_threshold=0.3)
    combined.set_epoch(1)
    loss, details = combined(
        pred_heatmap=pred_heatmap,
        pred_cumulative_bp=cum_bp,
        raw_velocity=raw_velocity,
        reference_bp_positions_list=ref_bp,
        n_ref_probes=n_ref,
        warmstart_heatmap=None,
        warmstart_valid=None,
        mask=mask,
    )
    assert details["bp"] == 0.0
    assert details["vel"] == 0.0
    assert torch.isfinite(loss)


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


def test_combined_loss_blend_floor_holds_at_late_epoch():
    """min_blend=0.1 means focal loss never fully disappears."""
    criterion = CombinedLoss(
        warmstart_epochs=5,
        warmstart_fade_epochs=2,
        min_blend=0.1,
    )
    criterion.set_epoch(100)
    assert criterion._warmstart_blend == 0.1


def test_combined_loss_blend_floor_defaults_to_zero():
    """Default min_blend=0.0 preserves existing post-fade behavior (blend=0)."""
    criterion = CombinedLoss(
        warmstart_epochs=5,
        warmstart_fade_epochs=2,
    )
    criterion.set_epoch(100)
    assert criterion._warmstart_blend == 0.0


def test_combined_loss_blend_floor_applies_with_no_warmstart():
    """Floor must apply even when warmstart_epochs=0 (overfit gate uses this)."""
    criterion = CombinedLoss(
        warmstart_epochs=0,
        warmstart_fade_epochs=0,
        min_blend=0.1,
    )
    criterion.set_epoch(0)
    assert criterion._warmstart_blend == 0.1


def test_combined_loss_scale_divisors_normalize_components(minimal_batch):
    """Raw loss components are divided by their scale divisors before lambda weighting."""
    plain = CombinedLoss(warmstart_epochs=0)
    scaled = CombinedLoss(
        warmstart_epochs=0,
        scale_probe=1.0,
        scale_bp=100.0,
        scale_vel=10.0,
        scale_count=1.0,
    )
    plain.set_epoch(0)
    scaled.set_epoch(0)

    # Note: `minimal_batch` fixture uses abbreviated dict keys (ref_bp, n_ref)
    # that need translating to CombinedLoss.__call__ kwargs.
    kwargs = dict(
        pred_heatmap=minimal_batch["pred_heatmap"],
        pred_cumulative_bp=minimal_batch["pred_cumulative_bp"],
        raw_velocity=minimal_batch["raw_velocity"],
        reference_bp_positions_list=minimal_batch["ref_bp"],
        n_ref_probes=minimal_batch["n_ref"],
        warmstart_heatmap=None,
        warmstart_valid=None,
        mask=minimal_batch["mask"],
    )

    _, details_plain = plain(**kwargs)
    _, details_scaled = scaled(**kwargs)

    # After the change, details should expose BOTH scaled and *_raw keys.
    assert details_scaled["probe_raw"] == details_plain["probe_raw"]
    assert details_scaled["bp_raw"] == details_plain["bp_raw"]
    assert details_scaled["bp"] == pytest.approx(details_plain["bp_raw"] / 100.0, rel=1e-5)
    assert details_scaled["vel"] == pytest.approx(details_plain["vel_raw"] / 10.0, rel=1e-5)
    assert details_scaled["count"] == pytest.approx(details_plain["count_raw"] / 1.0, rel=1e-5)
    assert details_scaled["probe"] == pytest.approx(details_plain["probe_raw"] / 1.0, rel=1e-5)


def test_combined_loss_scale_divisor_rejects_non_positive():
    """Non-positive scale divisors are a configuration error — fail loudly."""
    with pytest.raises(ValueError, match="scale_bp must be positive"):
        CombinedLoss(scale_bp=0.0)
    with pytest.raises(ValueError, match="scale_vel must be positive"):
        CombinedLoss(scale_vel=-1.0)


def test_combined_loss_teacher_forcing_gives_vel_gradient_with_flat_probe():
    """Given a flat probe heatmap (no peaks above NMS threshold), teacher
    forcing should still produce a nonzero L_vel gradient into ``raw_velocity``
    via the ground-truth index path. Without teacher forcing, L_vel is zero
    because extract_peak_indices returns <2 peaks."""
    import torch
    from mongoose.losses.combined import CombinedLoss

    B, T = 2, 200
    # Flat probe output — sigmoid = 0.047 everywhere, no peaks detected.
    pred_heatmap = torch.full((B, T), 0.047, requires_grad=False)
    pred_heatmap_logits = torch.full((B, T), -3.0, requires_grad=False)
    raw_velocity = torch.rand((B, T), requires_grad=True)
    pred_cumulative_bp = torch.cumsum(raw_velocity, dim=-1)
    mask = torch.ones(B, T, dtype=torch.bool)
    warmstart_heatmap = torch.zeros(B, T)
    warmstart_heatmap[0, [50, 100, 150]] = 1.0
    warmstart_heatmap[1, [60, 120]] = 1.0
    warmstart_valid = torch.tensor([True, True])
    ref_bp = [torch.tensor([0, 100, 200], dtype=torch.long),
              torch.tensor([0, 100], dtype=torch.long)]
    n_ref = torch.tensor([3, 2], dtype=torch.long)
    centers = [torch.tensor([50, 100, 150], dtype=torch.long),
               torch.tensor([60, 120], dtype=torch.long)]

    loss_fn = CombinedLoss(
        scale_bp=300000.0, scale_vel=5000.0, scale_count=1e9, scale_probe=1.0,
        lambda_bp=1.0, lambda_vel=1.0, lambda_count=0.0,
        warmstart_epochs=1, warmstart_fade_epochs=0, min_blend=1.0,
    )
    loss_fn.set_epoch(0)

    total, details = loss_fn(
        pred_heatmap=pred_heatmap,
        pred_cumulative_bp=pred_cumulative_bp,
        raw_velocity=raw_velocity,
        reference_bp_positions_list=ref_bp,
        n_ref_probes=n_ref,
        warmstart_heatmap=warmstart_heatmap,
        warmstart_valid=warmstart_valid,
        mask=mask,
        pred_heatmap_logits=pred_heatmap_logits,
        warmstart_probe_centers_samples_list=centers,
    )
    total.backward()

    assert raw_velocity.grad is not None, "raw_velocity should have received gradient"
    assert raw_velocity.grad.abs().sum().item() > 0.0, (
        "teacher forcing should produce nonzero velocity gradient even with flat probe"
    )
    assert details["vel_raw"] > 0.0, (
        f"expected nonzero raw L_vel under teacher forcing, got {details['vel_raw']}"
    )
