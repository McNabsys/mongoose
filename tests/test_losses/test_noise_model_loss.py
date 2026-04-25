"""Tests for the V4 ``NoiseModelLoss`` composite class."""
from __future__ import annotations

import math

import pytest
import torch

from mongoose.losses.noise_model import S_MAX, S_MIN
from mongoose.losses.noise_model_loss import NoiseModelLoss


def _build_synthetic_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 200,
    n_probes_per_mol: int = 4,
    pred_matches_ref: bool = True,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Construct a minimal synthetic batch shaped like the trainer's output.

    Layout: each molecule has ``n_probes_per_mol`` probes evenly spaced on
    the waveform. Reference bp positions form an arithmetic progression
    starting at ``1000`` with step ``1500``.

    When ``pred_matches_ref`` is True the predicted cumulative_bp ramps
    linearly so that ``pred_cumulative_bp[gt_centers]`` exactly matches the
    reference -- useful for checking the loss collapses to its constant
    log term.
    """
    torch.manual_seed(seed)
    device = "cpu"
    dtype = torch.float32

    pred_heatmap_logits = torch.full((batch_size, seq_len), -3.0, dtype=dtype)
    pred_heatmap = torch.sigmoid(pred_heatmap_logits)

    # Per-molecule probe sample indices, evenly spaced inside [10, seq_len-10].
    pad = 10
    spacing = (seq_len - 2 * pad) // (n_probes_per_mol - 1)
    centers = torch.tensor(
        [pad + i * spacing for i in range(n_probes_per_mol)], dtype=torch.long
    )
    warmstart_centers = [centers.clone() for _ in range(batch_size)]

    # Reference bp positions (forward direction): 1000, 2500, 4000, 5500, ...
    ref_bp = torch.tensor(
        [1000.0 + i * 1500.0 for i in range(n_probes_per_mol)], dtype=dtype
    )
    reference_bp = [ref_bp.clone() for _ in range(batch_size)]

    if pred_matches_ref:
        # Build a cumulative_bp that linearly interpolates between ref points.
        # bp_at_sample[c_i] = ref_bp[i]; in between, linear ramp.
        cum_bp = torch.zeros(seq_len, dtype=dtype)
        for i in range(n_probes_per_mol - 1):
            lo = int(centers[i].item())
            hi = int(centers[i + 1].item())
            cum_bp[lo : hi + 1] = torch.linspace(
                ref_bp[i].item(), ref_bp[i + 1].item(), hi - lo + 1
            )
        # Tail of waveform constant at last ref value.
        cum_bp[hi:] = ref_bp[-1]
        # Head of waveform constant at first ref value.
        cum_bp[: int(centers[0].item())] = ref_bp[0]
        pred_cumulative_bp = cum_bp.unsqueeze(0).expand(batch_size, -1).contiguous()

        # Velocity is the diff (zero-pad first sample).
        raw_velocity = torch.zeros_like(pred_cumulative_bp)
        raw_velocity[:, 1:] = pred_cumulative_bp[:, 1:] - pred_cumulative_bp[:, :-1]
    else:
        # Off-by-1000bp on the second-to-last interval. v_ML still close to 1.
        pred_cumulative_bp = torch.zeros((batch_size, seq_len), dtype=dtype)
        for i in range(n_probes_per_mol - 1):
            lo = int(centers[i].item())
            hi = int(centers[i + 1].item())
            ref_lo = ref_bp[i].item()
            ref_hi = ref_bp[i + 1].item()
            if i == n_probes_per_mol - 2:
                ref_hi += 1000.0  # injected error
            pred_cumulative_bp[:, lo : hi + 1] = torch.linspace(
                ref_lo, ref_hi, hi - lo + 1
            )
        pred_cumulative_bp[:, hi:] = pred_cumulative_bp[:, hi].unsqueeze(-1)
        raw_velocity = torch.zeros_like(pred_cumulative_bp)
        raw_velocity[:, 1:] = pred_cumulative_bp[:, 1:] - pred_cumulative_bp[:, :-1]

    # Warmstart heatmap: small Gaussians at GT centers.
    warmstart_heatmap = torch.zeros((batch_size, seq_len), dtype=dtype)
    sigma = 3.0
    for b in range(batch_size):
        x = torch.arange(seq_len, dtype=dtype)
        for c in centers:
            g = torch.exp(-0.5 * ((x - c.float()) / sigma) ** 2)
            warmstart_heatmap[b] = torch.maximum(warmstart_heatmap[b], g)

    warmstart_valid = torch.ones(batch_size, dtype=torch.bool)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    n_ref_probes = torch.tensor([n_probes_per_mol] * batch_size, dtype=torch.int64)

    return {
        "pred_heatmap": pred_heatmap,
        "pred_heatmap_logits": pred_heatmap_logits,
        "pred_cumulative_bp": pred_cumulative_bp,
        "raw_velocity": raw_velocity,
        "reference_bp_positions_list": reference_bp,
        "n_ref_probes": n_ref_probes,
        "warmstart_heatmap": warmstart_heatmap,
        "warmstart_valid": warmstart_valid,
        "mask": mask,
        "warmstart_probe_centers_samples_list": warmstart_centers,
    }


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_constructor_registers_log_S_as_parameter():
    loss = NoiseModelLoss(S_init=5.0)
    params = list(loss.parameters())
    assert len(params) == 1
    assert params[0] is loss.log_S
    assert params[0].requires_grad


def test_constructor_rejects_S_init_outside_range():
    with pytest.raises(ValueError, match="outside operational range"):
        NoiseModelLoss(S_init=10.0)
    with pytest.raises(ValueError, match="outside operational range"):
        NoiseModelLoss(S_init=2.0)


def test_constructor_rejects_nonpositive_scales():
    with pytest.raises(ValueError, match="must be positive"):
        NoiseModelLoss(scale_probe=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        NoiseModelLoss(scale_bp=-1.0)


# --------------------------------------------------------------------------
# Schedulers (interface match with CombinedLoss)
# --------------------------------------------------------------------------

def test_set_epoch_blend_full_during_warmstart():
    loss = NoiseModelLoss(warmstart_epochs=5, warmstart_fade_epochs=2)
    loss.set_epoch(0)
    assert loss._warmstart_blend == 1.0


def test_set_epoch_blend_zero_after_warmstart():
    loss = NoiseModelLoss(warmstart_epochs=5, warmstart_fade_epochs=2)
    loss.set_epoch(10)
    assert loss._warmstart_blend == 0.0


def test_set_epoch_blend_floor_via_min_blend():
    loss = NoiseModelLoss(warmstart_epochs=5, warmstart_fade_epochs=2, min_blend=0.05)
    loss.set_epoch(10)
    assert loss._warmstart_blend == 0.05


def test_set_epoch_lambda_ramps_up_during_warmstart():
    # Schedule (matches CombinedLoss): scale = 0.5 + 0.5 * (epoch/W) inside
    # warmstart, then 1.0 after. So with W=4, lambda=2.0:
    #   epoch 0 -> scale 0.50 -> lambda 1.0
    #   epoch 2 -> scale 0.75 -> lambda 1.5
    #   epoch 4 -> scale 1.00 -> lambda 2.0
    loss = NoiseModelLoss(
        lambda_bp=2.0, lambda_vel=2.0, lambda_count=2.0, warmstart_epochs=4
    )
    loss.set_epoch(0)
    assert loss.current_lambda_bp == pytest.approx(1.0)
    loss.set_epoch(2)
    assert loss.current_lambda_bp == pytest.approx(1.5)
    loss.set_epoch(4)
    assert loss.current_lambda_bp == pytest.approx(2.0)
    loss.set_epoch(10)
    assert loss.current_lambda_bp == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------

def test_forward_returns_finite_loss_and_expected_keys():
    loss = NoiseModelLoss(warmstart_epochs=5)
    loss.set_epoch(2)
    batch = _build_synthetic_batch(pred_matches_ref=True)

    total, details = loss(**batch)

    assert torch.isfinite(total).all()
    assert total.item() >= 0.0  # NLL terms are non-negative-ish (constants can be neg)

    expected_keys = {
        "probe",
        "bp",
        "vel",
        "count",
        "probe_raw",
        "bp_raw",
        "vel_raw",
        "count_raw",
        "stretch_prior",
        "stretch_v_ML",
        "S_value",
        "warmstart_blend",
    }
    assert expected_keys.issubset(details.keys())

    # All detail values must be float-cast finites.
    for k in expected_keys:
        v = details[k]
        assert isinstance(v, float)
        assert math.isfinite(v), f"detail {k!r} is not finite: {v!r}"


def test_forward_perfect_pred_recovers_v_close_to_one():
    loss = NoiseModelLoss(warmstart_epochs=5)
    loss.set_epoch(2)
    batch = _build_synthetic_batch(pred_matches_ref=True)

    _, details = loss(**batch)
    assert details["stretch_v_ML"] == pytest.approx(1.0, abs=1e-3)


def test_forward_S_value_in_operational_range():
    loss = NoiseModelLoss(S_init=5.0, warmstart_epochs=5)
    loss.set_epoch(2)
    batch = _build_synthetic_batch()

    _, details = loss(**batch)
    assert S_MIN - 1e-3 <= details["S_value"] <= S_MAX + 1e-3


def test_forward_loss_increases_when_pred_diverges_from_ref():
    loss = NoiseModelLoss(warmstart_epochs=0)  # full loss weight
    loss.set_epoch(10)

    batch_ok = _build_synthetic_batch(pred_matches_ref=True)
    batch_bad = _build_synthetic_batch(pred_matches_ref=False)

    total_ok, _ = loss(**batch_ok)
    total_bad, _ = loss(**batch_bad)

    assert total_bad.item() > total_ok.item()


def test_forward_logits_required_when_warmstart_active():
    loss = NoiseModelLoss(warmstart_epochs=5)
    loss.set_epoch(0)  # blend = 1.0
    batch = _build_synthetic_batch()
    batch["pred_heatmap_logits"] = None

    with pytest.raises(ValueError, match="requires pred_heatmap_logits"):
        loss(**batch)


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------

def test_backward_flows_to_log_S():
    loss = NoiseModelLoss(warmstart_epochs=5, lambda_bp=1.0)
    loss.set_epoch(2)
    batch = _build_synthetic_batch(pred_matches_ref=False)

    total, _ = loss(**batch)
    total.backward()

    assert loss.log_S.grad is not None
    assert torch.isfinite(loss.log_S.grad).all()


def test_backward_flows_to_pred_cumulative_bp():
    loss = NoiseModelLoss(warmstart_epochs=5, lambda_bp=1.0)
    loss.set_epoch(2)
    batch = _build_synthetic_batch(pred_matches_ref=False)

    # Make pred_cumulative_bp a leaf with grad.
    batch["pred_cumulative_bp"] = batch["pred_cumulative_bp"].detach().requires_grad_(True)
    total, _ = loss(**batch)
    total.backward()

    assert batch["pred_cumulative_bp"].grad is not None
    assert torch.isfinite(batch["pred_cumulative_bp"].grad).all()
    assert (batch["pred_cumulative_bp"].grad.abs() > 0).any()


def test_backward_flows_to_pred_heatmap_via_count_term():
    """The count term reads ``sum(heatmap)``; gradient must flow back."""
    loss = NoiseModelLoss(
        warmstart_epochs=0,  # no probe loss path to confound
        lambda_bp=0.0,  # isolate count gradient
        lambda_vel=0.0,
        lambda_count=1.0,
        lambda_stretch_prior=0.0,
        scale_count=1.0,
    )
    loss.set_epoch(10)

    batch = _build_synthetic_batch(pred_matches_ref=True)
    # Make heatmap a leaf with grad. Detach the cumulative_bp / velocity to
    # keep this test focused on the count term's gradient.
    batch["pred_heatmap"] = batch["pred_heatmap"].detach().requires_grad_(True)

    total, _ = loss(**batch)
    total.backward()

    assert batch["pred_heatmap"].grad is not None
    assert torch.isfinite(batch["pred_heatmap"].grad).all()
    assert (batch["pred_heatmap"].grad.abs() > 0).any()
