"""Tests for the Nabsys nanodetector noise-model NLL terms."""
from __future__ import annotations

import math

import pytest
import torch

from mongoose.losses.noise_model import (
    FP_MEAN_INTERVAL_BP,
    POSITION_SIGMA_BP,
    PROX_LOSS_CENTER_BP,
    PROX_LOSS_RANGE_BP,
    S_MAX,
    S_MIN,
    STRETCH_PRIOR_MEAN,
    STRETCH_PRIOR_SIGMA,
    interval_nll,
    position_nll,
    proximity_aware_count_loss,
    proximity_loss_probability,
    stretch_ml_estimate,
    stretch_prior_nll,
)


# --------------------------------------------------------------------------
# stretch_ml_estimate
# --------------------------------------------------------------------------

def test_stretch_ml_estimate_pred_equals_ref_returns_one():
    pred = torch.tensor([100.0, 200.0, 300.0])
    ref = torch.tensor([100.0, 200.0, 300.0])
    v = stretch_ml_estimate(pred, ref)
    assert v.item() == pytest.approx(1.0, abs=1e-6)


def test_stretch_ml_estimate_pred_double_ref_returns_two():
    pred = torch.tensor([200.0, 400.0, 600.0])
    ref = torch.tensor([100.0, 200.0, 300.0])
    v = stretch_ml_estimate(pred, ref)
    assert v.item() == pytest.approx(2.0, abs=1e-6)


def test_stretch_ml_estimate_empty_returns_one():
    v = stretch_ml_estimate(torch.empty(0), torch.empty(0))
    assert v.item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# interval_nll
# --------------------------------------------------------------------------

def test_interval_nll_below_two_probes_returns_zero():
    pred = torch.tensor([1000.0])
    ref = torch.tensor([5000.0])
    log_S = torch.log(torch.tensor(5.0))
    nll, v = interval_nll(pred, ref, log_S)
    assert nll.item() == 0.0
    assert v.item() == pytest.approx(1.0)


def test_interval_nll_perfect_match_is_only_log_term():
    # Perfect match: pred intervals = ref intervals, so v_ML = 1 and the
    # quadratic term vanishes. NLL collapses to the constant log term.
    intervals = [1000.0, 2000.0, 1500.0]
    pred_pos = torch.tensor([0.0] + list(torch.cumsum(torch.tensor(intervals), dim=0)))
    ref_pos = pred_pos.clone()
    log_S = torch.log(torch.tensor(5.0))

    nll, v = interval_nll(pred_pos, ref_pos, log_S)

    # Stretch must be 1 when pred equals ref
    assert v.item() == pytest.approx(1.0, abs=1e-5)

    # Expected: mean over intervals of 0.5 * log(S^2 * L_ref)
    S_sq = 25.0
    expected_per_interval = [0.5 * math.log(S_sq * L) for L in intervals]
    expected = sum(expected_per_interval) / len(expected_per_interval)
    assert nll.item() == pytest.approx(expected, rel=1e-4)


def test_interval_nll_one_sigma_off_adds_half():
    # Construct preds that are exactly +1 sigma off in every interval.
    # sigma_i = S * sqrt(L_ref,i), so adding S * sqrt(L_ref,i) to each
    # predicted interval produces a per-interval (z-score)^2 == 1, i.e.
    # an extra 0.5 in the NLL on top of the constant log term.
    S_val = 5.0
    log_S = torch.log(torch.tensor(S_val))
    ref_intervals = torch.tensor([1000.0, 2000.0, 4000.0])
    deviations = S_val * torch.sqrt(ref_intervals)
    # Shift each interval by sigma_i so v_ML stays close to 1.
    pred_intervals = ref_intervals + deviations
    pred_pos = torch.cat([torch.zeros(1), torch.cumsum(pred_intervals, dim=0)])
    ref_pos = torch.cat([torch.zeros(1), torch.cumsum(ref_intervals, dim=0)])

    nll, _ = interval_nll(pred_pos, ref_pos, log_S)

    # Expected: per-interval NLL = 0.5 * (1 + log(S^2 * L_ref))
    # but v_ML absorbs some of the deviation, so we expect a value close to
    # 0.5 * mean(1 + log(S^2 * L_ref)) but slightly less.
    log_terms = 0.5 * torch.log(S_val * S_val * ref_intervals)
    upper_bound = (0.5 + log_terms).mean().item()
    lower_bound = log_terms.mean().item()
    assert lower_bound < nll.item() < upper_bound


def test_interval_nll_S_clamped_to_operational_range():
    # Even if log_S is wildly outside [log(S_MIN), log(S_MAX)], the clamp
    # should keep S in range so the NLL is finite and consistent.
    pred = torch.tensor([0.0, 1000.0, 3000.0])
    ref = torch.tensor([0.0, 1000.0, 3000.0])

    nll_huge, _ = interval_nll(pred, ref, torch.log(torch.tensor(1000.0)))  # clamps to S_MAX
    nll_at_max, _ = interval_nll(pred, ref, torch.log(torch.tensor(S_MAX)))
    assert nll_huge.item() == pytest.approx(nll_at_max.item(), abs=1e-4)

    nll_tiny, _ = interval_nll(pred, ref, torch.log(torch.tensor(0.01)))  # clamps to S_MIN
    nll_at_min, _ = interval_nll(pred, ref, torch.log(torch.tensor(S_MIN)))
    assert nll_tiny.item() == pytest.approx(nll_at_min.item(), abs=1e-4)


# --------------------------------------------------------------------------
# position_nll
# --------------------------------------------------------------------------

def test_position_nll_perfect_match_is_only_log_term():
    pred = torch.tensor([0.0, 1000.0, 3000.0])
    ref = pred.clone()

    loss = position_nll(pred, ref, sigma_bp=50.0)
    expected = 0.5 * math.log(50.0 ** 2)
    assert loss.item() == pytest.approx(expected, rel=1e-4)


def test_position_nll_one_sigma_off_adds_half():
    pred = torch.tensor([0.0, 1050.0])
    ref = torch.tensor([0.0, 1000.0])

    loss = position_nll(pred, ref, sigma_bp=50.0)
    # Per-probe NLL: 0.5 * (z^2 + log(sigma^2)). Probe 0 has z=0, probe 1
    # has z=1. Mean = 0.5 * (0 + 1)/2 + 0.5 * log(2500) = 0.25 + 0.5*log(2500)
    expected = 0.25 + 0.5 * math.log(2500.0)
    assert loss.item() == pytest.approx(expected, rel=1e-4)


def test_position_nll_empty_returns_zero():
    loss = position_nll(torch.empty(0), torch.empty(0))
    assert loss.item() == 0.0


# --------------------------------------------------------------------------
# stretch_prior_nll
# --------------------------------------------------------------------------

def test_stretch_prior_nll_at_mean_is_zero():
    v = torch.tensor(STRETCH_PRIOR_MEAN)
    assert stretch_prior_nll(v).item() == pytest.approx(0.0, abs=1e-7)


def test_stretch_prior_nll_one_sigma_off_is_half():
    v = torch.tensor(STRETCH_PRIOR_MEAN + STRETCH_PRIOR_SIGMA)
    assert stretch_prior_nll(v).item() == pytest.approx(0.5, abs=1e-6)

    v = torch.tensor(STRETCH_PRIOR_MEAN - STRETCH_PRIOR_SIGMA)
    assert stretch_prior_nll(v).item() == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------
# proximity_loss_probability
# --------------------------------------------------------------------------

def test_proximity_loss_at_center_is_half():
    intervals = torch.tensor([PROX_LOSS_CENTER_BP])
    p = proximity_loss_probability(intervals)
    assert p[0].item() == pytest.approx(0.5, abs=1e-6)


def test_proximity_loss_at_center_plus_range_is_small():
    # tau = R / (4 ln 3); at d = C + R the logit is +4 ln 3
    # => P_resolved = 1/(1+1/81) ~= 0.988 => P_loss ~= 0.012
    intervals = torch.tensor([PROX_LOSS_CENTER_BP + PROX_LOSS_RANGE_BP])
    p = proximity_loss_probability(intervals)
    assert p[0].item() == pytest.approx(1.0 / 82.0, abs=1e-3)


def test_proximity_loss_at_center_minus_range_is_near_one():
    # Symmetric: at d = C - R, P_loss ~= 0.988
    intervals = torch.tensor([PROX_LOSS_CENTER_BP - PROX_LOSS_RANGE_BP])
    p = proximity_loss_probability(intervals)
    assert p[0].item() == pytest.approx(81.0 / 82.0, abs=1e-3)


def test_proximity_loss_monotonic_decreasing_in_distance():
    intervals = torch.linspace(50.0, 5000.0, 50)
    p = proximity_loss_probability(intervals)
    diffs = p[1:] - p[:-1]
    assert (diffs <= 0).all(), "P_loss must be non-increasing as interval grows"


# --------------------------------------------------------------------------
# proximity_aware_count_loss
# --------------------------------------------------------------------------

def test_proximity_aware_count_loss_zero_when_pred_matches_expected():
    # All intervals far apart -> P_loss ~ 0, expected ~ N_true + LM/100k.
    ref_intervals = torch.tensor([10_000.0, 10_000.0, 10_000.0])
    mol_length_bp = torch.tensor(50_000.0)
    n_ref = 4
    expected_fp = 50_000.0 / FP_MEAN_INTERVAL_BP  # 0.5
    expected_count = float(n_ref) + expected_fp  # ~ 4.5

    pred = torch.tensor(expected_count)
    loss = proximity_aware_count_loss(pred, ref_intervals, mol_length_bp, n_ref)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_proximity_aware_count_loss_close_intervals_reduce_expected():
    # Tightly-spaced probes -> high P_loss -> expected count drops below
    # n_ref. So the predicted count to hit zero loss is < n_ref, not == n_ref.
    ref_intervals = torch.tensor([100.0, 100.0, 100.0])  # well below C=400
    mol_length_bp = torch.tensor(1_000.0)
    n_ref = 4

    # Naive prediction at n_ref=4 should now be too high (not zero loss).
    naive_pred = torch.tensor(4.0)
    loss_naive = proximity_aware_count_loss(
        naive_pred, ref_intervals, mol_length_bp, n_ref
    )
    assert loss_naive.item() > 0.0

    # Predicting the proximity-discounted count should hit ~zero loss.
    p_loss = proximity_loss_probability(ref_intervals).mean()
    fp = float(mol_length_bp) / FP_MEAN_INTERVAL_BP
    expected = float(n_ref) * (1.0 - float(p_loss)) + fp
    pred_at_expected = torch.tensor(expected)
    loss_corrected = proximity_aware_count_loss(
        pred_at_expected, ref_intervals, mol_length_bp, n_ref
    )
    assert loss_corrected.item() == pytest.approx(0.0, abs=1e-3)


def test_proximity_aware_count_loss_no_intervals_falls_back_to_n_ref_plus_fp():
    # K=1 -> no intervals -> expected just N_true + LM/100k.
    pred = torch.tensor(1.5)
    mol_length_bp = torch.tensor(50_000.0)
    n_ref = 1
    loss = proximity_aware_count_loss(pred, torch.empty(0), mol_length_bp, n_ref)
    # Expected = 1 + 0.5 = 1.5, matches pred -> zero loss.
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------
# Differentiability sanity
# --------------------------------------------------------------------------

def test_interval_nll_gradient_flows_to_pred():
    pred = torch.tensor([0.0, 1000.0, 3000.0], requires_grad=True)
    ref = torch.tensor([0.0, 1100.0, 2800.0])
    log_S = torch.log(torch.tensor(5.0))

    nll, _ = interval_nll(pred, ref, log_S)
    nll.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad.abs() > 0).any()


def test_interval_nll_gradient_flows_to_log_S():
    pred = torch.tensor([0.0, 1000.0, 3000.0])
    ref = torch.tensor([0.0, 1100.0, 2800.0])
    log_S = torch.tensor(math.log(5.0), requires_grad=True)

    nll, _ = interval_nll(pred, ref, log_S)
    nll.backward()

    assert log_S.grad is not None
    assert torch.isfinite(log_S.grad).all()


def test_position_nll_gradient_flows_to_pred():
    pred = torch.tensor([0.0, 1000.0, 3000.0], requires_grad=True)
    ref = torch.tensor([0.0, 1050.0, 2950.0])

    loss = position_nll(pred, ref)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad.abs() > 0).any()
