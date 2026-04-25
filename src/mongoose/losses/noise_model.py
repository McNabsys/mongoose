"""NLL terms from the Nabsys nanodetector noise model.

Replaces the heuristic ``L_bp`` (soft-DTW), ``L_velocity`` (peak MSE), and
``L_count`` (smooth-L1) terms in :class:`CombinedLoss` with proper Gaussian /
proximity-aware likelihoods drawn from the production noise model.

The model treats each per-interval observation as
::

    L_pred,i ~ N(v * L_ref,i, S^2 * L_ref,i)

with ``S in [4.1, 5.5]`` and ``v`` a per-molecule stretch latent (Appendix 4
PDF: mean ~ 1.04, sigma ~ 0.07). Per-probe absolute positions carry an
additional Gaussian prior with sigma = 50 bp (Appendix 3). The expected
observed count is reduced by proximity loss (logistic with ``C=400``,
``R=300``) and inflated by an endogenous false-positive Poisson (mean
interval ``100,000`` bp).

Reference: ``support/Algorithm Confluence Export/Appendices/
NanodetectorReadErrorModel_03.pdf``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# Operational bounds on the per-interval scale parameter S (bp^0.5).
S_MIN = 4.1
S_MAX = 5.5
# Default per-probe absolute position prior sigma (bp).
POSITION_SIGMA_BP = 50.0
# Stretch latent prior (Appendix 4 empirical PDF).
STRETCH_PRIOR_MEAN = 1.04
STRETCH_PRIOR_SIGMA = 0.07
# Proximity-loss logistic parameters (Appendix 5).
PROX_LOSS_CENTER_BP = 400.0
PROX_LOSS_RANGE_BP = 300.0
# Endogenous false-positive Poisson mean interval (Appendix 6).
FP_MEAN_INTERVAL_BP = 100_000.0


def stretch_ml_estimate(
    pred_intervals: torch.Tensor,
    ref_intervals: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ML estimate of the per-molecule stretch ``v``.

    Under ``L_pred,i ~ N(v * L_ref,i, S^2 * L_ref,i)`` the maximum-likelihood
    estimator (in v, with S held fixed) is
    ::

        v_ML = sum_i L_pred,i / sum_i L_ref,i

    -- the ratio of total predicted span to total reference span. This
    captures a single global stretch factor without needing a separate
    learnable parameter or head.

    Args:
        pred_intervals: ``[K-1]`` predicted interval lengths (bp).
        ref_intervals: ``[K-1]`` reference interval lengths (bp). Must be
            positive; caller should ``.clamp(min=1.0)`` upstream.

    Returns:
        Scalar tensor with the ML stretch. Returns ``1.0`` if there are no
        intervals (degenerate).
    """
    if pred_intervals.numel() == 0:
        return torch.ones((), device=pred_intervals.device, dtype=pred_intervals.dtype)
    pred_total = pred_intervals.sum()
    ref_total = ref_intervals.sum().clamp(min=1.0)
    return pred_total / ref_total


def interval_nll(
    pred_bp_at_peaks: torch.Tensor,
    ref_bp_positions: torch.Tensor,
    log_S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-interval Gaussian NLL with sigma = S * sqrt(L_ref) and ML stretch.

    Computes the normalized negative log-likelihood
    ::

        NLL = (1 / (K - 1)) * sum_i [
                (L_pred,i - v * L_ref,i)^2 / (S^2 * L_ref,i)
              + log(S^2 * L_ref,i)
            ] / 2

    where ``L_pred,i = pred_bp[i+1] - pred_bp[i]``,
    ``L_ref,i = abs(ref_bp[i+1] - ref_bp[i])``, ``v`` is the closed-form ML
    stretch from :func:`stretch_ml_estimate`, and ``S = exp(log_S)`` clamped
    to ``[S_MIN, S_MAX]``. Both predicted and reference positions are
    zero-anchored to their first probe (the absolute genomic offset is not
    learnable from the waveform).

    Args:
        pred_bp_at_peaks: ``[K]`` predicted cumulative bp at GT probe sample
            indices. Must be ordered.
        ref_bp_positions: ``[K]`` reference bp positions (forward direction)
            paired 1:1 with ``pred_bp_at_peaks``.
        log_S: scalar tensor; ``S = exp(log_S)`` clamped to operational
            bounds. Treated as a learnable parameter when wrapped in
            :class:`torch.nn.Parameter`.

    Returns:
        ``(nll_mean, v_ML)`` -- the per-interval-mean NLL and the ML stretch
        estimate. The stretch is returned so callers can apply
        :func:`stretch_prior_nll` to it.
    """
    if pred_bp_at_peaks.numel() < 2:
        zero = torch.zeros((), device=pred_bp_at_peaks.device, dtype=pred_bp_at_peaks.dtype)
        one = torch.ones((), device=pred_bp_at_peaks.device, dtype=pred_bp_at_peaks.dtype)
        return zero, one

    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
    ref_f = ref_bp_positions.to(dtype=pred_norm.dtype, device=pred_norm.device)
    ref_norm = (ref_f - ref_f[0]).abs()

    pred_intervals = pred_norm[1:] - pred_norm[:-1]
    ref_intervals = (ref_norm[1:] - ref_norm[:-1]).clamp(min=1.0)

    v = stretch_ml_estimate(pred_intervals, ref_intervals)

    S = torch.exp(log_S).clamp(min=S_MIN, max=S_MAX)
    sigma_sq = (S * S) * ref_intervals  # bp^2 per interval

    diff = pred_intervals - v * ref_intervals
    nll = 0.5 * (diff.pow(2) / sigma_sq + torch.log(sigma_sq))
    return nll.mean(), v


def position_nll(
    pred_bp_at_peaks: torch.Tensor,
    ref_bp_positions: torch.Tensor,
    sigma_bp: float = POSITION_SIGMA_BP,
) -> torch.Tensor:
    """Per-probe Gaussian NLL with fixed sigma on absolute position.

    ::

        NLL = (1 / K) * sum_i 0.5 * [(p_pred,i - p_ref,i)^2 / sigma^2 + log(sigma^2)]

    Both predicted and reference positions are zero-anchored at the first
    probe -- the loss penalizes relative position error, not absolute
    genomic offset.

    Args:
        pred_bp_at_peaks: ``[K]`` predicted cumulative bp at GT probe
            sample indices.
        ref_bp_positions: ``[K]`` reference bp positions paired 1:1 with
            the predictions.
        sigma_bp: Per-probe Gaussian std in bp. Default 50 bp from the
            noise model appendix.

    Returns:
        Scalar mean per-probe NLL (zero if ``K == 0``).
    """
    if pred_bp_at_peaks.numel() == 0:
        return torch.zeros((), device=pred_bp_at_peaks.device, dtype=pred_bp_at_peaks.dtype)

    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
    ref_f = ref_bp_positions.to(dtype=pred_norm.dtype, device=pred_norm.device)
    ref_norm = (ref_f - ref_f[0]).abs()

    sigma_sq = float(sigma_bp) ** 2
    diff = pred_norm - ref_norm
    nll = 0.5 * (diff.pow(2) / sigma_sq + math.log(sigma_sq))
    return nll.mean()


def stretch_prior_nll(
    v: torch.Tensor,
    prior_mean: float = STRETCH_PRIOR_MEAN,
    prior_sigma: float = STRETCH_PRIOR_SIGMA,
) -> torch.Tensor:
    """Gaussian prior NLL on the per-molecule stretch ``v``.

    ::

        NLL = 0.5 * ((v - mu) / sigma)^2

    The constant ``log(sigma^2) / 2`` is omitted because it does not affect
    optimization. Drawn from the Appendix-4 empirical PDF (``mu ~ 1.04``,
    ``sigma ~ 0.07``, range ``[0.7, 1.3]``).

    Args:
        v: Scalar tensor with the stretch (typically the ML estimate from
            :func:`stretch_ml_estimate`).
        prior_mean: Center of the prior (bp/bp dimensionless).
        prior_sigma: Std of the prior.

    Returns:
        Scalar NLL.
    """
    return 0.5 * ((v - prior_mean) / prior_sigma).pow(2)


def proximity_loss_probability(
    interval_bp: torch.Tensor,
    center_bp: float = PROX_LOSS_CENTER_BP,
    range_bp: float = PROX_LOSS_RANGE_BP,
) -> torch.Tensor:
    """Per-interval probability that the two endpoints are NOT resolved.

    The noise model uses a logistic CDF for the probability that two
    adjacent probes are distinguishable:
    ::

        tau = R / (4 * ln 3)
        P_resolved(d) = 1 / (1 + exp(-(d - C) / tau))
        P_loss(d) = 1 - P_resolved(d)

    With ``C = 400 bp`` and ``R = 300 bp`` (Appendix 5), the loss
    probability is ~0.5 at d=400 and falls off quickly with distance.

    Args:
        interval_bp: ``[K-1]`` interval lengths in bp.
        center_bp: Logistic midpoint.
        range_bp: Logistic transition range.

    Returns:
        ``[K-1]`` proximity-loss probabilities, in ``[0, 1]``.
    """
    tau = float(range_bp) / (4.0 * math.log(3.0))
    interval_f = interval_bp.to(dtype=torch.float32)
    p_resolved = torch.sigmoid((interval_f - float(center_bp)) / tau)
    return 1.0 - p_resolved


def proximity_aware_count_loss(
    predicted_count: torch.Tensor,
    ref_intervals_bp: torch.Tensor,
    mol_length_bp: torch.Tensor,
    n_ref_probes: int,
    fp_mean_interval_bp: float = FP_MEAN_INTERVAL_BP,
    huber_delta: float = 5.0,
) -> torch.Tensor:
    """Smooth L1 between predicted count and proximity-aware expected count.

    Replaces the prior smooth-L1-against-``n_ref_probes`` term with a target
    that respects the noise-model expectation:
    ::

        E[N_observed] = N_true * (1 - mean(P_loss(intervals)))
                      + lambda_FP * mol_length / 100_000

    where ``lambda_FP = 1`` from the appendix (Poisson rate of one
    endogenous false probe per 100 kb). This sets a more accurate target
    than ``n_ref_probes`` alone, and naturally penalizes the model when it
    fails to merge close-spaced peaks (``P_loss`` should be high so its
    negative contribution shrinks the expected count).

    Args:
        predicted_count: Scalar predicted probe count for the molecule
            (e.g. ``sum(heatmap)`` or a separate count head's output).
        ref_intervals_bp: ``[K-1]`` reference interval lengths.
        mol_length_bp: Scalar molecule length in bp.
        n_ref_probes: Number of reference probes (integer).
        fp_mean_interval_bp: Mean interval between endogenous FP probes.
        huber_delta: Smooth-L1 transition parameter.

    Returns:
        Scalar smooth-L1 loss between ``predicted_count`` and the
        proximity-aware expected count.
    """
    if ref_intervals_bp.numel() == 0:
        # Only one probe (or zero) -- expected count is just N_true plus FP.
        expected_fp = mol_length_bp.to(dtype=predicted_count.dtype) / float(fp_mean_interval_bp)
        expected = float(n_ref_probes) + expected_fp
    else:
        p_loss = proximity_loss_probability(ref_intervals_bp).to(dtype=predicted_count.dtype)
        expected_resolved = float(n_ref_probes) * (1.0 - p_loss.mean())
        expected_fp = mol_length_bp.to(dtype=predicted_count.dtype) / float(fp_mean_interval_bp)
        expected = expected_resolved + expected_fp

    return F.smooth_l1_loss(
        predicted_count.to(dtype=torch.float32),
        expected.to(dtype=torch.float32).detach(),
        beta=float(huber_delta),
    )
