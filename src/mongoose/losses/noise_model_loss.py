"""V4 ``NoiseModelLoss`` -- composite loss using the Nabsys noise model.

Sibling of :class:`CombinedLoss` and :class:`L511Loss` with a matching
``__call__`` signature so the trainer can swap classes without structural
changes. The probe head supervision (CenterNet focal + peakiness) is
unchanged from :class:`CombinedLoss`; the bp / velocity / count terms are
replaced with proper Gaussian-likelihood / proximity-aware NLLs drawn from
``NanodetectorReadErrorModel_03.pdf``.

Detail-key semantic mapping (so existing TB / CLI logging works unchanged):

* ``probe``: CenterNet focal blended with peakiness regularizer (unchanged)
* ``bp``: per-interval Gaussian NLL with sigma = S * sqrt(L_ref) (was soft-DTW)
* ``vel``: per-probe position Gaussian NLL with sigma = 50 bp (was peak MSE)
* ``count``: proximity-aware smooth-L1 count loss (was smooth-L1 vs n_ref)

New diagnostic keys added: ``stretch_v_ML`` (batch-mean ML stretch),
``S_value`` (current S), ``stretch_prior`` (prior NLL on stretches).

The learnable parameter :pyattr:`log_S` is registered as
``nn.Parameter``; the trainer must add ``criterion.parameters()`` to its
optimizer's param groups for it to be updated.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mongoose.losses.centernet_focal import centernet_focal_loss
from mongoose.losses.count import peakiness_regularizer
from mongoose.losses.noise_model import (
    S_MAX,
    S_MIN,
    interval_nll,
    position_nll,
    proximity_aware_count_loss,
    stretch_prior_nll,
)
from mongoose.losses.peaks import extract_peak_indices


class NoiseModelLoss(nn.Module):
    """Composite V4 loss with Nabsys-noise-model NLLs replacing heuristics.

    ``__call__`` and ``set_epoch`` interfaces match :class:`CombinedLoss`
    so the trainer can swap classes without structural change.
    """

    def __init__(
        self,
        *,
        lambda_bp: float = 1.0,
        lambda_vel: float = 1.0,
        lambda_count: float = 1.0,
        lambda_stretch_prior: float = 1.0,
        warmstart_epochs: int = 5,
        warmstart_fade_epochs: int = 2,
        peakiness_window: int = 20,
        nms_threshold: float = 0.3,
        tag_width_bp: float = 511.0,
        sample_period_ms: float = 0.025,
        position_sigma_bp: float = 50.0,
        S_init: float = 5.0,
        min_blend: float = 0.0,
        scale_probe: float = 1.0,
        scale_bp: float = 1.0,
        scale_vel: float = 1.0,
        scale_count: float = 1.0,
    ) -> None:
        super().__init__()

        # Component weights. ``current_lambda_*`` are updated by set_epoch
        # to provide the same warmstart -> full-loss ramp as CombinedLoss.
        self.lambda_bp = float(lambda_bp)
        self.lambda_vel = float(lambda_vel)
        self.lambda_count = float(lambda_count)
        self.lambda_stretch_prior = float(lambda_stretch_prior)

        self.warmstart_epochs = int(warmstart_epochs)
        self.warmstart_fade_epochs = int(warmstart_fade_epochs)
        self.peakiness_window = int(peakiness_window)
        self.nms_threshold = float(nms_threshold)
        self.tag_width_bp = float(tag_width_bp)
        self.sample_period_ms = float(sample_period_ms)
        self.position_sigma_bp = float(position_sigma_bp)
        self.min_blend = float(min_blend)

        for name, val in (
            ("scale_probe", scale_probe),
            ("scale_bp", scale_bp),
            ("scale_vel", scale_vel),
            ("scale_count", scale_count),
        ):
            fval = float(val)
            if fval <= 0.0:
                raise ValueError(
                    f"{name} must be positive, got {fval!r}. "
                    "Pass a measured typical magnitude, e.g. scale_bp=1.0."
                )
            setattr(self, name, fval)

        # Learnable S, parameterized in log space so it stays positive and
        # the gradient is well-conditioned across the [4.1, 5.5] range.
        # interval_nll() applies the operational clamp internally.
        if not (S_MIN <= S_init <= S_MAX):
            raise ValueError(
                f"S_init={S_init!r} outside operational range [{S_MIN}, {S_MAX}]."
            )
        self.log_S = nn.Parameter(torch.tensor(math.log(S_init), dtype=torch.float32))

        # Schedule state.
        self.current_lambda_bp: float = 0.0
        self.current_lambda_vel: float = 0.0
        self.current_lambda_count: float = 0.0
        self.current_lambda_stretch_prior: float = 0.0
        self._warmstart_blend: float = 1.0

    # ------------------------------------------------------------------
    # Schedulers (interface-matched with CombinedLoss)
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Update warmstart blend and current lambda values for ``epoch``."""
        full_epochs = max(self.warmstart_epochs - self.warmstart_fade_epochs, 0)
        if self.warmstart_epochs <= 0:
            blend = 0.0
        elif epoch < full_epochs:
            blend = 1.0
        elif epoch < self.warmstart_epochs:
            frac = (epoch - full_epochs + 1) / max(self.warmstart_fade_epochs, 1)
            blend = max(0.0, 1.0 - frac)
        else:
            blend = 0.0
        if blend < self.min_blend:
            blend = self.min_blend
        self._warmstart_blend = float(blend)

        if self.warmstart_epochs <= 0:
            scale = 1.0
        elif epoch < self.warmstart_epochs:
            scale = 0.5 + 0.5 * (epoch / self.warmstart_epochs)
        else:
            scale = 1.0

        self.current_lambda_bp = self.lambda_bp * scale
        self.current_lambda_vel = self.lambda_vel * scale
        self.current_lambda_count = self.lambda_count * scale
        self.current_lambda_stretch_prior = self.lambda_stretch_prior * scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pred_heatmap: torch.Tensor,
        pred_cumulative_bp: torch.Tensor,
        raw_velocity: torch.Tensor,
        reference_bp_positions_list: list[torch.Tensor],
        n_ref_probes: torch.Tensor,
        warmstart_heatmap: torch.Tensor | None,
        warmstart_valid: torch.Tensor | None,
        mask: torch.Tensor,
        pred_heatmap_logits: torch.Tensor | None = None,
        warmstart_probe_centers_samples_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the V4 composite loss for a batch.

        Signature matches :class:`CombinedLoss`. See module docstring for
        the per-component math and the ``details`` key mapping.
        """
        device = pred_heatmap.device
        batch_size = pred_heatmap.shape[0]
        blend = float(self._warmstart_blend)

        probe_terms: list[torch.Tensor] = []
        bp_terms: list[torch.Tensor] = []
        vel_terms: list[torch.Tensor] = []
        count_terms: list[torch.Tensor] = []
        stretch_prior_terms: list[torch.Tensor] = []
        stretch_v_values: list[torch.Tensor] = []

        for b in range(batch_size):
            pred_h_b = pred_heatmap[b]
            pred_bp_b = pred_cumulative_bp[b]
            raw_v_b = raw_velocity[b]
            mask_b = mask[b]

            # Grad-attached zero -- used as the initial value for components
            # that may legitimately end up zero (degenerate batches). Without
            # this, torch.zeros(()) is a no-grad leaf, and a batch where all
            # terms degenerate produces a total loss with no grad_fn -- so
            # loss.backward() raises. Real training never hits this; synthetic
            # batches with warmstart_probability < 1 sometimes do.
            grad_zero = pred_h_b.sum() * 0.0

            # ----------------- L_probe (unchanged from CombinedLoss) -----------------
            probe_component = grad_zero
            if (
                blend > 0.0
                and warmstart_heatmap is not None
                and warmstart_valid is not None
                and bool(warmstart_valid[b].item())
            ):
                if pred_heatmap_logits is None:
                    raise ValueError(
                        "CenterNet focal probe loss requires pred_heatmap_logits; "
                        "the model forward must return logits."
                    )
                probe_component = probe_component + blend * centernet_focal_loss(
                    pred_heatmap_logits[b], warmstart_heatmap[b], mask_b
                )
            if blend < 1.0:
                masked_heatmap = pred_h_b * mask_b.to(pred_h_b.dtype)
                peaky = peakiness_regularizer(
                    masked_heatmap, window=self.peakiness_window
                )
                probe_component = probe_component + (1.0 - blend) * peaky
            probe_terms.append(probe_component)

            # Decide between teacher forcing and NMS-detected peaks.
            gt_centers: torch.Tensor | None = None
            if (
                warmstart_probe_centers_samples_list is not None
                and warmstart_probe_centers_samples_list[b] is not None
            ):
                raw_c = warmstart_probe_centers_samples_list[b]
                if raw_c.numel() >= 2:
                    gt_centers = raw_c.to(device=device, dtype=torch.long)

            ref_bp = reference_bp_positions_list[b]
            n_ref_b = int(n_ref_probes[b].item())

            if gt_centers is None and ref_bp.numel() >= 2:
                # Fallback: NMS-detected peaks paired 1:1 with ref_bp.
                peak_indices = extract_peak_indices(
                    pred_h_b,
                    raw_v_b,
                    threshold=self.nms_threshold,
                    tag_width_bp=self.tag_width_bp,
                )
                if peak_indices.numel() >= 2:
                    k = int(min(peak_indices.numel(), ref_bp.numel()))
                    gt_centers = peak_indices[:k].to(device=device, dtype=torch.long)
                    ref_bp = ref_bp[:k]

            if gt_centers is not None and ref_bp.numel() >= 2:
                gt_centers = gt_centers.clamp(0, pred_h_b.shape[0] - 1)
                pred_bp_at_peaks = pred_bp_b[gt_centers]
                ref_bp_f = ref_bp.to(device=device, dtype=pred_bp_at_peaks.dtype)

                # L_bp -> per-interval Gaussian NLL, returns ML stretch v.
                interval_nll_b, v_b = interval_nll(pred_bp_at_peaks, ref_bp_f, self.log_S)
                bp_terms.append(interval_nll_b)
                stretch_v_values.append(v_b.detach())

                # L_vel -> per-probe absolute position prior.
                vel_terms.append(
                    position_nll(pred_bp_at_peaks, ref_bp_f, sigma_bp=self.position_sigma_bp)
                )

                # Stretch-latent prior on v_ML.
                stretch_prior_terms.append(stretch_prior_nll(v_b))

                # L_count -> proximity-aware expected count vs sum(heatmap).
                ref_norm = (ref_bp_f - ref_bp_f[0]).abs()
                ref_intervals = (ref_norm[1:] - ref_norm[:-1]).clamp(min=1.0)
                # Molecule length proxy = absolute span between first and last
                # reference probe; clean molecules (the only ones in our
                # cache) are tail-anchored so this is a tight estimate.
                mol_length_bp = (ref_bp_f[-1] - ref_bp_f[0]).abs().detach()
                masked_heatmap = pred_h_b * mask_b.to(pred_h_b.dtype)
                predicted_count = masked_heatmap.sum()
                count_terms.append(
                    proximity_aware_count_loss(
                        predicted_count,
                        ref_intervals.detach(),
                        mol_length_bp,
                        n_ref_b,
                    )
                )
            else:
                # Degenerate: no usable peaks. Use grad-attached zero so a
                # whole batch of degenerate molecules still has a grad chain.
                bp_terms.append(grad_zero)
                vel_terms.append(grad_zero)
                count_terms.append(grad_zero)
                stretch_prior_terms.append(grad_zero)
                # No stretch_v emitted for this molecule.

        zero = torch.zeros((), device=device, dtype=pred_heatmap.dtype)

        probe_loss = torch.stack(probe_terms).mean() if probe_terms else zero
        bp_loss = torch.stack(bp_terms).mean() if bp_terms else zero
        vel_loss = torch.stack(vel_terms).mean() if vel_terms else zero
        count_loss_value = torch.stack(count_terms).mean() if count_terms else zero
        stretch_prior_loss = (
            torch.stack(stretch_prior_terms).mean() if stretch_prior_terms else zero
        )
        stretch_v_mean = (
            torch.stack(stretch_v_values).mean()
            if stretch_v_values
            else torch.ones((), device=device, dtype=pred_heatmap.dtype)
        )

        # Component scaling -- same pattern as CombinedLoss so lambdas
        # behave like contribute-equal-gradient knobs.
        scaled_probe = probe_loss / self.scale_probe
        scaled_bp = bp_loss / self.scale_bp
        scaled_vel = vel_loss / self.scale_vel
        scaled_count = count_loss_value / self.scale_count

        total = (
            scaled_probe
            + self.current_lambda_bp * scaled_bp
            + self.current_lambda_vel * scaled_vel
            + self.current_lambda_count * scaled_count
            + self.current_lambda_stretch_prior * stretch_prior_loss
        )

        S_value = float(torch.exp(self.log_S).clamp(min=S_MIN, max=S_MAX).detach().item())

        details: dict[str, Any] = {
            "probe": float(scaled_probe.detach().item()),
            "bp": float(scaled_bp.detach().item()),
            "vel": float(scaled_vel.detach().item()),
            "count": float(scaled_count.detach().item()),
            "probe_raw": float(probe_loss.detach().item()),
            "bp_raw": float(bp_loss.detach().item()),
            "vel_raw": float(vel_loss.detach().item()),
            "count_raw": float(count_loss_value.detach().item()),
            "stretch_prior": float(stretch_prior_loss.detach().item()),
            "stretch_v_ML": float(stretch_v_mean.detach().item()),
            "S_value": S_value,
            "warmstart_blend": blend,
        }
        return total, details
