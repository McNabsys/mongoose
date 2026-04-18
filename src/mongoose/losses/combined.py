"""Combined loss for V1 rearchitecture.

Composes four components per molecule:

- ``L_probe``: masked MSE against wfmproc Gaussian target during warmstart,
  fading into the peakiness regularizer after warmstart ends. MSE replaced
  focal loss after the latter produced degenerate flat-heatmap minima —
  focal's alpha=0.25 weight on positives and averaging over padded length
  diluted the sparse peak signal.
- ``L_bp``: soft-DTW between model-detected peaks in cumulative bp and the
  reference bp positions, zero-anchored and span-normalized.
- ``L_velocity``: MSE at detected peak positions with targets derived from
  heatmap FWHM (stop-grad on target).
- ``L_count``: smooth L1 on ``sum(heatmap)`` vs ``n_ref_probes``.

``set_epoch`` drives both the warmstart blend (L_probe character transition)
and the lambda scale (0.5 during warmstart, 1.0 after).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mongoose.losses.count import count_loss, peakiness_regularizer
from mongoose.losses.peaks import extract_peak_indices, measure_peak_widths_samples
from mongoose.losses.softdtw import soft_dtw


class CombinedLoss:
    """Composite training loss.

    L_total = L_probe
            + current_lambda_bp    * L_bp
            + current_lambda_vel   * L_velocity
            + current_lambda_count * L_count
    """

    def __init__(
        self,
        lambda_bp: float = 1.0,
        lambda_vel: float = 1.0,
        lambda_count: float = 1.0,
        warmup_epochs: int = 5,
        warmstart_epochs: int = 5,
        warmstart_fade_epochs: int = 2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        huber_delta_bp: float = 500.0,
        softdtw_gamma: float = 0.1,
        peakiness_window: int = 20,
        nms_threshold: float = 0.3,
        tag_width_bp: float = 511.0,
        sample_period_ms: float = 0.025,
        min_blend: float = 0.0,
        scale_probe: float = 1.0,
        scale_bp: float = 1.0,
        scale_vel: float = 1.0,
        scale_count: float = 1.0,
        probe_pos_weight: float = 0.0,
    ) -> None:
        self.lambda_bp = lambda_bp
        self.lambda_vel = lambda_vel
        self.lambda_count = lambda_count
        self.warmup_epochs = warmup_epochs
        self.warmstart_epochs = warmstart_epochs
        self.warmstart_fade_epochs = warmstart_fade_epochs
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.huber_delta_bp = huber_delta_bp
        self.softdtw_gamma = softdtw_gamma
        self.peakiness_window = peakiness_window
        self.nms_threshold = nms_threshold
        self.tag_width_bp = tag_width_bp
        self.sample_period_ms = sample_period_ms
        self.min_blend = float(min_blend)
        self.probe_pos_weight = float(probe_pos_weight)
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
                    "Pass a measured typical magnitude, e.g. scale_bp=30000.0."
                )
            setattr(self, name, fval)

        # Current effective lambdas and warmstart blend (updated by set_epoch).
        self.current_lambda_bp: float = 0.0
        self.current_lambda_vel: float = 0.0
        self.current_lambda_count: float = 0.0
        self._warmstart_blend: float = 1.0

    # ------------------------------------------------------------------
    # Schedulers
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Update ``_warmstart_blend`` and current lambda values for this epoch."""
        # Warmstart blend: 1.0 during full-warmstart, linearly fading to 0
        # over ``warmstart_fade_epochs``, then 0 afterward.
        full_epochs = max(self.warmstart_epochs - self.warmstart_fade_epochs, 0)
        if self.warmstart_epochs <= 0:
            self._warmstart_blend = 0.0
        elif epoch < full_epochs:
            self._warmstart_blend = 1.0
        elif epoch < self.warmstart_epochs:
            frac = (epoch - full_epochs + 1) / max(self.warmstart_fade_epochs, 1)
            self._warmstart_blend = max(0.0, 1.0 - frac)
        else:
            self._warmstart_blend = 0.0

        # Apply floor so focal supervision never fully disappears.
        if self._warmstart_blend < self.min_blend:
            self._warmstart_blend = self.min_blend

        # Lambda schedule: 0.5x target during warmstart, ramping to 1.0x after.
        if self.warmstart_epochs <= 0:
            scale = 1.0
        elif epoch < self.warmstart_epochs:
            progress = epoch / self.warmstart_epochs
            scale = 0.5 + 0.5 * progress
        else:
            scale = 1.0

        self.current_lambda_bp = self.lambda_bp * scale
        self.current_lambda_vel = self.lambda_vel * scale
        self.current_lambda_count = self.lambda_count * scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(
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
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute combined loss for a batch.

        Args:
            pred_heatmap: [B, T] predicted heatmap (after sigmoid).
            pred_cumulative_bp: [B, T] cumulative bp prediction.
            raw_velocity: [B, T] raw velocity (bp/sample).
            reference_bp_positions_list: list of [N_i] int64 tensors holding
                the ordered reference bp coordinates for each molecule.
            n_ref_probes: [B] int tensor with the number of reference probes.
            warmstart_heatmap: [B, T] float tensor (pre-built Gaussians), or
                ``None`` to skip the probe supervision term.
            pred_heatmap_logits: Optional [B, T] raw logits. When provided,
                the probe term uses ``BCEWithLogitsLoss`` for numerical
                stability under saturated sigmoid outputs. Without it, the
                probe term falls back to positive-weighted MSE on the
                sigmoid-activated heatmap (less stable once outputs saturate).
            warmstart_valid: [B] bool tensor flagging which molecules have
                valid warmstart labels, or ``None``.
            mask: [B, T] boolean mask for valid (non-padded) samples.

        Returns:
            ``(total_loss, details)`` where ``details`` contains the scalar
            per-component losses plus the current warmstart blend factor.
        """
        device = pred_heatmap.device
        batch_size = pred_heatmap.shape[0]
        blend = float(self._warmstart_blend)

        probe_terms: list[torch.Tensor] = []
        bp_terms: list[torch.Tensor] = []
        vel_terms: list[torch.Tensor] = []
        count_terms: list[torch.Tensor] = []

        for b in range(batch_size):
            pred_h_b = pred_heatmap[b]
            pred_bp_b = pred_cumulative_bp[b]
            raw_v_b = raw_velocity[b]
            mask_b = mask[b]

            # ----------------- L_probe -----------------
            probe_component = torch.zeros((), device=device, dtype=pred_h_b.dtype)

            if (
                blend > 0.0
                and warmstart_heatmap is not None
                and warmstart_valid is not None
                and bool(warmstart_valid[b].item())
            ):
                mask_f_b = mask_b.to(pred_h_b.dtype)
                target_b = warmstart_heatmap[b].to(pred_h_b.dtype)
                weight_b = 1.0 + self.probe_pos_weight * target_b

                if pred_heatmap_logits is not None:
                    # BCE-with-logits against the soft Gaussian target.
                    # Stable under saturated sigmoid outputs (a failure
                    # mode that killed pure-MSE training: once logits
                    # dropped below ~-6, sigmoid gradient vanished).
                    logits_b = pred_heatmap_logits[b]
                    bce_elem = F.binary_cross_entropy_with_logits(
                        logits_b, target_b, reduction="none"
                    )
                    weighted_err = bce_elem * weight_b * mask_f_b
                else:
                    # Fallback: positive-weighted MSE on post-sigmoid
                    # probabilities. Retained for backward compat.
                    sq_err = (pred_h_b - target_b) ** 2
                    weighted_err = sq_err * weight_b * mask_f_b

                denom = (weight_b * mask_f_b).sum().clamp(min=1.0)
                probe_loss_val = weighted_err.sum() / denom
                probe_component = probe_component + blend * probe_loss_val

            if blend < 1.0:
                masked_heatmap = pred_h_b * mask_b.to(pred_h_b.dtype)
                peaky = peakiness_regularizer(masked_heatmap, window=self.peakiness_window)
                probe_component = probe_component + (1.0 - blend) * peaky

            probe_terms.append(probe_component)

            # ----------------- Detect peaks (no grad) -----------------
            peak_indices = extract_peak_indices(
                pred_h_b,
                raw_v_b,
                threshold=self.nms_threshold,
                tag_width_bp=self.tag_width_bp,
            )

            ref_bp = reference_bp_positions_list[b]

            # ----------------- L_bp (soft-DTW) -----------------
            if peak_indices.numel() >= 2 and ref_bp.numel() >= 2:
                pred_bp_at_peaks = pred_bp_b[peak_indices]
                ref_bp_f = ref_bp.to(device=device, dtype=pred_bp_at_peaks.dtype)
                pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
                ref_norm = (ref_bp_f - ref_bp_f[0]).abs()
                span = (ref_bp_f[-1] - ref_bp_f[0]).abs()
                span = torch.clamp(span, min=1.0)
                dtw = soft_dtw(pred_norm, ref_norm, gamma=self.softdtw_gamma)
                bp_terms.append(dtw / span)

                # -------------- L_velocity (dense at peaks) --------------
                widths_samples = measure_peak_widths_samples(
                    pred_h_b, peak_indices, threshold_frac=0.5
                )
                widths_ms = widths_samples.to(device=device, dtype=raw_v_b.dtype) * float(
                    self.sample_period_ms
                )
                widths_ms = torch.clamp(widths_ms, min=1e-6)
                target_v = (
                    float(self.tag_width_bp) / widths_ms * float(self.sample_period_ms)
                )
                pred_v_at_peaks = raw_v_b[peak_indices]
                vel_terms.append(F.mse_loss(pred_v_at_peaks, target_v.detach()))

            # ----------------- L_count -----------------
            count_terms.append(
                count_loss(
                    pred_h_b,
                    float(n_ref_probes[b].item()),
                    mask=mask_b,
                )
            )

        zero = torch.zeros((), device=device, dtype=pred_heatmap.dtype)

        probe_loss = torch.stack(probe_terms).mean() if probe_terms else zero
        bp_loss = torch.stack(bp_terms).mean() if bp_terms else zero
        vel_loss = torch.stack(vel_terms).mean() if vel_terms else zero
        count_loss_value = torch.stack(count_terms).mean() if count_terms else zero

        # Scale each component by its hardcoded divisor so lambda values
        # mean "contribute roughly equal gradient" instead of being 4-5
        # orders of magnitude apart (see spec sec. 3.2).
        scaled_probe = probe_loss / self.scale_probe
        scaled_bp = bp_loss / self.scale_bp
        scaled_vel = vel_loss / self.scale_vel
        scaled_count = count_loss_value / self.scale_count

        total = (
            scaled_probe
            + self.current_lambda_bp * scaled_bp
            + self.current_lambda_vel * scaled_vel
            + self.current_lambda_count * scaled_count
        )

        details: dict[str, Any] = {
            "probe": scaled_probe.detach().item(),
            "bp": scaled_bp.detach().item(),
            "vel": scaled_vel.detach().item(),
            "count": scaled_count.detach().item(),
            "probe_raw": probe_loss.detach().item(),
            "bp_raw": bp_loss.detach().item(),
            "vel_raw": vel_loss.detach().item(),
            "count_raw": count_loss_value.detach().item(),
            "warmstart_blend": blend,
        }
        return total, details
