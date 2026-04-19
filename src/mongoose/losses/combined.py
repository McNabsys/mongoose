"""Combined loss for V1 rearchitecture.

Composes four components per molecule:

- ``L_probe``: CenterNet-style focal loss against wfmproc Gaussian targets
  during warmstart, blended with the peakiness regularizer after warmstart
  ends. Positives are samples with target >= 0.99 (peak centers); negatives
  in the Gaussian halo are down-weighted by ``(1 - target)**beta``. Normalized
  per-molecule by ``num_positives`` to keep gradient strength independent of
  sequence length.
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

from mongoose.losses.centernet_focal import centernet_focal_loss
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
    ) -> None:
        self.lambda_bp = lambda_bp
        self.lambda_vel = lambda_vel
        self.lambda_count = lambda_count
        self.warmup_epochs = warmup_epochs
        self.warmstart_epochs = warmstart_epochs
        self.warmstart_fade_epochs = warmstart_fade_epochs
        self.huber_delta_bp = huber_delta_bp
        self.softdtw_gamma = softdtw_gamma
        self.peakiness_window = peakiness_window
        self.nms_threshold = nms_threshold
        self.tag_width_bp = tag_width_bp
        self.sample_period_ms = sample_period_ms
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
        warmstart_probe_centers_samples_list: list[torch.Tensor] | None = None,
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
            pred_heatmap_logits: [B, T] raw logits. Required whenever
                ``warmstart_heatmap`` / ``warmstart_valid`` mark a molecule
                for probe supervision; the CenterNet focal loss operates on
                logits (not post-sigmoid probabilities) for numerical
                stability at saturated outputs. Passing ``None`` while
                warmstart is active raises ``ValueError``.
            warmstart_probe_centers_samples_list: Optional list of per-molecule
                ground-truth probe center sample indices (``LongTensor[K_i]``).
                When provided, L_bp and L_vel are evaluated at these indices
                (teacher forcing) instead of at NMS-detected peaks. This
                bypasses the non-differentiable NMS gate and lets the velocity
                head receive flawless supervision from step 1 regardless of
                the probe head's state. When ``None`` (or when a molecule's
                entry is ``None``), falls back to the legacy NMS-detected
                branch.
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
                if pred_heatmap_logits is None:
                    raise ValueError(
                        "CenterNet focal probe loss requires pred_heatmap_logits; "
                        "the model forward must return logits."
                    )
                logits_b = pred_heatmap_logits[b]
                target_b = warmstart_heatmap[b]
                probe_loss_val = centernet_focal_loss(logits_b, target_b, mask_b)
                probe_component = probe_component + blend * probe_loss_val

            if blend < 1.0:
                masked_heatmap = pred_h_b * mask_b.to(pred_h_b.dtype)
                peaky = peakiness_regularizer(masked_heatmap, window=self.peakiness_window)
                probe_component = probe_component + (1.0 - blend) * peaky

            probe_terms.append(probe_component)

            # Decide between teacher forcing and NMS-detected peak extraction.
            gt_centers: torch.Tensor | None = None
            if (
                warmstart_probe_centers_samples_list is not None
                and warmstart_probe_centers_samples_list[b] is not None
            ):
                gt_centers_raw = warmstart_probe_centers_samples_list[b]
                if gt_centers_raw.numel() >= 2:
                    gt_centers = gt_centers_raw.to(device=device, dtype=torch.long)

            if gt_centers is not None:
                # ---- Teacher-forced path ----
                # Clamp to valid range before gather (defensive; preprocess
                # should ensure all centers fall inside the waveform).
                gt_centers = gt_centers.clamp(0, pred_h_b.shape[0] - 1)

                # L_bp: soft-DTW between model-predicted cumulative-bp AT THE
                # GROUND-TRUTH SAMPLE INDICES and the reference bp positions.
                # Indices are the same length as ref_bp, so DTW is 1:1 and
                # could reduce to Huber; keeping soft-DTW for symmetry with
                # the inference path.
                pred_bp_at_peaks = pred_bp_b[gt_centers]
                ref_bp = reference_bp_positions_list[b]
                if ref_bp.numel() >= 2:
                    ref_bp_f = ref_bp.to(device=device, dtype=pred_bp_at_peaks.dtype)
                    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
                    ref_norm = (ref_bp_f - ref_bp_f[0]).abs()
                    span = (ref_bp_f[-1] - ref_bp_f[0]).abs().clamp(min=1.0)
                    dtw = soft_dtw(pred_norm, ref_norm, gamma=self.softdtw_gamma)
                    bp_terms.append(dtw / span)

                # L_vel: MSE between raw_velocity at the ground-truth indices
                # and the per-peak target derived from the GROUND-TRUTH
                # heatmap's FWHM at those same indices. Since target heatmaps
                # are Gaussians centered at the ground truth, we can measure
                # widths directly from the warmstart heatmap.
                if warmstart_heatmap is not None:
                    ws_b = warmstart_heatmap[b]
                    widths_samples = measure_peak_widths_samples(
                        ws_b, gt_centers, threshold_frac=0.5
                    )
                else:
                    # Falls back to predicted widths — less accurate but keeps
                    # the path defined for unit tests that omit warmstart.
                    widths_samples = measure_peak_widths_samples(
                        pred_h_b, gt_centers, threshold_frac=0.5
                    )
                widths_ms = widths_samples.to(
                    device=device, dtype=raw_v_b.dtype
                ) * float(self.sample_period_ms)
                widths_ms = torch.clamp(widths_ms, min=1e-6)
                target_v = (
                    float(self.tag_width_bp) / widths_ms * float(self.sample_period_ms)
                )
                pred_v_at_peaks = raw_v_b[gt_centers]
                vel_terms.append(F.mse_loss(pred_v_at_peaks, target_v.detach()))
            else:
                # ---- Legacy NMS-detected peaks path (unchanged) ----
                peak_indices = extract_peak_indices(
                    pred_h_b,
                    raw_v_b,
                    threshold=self.nms_threshold,
                    tag_width_bp=self.tag_width_bp,
                )

                ref_bp = reference_bp_positions_list[b]

                if peak_indices.numel() >= 2 and ref_bp.numel() >= 2:
                    pred_bp_at_peaks = pred_bp_b[peak_indices]
                    ref_bp_f = ref_bp.to(device=device, dtype=pred_bp_at_peaks.dtype)
                    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
                    ref_norm = (ref_bp_f - ref_bp_f[0]).abs()
                    span = (ref_bp_f[-1] - ref_bp_f[0]).abs()
                    span = torch.clamp(span, min=1.0)
                    dtw = soft_dtw(pred_norm, ref_norm, gamma=self.softdtw_gamma)
                    bp_terms.append(dtw / span)

                    widths_samples = measure_peak_widths_samples(
                        pred_h_b, peak_indices, threshold_frac=0.5
                    )
                    widths_ms = widths_samples.to(
                        device=device, dtype=raw_v_b.dtype
                    ) * float(self.sample_period_ms)
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
