"""L_511 physics-informed loss for V3 feasibility spike.

The tag width is a controlled physical constant (511 bp). This gives a
per-probe integral identity on the translocation velocity field:

    int_{t_enter}^{t_exit} v(t) dt = 511 bp   for every probe

Summed across thousands of probes per batch this is dense self-supervision
on the velocity head that does not require reference-genome alignment.

This module also provides an optional global span constraint
(``l_length_span``) and a smoothness regularizer (``l_smooth_velocity``).

Deep Think round-6 fix: the integration boundaries are **detached** before
being used as slice indices. This prevents a gradient path where the
network could "weaponize" the probe boundaries as a slack variable
(expand the mask into high-velocity baseline regions to trivially inflate
the integral). For this spike the boundaries come from wfmproc warmstart
data anyway (no gradient path exists), but the detach is belt-and-braces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TAG_WIDTH_BP = 511.0


def l511_per_probe(
    raw_velocity: torch.Tensor,
    centers_samples: torch.Tensor,
    widths_samples: torch.Tensor,
    tag_width_bp: float = TAG_WIDTH_BP,
) -> torch.Tensor:
    """Squared residual between integrated velocity and 511 bp per probe.

    ``raw_velocity`` has units of bp/sample, so ``sum(raw_velocity[i:j])``
    equals the integrated bp over that sample range. For each probe at
    ``center`` with ``width`` samples, we compute::

        integrated_bp = sum(raw_velocity[center - width/2 : center + width/2])
        residual_sq   = (integrated_bp - 511) ** 2

    and return the mean residual across probes.

    Args:
        raw_velocity: [T] per-sample predicted velocity (bp/sample).
        centers_samples: [K] LongTensor of probe center sample indices.
        widths_samples: [K] FloatTensor of probe widths in samples.
        tag_width_bp: Physical constant (default 511).

    Returns:
        Scalar mean squared residual over probes. Zero if K == 0.
    """
    if centers_samples.numel() == 0:
        return torch.zeros((), device=raw_velocity.device, dtype=raw_velocity.dtype)

    # Detach boundary definitions — gradient must not flow here
    # (round-6 slack-variable-trap fix).
    centers = centers_samples.detach().long()
    widths = widths_samples.detach().float().clamp(min=1.0)

    T = raw_velocity.shape[0]
    half = (widths * 0.5).long().clamp(min=1)
    lo = (centers - half).clamp(min=0, max=T - 1)
    hi = (centers + half).clamp(min=0, max=T - 1)

    residuals: list[torch.Tensor] = []
    for i in range(centers.shape[0]):
        lo_i = int(lo[i].item())
        hi_i = int(hi[i].item())
        if hi_i <= lo_i:
            continue
        integrated = raw_velocity[lo_i : hi_i + 1].sum()
        residuals.append((integrated - tag_width_bp).pow(2))

    if not residuals:
        return torch.zeros((), device=raw_velocity.device, dtype=raw_velocity.dtype)
    return torch.stack(residuals).mean()


def l_length_span(
    raw_velocity: torch.Tensor,
    centers_samples: torch.Tensor,
    reference_bp_positions: torch.Tensor,
) -> torch.Tensor:
    """Global span constraint: first-to-last integrated bp vs reference span.

    Anchors the global integration budget so v_macro doesn't drift. Uses
    integrated velocity between the first and last probe centers (detached
    indices), compared to the absolute reference bp span between the
    outer two matched probes.

    Args:
        raw_velocity: [T] per-sample predicted velocity.
        centers_samples: [K] LongTensor of probe center sample indices.
            Must have ``K >= 2`` or loss is zero.
        reference_bp_positions: [K] LongTensor of reference bp coords,
            paired 1:1 with centers in temporal order.

    Returns:
        Scalar squared residual. Zero if K < 2.
    """
    if centers_samples.numel() < 2 or reference_bp_positions.numel() < 2:
        return torch.zeros((), device=raw_velocity.device, dtype=raw_velocity.dtype)

    centers = centers_samples.detach().long()
    T = raw_velocity.shape[0]
    lo = int(centers[0].clamp(min=0, max=T - 1).item())
    hi = int(centers[-1].clamp(min=0, max=T - 1).item())
    if hi <= lo:
        return torch.zeros((), device=raw_velocity.device, dtype=raw_velocity.dtype)

    pred_span = raw_velocity[lo : hi + 1].sum()
    ref_span = (
        (reference_bp_positions[-1] - reference_bp_positions[0])
        .abs()
        .to(pred_span.dtype)
    )
    return (pred_span - ref_span).pow(2)


def l_smooth_velocity(
    raw_velocity: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """TV smoothness regularizer on the raw velocity field.

    Mean of squared first differences over the valid-mask region. Prevents
    high-frequency jitter where L_511 is silent (between probes).

    Args:
        raw_velocity: [T] per-sample velocity.
        mask: [T] bool tensor — True where sample is valid (non-padded).

    Returns:
        Scalar mean squared difference. Zero if fewer than 2 valid samples.
    """
    if raw_velocity.numel() < 2:
        return torch.zeros((), device=raw_velocity.device, dtype=raw_velocity.dtype)

    diffs = raw_velocity[1:] - raw_velocity[:-1]
    valid = mask[1:].to(raw_velocity.dtype)
    denom = valid.sum().clamp(min=1.0)
    return (diffs.pow(2) * valid).sum() / denom


class L511Loss:
    """V3-spike composite loss: L_probe + L_511 + L_smooth + L_length.

    Sibling of ``CombinedLoss`` with the same ``__call__`` signature so the
    trainer can drop it in without structural changes. The ``details`` dict
    uses the same keys (``probe``, ``bp``, ``vel``, ``count``) so the
    existing TB logging and CLI console output keep working; the semantic
    content of each slot shifts:

        probe  -> focal CenterNet loss (unchanged, for probe-head warmstart)
        bp     -> L_511 (per-probe integral constraint)  [the core signal]
        vel    -> L_smooth (TV regularizer on velocity)
        count  -> L_length_span (global integration anchor)
    """

    def __init__(
        self,
        *,
        lambda_511: float = 1.0,
        lambda_smooth: float = 0.001,
        lambda_length: float = 0.5,
        lambda_align: float = 0.0,
        align_min_confidence: float = 0.7,
        warmstart_epochs: int = 5,
        warmstart_fade_epochs: int = 2,
        min_blend: float = 0.05,
        scale_511: float = 511.0**2,  # residual is (~bp)^2; normalize by 511^2
        scale_smooth: float = 1.0,
        scale_length: float = 1.0e9,  # span residual can be (~30000 bp)^2 on long molecules
        scale_align: float = 1.0e6,  # L1 in bp, typical interval ~1000 bp
        tag_width_bp: float = TAG_WIDTH_BP,
    ) -> None:
        self.lambda_511 = lambda_511
        self.lambda_smooth = lambda_smooth
        self.lambda_length = lambda_length
        self.lambda_align = float(lambda_align)
        self.align_min_confidence = float(align_min_confidence)
        self.warmstart_epochs = int(warmstart_epochs)
        self.warmstart_fade_epochs = int(warmstart_fade_epochs)
        self.min_blend = float(min_blend)
        self.scale_511 = float(scale_511)
        self.scale_smooth = float(scale_smooth)
        self.scale_length = float(scale_length)
        self.scale_align = float(scale_align)
        self.tag_width_bp = float(tag_width_bp)
        self._warmstart_blend: float = 1.0

    # ------------------------------------------------------------------
    # Schedulers (shape-compatible with CombinedLoss)
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Linear decay of probe-head warmstart blend with a floor.

        Round-6 fix: never fully drop probe supervision — the floor
        ``min_blend`` keeps an elastic tether so the probe head cannot
        drift arbitrarily under L_511 pressure alone.
        """
        full = max(self.warmstart_epochs - self.warmstart_fade_epochs, 0)
        if self.warmstart_epochs <= 0:
            blend = 0.0
        elif epoch < full:
            blend = 1.0
        elif epoch < self.warmstart_epochs:
            frac = (epoch - full + 1) / max(self.warmstart_fade_epochs, 1)
            blend = max(0.0, 1.0 - frac)
        else:
            blend = 0.0
        self._warmstart_blend = max(blend, self.min_blend)

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute L511Loss for a batch.

        See module docstring for the math. Signature matches ``CombinedLoss``
        so the trainer can swap classes without change.
        """
        from mongoose.losses.centernet_focal import centernet_focal_loss
        from mongoose.losses.peaks import measure_peak_widths_samples
        import torch.nn.functional as F

        device = pred_heatmap.device
        batch_size = pred_heatmap.shape[0]
        blend = float(self._warmstart_blend)

        probe_terms: list[torch.Tensor] = []
        l511_terms: list[torch.Tensor] = []
        smooth_terms: list[torch.Tensor] = []
        length_terms: list[torch.Tensor] = []
        # Mixed-supervision alignment loss (only populated when lambda_align>0
        # AND molecule's match ratio exceeds align_min_confidence).
        align_terms: list[torch.Tensor] = []
        n_align_active = 0  # number of molecules contributing to alignment

        for b in range(batch_size):
            pred_h_b = pred_heatmap[b]
            raw_v_b = raw_velocity[b]
            mask_b = mask[b]

            # ---------- L_probe (same as CombinedLoss, but no peakiness fallback) ----------
            if (
                blend > 0.0
                and warmstart_heatmap is not None
                and warmstart_valid is not None
                and bool(warmstart_valid[b].item())
                and pred_heatmap_logits is not None
            ):
                probe_terms.append(
                    blend
                    * centernet_focal_loss(
                        pred_heatmap_logits[b], warmstart_heatmap[b], mask_b
                    )
                )

            # Pull per-molecule probe centers + widths (all boundary sources
            # are already gradient-free; extra .detach() in l511_per_probe).
            gt_centers: torch.Tensor | None = None
            if (
                warmstart_probe_centers_samples_list is not None
                and warmstart_probe_centers_samples_list[b] is not None
            ):
                raw_c = warmstart_probe_centers_samples_list[b]
                if raw_c.numel() >= 2:
                    gt_centers = raw_c.to(device=device, dtype=torch.long)

            if gt_centers is not None and warmstart_heatmap is not None:
                # Widths derived from the ground-truth heatmap (pre-built
                # Gaussian, no gradient path), not from model predictions.
                widths = measure_peak_widths_samples(
                    warmstart_heatmap[b], gt_centers, threshold_frac=0.5
                ).to(device=device, dtype=raw_v_b.dtype)
                l511_terms.append(
                    l511_per_probe(raw_v_b, gt_centers, widths, self.tag_width_bp)
                )

                # Global span anchor.
                ref_bp = reference_bp_positions_list[b]
                length_terms.append(l_length_span(raw_v_b, gt_centers, ref_bp))

                # ---------- L_align (mixed supervision, gated by confidence) ----------
                # For high-confidence remapped molecules, add a direct L1 loss
                # on cum_bp at probe centers vs reference_bp_positions (anchored
                # at the first probe so the absolute genome offset doesn't
                # matter; only relative bp positions across probes do).
                if self.lambda_align > 0 and ref_bp.numel() == gt_centers.numel():
                    n_ref_b = float(n_ref_probes[b].item())
                    n_matched = float(gt_centers.numel())
                    confidence = n_matched / max(n_ref_b, 1.0)
                    if confidence >= self.align_min_confidence:
                        # cum_bp is the cumulative integral of raw_velocity.
                        # Recompute here to keep gradient path through raw_v_b.
                        cum_bp_b = torch.cumsum(raw_v_b * mask_b.to(raw_v_b.dtype), dim=-1)
                        # Clamp center indices defensively.
                        T = cum_bp_b.shape[-1]
                        c = gt_centers.clamp(min=0, max=T - 1)
                        pred_at = cum_bp_b[c]  # [K]
                        # Anchor predictions and reference at the first probe.
                        pred_norm = pred_at - pred_at[0]
                        ref_norm = (ref_bp - ref_bp[0]).abs().to(pred_norm.dtype)
                        align_terms.append(F.l1_loss(pred_norm, ref_norm))
                        n_align_active += 1

            # ---------- L_smooth (always, needs only velocity + mask) ----------
            smooth_terms.append(l_smooth_velocity(raw_v_b, mask_b))

        zero = torch.zeros((), device=device, dtype=pred_heatmap.dtype)
        probe_loss = torch.stack(probe_terms).mean() if probe_terms else zero
        l511_loss_raw = torch.stack(l511_terms).mean() if l511_terms else zero
        smooth_loss = torch.stack(smooth_terms).mean() if smooth_terms else zero
        length_loss_raw = torch.stack(length_terms).mean() if length_terms else zero
        align_loss_raw = torch.stack(align_terms).mean() if align_terms else zero

        # Scaled components — keep magnitudes comparable for TB readability.
        scaled_511 = l511_loss_raw / self.scale_511
        scaled_smooth = smooth_loss / self.scale_smooth
        scaled_length = length_loss_raw / self.scale_length
        scaled_align = align_loss_raw / self.scale_align

        # L_align is folded into the "bp" slot (both push pred bp toward
        # ground truth at probes); we keep its raw value separately for log.
        bp_total_scaled = scaled_511 + self.lambda_align * scaled_align

        total = (
            probe_loss
            + self.lambda_511 * scaled_511
            + self.lambda_smooth * scaled_smooth
            + self.lambda_length * scaled_length
            + self.lambda_align * scaled_align
        )

        # Key mapping: map each V3 physics term into the existing slot name
        # so the trainer's logging doesn't need to change.
        details: dict[str, float] = {
            "probe": float(probe_loss.detach().item()),
            "bp": float(bp_total_scaled.detach().item()),
            "vel": float(scaled_smooth.detach().item()),
            "count": float(scaled_length.detach().item()),
            "probe_raw": float(probe_loss.detach().item()),
            "bp_raw": float(l511_loss_raw.detach().item()),  # mean sq residual (bp^2)
            "vel_raw": float(smooth_loss.detach().item()),
            "count_raw": float(length_loss_raw.detach().item()),  # span residual (bp^2)
            "align_raw": float(align_loss_raw.detach().item()),  # mean L1 (bp)
            "n_align_active": n_align_active,
            "warmstart_blend": blend,
        }
        return total, details
