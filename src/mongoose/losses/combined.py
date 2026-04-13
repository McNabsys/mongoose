"""Combined loss with linear lambda warmup."""

from __future__ import annotations

from typing import Any

import torch

from mongoose.losses.focal import focal_loss
from mongoose.losses.spatial import sparse_huber_delta_loss
from mongoose.losses.velocity import sparse_velocity_loss


class CombinedLoss:
    """L_total = L_probe + lambda_bp * L_bp + lambda_vel * L_velocity.

    Lambda warmup: linear ramp from 0 to target over the first N epochs.
    """

    def __init__(
        self,
        lambda_bp: float = 1.0,
        lambda_vel: float = 1.0,
        warmup_epochs: int = 5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        huber_delta: float = 500.0,
    ) -> None:
        self.lambda_bp = lambda_bp
        self.lambda_vel = lambda_vel
        self.warmup_epochs = warmup_epochs
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.huber_delta = huber_delta

        # Current effective lambdas (updated by set_epoch)
        self.current_lambda_bp: float = 0.0
        self.current_lambda_vel: float = 0.0

    def set_epoch(self, epoch: int) -> None:
        """Update effective lambdas based on current epoch.

        Linear warmup from 0 to target lambda over ``warmup_epochs``.
        """
        if self.warmup_epochs > 0:
            frac = min(1.0, epoch / self.warmup_epochs)
        else:
            frac = 1.0
        self.current_lambda_bp = self.lambda_bp * frac
        self.current_lambda_vel = self.lambda_vel * frac

    def __call__(
        self,
        pred_heatmap: torch.Tensor,
        pred_cumulative_bp: torch.Tensor,
        raw_velocity: torch.Tensor,
        target_heatmap: torch.Tensor,
        probe_indices_list: list[torch.Tensor],
        gt_deltas_list: list[torch.Tensor],
        velocity_targets_list: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute combined loss for a batch.

        Args:
            pred_heatmap: [B, T] predicted probe heatmap (after sigmoid).
            pred_cumulative_bp: [B, T] cumulative base-pair prediction.
            raw_velocity: [B, T] raw velocity head output (bp/sample).
            target_heatmap: [B, T] soft Gaussian target heatmap.
            probe_indices_list: List of B tensors, each [N_probes_i] with
                probe sample indices for that molecule.
            gt_deltas_list: List of B tensors, each [N_probes_i - 1] with
                ground-truth inter-probe deltas.
            velocity_targets_list: List of B tensors, each [N_probes_i] with
                target velocity (bp/sample) at each probe position.
            mask: [B, T] boolean mask for valid (non-padded) samples.

        Returns:
            (total_loss, details_dict) where details_dict contains the
            individual unweighted loss components.
        """
        batch_size = pred_heatmap.shape[0]

        # -- Focal (probe detection) loss --
        probe_loss = focal_loss(
            pred_heatmap,
            target_heatmap,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            mask=mask,
        )

        # -- Sparse Huber delta loss (bp accuracy) --
        bp_losses: list[torch.Tensor] = []
        for b in range(batch_size):
            idx = probe_indices_list[b]
            if idx.numel() < 2:
                continue
            pred_at_probes = pred_cumulative_bp[b, idx]
            bp_losses.append(
                sparse_huber_delta_loss(
                    pred_at_probes, gt_deltas_list[b], delta=self.huber_delta
                )
            )
        bp_loss = (
            torch.stack(bp_losses).mean()
            if bp_losses
            else torch.tensor(0.0, device=pred_heatmap.device)
        )

        # -- Sparse velocity loss --
        vel_losses: list[torch.Tensor] = []
        for b in range(batch_size):
            idx = probe_indices_list[b]
            if idx.numel() == 0:
                continue
            pred_v = raw_velocity[b, idx]
            vel_losses.append(
                sparse_velocity_loss(pred_v, velocity_targets_list[b])
            )
        vel_loss = (
            torch.stack(vel_losses).mean()
            if vel_losses
            else torch.tensor(0.0, device=pred_heatmap.device)
        )

        # -- Combine with warmup-scaled lambdas --
        total = probe_loss + self.current_lambda_bp * bp_loss + self.current_lambda_vel * vel_loss

        details = {
            "probe_loss": probe_loss.detach(),
            "bp_loss": bp_loss.detach(),
            "vel_loss": vel_loss.detach(),
        }
        return total, details
