"""Training loop with mixed precision, gradient clipping, and cosine annealing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from mongoose.data.collate import collate_molecules
from mongoose.data.dataset import SyntheticMoleculeDataset
from mongoose.losses.combined import CombinedLoss
from mongoose.model.unet import T2DUNet
from mongoose.training.config import TrainConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Trains the T2D U-Net model."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = T2DUNet(config.in_channels, config.conditioning_dim).to(
            self.device
        )

        # Loss
        self.criterion = CombinedLoss(
            config.lambda_bp, config.lambda_vel, config.warmup_epochs
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_lr
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.use_amp and self.device.type == "cuda"
        )

        # Data
        self._build_dataloaders()

        # Tracking
        self.epoch_metrics: list[dict[str, Any]] = []
        self.start_epoch: int = 0
        self.best_val_loss: float = float("inf")

        # Load checkpoint if one exists
        self._maybe_load_checkpoint()

    def _build_dataloaders(self) -> None:
        """Build train and optional validation dataloaders."""
        config = self.config

        if config.use_synthetic:
            full_dataset = SyntheticMoleculeDataset(
                num_molecules=config.synthetic_num_molecules,
                min_length=config.synthetic_min_length,
                max_length=config.synthetic_max_length,
            )
            # 80/20 train/val split
            n_val = max(1, len(full_dataset) // 5)
            n_train = len(full_dataset) - n_val
            train_dataset, val_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            raise NotImplementedError(
                "Real data loading not yet implemented. Use --synthetic."
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_molecules,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_molecules,
            pin_memory=self.device.type == "cuda",
        )

    def fit(self) -> None:
        """Run the full training loop."""
        for epoch in range(self.start_epoch, self.config.epochs):
            self.criterion.set_epoch(epoch)
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)

            self.scheduler.step()

            combined = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_probe": train_metrics["probe_loss"],
                "train_bp": train_metrics["bp_loss"],
                "train_vel": train_metrics["vel_loss"],
                "val_loss": val_metrics["loss"],
                "val_probe": val_metrics["probe_loss"],
                "val_bp": val_metrics["bp_loss"],
                "val_vel": val_metrics["vel_loss"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.epoch_metrics.append(combined)

            self._log(epoch, train_metrics, val_metrics)
            self._maybe_save_checkpoint(epoch, val_metrics)

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_probe = 0.0
        total_bp = 0.0
        total_vel = 0.0
        num_batches = 0

        for batch in self.train_loader:
            loss, details = self._step(batch)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_probe += details["probe_loss"].item()
            total_bp += details["bp_loss"].item()
            total_vel += details["vel_loss"].item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "probe_loss": total_probe / n,
            "bp_loss": total_bp / n,
            "vel_loss": total_vel / n,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_probe = 0.0
        total_bp = 0.0
        total_vel = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss, details = self._step(batch)
            total_loss += loss.item()
            total_probe += details["probe_loss"].item()
            total_bp += details["bp_loss"].item()
            total_vel += details["vel_loss"].item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "probe_loss": total_probe / n,
            "bp_loss": total_bp / n,
            "vel_loss": total_vel / n,
        }

    def _step(self, batch: dict) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass and loss computation for one batch."""
        waveform = batch["waveform"].to(self.device)
        conditioning = batch["conditioning"].to(self.device)
        mask = batch["mask"].to(self.device)
        target_heatmap = batch["probe_heatmap"].to(self.device)

        # Move per-sample tensors to device
        probe_indices_list = [t.to(self.device) for t in batch["probe_sample_indices"]]
        gt_deltas_list = [t.to(self.device) for t in batch["gt_deltas_bp"]]
        velocity_targets_list = [t.to(self.device) for t in batch["velocity_targets"]]

        with torch.amp.autocast(
            "cuda", enabled=self.config.use_amp and self.device.type == "cuda"
        ):
            probe_heatmap, cumulative_bp, raw_velocity = self.model(
                waveform, conditioning, mask
            )

            loss, details = self.criterion(
                pred_heatmap=probe_heatmap,
                pred_cumulative_bp=cumulative_bp,
                raw_velocity=raw_velocity,
                target_heatmap=target_heatmap,
                probe_indices_list=probe_indices_list,
                gt_deltas_list=gt_deltas_list,
                velocity_targets_list=velocity_targets_list,
                mask=mask,
            )

        return loss, details

    def _log(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        """Print epoch summary to stdout."""
        lr = self.optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch + 1}/{self.config.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"probe={train_metrics['probe_loss']:.4f} | "
            f"bp={train_metrics['bp_loss']:.4f} | "
            f"vel={train_metrics['vel_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"lr={lr:.6f}"
        )
        print(msg)
        logger.info(msg)

    def _maybe_save_checkpoint(
        self, epoch: int, val_metrics: dict[str, float]
    ) -> None:
        """Save checkpoint every N epochs and on best validation loss."""
        if epoch % self.config.save_every != 0 and val_metrics["loss"] >= self.best_val_loss:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch + 1,  # next epoch to train
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Periodic checkpoint
        if epoch % self.config.save_every == 0:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(state, path)
            print(f"  Saved checkpoint: {path}")

        # Best model checkpoint
        if val_metrics["loss"] < self.best_val_loss:
            self.best_val_loss = val_metrics["loss"]
            state["best_val_loss"] = self.best_val_loss
            path = checkpoint_dir / "best_model.pt"
            torch.save(state, path)
            print(f"  Saved best model: {path} (val_loss={self.best_val_loss:.4f})")

    def _maybe_load_checkpoint(self) -> None:
        """Load the latest checkpoint if one exists in checkpoint_dir."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return

        # Find the latest periodic checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return

        latest = checkpoints[-1]
        print(f"Resuming from checkpoint: {latest}")

        state = torch.load(latest, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.start_epoch = state["epoch"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
