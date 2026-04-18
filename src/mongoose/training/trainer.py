"""Training loop with mixed precision, gradient clipping, and cosine annealing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from mongoose.data.cached_dataset import CachedMoleculeDataset
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

        # Loss (V1 rearchitecture: CombinedLoss takes the full scheduler +
        # detection params from the training config).
        self.criterion = CombinedLoss(
            lambda_bp=config.lambda_bp,
            lambda_vel=config.lambda_vel,
            lambda_count=config.lambda_count,
            warmup_epochs=config.warmup_epochs,
            warmstart_epochs=config.warmstart_epochs,
            warmstart_fade_epochs=config.warmstart_fade_epochs,
            softdtw_gamma=config.softdtw_gamma,
            peakiness_window=config.peakiness_window,
            nms_threshold=config.nms_threshold,
            min_blend=config.min_blend,
            scale_probe=config.scale_probe,
            scale_bp=config.scale_bp,
            scale_vel=config.scale_vel,
            scale_count=config.scale_count,
            probe_pos_weight=config.probe_pos_weight,
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

        # bfloat16 autocast has fp32 exponent range, so GradScaler is unnecessary.
        # Keep the object so checkpoint load/save shape doesn't change.
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

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
        else:
            if not config.cache_dirs:
                raise ValueError(
                    "Non-synthetic training requires TrainConfig.cache_dirs to "
                    "be a non-empty list of preprocessed cache directories."
                )
            full_dataset = CachedMoleculeDataset(
                config.cache_dirs, augment=config.augment_train
            )
            if config.max_molecules is not None and config.max_molecules < len(
                full_dataset
            ):
                # Deterministic subsample for smoke training.
                rng = np.random.default_rng(config.split_seed)
                indices = rng.choice(
                    len(full_dataset), size=config.max_molecules, replace=False
                ).tolist()
                full_dataset = Subset(full_dataset, indices)

        # Train/val split (shared by synthetic and cached datasets).
        n_total = len(full_dataset)
        n_val = max(1, int(round(n_total * config.val_fraction)))
        n_val = min(n_val, n_total - 1) if n_total > 1 else 0
        n_train = n_total - n_val
        if n_val == 0:
            # Single-molecule edge case: reuse the sole sample for both loaders.
            train_dataset = full_dataset
            val_dataset = full_dataset
        else:
            train_dataset, val_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(config.split_seed),
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
                "train_count": train_metrics["count_loss"],
                "train_probe_raw": train_metrics["probe_raw"],
                "train_bp_raw": train_metrics["bp_raw"],
                "train_vel_raw": train_metrics["vel_raw"],
                "train_count_raw": train_metrics["count_raw"],
                "val_loss": val_metrics["loss"],
                "val_probe": val_metrics["probe_loss"],
                "val_bp": val_metrics["bp_loss"],
                "val_vel": val_metrics["vel_loss"],
                "val_count": val_metrics["count_loss"],
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
        total_count = 0.0
        total_probe_raw = 0.0
        total_bp_raw = 0.0
        total_vel_raw = 0.0
        total_count_raw = 0.0
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
            total_probe += float(details["probe"])
            total_bp += float(details["bp"])
            total_vel += float(details["vel"])
            total_count += float(details["count"])
            total_probe_raw += float(details["probe_raw"])
            total_bp_raw += float(details["bp_raw"])
            total_vel_raw += float(details["vel_raw"])
            total_count_raw += float(details["count_raw"])
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "probe_loss": total_probe / n,
            "bp_loss": total_bp / n,
            "vel_loss": total_vel / n,
            "count_loss": total_count / n,
            "probe_raw": total_probe_raw / n,
            "bp_raw": total_bp_raw / n,
            "vel_raw": total_vel_raw / n,
            "count_raw": total_count_raw / n,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_probe = 0.0
        total_bp = 0.0
        total_vel = 0.0
        total_count = 0.0
        total_probe_raw = 0.0
        total_bp_raw = 0.0
        total_vel_raw = 0.0
        total_count_raw = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss, details = self._step(batch)
            total_loss += loss.item()
            total_probe += float(details["probe"])
            total_bp += float(details["bp"])
            total_vel += float(details["vel"])
            total_count += float(details["count"])
            total_probe_raw += float(details["probe_raw"])
            total_bp_raw += float(details["bp_raw"])
            total_vel_raw += float(details["vel_raw"])
            total_count_raw += float(details["count_raw"])
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "probe_loss": total_probe / n,
            "bp_loss": total_bp / n,
            "vel_loss": total_vel / n,
            "count_loss": total_count / n,
            "probe_raw": total_probe_raw / n,
            "bp_raw": total_bp_raw / n,
            "vel_raw": total_vel_raw / n,
            "count_raw": total_count_raw / n,
        }

    def _step(self, batch: dict) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass and loss computation for one batch."""
        waveform = batch["waveform"].to(self.device)
        conditioning = batch["conditioning"].to(self.device)
        mask = batch["mask"].to(self.device)

        # Move per-molecule reference bp tensors to device.
        reference_bp_positions_list = [
            bp.to(self.device) for bp in batch["reference_bp_positions"]
        ]
        n_ref_probes = batch["n_ref_probes"].to(self.device)

        warmstart_heatmap = batch.get("warmstart_heatmap")
        if warmstart_heatmap is not None:
            warmstart_heatmap = warmstart_heatmap.to(self.device)
        warmstart_valid = batch.get("warmstart_valid")
        if warmstart_valid is not None:
            warmstart_valid = warmstart_valid.to(self.device)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.config.use_amp and self.device.type == "cuda",
        ):
            probe_heatmap, cumulative_bp, raw_velocity, probe_logits = self.model(
                waveform, conditioning, mask
            )

        # Loss runs in fp32: probe BCE's numeric stability, focal_loss eps
        # clamp, and soft_dtw's squared cost matrix all need fp32.
        with torch.amp.autocast("cuda", enabled=False):
            loss, details = self.criterion(
                pred_heatmap=probe_heatmap.float(),
                pred_cumulative_bp=cumulative_bp.float(),
                raw_velocity=raw_velocity.float(),
                reference_bp_positions_list=reference_bp_positions_list,
                n_ref_probes=n_ref_probes,
                warmstart_heatmap=warmstart_heatmap,
                warmstart_valid=warmstart_valid,
                mask=mask,
                pred_heatmap_logits=probe_logits.float(),
            )

        return loss, details

    def _log(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        """Print epoch summary to stdout."""
        lr = self.optimizer.param_groups[0]["lr"]
        blend = float(getattr(self.criterion, "_warmstart_blend", 0.0))
        msg = (
            f"Epoch {epoch + 1}/{self.config.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"probe={train_metrics['probe_loss']:.4f} | "
            f"bp={train_metrics['bp_loss']:.4f} | "
            f"vel={train_metrics['vel_loss']:.4f} | "
            f"count={train_metrics['count_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"blend={blend:.3f} | "
            f"lr={lr:.6f} | "
            f"raw[p={train_metrics['probe_raw']:.2f} "
            f"bp={train_metrics['bp_raw']:.0f} "
            f"vel={train_metrics['vel_raw']:.0f} "
            f"count={train_metrics['count_raw']:.2f}]"
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
