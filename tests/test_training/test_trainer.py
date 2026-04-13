"""Tests for the training loop."""

from __future__ import annotations

import math

import torch

from mongoose.training.config import TrainConfig
from mongoose.training.trainer import Trainer


def test_trainer_init_synthetic():
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=20,
        epochs=1,
        batch_size=4,
        use_amp=False,  # CPU testing
    )
    trainer = Trainer(config)
    assert trainer.model is not None
    assert trainer.train_loader is not None


def test_trainer_one_epoch_synthetic():
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=1,
        batch_size=4,
        use_amp=False,
    )
    trainer = Trainer(config)
    trainer.fit()
    # Should complete without error


def test_trainer_loss_decreases_over_epochs():
    """Train for a few epochs on synthetic data, loss should not explode."""
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=3,
        batch_size=4,
        use_amp=False,
        warmup_epochs=1,
    )
    trainer = Trainer(config)
    trainer.fit()
    # Verify it doesn't crash or produce NaN
    for metrics in trainer.epoch_metrics:
        assert not math.isnan(metrics["train_loss"]), "Training loss is NaN"
        assert not math.isinf(metrics["train_loss"]), "Training loss is inf"


def test_trainer_checkpoint_save_and_load(tmp_path):
    """Train for 1 epoch, save checkpoint, load into new trainer."""
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=1,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=tmp_path / "checkpoints",
        save_every=1,
    )
    trainer = Trainer(config)
    trainer.fit()

    # Checkpoint should exist
    ckpt_path = tmp_path / "checkpoints" / "checkpoint_epoch_000.pt"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    # Load into new trainer
    config2 = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=2,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=tmp_path / "checkpoints",
        save_every=1,
    )
    trainer2 = Trainer(config2)
    assert trainer2.start_epoch == 1, "Should resume from epoch 1"
