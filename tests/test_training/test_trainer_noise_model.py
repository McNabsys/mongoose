"""Trainer-level wiring tests for the V4 NoiseModelLoss path."""
from __future__ import annotations

import pytest
import torch

from mongoose.losses.noise_model_loss import NoiseModelLoss
from mongoose.training.config import TrainConfig
from mongoose.training.trainer import Trainer


def _synthetic_config(tmp_path, **overrides) -> TrainConfig:
    base = dict(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=1,
        batch_size=4,
        use_amp=False,  # CPU testing
        use_noise_model=True,
        warmstart_epochs=2,
        warmstart_fade_epochs=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_trainer_selects_noise_model_loss_when_flag_set(tmp_path):
    trainer = Trainer(_synthetic_config(tmp_path))
    assert isinstance(trainer.criterion, NoiseModelLoss)


def test_trainer_optimizer_includes_criterion_parameters(tmp_path):
    """log_S must end up inside the optimizer param groups so it gets updated."""
    trainer = Trainer(_synthetic_config(tmp_path))
    optim_param_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    log_S = trainer.criterion.log_S
    assert id(log_S) in optim_param_ids


def test_trainer_combined_loss_path_unaffected_by_noise_model_addition(tmp_path):
    """Default path (no flag) still uses CombinedLoss with no extra params."""
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=1,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    trainer = Trainer(config)
    assert not isinstance(trainer.criterion, NoiseModelLoss)
    # Optimizer param count should equal model param count (no criterion params).
    optim_params = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    model_params = list(trainer.model.parameters())
    assert len(optim_params) == len(model_params)


def test_trainer_rejects_l511_and_noise_model_simultaneously(tmp_path):
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=8,
        epochs=1,
        batch_size=4,
        use_amp=False,
        use_l511=True,
        use_noise_model=True,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        Trainer(config)


def test_trainer_one_epoch_with_noise_model_runs_and_updates_log_S(tmp_path):
    trainer = Trainer(_synthetic_config(tmp_path, epochs=2))
    log_S_before = trainer.criterion.log_S.detach().clone()
    trainer.fit()
    log_S_after = trainer.criterion.log_S.detach().clone()
    # log_S must have moved -- if it stayed exactly equal, the optimizer
    # is not seeing the parameter (most likely missing from param_groups).
    assert not torch.allclose(log_S_before, log_S_after, atol=1e-7), (
        f"log_S did not update: before={log_S_before.item()!r}, after={log_S_after.item()!r}"
    )
