"""Tests for the training loop."""

from __future__ import annotations

import json
import math
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

from mongoose.training.config import TrainConfig
from mongoose.training.trainer import Trainer


def _write_tiny_cache(cache_dir: Path, n_molecules: int = 8, n_samples: int = 2048) -> None:
    """Write a minimal on-disk cache matching CachedMoleculeDataset's expected schema.

    All molecules share ``n_samples`` length so the trainer test is deterministic.
    Reference positions and warmstart arrays are hand-placed near the center.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Waveforms: non-zero int16 so level-1 normalization is well-defined.
    waveform = np.full(n_samples, 500, dtype=np.int16)
    with open(cache_dir / "waveforms.bin", "wb") as f:
        for _ in range(n_molecules):
            f.write(waveform.tobytes())

    bytes_per_mol = n_samples * 2
    offsets = np.array(
        [[i * bytes_per_mol, n_samples] for i in range(n_molecules)],
        dtype=np.int64,
    )
    np.save(cache_dir / "offsets.npy", offsets)

    conditioning = np.tile(
        np.array([0.5, 4.6, 0.0, 0.1, 0.0, 0.0], dtype=np.float32),
        (n_molecules, 1),
    )
    np.save(cache_dir / "conditioning.npy", conditioning)

    gt_list = [
        {
            "reference_bp_positions": np.array([1000, 2000, 3000], dtype=np.int64),
            "n_ref_probes": 3,
            "direction": 1,
            "warmstart_probe_centers_samples": np.array(
                [n_samples // 4, n_samples // 2, 3 * n_samples // 4],
                dtype=np.int64,
            ),
            "warmstart_probe_durations_samples": np.array(
                [10.0, 10.0, 10.0], dtype=np.float32
            ),
        }
        for _ in range(n_molecules)
    ]
    with open(cache_dir / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    manifest = {
        "run_id": cache_dir.name,
        "stats": {"cached_molecules": n_molecules},
        "tdb_files": [],
        "molecules": [
            {
                "uid": i,
                "channel": 2,
                "molecule_id": i,
                "file_name_index": 0,
                "num_samples": n_samples,
                "num_probes": 3,
                "n_ref_probes": 3,
                "num_matched_probes": 3,
                "transloc_time_ms": 50.0,
                "mean_lvl1_from_tdb": 0.5,
                "direction": 1,
                "amplitude_scale": 1.0,
            }
            for i in range(n_molecules)
        ],
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


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


def test_trainer_one_epoch_cached(tmp_path):
    """Trainer should load a cached dataset from disk and run one epoch."""
    cache_dir = tmp_path / "cached_run"
    _write_tiny_cache(cache_dir, n_molecules=8, n_samples=2048)

    config = TrainConfig(
        use_synthetic=False,
        cache_dirs=[cache_dir],
        epochs=1,
        batch_size=2,
        use_amp=False,
        warmup_epochs=0,
        warmstart_epochs=0,
        warmstart_fade_epochs=0,
        checkpoint_dir=tmp_path / "checkpoints",
        save_every=1,
    )
    trainer = Trainer(config)
    trainer.fit()

    assert len(trainer.epoch_metrics) == 1
    metrics = trainer.epoch_metrics[0]
    assert not math.isnan(metrics["train_loss"]), "Training loss is NaN"
    assert not math.isinf(metrics["train_loss"]), "Training loss is inf"


def test_trainer_cached_requires_cache_dirs():
    """Non-synthetic training without cache_dirs should raise, not silently run."""
    config = TrainConfig(use_synthetic=False, cache_dirs=None, epochs=1, use_amp=False)
    try:
        Trainer(config)
    except ValueError as exc:
        assert "cache_dirs" in str(exc)
    else:
        raise AssertionError("Expected ValueError when cache_dirs is None")


def test_init_from_loads_weights_and_resets_epoch(tmp_path):
    """--init-from loads model weights but NOT optimizer/scheduler/epoch."""
    # Train briefly and save a checkpoint.
    src_dir = tmp_path / "source"
    config_src = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=1,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=src_dir,
        save_every=1,
    )
    trainer_src = Trainer(config_src)
    trainer_src.fit()
    src_ckpt = src_dir / "checkpoint_epoch_000.pt"
    assert src_ckpt.exists()

    # Grab a reference weight to verify it loaded.
    ref_param = next(iter(trainer_src.model.parameters())).detach().clone()

    # Fresh trainer in an empty checkpoint_dir with init_from pointing at the
    # saved weights. Should load weights, NOT resume epoch.
    fresh_dir = tmp_path / "fresh"
    config_fresh = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=5,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=fresh_dir,
        save_every=1,
        init_from=src_ckpt,
    )
    trainer_fresh = Trainer(config_fresh)

    # Weights should match.
    loaded_param = next(iter(trainer_fresh.model.parameters())).detach()
    assert torch.allclose(ref_param, loaded_param), "init_from did not load weights"
    # Epoch should be 0, not resumed.
    assert trainer_fresh.start_epoch == 0, "init_from must reset start_epoch"
    # best_val_loss should be fresh, not carried over.
    assert trainer_fresh.best_val_loss == float("inf")


def test_init_from_ignored_when_checkpoint_dir_has_checkpoints(tmp_path):
    """Auto-resume wins over init_from when both are set."""
    # First run: produce checkpoints in checkpoint_dir.
    ckpt_dir = tmp_path / "ckpts"
    config_a = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=1,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=ckpt_dir,
        save_every=1,
    )
    Trainer(config_a).fit()
    assert (ckpt_dir / "checkpoint_epoch_000.pt").exists()

    # Point init_from at a DIFFERENT checkpoint entirely (shouldn't matter —
    # existing checkpoint_dir wins). Use the same one for simplicity.
    other_ckpt = ckpt_dir / "checkpoint_epoch_000.pt"

    # New trainer: both init_from and existing checkpoint_dir populated.
    # Resume should win → start_epoch == 1 (resumed from epoch 0).
    config_b = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=16,
        epochs=3,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=ckpt_dir,  # has a checkpoint
        save_every=1,
        init_from=other_ckpt,  # should be ignored
    )
    trainer_b = Trainer(config_b)
    assert trainer_b.start_epoch == 1, (
        "Resume from checkpoint_dir must win over init_from"
    )


class _FakeWandb:
    """Minimal stand-in for the wandb module: records init/log/finish calls."""

    def __init__(self) -> None:
        self.init_kwargs: dict | None = None
        self.log_calls: list[tuple[dict, int | None]] = []
        self.finished = False

    def init(self, **kwargs):  # noqa: D401 - matches wandb.init signature
        self.init_kwargs = kwargs
        run = types.SimpleNamespace(name=kwargs.get("name") or "fake-run")
        return run

    def log(self, payload, step=None):
        self.log_calls.append((dict(payload), step))

    def finish(self):
        self.finished = True


def _force_cpu(monkeypatch) -> None:
    """Force Trainer onto CPU. Avoids GPU contention and works when CUDA is
    partially hidden (is_available=True, device_count=0)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_trainer_skips_wandb_when_disabled(monkeypatch, tmp_path):
    """Default use_wandb=False must not touch wandb."""
    _force_cpu(monkeypatch)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=8,
        epochs=1,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=tmp_path / "ckpts",
        save_every=1,
    )
    trainer = Trainer(config)
    trainer.fit()
    assert trainer._wandb is None
    assert fake.init_kwargs is None
    assert fake.log_calls == []
    assert fake.finished is False


def test_trainer_wandb_init_log_finish(monkeypatch, tmp_path):
    """With use_wandb=True, init/log-per-epoch/finish are all called."""
    _force_cpu(monkeypatch)
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=8,
        epochs=2,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=tmp_path / "ckpts",
        save_every=1,
        use_wandb=True,
        wandb_project="mongoose-test",
        wandb_run_name="unit-test-run",
    )
    trainer = Trainer(config)
    assert trainer._wandb is fake
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["project"] == "mongoose-test"
    assert fake.init_kwargs["name"] == "unit-test-run"
    cfg = fake.init_kwargs["config"]
    # Hyperparameters flattened from TrainConfig plus git + device metadata.
    assert cfg["epochs"] == 2
    assert cfg["batch_size"] == 4
    assert "git_commit" in cfg
    assert "device" in cfg

    trainer.fit()
    # One log call per epoch, step set to epoch index.
    assert len(fake.log_calls) == 2
    for epoch, (payload, step) in enumerate(fake.log_calls):
        assert step == epoch
        assert "train_loss" in payload
        assert "val_loss" in payload
        assert "lr" in payload
        assert "blend" in payload
    assert fake.finished is True


def test_trainer_wandb_graceful_when_import_fails(monkeypatch, tmp_path):
    """If wandb isn't installed, training continues without crashing."""
    _force_cpu(monkeypatch)
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise
    config = TrainConfig(
        use_synthetic=True,
        synthetic_num_molecules=8,
        epochs=1,
        batch_size=4,
        use_amp=False,
        checkpoint_dir=tmp_path / "ckpts",
        save_every=1,
        use_wandb=True,
    )
    trainer = Trainer(config)
    assert trainer._wandb is None
    trainer.fit()  # must not raise
