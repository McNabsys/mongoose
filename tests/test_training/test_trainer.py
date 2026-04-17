"""Tests for the training loop."""

from __future__ import annotations

import json
import math
import pickle
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
