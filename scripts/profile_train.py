"""Per-step timing breakdown for the training loop.

Temporary diagnostic: instruments Trainer with CUDA-synced timers at each
boundary (loader wait, H2D transfer, forward, criterion, backward, opt step)
and prints aggregate + per-step costs after a short run.

Usage:
    python scripts/profile_train.py --cache-dir <cache> --max-molecules 64 \\
        --batch-size 8 --num-workers 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from mongoose.training.config import TrainConfig
from mongoose.training.trainer import Trainer


class Timings:
    def __init__(self) -> None:
        self.buckets: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.buckets[name] = self.buckets.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, total_wall: float, n_molecules: int) -> None:
        tracked = sum(self.buckets.values())
        print(f"\n=== Per-phase timing ({n_molecules} molecules, {total_wall:.2f}s wall) ===")
        rows = sorted(self.buckets.items(), key=lambda kv: -kv[1])
        for name, total in rows:
            count = self.counts[name]
            per = total / count if count else 0.0
            share = 100.0 * total / total_wall if total_wall else 0.0
            print(f"  {name:24s}  total={total:7.2f}s  per_call={per*1000:8.2f}ms  calls={count:5d}  share={share:5.1f}%")
        untracked = total_wall - tracked
        print(f"  {'UNTRACKED':24s}  total={untracked:7.2f}s  share={100.0 * untracked / total_wall if total_wall else 0:5.1f}%")
        per_mol = total_wall / n_molecules if n_molecules else 0
        print(f"\n  per_molecule_wall = {per_mol*1000:.2f} ms  ({per_mol:.3f} s)")


def instrument_criterion(trainer: Trainer, timings: Timings) -> None:
    """Wrap criterion hot-spots with CUDA-synced timers."""
    from mongoose.losses import peaks as peaks_mod
    from mongoose.losses import softdtw as softdtw_mod

    def sync() -> None:
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()

    orig_extract = peaks_mod.extract_peak_indices
    orig_widths = peaks_mod.measure_peak_widths_samples
    orig_soft_dtw = softdtw_mod.soft_dtw

    def timed_extract(*args, **kwargs):
        sync()
        t0 = time.perf_counter()
        out = orig_extract(*args, **kwargs)
        sync()
        timings.add("crit.extract_peaks", time.perf_counter() - t0)
        return out

    def timed_widths(*args, **kwargs):
        sync()
        t0 = time.perf_counter()
        out = orig_widths(*args, **kwargs)
        sync()
        timings.add("crit.peak_widths", time.perf_counter() - t0)
        return out

    def timed_soft_dtw(*args, **kwargs):
        sync()
        t0 = time.perf_counter()
        out = orig_soft_dtw(*args, **kwargs)
        sync()
        timings.add("crit.soft_dtw", time.perf_counter() - t0)
        return out

    peaks_mod.extract_peak_indices = timed_extract
    peaks_mod.measure_peak_widths_samples = timed_widths
    softdtw_mod.soft_dtw = timed_soft_dtw
    # CombinedLoss imported these by name, so patch there too.
    from mongoose.losses import combined as combined_mod
    combined_mod.extract_peak_indices = timed_extract
    combined_mod.measure_peak_widths_samples = timed_widths
    combined_mod.soft_dtw = timed_soft_dtw


def instrument(trainer: Trainer, timings: Timings) -> None:
    """Monkey-patch trainer methods with CUDA-synced timers."""

    instrument_criterion(trainer, timings)

    def sync() -> None:
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()

    original_train_one_epoch = trainer._train_one_epoch

    def timed_train_one_epoch(epoch: int) -> dict[str, float]:
        trainer.model.train()
        total_loss = 0.0
        total_probe = total_bp = total_vel = total_count = 0.0
        num_batches = 0

        loader = trainer.train_loader
        loader_iter = iter(loader)
        sync()
        t_wait_start = time.perf_counter()
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            sync()
            timings.add("loader_wait", time.perf_counter() - t_wait_start)

            t_h2d_start = time.perf_counter()
            waveform = batch["waveform"].to(trainer.device)
            conditioning = batch["conditioning"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            reference_bp_positions_list = [
                bp.to(trainer.device) for bp in batch["reference_bp_positions"]
            ]
            n_ref_probes = batch["n_ref_probes"].to(trainer.device)
            warmstart_heatmap = batch.get("warmstart_heatmap")
            if warmstart_heatmap is not None:
                warmstart_heatmap = warmstart_heatmap.to(trainer.device)
            warmstart_valid = batch.get("warmstart_valid")
            if warmstart_valid is not None:
                warmstart_valid = warmstart_valid.to(trainer.device)
            sync()
            timings.add("h2d_transfer", time.perf_counter() - t_h2d_start)

            t_fwd_start = time.perf_counter()
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=trainer.config.use_amp and trainer.device.type == "cuda",
            ):
                probe_heatmap, cumulative_bp, raw_velocity = trainer.model(
                    waveform, conditioning, mask
                )
            sync()
            timings.add("forward", time.perf_counter() - t_fwd_start)

            t_crit_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=False):
                loss, details = trainer.criterion(
                    pred_heatmap=probe_heatmap.float(),
                    pred_cumulative_bp=cumulative_bp.float(),
                    raw_velocity=raw_velocity.float(),
                    reference_bp_positions_list=reference_bp_positions_list,
                    n_ref_probes=n_ref_probes,
                    warmstart_heatmap=warmstart_heatmap,
                    warmstart_valid=warmstart_valid,
                    mask=mask,
                )
            sync()
            timings.add("criterion", time.perf_counter() - t_crit_start)

            t_bwd_start = time.perf_counter()
            trainer.optimizer.zero_grad()
            loss.backward()
            sync()
            timings.add("backward", time.perf_counter() - t_bwd_start)

            t_opt_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), trainer.config.grad_clip_norm
            )
            trainer.optimizer.step()
            sync()
            timings.add("optimizer", time.perf_counter() - t_opt_start)

            total_loss += loss.item()
            total_probe += float(details["probe"])
            total_bp += float(details["bp"])
            total_vel += float(details["vel"])
            total_count += float(details["count"])
            num_batches += 1

            t_wait_start = time.perf_counter()

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "probe_loss": total_probe / n,
            "bp_loss": total_bp / n,
            "vel_loss": total_vel / n,
            "count_loss": total_count / n,
        }

    trainer._train_one_epoch = timed_train_one_epoch

    original_validate = trainer._validate

    def timed_validate(epoch: int) -> dict[str, float]:
        sync()
        t0 = time.perf_counter()
        out = original_validate(epoch)
        sync()
        timings.add("VALIDATE_total", time.perf_counter() - t0)
        return out

    trainer._validate = timed_validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        cache_dirs=[args.cache_dir],
        max_molecules=args.max_molecules,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=1,
        warmstart_epochs=1,
        warmstart_fade_epochs=1,
        use_amp=not args.no_amp,
        checkpoint_dir=Path("profile_checkpoints"),
        save_every=999,
    )

    trainer = Trainer(config)
    timings = Timings()
    instrument(trainer, timings)

    t0 = time.perf_counter()
    trainer.fit()
    total_wall = time.perf_counter() - t0

    # Count molecules processed in the train split.
    dataset_len = len(trainer.train_loader.dataset)  # type: ignore[arg-type]
    timings.report(total_wall, dataset_len)


if __name__ == "__main__":
    main()
