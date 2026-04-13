"""Dataset that reads from preprocessed cache directories.

Loads compact training data written by preprocess.py:
    waveforms.bin  -- memory-mapped int16 waveforms
    offsets.npy    -- byte offsets and sample counts
    conditioning.npy -- float32 conditioning vectors
    molecules.pkl  -- per-molecule ground truth dicts
    manifest.json  -- metadata and per-molecule info
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mongoose.data.ground_truth import SAMPLE_PERIOD_MS, TAG_WIDTH_BP
from mongoose.data.heatmap import build_probe_heatmap


class CachedMoleculeDataset(Dataset):
    """Dataset that reads from preprocessed cache directories.

    Supports loading from multiple cache directories (one per run),
    presenting them as a single flat dataset.
    """

    def __init__(self, cache_dirs: list[Path], augment: bool = False) -> None:
        """
        Args:
            cache_dirs: List of paths to cache/<run_id>/ directories.
            augment: Whether to apply data augmentations.
        """
        self.cache_dirs = [Path(d) for d in cache_dirs]
        self.augment = augment

        self.entries: list[tuple[int, int]] = []  # (dir_index, mol_index)
        self.manifests: list[dict] = []
        self.offsets_arrays: list[np.ndarray] = []
        self.conditioning_arrays: list[np.ndarray] = []
        self.gt_lists: list[list[dict]] = []
        self.waveform_files: list[np.ndarray] = []  # memory-mapped

        for cache_dir in self.cache_dirs:
            with open(cache_dir / "manifest.json") as f:
                manifest = json.load(f)
            self.manifests.append(manifest)

            offsets = np.load(cache_dir / "offsets.npy")
            self.offsets_arrays.append(offsets)

            conditioning = np.load(cache_dir / "conditioning.npy")
            self.conditioning_arrays.append(conditioning)

            with open(cache_dir / "molecules.pkl", "rb") as f:
                gt_list = pickle.load(f)
            self.gt_lists.append(gt_list)

            # Memory-map the waveform file for fast random access
            wfm_path = cache_dir / "waveforms.bin"
            wfm_size = wfm_path.stat().st_size
            if wfm_size > 0:
                self.waveform_files.append(
                    np.memmap(wfm_path, dtype=np.int16, mode="r")
                )
            else:
                self.waveform_files.append(np.array([], dtype=np.int16))

            # Add entries to global index
            dir_idx = len(self.manifests) - 1
            for mol_idx in range(len(offsets)):
                self.entries.append((dir_idx, mol_idx))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        dir_idx, mol_idx = self.entries[idx]
        manifest = self.manifests[dir_idx]
        mol_info = manifest["molecules"][mol_idx]

        # Read waveform from memory-mapped file
        byte_offset, num_samples = self.offsets_arrays[dir_idx][mol_idx]
        sample_offset = int(byte_offset) // 2  # int16 = 2 bytes per sample
        waveform = np.array(
            self.waveform_files[dir_idx][sample_offset : sample_offset + num_samples],
            dtype=np.float32,
        )

        # Normalize by level-1 amplitude
        amp_scale = mol_info["amplitude_scale"]  # uV per LSB
        mean_lvl1_uv = mol_info["mean_lvl1"] * 1000  # mV -> uV
        if mean_lvl1_uv > 0:
            waveform = (waveform * amp_scale) / mean_lvl1_uv

        # Load GT and conditioning
        gt = self.gt_lists[dir_idx][mol_idx]
        conditioning = self.conditioning_arrays[dir_idx][mol_idx].copy()

        # Build heatmap target
        probe_centers = gt["probe_sample_indices"]
        velocities = gt["velocity_targets_bp_per_ms"]
        durations_ms = TAG_WIDTH_BP / np.clip(velocities, 1e-6, None)
        durations_samples = durations_ms / SAMPLE_PERIOD_MS

        heatmap = build_probe_heatmap(num_samples, probe_centers, durations_samples)

        # Apply augmentations if enabled
        if self.augment:
            from mongoose.data.augment import add_noise, scale_amplitude, time_stretch

            rng = np.random.default_rng()
            # Time stretch
            stretch_factor = rng.uniform(0.9, 1.1)
            waveform = time_stretch(waveform, stretch_factor)
            heatmap = time_stretch(heatmap, stretch_factor)
            probe_centers = (probe_centers * stretch_factor).astype(np.int64)
            num_samples = len(waveform)

            # Noise
            waveform = add_noise(waveform, rms_scale=0.02, rng=rng)
            # Amplitude scale
            waveform = scale_amplitude(waveform, rng.uniform(0.95, 1.05))

        # Convert velocity targets from bp/ms to bp/sample
        velocity_targets_bp_per_sample = gt["velocity_targets_bp_per_ms"] * SAMPLE_PERIOD_MS

        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),  # [1, T]
            "conditioning": torch.from_numpy(conditioning),  # [6]
            "probe_heatmap": torch.from_numpy(heatmap),  # [T]
            "probe_sample_indices": torch.from_numpy(gt["probe_sample_indices"].copy()),
            "gt_deltas_bp": torch.from_numpy(gt["inter_probe_deltas_bp"].astype(np.float32).copy()),
            "velocity_targets": torch.from_numpy(
                velocity_targets_bp_per_sample.astype(np.float32).copy()
            ),
            "mask": torch.ones(num_samples, dtype=torch.bool),
            "molecule_uid": mol_info["uid"],
        }
