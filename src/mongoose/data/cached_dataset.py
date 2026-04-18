"""Dataset that reads from preprocessed cache directories.

Loads compact training data written by preprocess.py:
    waveforms.bin  -- memory-mapped int16 waveforms
    offsets.npy    -- byte offsets and sample counts
    conditioning.npy -- float32 conditioning vectors
    molecules.pkl  -- per-molecule ground truth dicts (V1 schema)
    manifest.json  -- metadata and per-molecule info

V1 rearchitecture (R6): items emitted by ``__getitem__`` carry the new
schema -- ``reference_bp_positions``, ``n_ref_probes``, and an optional
``warmstart_heatmap`` built in-memory from the cached
``warmstart_probe_centers_samples`` / ``warmstart_probe_durations_samples``
arrays. The legacy probe_sample_indices / gt_deltas_bp / velocity_targets
fields are no longer produced.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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

            # Memory-map the waveform file for fast random access.
            wfm_path = cache_dir / "waveforms.bin"
            wfm_size = wfm_path.stat().st_size
            if wfm_size > 0:
                self.waveform_files.append(
                    np.memmap(wfm_path, dtype=np.int16, mode="r")
                )
            else:
                self.waveform_files.append(np.array([], dtype=np.int16))

            dir_idx = len(self.manifests) - 1
            for mol_idx in range(len(offsets)):
                self.entries.append((dir_idx, mol_idx))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        dir_idx, mol_idx = self.entries[idx]
        manifest = self.manifests[dir_idx]
        mol_info = manifest["molecules"][mol_idx]

        # Read waveform from memory-mapped file.
        byte_offset, num_samples = self.offsets_arrays[dir_idx][mol_idx]
        sample_offset = int(byte_offset) // 2  # int16 = 2 bytes per sample
        waveform = np.array(
            self.waveform_files[dir_idx][sample_offset : sample_offset + num_samples],
            dtype=np.float32,
        )

        # Normalize by level-1 amplitude.
        amp_scale = mol_info["amplitude_scale"]  # uV per LSB
        mean_lvl1_uv = mol_info["mean_lvl1_from_tdb"] * 1000  # mV -> uV
        if mean_lvl1_uv > 0:
            waveform = (waveform * amp_scale) / mean_lvl1_uv

        # Load GT (new schema) and conditioning.
        gt = self.gt_lists[dir_idx][mol_idx]
        conditioning = self.conditioning_arrays[dir_idx][mol_idx].copy()

        reference_bp_positions = np.asarray(
            gt["reference_bp_positions"], dtype=np.int64
        ).copy()
        n_ref_probes = int(gt["n_ref_probes"])

        warmstart_centers = gt.get("warmstart_probe_centers_samples")
        warmstart_durations = gt.get("warmstart_probe_durations_samples")

        # Build warmstart heatmap in-memory when the cached arrays are
        # present; otherwise this molecule contributes no warmstart signal.
        if warmstart_centers is not None and warmstart_durations is not None:
            warmstart_np = build_probe_heatmap(
                int(num_samples), warmstart_centers, warmstart_durations
            )
        else:
            warmstart_np = None

        # Apply augmentations if enabled. Only time_stretch affects the
        # sample-space tensors; noise / amplitude only touch the waveform.
        if self.augment:
            from mongoose.data.augment import add_noise, scale_amplitude, time_stretch

            rng = np.random.default_rng()
            stretch_factor = rng.uniform(0.9, 1.1)
            waveform = time_stretch(waveform, stretch_factor)
            if warmstart_np is not None:
                warmstart_np = time_stretch(warmstart_np, stretch_factor)
            num_samples = len(waveform)

            waveform = add_noise(waveform, rms_scale=0.02, rng=rng)
            waveform = scale_amplitude(waveform, rng.uniform(0.95, 1.05))

        if warmstart_np is not None:
            warmstart_heatmap: torch.Tensor | None = torch.from_numpy(warmstart_np)
            warmstart_valid = True
        else:
            warmstart_heatmap = None
            warmstart_valid = False

        # Raw warmstart center positions for evaluator use; may be None.
        raw_centers = warmstart_centers  # numpy array (int64) or None

        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),  # [1, T]
            "conditioning": torch.from_numpy(conditioning),  # [6]
            "mask": torch.ones(int(num_samples), dtype=torch.bool),  # [T]
            "reference_bp_positions": torch.from_numpy(reference_bp_positions),  # [N]
            "n_ref_probes": torch.tensor(n_ref_probes, dtype=torch.long),
            "warmstart_heatmap": warmstart_heatmap,  # [T] or None
            "warmstart_valid": torch.tensor(warmstart_valid, dtype=torch.bool),
            "warmstart_probe_centers_samples": (
                torch.from_numpy(np.asarray(raw_centers, dtype=np.int64))
                if raw_centers is not None else None
            ),  # LongTensor[K] or None
            "molecule_uid": mol_info["uid"],
        }
