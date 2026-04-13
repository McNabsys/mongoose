"""PyTorch Dataset implementations for mongoose T2D training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mongoose.data.ground_truth import TAG_WIDTH_BP
from mongoose.data.heatmap import build_probe_heatmap


class SyntheticMoleculeDataset(Dataset):
    """Generates synthetic molecule data for testing the training loop.

    Produces realistic-looking fake waveforms with probe-like bumps,
    random GT deltas, and conditioning vectors. Not for real training.
    """

    def __init__(
        self,
        num_molecules: int = 100,
        min_length: int = 1000,
        max_length: int = 8000,
        min_probes: int = 4,
        max_probes: int = 20,
        seed: int = 0,
    ) -> None:
        self.num_molecules = num_molecules
        self.min_length = min_length
        self.max_length = max_length
        self.min_probes = min_probes
        self.max_probes = max_probes
        self.seed = seed

    def __len__(self) -> int:
        return self.num_molecules

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)

        # Random waveform length
        length = rng.integers(self.min_length, self.max_length + 1)

        # Baseline waveform: smooth random signal normalized around 0
        waveform = rng.normal(0.0, 0.1, size=length).astype(np.float32)

        # Number of probes
        n_probes = rng.integers(self.min_probes, self.max_probes + 1)

        # Place probes at sorted random positions (avoiding edges)
        margin = max(50, length // 20)
        probe_positions = np.sort(
            rng.integers(margin, length - margin, size=n_probes)
        )

        # Probe durations in samples (realistic range ~10-100 samples)
        probe_durations = rng.uniform(10.0, 80.0, size=n_probes).astype(np.float32)

        # Add probe-like bumps to the waveform
        for pos, dur in zip(probe_positions, probe_durations):
            sigma = max(2.0, dur / 6.0)
            lo = max(0, int(pos - 3 * sigma))
            hi = min(length, int(pos + 3 * sigma) + 1)
            if lo < hi:
                x = np.arange(lo, hi, dtype=np.float32)
                bump = 0.5 * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                waveform[lo:hi] += bump

        # Normalize waveform to roughly zero mean, unit variance
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

        # Build heatmap target
        probe_heatmap = build_probe_heatmap(length, probe_positions, probe_durations)

        # Random GT deltas (inter-probe distances in bp, realistic range)
        gt_deltas_bp = rng.uniform(500.0, 5000.0, size=n_probes - 1).astype(
            np.float32
        )

        # Velocity targets: TAG_WIDTH_BP / duration_samples (bp/sample)
        velocity_targets = (TAG_WIDTH_BP / probe_durations).astype(np.float32)

        # Conditioning vector: 6 physics features (fake but plausible)
        conditioning = rng.normal(0.0, 1.0, size=6).astype(np.float32)

        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),  # [1, T]
            "conditioning": torch.from_numpy(conditioning),  # [6]
            "probe_heatmap": torch.from_numpy(probe_heatmap),  # [T]
            "probe_sample_indices": torch.from_numpy(
                probe_positions.astype(np.int64)
            ),  # [N]
            "gt_deltas_bp": torch.from_numpy(gt_deltas_bp),  # [N-1]
            "velocity_targets": torch.from_numpy(velocity_targets),  # [N]
            "mask": torch.ones(length, dtype=torch.bool),  # [T]
            "molecule_uid": idx,
        }


class MoleculeDataset(Dataset):
    """Dataset from pre-built (waveform, MoleculeGT, conditioning) tuples.

    Use this when waveforms and ground truth have already been loaded and
    assembled outside the dataset (e.g., from TDB files + probes.bin).
    """

    def __init__(
        self,
        items: list[tuple[np.ndarray, "MoleculeGT", np.ndarray]],
    ) -> None:
        """
        Args:
            items: List of (waveform, molecule_gt, conditioning) tuples.
                waveform: np.ndarray float32 [T]
                molecule_gt: MoleculeGT with probe info and deltas
                conditioning: np.ndarray float32 [6]
        """
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        waveform, gt, conditioning = self.items[idx]

        length = len(waveform)

        # Level-1 normalize: zero mean, unit variance
        wf = waveform.astype(np.float32)
        wf = (wf - wf.mean()) / (wf.std() + 1e-8)

        # Convert probe sample indices to durations in samples for heatmap
        # Estimate probe duration from TAG_WIDTH_BP / velocity (in samples)
        probe_durations = TAG_WIDTH_BP / (
            gt.velocity_targets_bp_per_ms + 1e-8
        )

        probe_heatmap = build_probe_heatmap(
            length,
            gt.probe_sample_indices,
            probe_durations,
        )

        return {
            "waveform": torch.from_numpy(wf).unsqueeze(0),  # [1, T]
            "conditioning": torch.from_numpy(
                conditioning.astype(np.float32)
            ),  # [6]
            "probe_heatmap": torch.from_numpy(probe_heatmap),  # [T]
            "probe_sample_indices": torch.from_numpy(
                gt.probe_sample_indices.astype(np.int64)
            ),  # [N]
            "gt_deltas_bp": torch.from_numpy(
                gt.inter_probe_deltas_bp.astype(np.float32)
            ),  # [N-1]
            "velocity_targets": torch.from_numpy(
                gt.velocity_targets_bp_per_ms.astype(np.float32)
            ),  # [N]
            "mask": torch.ones(length, dtype=torch.bool),  # [T]
            "molecule_uid": idx,
        }
