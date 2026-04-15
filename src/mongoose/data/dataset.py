"""PyTorch Dataset implementations for mongoose T2D training.

V1 rearchitecture (R6): items now carry the new GT schema -- reference
bp positions, a reference probe count, and an optional pre-built
warmstart heatmap -- instead of the legacy probe_sample_indices /
gt_deltas_bp / velocity_targets trio.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mongoose.data.heatmap import build_probe_heatmap


class SyntheticMoleculeDataset(Dataset):
    """Generates synthetic molecule data for testing the training loop.

    Produces fake waveforms with probe-like bumps, synthetic
    ``reference_bp_positions`` (ascending integers simulating direction=1),
    and an optional warmstart heatmap for roughly half the molecules. Not
    for real training -- exists to exercise the trainer end-to-end.
    """

    def __init__(
        self,
        num_molecules: int = 100,
        min_length: int = 1000,
        max_length: int = 8000,
        min_probes: int = 5,
        max_probes: int = 20,
        warmstart_probability: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.num_molecules = num_molecules
        self.min_length = min_length
        self.max_length = max_length
        self.min_probes = min_probes
        self.max_probes = max_probes
        self.warmstart_probability = warmstart_probability
        self.seed = seed

    def __len__(self) -> int:
        return self.num_molecules

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)

        # Random waveform length.
        length = int(rng.integers(self.min_length, self.max_length + 1))

        # Baseline waveform: smooth noise, then add probe-like bumps.
        waveform = rng.normal(0.0, 0.1, size=length).astype(np.float32)

        # Number of probes in this molecule.
        n_probes = int(rng.integers(self.min_probes, self.max_probes + 1))

        # Place probes at sorted random sample positions (avoiding edges).
        margin = max(50, length // 20)
        probe_positions = np.sort(
            rng.integers(margin, length - margin, size=n_probes)
        ).astype(np.int64)

        # Probe durations in samples (realistic range ~10-80 samples).
        probe_durations = rng.uniform(10.0, 80.0, size=n_probes).astype(np.float32)

        # Add Gaussian-like bumps to the waveform so it has visible events.
        for pos, dur in zip(probe_positions, probe_durations):
            sigma = max(2.0, dur / 6.0)
            lo = max(0, int(pos - 3 * sigma))
            hi = min(length, int(pos + 3 * sigma) + 1)
            if lo < hi:
                x = np.arange(lo, hi, dtype=np.float32)
                bump = 0.5 * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                waveform[lo:hi] += bump

        # Normalize waveform to roughly zero mean, unit variance.
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

        # Synthetic reference_bp_positions: ascending bp coordinates with
        # ~500-5000 bp gaps (direction=1 semantics). These do NOT need to
        # match the temporal probe positions structurally for synthetic
        # training -- they only need to be valid, ascending int64.
        deltas = rng.uniform(500.0, 5000.0, size=n_probes - 1)
        reference_bp_positions = np.concatenate(
            ([0], np.cumsum(deltas))
        ).astype(np.int64)

        # Warmstart heatmap: produced for ~warmstart_probability of molecules.
        # Uses the probe bump positions so there's a plausible peaky signal.
        if rng.uniform() < self.warmstart_probability:
            warmstart_np = build_probe_heatmap(
                length, probe_positions, probe_durations
            )
            warmstart_heatmap: torch.Tensor | None = torch.from_numpy(warmstart_np)
            warmstart_valid = True
        else:
            warmstart_heatmap = None
            warmstart_valid = False

        # Conditioning vector: 6 physics features (fake but plausible).
        conditioning = rng.normal(0.0, 1.0, size=6).astype(np.float32)

        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),  # [1, T]
            "conditioning": torch.from_numpy(conditioning),  # [6]
            "mask": torch.ones(length, dtype=torch.bool),  # [T]
            "reference_bp_positions": torch.from_numpy(reference_bp_positions),  # [N]
            "n_ref_probes": torch.tensor(n_probes, dtype=torch.long),
            "warmstart_heatmap": warmstart_heatmap,  # [T] or None
            "warmstart_valid": torch.tensor(warmstart_valid, dtype=torch.bool),
            "molecule_uid": idx,
        }
