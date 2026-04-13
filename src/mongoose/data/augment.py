"""Data augmentation functions for 1D waveforms."""

from __future__ import annotations

import numpy as np


def time_stretch(waveform: np.ndarray, factor: float) -> np.ndarray:
    """Resample waveform by factor. factor>1 = stretch (slower), factor<1 = compress.

    Uses linear interpolation via np.interp.
    """
    orig_len = len(waveform)
    new_len = int(round(orig_len * factor))
    if new_len <= 0:
        return np.empty(0, dtype=waveform.dtype)
    old_x = np.arange(orig_len, dtype=np.float64)
    new_x = np.linspace(0, orig_len - 1, new_len, dtype=np.float64)
    return np.interp(new_x, old_x, waveform).astype(waveform.dtype)


def add_noise(
    waveform: np.ndarray, rms_scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise scaled to rms_scale."""
    noise = rng.normal(0.0, rms_scale, size=waveform.shape).astype(waveform.dtype)
    return waveform + noise


def scale_amplitude(waveform: np.ndarray, factor: float) -> np.ndarray:
    """Scale waveform amplitude by factor."""
    return waveform * factor
