"""Estimate level-1 backbone amplitude from raw TDB waveform.

This replaces the legacy wfmproc-provided mean_lvl1 value (from probes.bin).
The estimator uses the median of backbone samples (between molecule rise and fall
edges), which is robust to tag dips.
"""
from __future__ import annotations

import numpy as np


def estimate_level1(
    waveform: np.ndarray,
    rise_end_idx: int,
    fall_min_idx: int,
    trim_fraction: float = 0.1,
) -> float:
    """Estimate level-1 backbone amplitude using a trimmed median.

    Args:
        waveform: 1D int16 array of raw voltage samples.
        rise_end_idx: Sample index where the rising edge completes (molecule fully in channel).
        fall_min_idx: Sample index where the falling edge begins (molecule starting to exit).
        trim_fraction: Fraction to trim from each tail before median (0-0.5).

    Returns:
        Level-1 backbone amplitude as a Python float.
    """
    if fall_min_idx <= rise_end_idx:
        backbone = waveform.astype(np.float32)
    else:
        backbone = waveform[rise_end_idx:fall_min_idx].astype(np.float32)

    if backbone.size < 10:
        return float(np.median(waveform.astype(np.float32)))

    lo, hi = np.percentile(backbone, [trim_fraction * 100, (1.0 - trim_fraction) * 100])
    trimmed = backbone[(backbone >= lo) & (backbone <= hi)]
    if trimmed.size == 0:
        return float(np.median(backbone))
    return float(np.median(trimmed))
