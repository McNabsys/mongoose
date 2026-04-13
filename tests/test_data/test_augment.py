"""Tests for data augmentation functions."""

import numpy as np

from mongoose.data.augment import add_noise, scale_amplitude, time_stretch


def test_time_stretch_preserves_shape_approximately():
    waveform = np.ones(1000, dtype=np.float32)
    stretched = time_stretch(waveform, 1.1)
    assert len(stretched) == 1100  # 10% longer
    compressed = time_stretch(waveform, 0.9)
    assert len(compressed) == 900


def test_time_stretch_preserves_values():
    waveform = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    stretched = time_stretch(waveform, 2.0)
    assert len(stretched) == 10
    # Interpolated values should be between 0 and 1
    assert stretched.min() >= -0.01
    assert stretched.max() <= 1.01


def test_add_noise():
    rng = np.random.default_rng(42)
    waveform = np.zeros(10000, dtype=np.float32)
    noisy = add_noise(waveform, rms_scale=1.0, rng=rng)
    assert abs(noisy.std() - 1.0) < 0.1  # roughly unit variance


def test_scale_amplitude():
    waveform = np.ones(100, dtype=np.float32)
    scaled = scale_amplitude(waveform, 1.05)
    assert np.allclose(scaled, 1.05)
