import numpy as np
from mongoose.data.level1 import estimate_level1


def test_estimate_level1_flat_backbone():
    """Flat backbone value with higher pre-molecule baseline should yield the backbone value."""
    waveform = np.ones(1000, dtype=np.int16) * 1000
    waveform[:50] = 2000  # pre-molecule open channel (higher voltage, no DNA)
    waveform[950:] = 2000  # post-molecule open channel
    lvl1 = estimate_level1(waveform, rise_end_idx=50, fall_min_idx=950)
    assert abs(lvl1 - 1000) < 5.0


def test_estimate_level1_with_tag_dips():
    """Median is robust to tag-induced dips below backbone."""
    waveform = np.ones(1000, dtype=np.int16) * 1000
    waveform[:50] = 2000
    waveform[950:] = 2000
    # Three tag dips (tags cause voltage to drop further below open-channel)
    waveform[200:215] = 500
    waveform[400:418] = 500
    waveform[700:712] = 500
    lvl1 = estimate_level1(waveform, rise_end_idx=50, fall_min_idx=950)
    assert 990 < lvl1 < 1010


def test_estimate_level1_all_noise():
    """Noise around a mean value should give a sensible estimate."""
    rng = np.random.default_rng(42)
    waveform = rng.normal(1000, 50, 1000).astype(np.int16)
    lvl1 = estimate_level1(waveform, rise_end_idx=0, fall_min_idx=1000)
    assert 950 < lvl1 < 1050


def test_estimate_level1_degenerate_indices():
    """When rise_end >= fall_min, fall back to median of full waveform."""
    waveform = np.ones(1000, dtype=np.int16) * 1000
    lvl1 = estimate_level1(waveform, rise_end_idx=500, fall_min_idx=400)
    assert abs(lvl1 - 1000) < 1.0


def test_estimate_level1_short_backbone():
    """Very short backbone still returns a reasonable estimate."""
    waveform = np.ones(50, dtype=np.int16) * 800
    lvl1 = estimate_level1(waveform, rise_end_idx=10, fall_min_idx=40)
    assert abs(lvl1 - 800) < 5.0


def test_estimate_level1_returns_float():
    """Return value should be a Python float, not a numpy scalar."""
    waveform = np.ones(100, dtype=np.int16) * 500
    lvl1 = estimate_level1(waveform, rise_end_idx=10, fall_min_idx=90)
    assert isinstance(lvl1, float)
