"""Unit tests for the variable-length batch collator."""
from __future__ import annotations

import torch

from mongoose.data.collate import collate_molecules


def _make_item(t: int, wave_scale: float = 1e-4, wave_offset: float = 5e-4, *,
               centers: list[int] | None = None, warmstart: bool = True):
    """Synthesize one dataset item. Default mimics the real broken scale where
    post-normalize amplitudes land at ~1e-4."""
    waveform = torch.randn(1, t) * wave_scale + wave_offset
    mask = torch.ones(t, dtype=torch.bool)
    conditioning = torch.zeros(6, dtype=torch.float32)
    ref_bp = torch.tensor([100, 200, 300], dtype=torch.long)
    item = {
        "waveform": waveform,
        "conditioning": conditioning,
        "mask": mask,
        "reference_bp_positions": ref_bp,
        "n_ref_probes": torch.tensor(3, dtype=torch.long),
        "molecule_uid": 0,
    }
    if warmstart:
        item["warmstart_heatmap"] = torch.zeros(t, dtype=torch.float32)
        item["warmstart_valid"] = torch.tensor(True, dtype=torch.bool)
    else:
        item["warmstart_heatmap"] = None
        item["warmstart_valid"] = torch.tensor(False, dtype=torch.bool)
    item["warmstart_probe_centers_samples"] = (
        torch.tensor(centers or [10, 50, 100], dtype=torch.long)
    )
    return item


def test_zscore_valid_region_has_unit_stats():
    """After collate, each molecule's valid (mask=True) waveform region should
    have mean ~0 and std ~1 — regardless of the input scale."""
    items = [
        _make_item(t=500, wave_scale=1e-4, wave_offset=5e-4),
        _make_item(t=750, wave_scale=2e-4, wave_offset=3e-4),
        _make_item(t=300, wave_scale=5e-5, wave_offset=1e-3),
    ]
    batch = collate_molecules(items)
    wf = batch["waveform"]  # [B, 1, T]
    mask = batch["mask"]    # [B, T]
    for i in range(wf.shape[0]):
        m = mask[i]
        v = wf[i, 0, m]
        assert abs(float(v.mean())) < 1e-5, (
            f"molecule {i}: post-collate mean should be ~0, got {float(v.mean())}"
        )
        assert abs(float(v.std()) - 1.0) < 1e-3, (
            f"molecule {i}: post-collate std should be ~1, got {float(v.std())}"
        )


def test_zscore_does_not_affect_padding_region():
    """Samples outside the mask (padding) should stay at zero after z-score."""
    items = [
        _make_item(t=500, wave_scale=1e-4),
        _make_item(t=300, wave_scale=1e-4),
    ]
    batch = collate_molecules(items)
    wf = batch["waveform"]
    mask = batch["mask"]
    # Second molecule has t=300; samples [300, padded_len) are padding and
    # should remain exactly 0 (their pre-pad value from torch.zeros init).
    pad_region = wf[1, 0, ~mask[1]]
    assert pad_region.numel() > 0, "expected some padding for shorter molecule"
    assert torch.all(pad_region == 0.0), (
        f"padding region should be zeros, found max abs = {pad_region.abs().max()}"
    )


def test_zscore_constant_waveform_does_not_nan():
    """A molecule whose valid signal is all-constant (std=0) must not divide
    by zero. Eps in the denominator should leave the signal finite."""
    item = _make_item(t=200, wave_scale=0.0, wave_offset=7.0)
    batch = collate_molecules([item])
    wf = batch["waveform"]
    assert torch.isfinite(wf).all(), "NaN/Inf in z-scored constant waveform"
