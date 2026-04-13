"""Tests for the full inference pipeline and quality gating."""

import torch

from mongoose.inference.pipeline import InferredMolecule, InferredProbe, quality_gate, run_inference
from mongoose.model.unet import T2DUNet


def test_inference_returns_probes():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    model.eval()
    waveform = torch.randn(1, 1, 2048)
    conditioning = torch.randn(1, 6)
    mask = torch.ones(1, 2048, dtype=torch.bool)
    result = run_inference(model, waveform, conditioning, mask, threshold=0.1)
    # With random weights, we may or may not get probes
    # Just verify the structure
    assert hasattr(result, "probes")
    assert hasattr(result, "intervals_bp")
    assert hasattr(result, "total_length_bp")


def test_inference_intervals_match_probes():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    model.eval()
    waveform = torch.randn(1, 1, 4096)
    conditioning = torch.randn(1, 6)
    mask = torch.ones(1, 4096, dtype=torch.bool)
    result = run_inference(model, waveform, conditioning, mask, threshold=0.05)
    if len(result.probes) >= 2:
        assert len(result.intervals_bp) == len(result.probes) - 1
        # Intervals should be positive (cumsum is monotonic)
        assert all(d > 0 for d in result.intervals_bp)


def test_quality_gate_accepts_normal():
    mol = InferredMolecule(
        probes=[
            InferredProbe(100.0, 1000.0, 0.9, 510.0),
            InferredProbe(200.0, 5000.0, 0.8, 512.0),
        ],
        intervals_bp=[4000.0],
        total_length_bp=10000.0,
        molecule_uid=0,
    )
    assert quality_gate(mol) is True
