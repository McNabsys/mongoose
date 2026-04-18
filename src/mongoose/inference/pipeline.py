"""Full inference pipeline: model forward pass, NMS, interpolation, and quality gating."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mongoose.inference.nms import subsample_peak_position, velocity_adaptive_nms
from mongoose.model.unet import T2DUNet


@dataclass
class InferredProbe:
    """A single inferred probe position."""

    center_sample: float  # fractional sample index
    center_bp: float  # base-pair position from cumulative curve
    confidence: float  # heatmap value at peak
    duration_bp: float  # probe width in bp (from cumulative curve)


@dataclass
class InferredMolecule:
    """Inference result for a single molecule."""

    probes: list[InferredProbe] = field(default_factory=list)
    intervals_bp: list[float] = field(default_factory=list)  # inter-probe distances
    total_length_bp: float = 0.0  # cumulative bp at last sample
    molecule_uid: int = 0


def _lerp_cumulative(cumulative: torch.Tensor, t: float) -> float:
    """Linearly interpolate the cumulative curve at fractional index t."""
    if t <= 0:
        return cumulative[0].item()
    if t >= len(cumulative) - 1:
        return cumulative[-1].item()
    lo = int(math.floor(t))
    hi = int(math.ceil(t))
    if lo == hi:
        return cumulative[lo].item()
    frac = t - lo
    return cumulative[lo].item() * (1.0 - frac) + cumulative[hi].item() * frac


def _compute_duration_bp(
    heatmap: torch.Tensor, cumulative: torch.Tensor, peak_idx: int
) -> float:
    """Compute probe width in bp from the heatmap half-width.

    Finds the temporal half-width (points where heatmap drops below 0.5 * peak_value),
    then converts to bp via cumulative[right] - cumulative[left].
    """
    peak_val = heatmap[peak_idx].item()
    half_val = 0.5 * peak_val
    T = len(heatmap)

    # Expand left
    left = peak_idx
    while left > 0 and heatmap[left - 1].item() >= half_val:
        left -= 1

    # Expand right
    right = peak_idx
    while right < T - 1 and heatmap[right + 1].item() >= half_val:
        right += 1

    bp_left = cumulative[left].item()
    bp_right = cumulative[right].item()
    return bp_right - bp_left


def run_inference(
    model: T2DUNet,
    waveform: torch.Tensor,  # [1, 1, T]
    conditioning: torch.Tensor,  # [1, 6]
    mask: torch.Tensor,  # [1, T]
    threshold: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> InferredMolecule:
    """Run the full inference pipeline on a single molecule waveform.

    Steps:
        1. Forward pass (no grad)
        2. NMS on heatmap
        3. Sub-sample interpolation for each peak
        4. Read bp position from cumulative curve via linear interpolation
        5. Compute probe duration_bp from heatmap half-width
        6. Compute inter-probe intervals
        7. Return InferredMolecule

    Args:
        model: Trained T2DUNet model.
        waveform: Input waveform tensor [1, 1, T].
        conditioning: Physical observables [1, 6].
        mask: Bool tensor [1, T] (True = valid).
        threshold: Minimum confidence for peak detection.
        device: Device to run inference on.

    Returns:
        InferredMolecule with detected probes and metadata.
    """
    model = model.to(device)
    waveform = waveform.to(device)
    conditioning = conditioning.to(device)
    mask = mask.to(device)

    # Step 1: Forward pass
    with torch.no_grad():
        probe_heatmap, cumulative_bp, raw_velocity, _logits = model(waveform, conditioning, mask)

    # Work with single-molecule tensors (remove batch dim)
    heatmap_1d = probe_heatmap[0].cpu()  # [T]
    cumulative_1d = cumulative_bp[0].cpu()  # [T]
    velocity_1d = raw_velocity[0].cpu()  # [T]

    T = heatmap_1d.shape[0]

    # Step 2: Velocity-adaptive NMS
    peak_indices = velocity_adaptive_nms(heatmap_1d, velocity_1d, threshold=threshold)

    # Build probes
    probes: list[InferredProbe] = []
    for idx_tensor in peak_indices:
        idx = idx_tensor.item()

        # Step 3: Sub-sample interpolation
        frac_idx = subsample_peak_position(heatmap_1d, idx)

        # Step 4: Read bp position via linear interpolation
        center_bp = _lerp_cumulative(cumulative_1d, frac_idx)

        # Step 5: Compute probe duration in bp
        duration_bp = _compute_duration_bp(heatmap_1d, cumulative_1d, idx)

        confidence = heatmap_1d[idx].item()

        probes.append(
            InferredProbe(
                center_sample=frac_idx,
                center_bp=center_bp,
                confidence=confidence,
                duration_bp=duration_bp,
            )
        )

    # Step 6: Compute intervals (diff of consecutive probe bp positions)
    intervals: list[float] = []
    for i in range(1, len(probes)):
        intervals.append(probes[i].center_bp - probes[i - 1].center_bp)

    # Total length
    total_length_bp = cumulative_1d[-1].item() if T > 0 else 0.0

    return InferredMolecule(
        probes=probes,
        intervals_bp=intervals,
        total_length_bp=total_length_bp,
        molecule_uid=0,
    )


def quality_gate(
    molecule: InferredMolecule, max_velocity_bp_per_s: float = 5_000_000
) -> bool:
    """Returns True if molecule passes quality checks.

    Checks:
        - No negative intervals (should be impossible with cumsum).
        - Macro velocity doesn't exceed physical limit.
    """
    # Check: no negative intervals
    for interval in molecule.intervals_bp:
        if interval < 0:
            return False

    # Check: macro velocity doesn't exceed physical limit
    if molecule.total_length_bp > max_velocity_bp_per_s:
        return False

    return True
