"""Loss functions for mongoose velocity prediction model."""

from mongoose.losses.centernet_focal import centernet_focal_loss
from mongoose.losses.combined import CombinedLoss
from mongoose.losses.count import count_loss, peakiness_regularizer
from mongoose.losses.focal import focal_loss
from mongoose.losses.peaks import extract_peak_indices, measure_peak_widths_samples
from mongoose.losses.softdtw import soft_dtw
from mongoose.losses.spatial import sparse_huber_delta_loss
from mongoose.losses.velocity import sparse_velocity_loss

__all__ = [
    "CombinedLoss",
    "centernet_focal_loss",
    "count_loss",
    "extract_peak_indices",
    "focal_loss",
    "measure_peak_widths_samples",
    "peakiness_regularizer",
    "soft_dtw",
    "sparse_huber_delta_loss",
    "sparse_velocity_loss",
]
