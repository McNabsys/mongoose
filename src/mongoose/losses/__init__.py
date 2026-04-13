"""Loss functions for mongoose velocity prediction model."""

from mongoose.losses.combined import CombinedLoss
from mongoose.losses.focal import focal_loss
from mongoose.losses.spatial import sparse_huber_delta_loss
from mongoose.losses.velocity import sparse_velocity_loss

__all__ = [
    "CombinedLoss",
    "focal_loss",
    "sparse_huber_delta_loss",
    "sparse_velocity_loss",
]
