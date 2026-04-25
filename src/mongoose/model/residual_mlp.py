"""Per-probe MLP for Direction C residual learning.

Takes a fixed-shape feature vector (per
:mod:`mongoose.data.residual_dataset`) and predicts the per-probe bp
shift production applied (``residual_bp = post_position_bp -
pre_position_bp``).

Architecture: a small MLP with residual connections between hidden
blocks. Defaults are chosen for ~1M parameters total -- enough capacity
to learn TVC's interval-size dependence and head-dive's length-group
splines, small enough to train in seconds per epoch on the H100.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mongoose.data.residual_dataset import FEATURE_DIM


class ResidualBlock(nn.Module):
    """Linear -> LayerNorm -> GELU -> Linear -> LayerNorm -> Dropout, with skip."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.linear2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        return x + h


class ResidualMLP(nn.Module):
    """Per-probe residual-prediction MLP.

    Maps ``[B, FEATURE_DIM]`` feature vectors to ``[B]`` scalar bp-shift
    predictions. Internally: a stem projection to ``hidden_dim``, ``n_blocks``
    of pre-norm residual blocks, then a final head.

    Args:
        input_dim: Feature vector dimension. Defaults to
            :data:`mongoose.data.residual_dataset.FEATURE_DIM` and asserts a
            match against the dataset module to catch silent schema drift.
        hidden_dim: Hidden width inside the residual stack.
        n_blocks: Number of residual blocks.
        dropout: Dropout probability inside each block.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim != FEATURE_DIM:
            raise ValueError(
                f"input_dim={input_dim} does not match FEATURE_DIM={FEATURE_DIM}; "
                "the dataset and model schemas must agree."
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_blocks = int(n_blocks)

        self.stem = nn.Linear(input_dim, hidden_dim)
        self.stem_norm = nn.LayerNorm(hidden_dim)
        self.stem_act = nn.GELU()

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )

        self.head = nn.Linear(hidden_dim, 1)
        # Init head bias near the empirical residual mean (~+2,200 bp on
        # the example run) so the model starts from a non-zero baseline
        # and converges faster. Easy to override at training time if a
        # different prior is observed.
        nn.init.constant_(self.head.bias, 2200.0)
        nn.init.zeros_(self.head.weight)  # zero so initial pred is just the bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict per-sample bp shift.

        Args:
            features: ``[B, input_dim]`` float tensor.

        Returns:
            ``[B]`` float tensor of predicted ``residual_bp``.
        """
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected features shape [B, {self.input_dim}]; got {tuple(features.shape)!r}"
            )
        h = self.stem(features)
        h = self.stem_norm(h)
        h = self.stem_act(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
