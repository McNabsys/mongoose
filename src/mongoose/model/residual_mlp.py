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


# Index of the length_group_bin feature within the FEATURE_DIM vector.
# The bin is also a numeric feature in slot 21 (kept for backward-
# compatibility with the V4-C and V5-Lite checkpoints), but Phase 1.5
# also embeds it as a categorical so the model can learn distinct
# correction curves per length-group (the production head-dive Method
# 1 uses 16 length-group splines this way).
LENGTH_GROUP_BIN_FEATURE_IDX = 21
LENGTH_GROUP_NUM_BINS = 16  # bins 0..15
LENGTH_GROUP_SENTINEL_IDX = 16  # remap -1 (out-of-range) to this slot
LENGTH_GROUP_EMBED_DIM = 16


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
        use_length_group_embed: bool = True,
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
        self.use_length_group_embed = bool(use_length_group_embed)

        # Phase 1.5: length-group embedding. Production's head-dive
        # Method 1 uses 16 length-group splines with separate correction
        # curves; we mirror that structure with a learned embedding so
        # the model can produce length-group-conditioned corrections
        # directly.
        if self.use_length_group_embed:
            # +1 slot for the -1 sentinel (out-of-range molecules).
            self.length_group_embed = nn.Embedding(
                LENGTH_GROUP_NUM_BINS + 1, LENGTH_GROUP_EMBED_DIM
            )
            stem_input_dim = input_dim + LENGTH_GROUP_EMBED_DIM
        else:
            self.length_group_embed = None
            stem_input_dim = input_dim

        self.stem = nn.Linear(stem_input_dim, hidden_dim)
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

        # Length-group embedding (Phase 1.5): extract the length_group_bin
        # value from feature slot 21, map -1 sentinel to the dedicated
        # embedding slot, look up + concat. The numeric slot stays in the
        # feature vector too (cheap redundancy; lets the model use either
        # representation as needed).
        if self.length_group_embed is not None:
            bin_long = features[:, LENGTH_GROUP_BIN_FEATURE_IDX].long()
            bin_long = torch.where(
                bin_long < 0,
                torch.full_like(bin_long, LENGTH_GROUP_SENTINEL_IDX),
                bin_long.clamp(0, LENGTH_GROUP_NUM_BINS - 1),
            )
            bin_emb = self.length_group_embed(bin_long)
            stem_input = torch.cat([features, bin_emb], dim=-1)
        else:
            stem_input = features

        h = self.stem(stem_input)
        h = self.stem_norm(h)
        h = self.stem_act(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
