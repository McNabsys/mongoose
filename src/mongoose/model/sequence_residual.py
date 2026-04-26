"""V5 Phase 2: per-molecule sequence model with self-attention.

Where :class:`mongoose.model.residual_mlp.ResidualMLP` predicts each
probe's bp shift independently, this model encodes the entire molecule's
probe sequence with a transformer encoder, so each probe's predicted
shift can attend to all other probes in the same molecule.

Why this matters: production's head-dive Method 1 computes a per-
molecule severity by fitting all probe widths to a median curve, then
applies length-group-specific correction splines. The MLP cannot
compute "fit all widths to a curve" because it processes probes
independently; the sequence model can. Similarly, TVC's interval-size
dependence is most informative when contextualized against the
molecule's other intervals -- which the sequence model sees natively.

Architecture sketch (defaults are conservative for ~1-2M params):

    Linear(FEATURE_DIM -> hidden_dim)            ← per-probe feature embedding
    + position_embedding[probe_idx]              ← which probe within the molecule
    n_layers x TransformerEncoderLayer           ← self-attention + FFN, padding-aware
    Linear(hidden_dim -> 1)                      ← per-probe predicted shift
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mongoose.data.residual_dataset import FEATURE_DIM


# Position-encoding capacity. Probes-per-molecule distribution observed
# on the E. coli dataset: median 12, p95 26, p99 35, max 69. 80 covers
# >99.9% of molecules with margin; longer molecules are dropped at
# dataset-build time via the ``max_probes`` arg.
DEFAULT_MAX_SEQ_LEN = 80


class SequenceResidualModel(nn.Module):
    """Per-molecule sequence model for residual bp-shift prediction.

    Args:
        input_dim: Per-probe feature dim. Defaults to
            :data:`mongoose.data.residual_dataset.FEATURE_DIM`.
        hidden_dim: Transformer hidden / embed dim.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads. Must divide hidden_dim.
        dropout: Dropout probability inside encoder layers.
        max_seq_len: Maximum supported probe sequence length. The
            dataset must drop molecules with more probes than this
            (or pad/truncate consistently).
        head_bias_init: Initial value of the output linear layer's
            bias. Useful for centering predictions on the empirical
            target mean. The training script overrides this from
            data at runtime.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        head_bias_init: float = 2200.0,
    ) -> None:
        super().__init__()
        if input_dim != FEATURE_DIM:
            raise ValueError(
                f"input_dim={input_dim} does not match FEATURE_DIM={FEATURE_DIM}; "
                "the dataset and model schemas must agree."
            )
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)

        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.head.bias, float(head_bias_init))
        nn.init.zeros_(self.head.weight)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-probe bp shifts.

        Args:
            features: ``[B, K, FEATURE_DIM]`` float tensor (zero-padded
                at positions where ``padding_mask`` is True).
            padding_mask: ``[B, K]`` bool tensor; True where padded
                (no real probe). Convention matches
                ``nn.TransformerEncoder``'s ``src_key_padding_mask``.

        Returns:
            ``[B, K]`` float tensor of predicted shifts. Padded
            positions return arbitrary values; the loss / eval must
            mask them out using ``padding_mask``.
        """
        B, K, D = features.shape
        if D != self.input_dim:
            raise ValueError(
                f"expected features dim {self.input_dim}, got {D}"
            )
        if K > self.max_seq_len:
            raise ValueError(
                f"sequence length {K} exceeds max_seq_len {self.max_seq_len}; "
                "drop or split long molecules at dataset-build time"
            )

        # Project per-probe features to embedding space + add position
        # encoding. We use *probe index within molecule* as the position,
        # not a continuous position-frac (which is already in features
        # at slot 1). Both can coexist; the embedding adds a learnable
        # categorical signal.
        h = self.feature_proj(features)  # [B, K, H]
        positions = torch.arange(K, device=features.device)
        h = h + self.pos_embedding(positions).unsqueeze(0)  # broadcast over B

        # Encoder. ``src_key_padding_mask`` zeros out attention from
        # padded keys -- real probes don't attend to padding, and
        # padded queries get arbitrary outputs (handled in loss masking).
        h = self.encoder(h, src_key_padding_mask=padding_mask)  # [B, K, H]

        return self.head(h).squeeze(-1)  # [B, K]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
