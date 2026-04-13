"""Reusable building blocks for the 1D U-Net: ResBlock and FiLM conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with two dilated convolutions and BatchNorm.

    x -> Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+x) -> ReLU
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation from a conditioning vector.

    conditioning [B, cond_dim] -> shared MLP -> per-level (gamma, beta)
    Applied as: features = gamma * features + beta
    """

    def __init__(self, conditioning_dim: int, channel_sizes: list[int], hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # One projection head per conditioned level
        self.heads = nn.ModuleDict()
        for name, ch in channel_sizes:
            self.heads[name] = nn.Linear(hidden_dim, 2 * ch)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor, level_name: str) -> torch.Tensor:
        """Apply FiLM to *features* [B, C, T] using *conditioning* [B, cond_dim]."""
        h = self.shared(conditioning)  # [B, hidden]
        params = self.heads[level_name](h)  # [B, 2*C]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, C]
        return gamma.unsqueeze(-1) * features + beta.unsqueeze(-1)
