"""Physics-informed 1D U-Net that predicts probe heatmap and cumulative base pairs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mongoose.model.blocks import FiLMConditioning, ResBlock


class T2DUNet(nn.Module):
    """1D U-Net with FiLM conditioning, dilated bottleneck, and multi-head self-attention.

    Forward signature:
        forward(x, conditioning, mask) -> (probe_heatmap, cumulative_bp, raw_velocity)
    """

    # Encoder channel progression per level
    ENCODER_CHANNELS = [32, 64, 128, 256, 512]
    KERNEL_SIZE = 7
    NUM_LEVELS = 5
    PAD_MULTIPLE = 32  # 2**NUM_LEVELS

    def __init__(self, in_channels: int = 1, conditioning_dim: int = 6) -> None:
        super().__init__()

        # --- FiLM conditioning (level 0 and bottleneck) ---
        self.film = FiLMConditioning(
            conditioning_dim,
            channel_sizes=[("level0", self.ENCODER_CHANNELS[0]), ("bottleneck", self.ENCODER_CHANNELS[-1])],
        )

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        enc_in = in_channels
        for i, out_ch in enumerate(self.ENCODER_CHANNELS):
            # Channel-changing conv
            pad = (self.KERNEL_SIZE - 1) // 2
            self.encoder_convs.append(
                nn.Sequential(
                    nn.Conv1d(enc_in, out_ch, self.KERNEL_SIZE, padding=pad),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            # 2x ResBlock
            self.encoder_res.append(
                nn.Sequential(
                    ResBlock(out_ch, self.KERNEL_SIZE),
                    ResBlock(out_ch, self.KERNEL_SIZE),
                )
            )
            enc_in = out_ch

        self.pool = nn.MaxPool1d(2)

        # --- Bottleneck ---
        bottleneck_ch = self.ENCODER_CHANNELS[-1]  # 512
        self.bottleneck_res = nn.ModuleList(
            [ResBlock(bottleneck_ch, self.KERNEL_SIZE, dilation=d) for d in [1, 2, 4, 8]]
        )
        self.attention = nn.MultiheadAttention(embed_dim=bottleneck_ch, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(bottleneck_ch)

        # --- Decoder (5 levels, symmetric to encoder) ---
        # Skip connections in reverse: skip[4]=512ch, skip[3]=256ch, skip[2]=128ch, skip[1]=64ch, skip[0]=32ch
        decoder_target_channels = list(reversed(self.ENCODER_CHANNELS))  # [512, 256, 128, 64, 32]
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        dec_in = bottleneck_ch  # 512
        for i, out_ch in enumerate(decoder_target_channels):
            # Skip from encoder at symmetric level
            skip_ch = self.ENCODER_CHANNELS[self.NUM_LEVELS - 1 - i]
            pad = (self.KERNEL_SIZE - 1) // 2
            self.decoder_convs.append(
                nn.Sequential(
                    nn.Conv1d(dec_in + skip_ch, out_ch, self.KERNEL_SIZE, padding=pad),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.decoder_res.append(
                nn.Sequential(
                    ResBlock(out_ch, self.KERNEL_SIZE),
                    ResBlock(out_ch, self.KERNEL_SIZE),
                )
            )
            dec_in = out_ch

        # --- Probe head ---
        final_ch = decoder_target_channels[-1]  # 32
        self.probe_head = nn.Sequential(
            ResBlock(final_ch, self.KERNEL_SIZE),
            ResBlock(final_ch, self.KERNEL_SIZE),
            nn.Conv1d(final_ch, 1, 1),
        )

        # --- Velocity head (wide kernels for smoothing) ---
        self.velocity_head = nn.Sequential(
            ResBlock(final_ch, 31),
            ResBlock(final_ch, 31),
            nn.Conv1d(final_ch, 1, 1),
        )
        self.softplus = nn.Softplus()

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Args:
            x: Waveform tensor [B, 1, T].
            conditioning: Physical observables [B, 6].
            mask: Bool tensor [B, T] (True = valid).

        Returns:
            probe_heatmap [B, T], cumulative_bp [B, T], raw_velocity [B, T].
        """
        B, _, T_orig = x.shape

        # --- Internal padding to multiple of PAD_MULTIPLE ---
        remainder = T_orig % self.PAD_MULTIPLE
        if remainder != 0:
            pad_len = self.PAD_MULTIPLE - remainder
            x = F.pad(x, (0, pad_len))
            mask = F.pad(mask, (0, pad_len), value=False)

        T_padded = x.shape[2]

        # --- Encoder ---
        skips: list[torch.Tensor] = []
        h = x
        for i in range(self.NUM_LEVELS):
            h = self.encoder_convs[i](h)
            if i == 0:
                h = self.film(h, conditioning, "level0")
            h = self.encoder_res[i](h)
            skips.append(h)
            h = self.pool(h)

        # --- Bottleneck ---
        h = self.film(h, conditioning, "bottleneck")
        for res in self.bottleneck_res:
            h = res(h)

        # Multi-head self-attention: [B, C, T'] -> [B, T', C]
        h_t = h.permute(0, 2, 1)
        attn_out, _ = self.attention(h_t, h_t, h_t)
        h_t = self.attn_norm(h_t + attn_out)
        h = h_t.permute(0, 2, 1)  # back to [B, C, T']

        # --- Decoder ---
        # skips are [level0, level1, level2, level3, level4]
        # decoder uses them in reverse: level4, level3, level2, level1, level0
        for i in range(len(self.decoder_convs)):
            h = self.upsample(h)
            skip = skips[self.NUM_LEVELS - 1 - i]  # level 4, 3, 2, 1, 0
            # Handle potential size mismatch from pooling odd lengths
            if h.shape[2] != skip.shape[2]:
                h = h[:, :, : skip.shape[2]]
            h = torch.cat([h, skip], dim=1)
            h = self.decoder_convs[i](h)
            h = self.decoder_res[i](h)

        # h is now [B, 32, T_padded]

        # --- Probe head ---
        probe = self.probe_head(h)  # [B, 1, T_padded]
        probe = torch.sigmoid(probe).squeeze(1)  # [B, T_padded]

        # --- Velocity head ---
        vel = self.velocity_head(h)  # [B, 1, T_padded]
        raw_velocity = self.softplus(vel).squeeze(1)  # [B, T_padded], strictly positive

        # Mask velocity and compute cumulative bp
        masked_velocity = raw_velocity * mask.float()  # zero in padded regions
        cumulative_bp = torch.cumsum(masked_velocity, dim=-1)  # [B, T_padded]

        # --- Trim to original length ---
        probe = probe[:, :T_orig]
        cumulative_bp = cumulative_bp[:, :T_orig]
        raw_velocity = raw_velocity[:, :T_orig]

        return probe, cumulative_bp, raw_velocity
