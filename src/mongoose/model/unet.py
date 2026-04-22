"""Physics-informed 1D U-Net that predicts probe heatmap and cumulative base pairs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mongoose.model.blocks import FiLMConditioning, ResBlock

# TDB sample rate in Hz (used when computing v_T2D from per-molecule T2D
# params in hybrid mode).
TDB_SAMPLE_RATE_HZ = 32000
# Minimum ``t_from_tail_ms`` to avoid a singularity where the power-law
# derivative diverges at the trailing edge. 2 ms = 64 samples at 32 kHz —
# well inside the physically unmodeled tail transition region anyway.
T2D_MIN_T_FROM_TAIL_MS = 2.0
# Residual bound for the T2D-hybrid velocity modulation. The residual is
# tanh'd and scaled to ``[-0.5, +0.5]`` so ``v_final = v_T2D * (1 + residual)``
# stays within ``[0.5 * v_T2D, 1.5 * v_T2D]`` — physically plausible.
T2D_RESIDUAL_BOUND = 0.5


def compute_v_t2d(
    mult_const: torch.Tensor,
    alpha: torch.Tensor,
    tail_ms: torch.Tensor,
    T: int,
    *,
    sample_rate_hz: int = TDB_SAMPLE_RATE_HZ,
    min_t_from_tail_ms: float = T2D_MIN_T_FROM_TAIL_MS,
) -> torch.Tensor:
    """Compute per-sample v_T2D in bp/sample from per-molecule T2D params.

    Physics (per Oliver 2023 / ``support/T2D.pdf``):

        L(t) = mult_const * t_from_tail_ms ^ alpha + addit_const
        dL/dt_ms = mult_const * alpha * t_from_tail_ms ^ (alpha - 1)
        v_T2D_per_sample = dL/dt_ms * sample_period_ms

    ``addit_const`` vanishes in the derivative, so we don't need it here.

    Args:
        mult_const: [B, 1] per-molecule multiplicative constant.
        alpha: [B, 1] per-molecule exponent (typically ~0.55).
        tail_ms: [B, 1] per-molecule trailing-edge time in cached-waveform ms
            (= ``start_within_tdb_ms + fall_t50``).
        T: sequence length (number of samples per molecule).
        sample_rate_hz: Sample rate. Default 32000 (TDB standard).
        min_t_from_tail_ms: Floor on ``t_from_tail_ms`` to avoid the power-law
            singularity at the trailing edge. Default 2 ms.

    Returns:
        [B, T] per-sample v_T2D in bp/sample. Guaranteed finite and positive.
    """
    sample_period_ms = 1000.0 / sample_rate_hz
    sample_ms = torch.arange(T, device=mult_const.device, dtype=mult_const.dtype)
    sample_ms = sample_ms * sample_period_ms  # [T]
    # [B, T]
    t_from_tail_ms = (tail_ms - sample_ms.unsqueeze(0)).clamp(min=min_t_from_tail_ms)
    # v in bp/sample. Broadcasting: mult/alpha are [B,1], t_from_tail is [B,T].
    v_t2d = mult_const * alpha * t_from_tail_ms.pow(alpha - 1) * sample_period_ms
    return v_t2d


class T2DUNet(nn.Module):
    """1D U-Net with FiLM conditioning, dilated bottleneck, and multi-head self-attention.

    Two velocity-interpretation modes on ``forward``:

    * **Default** (``t2d_params=None``): velocity head output → softplus →
      positive raw velocity. Used by V1 and the L_511 spike.
    * **T2D-hybrid** (``t2d_params`` provided): velocity head output → tanh
      → residual in ``[-0.5, +0.5]``, composed as
      ``v_final = v_T2D(t; per-mol params) * (1 + residual)``. Collapses to
      pure T2D when residual → 0.
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
        # Initialize probe_head's final-conv bias so the model outputs
        # sigmoid(~-3) ~= 0.05 everywhere at init. This matches the
        # approximate peak-sample fraction on a sparse Gaussian target and
        # prevents the "1792 negatives pulling down from sigmoid=0.5" trap
        # that dominates the first few hundred steps otherwise (standard
        # sparse-detection init trick, e.g. RetinaNet focal-loss paper).
        final_conv = self.probe_head[-1]
        assert isinstance(final_conv, nn.Conv1d)
        nn.init.constant_(final_conv.bias, -3.0)

        # --- Velocity head (wide kernels for smoothing) ---
        self.velocity_head = nn.Sequential(
            ResBlock(final_ch, 31),
            ResBlock(final_ch, 31),
            nn.Conv1d(final_ch, 1, 1),
        )
        # (Velocity activation is applied inline in forward() — see the
        # t2d_params branch for hybrid vs softplus handling.)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        mask: torch.Tensor,
        t2d_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Args:
            x: Waveform tensor [B, 1, T].
            conditioning: Physical observables [B, 6].
            mask: Bool tensor [B, T] (True = valid).
            t2d_params: Optional [B, 3] tensor of per-molecule T2D params:
                columns ``[mult_const, alpha, tail_ms]``. When provided,
                the velocity head output is reinterpreted as a tanh-bounded
                residual modulating a physics-informed v_T2D baseline
                (T2D-hybrid mode). When ``None``, standard softplus-velocity
                behavior (backward compatible with V1 / L_511 spike).

        Returns:
            ``(probe_heatmap, cumulative_bp, raw_velocity, probe_logits)``.
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
        probe_logits = self.probe_head(h).squeeze(1)  # [B, T_padded], raw logits
        probe = torch.sigmoid(probe_logits)  # [B, T_padded], probabilities

        # --- Velocity head ---
        vel_logits = self.velocity_head(h).squeeze(1)  # [B, T_padded], raw logits

        if t2d_params is None:
            # Standard path (V1 / L_511 spike): softplus → positive velocity.
            raw_velocity = F.softplus(vel_logits)
        else:
            # T2D-hybrid path: logits → tanh-bounded residual, modulating v_T2D.
            # t2d_params: [B, 3] with columns [mult_const, alpha, tail_ms].
            mult_const = t2d_params[:, 0:1].to(dtype=vel_logits.dtype)  # [B, 1]
            alpha = t2d_params[:, 1:2].to(dtype=vel_logits.dtype)
            tail_ms = t2d_params[:, 2:3].to(dtype=vel_logits.dtype)

            v_t2d = compute_v_t2d(
                mult_const, alpha, tail_ms, T=vel_logits.shape[-1]
            )  # [B, T_padded]
            residual = T2D_RESIDUAL_BOUND * torch.tanh(vel_logits)  # [-0.5, +0.5]
            raw_velocity = v_t2d * (1.0 + residual)  # strictly positive

        # Mask velocity and compute cumulative bp
        masked_velocity = raw_velocity * mask.float()  # zero in padded regions
        cumulative_bp = torch.cumsum(masked_velocity, dim=-1)  # [B, T_padded]

        # --- Trim to original length ---
        probe = probe[:, :T_orig]
        probe_logits = probe_logits[:, :T_orig]
        cumulative_bp = cumulative_bp[:, :T_orig]
        raw_velocity = raw_velocity[:, :T_orig]

        return probe, cumulative_bp, raw_velocity, probe_logits
