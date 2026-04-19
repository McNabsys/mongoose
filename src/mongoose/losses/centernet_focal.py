"""CenterNet-style focal loss for sparse 1-D peak detection.

Per ``Zhou et al. 2019 (Objects as Points)``: the standard focal-loss penalty
for positive samples (target = 1 at exact peak center) is combined with a
penalty-reduction factor ``(1 - target)**beta`` for negative samples that fall
inside the Gaussian halo of a true peak.

The normalization divides the per-sample sum by the number of positive samples
in the molecule (not the sequence length), giving gradient strength that is
independent of padded/variable waveform length.

BF16 caveat: ``target.eq(1.0)`` is avoided because a ``wfmproc`` heatmap stored
as float32 and downcast inside ``torch.amp.autocast('cuda', dtype=bfloat16)``
may arrive at this loss slightly below exact 1.0 due to mantissa truncation.
``.ge(0.99)`` is used instead to identify peak-center samples.
"""
from __future__ import annotations

import torch


def centernet_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    pos_threshold: float = 0.99,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute CenterNet focal loss for a single molecule.

    Args:
        logits: ``[T]`` raw logits from the probe head (pre-sigmoid).
        target: ``[T]`` Gaussian heatmap target in ``[0, 1]``; values ``>= pos_threshold``
            are treated as peak-center samples.
        mask: ``[T]`` boolean mask, ``True`` at valid (non-padded) samples.
        alpha: Exponent on the prediction-confidence modulating factor.
        beta: Exponent on the Gaussian-halo penalty-reduction factor.
        pos_threshold: Minimum target value to count a sample as a positive
            peak-center (defaults to 0.99 to survive BF16 mantissa truncation).
        eps: Clamp applied to ``sigmoid(logits)`` before ``log`` for numerical
            stability.

    Returns:
        Scalar loss: sum over time divided by the number of positive samples
        (``clamp(min=1.0)`` to avoid division by zero when a molecule has no
        peaks inside the mask).
    """
    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    mask_f = mask.to(pred.dtype)
    target_f = target.to(pred.dtype)

    pos_inds = (target_f >= pos_threshold).to(pred.dtype) * mask_f
    neg_inds = (target_f < pos_threshold).to(pred.dtype) * mask_f

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_weights = torch.pow(1.0 - target_f, beta)
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum().clamp(min=1.0)
    return -(pos_loss + neg_loss).sum() / num_pos
