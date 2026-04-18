"""Soft-DTW loss (Cuturi & Blondel 2017).

Differentiable dynamic time warping for aligning two 1D sequences.
Used in V1 rearchitecture to match model-detected peak positions (in cumulative bp)
against E. coli reference probe positions, handling FN/FP detections natively.

Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series",
ICML 2017.

Implementation: forward + backward run on CPU as NumPy DPs wrapped in a custom
``torch.autograd.Function``. For the sequence lengths involved (tens of
peaks), CPU NumPy is orders of magnitude faster than a Python-level torch
DP that launches many tiny CUDA kernels per cell.
"""
from __future__ import annotations

import numpy as np
import torch


def _softmin3(a: float, b: float, c: float, gamma: float) -> tuple[float, float, float, float]:
    """Soft-min of three scalars with softmax weights returned.

    Returns (value, w_a, w_b, w_c) where value = -gamma*log(sum(exp(-x/gamma)))
    and (w_a, w_b, w_c) are the softmax weights (sum to 1) that contribute
    the gradient w.r.t. each input (d value / d a = w_a, etc.).
    """
    inv_g = 1.0 / gamma
    ra = -a * inv_g
    rb = -b * inv_g
    rc = -c * inv_g
    rmax = ra if ra >= rb else rb
    if rc > rmax:
        rmax = rc
    # If rmax == -inf then all are -inf (all inputs were +inf); return +inf.
    if not np.isfinite(rmax):
        return float("inf"), 0.0, 0.0, 0.0
    ea = np.exp(ra - rmax)
    eb = np.exp(rb - rmax)
    ec = np.exp(rc - rmax)
    s = ea + eb + ec
    # value = -gamma * (rmax + log(s))
    value = -gamma * (rmax + float(np.log(s)))
    inv_s = 1.0 / s
    return value, float(ea * inv_s), float(eb * inv_s), float(ec * inv_s)


class _SoftDTW(torch.autograd.Function):
    """Soft-DTW forward + backward as a single CPU/NumPy custom op."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        if x.dim() != 1 or y.dim() != 1:
            raise ValueError(
                f"soft_dtw expects 1D tensors; got shapes {tuple(x.shape)} and {tuple(y.shape)}"
            )
        n = int(x.shape[0])
        m = int(y.shape[0])
        if n == 0 or m == 0:
            raise ValueError("soft_dtw does not accept empty sequences")

        x_np = x.detach().to("cpu", dtype=torch.float32).numpy()
        y_np = y.detach().to("cpu", dtype=torch.float32).numpy()

        cost = (x_np[:, None] - y_np[None, :]) ** 2  # [N, M]

        # DP matrix with +inf borders; R[0,0] = 0.
        R = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        R[0, 0] = 0.0
        # Softmax weights per cell (N x M), indexed by (i-1, j-1).
        w_diag = np.zeros((n, m), dtype=np.float64)
        w_up = np.zeros((n, m), dtype=np.float64)
        w_left = np.zeros((n, m), dtype=np.float64)

        g = float(gamma)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sm, wd, wu, wl = _softmin3(
                    R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], g
                )
                R[i, j] = cost[i - 1, j - 1] + sm
                w_diag[i - 1, j - 1] = wd
                w_up[i - 1, j - 1] = wu
                w_left[i - 1, j - 1] = wl

        ctx.save_for_backward(x, y)
        ctx.intermediates = (cost, w_diag, w_up, w_left)
        ctx.n = n
        ctx.m = m

        out_dtype = x.dtype if x.is_floating_point() else torch.float32
        return torch.tensor(R[n, m], dtype=out_dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, y = ctx.saved_tensors
        cost, w_diag, w_up, w_left = ctx.intermediates
        n: int = ctx.n
        m: int = ctx.m

        # Back-recursion computes E[i, j] = d R[N, M] / d R[i, j].
        # With padded shape [N+2, M+2] so boundary weights can be zero
        # without special-casing.
        w_diag_p = np.zeros((n + 2, m + 2), dtype=np.float64)
        w_up_p = np.zeros((n + 2, m + 2), dtype=np.float64)
        w_left_p = np.zeros((n + 2, m + 2), dtype=np.float64)
        w_diag_p[1:n + 1, 1:m + 1] = w_diag
        w_up_p[1:n + 1, 1:m + 1] = w_up
        w_left_p[1:n + 1, 1:m + 1] = w_left

        E = np.zeros((n + 2, m + 2), dtype=np.float64)
        E[n, m] = 1.0
        for i in range(n, 0, -1):
            for j in range(m, 0, -1):
                if i == n and j == m:
                    continue
                # d R[N, M] / d R[i, j] = sum over successors of (weight_to_succ * E[succ]).
                # R[i+1, j] uses R[i, j] as its "up" term (weight w_up at cell (i+1, j)).
                # R[i+1, j+1] uses R[i, j] as its "diag" term.
                # R[i, j+1] uses R[i, j] as its "left" term.
                E[i, j] = (
                    w_up_p[i + 1, j] * E[i + 1, j]
                    + w_diag_p[i + 1, j + 1] * E[i + 1, j + 1]
                    + w_left_p[i, j + 1] * E[i, j + 1]
                )

        # d R[N, M] / d cost[i-1, j-1] = E[i, j]   (each R[i, j] adds cost[i-1, j-1]).
        # cost[i-1, j-1] = (x[i-1] - y[j-1])**2
        # d cost / d x[i-1] = 2*(x[i-1] - y[j-1]); summed over j.
        # d cost / d y[j-1] = -2*(x[i-1] - y[j-1]); summed over i.
        x_np = x.detach().to("cpu", dtype=torch.float32).numpy()
        y_np = y.detach().to("cpu", dtype=torch.float32).numpy()
        diff = x_np[:, None] - y_np[None, :]  # [N, M]
        E_inner = E[1:n + 1, 1:m + 1]
        grad_x = (2.0 * diff * E_inner).sum(axis=1)        # [N]
        grad_y = (-2.0 * diff * E_inner).sum(axis=0)       # [M]

        scale = float(grad_output.detach().to("cpu").item())
        grad_x_t = torch.from_numpy(grad_x.astype(np.float32) * scale).to(
            device=x.device, dtype=x.dtype if x.is_floating_point() else torch.float32
        )
        grad_y_t = torch.from_numpy(grad_y.astype(np.float32) * scale).to(
            device=y.device, dtype=y.dtype if y.is_floating_point() else torch.float32
        )
        # y may not require grad; autograd handles None vs. tensor either way.
        return grad_x_t, grad_y_t, None


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Soft-DTW distance between two 1D sequences.

    Args:
        x: [N] tensor of first sequence values.
        y: [M] tensor of second sequence values.
        gamma: smoothing parameter. Smaller gamma -> closer to hard DTW (less smooth gradient).
               Typical range: 0.01 - 1.0.

    Returns:
        Scalar tensor: the soft-DTW distance. Non-negative when x and y are numerical.

    Raises:
        ValueError: if either sequence is empty.
    """
    return _SoftDTW.apply(x, y, float(gamma))
