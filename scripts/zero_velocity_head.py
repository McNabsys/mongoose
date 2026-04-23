"""Zero out the velocity head weights in a checkpoint.

The T2D-hybrid training (--use-t2d-hybrid) reinterprets the velocity head's
output as a tanh-bounded residual (previously fed into softplus). If we
warm-start from a checkpoint whose velocity head was trained under the
softplus convention, its logits land in ~[1, 10] — saturating tanh and
killing the gradient there. Zeroing just the velocity head before
warm-start solves this: residual starts at 0, output starts at pure T2D,
and training learns corrections from a clean baseline.

All other weights (backbone, probe head, FiLM) carry over unchanged —
they're useful features, not affected by the velocity-interpretation
change.

Usage:
    python scripts/zero_velocity_head.py \\
        --input  l511_spike_ext_checkpoints/best_model.pt \\
        --output l511_spike_ext_checkpoints/best_model_zerovel.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # weights_only=False: our checkpoints bundle a TrainConfig dataclass.
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    n_zeroed = 0
    total_params = 0
    for key in list(state_dict.keys()):
        if "velocity_head" in key:
            state_dict[key] = torch.zeros_like(state_dict[key])
            n_zeroed += 1
            total_params += state_dict[key].numel()

    if n_zeroed == 0:
        raise SystemExit(
            "error: no velocity_head tensors found in checkpoint. "
            "Did you point at the right file?"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"Zeroed {n_zeroed} velocity_head tensors ({total_params:,} params).")
    print(f"Wrote {args.output}.")


if __name__ == "__main__":
    main()
