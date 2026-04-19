"""Plot ONE random post-collate batch for visual data-pipeline inspection.

Usage:
    python scripts/plot_random_dataloader_batch.py \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --batch-size 8 --seed 42 \\
        --output dataloader_batch_check.png

Purpose (Deep Think's Q5): after any data-pipeline change (preprocess,
collate, dataset), run this. Eyeball the PNG. If the heatmap overlays
(green) do NOT land on the waveform peaks, the data pipeline is broken.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=Path, default=Path("dataloader_batch_check.png")
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ds = CachedMoleculeDataset([args.cache_dir], augment=False)
    indices = rng.choice(len(ds), size=args.batch_size, replace=False).tolist()
    batch = collate_molecules([ds[i] for i in indices])

    wf = batch["waveform"]            # [B, 1, T]
    mask = batch["mask"]              # [B, T]
    hm = batch.get("warmstart_heatmap")  # [B, T] or None
    centers_list = batch["warmstart_probe_centers_samples"]

    fig, axes = plt.subplots(args.batch_size, 1, figsize=(16, 2 * args.batch_size))
    if args.batch_size == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        m = mask[i].numpy()
        v_len = int(m.sum())
        t = np.arange(v_len)
        ax.plot(t, wf[i, 0, :v_len].numpy(), color="steelblue", linewidth=0.5)
        ax.set_ylabel(f"mol {indices[i]}", fontsize=8)
        if hm is not None:
            ax2 = ax.twinx()
            ax2.plot(t, hm[i, :v_len].numpy(), color="black", linewidth=0.6, alpha=0.7)
            ax2.set_ylim(0, 1.1)
            ax2.tick_params(labelsize=6)
        centers = centers_list[i]
        if centers is not None:
            for c in centers.numpy():
                if 0 <= c < v_len:
                    ax.axvline(int(c), color="seagreen", linewidth=0.8, alpha=0.75)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, v_len)
    axes[-1].set_xlabel("sample index (valid region only)")
    plt.suptitle(
        f"Random post-collate batch from {args.cache_dir.name} (seed={args.seed})\n"
        f"green = warmstart_probe_centers_samples, black = warmstart_heatmap",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
