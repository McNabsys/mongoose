"""Render predicted vs reference peaks on a grid of molecules as one PNG.

Usage:
    python scripts/visualize_predictions.py \\
        --checkpoint overnight_training/best_model.pt \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --output preds_grid.png \\
        --n-molecules 20 \\
        --seed 42
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
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-molecules", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False required because our checkpoints include a TrainConfig
    # dataclass, not just raw tensors. We trust our own checkpoint files.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = CachedMoleculeDataset([args.cache_dir], augment=False)
    n = min(args.n_molecules, len(dataset))

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n))
    if n == 1:
        axes = [axes]

    for row_idx, dataset_idx in enumerate(indices):
        item = dataset[dataset_idx]
        waveform = item["waveform"].unsqueeze(0).to(device)
        conditioning = item["conditioning"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                heatmap, _, velocity = model(waveform, conditioning, mask)
            heatmap = heatmap.float().squeeze(0)
            velocity = velocity.float().squeeze(0)

        pred_idx = extract_peak_indices(heatmap, velocity, threshold=args.threshold)
        pred_np = pred_idx.cpu().numpy().astype(np.int64)

        # Reference peaks: use the new raw-centers field added to __getitem__
        # in Task 5's dataset fix. May be None if wfmproc ground truth is missing.
        centers_tensor = item.get("warmstart_probe_centers_samples")
        if centers_tensor is None:
            ref_centers = np.array([], dtype=np.int64)
        else:
            ref_centers = centers_tensor.detach().cpu().numpy().astype(np.int64)

        wf_np = item["waveform"].squeeze(0).numpy()
        hm_np = heatmap.cpu().numpy()
        t = np.arange(wf_np.shape[0])

        ax = axes[row_idx]
        ax.plot(t, wf_np, color="gray", linewidth=0.5, alpha=0.7, label="waveform")
        ax2 = ax.twinx()
        ax2.plot(t, hm_np[:wf_np.shape[0]], color="black", linewidth=0.8,
                 alpha=0.7, label="heatmap")
        ax2.set_ylim(0, 1.05)

        # Red = predicted peaks; blue = reference peaks.
        for p in pred_np:
            ax.axvline(p, color="red", linewidth=0.8, alpha=0.6)
        for r in ref_centers:
            ax.axvline(r, color="blue", linewidth=0.6, alpha=0.5, linestyle="--")

        uid = int(item.get("molecule_uid", dataset_idx))
        ax.set_title(
            f"mol uid={uid} | pred={len(pred_np)} (red) | ref={len(ref_centers)} (blue dashed)",
            fontsize=9,
        )
        ax.set_xlabel("sample")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    plt.close(fig)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
