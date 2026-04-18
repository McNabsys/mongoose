"""Sanity gate: can the model memorize a single batch?

Loads one fixed batch, disables augmentation and warmstart, trains on
that same batch for N steps, and prints raw + scaled per-component
losses. Also saves one visualization PNG for the first molecule.

Pass criteria (spec sec. 5, Phase 1.5):
  - scaled bp < 0.1 by step 300
  - probe < 0.1 by step 300
  - no NaN / Inf at any step
  - visualization shows sharp peaks aligned with reference
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.losses.combined import CombinedLoss
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-bp", type=float, default=30000.0)
    parser.add_argument("--scale-vel", type=float, default=5000.0)
    parser.add_argument("--scale-count", type=float, default=1.0)
    parser.add_argument("--scale-probe", type=float, default=1.0)
    parser.add_argument("--min-blend", type=float, default=0.1)
    parser.add_argument("--output-viz", type=Path, default=Path("overfit_gate_viz.png"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CachedMoleculeDataset([args.cache_dir], augment=False)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.batch_size, replace=False).tolist()
    items = [dataset[i] for i in indices]
    batch = collate_molecules(items)

    waveform = batch["waveform"].to(device)
    conditioning = batch["conditioning"].to(device)
    mask = batch["mask"].to(device)
    reference_bp_positions_list = [
        bp.to(device) for bp in batch["reference_bp_positions"]
    ]
    n_ref_probes = batch["n_ref_probes"].to(device)
    warmstart_heatmap = batch.get("warmstart_heatmap")
    if warmstart_heatmap is not None:
        warmstart_heatmap = warmstart_heatmap.to(device)
    warmstart_valid = batch.get("warmstart_valid")
    if warmstart_valid is not None:
        warmstart_valid = warmstart_valid.to(device)

    in_channels = waveform.shape[1]
    cond_dim = conditioning.shape[1]
    model = T2DUNet(in_channels, cond_dim).to(device)
    model.train()

    criterion = CombinedLoss(
        warmstart_epochs=0,  # No warmstart schedule; go straight to real target.
        warmstart_fade_epochs=0,
        min_blend=args.min_blend,
        scale_probe=args.scale_probe,
        scale_bp=args.scale_bp,
        scale_vel=args.scale_vel,
        scale_count=args.scale_count,
    )
    criterion.set_epoch(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            probe_heatmap, cumulative_bp, raw_velocity = model(
                waveform, conditioning, mask
            )
        with torch.amp.autocast("cuda", enabled=False):
            loss, details = criterion(
                pred_heatmap=probe_heatmap.float(),
                pred_cumulative_bp=cumulative_bp.float(),
                raw_velocity=raw_velocity.float(),
                reference_bp_positions_list=reference_bp_positions_list,
                n_ref_probes=n_ref_probes,
                warmstart_heatmap=warmstart_heatmap,
                warmstart_valid=warmstart_valid,
                mask=mask,
            )

        if not math.isfinite(loss.item()):
            print(f"NaN/Inf at step {step}: details={details}")
            raise SystemExit(2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(
                f"step {step:4d} | total={loss.item():10.5f} | "
                f"probe={details['probe']:.4f} bp={details['bp']:.4f} "
                f"vel={details['vel']:.4f} count={details['count']:.4f} | "
                f"raw_bp={details['bp_raw']:.1f} raw_vel={details['vel_raw']:.1f}"
            )

    # Final visualization for the first molecule in the batch.
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            heatmap, _, velocity = model(waveform, conditioning, mask)
        h0 = heatmap[0].float().cpu().numpy()
        v0 = velocity[0].float()
        pred_idx = extract_peak_indices(
            heatmap[0].float(), v0, threshold=0.3
        ).cpu().numpy()

    # Reference peaks for the first molecule via the new per-item field
    # added in Task 5's dataset fix (LongTensor | None).
    first_item = items[0]
    ref_centers_tensor = first_item.get("warmstart_probe_centers_samples")
    if ref_centers_tensor is None:
        ref_centers = np.array([], dtype=np.int64)
    else:
        ref_centers = ref_centers_tensor.detach().cpu().numpy().astype(np.int64)

    wf0 = waveform[0, 0].float().cpu().numpy()
    t = np.arange(wf0.shape[0])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, wf0, color="gray", linewidth=0.5, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(t, h0[:wf0.shape[0]], color="black", linewidth=0.8)
    ax2.set_ylim(0, 1.05)
    for p in pred_idx:
        ax.axvline(p, color="red", linewidth=0.8, alpha=0.7)
    for r in ref_centers:
        ax.axvline(r, color="blue", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_title(
        f"Overfit gate @ step {args.steps}: pred={len(pred_idx)} (red), ref={len(ref_centers)} (blue)"
    )

    args.output_viz.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.output_viz, dpi=100)
    plt.close(fig)
    print(f"\nSaved viz: {args.output_viz}")


if __name__ == "__main__":
    main()
