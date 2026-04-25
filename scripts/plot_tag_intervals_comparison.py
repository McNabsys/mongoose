"""Three-row spike plot: Legacy T2D vs V3 vs reference tag positions.

Picks a single molecule with a modest tag count (default 5-8 tags)
and high alignment score, runs V3 + T2D predictions, and draws the
three-row sparkline style used in the Nabsys deck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.inference.legacy_t2d import legacy_t2d_bp_positions
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.transform import load_transforms
from mongoose.model.unet import T2DUNet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--transform-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tags", type=int, default=5)
    parser.add_argument("--max-tags", type=int, default=8)
    parser.add_argument(
        "--molecule-uid", type=int, default=None,
        help="Specific uid to plot; otherwise auto-pick highest-align molecule "
             "in [min-tags, max-tags] range.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = T2DUNet(
        config.in_channels, config.conditioning_dim,
        probe_aware_velocity=bool(getattr(config, "probe_aware_velocity", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_hybrid = bool(getattr(config, "use_t2d_hybrid", False))

    transforms = load_transforms(args.transform_file)
    pbin_path = next(args.transform_file.parent.glob("*_probes.bin"))
    assigns_path = None
    for p in args.transform_file.parent.glob("*probeassignment.assigns"):
        if ".subset." not in p.name and ".tvcsubset." not in p.name:
            assigns_path = p
            break
    pf = load_probes_bin(pbin_path)
    mol_by_uid = {int(m.uid): m for m in pf.molecules}
    assigns_by_uid = {
        int(a.fragment_uid): a for a in load_assigns(assigns_path)
    }

    with open(args.cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    mol_entries = manifest["molecules"]

    # Find the target molecule.
    target_idx = None
    best_score = -1
    for i, me in enumerate(mol_entries):
        uid = int(me["uid"])
        if args.molecule_uid is not None and uid != args.molecule_uid:
            continue
        asgn = assigns_by_uid.get(uid)
        if asgn is None:
            continue
        n_matched = sum(1 for v in asgn.probe_indices if v > 0)
        if n_matched < args.min_tags or n_matched > args.max_tags:
            continue
        if args.molecule_uid is not None:
            target_idx = i
            break
        if asgn.alignment_score > best_score:
            best_score = asgn.alignment_score
            target_idx = i
    if target_idx is None:
        raise SystemExit(
            f"no molecule with {args.min_tags}-{args.max_tags} matched tags found"
        )
    target_uid = int(mol_entries[target_idx]["uid"])
    print(f"using molecule uid={target_uid} (manifest idx {target_idx}, "
          f"align_score={assigns_by_uid[target_uid].alignment_score})")

    # Load the batch containing the target molecule (1 per batch for simplicity).
    dataset = CachedMoleculeDataset([args.cache_dir], augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_molecules,
    )
    target_batch = None
    for i, batch in enumerate(loader):
        if i == target_idx:
            target_batch = batch
            break
    assert target_batch is not None

    waveform = target_batch["waveform"].to(device)
    conditioning = target_batch["conditioning"].to(device)
    mask = target_batch["mask"].to(device)
    t2d_params = None
    if use_hybrid:
        t2d_params = target_batch["t2d_params"].to(device)
    with torch.no_grad():
        _probe, cum_bp, _vel, _logits = model(
            waveform, conditioning, mask, t2d_params=t2d_params
        )
        cum_bp = cum_bp.float().cpu().numpy()[0]

    centers = target_batch["warmstart_probe_centers_samples"][0].numpy().astype(np.int64)
    ref_bp = target_batch["reference_bp_positions"][0].numpy().astype(np.int64)

    mol = mol_by_uid[target_uid]
    channel_key = f"Ch{mol.channel:03d}"
    ct = transforms[channel_key]
    t2d_bp = legacy_t2d_bp_positions(
        centers, mol=mol,
        mult_const=ct.mult_const, addit_const=ct.addit_const, alpha=ct.alpha,
    )
    v3_bp = cum_bp[centers]

    # Normalize to "molecule-local bp, starting at 0" so the three rows
    # share an x-axis. Each method's first tag sits at 0.
    ref_norm = ref_bp.astype(np.float64) - ref_bp[0]
    ref_norm = np.abs(ref_norm)  # handle direction=-1
    t2d_norm = np.abs(t2d_bp - t2d_bp[0])
    v3_norm = np.abs(v3_bp - v3_bp[0])

    xmax = max(float(ref_norm.max()), float(t2d_norm.max()), float(v3_norm.max()))
    xmax *= 1.05

    # Three-row spike plot.
    fig, ax = plt.subplots(figsize=(12, 5))
    row_ys = {"legacy": 2.0, "v3": 1.0, "ref": 0.0}
    row_colors = {
        "legacy": "#f4c20d",  # yellow
        "v3": "#4285f4",       # blue
        "ref": "#db4437",      # red
    }
    row_labels = {
        "legacy": "Legacy Model = Calc Tag Intervals",
        "v3": "Our Model = Learned Tag Intervals",
        "ref": "Reference = Perfect Tag Intervals",
    }
    spike_h = 0.4
    for name, xs in (("legacy", t2d_norm), ("v3", v3_norm), ("ref", ref_norm)):
        y = row_ys[name]
        color = row_colors[name]
        # Horizontal baseline.
        ax.hlines(y, 0, xmax, color=color, linewidth=1.6)
        # Vertical spikes at each tag position.
        for x in xs:
            ax.vlines(x, y, y + spike_h, color=color, linewidth=2.0)
        # Label on the right.
        ax.text(
            xmax * 1.02, y + spike_h / 2, row_labels[name],
            color=color, fontsize=11, va="center", ha="left",
        )

    # Faint dashed guides at reference positions -- anchors the visual
    # comparison just like the deck image.
    for x in ref_norm:
        ax.axvline(x, color="gray", linewidth=0.5, linestyle=(0, (2, 2)), alpha=0.5)

    ax.set_xlim(-xmax * 0.02, xmax * 1.45)
    ax.set_ylim(-0.4, 2.8)
    ax.set_xlabel("Tag Intervals in Base Pairs (molecule-local)")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        f"Tag interval comparison: uid {target_uid}  "
        f"({len(ref_norm)} matched tags, molecule span {xmax/1.05:.0f} bp)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
