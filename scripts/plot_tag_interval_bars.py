"""Inter-tag interval bar chart: Legacy vs V3 vs Reference per holdout.

For each holdout, picks one representative molecule (highest alignment
score with 5-8 matched tags) and draws a grouped bar chart of the
N-1 inter-tag intervals. y-axis = interval length in bp; groups of
three bars per interval index.

Companion to plot_tag_intervals_gallery.py. Shows the *numerical*
comparison rather than the visual offset comparison.
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


def _load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = T2DUNet(
        config.in_channels, config.conditioning_dim,
        probe_aware_velocity=bool(getattr(config, "probe_aware_velocity", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, bool(getattr(config, "use_t2d_hybrid", False))


def _pick_best_molecule(
    cache_dir: Path, assigns_path: Path, min_tags: int, max_tags: int,
) -> tuple[int, int] | None:
    with open(cache_dir / "manifest.json") as f:
        mol_entries = json.load(f)["molecules"]
    asgns = {int(a.fragment_uid): a for a in load_assigns(assigns_path)}
    best = None
    for idx, me in enumerate(mol_entries):
        uid = int(me["uid"])
        a = asgns.get(uid)
        if a is None:
            continue
        n_matched = sum(1 for v in a.probe_indices if v > 0)
        if n_matched < min_tags or n_matched > max_tags:
            continue
        if best is None or a.alignment_score > best[0]:
            best = (a.alignment_score, idx, uid)
    return (best[1], best[2]) if best is not None else None


def _predict_and_intervals(
    model, device, use_hybrid, dataset, idx, mol, transform,
) -> dict:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_molecules,
    )
    target = None
    for i, batch in enumerate(loader):
        if i == idx:
            target = batch
            break
    wf = target["waveform"].to(device)
    cond = target["conditioning"].to(device)
    mask = target["mask"].to(device)
    t2d_params = target["t2d_params"].to(device) if use_hybrid else None
    with torch.no_grad():
        _probe, cum_bp, _vel, _logits = model(
            wf, cond, mask, t2d_params=t2d_params,
        )
        cum_bp = cum_bp.float().cpu().numpy()[0]
    centers = target["warmstart_probe_centers_samples"][0].numpy().astype(np.int64)
    ref_bp = target["reference_bp_positions"][0].numpy().astype(np.int64)
    t2d_bp = legacy_t2d_bp_positions(
        centers, mol=mol, mult_const=transform.mult_const,
        addit_const=transform.addit_const, alpha=transform.alpha,
    )
    v3_bp = cum_bp[centers]
    ref_iv = np.abs(np.diff(ref_bp.astype(np.float64)))
    t2d_iv = np.abs(np.diff(t2d_bp))
    v3_iv = np.abs(np.diff(v3_bp))
    return {
        "ref_iv": ref_iv, "t2d_iv": t2d_iv, "v3_iv": v3_iv,
        "uid": int(mol.uid),
    }


def _draw_grouped_bars(ax, data: dict, title: str):
    ref = data["ref_iv"]
    t2d = data["t2d_iv"]
    v3 = data["v3_iv"]
    n = len(ref)
    x = np.arange(n)
    w = 0.27
    ax.bar(x - w, t2d / 1000.0, width=w, color="#f4c20d",
           edgecolor="#c9a107", label="Legacy T2D")
    ax.bar(x, v3 / 1000.0, width=w, color="#4285f4",
           edgecolor="#2556b8", label="V3 (Our Model)")
    ax.bar(x + w, ref / 1000.0, width=w, color="#db4437",
           edgecolor="#a22919", label="Reference")
    ax.set_xticks(x)
    ax.set_xticklabels([f"i{k+1}" for k in range(n)], fontsize=8)
    ax.set_ylabel("interval length (kbp)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    # Error annotations above each group: rel err for V3 vs ref and T2D vs ref.
    for k in range(n):
        if ref[k] <= 0:
            continue
        v3_rel = abs(v3[k] - ref[k]) / ref[k]
        t2d_rel = abs(t2d[k] - ref[k]) / ref[k]
        ax.annotate(
            f"T:{t2d_rel:.2f}\nV:{v3_rel:.2f}",
            xy=(x[k], max(t2d[k], v3[k], ref[k]) / 1000.0),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=7, color="#333",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, action="append", required=True,
    )
    parser.add_argument(
        "--transform-file", type=Path, action="append", required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tags", type=int, default=5)
    parser.add_argument("--max-tags", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_hybrid = _load_model(args.checkpoint, device)
    holdout_labels = ["Black (STB03-063B)", "Red (STB03-064D)", "Blue (STB03-065H)"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for row, (cache_dir, transform_file, label) in enumerate(
        zip(args.cache_dir, args.transform_file, holdout_labels)
    ):
        transforms = load_transforms(transform_file)
        pbin = load_probes_bin(next(transform_file.parent.glob("*_probes.bin")))
        mol_by_uid = {int(m.uid): m for m in pbin.molecules}
        asgn_path = None
        for p in transform_file.parent.glob("*probeassignment.assigns"):
            if ".subset." not in p.name and ".tvcsubset." not in p.name:
                asgn_path = p
                break
        pick = _pick_best_molecule(cache_dir, asgn_path, args.min_tags, args.max_tags)
        if pick is None:
            axes[row].text(0.5, 0.5, f"{label}: no eligible molecule",
                           ha="center", va="center")
            continue
        idx, uid = pick
        mol = mol_by_uid[uid]
        ct = transforms[f"Ch{mol.channel:03d}"]
        dataset = CachedMoleculeDataset([cache_dir], augment=False)
        data = _predict_and_intervals(
            model, device, use_hybrid, dataset, idx, mol, ct,
        )
        _draw_grouped_bars(
            axes[row], data,
            f"{label} / uid {uid}   (labels show relative error T:v3)",
        )
        if row == 0:
            axes[row].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("interval index along molecule")
    fig.suptitle(
        "Inter-tag intervals: Legacy T2D vs V3 vs Reference (single molecule per holdout)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
