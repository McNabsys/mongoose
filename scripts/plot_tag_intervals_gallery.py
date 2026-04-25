"""Gallery: 3-row sparkline per molecule, 6 molecules per holdout, 3 holdouts.

Produces a 6-row x 3-column grid of panels. Each panel shows the legacy
T2D / V3 / reference tag positions for one molecule on a shared
molecule-local bp axis.
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


ROW_COLORS = {"legacy": "#f4c20d", "v3": "#4285f4", "ref": "#db4437"}
ROW_YS = {"legacy": 2.0, "v3": 1.0, "ref": 0.0}
SPIKE_H = 0.4


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


def _pick_top_n_molecules(
    cache_dir: Path, assigns_path: Path,
    *, n: int, min_tags: int, max_tags: int,
) -> list[tuple[int, int]]:
    """Return (manifest_idx, uid) for top-N highest-align-score molecules
    with matched-tag count in [min_tags, max_tags]."""
    with open(cache_dir / "manifest.json") as f:
        mol_entries = json.load(f)["molecules"]
    asgns = {int(a.fragment_uid): a for a in load_assigns(assigns_path)}
    candidates: list[tuple[int, int, int]] = []  # (score, idx, uid)
    for idx, me in enumerate(mol_entries):
        uid = int(me["uid"])
        a = asgns.get(uid)
        if a is None:
            continue
        n_matched = sum(1 for v in a.probe_indices if v > 0)
        if n_matched < min_tags or n_matched > max_tags:
            continue
        candidates.append((a.alignment_score, idx, uid))
    candidates.sort(reverse=True)
    return [(idx, uid) for _score, idx, uid in candidates[:n]]


def _predict_one(
    model, device, use_hybrid: bool, dataset, idx: int,
    mol, transform,
) -> dict:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_molecules,
    )
    target_batch = None
    for i, batch in enumerate(loader):
        if i == idx:
            target_batch = batch
            break
    assert target_batch is not None
    wf = target_batch["waveform"].to(device)
    cond = target_batch["conditioning"].to(device)
    mask = target_batch["mask"].to(device)
    t2d_params = target_batch["t2d_params"].to(device) if use_hybrid else None
    with torch.no_grad():
        _probe, cum_bp, _vel, _logits = model(
            wf, cond, mask, t2d_params=t2d_params,
        )
        cum_bp = cum_bp.float().cpu().numpy()[0]
    centers = target_batch["warmstart_probe_centers_samples"][0].numpy().astype(np.int64)
    ref_bp = target_batch["reference_bp_positions"][0].numpy().astype(np.int64)
    t2d_bp = legacy_t2d_bp_positions(
        centers, mol=mol, mult_const=transform.mult_const,
        addit_const=transform.addit_const, alpha=transform.alpha,
    )
    v3_bp = cum_bp[centers]
    ref_norm = np.abs(ref_bp.astype(np.float64) - ref_bp[0])
    t2d_norm = np.abs(t2d_bp - t2d_bp[0])
    v3_norm = np.abs(v3_bp - v3_bp[0])
    return {"ref": ref_norm, "t2d": t2d_norm, "v3": v3_norm,
            "n": int(len(ref_norm))}


def _draw_row(ax, ys: np.ndarray, y_row: float, color: str):
    xmax = float(ys.max()) if ys.size else 1.0
    ax.hlines(y_row, 0, xmax, color=color, linewidth=1.0)
    ax.vlines(ys, y_row, y_row + SPIKE_H, color=color, linewidth=1.5)


def _draw_panel(ax, pred: dict, title: str):
    ref = pred["ref"]
    for ref_x in ref:
        ax.axvline(ref_x, color="gray", linewidth=0.35, linestyle=(0, (2, 2)), alpha=0.5)
    _draw_row(ax, pred["t2d"], ROW_YS["legacy"], ROW_COLORS["legacy"])
    _draw_row(ax, pred["v3"], ROW_YS["v3"], ROW_COLORS["v3"])
    _draw_row(ax, pred["ref"], ROW_YS["ref"], ROW_COLORS["ref"])
    xmax = max(float(pred["t2d"].max()), float(pred["v3"].max()), float(ref.max()))
    ax.set_xlim(-xmax * 0.02, xmax * 1.03)
    ax.set_ylim(-0.3, 2.7)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, action="append", required=True,
        help="Repeat per holdout. Order: Black, Red, Blue.",
    )
    parser.add_argument(
        "--transform-file", type=Path, action="append", required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-per-holdout", type=int, default=6)
    parser.add_argument("--min-tags", type=int, default=5)
    parser.add_argument("--max-tags", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_hybrid = _load_model(args.checkpoint, device)

    holdout_labels = ["Black (STB03-063B)", "Red (STB03-064D)", "Blue (STB03-065H)"]
    # Collect per-holdout panels
    all_preds: list[list[dict]] = []
    for cache_dir, transform_file in zip(args.cache_dir, args.transform_file):
        print(f"... processing {cache_dir.name}")
        transforms = load_transforms(transform_file)
        pbin = load_probes_bin(next(transform_file.parent.glob("*_probes.bin")))
        mol_by_uid = {int(m.uid): m for m in pbin.molecules}
        asgn_path = None
        for p in transform_file.parent.glob("*probeassignment.assigns"):
            if ".subset." not in p.name and ".tvcsubset." not in p.name:
                asgn_path = p
                break
        picks = _pick_top_n_molecules(
            cache_dir, asgn_path,
            n=args.n_per_holdout, min_tags=args.min_tags, max_tags=args.max_tags,
        )
        dataset = CachedMoleculeDataset([cache_dir], augment=False)
        preds_this_holdout: list[dict] = []
        for idx, uid in picks:
            mol = mol_by_uid.get(uid)
            if mol is None:
                continue
            ct = transforms.get(f"Ch{mol.channel:03d}")
            if ct is None:
                continue
            p = _predict_one(model, device, use_hybrid, dataset, idx, mol, ct)
            p["uid"] = uid
            preds_this_holdout.append(p)
        all_preds.append(preds_this_holdout)

    n_rows = args.n_per_holdout
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 2.2),
        constrained_layout=True,
    )
    for j, (label, preds) in enumerate(zip(holdout_labels, all_preds)):
        for i in range(n_rows):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            if i >= len(preds):
                ax.axis("off")
                continue
            pred = preds[i]
            _draw_panel(ax, pred, f"uid {pred['uid']}, {pred['n']} tags")
        axes[0, j].annotate(
            label, xy=(0.5, 1.25), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    fig.suptitle(
        "Tag interval comparison: Legacy T2D (yellow) / V3 (blue) / Reference (red)",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
