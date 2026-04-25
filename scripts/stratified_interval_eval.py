"""Stratified bp-interval eval: V3 vs T2D by attribute bit + velocity.

Augments the standard compare_bp_intervals harness with the Phase-0
stratifications we need to interpret holdout wins/losses properly:

  1. By attribute bit: attr_in_structure, attr_folded_start
     (an interval is "in_structure" if EITHER bounding probe has the bit).
  2. By molecule-mean velocity decile (bp / ms).
  3. By position-along-molecule of the interval's starting probe.

Per spec conversations 2026-04-24 following Phase 0a/0b findings: the
unstratified holdout median hides most of the relevant detail. A model
that improves only on clean-region, middle-velocity probes looks the
same in the headline as a model that improves specifically on the
10x-worse structured/folded population.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.inference.legacy_t2d import legacy_t2d_bp_positions
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.transform import load_transforms
from mongoose.model.unet import T2DUNet


# -- Per-interval record columns ------------------------------------------
#
# We keep one row per inter-probe interval with enough context to stratify
# by any axis Phase 0 pointed at. Intervals are between consecutive
# warmstart_probe_centers (as in compare_bp_intervals.py) so this mirrors
# the baseline's measurement exactly.


def _eval_v3_and_t2d_per_interval(
    *,
    checkpoint_path: Path,
    cache_dir: Path,
    transform_file: Path,
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = T2DUNet(
        config.in_channels,
        config.conditioning_dim,
        probe_aware_velocity=bool(getattr(config, "probe_aware_velocity", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_hybrid = bool(getattr(config, "use_t2d_hybrid", False))

    transforms = load_transforms(transform_file)
    pbin_path = next(transform_file.parent.glob("*_probes.bin"), None)
    if pbin_path is None:
        raise SystemExit(
            f"error: no *_probes.bin in {transform_file.parent}"
        )
    pf = load_probes_bin(pbin_path)
    mol_by_uid = {int(m.uid): m for m in pf.molecules}

    # Warmstart probes are the MATCHED subset of accepted probes (see Phase
    # 0 ETL join rule). To attach attribute bits correctly to each interval,
    # we need assigns.probe_indices to pick out which accepted probes are
    # matched, in detection order.
    assigns_path = next(
        transform_file.parent.glob("*_probeassignment.assigns"), None
    )
    # The glob may pick up the .subset/.tvcsubset variants -- prefer the
    # canonical one.
    for p in transform_file.parent.glob("*probeassignment.assigns"):
        if ".subset." not in p.name and ".tvcsubset." not in p.name:
            assigns_path = p
            break
    if assigns_path is None:
        raise SystemExit(
            f"error: no canonical probeassignment.assigns in {transform_file.parent}"
        )
    assigns_by_uid: dict[int, tuple[int, ...]] = {
        int(a.fragment_uid): a.probe_indices for a in load_assigns(assigns_path)
    }

    dataset = CachedMoleculeDataset([cache_dir], augment=False)
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    # The manifest order matches dataset order; use molecule_id entries for
    # per-molecule metadata (uid, channel, num_probes, transloc_time_ms).
    mol_entries = manifest["molecules"]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_molecules,
    )

    records: list[dict[str, Any]] = []
    idx_global = 0
    for batch in loader:
        waveform = batch["waveform"].to(device)
        conditioning = batch["conditioning"].to(device)
        mask = batch["mask"].to(device)
        t2d_params = None
        if use_hybrid:
            t2d_params = batch.get("t2d_params")
            if t2d_params is None:
                raise RuntimeError(
                    f"hybrid checkpoint needs t2d_params in {cache_dir}"
                )
            t2d_params = t2d_params.to(device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda",
        ):
            _probe, cum_bp, _vel, _logits = model(
                waveform, conditioning, mask, t2d_params=t2d_params,
            )
            cum_bp = cum_bp.float()

        b = waveform.shape[0]
        for i in range(b):
            mol_entry = mol_entries[idx_global]
            idx_global += 1
            uid = int(mol_entry["uid"])
            mol = mol_by_uid.get(uid)
            if mol is None:
                continue
            centers = batch["warmstart_probe_centers_samples"][i]
            ref_bp = batch["reference_bp_positions"][i]
            if centers is None:
                continue
            c = centers.detach().cpu().numpy().astype(np.int64)
            r = ref_bp.detach().cpu().numpy().astype(np.int64)
            if c.size < 2 or r.size < 2 or c.size != r.size:
                continue
            T = cum_bp.shape[-1]
            if c.max() >= T or c.min() < 0:
                continue

            # V3 interval prediction
            cum_at = cum_bp[i, c].detach().cpu().numpy().astype(np.float64)
            v3_iv = np.abs(np.diff(cum_at))

            # T2D interval prediction
            channel_key = f"Ch{mol.channel:03d}"
            ct = transforms.get(channel_key)
            if ct is None:
                continue
            t2d_bp = legacy_t2d_bp_positions(
                c, mol=mol,
                mult_const=ct.mult_const, addit_const=ct.addit_const, alpha=ct.alpha,
            )
            t2d_iv = np.abs(np.diff(t2d_bp))

            # Reference intervals
            ref_iv = np.abs(np.diff(r)).astype(np.float64)

            # Molecule-level velocity (bp/ms). bp span over transloc_time.
            span = float(r.max() - r.min())
            transloc_ms = float(mol.transloc_time_ms)
            mol_vel = span / transloc_ms if transloc_ms > 0 else float("nan")

            # Per-probe attr bits from probes.bin for the MATCHED probes
            # only (warmstart centers <-> matched probes under the ETL join
            # rule). Steps:
            #   1. accepted = detection-order list of accepted probes
            #   2. probe_indices = assigns row, length == len(accepted)
            #   3. matched_idxs = indices k where probe_indices[k] > 0
            #   4. k-th warmstart <-> accepted[matched_idxs[k]]
            accepted = [p for p in mol.probes if (p.attribute >> 7) & 1]
            probe_indices = assigns_by_uid.get(uid, ())
            if len(probe_indices) > len(accepted):
                # Unexpected per ETL invariant; skip this molecule.
                continue
            matched_idxs = [
                j for j, v in enumerate(probe_indices) if v > 0
            ]
            if len(matched_idxs) != len(c):
                # Warmstart length disagrees with matched count; skip.
                continue
            n_interval = v3_iv.size
            for k in range(n_interval):
                if ref_iv[k] <= 0:
                    continue
                left_bits = accepted[matched_idxs[k]].attribute
                right_bits = accepted[matched_idxs[k + 1]].attribute
                any_in_structure = bool(
                    ((left_bits >> 3) & 1) or ((right_bits >> 3) & 1)
                )
                any_folded_start = bool(
                    ((left_bits >> 2) & 1) or ((right_bits >> 2) & 1)
                )
                # Position along molecule: start probe's center/transloc.
                center_ms_left = accepted[matched_idxs[k]].center_ms
                pos_frac = (
                    (mol.start_within_tdb_ms + center_ms_left) / transloc_ms
                    if transloc_ms > 0 else float("nan")
                )
                records.append({
                    "run_id": cache_dir.name,
                    "mol_uid": uid,
                    "k": k,
                    "ref_iv_bp": float(ref_iv[k]),
                    "v3_iv_bp": float(v3_iv[k]),
                    "t2d_iv_bp": float(t2d_iv[k]),
                    "v3_rel_err": float(abs(v3_iv[k] - ref_iv[k]) / ref_iv[k]),
                    "t2d_rel_err": float(abs(t2d_iv[k] - ref_iv[k]) / ref_iv[k]),
                    "mol_vel_bp_per_ms": mol_vel,
                    "pos_frac": pos_frac,
                    "any_in_structure": any_in_structure,
                    "any_folded_start": any_folded_start,
                })
    return pd.DataFrame(records)


def _stratify(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows: list[dict] = []
    for value, sub in df.groupby(key, dropna=False, observed=True):
        rows.append({
            "stratum": key,
            "value": str(value),
            "n": int(len(sub)),
            "v3_median_rel": float(sub["v3_rel_err"].median()),
            "t2d_median_rel": float(sub["t2d_rel_err"].median()),
            "v3_p90_rel": float(sub["v3_rel_err"].quantile(0.9)),
            "t2d_p90_rel": float(sub["t2d_rel_err"].quantile(0.9)),
            "delta_median": float(
                sub["v3_rel_err"].median() - sub["t2d_rel_err"].median()
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--transform-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.cache_dir) != len(args.transform_file):
        raise SystemExit("--cache-dir and --transform-file must be same count")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_dfs: list[pd.DataFrame] = []
    for cache, transform in zip(args.cache_dir, args.transform_file):
        print(f"... {cache.name}")
        df = _eval_v3_and_t2d_per_interval(
            checkpoint_path=args.checkpoint,
            cache_dir=cache,
            transform_file=transform,
            device=device,
        )
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)

    # Velocity deciles (computed over ALL intervals globally)
    full["vel_decile"] = pd.qcut(
        full["mol_vel_bp_per_ms"], q=10,
        labels=[f"D{i+1}" for i in range(10)], duplicates="drop",
    ).astype("string")
    # Position-along-molecule bins
    full["pos_bin"] = pd.cut(
        full["pos_frac"], bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01],
        labels=["0-10%", "10-30%", "30-50%", "50-70%", "70-90%", "90-100%"],
        include_lowest=True,
    ).astype("string")

    out: dict[str, Any] = {
        "n_intervals_total": int(len(full)),
        "headline": {
            "v3_median_rel": float(full["v3_rel_err"].median()),
            "t2d_median_rel": float(full["t2d_rel_err"].median()),
            "v3_p95_rel": float(full["v3_rel_err"].quantile(0.95)),
            "t2d_p95_rel": float(full["t2d_rel_err"].quantile(0.95)),
        },
        "by_run_id": _stratify(full, "run_id").to_dict(orient="records"),
        "by_any_in_structure": _stratify(full, "any_in_structure").to_dict(orient="records"),
        "by_any_folded_start": _stratify(full, "any_folded_start").to_dict(orient="records"),
        "by_vel_decile": _stratify(full, "vel_decile").to_dict(orient="records"),
        "by_pos_bin": _stratify(full, "pos_bin").to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str))

    # Also a human-readable print
    print(f"\n=== headline ===")
    print(f"  n intervals: {out['n_intervals_total']:,}")
    print(f"  V3  median rel err: {out['headline']['v3_median_rel']:.4f}")
    print(f"  T2D median rel err: {out['headline']['t2d_median_rel']:.4f}")
    for key in ("by_run_id", "by_any_in_structure", "by_any_folded_start",
                "by_vel_decile", "by_pos_bin"):
        print(f"\n=== {key} ===")
        print(pd.DataFrame(out[key]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
