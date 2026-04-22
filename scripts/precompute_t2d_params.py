"""Precompute per-molecule T2D parameters and write them into a cache directory.

Option A (T2D-hybrid) training needs per-molecule ``(mult_const, alpha, tail_ms)``
triples at data-loading time so the model can compute ``v_T2D`` on-the-fly.

This script enriches an existing cache directory with a ``t2d_params.npy`` file
shape ``(n_molecules, 3)`` dtype float32 with columns
``[mult_const, alpha, tail_ms]``. Pairs are looked up by molecule channel
against a ``_transForm.txt`` (per-channel T2D constants) and by molecule uid
against the original ``probes.bin`` (for ``start_within_tdb_ms`` and
``fall_t50``, which determine ``tail_ms``).

Usage:
    python scripts/precompute_t2d_params.py \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --transform-file ".../STB03-060A..._transForm.txt"

If ``--probes-bin`` is omitted, the script auto-locates the probes.bin in the
same directory as the transform file (the Nabsys tooling convention).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.transform import load_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--transform-file", type=Path, required=True)
    parser.add_argument(
        "--probes-bin",
        type=Path,
        default=None,
        help="Override probes.bin path (default: auto-detect next to transform file).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="t2d_params.npy",
        help="Filename to write inside --cache-dir.",
    )
    args = parser.parse_args()

    transforms = load_transforms(args.transform_file)

    pbin = args.probes_bin
    if pbin is None:
        pbin = next(args.transform_file.parent.glob("*_probes.bin"), None)
        if pbin is None:
            raise SystemExit(
                f"error: no *_probes.bin found in {args.transform_file.parent}; "
                "pass --probes-bin explicitly."
            )
    pf = load_probes_bin(pbin)
    mols_by_uid = {int(m.uid): m for m in pf.molecules}

    with open(args.cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    molecules = manifest["molecules"]

    # Output shape: (n_molecules, 3) with columns [mult_const, alpha, tail_ms].
    # NaN rows flag molecules we couldn't resolve — trainer must handle these.
    params = np.full((len(molecules), 3), np.nan, dtype=np.float32)

    n_resolved = 0
    n_missing_channel = 0
    n_missing_mol = 0

    for i, mol_info in enumerate(molecules):
        channel_key = f"Ch{int(mol_info['channel']):03d}"
        tf = transforms.get(channel_key)
        if tf is None:
            n_missing_channel += 1
            continue
        mol = mols_by_uid.get(int(mol_info["uid"]))
        if mol is None:
            n_missing_mol += 1
            continue
        tail_ms = float(mol.start_within_tdb_ms) + float(mol.fall_t50)
        params[i, 0] = tf.mult_const
        params[i, 1] = tf.alpha
        params[i, 2] = tail_ms
        n_resolved += 1

    out_path = args.cache_dir / args.output_name
    np.save(out_path, params)

    print(f"Wrote {out_path}")
    print(f"  resolved: {n_resolved}/{len(molecules)} molecules")
    if n_missing_channel > 0:
        print(f"  missing channel transform: {n_missing_channel}")
    if n_missing_mol > 0:
        print(f"  missing probes.bin entry: {n_missing_mol}")


if __name__ == "__main__":
    main()
