"""Migrate cached ``molecules.pkl`` files to the V4 warmstart-paired schema.

Pre-V4 caches store ``warmstart_probe_centers_samples`` as a strict
*subset* of ``reference_bp_positions`` -- probes whose
``duration_ms <= 0`` are dropped. This breaks the 1:1 pairing the
NoiseModelLoss interval / position NLL terms depend on. V4's
:func:`build_molecule_gt` emits paired arrays with ``-1`` sentinels for
dropped probes; this script regenerates the GT in place against the
existing caches without re-cooking ``waveforms.bin`` (which is unchanged
by the schema fix).

Usage:
    python scripts/migrate_cache_warmstart_schema.py \\
        --cache-dir <CACHE> \\
        --remap-dir <REMAP_ALLCH> \\
        --probes-bin <PROBES_BIN_PATH> \\
        --tdb-source-root <RESULTS_DIR>

For batch migration, repeat the flags for each (cache, remap, probes,
tdb-source) tuple. The script preserves the molecule order in the cache
manifest so existing ``offsets.npy`` and ``waveforms.bin`` references
stay valid.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

from mongoose.data.ground_truth import build_molecule_gt
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.reference_map import load_reference_map
from mongoose.io.tdb import load_tdb_header


LOG = logging.getLogger("migrate_warmstart_schema")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate cached molecules.pkl to V4 paired-warmstart schema."
    )
    parser.add_argument("--cache-dir", type=Path, action="append", required=True,
                        help="Cache directory containing molecules.pkl.")
    parser.add_argument("--remap-dir", type=Path, action="append", required=True,
                        help="Path to Remapped/AllCh/ for the run "
                             "(.assigns + _referenceMap.txt).")
    parser.add_argument("--probes-bin", type=Path, action="append", required=True,
                        help="Path to the run's probes.bin file.")
    parser.add_argument("--tdb-header", type=Path, action="append", required=True,
                        help="Path to a TDB file from the run (used to extract "
                             "the sample_rate).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the new GT but do not overwrite molecules.pkl.")
    return parser


def _backup_existing(path: Path) -> Path:
    """Move ``molecules.pkl`` aside as ``molecules.pkl.pre_v4`` before overwrite."""
    backup = path.with_suffix(".pkl.pre_v4")
    if not backup.exists():
        path.rename(backup)
    else:
        # Already backed up earlier; leave the original backup intact and
        # remove the present molecules.pkl so the new write succeeds.
        path.unlink(missing_ok=True)
    return backup


def migrate_one(
    cache_dir: Path,
    remap_dir: Path,
    probes_bin_path: Path,
    tdb_header_path: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Migrate a single cache directory in place.

    Returns a small summary dict for logging.
    """
    LOG.info("migrating cache: %s", cache_dir)

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing under {cache_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    run_id = manifest["run_id"]
    LOG.info("  run_id=%s", run_id)

    assigns_path = remap_dir / f"{run_id}_probes.txt_probeassignment.assigns"
    refmap_path = remap_dir / f"{run_id}_probes.txt_referenceMap.txt"
    for p in (assigns_path, refmap_path, probes_bin_path, tdb_header_path):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    LOG.info("  loading sources...")
    probes_file = load_probes_bin(probes_bin_path)
    assigns_list = load_assigns(assigns_path)
    refmap = load_reference_map(refmap_path)
    tdb_header = load_tdb_header(tdb_header_path)
    sample_rate_hz = int(tdb_header.sample_rate)
    LOG.info("  sample_rate=%d Hz; %d probes-bin molecules; %d assignments; "
             "%d ref probes",
             sample_rate_hz, len(probes_file.molecules), len(assigns_list),
             len(refmap.probe_positions))

    # uid -> molecule lookup (probes.bin order doesn't always match cache order)
    mol_by_uid = {int(m.uid): m for m in probes_file.molecules}
    assign_by_uid = {int(a.fragment_uid): a for a in assigns_list}

    # Iterate manifest molecules in cached order so the rebuilt GT list
    # stays index-aligned with offsets.npy / conditioning.npy / waveforms.bin.
    new_gt_list: list[dict] = []
    n_paired = 0
    n_old_subset = 0
    n_skipped = 0
    for entry in manifest["molecules"]:
        uid = int(entry["uid"])
        mol = mol_by_uid.get(uid)
        assign = assign_by_uid.get(uid)
        if mol is None or assign is None:
            n_skipped += 1
            new_gt_list.append({})  # placeholder; should not happen on real caches
            continue
        gt = build_molecule_gt(
            mol, assign, refmap,
            sample_rate_hz=sample_rate_hz,
            min_matched_probes=1,  # cache already filtered; don't drop here
            include_warmstart=True,
        )
        if gt is None:
            n_skipped += 1
            new_gt_list.append({})
            continue
        ws_centers = gt.warmstart_probe_centers_samples
        ref_pos = gt.reference_bp_positions
        if ws_centers is None or len(ws_centers) != len(ref_pos):
            n_old_subset += 1
        else:
            n_paired += 1
        new_gt_list.append(
            {
                "reference_bp_positions": gt.reference_bp_positions,
                "n_ref_probes": gt.n_ref_probes,
                "direction": gt.direction,
                "warmstart_probe_centers_samples": gt.warmstart_probe_centers_samples,
                "warmstart_probe_durations_samples": gt.warmstart_probe_durations_samples,
            }
        )

    LOG.info("  rebuilt %d GT entries; paired=%d, old-subset-fallback=%d, skipped=%d",
             len(new_gt_list), n_paired, n_old_subset, n_skipped)

    pkl_path = cache_dir / "molecules.pkl"
    if dry_run:
        LOG.info("  --dry-run: would overwrite %s", pkl_path)
    else:
        backup = _backup_existing(pkl_path)
        LOG.info("  backed up old molecules.pkl -> %s", backup.name)
        with open(pkl_path, "wb") as f:
            pickle.dump(new_gt_list, f)
        LOG.info("  wrote new molecules.pkl")

    return {
        "cache_dir": str(cache_dir),
        "run_id": run_id,
        "n_total": len(new_gt_list),
        "n_paired": n_paired,
        "n_old_subset": n_old_subset,
        "n_skipped": n_skipped,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    counts = [
        len(args.cache_dir), len(args.remap_dir),
        len(args.probes_bin), len(args.tdb_header),
    ]
    if len(set(counts)) != 1:
        raise SystemExit(
            f"flag counts must match: cache-dir={counts[0]} "
            f"remap-dir={counts[1]} probes-bin={counts[2]} "
            f"tdb-header={counts[3]}"
        )

    summaries = []
    for cache_dir, remap_dir, probes_bin, tdb_header in zip(
        args.cache_dir, args.remap_dir, args.probes_bin, args.tdb_header,
        strict=True,
    ):
        summaries.append(migrate_one(
            cache_dir, remap_dir, probes_bin, tdb_header,
            dry_run=args.dry_run,
        ))

    LOG.info("=== migration summary ===")
    for s in summaries:
        LOG.info("  %s: paired=%d / total=%d (subset_fallback=%d, skipped=%d)",
                 s["run_id"], s["n_paired"], s["n_total"],
                 s["n_old_subset"], s["n_skipped"])


if __name__ == "__main__":
    main()
