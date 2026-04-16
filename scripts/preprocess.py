"""Preprocess TDB files into compact training cache.

Usage (standard layout):
    python scripts/preprocess.py \\
        --run-dir V:/E.\\ coli/Black/<run_id>/<YYYY-MM-DD> \\
        --run-id <run_id> \\
        --output V:/E.\\ coli/cache/

Usage (explicit paths, for non-standard layouts):
    python scripts/preprocess.py \\
        --tdbs /path/to/first.tdb /path/to/second.tdb \\
        --tdb-indexes /path/to/first.tdb_index /path/to/second.tdb_index \\
        --probes-bin /path/to/file_probes.bin \\
        --assigns /path/to/file_probeassignment.assigns \\
        --reference-map /path/to/file_referenceMap.txt \\
        --run-id run_001 \\
        --output cache/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mongoose.data.preprocess import preprocess_run
from mongoose.data.run_inputs import resolve_run_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess TDB files into compact training cache")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run's date directory (contains TDBs + Remapped/AllCh). If set, all input paths are resolved automatically.")
    parser.add_argument("--tdbs", type=Path, nargs="+", default=None)
    parser.add_argument("--tdb-indexes", type=Path, nargs="+", default=None)
    parser.add_argument("--probes-bin", type=Path, default=None)
    parser.add_argument("--assigns", type=Path, default=None)
    parser.add_argument("--reference-map", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("cache"))
    parser.add_argument("--min-probes", type=int, default=8)
    parser.add_argument("--min-transloc-ms", type=float, default=30.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.run_dir is not None:
        inputs = resolve_run_inputs(args.run_dir, args.run_id)
        tdbs = inputs.tdb_paths
        tdb_indexes = inputs.tdb_index_paths
        probes_bin = inputs.probes_bin_path
        assigns = inputs.assigns_path
        ref_map = inputs.reference_map_path
    else:
        missing = [
            name for name, val in {
                "--tdbs": args.tdbs,
                "--tdb-indexes": args.tdb_indexes,
                "--probes-bin": args.probes_bin,
                "--assigns": args.assigns,
                "--reference-map": args.reference_map,
            }.items() if val is None
        ]
        if missing:
            print(f"Either --run-dir, or all of: {missing}", file=sys.stderr)
            sys.exit(2)
        tdbs = args.tdbs
        tdb_indexes = args.tdb_indexes
        probes_bin = args.probes_bin
        assigns = args.assigns
        ref_map = args.reference_map

    stats = preprocess_run(
        run_id=args.run_id,
        tdb_paths=tdbs,
        tdb_index_paths=tdb_indexes,
        probes_bin_path=probes_bin,
        assigns_path=assigns,
        reference_map_path=ref_map,
        output_dir=args.output,
        min_probes=args.min_probes,
        min_transloc_ms=args.min_transloc_ms,
    )

    print(f"Run: {stats.run_id}")
    print(f"  TDB files: {len(tdbs)}")
    print(f"  Total molecules: {stats.total_molecules}")
    print(f"  Clean molecules: {stats.clean_molecules}")
    print(f"  Remapped molecules: {stats.remapped_molecules}")
    print(f"  Cached molecules: {stats.cached_molecules}")
    print(f"  Waveform data: {stats.total_waveform_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
