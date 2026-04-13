"""Preprocess TDB files into compact training cache.

Usage:
    python scripts/preprocess.py \
        --tdb /path/to/file.tdb \
        --probes-bin /path/to/file_probes.bin \
        --assigns /path/to/file_probeassignment.assigns \
        --reference-map /path/to/file_referenceMap.txt \
        --run-id run_001 \
        --output cache/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mongoose.data.preprocess import preprocess_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess TDB files into compact training cache"
    )
    parser.add_argument("--tdb", type=Path, required=True, help="Path to .tdb file")
    parser.add_argument(
        "--probes-bin", type=Path, required=True, help="Path to _probes.bin"
    )
    parser.add_argument(
        "--assigns", type=Path, required=True, help="Path to _probeassignment.assigns"
    )
    parser.add_argument(
        "--reference-map", type=Path, required=True, help="Path to _referenceMap.txt"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cache"), help="Output directory"
    )
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument("--min-probes", type=int, default=8)
    parser.add_argument("--min-transloc-ms", type=float, default=30.0)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stats = preprocess_run(
        run_id=args.run_id,
        tdb_path=args.tdb,
        probes_bin_path=args.probes_bin,
        assigns_path=args.assigns,
        reference_map_path=args.reference_map,
        output_dir=args.output,
        min_probes=args.min_probes,
        min_transloc_ms=args.min_transloc_ms,
    )

    print(f"Run: {stats.run_id}")
    print(f"  Total molecules: {stats.total_molecules}")
    print(f"  Clean molecules: {stats.clean_molecules}")
    print(f"  Remapped molecules: {stats.remapped_molecules}")
    print(f"  Cached molecules: {stats.cached_molecules}")
    print(f"  Waveform data: {stats.total_waveform_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
