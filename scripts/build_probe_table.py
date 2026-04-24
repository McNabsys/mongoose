"""CLI wrapper for the Phase 0 unified probe-table ETL.

Usage (run from the worktree root):

    PYTHONPATH=src python scripts/build_probe_table.py \\
        --excel "C:/git/mongoose/E. coli/support/Project Mongoose - input data.xlsx" \\
        --data-root "C:/git/mongoose/E. coli" \\
        --output-dir "data/derived" \\
        --workers 4

See src/mongoose/etl/build.py for the core logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mongoose.etl.build import build_probe_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase 0 unified probe table."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        required=True,
        help="Path to 'Project Mongoose - input data.xlsx'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root of the E. coli data (contains Black/, Red/, Blue/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived"),
        help="Output directory. Shards -> <output-dir>/probe_table/*.parquet, "
             "merged -> <output-dir>/probe_table.parquet, "
             "manifest -> <output-dir>/probe_table_manifest.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--no-compute-t2d",
        action="store_true",
        help="Skip t2d_predicted_bp_pos computation (faster; column stays null).",
    )
    parser.add_argument(
        "--no-merged",
        action="store_true",
        help="Skip writing the merged single-file parquet (shards only).",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated Run IDs to ingest (default: all 30 short runs).",
    )
    args = parser.parse_args()

    run_subset = (
        [r.strip() for r in args.runs.split(",") if r.strip()]
        if args.runs
        else None
    )

    manifest = build_probe_table(
        excel_path=args.excel,
        data_root=args.data_root,
        output_dir=args.output_dir,
        compute_t2d=not args.no_compute_t2d,
        workers=args.workers,
        run_subset=run_subset,
        write_merged=not args.no_merged,
    )

    print("\n=== manifest summary ===")
    print(f"  schema_version       : {manifest['schema_version']}")
    print(f"  git_commit           : {manifest['git_commit']}")
    print(f"  runs requested       : {manifest['n_runs_requested']}")
    print(f"  runs ingested (ok)   : {manifest['n_runs_ingested']}")
    print(f"  runs skipped         : {manifest['n_runs_skipped']}")
    print(f"  runs failed          : {manifest['n_runs_failed']}")
    print(f"  total probes         : {manifest['total_probe_count']:,}")
    print(f"  total molecules      : {manifest['total_molecule_count']:,}")
    print(f"  biochem flagged      : {manifest['biochem_flagged_count']}")
    print(f"  wall time            : {manifest['wall_time_sec']:.1f}s")

    return 0 if manifest["n_runs_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
