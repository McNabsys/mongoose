"""CLI wrapper for Phase 0a T2D residual decomposition.

Usage (from worktree root):

    PYTHONPATH=src python scripts/analysis/run_phase0a.py \\
        --probe-table data/derived/probe_table/ \\
        --example-reference-map "C:/git/mongoose/E. coli/Red/STB03-064B.../AllCh/STB03-064B..._referenceMap.txt" \\
        --output-dir reports/phase0a
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mongoose.analysis.phase0a_t2d_residual_decomposition import decompose


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0a decomposition.")
    parser.add_argument("--probe-table", type=Path, required=True)
    parser.add_argument("--example-reference-map", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/phase0a")
    )
    args = parser.parse_args()

    metrics = decompose(
        probe_table_path=args.probe_table,
        example_reference_map=args.example_reference_map,
        output_dir=args.output_dir,
    )

    print(f"\nwall time: {metrics['wall_time_sec']:.1f}s")
    print(f"rows analyzed: {metrics['filter_cascade']['after_affine_fit_available']:,}")
    print(f"global abs_median_bp: {metrics['global_residual_stats']['abs_median_bp']:.1f}")
    print(f"structured variance: {metrics['multivariate_ols']['structured_variance_fraction']:.4f}")
    print(f"unstructured       : {metrics['multivariate_ols']['unstructured_variance_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
