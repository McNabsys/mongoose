"""CLI wrapper for Phase 0b characterization.

Usage (from worktree root):

    PYTHONPATH=src python scripts/analysis/run_phase0b.py \\
        --probe-table data/derived/probe_table/ \\
        --example-remap-settings "C:/git/mongoose/E. coli/Red/STB03-064B.../AllCh/STB03-064B.txt_remapSettings.txt" \\
        --example-reference-map  "C:/git/mongoose/E. coli/Red/STB03-064B.../AllCh/STB03-064B..._referenceMap.txt" \\
        --output-dir reports/phase0b

Artifacts end up under ``<output-dir>/``; narrative report
(``phase0b_report.md``) is intended to be written separately so the
numbers can be regenerated without touching the prose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mongoose.analysis.phase0b_classifier_characterization import characterize


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0b characterization.")
    parser.add_argument(
        "--probe-table",
        type=Path,
        required=True,
        help="Path to the probe-table parquet file or shard directory.",
    )
    parser.add_argument(
        "--example-remap-settings",
        type=Path,
        required=True,
        help="One run's _remapSettings.txt (constants verified across all runs).",
    )
    parser.add_argument(
        "--example-reference-map",
        type=Path,
        required=True,
        help="One run's referenceMap.txt (all 30 runs are hash-identical).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase0b"),
    )
    parser.add_argument(
        "--envelope-threshold",
        type=float,
        default=3.0,
        help="Mahalanobis threshold for inside/outside envelope (default: 3.0, "
             "~chi2(4) p90).",
    )
    parser.add_argument(
        "--align-score-pct",
        type=float,
        default=75.0,
        help="Percentile of molecule_align_score (within is_assigned) to use as "
             "the known-good cutoff. Sensitivity can be checked by re-running "
             "with different values.",
    )
    args = parser.parse_args()

    metrics = characterize(
        probe_table_path=args.probe_table,
        example_remap_settings=args.example_remap_settings,
        example_reference_map=args.example_reference_map,
        output_dir=args.output_dir,
        envelope_threshold=args.envelope_threshold,
        align_score_pct=args.align_score_pct,
    )

    print(f"\nwall time: {metrics['wall_time_sec']:.1f}s")
    print(
        f"narrow: precision={metrics['confusion']['narrow']['precision']:.4f}  "
        f"recall={metrics['confusion']['narrow']['recall']:.4f}  "
        f"mcc={metrics['confusion']['narrow']['mcc']:.4f}"
    )
    print(
        f"broad:  precision={metrics['confusion']['broad']['precision']:.4f}  "
        f"recall={metrics['confusion']['broad']['recall']:.4f}  "
        f"mcc={metrics['confusion']['broad']['mcc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
