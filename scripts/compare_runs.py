"""Compare multiple evaluation JSON result files side-by-side.

Intended for the V1 lambda_vel ablation: train two models (lambda_vel=1.0 and
lambda_vel=0.0), evaluate each to a JSON file, then run this script to compare.

Usage:
    python scripts/compare_runs.py \
        --results results_v1.json results_v1_no_velocity.json \
        --labels "V1 (lambda_vel=1.0)" "V1 (lambda_vel=0.0)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path: str | Path) -> dict:
    """Load a single evaluation JSON result file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt_float(value: float | None, spec: str = ".1f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _fmt_signed(value: float | None, spec: str = "+.2f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def build_comparison_table(results: list[dict], labels: list[str]) -> str:
    """Build a formatted comparison table from evaluation result dicts.

    Args:
        results: List of parsed JSON dicts (one per run).
        labels: One label per result; must be the same length as results.

    Returns:
        A formatted multi-line string suitable for printing.

    Raises:
        ValueError: If lengths don't match or lists are empty.
    """
    if len(results) != len(labels):
        raise ValueError(
            f"results and labels must have equal length "
            f"(got {len(results)} and {len(labels)})"
        )
    if len(results) == 0:
        raise ValueError("Need at least one result to compare")

    # Column width: max(label, 20) + padding
    col_width = max(max(len(label) for label in labels), 20) + 2
    label_col = 28

    header_title = "Comparison: " + " vs ".join(labels)
    lines: list[str] = [header_title, "=" * max(len(header_title), 32)]

    # Header row
    header = " " * label_col + "".join(f"{label:<{col_width}s}" for label in labels)
    lines.append(header)

    def row(name: str, values: list[str]) -> str:
        return f"{name:<{label_col}s}" + "".join(
            f"{v:<{col_width}s}" for v in values
        )

    # DL model metrics
    lines.append(
        row(
            "DL MAE (bp):",
            [_fmt_float(r.get("dl_model", {}).get("mae_bp")) for r in results],
        )
    )
    lines.append(
        row(
            "DL median AE (bp):",
            [
                _fmt_float(r.get("dl_model", {}).get("median_ae_bp"))
                for r in results
            ],
        )
    )
    lines.append(
        row(
            "DL std AE (bp):",
            [_fmt_float(r.get("dl_model", {}).get("std_ae_bp")) for r in results],
        )
    )

    # Legacy T2D metrics (may be absent)
    has_legacy = any("legacy_t2d" in r for r in results)
    if has_legacy:
        lines.append(
            row(
                "Legacy T2D MAE (bp):",
                [
                    _fmt_float(r.get("legacy_t2d", {}).get("mae_bp"))
                    for r in results
                ],
            )
        )
        lines.append(
            row(
                "Legacy T2D median AE (bp):",
                [
                    _fmt_float(r.get("legacy_t2d", {}).get("median_ae_bp"))
                    for r in results
                ],
            )
        )

    # Peak-count discrepancy (may be absent)
    has_peak = any("peak_count" in r for r in results)
    if has_peak:
        lines.append(
            row(
                "Peak discrepancy mean:",
                [
                    _fmt_signed(r.get("peak_count", {}).get("mean_discrepancy"))
                    for r in results
                ],
            )
        )
        lines.append(
            row(
                "Peak discrepancy median:",
                [
                    _fmt_signed(
                        r.get("peak_count", {}).get("median_discrepancy")
                    )
                    for r in results
                ],
            )
        )
        lines.append(
            row(
                "Peak more detections:",
                [
                    _fmt_percent(
                        r.get("peak_count", {}).get("fraction_more_detections")
                    )
                    for r in results
                ],
            )
        )
        lines.append(
            row(
                "Peak fewer detections:",
                [
                    _fmt_percent(
                        r.get("peak_count", {}).get("fraction_fewer_detections")
                    )
                    for r in results
                ],
            )
        )
        lines.append(
            row(
                "Peak equal detections:",
                [
                    _fmt_percent(
                        r.get("peak_count", {}).get("fraction_equal_detections")
                    )
                    for r in results
                ],
            )
        )

    # Molecule count row (sanity check)
    lines.append(
        row(
            "Molecules evaluated:",
            [
                str(r.get("metadata", {}).get("num_molecules_evaluated", "n/a"))
                for r in results
            ],
        )
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple evaluation JSON result files"
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluation JSON files (produced by evaluate.py --output-json)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Human-readable label for each results file",
    )
    args = parser.parse_args()

    if len(args.results) != len(args.labels):
        print(
            f"error: --results ({len(args.results)}) and --labels "
            f"({len(args.labels)}) must have the same number of entries",
            file=sys.stderr,
        )
        sys.exit(2)

    results = [load_result(p) for p in args.results]
    print(build_comparison_table(results, list(args.labels)))


if __name__ == "__main__":
    main()
