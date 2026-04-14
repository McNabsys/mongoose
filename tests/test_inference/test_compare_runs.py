"""Tests for the scripts/compare_runs.py comparison-table helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is a sibling of src/, not an installed package, so add it to path.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_runs  # noqa: E402


def _make_result(
    mae: float,
    median_ae: float = 100.0,
    std_ae: float = 50.0,
    num_intervals: int = 100,
    num_mols: int = 40,
    peak_mean: float = 0.0,
    peak_more: float = 0.3,
    peak_fewer: float = 0.2,
    peak_equal: float = 0.5,
    include_legacy: bool = True,
    include_peak: bool = True,
) -> dict:
    result = {
        "metadata": {
            "checkpoint": "x.pt",
            "run_id": "test",
            "num_molecules_evaluated": num_mols,
            "timestamp": "2026-04-13T12:00:00",
        },
        "dl_model": {
            "mae_bp": mae,
            "median_ae_bp": median_ae,
            "std_ae_bp": std_ae,
            "num_intervals": num_intervals,
        },
    }
    if include_legacy:
        result["legacy_t2d"] = {
            "mae_bp": 200.0,
            "median_ae_bp": 180.0,
            "std_ae_bp": 80.0,
            "num_intervals": num_intervals,
        }
    if include_peak:
        result["peak_count"] = {
            "mean_discrepancy": peak_mean,
            "median_discrepancy": 0.0,
            "std_discrepancy": 1.0,
            "fraction_more_detections": peak_more,
            "fraction_fewer_detections": peak_fewer,
            "fraction_equal_detections": peak_equal,
            "num_molecules": num_mols,
        }
    return result


def test_build_comparison_table_basic():
    r1 = _make_result(mae=123.4, peak_mean=0.5, peak_more=0.35, peak_fewer=0.15)
    r2 = _make_result(mae=145.2, peak_mean=0.3, peak_more=0.30, peak_fewer=0.20)

    table = compare_runs.build_comparison_table(
        [r1, r2], ["baseline", "no_velocity"]
    )

    assert "baseline" in table
    assert "no_velocity" in table
    assert "DL MAE (bp):" in table
    assert "123.4" in table
    assert "145.2" in table
    assert "Peak discrepancy mean:" in table
    assert "+0.50" in table
    assert "+0.30" in table
    assert "Peak more detections:" in table
    assert "35%" in table
    assert "30%" in table


def test_build_comparison_table_no_legacy():
    r1 = _make_result(mae=100.0, include_legacy=False)
    r2 = _make_result(mae=110.0, include_legacy=False)
    table = compare_runs.build_comparison_table([r1, r2], ["a", "b"])
    assert "Legacy T2D" not in table
    assert "DL MAE" in table


def test_build_comparison_table_no_peak_count():
    r1 = _make_result(mae=100.0, include_peak=False)
    r2 = _make_result(mae=110.0, include_peak=False)
    table = compare_runs.build_comparison_table([r1, r2], ["a", "b"])
    assert "Peak discrepancy" not in table
    assert "DL MAE" in table


def test_build_comparison_table_length_mismatch_raises():
    r1 = _make_result(mae=100.0)
    with pytest.raises(ValueError):
        compare_runs.build_comparison_table([r1], ["a", "b"])


def test_build_comparison_table_empty_raises():
    with pytest.raises(ValueError):
        compare_runs.build_comparison_table([], [])


def test_load_result_roundtrip(tmp_path):
    payload = _make_result(mae=77.7)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = compare_runs.load_result(path)
    assert loaded["dl_model"]["mae_bp"] == 77.7
    assert loaded["peak_count"]["fraction_more_detections"] == 0.3


def test_build_comparison_table_three_runs():
    r1 = _make_result(mae=100.0)
    r2 = _make_result(mae=200.0)
    r3 = _make_result(mae=300.0)
    table = compare_runs.build_comparison_table(
        [r1, r2, r3], ["lo", "mid", "hi"]
    )
    assert "lo" in table and "mid" in table and "hi" in table
    assert "100.0" in table and "200.0" in table and "300.0" in table
