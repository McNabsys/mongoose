"""Synthetic-only unit tests for Phase 0a residual decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mongoose.analysis.phase0a_t2d_residual_decomposition import (
    _looks_skewed_from_zero,
    compute_per_molecule_dispersion,
    global_residual_stats,
    per_molecule_headline,
    residual_by_position_along_molecule,
    residual_by_run_metadata,
    variance_decomposition_ols,
)


def _one_molecule_df(
    *, run_id: str, molecule_uid: int,
    t2d: np.ndarray, ref: np.ndarray,
) -> pd.DataFrame:
    n = t2d.size
    return pd.DataFrame({
        "run_id": [run_id] * n,
        "molecule_uid": [molecule_uid] * n,
        "probe_idx_in_molecule": np.arange(n, dtype=np.uint16),
        "t2d_predicted_bp_pos": t2d.astype(np.float64),
        "ref_genomic_pos_bp": ref.astype(np.float64),
        "predicted_genomic_bp": ref.astype(np.float64),  # assume perfect for tests; test sets residual explicitly
    })


def test_looks_skewed_from_zero_rejects_centered_wide_distribution():
    # Symmetric around zero: median ~= 0; NOT skewed.
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=0.0, scale=500.0, size=10_000)
    assert _looks_skewed_from_zero(arr) is False


def test_looks_skewed_from_zero_detects_shifted_distribution():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=300.0, scale=500.0, size=10_000)
    # median ~= 300, IQR ~= 680 -> 300 > 680/6 = 113 -> TRUE.
    assert _looks_skewed_from_zero(arr) is True


def test_compute_per_molecule_dispersion_recovers_slope_and_std():
    # Two molecules, known residual patterns.
    # Mol 1: residual = 100 + 200 * pos_frac. slope = 200, std of residuals
    # computed from actual values.
    pos1 = np.linspace(0.0, 1.0, 5)
    res1 = 100.0 + 200.0 * pos1
    # Mol 2: residual = -50 + 0 * pos_frac. slope = 0. std = 0.
    pos2 = np.linspace(0.0, 1.0, 4)
    res2 = np.full(4, -50.0)
    df = pd.DataFrame({
        "run_id": ["R"] * 9,
        "molecule_uid": [1] * 5 + [2] * 4,
        "pos_frac": np.concatenate([pos1, pos2]),
        "residual_bp": np.concatenate([res1, res2]),
    })
    per_mol = compute_per_molecule_dispersion(df)
    m1 = per_mol[per_mol["molecule_uid"] == 1].iloc[0]
    m2 = per_mol[per_mol["molecule_uid"] == 2].iloc[0]
    assert m1["trend_slope_bp_per_posfrac"] == pytest.approx(200.0)
    assert m2["trend_slope_bp_per_posfrac"] == pytest.approx(0.0)
    # Mol 1 residual_mean = mean of (100,150,200,250,300) = 200.
    assert m1["residual_mean_bp"] == pytest.approx(200.0)
    # Mol 2 std of constants = 0.
    assert m2["residual_std_bp"] == pytest.approx(0.0)


def test_per_molecule_headline_reports_stop_flags():
    # Fabricate per_mol with a tight std distribution (median < 100) and
    # a centered-on-zero slope distribution. Should NOT trip slope flag
    # but SHOULD trip std<100 flag.
    rng = np.random.default_rng(0)
    n = 500
    per_mol = pd.DataFrame({
        "run_id": ["R"] * n,
        "molecule_uid": np.arange(n),
        "n": [5] * n,
        "residual_mean_bp": np.zeros(n),
        "residual_std_bp": rng.uniform(low=10.0, high=90.0, size=n),  # median < 100
        "trend_slope_bp_per_posfrac": rng.normal(loc=0.0, scale=200.0, size=n),
    })
    h = per_molecule_headline(per_mol)
    assert h["stop_condition_std_below_100"] is True
    assert h["stop_condition_slope_skewed_from_zero"] is False


def test_residual_by_position_along_molecule_detects_head_dive():
    # Fabricate head-elevated residuals: bin 0 has abs_median 2000,
    # rest have abs_median 500.
    rng = np.random.default_rng(0)
    n_head = 1000
    n_other = 9000
    df = pd.DataFrame({
        "pos_frac": np.concatenate([
            rng.uniform(0.0, 0.05, size=n_head),
            rng.uniform(0.1, 1.0, size=n_other),
        ]),
        "residual_bp": np.concatenate([
            rng.normal(loc=0.0, scale=2500.0, size=n_head),
            rng.normal(loc=0.0, scale=600.0, size=n_other),
        ]),
    })
    df["abs_residual_bp"] = df["residual_bp"].abs()
    out = residual_by_position_along_molecule(df, n_bins=20)
    # Head bin (0.00-0.05) should have much higher abs_median than later bins.
    head_abs = float(out.iloc[0]["abs_median"])
    mid_abs = float(out.iloc[10]["abs_median"])
    assert head_abs > 2 * mid_abs


def test_residual_by_run_metadata_preserves_counts():
    df = pd.DataFrame({
        "residual_bp": [100.0, -200.0, 50.0, 300.0],
        "abs_residual_bp": [100.0, 200.0, 50.0, 300.0],
        "concentration_group": ["std", "low", "std", "low"],
        "biochem_flagged_good": [True, False, True, False],
        "excel_instrument": [433.0] * 4,
    })
    out = residual_by_run_metadata(df)
    # Per-stratum counts should sum to 4 per stratum.
    for stratum in ("concentration_group", "biochem_flagged_good", "excel_instrument"):
        sub = out[out["stratum"] == stratum]
        assert int(sub["n"].sum()) == 4


def test_variance_decomposition_ols_recovers_known_structure():
    """OLS should pull meaningful R^2 when the target has real structure,
    even on noise. Uses a synthetic |residual| with a linear dependence
    on pos_frac plus attribute bit."""
    rng = np.random.default_rng(0)
    n = 50_000
    pos_frac = rng.uniform(0.0, 1.0, size=n)
    folded = rng.integers(0, 2, size=n).astype(np.float64)
    mol_vel = rng.uniform(300.0, 1200.0, size=n)
    # Target: has real structure from pos_frac, folded flag, velocity.
    # All components chosen so the target is comfortably positive -- we
    # want the winsorized OLS to fit real variance, not a degenerate
    # clipped-to-one distribution.
    true_y = (
        2000.0
        + 1500.0 * pos_frac
        + 2000.0 * folded
        - 500.0 * np.log(mol_vel / 100.0)
    )
    noise = rng.normal(loc=0.0, scale=300.0, size=n)
    abs_resid = np.abs(true_y + noise) + 10.0  # always > 0

    df = pd.DataFrame({
        "pos_frac": pos_frac,
        "abs_residual_bp": abs_resid,
        "molecule_velocity_bp_per_ms": mol_vel,
        "num_probes": rng.integers(6, 100, size=n),
        "duration_ms": rng.uniform(0.1, 2.0, size=n),
        "attr_folded_start": folded.astype(bool),
        "attr_in_structure": np.zeros(n, dtype=bool),
        "probe_local_density": rng.integers(1, 20, size=n),
    })
    out = variance_decomposition_ols(df)
    # R^2 should be meaningfully above zero (synthetic data has real structure).
    assert out["r_squared"] > 0.3
    # Coefficient signs should match the synthetic generation:
    # pos_frac positive, folded positive, log_mol_vel negative.
    by_name = {c["feature"]: c["coefficient_bp"] for c in out["coefficients"]}
    assert by_name["pos_frac"] > 0
    assert by_name["attr_folded_start"] > 0
    assert by_name["log_mol_vel"] < 0


def test_global_residual_stats_contains_expected_keys():
    df = pd.DataFrame({
        "residual_bp": np.linspace(-1000, 1000, 100),
        "abs_residual_bp": np.abs(np.linspace(-1000, 1000, 100)),
    })
    g = global_residual_stats(df)
    for k in ("n", "signed_mean_bp", "signed_median_bp", "abs_median_bp",
              "abs_p90_bp", "frac_gt_250_bp", "frac_gt_1000_bp"):
        assert k in g
    assert g["n"] == 100
