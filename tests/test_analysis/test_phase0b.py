"""Unit tests for Phase 0b classifier characterization.

Synthetic-only — does not depend on the full probe table. The tests
nail down the confusion-matrix arithmetic (easy to get wrong per
spec §235) and the envelope's TP-sanity property on controlled data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mongoose.analysis.phase0b_classifier_characterization import (
    _cohen_kappa,
    _mcc,
    _metrics_from_cells,
    build_known_good_mask,
    compute_agreement_matrix,
    compute_confusion_matrix,
    fit_envelope,
    fit_per_molecule_affine,
    rejection_envelope_breakdown,
    stratified_confusion,
    add_scaled_features,
    apply_envelope,
    REJECTION_BITS,
)


def test_metrics_from_cells_exact():
    # Known numbers: 9 TP, 1 FP, 2 FN, 8 TN.
    m = _metrics_from_cells(tp=9, fp=1, fn=2, tn=8)
    assert m["tp"] == 9 and m["fp"] == 1 and m["fn"] == 2 and m["tn"] == 8
    assert m["n"] == 20
    assert m["precision"] == pytest.approx(9 / 10)
    assert m["recall"] == pytest.approx(9 / 11)
    assert m["specificity"] == pytest.approx(8 / 9)
    # F1 = 2 * 0.9 * (9/11) / (0.9 + 9/11)
    expected_f1 = 2 * (9 / 10) * (9 / 11) / ((9 / 10) + (9 / 11))
    assert m["f1"] == pytest.approx(expected_f1)


def test_metrics_degenerate_rows_return_nan():
    # Zero ACCEPT cells -> precision undefined.
    m = _metrics_from_cells(tp=0, fp=0, fn=5, tn=5)
    assert np.isnan(m["precision"])
    assert m["recall"] == 0.0  # TP / (TP + FN) = 0 / 5 = 0
    assert m["specificity"] == pytest.approx(1.0)


def test_cohen_kappa_perfect_agreement():
    # All labels agree: kappa = 1.
    assert _cohen_kappa(tt=10, tf=0, ft=0, ff=10) == pytest.approx(1.0)


def test_cohen_kappa_random_agreement():
    # 50/50 split with 50% observed agreement -> expected = 50%, kappa = 0.
    # tt=25, tf=25, ft=25, ff=25: p_o=0.5, p_yes_1=0.5, p_yes_2=0.5, p_e=0.5
    assert _cohen_kappa(tt=25, tf=25, ft=25, ff=25) == pytest.approx(0.0)


def test_mcc_perfect_correlation():
    assert _mcc(tt=10, tf=0, ft=0, ff=10) == pytest.approx(1.0)
    assert _mcc(tt=0, tf=10, ft=10, ff=0) == pytest.approx(-1.0)


def test_mcc_handles_large_integer_overflow():
    # Real-scale cells that would overflow numpy int64 on ufunc sqrt.
    m = _mcc(tt=24_000_000, tf=10_000, ft=5_000, ff=8_000_000)
    assert not np.isnan(m)
    assert 0.0 < m < 1.0


def _fake_df(cells: dict[str, int]) -> pd.DataFrame:
    """Build a 4-row DataFrame that yields the requested confusion cells.

    ``cells`` maps one of ``{'TP','FP','FN','TN'}`` to a count.
    Rows repeat to get the requested counts.
    """
    rows: list[dict] = []
    for cell, n in cells.items():
        if cell == "TP":
            row = {"attr_accepted": True, "is_assigned": True}
        elif cell == "FP":
            row = {"attr_accepted": True, "is_assigned": False}
        elif cell == "FN":
            row = {"attr_accepted": False, "is_assigned": True}
        elif cell == "TN":
            row = {"attr_accepted": False, "is_assigned": False}
        else:
            raise KeyError(cell)
        rows.extend([row] * n)
    return pd.DataFrame(rows)


def test_compute_confusion_matrix_exact_counts():
    df = _fake_df({"TP": 7, "FP": 3, "FN": 2, "TN": 11})
    cm = compute_confusion_matrix(df, "attr_accepted")
    assert cm["tp"] == 7 and cm["fp"] == 3 and cm["fn"] == 2 and cm["tn"] == 11
    assert cm["n"] == 23
    assert cm["precision"] == pytest.approx(7 / 10)
    assert cm["recall"] == pytest.approx(7 / 9)


def test_stratified_confusion_preserves_per_group_counts():
    df = _fake_df({"TP": 4, "FP": 2, "FN": 1, "TN": 3})
    df["group"] = ["A"] * 5 + ["B"] * 5  # split 5/5
    out = stratified_confusion(df, "group", "attr_accepted")
    assert set(out["stratum_value"]) == {"A", "B"}
    # Sum of per-stratum n equals total df length.
    assert int(out["n"].sum()) == len(df)


# --- Mahalanobis envelope ------------------------------------------------


def test_mahalanobis_known_good_has_small_distance():
    """TP probes sampled from the same multivariate normal as the known-good
    set should have Mahalanobis distances that track chi-square(4)."""
    rng = np.random.default_rng(0)
    n_good = 5000
    n_test = 2000
    mean = np.array([1.0, 2.0, 3.0, 4.0])
    cov_sqrt = np.array([
        [1.0, 0.2, 0.1, 0.0],
        [0.2, 1.5, 0.0, 0.1],
        [0.1, 0.0, 2.0, 0.3],
        [0.0, 0.1, 0.3, 1.0],
    ])
    cov = cov_sqrt @ cov_sqrt.T
    X_good = rng.multivariate_normal(mean, cov, size=n_good)
    X_test = rng.multivariate_normal(mean, cov, size=n_test)

    # Build a fake DataFrame the module can consume.
    cols = ["f1", "f2", "f3", "f4"]
    df = pd.DataFrame(np.vstack([X_good, X_test]), columns=cols)
    df["is_good"] = [True] * n_good + [False] * n_test

    env = fit_envelope(
        df, df["is_good"], features=tuple(cols),
    )
    # Envelope should recover the mean / cov closely.
    np.testing.assert_allclose(env.mean, mean, atol=0.1)
    # Mahalanobis of test points (from same distribution) -- median close to
    # sqrt(chi2(4).ppf(0.5)) = 1.832.
    X = df.loc[~df["is_good"], cols].to_numpy()
    d = env.mahalanobis(X)
    assert 1.5 < np.median(d) < 2.2


def test_add_scaled_features_guards_zero_baseline():
    df = pd.DataFrame({
        "duration_ms": [10.0, 20.0],
        "molecule_velocity_bp_per_ms": [1000.0, 500.0],
        "max_amp_uv": [100.0, 200.0],
        "area_samples_uv": [5000.0, 10000.0],
        "mean_lvl1_mv": [0.0, 0.5],  # first row has zero baseline
        "probe_local_density": [3, 5],
    })
    out = add_scaled_features(df)
    # Row 0 has mean_lvl1 = 0 -> amplitude_scaled / area_scaled should be NaN.
    assert np.isnan(out.loc[0, "amplitude_scaled"])
    assert np.isnan(out.loc[0, "area_scaled"])
    assert out.loc[0, "duration_scaled"] == pytest.approx(10_000.0)
    # Row 1 should have finite values.
    assert out.loc[1, "amplitude_scaled"] == pytest.approx(400.0)
    assert out.loc[1, "density_scaled"] == pytest.approx(5.0)


def test_build_known_good_mask_threshold_and_position():
    df = pd.DataFrame({
        "is_assigned": [True, True, True, True, False, True],
        "molecule_align_score": [10, 100, 500, 1000, 5000, 500],
        "center_ms": [50, 50, 50, 5, 50, 95],    # position fractions: .5 .5 .5 .05 .5 .95
        "translocation_time_ms": [100, 100, 100, 100, 100, 100],
        "duration_scaled": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "amplitude_scaled": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "area_scaled": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "density_scaled": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    mask, thr = build_known_good_mask(df, align_score_pct=50.0)
    # p50 of align_score among is_assigned=True: [10, 100, 500, 1000, 500] -> sorted [10,100,500,500,1000] -> p50 = 500
    assert thr == 500
    # Row 3 excluded (position=.05 < .10), row 4 excluded (not assigned),
    # row 5 excluded (position=.95 > .90). Rows 2 and 5 have align_score >= 500.
    assert mask.tolist() == [False, False, True, False, False, False]


# --- Rejection-envelope breakdown ----------------------------------------


def test_rejection_envelope_breakdown_columns_and_order():
    """Given known bit firings + known mahalanobis values, verify the
    per-bit inside-envelope fractions are computed correctly and the
    table is ordered by fraction descending."""
    n = 100
    df = pd.DataFrame({
        "attr_accepted": [False] * n,
        "is_assigned": [False] * n,
        "mahal_envelope": np.concatenate([
            np.full(50, 1.0),  # inside (<=3)
            np.full(50, 4.0),  # outside
        ]),
    })
    # All rejection bits default False
    for bit in REJECTION_BITS:
        df[bit] = False
    # bit A fires on first 40 rows (all inside) -> 100% inside
    df.loc[:39, "attr_folded_start"] = True
    # bit B fires on rows 10..70 (20 inside + 40 outside -> 20/60 = 33%)
    df.loc[10:70, "attr_in_structure"] = True  # .loc slice is INCLUSIVE

    out = rejection_envelope_breakdown(df, envelope_threshold=3.0)
    top = out.iloc[0]
    assert top["bit"] == "attr_folded_start"
    assert top["fraction_inside_envelope"] == pytest.approx(1.0)
    second = out[out["bit"] == "attr_in_structure"].iloc[0]
    # 10..70 inclusive = 61 rows, 40 of which are inside (rows 10..49).
    # fraction = 40 / 61.
    assert second["fraction_inside_envelope"] == pytest.approx(40 / 61)


# --- Per-molecule affine fit ---------------------------------------------


def test_fit_per_molecule_affine_recovers_known_slope():
    """3 assigned probes on a molecule, known linear relationship."""
    df = pd.DataFrame({
        "is_assigned": [True, True, True, False, True, True, True],
        "t2d_predicted_bp_pos": [0.0, 1000.0, 2000.0, 500.0, 100.0, 300.0, 500.0],
        "ref_genomic_pos_bp": [5000.0, 4500.0, 4000.0, np.nan,  2000.0, 3000.0, 4000.0],
        "run_id": ["R1"] * 4 + ["R2"] * 3,
        "molecule_uid": [1, 1, 1, 1, 2, 2, 2],
    })
    # Molecule R1/1: expected slope = (4000-5000)/2000 = -0.5, intercept = 5000.
    # Molecule R2/2: slope = (4000-2000)/(500-100) = 5.0, intercept = 1500.
    fits = fit_per_molecule_affine(df)
    assert len(fits) == 2
    r1 = fits[fits["molecule_uid"] == 1].iloc[0]
    r2 = fits[fits["molecule_uid"] == 2].iloc[0]
    assert r1["slope"] == pytest.approx(-0.5)
    assert r1["intercept"] == pytest.approx(5000.0)
    assert r2["slope"] == pytest.approx(5.0)
    assert r2["intercept"] == pytest.approx(1500.0)


def test_fit_per_molecule_affine_null_below_min_points():
    df = pd.DataFrame({
        "is_assigned": [True, True],
        "t2d_predicted_bp_pos": [0.0, 100.0],
        "ref_genomic_pos_bp": [1000.0, 1100.0],
        "run_id": ["R"] * 2,
        "molecule_uid": [1, 1],
    })
    # Only 2 assigned probes (< MIN_FIT_POINTS=3) -> slope/intercept are NaN.
    fits = fit_per_molecule_affine(df)
    row = fits.iloc[0]
    assert np.isnan(row["slope"])
    assert np.isnan(row["intercept"])


# --- Proximity oracle agreement (kept for reproducibility) ---------------


def test_agreement_matrix_perfect_agreement():
    df = pd.DataFrame({
        "attr_accepted": [True] * 4,
        "ref_idx": [1, 0, 2, 0],  # all broad-eligible
        "is_assigned": [True, False, True, False],
        "t2d_plausible_match_250": pd.array([True, False, True, False], dtype="boolean"),
    })
    out = compute_agreement_matrix(df, tolerance_bp=250)
    assert out["broad_accept"]["tt"] == 2
    assert out["broad_accept"]["ff"] == 2
    assert out["broad_accept"]["tf"] == 0 and out["broad_accept"]["ft"] == 0
    assert out["broad_accept"]["agreement_rate"] == pytest.approx(1.0)
    assert out["broad_accept"]["cohen_kappa"] == pytest.approx(1.0)
