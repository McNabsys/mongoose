"""Phase 0a: T2D per-probe residual decomposition.

Entry point: :func:`decompose`. See ``phase0a_t2d_residual_spec.md``.

Residual definition (IMPORTANT):

  We reuse Phase 0b's ``fit_per_molecule_affine`` +
  ``predict_genomic_positions`` so each residual is
  ``predicted_genomic_bp - ref_genomic_pos_bp`` after an OLS affine fit
  per aligned molecule. That fit zeros the per-molecule MEAN residual
  by construction. Every finding in this phase is a DEVIATION FROM
  THAT PER-MOLECULE LINEAR FIT, not an absolute T2D error.

  Axis 1 (position along molecule) is therefore a nonlinearity finder,
  not a bias finder. Head-dive as a CONSTANT offset would be absorbed
  into the affine. What survives at axis 1 is head-dive nonlinearity
  (curvature of the velocity profile near molecule ends) — which is
  the more interesting finding anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from mongoose.analysis.phase0b_classifier_characterization import (
    ANALYSIS_COLUMNS as PHASE0B_COLUMNS,
    MIN_PROBES_REMAPPING,
    fit_per_molecule_affine,
    load_shared_reference_map,
    predict_genomic_positions,
)


RESIDUAL_COLUMNS: tuple[str, ...] = tuple(set(PHASE0B_COLUMNS) | {
    "mean_lvl1_mv",
    "rise_time_t50_ms",
    "fall_time_t50_ms",
    "molecule_stretch_factor",
    "molecule_stretch_offset",
    "ref_strand",
})


@dataclass(frozen=True)
class ResidualCascade:
    total: int
    after_aligned: int
    after_do_not_use: int
    after_min_probes: int
    after_is_assigned: int
    after_t2d_non_null: int
    after_affine_fit: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_rows": self.total,
            "after_molecule_aligned": self.after_aligned,
            "after_not_do_not_use": self.after_do_not_use,
            "after_min_probes": self.after_min_probes,
            "after_is_assigned": self.after_is_assigned,
            "after_t2d_non_null": self.after_t2d_non_null,
            "after_affine_fit_available": self.after_affine_fit,
        }


def load_residual_eligible(
    probe_table_path: Path | str,
    *,
    columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, ResidualCascade]:
    """Filter cascade + residual computation.

    Emits a DataFrame with one row per residual-eligible assigned probe,
    with columns:
      - pos_frac                 : probe center / translocation_time_ms
      - predicted_genomic_bp     : from per-molecule OLS fit
      - residual_bp              : predicted_genomic_bp - ref_genomic_pos_bp
      - abs_residual_bp          : |residual_bp|
    plus everything listed in :data:`RESIDUAL_COLUMNS`.
    """
    cols = set(columns) if columns is not None else set(RESIDUAL_COLUMNS)
    cols |= {"run_id", "molecule_uid", "is_assigned", "molecule_aligned",
             "molecule_do_not_use", "num_probes", "t2d_predicted_bp_pos",
             "ref_genomic_pos_bp", "center_ms", "translocation_time_ms"}

    dataset = ds.dataset(Path(probe_table_path), format="parquet")
    total = dataset.count_rows()
    f_aligned = ds.field("molecule_aligned") == True  # noqa: E712
    after_aligned = dataset.count_rows(filter=f_aligned)
    f_dnu = f_aligned & (ds.field("molecule_do_not_use") == False)  # noqa: E712
    after_dnu = dataset.count_rows(filter=f_dnu)
    f_min = f_dnu & (ds.field("num_probes") >= MIN_PROBES_REMAPPING)
    after_min = dataset.count_rows(filter=f_min)
    f_assigned = f_min & (ds.field("is_assigned") == True)  # noqa: E712
    after_assigned = dataset.count_rows(filter=f_assigned)
    f_t2d = f_assigned & ds.field("t2d_predicted_bp_pos").is_valid()
    after_t2d = dataset.count_rows(filter=f_t2d)

    # Load the full analysis-eligible set (not just assigned) because we
    # need all probes to fit the per-molecule affine. Then filter to
    # assigned-with-t2d for the residual dataframe.
    full = dataset.to_table(columns=list(cols), filter=f_min).to_pandas()
    fits = fit_per_molecule_affine(full)
    full = predict_genomic_positions(full, fits)

    df = full[
        full["is_assigned"]
        & full["t2d_predicted_bp_pos"].notna()
        & full["predicted_genomic_bp"].notna()
    ].copy()
    df["residual_bp"] = (
        df["predicted_genomic_bp"].astype(np.float64)
        - df["ref_genomic_pos_bp"].astype(np.float64)
    )
    df["abs_residual_bp"] = df["residual_bp"].abs()
    df["pos_frac"] = (
        df["center_ms"].astype(np.float64)
        / df["translocation_time_ms"].astype(np.float64)
    )
    cascade = ResidualCascade(
        total=total,
        after_aligned=after_aligned,
        after_do_not_use=after_dnu,
        after_min_probes=after_min,
        after_is_assigned=after_assigned,
        after_t2d_non_null=after_t2d,
        after_affine_fit=len(df),
    )
    return df, cascade


def global_residual_stats(df: pd.DataFrame) -> dict[str, float]:
    """Headline numbers on the global signed + absolute residual."""
    signed = df["residual_bp"].astype(np.float64)
    absval = df["abs_residual_bp"].astype(np.float64)
    return {
        "n": int(len(df)),
        "signed_mean_bp": float(signed.mean()),
        "signed_median_bp": float(signed.median()),
        "signed_p1_bp": float(signed.quantile(0.01)),
        "signed_p99_bp": float(signed.quantile(0.99)),
        "abs_median_bp": float(absval.median()),
        "abs_p75_bp": float(absval.quantile(0.75)),
        "abs_p90_bp": float(absval.quantile(0.9)),
        "abs_p95_bp": float(absval.quantile(0.95)),
        "abs_p99_bp": float(absval.quantile(0.99)),
        "frac_gt_250_bp": float((absval > 250).mean()),
        "frac_gt_1000_bp": float((absval > 1000).mean()),
    }


# --- Axis 2 (my ordering; spec axis 4 partial): per-molecule statistics --
#
# Runs BEFORE axis 1 because per-molecule dispersion + per-molecule
# trend-slope tell us the fork between "calibration-fixable" (uniform
# offsets per molecule -- but OLS has already zeroed those) and
# "structure-fixable" (within-molecule shape).
#
# Stop conditions (surface if hit, per Jon 2026-04-23):
#   1. median per-molecule residual std < 100 bp
#      -> T2D is nearly perfect within-molecule post-affine; all the
#         error is in the affine calibration itself. Surprising.
#   2. per-molecule trend-slope distribution is bimodal or heavily
#      skewed from zero -> shared within-molecule structure T2D isn't
#      capturing. Strong signal for a nonlinear correction.


def compute_per_molecule_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """Per-molecule residual std + per-molecule trend slope.

    The trend slope is the OLS slope of ``residual_bp`` against
    ``pos_frac`` within a single molecule. If this slope distribution
    is centered on zero with modest spread, T2D has no shared
    within-molecule trend. If it's off-zero or bimodal, there's a
    shared shape the affine fit didn't absorb.
    """
    grp = df.groupby(["run_id", "molecule_uid"], sort=False)

    # Fast closed-form slope via pre-computed sums.
    x = df["pos_frac"].astype(np.float64).to_numpy()
    y = df["residual_bp"].astype(np.float64).to_numpy()
    helper = pd.DataFrame({
        "run_id": df["run_id"].to_numpy(),
        "molecule_uid": df["molecule_uid"].to_numpy(),
        "x": x,
        "y": y,
        "xx": x * x,
        "xy": x * y,
        "yy": y * y,
    })
    agg = helper.groupby(["run_id", "molecule_uid"], sort=False).agg(
        n=("x", "size"),
        sum_x=("x", "sum"),
        sum_y=("y", "sum"),
        sum_xx=("xx", "sum"),
        sum_xy=("xy", "sum"),
        mean_y=("y", "mean"),
        std_y=("y", "std"),
    ).reset_index()

    n = agg["n"].to_numpy(dtype=np.float64)
    sum_x = agg["sum_x"].to_numpy(dtype=np.float64)
    sum_y = agg["sum_y"].to_numpy(dtype=np.float64)
    sum_xx = agg["sum_xx"].to_numpy(dtype=np.float64)
    sum_xy = agg["sum_xy"].to_numpy(dtype=np.float64)
    denom = n * sum_xx - sum_x ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["trend_slope_bp_per_posfrac"] = np.where(
            denom > 0, (n * sum_xy - sum_x * sum_y) / denom, np.nan
        )
    agg = agg.rename(columns={"std_y": "residual_std_bp", "mean_y": "residual_mean_bp"})
    return agg[
        ["run_id", "molecule_uid", "n", "residual_mean_bp",
         "residual_std_bp", "trend_slope_bp_per_posfrac"]
    ]


def per_molecule_headline(per_mol: pd.DataFrame) -> dict[str, float]:
    """Headline numbers + stop-condition checks."""
    std = per_mol["residual_std_bp"].dropna()
    slope = per_mol["trend_slope_bp_per_posfrac"].dropna()
    return {
        "n_molecules": int(len(per_mol)),
        "residual_std_median_bp": float(std.median()),
        "residual_std_p90_bp": float(std.quantile(0.9)),
        "residual_std_p99_bp": float(std.quantile(0.99)),
        "residual_mean_abs_median_bp": float(per_mol["residual_mean_bp"].abs().median()),
        "trend_slope_median_bp_per_posfrac": float(slope.median()),
        "trend_slope_iqr_bp_per_posfrac": float(
            slope.quantile(0.75) - slope.quantile(0.25)
        ),
        "trend_slope_frac_positive": float((slope > 0).mean()),
        "trend_slope_frac_negative": float((slope < 0).mean()),
        "trend_slope_abs_median_bp_per_posfrac": float(slope.abs().median()),
        # Stop-condition flags
        "stop_condition_std_below_100": bool(std.median() < 100.0),
        "stop_condition_slope_skewed_from_zero": _looks_skewed_from_zero(
            slope.to_numpy()
        ),
    }


# --- Axis 1: position along molecule ------------------------------------
#
# This is a NONLINEARITY finder, not a bias finder. A constant head-dive
# offset would have been absorbed by the per-molecule affine. What we're
# looking for at this axis is head/tail CURVATURE that the affine could
# not absorb: the median residual going systematically positive or
# negative as pos_frac approaches 0 or 1.


def residual_by_position_along_molecule(
    df: pd.DataFrame, *, n_bins: int = 20,
) -> pd.DataFrame:
    """Residual distribution stratified by probe center position fraction.

    Reported quantities per bin: signed median (the bias signal),
    abs median, signed p10 / p90 (envelope width), count.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned = df.copy()
    binned["pos_bin"] = pd.cut(
        binned["pos_frac"], bins=edges, include_lowest=True,
        labels=[f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(n_bins)],
    )
    grp = binned.groupby("pos_bin", observed=True)
    out = grp["residual_bp"].agg(
        n="size",
        signed_median="median",
        signed_p10=lambda s: float(s.quantile(0.1)),
        signed_p90=lambda s: float(s.quantile(0.9)),
    )
    abs_med = grp["abs_residual_bp"].median()
    out["abs_median"] = abs_med
    return out.reset_index()


# --- Axis 3 (spec axis 5): run-level covariates --------------------------


def residual_by_run_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (stratum, value) covering the spec §5 covariates."""
    rows: list[dict] = []
    for stratum in ("concentration_group", "biochem_flagged_good",
                    "excel_instrument"):
        if stratum not in df.columns:
            continue
        for value, sub in df.groupby(stratum, dropna=False, observed=True):
            rows.append(_run_stratum_row(stratum, value, sub))
    # SNR tercile
    if "excel_snr" in df.columns and df["excel_snr"].notna().any():
        snr = df["excel_snr"].astype(np.float64)
        df = df.copy()
        df["snr_tercile"] = pd.qcut(
            snr, q=3, labels=["SNR_low", "SNR_mid", "SNR_high"]
        ).astype("string")
        for value, sub in df.groupby("snr_tercile", dropna=False, observed=True):
            rows.append(_run_stratum_row("snr_tercile", value, sub))
    return pd.DataFrame(rows)


def _run_stratum_row(stratum: str, value, sub: pd.DataFrame) -> dict:
    signed = sub["residual_bp"].astype(np.float64)
    absval = sub["abs_residual_bp"].astype(np.float64)
    return {
        "stratum": stratum,
        "value": value,
        "n": int(len(sub)),
        "signed_median_bp": float(signed.median()),
        "abs_median_bp": float(absval.median()),
        "abs_p90_bp": float(absval.quantile(0.9)),
        "frac_gt_250_bp": float((absval > 250).mean()),
        "frac_gt_1000_bp": float((absval > 1000).mean()),
    }


# --- Axis 4 (spec axis 2): velocity --------------------------------------


def residual_by_velocity(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Residual vs molecule_velocity_bp_per_ms + local velocity proxy."""
    out: dict[str, pd.DataFrame] = {}

    # Molecule-mean velocity deciles
    mv = df["molecule_velocity_bp_per_ms"].astype(np.float64)
    df_mv = df[mv.notna()].copy()
    df_mv["vel_decile"] = pd.qcut(
        df_mv["molecule_velocity_bp_per_ms"].astype(np.float64),
        q=10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop",
    )
    mv_grp = df_mv.groupby("vel_decile", observed=True)
    out["molecule_velocity_deciles"] = pd.DataFrame({
        "n": mv_grp.size(),
        "velocity_median_bp_per_ms": mv_grp[
            "molecule_velocity_bp_per_ms"
        ].median(),
        "signed_median_residual_bp": mv_grp["residual_bp"].median(),
        "abs_median_residual_bp": mv_grp["abs_residual_bp"].median(),
        "abs_p90_residual_bp": mv_grp["abs_residual_bp"].quantile(0.9),
    }).reset_index()

    # Local velocity proxy: bp/ms between a probe and its predecessor-in-
    # detection-order assigned probe on the same molecule. This is the
    # average translocation velocity over the inter-probe segment.
    df_sorted = df.sort_values(["run_id", "molecule_uid", "probe_idx_in_molecule"])
    ref = df_sorted["ref_genomic_pos_bp"].astype(np.float64).to_numpy()
    center = df_sorted["center_ms"].astype(np.float64).to_numpy()
    bp_diff = np.abs(np.concatenate([[np.nan], np.diff(ref)]))
    ms_diff = np.abs(np.concatenate([[np.nan], np.diff(center)]))
    first_mask = df_sorted.groupby(
        ["run_id", "molecule_uid"], sort=False
    ).cumcount() == 0
    bp_diff[first_mask.to_numpy()] = np.nan
    ms_diff[first_mask.to_numpy()] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        local_vel = np.where(ms_diff > 0, bp_diff / ms_diff, np.nan)
    df_sorted["local_velocity_bp_per_ms"] = local_vel
    df_sorted["local_mean_vel_ratio"] = (
        local_vel / df_sorted["molecule_velocity_bp_per_ms"].astype(np.float64)
    )
    lv = df_sorted[np.isfinite(df_sorted["local_mean_vel_ratio"])].copy()
    if len(lv):
        lv["local_ratio_decile"] = pd.qcut(
            lv["local_mean_vel_ratio"], q=10,
            labels=[f"D{i+1}" for i in range(10)], duplicates="drop",
        )
        lv_grp = lv.groupby("local_ratio_decile", observed=True)
        out["local_over_mean_velocity_deciles"] = pd.DataFrame({
            "n": lv_grp.size(),
            "local_over_mean_median": lv_grp["local_mean_vel_ratio"].median(),
            "signed_median_residual_bp": lv_grp["residual_bp"].median(),
            "abs_median_residual_bp": lv_grp["abs_residual_bp"].median(),
            "abs_p90_residual_bp": lv_grp["abs_residual_bp"].quantile(0.9),
        }).reset_index()
    return out


# --- Axis 5 (spec axis 3): per-probe features ----------------------------


PROBE_FEATURE_COLS: tuple[str, ...] = (
    "duration_ms", "max_amp_uv", "area_samples_uv", "probe_local_density",
    "prev_probe_gap_ms", "next_probe_gap_ms",
)


def residual_by_probe_features(df: pd.DataFrame) -> dict[str, object]:
    """Per-feature decile breakdown + per-attribute-bit residual profile."""
    out: dict[str, object] = {}

    feature_frames: dict[str, pd.DataFrame] = {}
    for feat in PROBE_FEATURE_COLS:
        if feat not in df.columns:
            continue
        vals = df[feat].astype(np.float64)
        mask = vals.notna() & np.isfinite(vals)
        sub = df[mask].copy()
        sub[f"{feat}_decile"] = pd.qcut(
            sub[feat].astype(np.float64), q=10,
            labels=[f"D{i+1}" for i in range(10)], duplicates="drop",
        )
        grp = sub.groupby(f"{feat}_decile", observed=True)
        feature_frames[feat] = pd.DataFrame({
            "n": grp.size(),
            "feature_median": grp[feat].median(),
            "signed_median_residual_bp": grp["residual_bp"].median(),
            "abs_median_residual_bp": grp["abs_residual_bp"].median(),
            "abs_p90_residual_bp": grp["abs_residual_bp"].quantile(0.9),
        }).reset_index()
    out["per_feature_deciles"] = feature_frames

    # Pearson correlation, absolute residual vs each feature.
    corrs: list[dict] = []
    absres = df["abs_residual_bp"].astype(np.float64).to_numpy()
    for feat in PROBE_FEATURE_COLS:
        if feat not in df.columns:
            continue
        v = df[feat].astype(np.float64).to_numpy()
        ok = np.isfinite(absres) & np.isfinite(v)
        if ok.sum() < 1000:
            continue
        # Pearson (linear) + Spearman-like via rank would be better but
        # costly at 25M rows; keep Pearson and note the linearity in
        # the report.
        r = float(np.corrcoef(absres[ok], v[ok])[0, 1])
        corrs.append({"feature": feat, "pearson_r_vs_abs_residual": r, "n": int(ok.sum())})
    out["pearson_correlations"] = pd.DataFrame(corrs)

    # Per-attribute-bit: signed + abs median residual when bit is set vs clear.
    bit_rows: list[dict] = []
    attr_bits = [c for c in df.columns if c.startswith("attr_")
                 and c not in ("attr_bitfield_raw",)]
    for bit in attr_bits:
        set_mask = df[bit].astype(bool)
        if int(set_mask.sum()) < 100:
            continue
        on = df[set_mask]
        off = df[~set_mask]
        bit_rows.append({
            "bit": bit,
            "n_set": int(len(on)),
            "n_clear": int(len(off)),
            "signed_median_residual_bp_set": float(on["residual_bp"].median()),
            "signed_median_residual_bp_clear": float(off["residual_bp"].median()),
            "abs_median_residual_bp_set": float(on["abs_residual_bp"].median()),
            "abs_median_residual_bp_clear": float(off["abs_residual_bp"].median()),
            "abs_ratio_set_over_clear": float(
                on["abs_residual_bp"].median() / off["abs_residual_bp"].median()
            ) if len(off) and off["abs_residual_bp"].median() > 0 else float("nan"),
        })
    out["per_attribute_bit"] = pd.DataFrame(bit_rows)
    return out


# --- Axis 6 (spec axis 4 remainder): molecule-level features -------------


MOLECULE_FEATURE_COLS: tuple[str, ...] = (
    "translocation_time_ms", "molecule_align_score",
    "molecule_stretch_factor", "num_probes", "mean_lvl1_mv",
    "rise_time_t50_ms", "fall_time_t50_ms", "molecule_velocity_bp_per_ms",
)


def residual_by_molecule_features(
    df: pd.DataFrame, per_mol: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Correlate per-molecule residual std / trend with molecule features.

    ``per_mol`` is :func:`compute_per_molecule_dispersion` output;
    ``df`` supplies molecule-level covariates (one value per molecule,
    taken as the first probe's value for broadcast cols).
    """
    mol_feats = (
        df.groupby(["run_id", "molecule_uid"], sort=False)[
            list(MOLECULE_FEATURE_COLS)
        ].first().reset_index()
    )
    merged = per_mol.merge(mol_feats, on=["run_id", "molecule_uid"], how="left")

    out: dict[str, pd.DataFrame] = {}

    # Decile breakdowns of per-molecule residual_std_bp by each feature.
    decile_frames: dict[str, pd.DataFrame] = {}
    for feat in MOLECULE_FEATURE_COLS:
        if feat not in merged.columns:
            continue
        vals = merged[feat].astype(np.float64)
        sub = merged[vals.notna() & np.isfinite(vals)].copy()
        if len(sub) < 1000:
            continue
        sub[f"{feat}_decile"] = pd.qcut(
            sub[feat].astype(np.float64), q=10,
            labels=[f"D{i+1}" for i in range(10)], duplicates="drop",
        )
        sub["abs_trend_slope"] = sub["trend_slope_bp_per_posfrac"].abs()
        grp = sub.groupby(f"{feat}_decile", observed=True)
        decile_frames[feat] = pd.DataFrame({
            "n_mols": grp.size(),
            "feature_median": grp[feat].median(),
            "residual_std_median_bp": grp["residual_std_bp"].median(),
            "trend_slope_median": grp["trend_slope_bp_per_posfrac"].median(),
            "trend_slope_abs_median": grp["abs_trend_slope"].median(),
        }).reset_index()
    out["per_molecule_feature_deciles"] = decile_frames

    # Pearson of residual_std_bp vs each feature.
    corrs: list[dict] = []
    s = merged["residual_std_bp"].to_numpy()
    for feat in MOLECULE_FEATURE_COLS:
        if feat not in merged.columns:
            continue
        v = merged[feat].astype(np.float64).to_numpy()
        ok = np.isfinite(s) & np.isfinite(v)
        if ok.sum() < 1000:
            continue
        r = float(np.corrcoef(s[ok], v[ok])[0, 1])
        corrs.append({
            "feature": feat, "n": int(ok.sum()),
            "pearson_r_std_bp_vs_feature": r,
        })
    out["pearson_correlations"] = pd.DataFrame(corrs)
    return out


# --- Axis 7 (spec axis 6): genomic context -------------------------------


def residual_by_genomic_context(
    df: pd.DataFrame, *, ref_positions: np.ndarray, genome_length: int,
) -> dict[str, pd.DataFrame]:
    """Genomic-frame correlates: local density, nearest-neighbor distance,
    strand, genome position. Genome is circular (4.64 Mbp E. coli)."""
    out: dict[str, pd.DataFrame] = {}
    ref = np.sort(ref_positions.astype(np.int64))

    # Strand
    if "ref_strand" in df.columns:
        by_strand = df.groupby(df["ref_strand"].astype("Int8"), observed=True).agg(
            n=("residual_bp", "size"),
            signed_median_bp=("residual_bp", "median"),
            abs_median_bp=("abs_residual_bp", "median"),
            abs_p90_bp=("abs_residual_bp", lambda s: float(s.quantile(0.9))),
        ).reset_index().rename(columns={"ref_strand": "strand"})
        out["by_strand"] = by_strand

    # Distance to nearest neighbor ref site (for each assigned probe's
    # own ref_genomic_pos_bp). Circular-aware.
    positions = df["ref_genomic_pos_bp"].astype(np.int64).to_numpy()
    idx = np.searchsorted(ref, positions)
    # Neighbors on each side, wrapping around
    left_idx = (idx - 1) % len(ref)
    right_idx = idx % len(ref)
    left_pos = ref[left_idx]
    right_pos = ref[right_idx]
    # Circular distance (consider both directions on the ring)
    def _circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.abs(a - b).astype(np.float64)
        return np.minimum(d, genome_length - d)
    # Distance to immediate left / right neighbors.
    d_left = _circular_distance(positions, left_pos)
    d_right = _circular_distance(positions, right_pos)
    # Exclude the probe itself (same position) -- pick the second-nearest.
    d_stacked = np.where(d_left == 0, np.inf, d_left)
    d_stacked2 = np.where(d_right == 0, np.inf, d_right)
    nearest = np.minimum(d_stacked, d_stacked2)
    df_g = df.copy()
    df_g["nearest_neighbor_bp"] = nearest
    bins = [0, 500, 1000, 2000, 5000, 10000, 20000, np.inf]
    labels = ["0-500", "500-1k", "1k-2k", "2k-5k", "5k-10k", "10k-20k", "20k+"]
    df_g["nn_bucket"] = pd.cut(
        df_g["nearest_neighbor_bp"], bins=bins, labels=labels, include_lowest=True,
    )
    by_nn = df_g.groupby("nn_bucket", observed=True).agg(
        n=("residual_bp", "size"),
        signed_median_bp=("residual_bp", "median"),
        abs_median_bp=("abs_residual_bp", "median"),
        abs_p90_bp=("abs_residual_bp", lambda s: float(s.quantile(0.9))),
    ).reset_index()
    out["by_nearest_neighbor_bucket"] = by_nn

    # Position in genome (10% bins of the 4.64 Mbp)
    df_g["genome_pct"] = (df_g["ref_genomic_pos_bp"].astype(np.float64)
                          / genome_length * 100.0)
    df_g["genome_bin"] = pd.cut(
        df_g["genome_pct"], bins=10,
        labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)], include_lowest=True,
    )
    by_gpos = df_g.groupby("genome_bin", observed=True).agg(
        n=("residual_bp", "size"),
        signed_median_bp=("residual_bp", "median"),
        abs_median_bp=("abs_residual_bp", "median"),
        abs_p90_bp=("abs_residual_bp", lambda s: float(s.quantile(0.9))),
    ).reset_index()
    out["by_genome_position"] = by_gpos
    return out

    """Cheap heuristic for "distribution median is far from zero."

    True when |median| exceeds one-sixth of the IQR. A symmetric
    distribution centered at zero has ``|median| ~= 0``; a distribution
    with a shared sign has ``|median|`` at roughly half the IQR.
    The 1/6 cutoff flags meaningful skew without firing on small
    population drift. Exists to trigger "stop and look" — not a
    rigorous test.
    """
    if arr.size < 100:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size < 100:
        return False
    iqr = float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
    if iqr <= 0:
        return False
    median = float(np.median(finite))
    return abs(median) > (iqr / 6.0)
