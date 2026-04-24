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


def _looks_skewed_from_zero(arr: np.ndarray) -> bool:
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
