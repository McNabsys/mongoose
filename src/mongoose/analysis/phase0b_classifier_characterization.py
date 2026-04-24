"""Phase 0b: classifier characterization against reference-map ground truth.

Entry point: :func:`characterize`. See ``docs/plans/phase0b_classifier_spec.md``
for the full spec.

Naming conventions (spec-critical):
- ``is_assigned`` is the *assigns-based* ground truth. Authoritative for
  narrow-ACCEPT probes; structurally zero for narrow-REJECT probes (the
  tautology documented in the report's Method section).
- ``t2d_plausible_match_{tol}`` is a *proximity-based* secondary signal —
  NOT ground truth. It answers: "the classifier rejected a probe whose
  predicted position coincides with a reference site at better-than-chance
  rate, given this molecule's reference-site density." Column names in
  tables that combine a classifier decision with this signal use
  ``rejection_fired_on_t2d_plausible_probe`` (or similar), not anything
  involving "false negative."
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from mongoose.etl.remap_settings import load_remap_settings
from mongoose.io.reference_map import ReferenceMap, load_reference_map


# -- Constants confirmed 2026-04-23 across all 30 short-runs' _remapSettings.
# Runtime asserted in load_reference_settings() below so a future dataset
# with a different configuration fails loudly instead of silently drifting.
MIN_PROBES_REMAPPING: int = 6
INTERVAL_MATCH_TOL_BP: int = 250
PROXIMITY_THRESHOLDS_BP: tuple[int, ...] = (100, 250, 500)

# Minimum assigned-probe count per molecule to fit a stable affine
# molecule-local -> genomic mapping. 2 is mathematically sufficient but
# 3 gives one degree of freedom for residual sanity-check.
MIN_FIT_POINTS: int = 3


# --- Filter cascade -------------------------------------------------------


ANALYSIS_COLUMNS: tuple[str, ...] = (
    "probe_uid",
    "run_id",
    "molecule_uid",
    "probe_idx_in_molecule",
    # Probe features
    "start_ms",
    "duration_ms",
    "center_ms",
    "area_samples_uv",
    "max_amp_uv",
    "attr_bitfield_raw",
    "attr_clean_region",
    "attr_folded_end",
    "attr_folded_start",
    "attr_in_structure",
    "attr_excl_amp_high",
    "attr_excl_width_sp",
    "attr_excl_width_remap",
    "attr_accepted",
    "attr_excl_outside_partial",
    # Molecule features
    "molecule_aligned",
    "molecule_do_not_use",
    "num_probes",
    "translocation_time_ms",
    "molecule_direction",
    "molecule_align_score",
    "molecule_velocity_bp_per_ms",
    "mean_lvl1_mv",
    # Labels
    "ref_idx",
    "ref_genomic_pos_bp",
    "is_assigned",
    "t2d_predicted_bp_pos",
    # Derived neighborhood
    "probe_local_density",
    "prev_probe_gap_ms",
    "next_probe_gap_ms",
    # Stratification metadata
    "concentration_group",
    "biochem_flagged_good",
    "excel_instrument",
    "excel_snr",
)


@dataclass(frozen=True)
class FilterCascade:
    total: int
    after_aligned: int
    after_do_not_use: int
    after_min_probes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_rows": self.total,
            "after_molecule_aligned": self.after_aligned,
            "after_not_do_not_use": self.after_do_not_use,
            "after_min_probes": self.after_min_probes,
        }


def load_analysis_eligible(
    probe_table_path: Path | str,
    *,
    columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, FilterCascade]:
    """Load the Phase 0 probe table and apply spec §41-47 filter cascade.

    The three filters collapse to one on the current dataset (the latter
    two are no-ops because aligned molecules already satisfy them), but
    we run all three so the cascade is visible in the report and future
    datasets with different properties fail informatively.
    """
    cols = list(columns) if columns is not None else list(ANALYSIS_COLUMNS)
    dataset = ds.dataset(Path(probe_table_path), format="parquet")
    total = dataset.count_rows()

    aligned_filter = ds.field("molecule_aligned") == True  # noqa: E712
    after_aligned = dataset.count_rows(filter=aligned_filter)

    not_dnu_filter = aligned_filter & (ds.field("molecule_do_not_use") == False)  # noqa: E712
    after_dnu = dataset.count_rows(filter=not_dnu_filter)

    min_probes_filter = not_dnu_filter & (
        ds.field("num_probes") >= MIN_PROBES_REMAPPING
    )
    after_min = dataset.count_rows(filter=min_probes_filter)

    df = dataset.to_table(columns=cols, filter=min_probes_filter).to_pandas()
    cascade = FilterCascade(
        total=total,
        after_aligned=after_aligned,
        after_do_not_use=after_dnu,
        after_min_probes=after_min,
    )
    return df, cascade


def load_reference_settings(example_remap_settings: Path) -> dict[str, int]:
    """Load and runtime-assert the two spec-critical remap constants.

    ``example_remap_settings`` is any one run's ``_remapSettings.txt`` —
    we asserted 2026-04-23 that ``min_probes_remapping`` and
    ``interval_match_tol_bp`` are constant across all 30 runs.
    """
    s = load_remap_settings(example_remap_settings)
    mp = s.get_int("min_probes_remapping")
    tol = s.get_int("interval_match_tol_bp")
    if mp != MIN_PROBES_REMAPPING:
        raise ValueError(
            f"{example_remap_settings}: min_probes_remapping={mp}, "
            f"expected {MIN_PROBES_REMAPPING} (hardcoded 2026-04-23 after "
            f"confirming constancy across 30 short-runs). Update the "
            f"constant if the dataset has changed."
        )
    if tol != INTERVAL_MATCH_TOL_BP:
        raise ValueError(
            f"{example_remap_settings}: interval_match_tol_bp={tol}, "
            f"expected {INTERVAL_MATCH_TOL_BP}. Same provenance as above."
        )
    return {"min_probes_remapping": mp, "interval_match_tol_bp": tol}


def load_shared_reference_map(
    refmap_paths: list[Path] | Path,
) -> ReferenceMap:
    """Load a single referenceMap.txt and assert identity across inputs.

    All 30 short-runs' referenceMaps hash-identically, so any one is
    representative. If ``refmap_paths`` is a list, we load all, hash
    their (positions, strands) bytes, and require the set has size 1.
    """
    paths = [Path(refmap_paths)] if isinstance(refmap_paths, Path) else list(refmap_paths)
    if not paths:
        raise ValueError("refmap_paths must contain at least one path")
    hashes: set[str] = set()
    refmap: ReferenceMap | None = None
    for p in paths:
        rm = load_reference_map(p)
        h = hashlib.md5(rm.probe_positions.tobytes() + rm.strands.tobytes()).hexdigest()
        hashes.add(h)
        refmap = rm
    if len(hashes) > 1:
        raise ValueError(
            f"Reference maps are not identical across the {len(paths)} "
            f"supplied paths; distinct hashes: {sorted(hashes)}. "
            f"The analysis assumes a shared map — revisit before proceeding."
        )
    assert refmap is not None  # mypy
    return refmap


# --- Per-molecule affine fit: t2d (molecule-local bp) -> genomic bp -------


def fit_per_molecule_affine(df: pd.DataFrame) -> pd.DataFrame:
    """For each aligned molecule, fit ``ref_bp ~= slope * t2d_bp + intercept``.

    Uses ordinary-least-squares on the molecule's assigned probes only.
    Molecules with fewer than :data:`MIN_FIT_POINTS` assigned probes
    receive NaN slope/intercept — their rejected-probe rows will have
    ``t2d_plausible_match`` set to NaN downstream.

    Returns a DataFrame keyed by ``(run_id, molecule_uid)`` with
    columns ``slope``, ``intercept``, ``n_fit_points``, and the bounds
    ``t2d_bp_min`` / ``t2d_bp_max`` needed for null-rate computation.
    """
    mask = df["is_assigned"] & df["t2d_predicted_bp_pos"].notna()
    x = df.loc[mask, "t2d_predicted_bp_pos"].astype(np.float64).to_numpy()
    y = df.loc[mask, "ref_genomic_pos_bp"].astype(np.float64).to_numpy()
    assigned = pd.DataFrame({
        "run_id": df.loc[mask, "run_id"].to_numpy(),
        "molecule_uid": df.loc[mask, "molecule_uid"].to_numpy(),
        "x": x,
        "y": y,
        "xx": x * x,
        "xy": x * y,
    })
    # Closed-form OLS per group via pre-computed sums — avoids per-group
    # lambdas (the slow path in the first implementation).
    agg = assigned.groupby(["run_id", "molecule_uid"], sort=False).agg(
        n=("x", "size"),
        sum_x=("x", "sum"),
        sum_y=("y", "sum"),
        sum_xx=("xx", "sum"),
        sum_xy=("xy", "sum"),
        x_min=("x", "min"),
        x_max=("x", "max"),
    )
    n = agg["n"].to_numpy(dtype=np.float64)
    sum_x = agg["sum_x"].to_numpy(dtype=np.float64)
    sum_y = agg["sum_y"].to_numpy(dtype=np.float64)
    sum_xx = agg["sum_xx"].to_numpy(dtype=np.float64)
    sum_xy = agg["sum_xy"].to_numpy(dtype=np.float64)

    denom = n * sum_xx - sum_x ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denom > 0, (n * sum_xy - sum_x * sum_y) / denom, np.nan)
        intercept = np.where(denom > 0, (sum_y - slope * sum_x) / n, np.nan)
    agg["slope"] = slope
    agg["intercept"] = intercept

    # Molecules below the fit-point threshold: null out the params.
    below_min = agg["n"] < MIN_FIT_POINTS
    agg.loc[below_min, ["slope", "intercept"]] = np.nan

    agg = agg.rename(
        columns={"n": "n_fit_points", "x_min": "t2d_bp_min", "x_max": "t2d_bp_max"}
    )
    return agg.reset_index()[
        [
            "run_id",
            "molecule_uid",
            "n_fit_points",
            "slope",
            "intercept",
            "t2d_bp_min",
            "t2d_bp_max",
        ]
    ]


def predict_genomic_positions(
    df: pd.DataFrame, fits: pd.DataFrame
) -> pd.DataFrame:
    """Attach per-probe ``predicted_genomic_bp`` by applying the molecule's fit.

    NaN for molecules without a fit (too few assigned probes) and for
    probes with ``t2d_predicted_bp_pos`` NaN.
    """
    merged = df.merge(
        fits[["run_id", "molecule_uid", "slope", "intercept"]],
        on=["run_id", "molecule_uid"],
        how="left",
    )
    merged["predicted_genomic_bp"] = (
        merged["slope"] * merged["t2d_predicted_bp_pos"].astype(np.float64)
        + merged["intercept"]
    )
    return merged.drop(columns=["slope", "intercept"])


# --- Proximity oracle -----------------------------------------------------


def nearest_reference_distance(
    predicted_positions: np.ndarray, refmap: ReferenceMap
) -> np.ndarray:
    """|predicted - nearest ref site| in bp. NaN passes through."""
    ref = refmap.probe_positions.astype(np.int64)
    predicted = np.asarray(predicted_positions, dtype=np.float64)
    out = np.full(predicted.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(predicted)
    if not finite.any():
        return out
    # bisect on a sorted reference array -> neighbor indices.
    valid_pred = predicted[finite]
    idx_right = np.searchsorted(ref, valid_pred)
    left = np.clip(idx_right - 1, 0, len(ref) - 1)
    right = np.clip(idx_right, 0, len(ref) - 1)
    d_left = np.abs(valid_pred - ref[left].astype(np.float64))
    d_right = np.abs(ref[right].astype(np.float64) - valid_pred)
    out[finite] = np.minimum(d_left, d_right)
    return out


def compute_t2d_plausible_match(
    df: pd.DataFrame,
    refmap: ReferenceMap,
    *,
    thresholds_bp: Iterable[int] = PROXIMITY_THRESHOLDS_BP,
) -> pd.DataFrame:
    """Attach ``t2d_plausible_match_{tol}`` boolean columns (one per tol).

    Requires ``predicted_genomic_bp`` to already be on the DataFrame
    (via :func:`predict_genomic_positions`). Probes on molecules
    without a fit get NaN (stored as pandas nullable boolean).
    """
    if "predicted_genomic_bp" not in df.columns:
        raise KeyError(
            "compute_t2d_plausible_match requires predicted_genomic_bp; "
            "call predict_genomic_positions first."
        )
    predicted = df["predicted_genomic_bp"].to_numpy(dtype=np.float64)
    dist = nearest_reference_distance(predicted, refmap)
    df = df.copy()
    df["nearest_ref_dist_bp"] = dist
    for tol in thresholds_bp:
        col = f"t2d_plausible_match_{tol}"
        mask = np.where(np.isfinite(dist), dist <= float(tol), np.nan)
        # pandas nullable boolean so "no fit" stays NA rather than False.
        df[col] = pd.array(
            np.where(np.isnan(mask), pd.NA, mask.astype(bool)),
            dtype="boolean",
        )
    return df


# --- Null-rate guard ------------------------------------------------------


def compute_null_rates(
    fits: pd.DataFrame,
    refmap: ReferenceMap,
    *,
    thresholds_bp: Iterable[int] = PROXIMITY_THRESHOLDS_BP,
) -> pd.DataFrame:
    """Per-molecule base-rate: probability that a random predicted position
    on [y_min, y_max] lands within ``tol`` of some reference site.

    Framed per spec: ``t2d_plausible_match`` on a molecule only carries
    lift over random *above this rate*.
    """
    ref = refmap.probe_positions.astype(np.int64)
    out = fits[["run_id", "molecule_uid", "slope", "intercept",
                "t2d_bp_min", "t2d_bp_max", "n_fit_points"]].copy()

    # Molecule-local → genomic range. Direction-sign handled via min/max.
    y_at_xmin = out["slope"] * out["t2d_bp_min"] + out["intercept"]
    y_at_xmax = out["slope"] * out["t2d_bp_max"] + out["intercept"]
    out["genomic_bp_min"] = np.minimum(y_at_xmin, y_at_xmax)
    out["genomic_bp_max"] = np.maximum(y_at_xmin, y_at_xmax)

    for tol in thresholds_bp:
        col = f"null_rate_{tol}"
        out[col] = _null_rates_for_tolerance(
            out["genomic_bp_min"].to_numpy(),
            out["genomic_bp_max"].to_numpy(),
            ref,
            float(tol),
        )
    return out


# --- Agreement between assigns oracle and proximity signal ---------------


def compute_agreement_matrix(
    df: pd.DataFrame,
    *,
    tolerance_bp: int,
) -> dict[str, dict]:
    """is_assigned (assigns oracle) vs t2d_plausible_match_{tol} (proximity).

    Spec §refinement-2: this is the calibration pass that frames every
    proximity-oracle claim downstream. We compute the matrix on three
    subsets, each informative for a different reason:

    - ``narrow_accept``: the spec's requested subset. Includes
      narrow-ACCEPT probes on aligned molecules regardless of whether
      the aligner offered them (the 6.1M "broad-REJECT tail" cases have
      ``is_assigned=False`` by construction, which biases the matrix
      toward the "proximity TRUE + assigns FALSE" cell).
    - ``broad_accept``: restricted to probes the aligner actually had a
      chance to place (``ref_idx.notna()``). This is the clean
      agreement signal, since both labels are meaningfully defined.
    - ``narrow_accept_minus_broad``: the tail cases only. Tells us what
      fraction of "offered by SP but dropped before assignment" probes
      look plausible by proximity -- an upper bound on FN cost of
      whatever downstream filter dropped them.

    Returns a dict keyed by subset name, with cells ``tt, tf, ft, ff``
    (is_assigned × t2d_plausible_match, T=assigned first, T=plausible
    second), plus derived metrics ``agreement_rate``, ``cohen_kappa``,
    ``mcc``, ``n``.
    """
    col = f"t2d_plausible_match_{tolerance_bp}"
    if col not in df.columns:
        raise KeyError(
            f"{col} not in df; call compute_t2d_plausible_match first."
        )

    narrow = df["attr_accepted"].astype(bool)
    broad = df["ref_idx"].notna()
    subsets = {
        "narrow_accept": narrow,
        "broad_accept": narrow & broad,
        "narrow_accept_minus_broad": narrow & ~broad,
    }

    results: dict[str, dict] = {}
    for name, mask in subsets.items():
        sub = df.loc[mask, ["is_assigned", col]].copy()
        sub = sub[sub[col].notna()]  # drop rows with no fit
        y_assigned = sub["is_assigned"].astype(bool).to_numpy()
        y_plausible = sub[col].astype("boolean").astype(bool).to_numpy()
        tt = int(np.sum(y_assigned & y_plausible))
        tf = int(np.sum(y_assigned & ~y_plausible))
        ft = int(np.sum(~y_assigned & y_plausible))
        ff = int(np.sum(~y_assigned & ~y_plausible))
        total = tt + tf + ft + ff
        results[name] = {
            "tt": tt, "tf": tf, "ft": ft, "ff": ff, "n": total,
            "agreement_rate": (tt + ff) / total if total else float("nan"),
            "cohen_kappa": _cohen_kappa(tt, tf, ft, ff),
            "mcc": _mcc(tt, tf, ft, ff),
        }
    return results


def _cohen_kappa(tt: int, tf: int, ft: int, ff: int) -> float:
    n = tt + tf + ft + ff
    if n == 0:
        return float("nan")
    p_o = (tt + ff) / n
    p_yes_assigned = (tt + tf) / n
    p_yes_plausible = (tt + ft) / n
    p_e = p_yes_assigned * p_yes_plausible + (1 - p_yes_assigned) * (1 - p_yes_plausible)
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def _mcc(tt: int, tf: int, ft: int, ff: int) -> float:
    # Matthews correlation coefficient; is_assigned=T treated as +.
    # Cast to float before sqrt -- Python ints can overflow numpy ufuncs
    # at the scales we hit (35M rows -> denom_sq up to ~1e30).
    num = float(tt) * float(ff) - float(tf) * float(ft)
    denom_sq = (
        float(tt + tf) * float(tt + ft) * float(ff + tf) * float(ff + ft)
    )
    if denom_sq <= 0.0:
        return float("nan")
    return num / float(np.sqrt(denom_sq))


# --- Feature-space envelope (Mahalanobis against known-good) -------------
#
# Rationale (see design notes 2026-04-23 following the T2D-proximity
# structural failure):
#
# The assigns-based oracle is tautological for narrow-REJECT probes
# (is_assigned = True implies attr_accepted = True). A T2D-proximity
# secondary signal is too noisy to repair the tautology: per-probe
# T2D error in the genomic frame has p50 = 532 bp, p90 = 5.9 kbp,
# far wider than any usefully-tight tolerance. So we abandon
# "would this rejected probe have been assigned" questions and instead
# ask the weaker, cleanly answerable question: "does this rejected
# probe look, featurewise, like probes we know are real bound tags?"
#
# The column name is ``inside_known_good_envelope``, not anything with
# "false negative" in it. The report language throughout is "looks
# featurewise similar to known bound tags" -- not "is a bound tag."

ENVELOPE_FEATURES: tuple[str, ...] = (
    "duration_scaled",    # duration_ms * molecule_velocity_bp_per_ms
    "amplitude_scaled",   # max_amp_uv / mean_lvl1_mv  (dimensionless ratio)
    "area_scaled",        # area_samples_uv / mean_lvl1_mv
    "density_scaled",     # probe_local_density (float copy for dtype uniformity)
)


@dataclass(frozen=True)
class EnvelopeModel:
    """Mahalanobis envelope fit on a known-good probe set."""

    features: tuple[str, ...]
    mean: np.ndarray       # shape (n_features,)
    covariance: np.ndarray # shape (n_features, n_features)
    inv_covariance: np.ndarray
    n_known_good: int
    align_score_threshold: float
    center_frac_lo: float
    center_frac_hi: float

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Per-row Mahalanobis distance. NaN rows return NaN."""
        diff = X - self.mean
        row_ok = ~np.any(np.isnan(diff), axis=1)
        dist_sq = np.full(diff.shape[0], np.nan, dtype=np.float64)
        if row_ok.any():
            dist_sq[row_ok] = np.einsum(
                "ni,ij,nj->n", diff[row_ok], self.inv_covariance, diff[row_ok]
            )
        return np.sqrt(np.maximum(dist_sq, 0.0))


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the scaled-feature columns that feed the Mahalanobis envelope.

    Scaling choices per the design refinements:
    - ``duration_scaled = duration_ms * molecule_velocity_bp_per_ms``
      -- probe width expressed in bp-equivalent; removes the first-order
      molecule-speed effect.
    - ``amplitude_scaled = max_amp_uv / mean_lvl1_mv`` -- relative
      amplitude against the molecule's baseline current; dimensionless
      (units differ between numerator and denominator, but the ratio is
      stable within a molecule and removes the baseline drift).
    - ``area_scaled = area_samples_uv / mean_lvl1_mv`` -- same principle.
    - ``density_scaled = probe_local_density`` -- already local, cast to
      float for uniform dtype in the feature matrix.

    Guards against division by zero on mean_lvl1_mv with NaN propagation.
    """
    df = df.copy()
    lvl1 = df["mean_lvl1_mv"].astype(np.float64)
    lvl1 = lvl1.where(lvl1 > 0, np.nan)
    df["duration_scaled"] = (
        df["duration_ms"].astype(np.float64)
        * df["molecule_velocity_bp_per_ms"].astype(np.float64)
    )
    df["amplitude_scaled"] = df["max_amp_uv"].astype(np.float64) / lvl1
    df["area_scaled"] = df["area_samples_uv"].astype(np.float64) / lvl1
    df["density_scaled"] = df["probe_local_density"].astype(np.float64)
    return df


def build_known_good_mask(
    df: pd.DataFrame,
    *,
    align_score_pct: float = 75.0,
    center_frac_lo: float = 0.1,
    center_frac_hi: float = 0.9,
    features: tuple[str, ...] = ENVELOPE_FEATURES,
) -> tuple[pd.Series, float]:
    """High-confidence bound-tag mask for envelope construction.

    Definition: ``is_assigned=True`` AND molecule alignment score above
    the supplied percentile (computed over the assigned population) AND
    probe center inside ``[center_frac_lo, center_frac_hi]`` of the
    translocation (avoid molecule ends, where T2D is most unstable and
    folded-region artifacts concentrate). Rows with any NaN in the
    envelope features are excluded from the known-good set.

    Returns the boolean mask and the computed align-score threshold.
    """
    aligned = df[df["is_assigned"]]
    align_score_threshold = float(
        aligned["molecule_align_score"].quantile(align_score_pct / 100.0)
    )
    pos_frac = df["center_ms"].astype(np.float64) / df[
        "translocation_time_ms"
    ].astype(np.float64)
    mask = (
        df["is_assigned"]
        & (df["molecule_align_score"] >= align_score_threshold)
        & (pos_frac >= center_frac_lo)
        & (pos_frac <= center_frac_hi)
        & df[list(features)].notna().all(axis=1)
    )
    return mask, align_score_threshold


def fit_envelope(
    df: pd.DataFrame,
    known_good_mask: pd.Series,
    *,
    features: tuple[str, ...] = ENVELOPE_FEATURES,
    align_score_threshold: float = float("nan"),
    center_frac_lo: float = 0.1,
    center_frac_hi: float = 0.9,
) -> EnvelopeModel:
    """Fit the mean + covariance of the Mahalanobis envelope."""
    X_good = df.loc[known_good_mask, list(features)].to_numpy(dtype=np.float64)
    X_good = X_good[~np.any(np.isnan(X_good), axis=1)]
    if X_good.shape[0] < len(features) + 1:
        raise ValueError(
            f"known-good set too small ({X_good.shape[0]}) to fit a "
            f"covariance for {len(features)} features."
        )
    mu = X_good.mean(axis=0)
    cov = np.cov(X_good, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    return EnvelopeModel(
        features=tuple(features),
        mean=mu,
        covariance=cov,
        inv_covariance=inv_cov,
        n_known_good=int(X_good.shape[0]),
        align_score_threshold=align_score_threshold,
        center_frac_lo=center_frac_lo,
        center_frac_hi=center_frac_hi,
    )


def apply_envelope(
    df: pd.DataFrame,
    envelope: EnvelopeModel,
) -> pd.DataFrame:
    """Attach ``mahal_envelope`` (float) and ``inside_known_good_envelope``
    (bool, nullable) columns."""
    df = df.copy()
    X = df[list(envelope.features)].to_numpy(dtype=np.float64)
    df["mahal_envelope"] = envelope.mahalanobis(X)
    return df


def tp_sanity_report(df: pd.DataFrame) -> dict:
    """Return headline stats on the TP Mahalanobis distribution.

    Spec-critical: median should be close to sqrt(chi2(k).ppf(0.5))
    where k = len(features). If the observed distribution is far
    heavier-tailed than chi-square, the envelope concept is broken
    on this feature set and downstream rejection-breakdown analyses
    must be reconsidered. The value returned here is what the report
    cites to justify the envelope's use.
    """
    tp = df.loc[
        df["attr_accepted"] & df["is_assigned"] & df["mahal_envelope"].notna(),
        "mahal_envelope",
    ]
    return {
        "n_tp": int(tp.size),
        "median": float(tp.quantile(0.5)),
        "p75": float(tp.quantile(0.75)),
        "p90": float(tp.quantile(0.9)),
        "p95": float(tp.quantile(0.95)),
        "p99": float(tp.quantile(0.99)),
        "mean": float(tp.mean()),
        "frac_inside_3": float((tp <= 3.0).mean()),
        "frac_inside_4": float((tp <= 4.0).mean()),
        "frac_inside_5": float((tp <= 5.0).mean()),
    }


def _null_rates_for_tolerance(
    lo_arr: np.ndarray, hi_arr: np.ndarray, ref: np.ndarray, tol: float,
) -> np.ndarray:
    """Length-proportional coverage of [lo-tol, hi+tol] hit windows,
    clipped to [lo, hi]. Vectorized over molecules."""
    out = np.full(lo_arr.shape, np.nan, dtype=np.float64)
    ref_f = ref.astype(np.float64)
    for i in range(lo_arr.size):
        lo = lo_arr[i]
        hi = hi_arr[i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        span = hi - lo
        # Reference sites whose ±tol window overlaps [lo, hi].
        left = bisect_left(ref_f, lo - tol)
        right = bisect_right(ref_f, hi + tol)
        sel = ref_f[left:right]
        if sel.size == 0:
            out[i] = 0.0
            continue
        # Windows: [r - tol, r + tol], clipped to [lo, hi]. Merge overlaps.
        starts = np.maximum(sel - tol, lo)
        ends = np.minimum(sel + tol, hi)
        # Already sorted because ref is sorted.
        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]
        # Merge adjacents.
        merged_total = 0.0
        cur_s, cur_e = starts[0], ends[0]
        for s, e in zip(starts[1:], ends[1:]):
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged_total += cur_e - cur_s
                cur_s, cur_e = s, e
        merged_total += cur_e - cur_s
        out[i] = merged_total / span
    return out
