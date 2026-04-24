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
