"""Per-probe residual dataset for Direction C.

Wraps the long-format DataFrame produced by
:func:`mongoose.etl.reads_maps_table.build_residual_table` as a PyTorch
:class:`Dataset`. Each item is one accepted probe with:

* a fixed-shape feature vector (``[D]``) -- per-probe, plus broadcast
  per-molecule and aggregate features.
* a scalar regression target: ``residual_bp`` (production's per-probe bp
  shift = post_position_bp - pre_position_bp).
* a per-probe weight (``1.0`` by default; reserved for class-balancing or
  importance sampling extensions).

Filtering: by default we keep only probes where ``accepted == True``.
Production excluded probes (PF-rejected, outside-partial, etc.) have a
post-position but it isn't a meaningful prediction target.

Feature schema (deterministic order):
    Per-probe continuous (5):
        bp_position_frac              pre_position / molecule_length
        frac_position_in_molecule     probe_idx / (num_probes - 1)
        log_prev_interval_bp          log1p(max(prev_interval_bp, 0))
        log_next_interval_bp          log1p(max(next_interval_bp, 0))
        normalized_width              width_bp / 831 (analytical expected)
    Per-probe binary (8):
        has_prev_interval, has_next_interval
        in_clean_region, in_structure, in_folded_start, in_folded_end
        excluded_width_sp, excluded_width_rm
    Per-molecule continuous (5):
        log_molecule_length_bp
        log_num_probes
        num_probes_accepted_frac      num_probes_accepted / num_probes
        log_mean_probe_width
        log_median_probe_width
    Per-molecule categorical-encoded (3):
        direction_forward, direction_reverse, direction_unknown (one-hot)
    Per-molecule discrete (1):
        length_group_bin              integer in [-1, 15], embedded raw

Total: 22 features.

When the model architecture changes the schema must change in lockstep
with :class:`mongoose.model.residual_mlp.ResidualMLP`'s ``input_dim``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROBE_WIDTH_NORM_BP = 831.0  # analytical expected width (FOM, default channel)
FEATURE_DIM = 22


def add_molecule_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-molecule aggregate columns used as broadcast features.

    Adds (in place via assign):
        mean_probe_width_bp      mean of width_bp across the molecule's probes
        median_probe_width_bp    median of width_bp

    Aggregates use ALL probes (accepted + rejected) to capture the
    full molecule probe-width signature. The head-dive Method 1
    severity computation in production uses all probes to fit
    against the median curve.

    Args:
        df: A residual table from :func:`build_residual_table`.

    Returns:
        New DataFrame with two extra columns; original input is not
        mutated.
    """
    if df.empty:
        out = df.copy()
        out["mean_probe_width_bp"] = 0.0
        out["median_probe_width_bp"] = 0.0
        return out

    by_uid = df.groupby("uid", sort=False)["width_bp"]
    aggs = pd.DataFrame(
        {
            "mean_probe_width_bp": by_uid.transform("mean").astype(float),
            "median_probe_width_bp": by_uid.transform("median").astype(float),
        },
        index=df.index,
    )
    return pd.concat([df, aggs], axis=1)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Convert a residual-table DataFrame to a fixed-shape feature array.

    Required columns: see module docstring. The aggregate columns
    (mean_probe_width_bp, median_probe_width_bp) must be present --
    call :func:`add_molecule_aggregates` first.

    Args:
        df: Residual table with aggregates joined.

    Returns:
        ``np.ndarray`` of shape ``(N, FEATURE_DIM)`` and dtype float32.
    """
    n = len(df)
    if n == 0:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    out = np.empty((n, FEATURE_DIM), dtype=np.float32)

    # Per-probe continuous (5)
    out[:, 0] = df["bp_position_frac"].to_numpy(dtype=np.float32)
    out[:, 1] = df["frac_position_in_molecule"].to_numpy(dtype=np.float32)
    out[:, 2] = np.log1p(np.clip(df["prev_interval_bp"].to_numpy(dtype=np.float32), 0, None))
    out[:, 3] = np.log1p(np.clip(df["next_interval_bp"].to_numpy(dtype=np.float32), 0, None))
    out[:, 4] = df["width_bp"].to_numpy(dtype=np.float32) / PROBE_WIDTH_NORM_BP

    # Per-probe binary (8)
    out[:, 5] = (df["prev_interval_bp"] != -1).to_numpy(dtype=np.float32)
    out[:, 6] = (df["next_interval_bp"] != -1).to_numpy(dtype=np.float32)
    out[:, 7] = df["in_clean_region"].to_numpy(dtype=np.float32)
    out[:, 8] = df["in_structure"].to_numpy(dtype=np.float32)
    out[:, 9] = df["in_folded_start"].to_numpy(dtype=np.float32)
    out[:, 10] = df["in_folded_end"].to_numpy(dtype=np.float32)
    out[:, 11] = df["excluded_width_sp"].to_numpy(dtype=np.float32)
    out[:, 12] = df["excluded_width_rm"].to_numpy(dtype=np.float32)

    # Per-molecule continuous (5)
    out[:, 13] = np.log(np.clip(df["molecule_length_bp"].to_numpy(dtype=np.float32), 1, None))
    out[:, 14] = np.log(np.clip(df["num_probes"].to_numpy(dtype=np.float32), 1, None))
    n_probes = df["num_probes"].to_numpy(dtype=np.float32)
    out[:, 15] = df["num_probes_accepted"].to_numpy(dtype=np.float32) / np.clip(n_probes, 1, None)
    mean_w = df["mean_probe_width_bp"].to_numpy(dtype=np.float32)
    median_w = df["median_probe_width_bp"].to_numpy(dtype=np.float32)
    out[:, 16] = np.log(np.clip(mean_w, 1, None))
    out[:, 17] = np.log(np.clip(median_w, 1, None))

    # Direction one-hot (3): forward (+1), reverse (-1), unknown (0)
    direction = df["direction"].to_numpy(dtype=np.int32)
    out[:, 18] = (direction == 1).astype(np.float32)   # forward
    out[:, 19] = (direction == -1).astype(np.float32)  # reverse
    out[:, 20] = (direction == 0).astype(np.float32)   # unknown

    # Length-group bin (1) -- pass through, models can embed if needed
    out[:, 21] = df["length_group_bin"].to_numpy(dtype=np.float32)

    return out


@dataclass(frozen=True)
class ResidualDatasetSplit:
    """Index split for a residual dataset (train / val)."""

    train_indices: np.ndarray  # int64
    val_indices: np.ndarray  # int64


def make_split(
    n: int,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> ResidualDatasetSplit:
    """Build a deterministic train/val index split.

    Per-probe random split; for stratification by molecule, group_by
    uid externally and call this on group ids instead.

    Args:
        n: Total number of samples.
        val_fraction: Fraction in ``[0, 1)`` allocated to val.
        seed: RNG seed for reproducibility.

    Returns:
        :class:`ResidualDatasetSplit` with int64 index arrays.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction!r}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * (1.0 - val_fraction)))
    return ResidualDatasetSplit(
        train_indices=perm[:cut].astype(np.int64),
        val_indices=perm[cut:].astype(np.int64),
    )


class ResidualDataset(Dataset):
    """In-memory per-probe residual dataset.

    Wraps a residual-table DataFrame as a PyTorch Dataset. Each item is
    a tuple ``(features, target)`` where features is a ``[FEATURE_DIM]``
    float32 tensor and target is a scalar float32 tensor.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        accepted_only: bool = True,
        compute_aggregates: bool = True,
        target_column: str = "residual_bp",
        require_reference: bool = False,
    ) -> None:
        """
        Args:
            df: Residual table from
                :func:`mongoose.etl.reads_maps_table.build_residual_table`.
            accepted_only: When True (default), retain only probes with
                ``accepted == True``. Production-rejected probes have a
                post-position but it isn't a meaningful prediction target.
            compute_aggregates: When True (default), call
                :func:`add_molecule_aggregates` first. Pass False if
                ``df`` already has the aggregate columns.
            target_column: Which column to use as the regression target.
                ``"residual_bp"`` (default) trains the model to mimic
                production's correction (post - pre, ceiling = production).
                ``"ref_anchored_residual_bp"`` trains against the genome
                directly (anchored at the first matched probe), which has
                no a-priori ceiling and could in principle beat production.
            require_reference: When True, additionally drop probes
                without a reference match (``has_reference == False``).
                Set this to True when ``target_column ==
                "ref_anchored_residual_bp"`` because the target is
                undefined for unmatched probes.
        """
        if compute_aggregates and "mean_probe_width_bp" not in df.columns:
            df = add_molecule_aggregates(df)
        elif "mean_probe_width_bp" not in df.columns:
            raise ValueError(
                "df missing aggregate columns; call add_molecule_aggregates "
                "or pass compute_aggregates=True."
            )

        if accepted_only:
            df = df[df["accepted"]].reset_index(drop=True)
        if require_reference:
            df = df[df["has_reference"]].reset_index(drop=True)

        if target_column not in df.columns:
            raise ValueError(
                f"target_column={target_column!r} not in df. "
                f"Available columns: {sorted(df.columns)}"
            )

        self.features = torch.from_numpy(extract_features(df))
        # ``torch.tensor`` always copies, sidestepping pandas' non-writable
        # views (which torch.from_numpy warns about).
        self.target = torch.tensor(
            df[target_column].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.target_column = target_column
        # Keep uid/run_id around for downstream stratified eval.
        self.uid = torch.tensor(
            df["uid"].to_numpy(dtype=np.int64),
            dtype=torch.int64,
        )
        self.run_id_index = self._build_run_id_index(df)

    @staticmethod
    def _build_run_id_index(df: pd.DataFrame) -> torch.Tensor:
        if "run_id" not in df.columns or len(df) == 0:
            return torch.zeros(len(df), dtype=torch.int64)
        codes, _ = pd.factorize(df["run_id"])
        return torch.tensor(codes.astype(np.int64), dtype=torch.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.target[idx]
