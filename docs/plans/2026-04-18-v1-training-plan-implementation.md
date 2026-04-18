# V1 Training Plan — 48-Hour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a trained V1 T2D U-Net model evaluated with quantitative peak-match F1 and visual inspection, first on a single preprocessed E. coli run and then scaled to a multi-run (27 train / 3 held-out test) configuration, within 48 hours on an RTX A4500 workstation.

**Architecture:** Targeted changes to `CombinedLoss` (blend floor + static loss-scale divisors) to prevent the heatmap-collapse + loss-imbalance issues seen in prior runs. New, independently testable peak-match evaluation library in `src/mongoose/inference/peak_match.py` plus three thin CLI scripts (`overfit_one_batch.py`, `evaluate_peak_match.py`, `visualize_predictions.py`). The training flow is gated: an overfit-one-batch sanity check must pass before any long smoke, and every long run has an abort criterion tied to early-epoch F1.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu128 (bf16 autocast, fp32 criterion), NumPy 2.4, `scipy.optimize.linear_sum_assignment` for Hungarian peak-matching, matplotlib for visualization, pytest. Branch: `training/v1-recipe-design` (off `perf/criterion-amp-fixes`). All changes committed before any long training run.

**Spec reference:** [docs/plans/2026-04-18-v1-training-plan-design.md](2026-04-18-v1-training-plan-design.md)

**Working directory:** `C:/git/mongoose`. All commands assume `.venv/Scripts/python.exe` interpreter.

---

## Task 1: Add scipy to Dev Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current dev-deps section**

Run: `head -25 pyproject.toml`
Note the current `[project.optional-dependencies] dev` entries.

- [ ] **Step 2: Add scipy**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "openpyxl>=3.1",
    "scipy>=1.13",
    "matplotlib>=3.8",
]
```

- [ ] **Step 3: Install**

Run: `.venv/Scripts/python.exe -m pip install -e ".[dev]"`
Expected: "Successfully installed scipy-… matplotlib-…"

- [ ] **Step 4: Verify imports**

Run:
```
.venv/Scripts/python.exe -c "from scipy.optimize import linear_sum_assignment; import matplotlib; print('scipy+matplotlib ok')"
```
Expected output: `scipy+matplotlib ok`

- [ ] **Step 5: Full test suite still passes**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `199 passed` (no regressions from new deps).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "build: add scipy and matplotlib to dev dependencies"
```

---

## Task 2: CombinedLoss `min_blend` Floor

**Files:**
- Modify: `src/mongoose/losses/combined.py` (constructor + `set_epoch`)
- Modify: `tests/test_losses/test_losses.py` (add two tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_losses/test_losses.py`:

```python
def test_combined_loss_blend_floor_holds_at_late_epoch():
    """min_blend=0.1 means focal loss never fully disappears."""
    criterion = CombinedLoss(
        warmstart_epochs=5,
        warmstart_fade_epochs=2,
        min_blend=0.1,
    )
    criterion.set_epoch(100)
    assert criterion._warmstart_blend == 0.1


def test_combined_loss_blend_floor_defaults_to_zero():
    """Default min_blend=0.0 preserves existing post-fade behavior (blend=0)."""
    criterion = CombinedLoss(
        warmstart_epochs=5,
        warmstart_fade_epochs=2,
    )
    criterion.set_epoch(100)
    assert criterion._warmstart_blend == 0.0
```

- [ ] **Step 2: Run tests — expect failures**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_losses.py::test_combined_loss_blend_floor_holds_at_late_epoch tests/test_losses/test_losses.py::test_combined_loss_blend_floor_defaults_to_zero -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'min_blend'`

- [ ] **Step 3: Add `min_blend` parameter to `CombinedLoss.__init__`**

In `src/mongoose/losses/combined.py`, modify the constructor signature (after existing args, before `sample_period_ms`):

```python
    def __init__(
        self,
        lambda_bp: float = 1.0,
        lambda_vel: float = 1.0,
        lambda_count: float = 1.0,
        warmup_epochs: int = 5,
        warmstart_epochs: int = 5,
        warmstart_fade_epochs: int = 2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        huber_delta_bp: float = 500.0,
        softdtw_gamma: float = 0.1,
        peakiness_window: int = 20,
        nms_threshold: float = 0.3,
        tag_width_bp: float = 511.0,
        sample_period_ms: float = 0.025,
        min_blend: float = 0.0,
    ) -> None:
```

And store:

```python
        self.min_blend = float(min_blend)
```

(Place immediately after the other scalar fields near the top of `__init__`.)

- [ ] **Step 4: Apply floor in `set_epoch`**

In `src/mongoose/losses/combined.py`, modify the last line of `set_epoch`'s blend computation:

Before:
```python
        else:
            self._warmstart_blend = 0.0
```
After:
```python
        else:
            self._warmstart_blend = 0.0

        # Apply floor so focal supervision never fully disappears.
        if self._warmstart_blend < self.min_blend:
            self._warmstart_blend = self.min_blend
```

(The floor is applied after the full blend computation, so it applies in all branches: pre-warmstart, during fade, and post-warmstart.)

- [ ] **Step 5: Run tests — expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_losses.py -v -k "blend"`
Expected: all blend-related tests PASS (including existing `test_combined_loss_warmstart_blend`).

- [ ] **Step 6: Run full loss test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/mongoose/losses/combined.py tests/test_losses/test_losses.py
git commit -m "feat(loss): add min_blend floor so focal supervision never fully disappears"
```

---

## Task 3: CombinedLoss Static Loss-Scale Divisors

**Files:**
- Modify: `src/mongoose/losses/combined.py` (constructor + `__call__`)
- Modify: `tests/test_losses/test_losses.py` (add one test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_losses/test_losses.py`:

```python
def test_combined_loss_scale_divisors_normalize_components(minimal_batch):
    """Raw loss components are divided by their scale divisors before lambda weighting."""
    # Arrange: two identical criterions except for scale divisors.
    plain = CombinedLoss(warmstart_epochs=0)
    scaled = CombinedLoss(
        warmstart_epochs=0,
        scale_probe=1.0,
        scale_bp=100.0,
        scale_vel=10.0,
        scale_count=1.0,
    )
    plain.set_epoch(0)
    scaled.set_epoch(0)

    # Note: `minimal_batch` fixture uses abbreviated dict keys (ref_bp, n_ref)
    # that need translating to CombinedLoss.__call__ kwargs.
    kwargs = dict(
        pred_heatmap=minimal_batch["pred_heatmap"],
        pred_cumulative_bp=minimal_batch["pred_cumulative_bp"],
        raw_velocity=minimal_batch["raw_velocity"],
        reference_bp_positions_list=minimal_batch["ref_bp"],
        n_ref_probes=minimal_batch["n_ref"],
        warmstart_heatmap=None,
        warmstart_valid=None,
        mask=minimal_batch["mask"],
    )

    _, details_plain = plain(**kwargs)
    _, details_scaled = scaled(**kwargs)

    # Assert: per-component details are the raw value / divisor.
    # Now that `details` exposes both scaled and raw, cross-check both sides.
    assert details_scaled["probe_raw"] == details_plain["probe_raw"]
    assert details_scaled["bp_raw"] == details_plain["bp_raw"]
    assert abs(details_scaled["bp"] - details_plain["bp_raw"] / 100.0) < 1e-5
    assert abs(details_scaled["vel"] - details_plain["vel_raw"] / 10.0) < 1e-5
    assert abs(details_scaled["count"] - details_plain["count_raw"] / 1.0) < 1e-5
    assert abs(details_scaled["probe"] - details_plain["probe_raw"] / 1.0) < 1e-5
```

- [ ] **Step 2: Run test — expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_losses.py::test_combined_loss_scale_divisors_normalize_components -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'scale_probe'`

- [ ] **Step 3: Add scale_* parameters to `CombinedLoss.__init__`**

In `src/mongoose/losses/combined.py`, extend the constructor signature:

```python
    def __init__(
        self,
        lambda_bp: float = 1.0,
        lambda_vel: float = 1.0,
        lambda_count: float = 1.0,
        warmup_epochs: int = 5,
        warmstart_epochs: int = 5,
        warmstart_fade_epochs: int = 2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        huber_delta_bp: float = 500.0,
        softdtw_gamma: float = 0.1,
        peakiness_window: int = 20,
        nms_threshold: float = 0.3,
        tag_width_bp: float = 511.0,
        sample_period_ms: float = 0.025,
        min_blend: float = 0.0,
        scale_probe: float = 1.0,
        scale_bp: float = 1.0,
        scale_vel: float = 1.0,
        scale_count: float = 1.0,
    ) -> None:
```

And store, grouped with the other scalars:

```python
        self.scale_probe = max(float(scale_probe), 1e-6)
        self.scale_bp = max(float(scale_bp), 1e-6)
        self.scale_vel = max(float(scale_vel), 1e-6)
        self.scale_count = max(float(scale_count), 1e-6)
```

- [ ] **Step 4: Apply divisors in `__call__`**

In `src/mongoose/losses/combined.py`, modify the final aggregation. Before:

```python
        probe_loss = torch.stack(probe_terms).mean() if probe_terms else zero
        bp_loss = torch.stack(bp_terms).mean() if bp_terms else zero
        vel_loss = torch.stack(vel_terms).mean() if vel_terms else zero
        count_loss_value = torch.stack(count_terms).mean() if count_terms else zero

        total = (
            probe_loss
            + self.current_lambda_bp * bp_loss
            + self.current_lambda_vel * vel_loss
            + self.current_lambda_count * count_loss_value
        )

        details: dict[str, Any] = {
            "probe": probe_loss.detach().item(),
            "bp": bp_loss.detach().item(),
            "vel": vel_loss.detach().item(),
            "count": count_loss_value.detach().item(),
            "warmstart_blend": blend,
        }
```

After:

```python
        probe_loss = torch.stack(probe_terms).mean() if probe_terms else zero
        bp_loss = torch.stack(bp_terms).mean() if bp_terms else zero
        vel_loss = torch.stack(vel_terms).mean() if vel_terms else zero
        count_loss_value = torch.stack(count_terms).mean() if count_terms else zero

        # Scale each component by its hardcoded divisor so lambda values
        # mean "contribute roughly equal gradient" instead of being 4-5
        # orders of magnitude apart (see spec sec. 3.2).
        scaled_probe = probe_loss / self.scale_probe
        scaled_bp = bp_loss / self.scale_bp
        scaled_vel = vel_loss / self.scale_vel
        scaled_count = count_loss_value / self.scale_count

        total = (
            scaled_probe
            + self.current_lambda_bp * scaled_bp
            + self.current_lambda_vel * scaled_vel
            + self.current_lambda_count * scaled_count
        )

        details: dict[str, Any] = {
            "probe": scaled_probe.detach().item(),
            "bp": scaled_bp.detach().item(),
            "vel": scaled_vel.detach().item(),
            "count": scaled_count.detach().item(),
            "probe_raw": probe_loss.detach().item(),
            "bp_raw": bp_loss.detach().item(),
            "vel_raw": vel_loss.detach().item(),
            "count_raw": count_loss_value.detach().item(),
            "warmstart_blend": blend,
        }
```

(Both scaled and raw values are reported so the training log shows
gradient-balance AND raw-magnitude drift.)

- [ ] **Step 5: Run the scale test — expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_losses.py::test_combined_loss_scale_divisors_normalize_components -v`
Expected: PASS.

- [ ] **Step 6: Regression check — default-args behavior unchanged**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/ tests/test_training/ -q`
Expected: all pass. With divisors defaulting to 1.0, scaled_* == raw_*, so all existing tests see identical values in `details` (the new `*_raw` keys are additive).

- [ ] **Step 7: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `200+ passed` (199 original + the new tests).

- [ ] **Step 8: Commit**

```bash
git add src/mongoose/losses/combined.py tests/test_losses/test_losses.py
git commit -m "feat(loss): add static scale divisors for per-component gradient balance"
```

---

## Task 4: Peak-Matching Library

**Files:**
- Create: `src/mongoose/inference/peak_match.py`
- Create: `tests/test_inference/test_peak_match.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_inference/test_peak_match.py`:

```python
"""Tests for Hungarian peak matching between predicted and reference peaks."""
from __future__ import annotations

import numpy as np
import pytest

from mongoose.inference.peak_match import (
    compute_metrics,
    match_peaks,
)


def test_match_peaks_perfect_alignment():
    pred = np.array([100, 500, 1000])
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 3
    assert fp == []
    assert fn == []


def test_match_peaks_within_tolerance():
    pred = np.array([100, 525, 1000])  # second one is 25 off
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 3
    assert fp == []
    assert fn == []


def test_match_peaks_outside_tolerance():
    pred = np.array([100, 700, 1000])  # second one is 200 off — too far
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 2
    assert fp == [1]  # pred index 1 unmatched
    assert fn == [1]  # ref index 1 unmatched


def test_match_peaks_extra_prediction():
    pred = np.array([100, 500, 1000, 2000])  # extra peak at 2000
    ref = np.array([100, 500, 1000])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 3
    assert fp == [3]  # pred index 3 (the 2000) unmatched
    assert fn == []


def test_match_peaks_missing_reference():
    pred = np.array([100, 500])
    ref = np.array([100, 500, 1000])  # no pred near 1000
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert len(matches) == 2
    assert fp == []
    assert fn == [2]


def test_match_peaks_optimal_not_greedy():
    """Two predicted peaks both near the same reference — Hungarian finds the optimal 1:1."""
    pred = np.array([100, 130])       # both within tolerance of ref 120
    ref = np.array([120, 200])         # 200 is unreachable by 130 (70 off)
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    # Optimal: pred 130 -> ref 120 (dist 10); pred 100 -> ref 200 is 100 dist (out of tol, FP+FN).
    # Or pred 100 -> ref 120 (dist 20); pred 130 -> ref 200 is 70 dist (out of tol, FP+FN).
    # Either way exactly one match and one FP + one FN.
    assert len(matches) == 1
    assert len(fp) == 1
    assert len(fn) == 1


def test_match_peaks_empty_predictions():
    pred = np.array([], dtype=np.int64)
    ref = np.array([100, 500])
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert matches == []
    assert fp == []
    assert fn == [0, 1]


def test_match_peaks_empty_references():
    pred = np.array([100, 500])
    ref = np.array([], dtype=np.int64)
    matches, fp, fn = match_peaks(pred, ref, tolerance=50)
    assert matches == []
    assert fp == [0, 1]
    assert fn == []


def test_match_peaks_both_empty():
    matches, fp, fn = match_peaks(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        tolerance=50,
    )
    assert matches == []
    assert fp == []
    assert fn == []


def test_compute_metrics_perfect():
    m = compute_metrics(tp=10, fp=0, fn=0)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_half_recall():
    m = compute_metrics(tp=5, fp=0, fn=5)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert abs(m["f1"] - 2 * 1.0 * 0.5 / 1.5) < 1e-6


def test_compute_metrics_zero_tp_returns_zero_f1():
    m = compute_metrics(tp=0, fp=5, fn=5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
```

- [ ] **Step 2: Run tests — expect import failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_inference/test_peak_match.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mongoose.inference.peak_match'`

- [ ] **Step 3: Implement the peak-matching library**

Create `src/mongoose/inference/peak_match.py`:

```python
"""Optimal 1:1 peak matching between predicted and reference peak positions.

Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) over
the pairwise absolute-distance matrix, with out-of-tolerance pairs blocked
by a large-cost sentinel so the optimizer routes around them when a
feasible alternative exists.

Note on matching semantics: this is optimal 1:1 assignment, not greedy-by-
confidence as in COCO/VOC F1. Good for internal model comparisons; not
directly benchmark-comparable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG_COST: float = 1e9


def match_peaks(
    pred_positions: np.ndarray,
    ref_positions: np.ndarray,
    tolerance: float = 50.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted peaks to reference peaks optimally at ``tolerance``.

    Args:
        pred_positions: 1D integer array of predicted peak positions (sample indices).
        ref_positions: 1D integer array of reference peak positions.
        tolerance: Absolute-distance tolerance for a valid match.

    Returns:
        ``(matches, unmatched_preds, unmatched_refs)`` where ``matches`` is a
        list of ``(pred_idx, ref_idx)`` index pairs and the two lists hold
        indices into ``pred_positions`` / ``ref_positions`` for unmatched
        peaks.
    """
    n_pred = int(pred_positions.shape[0])
    n_ref = int(ref_positions.shape[0])

    if n_pred == 0 and n_ref == 0:
        return [], [], []
    if n_pred == 0:
        return [], [], list(range(n_ref))
    if n_ref == 0:
        return [], list(range(n_pred)), []

    # Pairwise absolute distances. Shape: (n_pred, n_ref).
    pred_f = pred_positions.astype(np.float64)
    ref_f = ref_positions.astype(np.float64)
    dist = np.abs(pred_f[:, None] - ref_f[None, :])

    # Block out-of-tolerance pairs by giving them a huge cost so Hungarian
    # avoids them unless no better option exists for a given row/column.
    cost = np.where(dist <= tolerance, dist, _BIG_COST)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if dist[r, c] <= tolerance:
            matches.append((int(r), int(c)))
            matched_pred.add(int(r))
            matched_ref.add(int(c))

    unmatched_preds = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_refs = [i for i in range(n_ref) if i not in matched_ref]
    return matches, unmatched_preds, unmatched_refs


def compute_metrics(*, tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision / recall / F1 from TP / FP / FN counts.

    Returns zeros for undefined cases (tp=fp=0 or tp=fn=0) instead of raising.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_per_molecule_metrics(
    per_molecule: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate mean precision / recall / F1 across a list of per-molecule metric dicts.

    Each input dict must contain ``"precision"``, ``"recall"``, ``"f1"``.
    Returns a dict with the same keys whose values are the arithmetic means.
    """
    if not per_molecule:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_molecules": 0}
    p = float(np.mean([m["precision"] for m in per_molecule]))
    r = float(np.mean([m["recall"] for m in per_molecule]))
    f = float(np.mean([m["f1"] for m in per_molecule]))
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "n_molecules": len(per_molecule),
    }
```

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_inference/test_peak_match.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: `212+ passed`.

- [ ] **Step 6: Commit**

```bash
git add src/mongoose/inference/peak_match.py tests/test_inference/test_peak_match.py
git commit -m "feat(inference): add Hungarian peak-matching library with F1 metrics"
```

---

## Task 5: `evaluate_peak_match.py` CLI Script

**Files:**
- Create: `scripts/evaluate_peak_match.py`

- [ ] **Step 1: Implement the script**

Create `scripts/evaluate_peak_match.py`:

```python
"""Evaluate a trained checkpoint by peak-match F1 against wfmproc probe centers.

Usage:
    python scripts/evaluate_peak_match.py \\
        --checkpoint overnight_training/best_model.pt \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --output peak_match.json

Output: JSON with overall + per-run + per-molecule metrics, plus a stdout
summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.inference.peak_match import (
    aggregate_per_molecule_metrics,
    compute_metrics,
    match_peaks,
)
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def _run_id_for_cache(cache_dir: Path) -> str:
    import json as _json
    with open(cache_dir / "manifest.json") as f:
        manifest = _json.load(f)
    return str(manifest["run_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True,
                        help="Repeat for multiple caches.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=None,
                        help="Cap per-cache molecule count (debug).")
    parser.add_argument("--tolerance", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Peak-extraction confidence threshold.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    per_run: dict[str, dict[str, Any]] = {}
    per_molecule: list[dict[str, Any]] = []

    for cache_dir in args.cache_dir:
        run_id = _run_id_for_cache(cache_dir)
        dataset = CachedMoleculeDataset([cache_dir], augment=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_molecules,
        )

        run_tp = run_fp = run_fn = 0
        run_per_mol: list[dict[str, Any]] = []

        for batch_idx, batch in enumerate(loader):
            if args.max_molecules is not None:
                if batch_idx * args.batch_size >= args.max_molecules:
                    break

            waveform = batch["waveform"].to(device)
            conditioning = batch["conditioning"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    probe_heatmap, _, raw_velocity = model(waveform, conditioning, mask)
                probe_heatmap = probe_heatmap.float()
                raw_velocity = raw_velocity.float()

            b = waveform.shape[0]
            for i in range(b):
                h = probe_heatmap[i]
                v = raw_velocity[i]
                m = mask[i]
                h_masked = h * m.to(h.dtype)
                v_masked = torch.where(m, v, torch.zeros_like(v))
                pred_idx = extract_peak_indices(
                    h_masked, v_masked, threshold=args.threshold
                )
                pred_np = pred_idx.detach().cpu().numpy().astype(np.int64)

                # Reference peak positions (in sample space) come from the
                # cached wfmproc centers via the gt dict; loader has them in
                # batch["warmstart_heatmap"] as a built Gaussian but we want
                # the raw centers. Pull from the underlying dataset gt.
                # dataset.entries is [(dir_idx, mol_idx)]; but here our
                # dataset is single-cache, so dir_idx=0 always.
                global_idx = batch_idx * args.batch_size + i
                if global_idx >= len(dataset):
                    break
                dir_idx, mol_idx = dataset.entries[global_idx]
                gt = dataset.gt_lists[dir_idx][mol_idx]
                centers = gt.get("warmstart_probe_centers_samples")
                if centers is None:
                    # Skip molecules with no wfmproc ground truth.
                    continue
                ref_np = np.asarray(centers, dtype=np.int64)

                matches, fps, fns = match_peaks(
                    pred_np, ref_np, tolerance=args.tolerance
                )
                tp, fp, fn = len(matches), len(fps), len(fns)
                run_tp += tp; run_fp += fp; run_fn += fn
                metrics = compute_metrics(tp=tp, fp=fp, fn=fn)
                row = {
                    "run_id": run_id,
                    "molecule_uid": int(batch["molecule_uid"][i]),
                    "n_pred": int(pred_np.size),
                    "n_ref": int(ref_np.size),
                    "tp": tp, "fp": fp, "fn": fn,
                    **metrics,
                }
                run_per_mol.append(row)
                per_molecule.append(row)

        overall_tp += run_tp; overall_fp += run_fp; overall_fn += run_fn
        per_run[run_id] = {
            "tp": run_tp, "fp": run_fp, "fn": run_fn,
            **compute_metrics(tp=run_tp, fp=run_fp, fn=run_fn),
            "per_molecule_mean": aggregate_per_molecule_metrics(run_per_mol),
        }

    summary = {
        "overall": {
            "tp": overall_tp, "fp": overall_fp, "fn": overall_fn,
            **compute_metrics(tp=overall_tp, fp=overall_fp, fn=overall_fn),
            "per_molecule_mean": aggregate_per_molecule_metrics(per_molecule),
        },
        "per_run": per_run,
        "tolerance": args.tolerance,
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    # Print a short summary.
    o = summary["overall"]
    print(f"\n=== Peak-Match F1 @ tolerance={args.tolerance} samples ===")
    print(f"  Overall (sum-of-counts): P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}")
    m = o["per_molecule_mean"]
    print(f"  Per-molecule mean:       P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['n_molecules']})")
    for run_id, stats in per_run.items():
        print(f"  Run {run_id}: F1_sum={stats['f1']:.3f}  F1_mol={stats['per_molecule_mean']['f1']:.3f}  (n={stats['per_molecule_mean']['n_molecules']})")
    print(f"\n  Wrote: {args.output}")
    per_mol_json = per_molecule[:0]  # keep the main JSON compact if needed later


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — invoke on 16 molecules from the existing checkpoint**

Prerequisite: the existing `overnight_training/best_model.pt` from the diverged run is available.

Run:
```
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint overnight_training/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output smoke_eval.json \
    --max-molecules 16
```

Expected: script runs to completion; stdout shows a "Peak-Match F1" block with numeric precision/recall/F1; `smoke_eval.json` contains valid JSON with `overall`, `per_run`, `tolerance`, `threshold`, `checkpoint` keys. Exact F1 value is whatever the diverged model produces — we only care that the script executes and emits sane-looking structure.

Delete the smoke output: `rm smoke_eval.json`

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_peak_match.py
git commit -m "feat(eval): add peak-match F1 evaluator CLI"
```

---

## Task 6: `visualize_predictions.py` CLI Script

**Files:**
- Create: `scripts/visualize_predictions.py`

- [ ] **Step 1: Implement the script**

Create `scripts/visualize_predictions.py`:

```python
"""Render predicted vs reference peaks on a grid of molecules as one PNG.

Usage:
    python scripts/visualize_predictions.py \\
        --checkpoint overnight_training/best_model.pt \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --output preds_grid.png \\
        --n-molecules 20 \\
        --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-molecules", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = T2DUNet(config.in_channels, config.conditioning_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = CachedMoleculeDataset([args.cache_dir], augment=False)
    n = min(args.n_molecules, len(dataset))

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n))
    if n == 1:
        axes = [axes]

    for row_idx, dataset_idx in enumerate(indices):
        item = dataset[dataset_idx]
        waveform = item["waveform"].unsqueeze(0).to(device)
        conditioning = item["conditioning"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                heatmap, _, velocity = model(waveform, conditioning, mask)
            heatmap = heatmap.float().squeeze(0)
            velocity = velocity.float().squeeze(0)

        pred_idx = extract_peak_indices(heatmap, velocity, threshold=args.threshold)
        pred_np = pred_idx.cpu().numpy().astype(np.int64)

        dir_idx, mol_idx = dataset.entries[dataset_idx]
        gt = dataset.gt_lists[dir_idx][mol_idx]
        ref_centers = np.asarray(
            gt.get("warmstart_probe_centers_samples", []), dtype=np.int64
        )

        wf_np = item["waveform"].squeeze(0).numpy()
        hm_np = heatmap.cpu().numpy()
        t = np.arange(wf_np.shape[0])

        ax = axes[row_idx]
        ax.plot(t, wf_np, color="gray", linewidth=0.5, alpha=0.7, label="waveform")
        ax2 = ax.twinx()
        ax2.plot(t, hm_np[:wf_np.shape[0]], color="black", linewidth=0.8,
                 alpha=0.7, label="heatmap")
        ax2.set_ylim(0, 1.05)

        # Red = predicted peaks; blue = reference peaks.
        for p in pred_np:
            ax.axvline(p, color="red", linewidth=0.8, alpha=0.6)
        for r in ref_centers:
            ax.axvline(r, color="blue", linewidth=0.6, alpha=0.5, linestyle="--")

        uid = int(item.get("molecule_uid", dataset_idx))
        ax.set_title(
            f"mol uid={uid} | pred={len(pred_np)} (red) | ref={len(ref_centers)} (blue dashed)",
            fontsize=9,
        )
        ax.set_xlabel("sample")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    plt.close(fig)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run:
```
.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
    --checkpoint overnight_training/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output smoke_viz.png \
    --n-molecules 4 \
    --seed 42
```
Expected: script runs cleanly; `smoke_viz.png` exists and is > 20 KB (non-empty image).

Verify:
```
ls -la smoke_viz.png
```

Then delete: `rm smoke_viz.png`

- [ ] **Step 3: Commit**

```bash
git add scripts/visualize_predictions.py
git commit -m "feat(eval): add prediction visualization grid script"
```

---

## Task 7: `overfit_one_batch.py` Gate Script

**Files:**
- Create: `scripts/overfit_one_batch.py`

- [ ] **Step 1: Implement the script**

Create `scripts/overfit_one_batch.py`:

```python
"""Sanity gate: can the model memorize a single batch?

Loads one fixed batch, disables augmentation and warmstart, trains on
that same batch for N steps, and prints raw + scaled per-component
losses. Also saves one visualization PNG for the first molecule.

Pass criteria (spec sec. 5, Phase 1.5):
  - scaled bp < 0.1 by step 300
  - probe < 0.1 by step 300
  - no NaN / Inf at any step
  - visualization shows sharp peaks aligned with reference
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.collate import collate_molecules
from mongoose.losses.combined import CombinedLoss
from mongoose.losses.peaks import extract_peak_indices
from mongoose.model.unet import T2DUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-bp", type=float, default=30000.0)
    parser.add_argument("--scale-vel", type=float, default=5000.0)
    parser.add_argument("--scale-count", type=float, default=1.0)
    parser.add_argument("--scale-probe", type=float, default=1.0)
    parser.add_argument("--min-blend", type=float, default=0.1)
    parser.add_argument("--output-viz", type=Path, default=Path("overfit_gate_viz.png"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CachedMoleculeDataset([args.cache_dir], augment=False)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.batch_size, replace=False).tolist()
    items = [dataset[i] for i in indices]
    batch = collate_molecules(items)

    waveform = batch["waveform"].to(device)
    conditioning = batch["conditioning"].to(device)
    mask = batch["mask"].to(device)
    reference_bp_positions_list = [
        bp.to(device) for bp in batch["reference_bp_positions"]
    ]
    n_ref_probes = batch["n_ref_probes"].to(device)
    warmstart_heatmap = batch.get("warmstart_heatmap")
    if warmstart_heatmap is not None:
        warmstart_heatmap = warmstart_heatmap.to(device)
    warmstart_valid = batch.get("warmstart_valid")
    if warmstart_valid is not None:
        warmstart_valid = warmstart_valid.to(device)

    in_channels = waveform.shape[1]
    cond_dim = conditioning.shape[1]
    model = T2DUNet(in_channels, cond_dim).to(device)
    model.train()

    criterion = CombinedLoss(
        warmstart_epochs=0,  # No warmstart schedule; go straight to real target.
        warmstart_fade_epochs=0,
        min_blend=args.min_blend,
        scale_probe=args.scale_probe,
        scale_bp=args.scale_bp,
        scale_vel=args.scale_vel,
        scale_count=args.scale_count,
    )
    criterion.set_epoch(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            probe_heatmap, cumulative_bp, raw_velocity = model(
                waveform, conditioning, mask
            )
        with torch.amp.autocast("cuda", enabled=False):
            loss, details = criterion(
                pred_heatmap=probe_heatmap.float(),
                pred_cumulative_bp=cumulative_bp.float(),
                raw_velocity=raw_velocity.float(),
                reference_bp_positions_list=reference_bp_positions_list,
                n_ref_probes=n_ref_probes,
                warmstart_heatmap=warmstart_heatmap,
                warmstart_valid=warmstart_valid,
                mask=mask,
            )

        if not math.isfinite(loss.item()):
            print(f"NaN/Inf at step {step}: details={details}")
            raise SystemExit(2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(
                f"step {step:4d} | total={loss.item():10.5f} | "
                f"probe={details['probe']:.4f} bp={details['bp']:.4f} "
                f"vel={details['vel']:.4f} count={details['count']:.4f} | "
                f"raw_bp={details['bp_raw']:.1f} raw_vel={details['vel_raw']:.1f}"
            )

    # Final visualization for the first molecule in the batch.
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            heatmap, _, velocity = model(waveform, conditioning, mask)
        h0 = heatmap[0].float().cpu().numpy()
        v0 = velocity[0].float()
        pred_idx = extract_peak_indices(
            heatmap[0].float(), v0, threshold=0.3
        ).cpu().numpy()

    dir_idx, mol_idx = dataset.entries[indices[0]]
    gt = dataset.gt_lists[dir_idx][mol_idx]
    ref_centers = np.asarray(
        gt.get("warmstart_probe_centers_samples", []), dtype=np.int64
    )
    wf0 = waveform[0, 0].float().cpu().numpy()
    t = np.arange(wf0.shape[0])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, wf0, color="gray", linewidth=0.5, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(t, h0[:wf0.shape[0]], color="black", linewidth=0.8)
    ax2.set_ylim(0, 1.05)
    for p in pred_idx:
        ax.axvline(p, color="red", linewidth=0.8, alpha=0.7)
    for r in ref_centers:
        ax.axvline(r, color="blue", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_title(
        f"Overfit gate @ step {args.steps}: pred={len(pred_idx)} (red), ref={len(ref_centers)} (blue)"
    )

    args.output_viz.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.output_viz, dpi=100)
    plt.close(fig)
    print(f"\nSaved viz: {args.output_viz}")

    # Exit code 0 on clean completion. Decide pass/fail by inspecting the
    # printed metrics and viz.


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — 10 steps, 4 molecules**

Run:
```
.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --steps 10 \
    --batch-size 4 \
    --output-viz smoke_overfit.png
```

Expected: script runs cleanly, prints per-step loss lines, no NaN, creates `smoke_overfit.png`. Final-step scaled losses will be worse than converged (only 10 steps) — this just confirms plumbing.

Delete: `rm smoke_overfit.png`

- [ ] **Step 3: Commit**

```bash
git add scripts/overfit_one_batch.py
git commit -m "feat(training): add overfit-one-batch gate script"
```

---

## Task 8: Phase 1.5 — Overfit Gate

**Type:** Operational. No new code.

- [ ] **Step 1: Run the gate**

```
.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --steps 300 \
    --batch-size 32 \
    --output-viz phase1_5_overfit.png
```

Expected wall: ~10-15 min on the A4500.

- [ ] **Step 2: Check pass criteria**

Review the stdout log and `phase1_5_overfit.png`. The four criteria from the spec:

1. `bp` (scaled) drops below 0.1 by step 300.
2. `probe` drops below 0.1 by step 300.
3. No NaN/Inf at any step (script would have exited with code 2 if so).
4. Visualization shows sharp peaks (red ticks) aligned with reference peaks (blue dashed ticks) — not a flat heatmap.

- [ ] **Step 3: On pass — advance to Task 9**

If all four criteria met, proceed to Task 9 (Phase 2a smoke).

- [ ] **Step 4: On fail — triage**

If any criterion fails: STOP. Investigate. The failure mode tells you what to fix:
- bp or probe not dropping → loss-weighting wrong; try tweaking `--scale-bp` or `--min-blend`.
- NaN/Inf → something in the pipeline lost numerical stability; systematic-debugging skill.
- Flat heatmap at step 300 → focal loss not driving peak structure; suspect `min_blend` too low, or the model is still getting dominated by a different loss term.

Resolve, re-run Step 1. Do not advance to Task 9 until all four criteria are met.

---

## Task 9: Phase 2a — First Smoke Run

**Type:** Operational. No new code. Uses existing `scripts/train.py` with new loss flags.

Note: the new `scale_*` and `min_blend` parameters are on `CombinedLoss` but `scripts/train.py` does not currently expose them via CLI. For this task we pass them via a small Python one-liner that imports the training config and overrides directly. Alternative: add flags to `train.py`. Adding the flags is cleaner and part of this task.

**Files to modify:**
- Modify: `src/mongoose/training/cli.py` (add `--min-blend`, `--scale-bp`, `--scale-vel`, `--scale-count`, `--scale-probe` flags and thread them into `TrainConfig`)
- Modify: `src/mongoose/training/config.py` (add fields to `TrainConfig`)
- Modify: `src/mongoose/training/trainer.py` (pass config fields through to `CombinedLoss`)

Note: `scripts/train.py` is a thin wrapper around `mongoose.training.cli.main`; the parser and config assembly both live in the package for unit-testability, so we modify the package not the script.

- [ ] **Step 1: Extend `TrainConfig`**

In `src/mongoose/training/config.py`, add these fields under `# Loss` (after `nms_threshold`):

```python
    min_blend: float = 0.0
    scale_probe: float = 1.0
    scale_bp: float = 1.0
    scale_vel: float = 1.0
    scale_count: float = 1.0
```

- [ ] **Step 2: Thread through `Trainer`**

In `src/mongoose/training/trainer.py`, modify the `CombinedLoss(...)` construction inside `__init__`:

```python
        self.criterion = CombinedLoss(
            lambda_bp=config.lambda_bp,
            lambda_vel=config.lambda_vel,
            lambda_count=config.lambda_count,
            warmup_epochs=config.warmup_epochs,
            warmstart_epochs=config.warmstart_epochs,
            warmstart_fade_epochs=config.warmstart_fade_epochs,
            softdtw_gamma=config.softdtw_gamma,
            peakiness_window=config.peakiness_window,
            nms_threshold=config.nms_threshold,
            min_blend=config.min_blend,
            scale_probe=config.scale_probe,
            scale_bp=config.scale_bp,
            scale_vel=config.scale_vel,
            scale_count=config.scale_count,
        )
```

- [ ] **Step 3: Add CLI flags in `src/mongoose/training/cli.py`**

In `build_arg_parser`, add these arguments alongside `--warmstart-epochs` / `--warmstart-fade-epochs`:

```python
    parser.add_argument(
        "--min-blend",
        type=float,
        default=None,
        help="Floor for the focal-vs-peakiness blend; 0.1 keeps focal supervision on forever.",
    )
    parser.add_argument(
        "--scale-bp",
        type=float,
        default=None,
        help="Divisor applied to raw bp_loss for per-component gradient balance.",
    )
    parser.add_argument("--scale-vel", type=float, default=None)
    parser.add_argument("--scale-count", type=float, default=None)
    parser.add_argument("--scale-probe", type=float, default=None)
```

In `config_from_args`, after the existing `if args.warmstart_*` blocks, add:

```python
    if args.min_blend is not None:
        config.min_blend = args.min_blend
    if args.scale_bp is not None:
        config.scale_bp = args.scale_bp
    if args.scale_vel is not None:
        config.scale_vel = args.scale_vel
    if args.scale_count is not None:
        config.scale_count = args.scale_count
    if args.scale_probe is not None:
        config.scale_probe = args.scale_probe
```

- [ ] **Step 4: Full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass. (If any existing CLI test broke, look at `tests/test_training/test_cli.py`.)

- [ ] **Step 5: Commit the CLI plumbing**

```bash
git add src/mongoose/training/config.py src/mongoose/training/trainer.py src/mongoose/training/cli.py
git commit -m "feat(training): expose min_blend and loss-scale divisors via CLI"
```

- [ ] **Step 6: Launch the Phase 2a smoke**

```
rm -rf phase2a_checkpoints 2>/dev/null
.venv/Scripts/python.exe -u scripts/train.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --epochs 10 \
    --batch-size 32 \
    --lr 3e-4 \
    --warmstart-epochs 8 \
    --warmstart-fade-epochs 4 \
    --min-blend 0.1 \
    --scale-bp 30000 \
    --scale-vel 5000 \
    --scale-count 1.0 \
    --scale-probe 1.0 \
    --checkpoint-dir phase2a_checkpoints \
    --save-every 1 2>&1 | tee phase2a_train.log
```

Expected wall: ~4 hours.

- [ ] **Step 7: Evaluate Phase 2a**

```
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint phase2a_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2a_eval.json

.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
    --checkpoint phase2a_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2a_viz.png \
    --n-molecules 20 --seed 42
```

- [ ] **Step 8: Go / no-go for Phase 3**

Read `phase2a_eval.json` overall F1.

- **F1 ≥ 0.3** → proceed to Task 10 (long single-run train).
- **F1 < 0.3** → proceed to Task 9b (one tuning iteration) before advancing.

---

## Task 9b: (Conditional) Phase 2c — Single Tuning Iteration

**Skip this task if Phase 2a's F1 was ≥ 0.3.**

- [ ] **Step 1: Decide what to tune**

Based on `phase2a_train.log` and `phase2a_eval.json`, pick ONE adjustment:
- If scaled losses are wildly off (e.g. scaled_bp stayed > 10 across epochs) → scale divisors are wrong. Set `--scale-bp <observed-raw-bp-mean>` instead of 30000.
- If probe stayed at ≈0.998 throughout → `--min-blend 0.2` (double the floor).
- If loss exploded mid-run → reduce `--lr 1e-4`.

- [ ] **Step 2: Re-run smoke with the one change**

```
rm -rf phase2c_checkpoints 2>/dev/null
# Same as Task 9 Step 6 but substitute the one tuned flag, and
# --checkpoint-dir phase2c_checkpoints.
```

- [ ] **Step 3: Evaluate**

```
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint phase2c_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2c_eval.json
```

- [ ] **Step 4: Go / escalate**

If F1 ≥ 0.3, proceed to Task 10. If still < 0.3, stop and escalate to the user — design assumptions are wrong and we need a re-scope before burning more GPU.

---

## Task 10: Phase 3 — Preprocess Remaining 29 E. coli Runs

**Type:** Operational. Can run in parallel with any GPU task.

- [ ] **Step 1: Enumerate runs that still need preprocessing**

```
ls "E. coli/Black/" "E. coli/Blue/" "E. coli/Red/" | sort -u > /tmp/all_runs.txt
ls "E. coli/cache/" > /tmp/cached_runs.txt
comm -23 <(sort /tmp/all_runs.txt) <(sort /tmp/cached_runs.txt) > /tmp/todo_runs.txt
cat /tmp/todo_runs.txt
```

Expected: ~28 run IDs (30 total minus the 2 already cached).

- [ ] **Step 2: Loop preprocess**

For each run ID `R` listed, locate the color (`Black`, `Blue`, `Red`) and date subdir, then:

```bash
# Shell loop (example for one color — repeat per color):
for run_id in $(cat /tmp/todo_runs.txt); do
  for color in Black Blue Red; do
    run_dir="E. coli/$color/$run_id"
    if [ -d "$run_dir" ]; then
      date_dir=$(ls -d "$run_dir"/*/ | head -1)
      echo "=== Preprocessing $run_id (color=$color, date=$date_dir) ==="
      .venv/Scripts/python.exe scripts/preprocess.py \
          --run-id "$run_id" \
          --run-dir "$date_dir" \
          --output "E. coli/cache"
      break
    fi
  done
done
```

Expected total wall: ~30 runs × ~35 s = ~17 min.

- [ ] **Step 3: Verify 30 caches exist**

```
ls "E. coli/cache/" | wc -l
```

Expected: 30.

- [ ] **Step 4: Spot-check a per-color cache manifest**

```
cat "E. coli/cache/$(ls 'E. coli/Black' | head -1)/manifest.json" | head -15
```

Expected: valid JSON with `run_id`, `stats.cached_molecules` > 30000. If any run failed to preprocess (e.g., missing probes.bin), log it and continue — one or two missing runs is acceptable.

- [ ] **Step 5: Identify the 3 held-out test runs**

```
ls "E. coli/Black/" | sort | tail -1 > /tmp/holdout_black.txt
ls "E. coli/Blue/"  | sort | tail -1 > /tmp/holdout_blue.txt
ls "E. coli/Red/"   | sort | tail -1 > /tmp/holdout_red.txt
cat /tmp/holdout_{black,blue,red}.txt
```

Record these three run IDs in a note file for use in Tasks 13 and 14:
```bash
cat /tmp/holdout_*.txt > overnight_holdout_runs.txt
cat overnight_holdout_runs.txt
```

- [ ] **Step 6: Commit the holdout manifest**

```bash
git add overnight_holdout_runs.txt
git commit -m "chore: pin 3 held-out test run IDs (last alphabetical per color)"
```

---

## Task 11: Phase 4 — Long Single-Run Training

**Type:** Operational.

- [ ] **Step 1: Launch**

```
rm -rf single_run_training 2>/dev/null
.venv/Scripts/python.exe -u scripts/train.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --epochs 35 \
    --batch-size 32 \
    --lr 3e-4 \
    --warmstart-epochs 8 \
    --warmstart-fade-epochs 4 \
    --min-blend 0.1 \
    --scale-bp 30000 \
    --scale-vel 5000 \
    --scale-count 1.0 \
    --scale-probe 1.0 \
    --checkpoint-dir single_run_training \
    --save-every 5 2>&1 | tee single_run_train.log
```

Expected wall: ~15 h.

- [ ] **Step 2: Mid-run sanity check (after ~3 epochs)**

Tail the log, confirm losses are decreasing or at least stable, no NaN. If exploding, stop and revise.

- [ ] **Step 3: Evaluate final model**

```
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint single_run_training/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output single_run_eval.json

.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
    --checkpoint single_run_training/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output single_run_viz.png \
    --n-molecules 20 --seed 42
```

- [ ] **Step 4: Decide on Phase 6**

Record `single_run_eval.json` overall F1.

- **F1 ≥ 0.5** → strong result, proceed to Task 12 (multi-run).
- **F1 ∈ [0.3, 0.5]** → OK result, proceed to Task 12 with tempered expectations.
- **F1 < 0.3** → skip Task 12. Use remaining 20h budget to iterate on single-run recipe instead.

---

## Task 12: Phase 6 — Multi-Run Training

**Type:** Operational. Skip if Task 11 F1 < 0.3.

- [ ] **Step 1: Build an array of training cache-dir flags (27 runs, excluding 3 holdouts)**

```bash
# Use a bash array so filenames with spaces (like "E. coli") survive word-splitting.
train_flags=()
while read -r run_id; do
  if ! grep -qx "$run_id" overnight_holdout_runs.txt; then
    train_flags+=("--cache-dir" "E. coli/cache/$run_id")
  fi
done < <(ls "E. coli/cache/")

# Sanity check:
n_cache=$(printf '%s\n' "${train_flags[@]}" | grep -c -- '^--cache-dir$')
echo "Training caches: $n_cache"
```

Expected count: 27 (or 28 if a holdout didn't preprocess cleanly — acceptable).

- [ ] **Step 2: Confirm loss scales are still appropriate**

Run the overfit-one-batch gate on one Blue and one Red cache to check raw magnitudes stay comparable to single-run. Pick one Blue run (not the holdout) and one Red run (not the holdout):

```bash
blue_run=$(ls "E. coli/Blue/" | head -1)
red_run=$(ls "E. coli/Red/"  | head -1)
echo "Checking Blue=$blue_run, Red=$red_run"

.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "E. coli/cache/$blue_run" \
    --steps 30 --batch-size 16 \
    --output-viz cross_color_blue.png

.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "E. coli/cache/$red_run" \
    --steps 30 --batch-size 16 \
    --output-viz cross_color_red.png
```

Check the first-step `raw_bp` and `raw_vel` values in each log. If they are within ~10× of the single-run values (bp ~30000, vel ~5000), proceed to Step 3. If wildly different, adjust `--scale-bp` / `--scale-vel` in Step 3 to match the cross-color median.

- [ ] **Step 3: Launch multi-run training**

```bash
rm -rf multi_run_training 2>/dev/null
.venv/Scripts/python.exe -u scripts/train.py "${train_flags[@]}" \
    --epochs 2 \
    --batch-size 32 \
    --lr 3e-4 \
    --warmstart-epochs 1 \
    --warmstart-fade-epochs 1 \
    --min-blend 0.1 \
    --scale-bp 30000 \
    --scale-vel 5000 \
    --scale-count 1.0 \
    --scale-probe 1.0 \
    --checkpoint-dir multi_run_training \
    --save-every 1 2>&1 | tee multi_run_train.log
```

Note: this must run in the **same shell session** as Step 1, because the
`train_flags` array is local to that shell. If resuming in a new shell,
rebuild the array first by re-running Step 1.

Note: warmstart length reduced to 1 because each epoch over 1.5M molecules
is the equivalent of many single-run epochs; the model should be past the
warmstart quickly.

Expected wall: ~20-22 h.

- [ ] **Step 4: Mid-epoch sanity**

After ~5 hours (half of epoch 1), tail the log. Confirm losses decreasing, no NaN. If failing, stop and triage.

---

## Task 13: Phase 7 — Evaluate Multi-Run Model on Held-Out Test Runs

**Type:** Operational. Skip if Task 12 was skipped.

- [ ] **Step 1: Evaluate per held-out run**

```bash
for run_id in $(cat overnight_holdout_runs.txt); do
  .venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
      --checkpoint multi_run_training/best_model.pt \
      --cache-dir "E. coli/cache/$run_id" \
      --output "holdout_${run_id}_eval.json"
  .venv/Scripts/python.exe -u scripts/visualize_predictions.py \
      --checkpoint multi_run_training/best_model.pt \
      --cache-dir "E. coli/cache/$run_id" \
      --output "holdout_${run_id}_viz.png" \
      --n-molecules 20 --seed 42
done
```

- [ ] **Step 2: Also evaluate on a training-set run for comparison**

Pick one non-holdout, non-single-run training run:

```bash
train_sample=$(ls "E. coli/cache/" | grep -v -f overnight_holdout_runs.txt | head -1)
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint multi_run_training/best_model.pt \
    --cache-dir "E. coli/cache/$train_sample" \
    --output "train_sample_${train_sample}_eval.json"
```

The comparison of F1 on held-out vs training runs tells us whether the model generalizes or memorizes.

- [ ] **Step 3: Collect all F1 numbers into one summary**

```bash
echo "=== Held-out F1 ==="
for run_id in $(cat overnight_holdout_runs.txt); do
  .venv/Scripts/python.exe -c "
import json, sys
d = json.load(open(sys.argv[1]))
o = d['overall']
m = o['per_molecule_mean']
print(f'{sys.argv[2]}: F1_sum={o[\"f1\"]:.3f}  F1_mol_mean={m[\"f1\"]:.3f}  (n={m[\"n_molecules\"]})')
" "holdout_${run_id}_eval.json" "$run_id"
done
echo "=== Training-set sample F1 ==="
.venv/Scripts/python.exe -c "
import json, sys
d = json.load(open(sys.argv[1]))
o = d['overall']
m = o['per_molecule_mean']
print(f'{sys.argv[2]}: F1_sum={o[\"f1\"]:.3f}  F1_mol_mean={m[\"f1\"]:.3f}  (n={m[\"n_molecules\"]})')
" "train_sample_${train_sample}_eval.json" "$train_sample"
```

---

## Task 14: Phase 8 — Final Written Summary

**Files:**
- Create: `docs/plans/2026-04-18-v1-training-plan-results.md`

- [ ] **Step 1: Draft results document**

Create `docs/plans/2026-04-18-v1-training-plan-results.md` with this structure:

```markdown
# V1 Training — 48-Hour Results

**Date:** 2026-04-18 / 2026-04-19
**Hardware:** NVIDIA RTX A4500, 20 GB VRAM
**Plan:** [2026-04-18-v1-training-plan-implementation.md](2026-04-18-v1-training-plan-implementation.md)
**Spec:** [2026-04-18-v1-training-plan-design.md](2026-04-18-v1-training-plan-design.md)

## Executive Summary

[1-2 sentences: did we succeed or not; primary F1 numbers.]

## Configuration

- **Single-run training:** [config flags used]
- **Multi-run training:** [config flags, number of runs, total molecules]
- **Held-out test runs:** [3 IDs from `overnight_holdout_runs.txt`]

## Results

### Phase 1.5 (Overfit Gate)
[scaled bp / probe final values; pass criteria met?]

### Phase 2a (Smoke)
[F1; link to viz; notes]

### Phase 4 (Single-Run)
[F1 + precision + recall; best epoch; link to viz]

### Phase 6 (Multi-Run)
Per held-out run:
- [run_id 1]: F1=..., P=..., R=...
- [run_id 2]: F1=..., P=..., R=...
- [run_id 3]: F1=..., P=..., R=...
Training-set comparator: F1=...

## What Worked

[bullet points with specifics]

## What Didn't

[bullet points with specifics; include anomalies observed]

## Recommended Next Steps

[2-5 bullets: hyperparameters to try, code changes to explore, unanswered questions]

## Artifacts

- Checkpoints: [paths]
- JSON: [paths]
- Visualizations: [paths]
```

Fill in the actual values observed from the runs. Keep it terse; this is a working document, not a publication.

- [ ] **Step 2: Commit**

```bash
git add docs/plans/2026-04-18-v1-training-plan-results.md
git commit -m "docs(plans): 48-hour training results summary"
```

---

## File Structure Summary

```
src/mongoose/
├── losses/
│   └── combined.py          [MODIFY Tasks 2, 3]
├── inference/
│   └── peak_match.py        [CREATE Task 4]
└── training/
    ├── config.py            [MODIFY Task 9 Step 1]
    └── trainer.py           [MODIFY Task 9 Step 2]

src/mongoose/training/
└── cli.py                   [MODIFY Task 9 Step 3]

scripts/
├── evaluate_peak_match.py   [CREATE Task 5]
├── visualize_predictions.py [CREATE Task 6]
└── overfit_one_batch.py     [CREATE Task 7]

tests/
├── test_losses/
│   └── test_losses.py       [MODIFY Tasks 2, 3]
└── test_inference/
    └── test_peak_match.py   [CREATE Task 4]

docs/plans/
└── 2026-04-18-v1-training-plan-results.md  [CREATE Task 14]

pyproject.toml               [MODIFY Task 1]
overnight_holdout_runs.txt   [CREATE Task 10 Step 5]
```

## Branch Handling

- All of Task 1-7 and Task 9 Steps 1-5 are code commits on branch `training/v1-recipe-design`.
- Task 9 Step 6 onward (operational tasks) do not produce code commits (except the pinned holdout list in Task 10 and the results doc in Task 14).
- Final state at 48h: branch `training/v1-recipe-design` contains the loss recipe changes, evaluation library, three scripts, the holdout pin, and the results doc. `perf/criterion-amp-fixes` upstream stays focused on GPU perf + AMP.
