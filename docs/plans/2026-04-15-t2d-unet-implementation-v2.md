# T2D U-Net Implementation Plan -- V2 (Delta from V1 Original)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rearchitecture V1 to replace wfmproc probe labels with reference-anchored ground truth and self-supervised probe detection. Executed on branch `v1-rearchitecture`.

**Architecture reference:** `docs/plans/2026-04-15-t2d-unet-design-v2.md`

**Base state:** V1 original is complete (124 passing tests) on branch `main`. This plan modifies existing modules and adds new ones on top of that foundation.

**Tech stack unchanged:** Python 3.12+, PyTorch 2.11+, NumPy, pytest.

**Soft-DTW dependency:** We will use a pure-PyTorch soft-DTW implementation (no external library). Simple DP-based, ~100 lines, tested.

---

## What Stays The Same

- Model architecture (encoder, bottleneck, decoder, bifurcation, FiLM) -- no code changes
- All IO parsers (TDB, probes.bin, assigns, reference map, transform) -- no code changes
- Inference pipeline (NMS, sub-sample interpolation, output formatting) -- no code changes to core logic
- Test infrastructure -- add new tests, existing tests continue to pass

## What Changes

Six tasks, each with TDD steps. Tasks are sequenced but tasks 2 and 3 can run in parallel (independent code paths).

---

### Task R1: Level-1 estimator from raw waveform

**Files:**
- Create: `src/mongoose/data/level1.py`
- Create: `tests/test_data/test_level1.py`

**Step 1: Write failing tests**

```python
# tests/test_data/test_level1.py
import numpy as np
from mongoose.data.level1 import estimate_level1

def test_estimate_level1_flat_backbone():
    # Synthetic: flat waveform at 1000, rise at sample 50, fall at sample 950
    waveform = np.ones(1000, dtype=np.int16) * 1000
    waveform[:50] = 2000  # pre-molecule baseline (higher)
    waveform[950:] = 2000
    lvl1 = estimate_level1(waveform, rise_end_idx=50, fall_min_idx=950)
    assert abs(lvl1 - 1000) < 5.0

def test_estimate_level1_with_tag_dips():
    # Backbone at 1000 with tag dips to 500 at various points
    waveform = np.ones(1000, dtype=np.int16) * 1000
    waveform[:50] = 2000
    waveform[950:] = 2000
    waveform[200:215] = 500
    waveform[400:418] = 500
    waveform[700:712] = 500
    lvl1 = estimate_level1(waveform, rise_end_idx=50, fall_min_idx=950)
    assert 990 < lvl1 < 1010  # median is robust to dips

def test_estimate_level1_all_noise():
    # Pure noise backbone should give a sensible estimate
    rng = np.random.default_rng(42)
    waveform = rng.normal(1000, 50, 1000).astype(np.int16)
    lvl1 = estimate_level1(waveform, rise_end_idx=0, fall_min_idx=1000)
    assert 950 < lvl1 < 1050
```

**Step 2: Implement**

```python
# src/mongoose/data/level1.py
"""Estimate level-1 backbone amplitude from raw TDB waveform."""
import numpy as np

def estimate_level1(
    waveform: np.ndarray,
    rise_end_idx: int,
    fall_min_idx: int,
    trim_fraction: float = 0.1,
) -> float:
    """Estimate level-1 using median of backbone samples.
    
    The backbone is between rise_end_idx and fall_min_idx. Median is robust
    to tag dips. Trimming removes extreme outliers (both dips and noise spikes).
    """
    if fall_min_idx <= rise_end_idx:
        # Degenerate case -- use full waveform
        backbone = waveform.astype(np.float32)
    else:
        backbone = waveform[rise_end_idx:fall_min_idx].astype(np.float32)
    
    if len(backbone) < 10:
        return float(np.median(waveform.astype(np.float32)))
    
    # Trimmed median for robustness
    lo, hi = np.percentile(backbone, [trim_fraction * 100, (1 - trim_fraction) * 100])
    trimmed = backbone[(backbone >= lo) & (backbone <= hi)]
    return float(np.median(trimmed)) if len(trimmed) > 0 else float(np.median(backbone))
```

**Step 3: Verify and commit**

```bash
pytest tests/test_data/test_level1.py -v
git add src/mongoose/data/level1.py tests/test_data/test_level1.py
git commit -m "feat: level-1 estimator from raw TDB waveform backbone"
```

---

### Task R2: Soft-DTW loss module

**Files:**
- Create: `src/mongoose/losses/softdtw.py`
- Create: `tests/test_losses/test_softdtw.py`

**Reference:** Cuturi & Blondel 2017, "Soft-DTW: a Differentiable Loss Function for Time-Series". Pure PyTorch implementation.

**Step 1: Write failing tests**

```python
# tests/test_losses/test_softdtw.py
import torch
from mongoose.losses.softdtw import soft_dtw

def test_soft_dtw_identical_sequences():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = soft_dtw(x, y, gamma=0.1)
    assert loss.item() < 0.01

def test_soft_dtw_shifted_sequences():
    """Same sequence, shifted in time; DTW should be near zero."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])  # shifted by 1
    # DTW aligns by matching similar values, so this isn't zero
    # but should be meaningfully smaller than a totally different sequence
    loss_shifted = soft_dtw(x, y, gamma=0.1)
    loss_different = soft_dtw(x, torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]), gamma=0.1)
    assert loss_shifted < loss_different

def test_soft_dtw_different_lengths():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 2.5, 3.0])
    loss = soft_dtw(x, y, gamma=0.1)
    # Should complete without error
    assert loss.item() >= 0

def test_soft_dtw_differentiable():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([1.5, 2.5, 3.5])
    loss = soft_dtw(x, y, gamma=0.1)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
```

**Step 2: Implement**

```python
# src/mongoose/losses/softdtw.py
"""Soft-DTW loss (Cuturi & Blondel 2017) for matching detected peaks to reference probes.

Pure PyTorch; differentiable; handles sequences of different lengths natively.
"""
import torch

def _soft_min(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Smooth minimum via log-sum-exp."""
    stacked = torch.stack([a, b, c])
    return -gamma * torch.logsumexp(-stacked / gamma, dim=0)


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Soft-DTW between two 1D sequences.
    
    Args:
        x: [N] tensor
        y: [M] tensor
        gamma: smoothing parameter; smaller = closer to hard DTW
    
    Returns: scalar loss
    """
    n, m = x.shape[0], y.shape[0]
    
    # Pairwise squared distance cost matrix
    cost = (x.unsqueeze(1) - y.unsqueeze(0)) ** 2  # [N, M]
    
    # DP: r[i, j] = cost[i, j] + soft_min(r[i-1, j], r[i-1, j-1], r[i, j-1])
    # Pad with infinities on the border
    inf = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
    r = torch.full((n + 1, m + 1), float('inf'), device=x.device, dtype=x.dtype)
    r[0, 0] = 0
    
    # Iterative DP build (gradient flows through tensor operations)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r[i, j] = cost[i - 1, j - 1] + _soft_min(r[i - 1, j], r[i - 1, j - 1], r[i, j - 1], gamma)
    
    return r[n, m]
```

**Step 3: Verify and commit**

```bash
pytest tests/test_losses/test_softdtw.py -v
git add src/mongoose/losses/softdtw.py tests/test_losses/test_softdtw.py
git commit -m "feat: pure-PyTorch soft-DTW loss for peak-to-reference matching"
```

---

### Task R3: Count loss and peakiness regularizer

**Files:**
- Create: `src/mongoose/losses/count.py`
- Modify: existing `tests/test_losses/test_losses.py` (add tests)

**Step 1: Write failing tests (add to test_losses.py)**

```python
def test_count_loss_exact_match():
    heatmap = torch.zeros(100)
    heatmap[10] = heatmap[30] = heatmap[60] = 1.0  # 3 peaks
    mask = torch.ones(100, dtype=torch.bool)
    from mongoose.losses.count import count_loss
    loss = count_loss(heatmap, target_count=3, mask=mask)
    assert loss.item() < 0.1

def test_count_loss_too_few():
    heatmap = torch.zeros(100)
    heatmap[10] = 1.0
    mask = torch.ones(100, dtype=torch.bool)
    from mongoose.losses.count import count_loss
    loss_low = count_loss(heatmap, target_count=3, mask=mask)
    loss_ok = count_loss(heatmap.clone(), target_count=1, mask=mask)
    assert loss_low > loss_ok

def test_peakiness_regularizer_flat():
    """Flat heatmap should have high peakiness loss."""
    from mongoose.losses.count import peakiness_regularizer
    heatmap_flat = torch.ones(100) * 0.3
    heatmap_peaky = torch.zeros(100)
    heatmap_peaky[10] = heatmap_peaky[50] = heatmap_peaky[90] = 1.0
    loss_flat = peakiness_regularizer(heatmap_flat, window=20)
    loss_peaky = peakiness_regularizer(heatmap_peaky, window=20)
    assert loss_flat > loss_peaky
```

**Step 2: Implement**

```python
# src/mongoose/losses/count.py
"""Count and peakiness losses for self-supervised probe detection."""
import torch
import torch.nn.functional as F


def count_loss(heatmap: torch.Tensor, target_count: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth L1 loss between sum(heatmap) and target_count.
    
    Args:
        heatmap: [T] per-sample probability
        target_count: expected number of peaks (float for smoothness)
        mask: [T] valid-sample mask
    """
    if mask is not None:
        heatmap = heatmap * mask.float()
    predicted_count = heatmap.sum()
    target = torch.tensor(float(target_count), device=heatmap.device, dtype=heatmap.dtype)
    # Normalize by target to make scale-invariant
    return F.smooth_l1_loss(predicted_count, target) / max(target_count, 1.0)


def peakiness_regularizer(heatmap: torch.Tensor, window: int = 20) -> torch.Tensor:
    """L1 penalty on (1 - max_over_local_window).
    
    Encourages at least one sharp peak per window. Applied across all valid samples.
    """
    # Max-pool the heatmap with a sliding window
    padded = F.pad(heatmap.unsqueeze(0).unsqueeze(0), (window // 2, window // 2), mode='replicate')
    max_pooled = F.max_pool1d(padded, kernel_size=window, stride=1)[0, 0]
    # Trim to original length
    max_pooled = max_pooled[:heatmap.shape[0]]
    return (1.0 - max_pooled).mean()
```

**Step 3: Verify and commit**

```bash
pytest tests/test_losses/test_losses.py -v
git add src/mongoose/losses/count.py tests/test_losses/test_losses.py
git commit -m "feat: count loss and peakiness regularizer for self-supervised probe detection"
```

---

### Task R4: Ground truth builder revision + preprocessing

**Files:**
- Modify: `src/mongoose/data/ground_truth.py`
- Modify: `src/mongoose/data/preprocess.py`
- Modify: `tests/test_data/test_ground_truth.py`
- Modify: `tests/test_data/test_preprocess.py`

**Scope:**
1. `MoleculeGT` dataclass gains `reference_bp_positions` (from aligner + FASTA), drops `inter_probe_deltas_bp` (now computed from reference), drops `velocity_targets_bp_per_ms` (now computed from heatmap dynamically). Keeps optional `warmstart_probe_centers` (from wfmproc, used for warmstart phase only).
2. `build_molecule_gt` rewritten: takes the same inputs but produces the new structure. The reference probe lookup uses `assign.ref_index`, `assign.direction`, and a new helper that extracts the `[ref_start, ref_end]` interval from the alignment.
3. `preprocess.py` gains level-1 computation from TDB, caches the new GT structure, optionally caches warmstart labels.

**Detailed test-driven sequence:** (write tests first for each behavior change; I'll produce them in the implementer subagent prompt when this task executes)

**Commit:**

```bash
git commit -m "refactor: ground truth builder + preprocessing for reference-anchored labels"
```

---

### Task R5: Loss composition revision + warmstart scheduler

**Files:**
- Modify: `src/mongoose/losses/combined.py`
- Modify: `tests/test_losses/test_losses.py`
- Modify: `src/mongoose/training/config.py` (add `warmstart_epochs`)

**Scope:**
1. `CombinedLoss` accepts the new inputs: heatmap, pred_velocity, pred_cumulative_bp, reference_bp_positions_list, warmstart_heatmap_list (optional), warmstart_valid_list (bool), n_ref_probes, mask
2. Internally runs NMS on current heatmap to extract detected peak positions (no gradient through NMS)
3. Computes L_bp via soft-DTW between `pred_cumulative_bp[detected_peaks]` and `reference_bp_positions`, with position-normalized cost
4. Computes L_velocity at detected peaks with target = 511 / heatmap_FWHM (stop-gradient on target)
5. Computes L_count on sum(heatmap) vs n_ref_probes
6. L_probe depends on `is_warmstart_active(epoch)`: focal on warmstart Gaussian if yes, peakiness regularizer if no. Linear blend during transition epochs
7. `set_epoch(epoch)` method now drives both lambda warmup AND warmstart activity

**Commit:**

```bash
git commit -m "refactor: combined loss with soft-DTW, count loss, warmstart scheduler"
```

---

### Task R6: Dataset revision + training loop integration

**Files:**
- Modify: `src/mongoose/data/cached_dataset.py`
- Modify: `src/mongoose/data/collate.py`
- Modify: `src/mongoose/training/trainer.py`
- Modify: `tests/test_data/test_cached_dataset.py`
- Modify: `tests/test_training/test_trainer.py`

**Scope:**
1. `CachedMoleculeDataset.__getitem__` returns new dict schema (reference_bp_positions, n_ref_probes, optional warmstart_heatmap + warmstart_valid)
2. `collate_molecules` handles the new variable-length fields
3. `Trainer._step` passes the new tensors to `CombinedLoss`
4. `Trainer.fit` calls `criterion.set_epoch(epoch)` which drives warmstart scheduling
5. `SyntheticMoleculeDataset` updated to produce synthetic reference_bp_positions for smoke-test compatibility

**Commit:**

```bash
git commit -m "refactor: dataset + trainer integration for V1 rearchitecture"
```

---

### Task R7: End-to-end smoke test verification

**Files:**
- Modify: `scripts/train.py` (expose `--warmstart-epochs` CLI flag)
- Run: existing synthetic smoke test

**Steps:**
1. Run full test suite: `pytest -q` -- all 124+ tests should pass
2. Run synthetic smoke test: `python scripts/train.py --synthetic --epochs 3 --warmstart-epochs 1 --batch-size 2 --no-amp`
3. Verify no NaN losses, warmstart -> post-warmstart transition works, all four loss components report

**Commit:**

```bash
git commit -m "feat: V1 rearchitecture end-to-end smoke test passing"
```

---

## Execution Order

Task R1 and R2 can run in parallel (independent code paths).
Task R3 depends on neither R1 nor R2 -- can also run in parallel.
Task R4 depends on R1 (level-1 used in preprocessing).
Task R5 depends on R2, R3 (uses both losses).
Task R6 depends on R4, R5.
Task R7 is the final verification gate.

Recommended sequential order (conservative, avoids subagent parallelism conflicts):
R1 -> R2 -> R3 -> R4 -> R5 -> R6 -> R7

Or optimistic parallel batches:
Batch 1: R1, R2, R3 (three parallel subagents)
Batch 2: R4 (sequential, depends on R1)
Batch 3: R5 (sequential, depends on R2 and R3)
Batch 4: R6 (sequential, depends on R4 and R5)
Batch 5: R7 (final)

## Success Gate

All of the following must hold before this branch is considered ready for cloud training:

1. Full test suite passes (124 existing + new tests for R1-R5)
2. Synthetic smoke test runs 3 epochs with warmstart=1 without NaN losses
3. The four loss components (L_probe, L_bp, L_velocity, L_count) all report non-zero, non-NaN values after epoch 2
4. Warmstart -> post-warmstart transition visible in loss trajectory (L_probe changes character around the transition)
5. No existing inference or architecture code was modified (only data pipeline, losses, and trainer)

## Out of Scope for This Plan

- Running real-data preprocessing (that's the desktop playbook)
- Cloud training (that's the $75 experiment)
- Evaluation script changes (V2 evaluation is similar enough to V1 original; changes deferred until after first training results are in)
- Full probes.bin removal from preprocess cache (we still cache wfmproc probe centers for the warmstart phase; can remove after we confirm warmstart isn't needed)
