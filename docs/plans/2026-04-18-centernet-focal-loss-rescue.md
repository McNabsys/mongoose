# CenterNet Focal Loss Rescue — Phase 2a Probe Head Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the positive-weighted BCE probe loss in `CombinedLoss` with a CenterNet-style focal loss that normalizes by number of positives per molecule, then gate on the overfit-one-batch sanity test before attempting another Phase 2a smoke run.

**Architecture:** Add a new `centernet_focal_loss` function to `src/mongoose/losses/centernet_focal.py` that consumes per-molecule `[T]` tensors (logits, target, mask). Splice it into `CombinedLoss.__call__` in place of the existing BCE/MSE branches. Remove the now-dead `probe_pos_weight` parameter and its CLI flag. Verify via TDD, the existing `tests/test_losses/` suite, and `scripts/overfit_one_batch.py`.

**Tech Stack:** PyTorch 2.11+cu128, Python 3.12, pytest. Training in `torch.amp.autocast('cuda', dtype=torch.bfloat16)`.

**Context this plan assumes the engineer does NOT have:**
- Prior conversation with Deep Think. See `deep_think_prompt_DRAFT.md` (the peer-review question) and `C:/Users/jonmc/Downloads/U-Net Peak Detection Debugging.md` + `C:/Users/jonmc/Downloads/Debugging Probe Localization Loss.md` (Deep Think's responses) for full context on why this is the right fix.
- Observation from Phase 2a training log: `raw_probe` stuck at ~1.13 for all 10 epochs. Best-model F1 = 0.033, recall = 0.017. The probe head never learned peak localization.

**What is intentionally NOT in this plan:**
- Teacher forcing L_bp/L_vel at ground-truth reference sample indices. This requires plumbing `probe_centers_samples` through `CachedMoleculeDataset` → `collate_molecules` → `CombinedLoss.__call__`, which is a data-contract refactor. Deferred to a follow-up plan so we can iterate on the loss math in isolation.
- Correcting the L_count target from `n_ref_probes` to `sum(target_heatmap)`. The immediate fix is to disable L_count entirely at the CLI (`--scale-count 1e9`). The target-correction fix can be a follow-up once we know the CenterNet change works.

---

## File Structure

**Created:**
- `src/mongoose/losses/centernet_focal.py` — new module with `centernet_focal_loss` function
- `tests/test_losses/test_centernet_focal.py` — unit tests for the new function

**Modified:**
- `src/mongoose/losses/combined.py:40-305` — swap BCE/MSE probe-term block for a call to `centernet_focal_loss`; remove `probe_pos_weight` parameter
- `src/mongoose/losses/__init__.py` — export `centernet_focal_loss`
- `src/mongoose/training/config.py` — remove `probe_pos_weight` field
- `src/mongoose/training/cli.py:197-198` — remove `--probe-pos-weight` CLI flag
- `src/mongoose/training/trainer.py` — remove `probe_pos_weight` from `CombinedLoss` kwargs
- `scripts/overfit_one_batch.py:52-53` — remove `--probe-pos-weight` arg
- `tests/test_losses/test_losses.py` — update `CombinedLoss` tests that referenced BCE behavior or `probe_pos_weight`
- `tests/test_training/` — update any fixture/config tests referencing `probe_pos_weight`

**Unchanged but invoked:**
- `scripts/train.py` — no code changes; will be invoked with new CLI flags in Task 8
- `scripts/evaluate_peak_match.py` — no code changes; invoked in Task 9

---

## Task 1: Add `centernet_focal_loss` function skeleton with first failing test

**Files:**
- Create: `src/mongoose/losses/centernet_focal.py`
- Create: `tests/test_losses/test_centernet_focal.py`

- [ ] **Step 1: Write the failing test for perfect prediction**

Create `tests/test_losses/test_centernet_focal.py`:

```python
"""Tests for CenterNet-style focal loss used on sparse 1-D probe heatmaps."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mongoose.losses.centernet_focal import centernet_focal_loss


def _gaussian_target(length: int, centers: list[int], sigma: float = 2.0) -> torch.Tensor:
    x = np.arange(length, dtype=np.float32)
    hm = np.zeros(length, dtype=np.float32)
    for c in centers:
        g = np.exp(-0.5 * ((x - float(c)) / sigma) ** 2)
        hm = np.maximum(hm, g)
    return torch.from_numpy(hm)


def test_centernet_focal_loss_perfect_prediction():
    target = _gaussian_target(length=100, centers=[20, 50, 80])
    # Predict logits that sigmoid to (approximately) the target distribution.
    # For target=1, sigmoid=1 means logit=+inf; approximate with large positive.
    logits = torch.where(target > 0.99, torch.tensor(8.0), torch.logit(target.clamp(1e-4, 0.9999)))
    mask = torch.ones(100, dtype=torch.bool)

    loss = centernet_focal_loss(logits, target, mask)
    assert loss.item() < 0.5, f"expected low loss for near-perfect prediction, got {loss.item()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_centernet_focal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mongoose.losses.centernet_focal'`

- [ ] **Step 3: Create the minimal implementation**

Create `src/mongoose/losses/centernet_focal.py`:

```python
"""CenterNet-style focal loss for sparse 1-D peak detection.

Per ``Zhou et al. 2019 (Objects as Points)``: the standard focal-loss penalty
for positive samples (target = 1 at exact peak center) is combined with a
penalty-reduction factor ``(1 - target)**beta`` for negative samples that fall
inside the Gaussian halo of a true peak.

The normalization divides the per-sample sum by the number of positive samples
in the molecule (not the sequence length), giving gradient strength that is
independent of padded/variable waveform length.

BF16 caveat: ``target.eq(1.0)`` is avoided because a ``wfmproc`` heatmap stored
as float32 and downcast inside ``torch.amp.autocast('cuda', dtype=bfloat16)``
may arrive at this loss slightly below exact 1.0 due to mantissa truncation.
``.ge(0.99)`` is used instead to identify peak-center samples.
"""
from __future__ import annotations

import torch


def centernet_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    pos_threshold: float = 0.99,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute CenterNet focal loss for a single molecule.

    Args:
        logits: ``[T]`` raw logits from the probe head (pre-sigmoid).
        target: ``[T]`` Gaussian heatmap target in ``[0, 1]``; values ``>= pos_threshold``
            are treated as peak-center samples.
        mask: ``[T]`` boolean mask, ``True`` at valid (non-padded) samples.
        alpha: Exponent on the prediction-confidence modulating factor.
        beta: Exponent on the Gaussian-halo penalty-reduction factor.
        pos_threshold: Minimum target value to count a sample as a positive
            peak-center (defaults to 0.99 to survive BF16 mantissa truncation).
        eps: Clamp applied to ``sigmoid(logits)`` before ``log`` for numerical
            stability.

    Returns:
        Scalar loss: sum over time divided by the number of positive samples
        (``clamp(min=1.0)`` to avoid division by zero when a molecule has no
        peaks inside the mask).
    """
    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    mask_f = mask.to(pred.dtype)
    target_f = target.to(pred.dtype)

    pos_inds = (target_f >= pos_threshold).to(pred.dtype) * mask_f
    neg_inds = (target_f < pos_threshold).to(pred.dtype) * mask_f

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_weights = torch.pow(1.0 - target_f, beta)
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum().clamp(min=1.0)
    return -(pos_loss + neg_loss).sum() / num_pos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_centernet_focal.py::test_centernet_focal_loss_perfect_prediction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mongoose/losses/centernet_focal.py tests/test_losses/test_centernet_focal.py
git commit -m "feat(loss): add CenterNet focal loss for sparse 1-D peak detection"
```

---

## Task 2: Add test for flat-zero prediction (worst case)

**Files:**
- Modify: `tests/test_losses/test_centernet_focal.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/test_losses/test_centernet_focal.py`:

```python
def test_centernet_focal_loss_flat_zero_prediction():
    target = _gaussian_target(length=100, centers=[20, 50, 80])
    # Model outputs near-zero probability everywhere: logit << 0
    logits = torch.full((100,), -5.0)
    mask = torch.ones(100, dtype=torch.bool)

    loss = centernet_focal_loss(logits, target, mask)
    # Three missed peaks should produce a substantial per-peak loss (>> perfect case)
    assert loss.item() > 2.0, f"expected high loss for flat-zero prediction, got {loss.item()}"
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_centernet_focal.py::test_centernet_focal_loss_flat_zero_prediction -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses/test_centernet_focal.py
git commit -m "test(loss): assert CenterNet focal loss penalizes flat-zero predictions"
```

---

## Task 3: Add test for per-molecule normalization (length invariance)

**Files:**
- Modify: `tests/test_losses/test_centernet_focal.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/test_losses/test_centernet_focal.py`:

```python
def test_centernet_focal_loss_length_invariant_with_same_num_positives():
    """Two molecules with same #peaks, same per-peak quality, different lengths
    should produce similar losses (modulo halo effect)."""
    target_short = _gaussian_target(length=100, centers=[20, 50, 80])
    target_long  = _gaussian_target(length=300, centers=[20, 50, 80])
    logits_short = torch.full((100,), -5.0)
    logits_long  = torch.full((300,), -5.0)
    mask_short = torch.ones(100, dtype=torch.bool)
    mask_long  = torch.ones(300, dtype=torch.bool)

    loss_short = centernet_focal_loss(logits_short, target_short, mask_short)
    loss_long  = centernet_focal_loss(logits_long,  target_long,  mask_long)

    # Losses should be within 25% of each other — not 3x as would happen
    # with a seq-length denominator.
    ratio = loss_long.item() / loss_short.item()
    assert 0.75 < ratio < 1.25, (
        f"length dilution detected: short={loss_short.item()}, long={loss_long.item()}, "
        f"ratio={ratio}"
    )
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_centernet_focal.py::test_centernet_focal_loss_length_invariant_with_same_num_positives -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses/test_centernet_focal.py
git commit -m "test(loss): assert CenterNet focal loss is sequence-length invariant per peak"
```

---

## Task 4: Add test for mask correctness

**Files:**
- Modify: `tests/test_losses/test_centernet_focal.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/test_losses/test_centernet_focal.py`:

```python
def test_centernet_focal_loss_masked_region_has_no_effect():
    target = _gaussian_target(length=100, centers=[20, 50, 80])
    logits = torch.full((100,), -5.0)
    mask_full = torch.ones(100, dtype=torch.bool)
    mask_half = torch.ones(100, dtype=torch.bool)
    mask_half[60:] = False  # drop the peak at 80 and everything after

    loss_full = centernet_focal_loss(logits, target, mask_full)
    loss_half = centernet_focal_loss(logits, target, mask_half)

    # Masking-out the region containing 1 of 3 peaks should change the loss.
    # Since num_pos drops from 3 to 2 and one missed peak's error is removed,
    # the result is close in magnitude but not equal.
    assert abs(loss_full.item() - loss_half.item()) > 1e-3, (
        f"mask had no effect: full={loss_full.item()}, half={loss_half.item()}"
    )
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_centernet_focal.py::test_centernet_focal_loss_masked_region_has_no_effect -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses/test_centernet_focal.py
git commit -m "test(loss): assert CenterNet focal loss honors per-sample mask"
```

---

## Task 5: Add test for BF16 autocast safety

**Files:**
- Modify: `tests/test_losses/test_centernet_focal.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/test_losses/test_centernet_focal.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for bf16 autocast")
def test_centernet_focal_loss_bf16_autocast_no_nan():
    target = _gaussian_target(length=200, centers=[30, 80, 130, 180]).cuda()
    logits = torch.randn(200, device="cuda", requires_grad=True)
    mask = torch.ones(200, dtype=torch.bool, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = centernet_focal_loss(logits, target, mask)
    loss.backward()

    assert torch.isfinite(loss), f"loss is not finite under bf16 autocast: {loss}"
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), (
        "gradient contains NaN or Inf under bf16 autocast"
    )
```

- [ ] **Step 2: Run to verify it passes on the GPU box**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/test_centernet_focal.py::test_centernet_focal_loss_bf16_autocast_no_nan -v`
Expected: PASS (or SKIPPED if no CUDA — that is fine; we will hit it on the training box)

- [ ] **Step 3: Commit**

```bash
git add tests/test_losses/test_centernet_focal.py
git commit -m "test(loss): assert CenterNet focal loss is stable under bf16 autocast"
```

---

## Task 6: Swap CenterNet focal loss into `CombinedLoss`, remove `probe_pos_weight`

**Files:**
- Modify: `src/mongoose/losses/combined.py:40-305`
- Modify: `src/mongoose/losses/__init__.py`

- [ ] **Step 1: Update `src/mongoose/losses/__init__.py` to export the new loss**

Edit `src/mongoose/losses/__init__.py`. Add the import near the other `from mongoose.losses.*` imports:

```python
from mongoose.losses.centernet_focal import centernet_focal_loss
```

And add `"centernet_focal_loss"` to `__all__` (keep alphabetical).

- [ ] **Step 2: Replace the BCE/MSE probe block in `CombinedLoss.__call__` with a CenterNet call**

Edit `src/mongoose/losses/combined.py`. Locate the block from line ~188 through ~226 starting with `# ----------------- L_probe -----------------` and ending with `probe_terms.append(probe_component)`. Replace with:

```python
            # ----------------- L_probe -----------------
            probe_component = torch.zeros((), device=device, dtype=pred_h_b.dtype)

            if (
                blend > 0.0
                and warmstart_heatmap is not None
                and warmstart_valid is not None
                and bool(warmstart_valid[b].item())
            ):
                if pred_heatmap_logits is None:
                    raise ValueError(
                        "CenterNet focal probe loss requires pred_heatmap_logits; "
                        "the model forward must return logits."
                    )
                logits_b = pred_heatmap_logits[b]
                target_b = warmstart_heatmap[b]
                probe_loss_val = centernet_focal_loss(logits_b, target_b, mask_b)
                probe_component = probe_component + blend * probe_loss_val

            if blend < 1.0:
                masked_heatmap = pred_h_b * mask_b.to(pred_h_b.dtype)
                peaky = peakiness_regularizer(masked_heatmap, window=self.peakiness_window)
                probe_component = probe_component + (1.0 - blend) * peaky

            probe_terms.append(probe_component)
```

- [ ] **Step 3: Add the import at the top of `combined.py`**

Edit `src/mongoose/losses/combined.py`. After the existing `from mongoose.losses.*` imports, add:

```python
from mongoose.losses.centernet_focal import centernet_focal_loss
```

- [ ] **Step 4: Remove `probe_pos_weight` from `CombinedLoss.__init__`**

Edit `src/mongoose/losses/combined.py`:

Remove the parameter from the constructor signature (`probe_pos_weight: float = 0.0,`), remove the assignment `self.probe_pos_weight = float(probe_pos_weight)`, and remove the focal-loss-era parameters that are now dead: `focal_gamma`, `focal_alpha`. Also update the module docstring at the top of the file to describe the CenterNet focal loss instead of "masked MSE against wfmproc Gaussian target".

Replacement docstring:

```python
"""Combined loss for V1 rearchitecture.

Composes four components per molecule:

- ``L_probe``: CenterNet-style focal loss against wfmproc Gaussian targets
  during warmstart, blended with the peakiness regularizer after warmstart
  ends. Positives are samples with target >= 0.99 (peak centers); negatives
  in the Gaussian halo are down-weighted by ``(1 - target)**beta``. Normalized
  per-molecule by ``num_positives`` to keep gradient strength independent of
  sequence length.
- ``L_bp``: soft-DTW between model-detected peaks in cumulative bp and the
  reference bp positions, zero-anchored and span-normalized.
- ``L_velocity``: MSE at detected peak positions with targets derived from
  heatmap FWHM (stop-grad on target).
- ``L_count``: smooth L1 on ``sum(heatmap)`` vs ``n_ref_probes``.

``set_epoch`` drives both the warmstart blend (L_probe character transition)
and the lambda scale (0.5 during warmstart, 1.0 after).
"""
```

- [ ] **Step 5: Run full loss-test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_losses/ -v`
Expected: all new tests PASS; pre-existing tests in `test_losses.py` may fail if they referenced `probe_pos_weight` or the BCE/MSE behavior. Task 7 fixes those.

- [ ] **Step 6: Commit**

```bash
git add src/mongoose/losses/combined.py src/mongoose/losses/__init__.py
git commit -m "feat(loss): swap probe BCE for CenterNet focal in CombinedLoss"
```

---

## Task 7: Remove `probe_pos_weight` from config, CLI, trainer, and tests

**Files:**
- Modify: `src/mongoose/training/config.py`
- Modify: `src/mongoose/training/cli.py:197-198`
- Modify: `src/mongoose/training/trainer.py`
- Modify: `scripts/overfit_one_batch.py:52-53`
- Modify: `tests/test_losses/test_losses.py`
- Modify: `tests/test_training/` (any file referencing `probe_pos_weight`)

- [ ] **Step 1: Find every reference that still mentions the removed parameter**

Run: `grep -rn probe_pos_weight src/ tests/ scripts/` (or `rg probe_pos_weight src/ tests/ scripts/`). Expected: hits in `config.py`, `cli.py` (args.probe_pos_weight + add_argument), `trainer.py` (loss kwargs), `overfit_one_batch.py` (arg + kwargs), `test_losses.py` (loss construction), and any test_training fixtures.

- [ ] **Step 2: Remove `probe_pos_weight` from `TrainConfig`**

Edit `src/mongoose/training/config.py`: remove the `probe_pos_weight: float = 0.0` field (or whatever its current default is).

- [ ] **Step 3: Remove the CLI flag**

Edit `src/mongoose/training/cli.py`: remove the `parser.add_argument("--probe-pos-weight", ...)` block (earlier in the file than line 197) and the `if args.probe_pos_weight is not None: config.probe_pos_weight = args.probe_pos_weight` block at lines 197-198.

- [ ] **Step 4: Remove from trainer's loss construction**

Edit `src/mongoose/training/trainer.py`: find where `CombinedLoss(...)` is instantiated and remove `probe_pos_weight=config.probe_pos_weight` from the kwargs. Same for any `focal_gamma=...` / `focal_alpha=...` kwargs if they are passed (they were already dead but may still be wired).

- [ ] **Step 5: Remove from overfit_one_batch.py**

Edit `scripts/overfit_one_batch.py`:
- Remove the `parser.add_argument("--probe-pos-weight", ...)` block at lines 52-53.
- Remove `probe_pos_weight=args.probe_pos_weight` from the `CombinedLoss(...)` construction further down in the file.

- [ ] **Step 6: Update existing loss tests**

Edit `tests/test_losses/test_losses.py`:
- Any `CombinedLoss(..., probe_pos_weight=...)` call: remove the kwarg.
- Any assertion that relied on specific BCE loss magnitudes (search for `probe` in assertions and sanity-check).
- Any test that was specifically testing BCE behavior (`test_combined_loss_probe_bce_branch`, etc.): delete it — it's no longer applicable.

- [ ] **Step 7: Update training tests**

Search `tests/test_training/` for `probe_pos_weight`. Remove references. If a fixture defaulted `probe_pos_weight=50.0`, just drop the field.

- [ ] **Step 8: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v -x`
Expected: all 221 tests pass (minus any that were specifically BCE-branch-dependent and got deleted in Step 6). No failures. No NaN warnings.

- [ ] **Step 9: Commit**

```bash
git add src/mongoose/training/config.py src/mongoose/training/cli.py src/mongoose/training/trainer.py scripts/overfit_one_batch.py tests/
git commit -m "refactor(loss): drop probe_pos_weight config/CLI/tests (BCE fallback gone)"
```

---

## Task 8: Run overfit-one-batch gate with the rescue recipe

**Files:**
- No code changes; this is a verification task.

**Gate criteria (from `scripts/overfit_one_batch.py` docstring):**
- `scaled bp < 0.1` by step 300
- `probe < 0.1` by step 300
- no NaN / Inf at any step
- viz shows sharp peaks aligned with reference

With the CenterNet loss in place, we should also see `raw_probe` dropping from its ~1.1 floor within the first ~50 steps — that is the load-bearing signal that the fix worked.

- [ ] **Step 1: Sanity-check the cache path resolves**

Run: `ls "E. coli/cache/STB03-060A-02L58270w05-433B23e/" | head -3`
Expected: a few `molecule_*.pt` files (or equivalent). If the path is wrong or the cache is gone, fix before continuing.

- [ ] **Step 2: Run the gate with the rescue recipe**

Run (from repo root):

```bash
.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --steps 300 --batch-size 32 --lr 3e-4 \
    --scale-bp 300000 --scale-vel 5000 \
    --scale-count 1e9 --scale-probe 1.0 \
    --min-blend 1.0 \
    --warmstart-epochs 1 --warmstart-fade-epochs 0 \
    --output-viz overfit_gate_centernet.png \
    2>&1 | tee overfit_gate_centernet.log
```

Rationale for the flags:
- `--scale-count 1e9` → effectively kills L_count (Deep Think's Rank 1: L_count target contradicts Gaussian area).
- `--min-blend 1.0` → keeps the CenterNet probe supervision at full strength forever (Deep Think's Q4 answer).
- `--warmstart-epochs 1 --warmstart-fade-epochs 0` → no blend fade (belt-and-braces with `--min-blend 1.0`).

Expected output (in the last few log lines):
- `probe` (scaled) < 0.1
- `bp` (scaled) < 0.1
- no `nan` or `inf` in any step line

- [ ] **Step 3: Inspect the viz**

Open `overfit_gate_centernet.png`. Expected: sharp predicted peaks (not a flat line near zero) visually aligned with the reference-probe markers.

- [ ] **Step 4: Decide whether to proceed**

- **If gate passes:** proceed to Task 9 (Phase 2a rerun). Note the final step's `raw_probe` — that is our new baseline.
- **If gate fails (probe still stuck high, or viz flat):** STOP. Do not launch Phase 2a. Re-read Deep Think's response and the loss code; look for either (a) a bug in the CenterNet implementation (check `num_pos` is really being computed per-molecule), (b) a dtype contamination issue surviving into the loss (print `target_b.dtype`, `logits_b.dtype`), (c) a mask issue (zero valid samples in the batch?).

- [ ] **Step 5: Commit the gate artifacts**

```bash
git add overfit_gate_centernet.log overfit_gate_centernet.png
git commit -m "chore(training): overfit-one-batch gate artifacts with CenterNet probe loss"
```

---

## Task 9: Relaunch Phase 2a smoke run with the rescue recipe

**Files:**
- No code changes; this is a verification task.

**Gate criteria:** by epoch 10, peak-match F1 on the held-out split should be meaningfully above the Phase 2a-v1 baseline of **F1 = 0.033 / Recall = 0.017**. A plausible success bar is F1 > 0.3 and Recall > 0.3, but any substantial lift from 0.017 recall is evidence the recipe works.

**Do not launch Phase 4 automatically on success.** Report results and wait for explicit user approval before launching any 35-epoch run. This rule is load-bearing; see `SESSION_STATE.md` §13.

- [ ] **Step 1: Archive the v1 Phase 2a outputs**

Preserve the old run so it can be compared against. Run:

```bash
mv phase2a_checkpoints phase2a_v1_bce_checkpoints
mv phase2a_train.log phase2a_v1_bce_train.log
mv phase2a_eval.json phase2a_v1_bce_eval.json
mv phase2a_viz.png phase2a_v1_bce_viz.png 2>/dev/null || true
```

- [ ] **Step 2: Launch Phase 2a v2 with the rescue recipe**

Run:

```bash
.venv/Scripts/python.exe -u scripts/train.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --epochs 10 --batch-size 32 --lr 3e-4 \
    --warmstart-epochs 10 --warmstart-fade-epochs 0 \
    --min-blend 1.0 \
    --scale-bp 300000 --scale-vel 5000 \
    --scale-count 1e9 --scale-probe 1.0 \
    --min-lr 1.5e-5 \
    --checkpoint-dir phase2a_checkpoints --save-every 1 \
    2>&1 | tee phase2a_train.log
```

Differences from the v1 Phase 2a command:
- `--warmstart-epochs 10 --warmstart-fade-epochs 0` (keep CenterNet supervision live for the whole run instead of fading after epoch 4).
- `--min-blend 1.0` (belt-and-braces against any future schedule change).
- `--scale-count 1e9` (disable L_count).
- Dropped `--probe-pos-weight 50` (parameter no longer exists).

Expected wall time: ~25–33 min/epoch × 10 = ~4–5 hr. Let it run; do not kill it early unless `raw_probe` is still flat by epoch 3.

- [ ] **Step 3: Run evaluation and viz on the best model**

Once training completes, run:

```bash
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint phase2a_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2a_eval.json

.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
    --checkpoint phase2a_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2a_viz.png --n-molecules 20 --seed 42
```

Print the F1 summary:

```bash
.venv/Scripts/python.exe -c "import json; d=json.load(open('phase2a_eval.json')); o=d['overall']; m=o['per_molecule_mean']; print(f'F1_sum={o[\"f1\"]:.3f}  F1_mol_mean={m[\"f1\"]:.3f}  P={o[\"precision\"]:.3f} R={o[\"recall\"]:.3f}  n={m[\"n_molecules\"]}')"
```

- [ ] **Step 4: Report to the user and STOP**

Report:
- Per-epoch `raw_probe` trajectory (did it drop from 1.1?).
- Final F1, Recall, Precision.
- Viz artifact path.
- Comparison to v1 Phase 2a (F1=0.033, R=0.017).

**Do not launch Phase 4.** Wait for user approval. (See `SESSION_STATE.md` §13.)

- [ ] **Step 5: Commit Phase 2a v2 artifacts**

```bash
git add phase2a_train.log phase2a_eval.json phase2a_viz.png phase2a_v1_bce_*
git commit -m "chore(training): Phase 2a v2 artifacts (CenterNet focal loss)"
```

---

## Rollback Plan

If Task 8 gate fails and the cause turns out to be a fundamental issue with the CenterNet formulation (not a bug in our implementation), revert Tasks 1-7 with:

```bash
git log --oneline  # find the commit just before Task 1
git revert <hash>..HEAD  # or git reset --hard to that hash if no one else has pulled
```

The original BCE loss is preserved in git history at `d45e9fb`.
