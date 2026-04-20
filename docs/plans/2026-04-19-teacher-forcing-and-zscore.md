# Teacher Forcing + Z-score Normalization + Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two remaining bottlenecks flagged by Deep Think (round 2): (a) post-normalize waveform amplitudes at ~1e-4 are incompatible with BF16 autocast and PyTorch weight init assumptions, and (b) L_bp / L_vel can't train the velocity head until the probe head crosses the NMS threshold, producing a "reward-hacking" oscillation. Then launch a 5-epoch multi-run smoke on 27 of 30 cached runs with 1 holdout per color.

**Architecture:** (1) Dynamic per-molecule z-score standardization of the valid-mask region in `collate_molecules`, no cache rebuild needed. (2) Plumb `warmstart_probe_centers_samples` through the trainer to `CombinedLoss.__call__`, and rewrite the L_bp and L_vel blocks to evaluate at those ground-truth indices (teacher forcing) instead of at NMS-detected peaks. (3) One-shot overfit gate with physics ON to confirm both heads train simultaneously, then launch Phase 6.

**Tech Stack:** PyTorch 2.11+cu128, Python 3.12, bfloat16 autocast on CUDA, pytest. Branch `claude/peaceful-rubin-bfb7a9` in worktree `C:\git\mongoose\.claude\worktrees\peaceful-rubin-bfb7a9`.

**Context assumed not in the implementer's head:**

- The label-mapping bug was fixed in commit `26f84ec` (see `docs/plans/2026-04-18-centernet-focal-loss-rescue.md` and the two Deep Think response files in `C:/Users/jonmc/Downloads/`). All 30 caches have been regenerated.
- The CenterNet focal probe loss at `src/mongoose/losses/centernet_focal.py` is proven to converge on the overfit-one-batch gate: probe went 6.19 → 0.066 in 300 steps with `--lambda-bp 0 --lambda-vel 0`.
- The overfit-one-batch gate with physics ON (`--lambda-bp 1 --lambda-vel 1`) shows a "reward-hacking" event at step 100–300: probe drops to 0.30 then regresses to 0.70 as the velocity head's random-init outputs produce chaotic gradients that disrupt the shared encoder. Teacher forcing bypasses this entirely.

**Python invocation pattern for this plan:**

```bash
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe <script>
```

---

## File Structure

**Modified:**
- `src/mongoose/data/collate.py` — add per-molecule z-score on the valid-mask region after padding.
- `src/mongoose/losses/combined.py` — accept new `warmstart_probe_centers_samples_list` parameter; rewrite L_bp and L_vel blocks to teacher-force from that list; keep the existing NMS branch intact for the old pre-teacher-forcing path, used only if the parameter is `None`.
- `src/mongoose/training/trainer.py` — thread `warmstart_probe_centers_samples` from the batch dict to `CombinedLoss.__call__`.
- `scripts/overfit_one_batch.py` — same threading.

**Created:**
- `tests/test_data/test_collate.py` — new file with z-score tests (the existing `test_cached_dataset.py` covers the `cached_dataset → collate` path end-to-end but not collate-only unit tests).
- `scripts/plot_random_dataloader_batch.py` — diagnostic that pulls a random post-collate batch, plots waveform[0] with warmstart_heatmap[0] overlaid, saves PNG. This is Deep Think's Q5 recommendation: a mandatory visual gate after any data-pipeline change.

**Unchanged but invoked:**
- `scripts/train.py` — no code changes; invoked with new CLI flags in Task 7.
- `scripts/evaluate_peak_match.py` — invoked in Task 7.

---

## Task 1: Z-score normalize waveforms in `collate_molecules`

**Files:**
- Create: `tests/test_data/test_collate.py`
- Modify: `src/mongoose/data/collate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data/test_collate.py`:

```python
"""Unit tests for the variable-length batch collator."""
from __future__ import annotations

import torch

from mongoose.data.collate import collate_molecules


def _make_item(t: int, wave_scale: float = 1e-4, wave_offset: float = 5e-4, *,
               centers: list[int] | None = None, warmstart: bool = True):
    """Synthesize one dataset item. Default mimics the real broken scale where
    post-normalize amplitudes land at ~1e-4."""
    waveform = torch.randn(1, t) * wave_scale + wave_offset
    mask = torch.ones(t, dtype=torch.bool)
    conditioning = torch.zeros(6, dtype=torch.float32)
    ref_bp = torch.tensor([100, 200, 300], dtype=torch.long)
    item = {
        "waveform": waveform,
        "conditioning": conditioning,
        "mask": mask,
        "reference_bp_positions": ref_bp,
        "n_ref_probes": torch.tensor(3, dtype=torch.long),
        "molecule_uid": 0,
    }
    if warmstart:
        item["warmstart_heatmap"] = torch.zeros(t, dtype=torch.float32)
        item["warmstart_valid"] = torch.tensor(True, dtype=torch.bool)
    else:
        item["warmstart_heatmap"] = None
        item["warmstart_valid"] = torch.tensor(False, dtype=torch.bool)
    item["warmstart_probe_centers_samples"] = (
        torch.tensor(centers or [10, 50, 100], dtype=torch.long)
    )
    return item


def test_zscore_valid_region_has_unit_stats():
    """After collate, each molecule's valid (mask=True) waveform region should
    have mean ~0 and std ~1 — regardless of the input scale."""
    items = [
        _make_item(t=500, wave_scale=1e-4, wave_offset=5e-4),
        _make_item(t=750, wave_scale=2e-4, wave_offset=3e-4),
        _make_item(t=300, wave_scale=5e-5, wave_offset=1e-3),
    ]
    batch = collate_molecules(items)
    wf = batch["waveform"]  # [B, 1, T]
    mask = batch["mask"]    # [B, T]
    for i in range(wf.shape[0]):
        m = mask[i]
        v = wf[i, 0, m]
        assert abs(float(v.mean())) < 1e-5, (
            f"molecule {i}: post-collate mean should be ~0, got {float(v.mean())}"
        )
        assert abs(float(v.std()) - 1.0) < 1e-3, (
            f"molecule {i}: post-collate std should be ~1, got {float(v.std())}"
        )


def test_zscore_does_not_affect_padding_region():
    """Samples outside the mask (padding) should stay at zero after z-score."""
    items = [
        _make_item(t=500, wave_scale=1e-4),
        _make_item(t=300, wave_scale=1e-4),
    ]
    batch = collate_molecules(items)
    wf = batch["waveform"]
    mask = batch["mask"]
    # Second molecule has t=300; samples [300, padded_len) are padding and
    # should remain exactly 0 (their pre-pad value from torch.zeros init).
    pad_region = wf[1, 0, ~mask[1]]
    assert pad_region.numel() > 0, "expected some padding for shorter molecule"
    assert torch.all(pad_region == 0.0), (
        f"padding region should be zeros, found max abs = {pad_region.abs().max()}"
    )


def test_zscore_constant_waveform_does_not_nan():
    """A molecule whose valid signal is all-constant (std=0) must not divide
    by zero. Eps in the denominator should leave the signal finite."""
    item = _make_item(t=200, wave_scale=0.0, wave_offset=7.0)
    batch = collate_molecules([item])
    wf = batch["waveform"]
    assert torch.isfinite(wf).all(), "NaN/Inf in z-scored constant waveform"
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_data/test_collate.py -v
```
Expected: `test_zscore_valid_region_has_unit_stats` and `test_zscore_constant_waveform_does_not_nan` FAIL (the current collate doesn't normalize). `test_zscore_does_not_affect_padding_region` may pass by coincidence (torch.zeros init) but doesn't yet verify the interaction.

- [ ] **Step 3: Implement z-score in collate**

Edit `src/mongoose/data/collate.py`. Locate the block that populates `waveforms[i, :, :t] = item["waveform"]` and add the z-score immediately after:

```python
    for i, item in enumerate(items):
        t = item["waveform"].shape[-1]
        waveforms[i, :, :t] = item["waveform"]
        masks[i, :t] = item["mask"]
        if all_have_warmstart:
            warmstart_heatmaps[i, :t] = item["warmstart_heatmap"]  # type: ignore[index]

        # Per-molecule z-score on the valid-mask region only. Padding stays at 0.
        # Rationale: post-preprocess waveform amplitudes come out at ~1e-4
        # (unit scaling in preprocess is suspect), which is at the BF16
        # precision floor and incompatible with Kaiming/Xavier N(0,1) init
        # assumptions. Normalizing here is cheaper than rebuilding 1.25M
        # cached molecules and is robust to whatever units the raw waveform
        # ends up in.
        valid = waveforms[i, 0, :t]
        std = valid.std().clamp(min=1e-8)
        waveforms[i, 0, :t] = (valid - valid.mean()) / std
```

- [ ] **Step 4: Run tests to confirm they pass**

```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_data/test_collate.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Run the full test suite to check nothing else broke**

```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_data/ tests/test_losses/ tests/test_training/ tests/test_model/ -v --ignore=tests/test_data/test_ground_truth.py
```

Note: `tests/test_data/test_ground_truth.py`, `tests/test_io/`, `tests/test_data/test_cached_dataset.py`, and a few others depend on fixture data that's missing from the worktree — those are known pre-existing FileNotFoundError failures, NOT caused by this change. Limit the expected-green set to what runs without fixture data.

Expected: all tests in the included paths pass. If any test_losses or test_training tests fail with shape mismatches or statistic assertion failures, investigate — it's likely a fixture that assumed raw-amplitude waveforms and needs its expected values updated for z-scored inputs. Report any such failures in the status report; we'll triage case by case.

- [ ] **Step 6: Commit**

```bash
git add src/mongoose/data/collate.py tests/test_data/test_collate.py
git commit -m "feat(collate): z-score waveform valid region per molecule"
```

---

## Task 2: Accept `warmstart_probe_centers_samples_list` in `CombinedLoss.__call__`

**Files:**
- Modify: `src/mongoose/losses/combined.py`

This task only adds the PARAMETER and a docstring; it doesn't yet use it. That way Task 3 can fail-fast if the parameter isn't threaded through, but existing callers don't break yet.

- [ ] **Step 1: Add the optional parameter to the signature**

Edit `src/mongoose/losses/combined.py`. In the `__call__` method signature (around line 133–144), add a new keyword-only parameter after `pred_heatmap_logits`:

```python
    def __call__(
        self,
        pred_heatmap: torch.Tensor,
        pred_cumulative_bp: torch.Tensor,
        raw_velocity: torch.Tensor,
        reference_bp_positions_list: list[torch.Tensor],
        n_ref_probes: torch.Tensor,
        warmstart_heatmap: torch.Tensor | None,
        warmstart_valid: torch.Tensor | None,
        mask: torch.Tensor,
        pred_heatmap_logits: torch.Tensor | None = None,
        warmstart_probe_centers_samples_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
```

Also extend the docstring to document the new parameter:

```
            warmstart_probe_centers_samples_list: Optional list of per-molecule
                ground-truth probe center sample indices (``LongTensor[K_i]``).
                When provided, L_bp and L_vel are evaluated at these indices
                (teacher forcing) instead of at NMS-detected peaks. This
                bypasses the non-differentiable NMS gate and lets the velocity
                head receive flawless supervision from step 1 regardless of
                the probe head's state. When ``None`` (or when a molecule's
                entry is ``None``), falls back to the legacy NMS-detected
                branch.
```

- [ ] **Step 2: Run the loss tests to confirm no breakage**

```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_losses/ -v
```

Expected: all pass. No behavior change yet, just a parameter addition.

- [ ] **Step 3: Commit**

```bash
git add src/mongoose/losses/combined.py
git commit -m "feat(loss): add optional warmstart_probe_centers_samples_list to CombinedLoss"
```

---

## Task 3: Teacher-force L_vel at ground-truth indices

**Files:**
- Modify: `src/mongoose/losses/combined.py`
- Modify: `tests/test_losses/test_losses.py`

Deep Think's argument for this: when `warmstart_probe_centers_samples_list` is provided, we know exactly where the peaks SHOULD be. Evaluating the velocity loss at those positions (instead of at NMS-detected predicted positions) gives the velocity head gradient from step 1 regardless of probe-head output. The existing legacy NMS branch stays for the `None` case so no caller gets silently broken.

- [ ] **Step 1: Write a failing test that forces gradient into the velocity head with a flat probe output**

Append to `tests/test_losses/test_losses.py`:

```python
def test_combined_loss_teacher_forcing_gives_vel_gradient_with_flat_probe():
    """Given a flat probe heatmap (no peaks above NMS threshold), teacher
    forcing should still produce a nonzero L_vel gradient into ``raw_velocity``
    via the ground-truth index path. Without teacher forcing, L_vel is zero
    because extract_peak_indices returns <2 peaks."""
    import torch
    from mongoose.losses.combined import CombinedLoss

    B, T = 2, 200
    # Flat probe output — sigmoid = 0.047 everywhere, no peaks detected.
    pred_heatmap = torch.full((B, T), 0.047, requires_grad=False)
    pred_heatmap_logits = torch.full((B, T), -3.0, requires_grad=False)
    raw_velocity = torch.rand((B, T), requires_grad=True)
    pred_cumulative_bp = torch.cumsum(raw_velocity, dim=-1)
    mask = torch.ones(B, T, dtype=torch.bool)
    warmstart_heatmap = torch.zeros(B, T)
    warmstart_heatmap[0, [50, 100, 150]] = 1.0
    warmstart_heatmap[1, [60, 120]] = 1.0
    warmstart_valid = torch.tensor([True, True])
    ref_bp = [torch.tensor([0, 100, 200], dtype=torch.long),
              torch.tensor([0, 100], dtype=torch.long)]
    n_ref = torch.tensor([3, 2], dtype=torch.long)
    centers = [torch.tensor([50, 100, 150], dtype=torch.long),
               torch.tensor([60, 120], dtype=torch.long)]

    loss_fn = CombinedLoss(
        scale_bp=300000.0, scale_vel=5000.0, scale_count=1e9, scale_probe=1.0,
        lambda_bp=1.0, lambda_vel=1.0, lambda_count=0.0,
        warmstart_epochs=1, warmstart_fade_epochs=0, min_blend=1.0,
    )
    loss_fn.set_epoch(0)

    total, details = loss_fn(
        pred_heatmap=pred_heatmap,
        pred_cumulative_bp=pred_cumulative_bp,
        raw_velocity=raw_velocity,
        reference_bp_positions_list=ref_bp,
        n_ref_probes=n_ref,
        warmstart_heatmap=warmstart_heatmap,
        warmstart_valid=warmstart_valid,
        mask=mask,
        pred_heatmap_logits=pred_heatmap_logits,
        warmstart_probe_centers_samples_list=centers,
    )
    total.backward()

    assert raw_velocity.grad is not None, "raw_velocity should have received gradient"
    assert raw_velocity.grad.abs().sum().item() > 0.0, (
        "teacher forcing should produce nonzero velocity gradient even with flat probe"
    )
    assert details["vel_raw"] > 0.0, (
        f"expected nonzero raw L_vel under teacher forcing, got {details['vel_raw']}"
    )
```

- [ ] **Step 2: Run to confirm it FAILS**

```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_losses/test_losses.py::test_combined_loss_teacher_forcing_gives_vel_gradient_with_flat_probe -v
```
Expected: FAIL with `vel_raw == 0.0` (the current NMS branch returns zero for flat probe, and the teacher-forced branch doesn't exist yet).

- [ ] **Step 3: Implement teacher-forced L_vel**

Edit `src/mongoose/losses/combined.py`. Locate the per-molecule loop at the top of `__call__` where L_bp and L_vel are computed. Replace the `# ----------------- Detect peaks (no grad) -----------------` through `# -------------- L_velocity (dense at peaks) --------------` block so that when `warmstart_probe_centers_samples_list is not None and warmstart_probe_centers_samples_list[b] is not None`, we use the ground-truth indices. Keep the legacy NMS branch as the `else` case for backward compatibility.

The replacement shape is:

```python
            # Decide between teacher forcing and NMS-detected peak extraction.
            gt_centers: torch.Tensor | None = None
            if (
                warmstart_probe_centers_samples_list is not None
                and warmstart_probe_centers_samples_list[b] is not None
            ):
                gt_centers_raw = warmstart_probe_centers_samples_list[b]
                if gt_centers_raw.numel() >= 2:
                    gt_centers = gt_centers_raw.to(device=device, dtype=torch.long)

            if gt_centers is not None:
                # ---- Teacher-forced path ----
                # Clamp to valid range before gather (defensive; preprocess
                # should ensure all centers fall inside the waveform).
                gt_centers = gt_centers.clamp(0, pred_h_b.shape[0] - 1)

                # L_bp: soft-DTW between model-predicted cumulative-bp AT THE
                # GROUND-TRUTH SAMPLE INDICES and the reference bp positions.
                # Indices are the same length as ref_bp, so DTW is 1:1 and
                # could reduce to Huber; keeping soft-DTW for symmetry with
                # the inference path.
                pred_bp_at_peaks = pred_bp_b[gt_centers]
                ref_bp = reference_bp_positions_list[b]
                if ref_bp.numel() >= 2:
                    ref_bp_f = ref_bp.to(device=device, dtype=pred_bp_at_peaks.dtype)
                    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
                    ref_norm = (ref_bp_f - ref_bp_f[0]).abs()
                    span = (ref_bp_f[-1] - ref_bp_f[0]).abs().clamp(min=1.0)
                    dtw = soft_dtw(pred_norm, ref_norm, gamma=self.softdtw_gamma)
                    bp_terms.append(dtw / span)

                # L_vel: MSE between raw_velocity at the ground-truth indices
                # and the per-peak target derived from the GROUND-TRUTH
                # heatmap's FWHM at those same indices. Since target heatmaps
                # are Gaussians centered at the ground truth, we can measure
                # widths directly from the warmstart heatmap.
                if warmstart_heatmap is not None:
                    ws_b = warmstart_heatmap[b]
                    widths_samples = measure_peak_widths_samples(
                        ws_b, gt_centers, threshold_frac=0.5
                    )
                else:
                    # Falls back to predicted widths — less accurate but keeps
                    # the path defined for unit tests that omit warmstart.
                    widths_samples = measure_peak_widths_samples(
                        pred_h_b, gt_centers, threshold_frac=0.5
                    )
                widths_ms = widths_samples.to(
                    device=device, dtype=raw_v_b.dtype
                ) * float(self.sample_period_ms)
                widths_ms = torch.clamp(widths_ms, min=1e-6)
                target_v = (
                    float(self.tag_width_bp) / widths_ms * float(self.sample_period_ms)
                )
                pred_v_at_peaks = raw_v_b[gt_centers]
                vel_terms.append(F.mse_loss(pred_v_at_peaks, target_v.detach()))
            else:
                # ---- Legacy NMS-detected peaks path (unchanged) ----
                peak_indices = extract_peak_indices(
                    pred_h_b,
                    raw_v_b,
                    threshold=self.nms_threshold,
                    tag_width_bp=self.tag_width_bp,
                )

                ref_bp = reference_bp_positions_list[b]

                if peak_indices.numel() >= 2 and ref_bp.numel() >= 2:
                    pred_bp_at_peaks = pred_bp_b[peak_indices]
                    ref_bp_f = ref_bp.to(device=device, dtype=pred_bp_at_peaks.dtype)
                    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
                    ref_norm = (ref_bp_f - ref_bp_f[0]).abs()
                    span = (ref_bp_f[-1] - ref_bp_f[0]).abs()
                    span = torch.clamp(span, min=1.0)
                    dtw = soft_dtw(pred_norm, ref_norm, gamma=self.softdtw_gamma)
                    bp_terms.append(dtw / span)

                    widths_samples = measure_peak_widths_samples(
                        pred_h_b, peak_indices, threshold_frac=0.5
                    )
                    widths_ms = widths_samples.to(
                        device=device, dtype=raw_v_b.dtype
                    ) * float(self.sample_period_ms)
                    widths_ms = torch.clamp(widths_ms, min=1e-6)
                    target_v = (
                        float(self.tag_width_bp) / widths_ms * float(self.sample_period_ms)
                    )
                    pred_v_at_peaks = raw_v_b[peak_indices]
                    vel_terms.append(F.mse_loss(pred_v_at_peaks, target_v.detach()))
```

Note: this reuses `measure_peak_widths_samples` and `soft_dtw` which are already imported in `combined.py`. `extract_peak_indices` is no longer called in the teacher-forced path but must remain imported for the legacy `else` branch.

- [ ] **Step 4: Run the new test to confirm it passes**

```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_losses/test_losses.py::test_combined_loss_teacher_forcing_gives_vel_gradient_with_flat_probe -v
```

Expected: PASS.

- [ ] **Step 5: Run the full loss test suite to confirm no regressions**

```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_losses/ -v
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/mongoose/losses/combined.py tests/test_losses/test_losses.py
git commit -m "feat(loss): teacher-force L_bp and L_vel at ground-truth indices"
```

---

## Task 4: Thread `warmstart_probe_centers_samples` through the trainer and overfit script

**Files:**
- Modify: `src/mongoose/training/trainer.py`
- Modify: `scripts/overfit_one_batch.py`

The collate function already returns this field as a list (see `src/mongoose/data/collate.py:95-97`). The trainer and the overfit script just need to read it from the batch dict and pass it to `CombinedLoss.__call__`.

- [ ] **Step 1: Modify `trainer.py`**

Edit `src/mongoose/training/trainer.py`, in `_step` (around line 295). After the existing `warmstart_valid` block, add:

```python
        warmstart_probe_centers_samples = batch.get("warmstart_probe_centers_samples")
        if warmstart_probe_centers_samples is not None:
            warmstart_probe_centers_samples = [
                (c.to(self.device) if c is not None else None)
                for c in warmstart_probe_centers_samples
            ]
```

Then in the `self.criterion(...)` call, add the kwarg:

```python
            loss, details = self.criterion(
                pred_heatmap=probe_heatmap.float(),
                pred_cumulative_bp=cumulative_bp.float(),
                raw_velocity=raw_velocity.float(),
                reference_bp_positions_list=reference_bp_positions_list,
                n_ref_probes=n_ref_probes,
                warmstart_heatmap=warmstart_heatmap,
                warmstart_valid=warmstart_valid,
                mask=mask,
                pred_heatmap_logits=probe_logits.float(),
                warmstart_probe_centers_samples_list=warmstart_probe_centers_samples,
            )
```

- [ ] **Step 2: Modify `scripts/overfit_one_batch.py`**

Find where `CombinedLoss.__call__` is invoked inside `main()` (the call accepting `pred_heatmap=`, `pred_cumulative_bp=` etc.). Add the same kwarg:

```python
            loss, details = criterion(
                pred_heatmap=probe_heatmap.float(),
                pred_cumulative_bp=cumulative_bp.float(),
                raw_velocity=raw_velocity.float(),
                reference_bp_positions_list=reference_bp_positions_list,
                n_ref_probes=n_ref_probes,
                warmstart_heatmap=warmstart_heatmap,
                warmstart_valid=warmstart_valid,
                mask=mask,
                pred_heatmap_logits=probe_logits.float(),
                warmstart_probe_centers_samples_list=[
                    (c.to(device) if c is not None else None)
                    for c in batch["warmstart_probe_centers_samples"]
                ],
            )
```

(Exact surrounding syntax may differ; match the existing indentation.)

- [ ] **Step 3: Run trainer and overfit smoke tests**

First the unit tests:
```
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -m pytest tests/test_training/ tests/test_losses/ -v
```

Then a fast overfit smoke (10 steps only, physics ON) to confirm the plumbing doesn't crash:
```
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "C:/git/mongoose/E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --steps 10 --batch-size 8 --lr 3e-4 \
    --scale-bp 300000 --scale-vel 5000 --scale-count 1e9 --scale-probe 1.0 \
    --lambda-bp 1.0 --lambda-vel 1.0 \
    --min-blend 1.0 \
    --warmstart-epochs 1 --warmstart-fade-epochs 0 \
    --output-viz /tmp/smoke_10step.png 2>&1 | tail -15
```

Expected: 10 lines of step output, finite loss values, no NaN / no crash. `vel` should be nonzero from step 1 (teacher forcing) — this is the key signal that the plumbing works.

- [ ] **Step 4: Commit**

```bash
git add src/mongoose/training/trainer.py scripts/overfit_one_batch.py
git commit -m "feat(training): thread warmstart_probe_centers_samples to CombinedLoss"
```

---

## Task 5: Create `scripts/plot_random_dataloader_batch.py` (Deep Think's Q5 gate)

**Files:**
- Create: `scripts/plot_random_dataloader_batch.py`

This is the mandatory post-data-pipeline-change visualization Deep Think recommended. It catches label-mapping bugs like the one we just fixed in 5 seconds instead of weeks.

- [ ] **Step 1: Write the script**

Create `scripts/plot_random_dataloader_batch.py`:

```python
"""Plot ONE random post-collate batch for visual data-pipeline inspection.

Usage:
    python scripts/plot_random_dataloader_batch.py \\
        --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \\
        --batch-size 8 --seed 42 \\
        --output dataloader_batch_check.png

Purpose (Deep Think's Q5): after any data-pipeline change (preprocess,
collate, dataset), run this. Eyeball the PNG. If the heatmap overlays
(green) do NOT land on the waveform peaks, the data pipeline is broken.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=Path, default=Path("dataloader_batch_check.png")
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ds = CachedMoleculeDataset([args.cache_dir], augment=False)
    indices = rng.choice(len(ds), size=args.batch_size, replace=False).tolist()
    batch = collate_molecules([ds[i] for i in indices])

    wf = batch["waveform"]            # [B, 1, T]
    mask = batch["mask"]              # [B, T]
    hm = batch.get("warmstart_heatmap")  # [B, T] or None
    centers_list = batch["warmstart_probe_centers_samples"]

    fig, axes = plt.subplots(args.batch_size, 1, figsize=(16, 2 * args.batch_size))
    if args.batch_size == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        m = mask[i].numpy()
        v_len = int(m.sum())
        t = np.arange(v_len)
        ax.plot(t, wf[i, 0, :v_len].numpy(), color="steelblue", linewidth=0.5)
        ax.set_ylabel(f"mol {indices[i]}", fontsize=8)
        if hm is not None:
            ax2 = ax.twinx()
            ax2.plot(t, hm[i, :v_len].numpy(), color="black", linewidth=0.6, alpha=0.7)
            ax2.set_ylim(0, 1.1)
            ax2.tick_params(labelsize=6)
        centers = centers_list[i]
        if centers is not None:
            for c in centers.numpy():
                if 0 <= c < v_len:
                    ax.axvline(int(c), color="seagreen", linewidth=0.8, alpha=0.75)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, v_len)
    axes[-1].set_xlabel("sample index (valid region only)")
    plt.suptitle(
        f"Random post-collate batch from {args.cache_dir.name} (seed={args.seed})\n"
        f"green = warmstart_probe_centers_samples, black = warmstart_heatmap",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=110, bbox_inches="tight")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and visually verify**

```
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe scripts/plot_random_dataloader_batch.py \
    --cache-dir "C:/git/mongoose/E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --batch-size 8 --seed 42 \
    --output dataloader_batch_check_after_zscore.png
```

Report the resulting image in your status. The implementer should LOOK at the image. For every panel, green lines should land on upward spikes of the (now z-scored) waveform. If any panel shows green lines on flat regions, STOP and report BLOCKED — the label bug has returned.

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_random_dataloader_batch.py
git commit -m "feat(tools): add post-collate random-batch viz for data-pipeline QA"
```

---

## Task 6: Final overfit gate — physics ON, teacher forcing ON

**Files:**
- No code changes; verification only.

This is the gate Deep Think specified: probe loss should drop below 0.1 AND the velocity head should converge simultaneously without the reward-hacking oscillation.

- [ ] **Step 1: Run the gate**

```
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -u scripts/overfit_one_batch.py \
    --cache-dir "C:/git/mongoose/E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --steps 300 --batch-size 32 --lr 3e-4 \
    --scale-bp 300000 --scale-vel 5000 \
    --scale-count 1e9 --scale-probe 1.0 \
    --lambda-bp 1.0 --lambda-vel 1.0 \
    --min-blend 1.0 \
    --warmstart-epochs 1 --warmstart-fade-epochs 0 \
    --output-viz overfit_gate_teacherforcing.png \
    2>&1 | tee overfit_gate_teacherforcing.log
```

Expected runtime: 2–3 min.

- [ ] **Step 2: Interpret the result**

Record the last 5 step lines and the viz state. Pass criteria:

- `probe` (scaled) at step 300 < 0.1 **AND**
- `vel` (scaled) descending monotonically from its initial value **AND**
- `bp` (scaled) descending monotonically from its initial value **AND**
- No NaN/Inf anywhere in the log **AND**
- Viz shows pred-peaks near the reference peaks on the first molecule.

If any criterion fails, STOP. Report DONE_WITH_CONCERNS with the log, don't proceed to Task 7. The controller will decide next step (e.g., bump LR, change λ weights, investigate).

- [ ] **Step 3: Commit the gate artifacts**

```bash
git add overfit_gate_teacherforcing.log overfit_gate_teacherforcing.png \
        dataloader_batch_check_after_zscore.png
git commit -m "chore(training): teacher-forcing overfit gate artifacts"
```

---

## Task 7: Launch Phase 6 multi-run smoke (5 epochs, 27 runs, 3 holdouts)

**Files:**
- No code changes; verification only.

This is the real payoff. With teacher-forced physics losses, normalized inputs, and correct labels, a 5-epoch multi-run training should now produce a meaningful F1 on held-out runs.

- [ ] **Step 1: Define the holdout set**

Hold out the alphabetically-last run from each color:
- Black: `STB03-063B-02L58270w05-433B23b`
- Blue:  `STB03-065H-02L58270w05-433H09j`
- Red:   `STB03-064D-02L58270w05-433H09d`

Verify these exist:
```
ls "C:/git/mongoose/E. coli/cache/STB03-063B-02L58270w05-433B23b/manifest.json"
ls "C:/git/mongoose/E. coli/cache/STB03-065H-02L58270w05-433H09j/manifest.json"
ls "C:/git/mongoose/E. coli/cache/STB03-064D-02L58270w05-433H09d/manifest.json"
```

- [ ] **Step 2: Build the `--cache-dir` list for training (27 runs)**

All cached run directories EXCEPT the three holdouts. Use bash:

```bash
TRAIN_ARGS=""
for d in "C:/git/mongoose/E. coli/cache"/*/; do
    run=$(basename "$d")
    case "$run" in
        STB03-063B-02L58270w05-433B23b|STB03-065H-02L58270w05-433H09j|STB03-064D-02L58270w05-433H09d)
            echo "HOLDOUT: $run"
            ;;
        *)
            TRAIN_ARGS="$TRAIN_ARGS --cache-dir $d"
            ;;
    esac
done
echo "TRAIN_ARGS len: $(echo $TRAIN_ARGS | wc -w)"  # expect 54 (27 * 2 flags)
```

- [ ] **Step 3: Launch training**

```bash
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe -u scripts/train.py \
    $TRAIN_ARGS \
    --epochs 5 --batch-size 32 --lr 3e-4 \
    --warmstart-epochs 5 --warmstart-fade-epochs 0 \
    --min-blend 1.0 \
    --scale-bp 300000 --scale-vel 5000 \
    --scale-count 1e9 --scale-probe 1.0 \
    --min-lr 1.5e-5 \
    --checkpoint-dir phase6_smoke_checkpoints --save-every 1 \
    2>&1 | tee phase6_smoke_train.log
```

ETA: roughly 4-6 hours (1.12M molecules × 5 epochs at the per-epoch rate we saw in Phase 2a v1, ~half an hour per epoch for single-cache 55k molecules × ~20 for the full 1.1M scale).

**Do NOT auto-launch Phase 4 or Phase 6 production (35 epochs) after this completes.** Report results and wait for explicit user approval.

- [ ] **Step 4: Evaluate on held-out runs**

For each of the 3 holdouts, run peak-match eval:

```bash
for holdout in STB03-063B-02L58270w05-433B23b STB03-065H-02L58270w05-433H09j STB03-064D-02L58270w05-433H09d; do
    PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
    C:/git/mongoose/.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
        --checkpoint phase6_smoke_checkpoints/best_model.pt \
        --cache-dir "C:/git/mongoose/E. coli/cache/$holdout" \
        --output "phase6_eval_${holdout}.json"
done
```

Also run viz on one randomly-chosen holdout molecule per run:

```bash
for holdout in STB03-063B-02L58270w05-433B23b STB03-065H-02L58270w05-433H09j STB03-064D-02L58270w05-433H09d; do
    PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
    C:/git/mongoose/.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
        --checkpoint phase6_smoke_checkpoints/best_model.pt \
        --cache-dir "C:/git/mongoose/E. coli/cache/$holdout" \
        --output "phase6_viz_${holdout}.png" --n-molecules 5 --seed 42
done
```

- [ ] **Step 5: Report results and STOP**

Report:
- Per-epoch loss trajectory (probe, bp, vel, count) from `phase6_smoke_train.log`
- Per-holdout F1 summary: `F1_sum`, `F1_mol_mean`, `precision`, `recall`, `n_molecules`
- Overall averaged F1 across all 3 holdouts
- Paths to the 3 viz PNGs
- Any anomalies (NaN during training, missing checkpoints, etc.)

**Do NOT proceed to Phase 4 or a 35-epoch run.** Wait for user approval.

- [ ] **Step 6: Commit the artifacts**

```bash
git add phase6_smoke_train.log phase6_eval_*.json phase6_viz_*.png
git commit -m "chore(training): Phase 6 smoke (5 epochs, 27/30 runs) artifacts"
```

---

## Rollback Plan

If any task fails in a way that makes the codebase worse than the pre-plan state:

```bash
cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
git log --oneline -5  # find the last known-good commit (probably 26f84ec: the mapping fix)
git revert <hash>..HEAD  # revert only plan-specific commits
```

The label-mapping fix at `26f84ec` is the desired baseline — do NOT revert past it.
