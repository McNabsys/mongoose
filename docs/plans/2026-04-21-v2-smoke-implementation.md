# V2 Strong-Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal.** Implement enough of the V2 design (`docs/plans/2026-04-20-v2-design.md`) to pass a "Strong" smoke test on the local A4500 desktop: (a) gradient-sanity test A2a passing at every training phase, (b) overfit-one-batch gate converging probe < 0.1 in ≤ 500 steps, (c) a Micro V2 Phase 6 — 1 epoch on 3 caches with all V2 losses live, showing clean train/val descent.

**What this is NOT.** Not a full V2 implementation. No ablations (A1/A3/A4/etc.), no production training, no 30-cache regen. Just "does V2's architecture + losses work end-to-end on the known Micro-Phase-6 data distribution."

**Architecture base:** `docs/plans/2026-04-20-v2-design.md` is authoritative. This plan implements a Minimum Viable subset of it.

**Branch:** `v2-smoke-test` off `training/v1-recipe-design` (after V1 sprint merges). New worktree at `.claude/worktrees/v2-smoke/`.

**Python env:** `C:/git/mongoose/.venv/Scripts/python.exe` with two new deps:
- `einops>=0.7` (for PixelShuffle1d substitute per V2 spec §5.5)
- Hand-rolled Sinkhorn (NOT geomloss — KeOps dependency is painful on Windows; fall back to `geomloss` only if hand-rolled has a numerical issue we can't fix).

---

## Scope trimming: what Strong Smoke SKIPS vs full V2

| V2 full design | Strong Smoke |
|---|---|
| New preprocessing (relaxed filter + seg_labels + accepted/rejected anchors) | ✅ full |
| 8x-Res Transformer U-Net backbone | ✅ full |
| 6-class segmentation head | ✅ full |
| Batched bidirectional Sinkhorn L_bp | ✅ forward only (drop reverse soft-min for now — the Micro Phase 6 dataset is all direction=1 or =-1 per molecule; direction-robust tuning is an ablation for later) |
| Dynamic-target L_count | ✅ full |
| Wide-boxcar warmstart L_probe_anchor | ✅ full |
| L_heatmap_sharpness (negative L2) | ✅ full |
| L_velocity REMOVED | ✅ enforced |
| 12-value FiLM/AdaLN conditioning | ✅ full |
| RoPE + AdaLN + FlashAttention | ✅ full (verify SDPA flash-dispatch works on A4500 Ampere) |
| Class-weight calibration from cache stats | ⚠️ smoke-time hack: use uniform weights [1, 10, 10, 5, 15, 30] as rough estimate; recompute properly in follow-up |
| Quality floor (match_rate ≥ 0.5, min_ref ≥ 10) | ✅ full |
| Ablations A1-A12 | ❌ deferred (Strong is not an ablation pass — just "does baseline V2 converge?") |
| Full 30-cache regeneration | ❌ skip — reuse 3 caches already cached for Micro Phase 6 (black/blue/red from V1 cache) **BUT with updated V2 preprocess so they include seg_labels and anchor positions** |
| Full 35-epoch training | ❌ skip — Micro V2 Phase 6 = 1 epoch × 3 training caches only |

---

## File Structure

**Created:**
- `src/mongoose/data/seg_labels.py` — densify seg_labels from probes.bin structure fields
- `src/mongoose/data/anchors.py` — extract accepted/rejected peak positions from probes
- `src/mongoose/model/rope.py` — 1D Rotary Position Embeddings
- `src/mongoose/model/adaln.py` — DiT-style AdaLN block
- `src/mongoose/model/pixelshuffle1d.py` — einops-based 1D PixelShuffle substitute
- `src/mongoose/model/transformer_unet.py` — the V2 8x-Res Transformer U-Net
- `src/mongoose/losses/sinkhorn_bp.py` — batched forward-direction Sinkhorn-OT L_bp (reverse/soft-min deferred)
- `src/mongoose/losses/probe_anchor.py` — wide-boxcar + narrow-negative anchor loss
- `src/mongoose/losses/heatmap_sharpness.py` — negative-L2 concentration regularizer
- `src/mongoose/losses/combined_v2.py` — CombinedLossV2 composing L_bp + L_count + L_seg + L_probe_anchor + L_sharpness
- `tests/test_model/test_rope.py`, `test_adaln.py`, `test_transformer_unet.py`, `test_pixelshuffle1d.py`
- `tests/test_losses/test_sinkhorn_bp.py`, `test_probe_anchor.py`, `test_heatmap_sharpness.py`, `test_combined_v2.py`
- `tests/test_data/test_seg_labels.py`, `test_anchors.py`
- `scripts/train_v2.py` — V2 training entrypoint (copy + adapt from V1's train.py)
- `scripts/overfit_one_batch_v2.py` — V2 overfit gate
- `scripts/v2_gradient_sanity.py` — Ablation A2a (gradient sanity test at warmstart, mid-fade, post-peel)

**Modified:**
- `src/mongoose/data/preprocess.py` — remove structured/folded filter; add seg_labels + anchor positions to cache; add match-rate quality floor
- `src/mongoose/data/cached_dataset.py` — emit seg_labels + anchor positions in `__getitem__`
- `src/mongoose/data/collate.py` — emit `ref_padded [B, max_N]` + `ref_valid [B, max_N]` tensors; emit anchor/seg batch tensors
- `src/mongoose/training/trainer.py` — support V2 model + loss (may need `--model-version v2` flag or separate V2 trainer)
- `pyproject.toml` — add `einops>=0.7` to deps

**Unchanged:**
- `src/mongoose/model/unet.py` (V1 model, kept for ablations)
- `src/mongoose/losses/combined.py` (V1 loss, kept for ablations)
- `src/mongoose/losses/centernet_focal.py` (V1 probe loss, kept for ablations)

---

## Implementation tasks (in dependency order)

### Tier 1: Data pipeline (foundational, everything depends on this)

- [ ] **Task 1.** Install einops: `pip install einops` + add to pyproject.toml deps. Commit.

- [ ] **Task 2.** `src/mongoose/data/seg_labels.py` with `build_seg_labels(mol, n_samples, sample_rate_hz)` returning `[n_samples]` int64 with 6 classes per V2 spec §4.1:
  - clean=0 default
  - folded_start=1: samples ≤ `folded_start_end_ms * sample_rate / 1000`
  - folded_end=2: samples ≥ `folded_end_start_ms * sample_rate / 1000`
  - struct_recoverable=3, struct_amp_high=4, excluded_other=5: from `mol.structures[i].attribute` bits per probes.bin V5 Table 5
  - TDD: synthetic molecule with known structure events, verify labeling

- [ ] **Task 3.** `src/mongoose/data/anchors.py` with `extract_anchors(mol, sample_rate_hz, start_within_tdb_ms)` returning `(accepted_pos [M], rejected_pos [R])`:
  - accepted: `probe.attribute & 0x80` set (bit 7 = accepted) AND `probe.in_clean_region` AND duration > 0
  - rejected: attribute bits 4/5/6/8 set (rejection flags from probes.bin V5 Table 3 — verify exact bit assignments against the spec)
  - Sample-idx conversion uses the same formula as V1 post-mapping-fix: `(start_within_tdb_ms + probe.center_ms) * sample_rate / 1000`
  - TDD: synthetic probes with known attribute bits, verify split

- [ ] **Task 4.** Modify `src/mongoose/data/preprocess.py` to:
  - Remove structured/folded/do_not_use exclusion (line 142)
  - Add quality floor: compute `match_rate = n_ref_probes / num_probes` and skip molecules with `match_rate < 0.5` OR `n_ref_probes < 10`
  - For each kept molecule: call `build_seg_labels` and `extract_anchors`, store in gt_dict:
    - `seg_labels`: np.int64 array [n_samples]
    - `accepted_peak_positions`: np.int64 array
    - `rejected_peak_positions`: np.int64 array
  - Update manifest.json molecule entries with `num_accepted`, `num_rejected`, `seg_class_counts [K]` (for later class-weight computation)
  - Extend `tests/test_data/test_preprocess.py` with a test verifying a known structured molecule now appears in the cache with correct seg_labels

- [ ] **Task 5.** Modify `src/mongoose/data/cached_dataset.py:__getitem__` to emit the new fields:
  - `seg_labels`: torch.LongTensor [T]
  - `accepted_peak_positions`: torch.LongTensor [M] (or None if empty)
  - `rejected_peak_positions`: torch.LongTensor [R] (or None)
  - Backward compat: existing fields stay unchanged so V1 still works

- [ ] **Task 6.** Modify `src/mongoose/data/collate.py` to:
  - Pad and stack `seg_labels` → `[B, padded_T]` with fill value = 5 (excluded_other, safely gated out everywhere)
  - Pad reference_bp_positions to `ref_padded [B, max_N]` + `ref_valid [B, max_N] bool`. Keep the V1 list-of-tensors output as `reference_bp_positions_list` for V1 model backward compat.
  - Collect accepted/rejected positions as lists (variable length per molecule; CombinedLossV2 will vectorize inside)
  - Tests: verify padding correctness, dtype, mask validity

- [ ] **Task 7.** Regenerate 3 smoke caches with the new V2 preprocess:
  - Black/060B + Blue/065C + Red/062B (same as V1 Micro Phase 6, for direct comparison)
  - Run `scripts/preprocess.py` for each → writes to `E. coli/cache_v2/<run_id>/`
  - Verify new fields present via manual inspection: load molecule 0, print seg_labels distribution, print accepted/rejected counts

### Tier 2: Model architecture

- [ ] **Task 8.** `src/mongoose/model/pixelshuffle1d.py`: einops-based PixelShuffle1d layer per V2 spec §5.5 option 1. Unit test: input shape `[B, C*r, T]` → output `[B, C, T*r]`.

- [ ] **Task 9.** `src/mongoose/model/rope.py`: 1D Rotary Position Embeddings, θ_base=50000, apply_rope(q, k, positions) per V2 spec §5.4. Unit test: rotation preserves norm, different positions produce different outputs.

- [ ] **Task 10.** `src/mongoose/model/adaln.py`: AdaLN block with zero-init gates per V2 spec §5.4. Unit test: block is identity at initialization (gate=0 → x == block(x)).

- [ ] **Task 11.** `src/mongoose/model/transformer_unet.py`: the full V2 architecture assembled from sections 5.3–5.7:
  - 12-value metadata embed MLP → cond_embed[B, 128]
  - Encoder 3-stage overlapping-conv CNN (8x downsample) with FiLM at Stage 0
  - Transformer stack (8 blocks at 8x, RoPE + AdaLN + SDPA-flash attention, key-padding-mask aware)
  - Decoder 2-stage (U3 + U2) with einops PixelShuffle1d and channel-reduction convs after skip concats
  - Bifurcation at 2x into heatmap+seg branch (narrow kernel) and velocity branch (wide kernel)
  - 4 output heads: heatmap [B,T] logits, velocity [B,T] softplus, cumulative_bp [B,T] cumsum(velocity*mask), seg_logits [B,6,T]
  - Tests: forward pass on synthetic input produces all 4 outputs with correct shapes; seg+heatmap come from the same branch; velocity cumsum respects mask

### Tier 3: Losses

- [ ] **Task 12.** `src/mongoose/losses/sinkhorn_bp.py`: forward-direction batched Sinkhorn-OT per V2 spec §6.2 (hand-rolled, not geomloss). Key correctness items:
  - Translation-invariant mean-centering on BOTH source (pred_cum_bp weighted by heatmap mass) and target (mean of valid ref_bp entries)
  - NaN-safe `torch.where` pattern on fully-padded rows (V2 spec §6.10 #3)
  - Return per-molecule `[B]` tensor; caller averages
  - Unit test A: point-cloud matching with known permutation → loss ≈ 0
  - Unit test B: translation invariance — add constant to all ref_bp, loss unchanged
  - Unit test C: NaN safety — batch with 1 fully-padded element, verify other elements' gradients are finite

- [ ] **Task 13.** `src/mongoose/losses/probe_anchor.py`: wide-boxcar + narrow-negative anchor loss per V2 spec §6.6. Returns `(L, anchor_mass)` tuple for passing into L_count.

- [ ] **Task 14.** `src/mongoose/losses/heatmap_sharpness.py`: negative L2 norm regularizer per V2 spec §6.7. Sign convention: returns a value we MINIMIZE (i.e., `-l2_norm`). Unit test: flat heatmap → high loss; peaked heatmap → low loss.

- [ ] **Task 15.** `src/mongoose/losses/combined_v2.py` = `CombinedLossV2` composing:
  - L_bp (from Task 12, weighted by lambda_bp)
  - L_count (new dynamic-target version per V2 spec §6.3)
  - L_segmentation (cross-entropy with class weights, per V2 spec §6.5)
  - L_probe_anchor (weighted by `_blend`, faded per V2 spec §6.8)
  - L_heatmap_sharpness (weighted by lambda_sharp, always on post-warmstart)
  - `set_epoch(epoch)` drives the blend fade
  - Full batching discipline: NO `for b in range(B)` in the loss path (per V2 spec §6.10 #8)
  - Returns `(total, details)` dict compatible with trainer's expectations

- [ ] **Task 16.** Gradient sanity test `scripts/v2_gradient_sanity.py` (Ablation A2a, free). Run at each of 3 phases:
  - `set_epoch(0)` — blend=1.0 (full warmstart)
  - `set_epoch(3)` — blend≈0.33 (mid-fade)
  - `set_epoch(10)` — blend=0.0 (post-peel)
  Assert at each phase: `pred_heatmap_logits.grad`, `raw_velocity.grad`, `pred_seg_logits.grad` all non-zero AND finite. This is Strong-Smoke's first pass criterion.

### Tier 4: Training + overfit gate

- [ ] **Task 17.** `scripts/overfit_one_batch_v2.py`: copy of V1 overfit script adapted for V2 model + loss. Accepts `--model v2 --cache-dir <v2_cache>`. Success: probe loss < 0.1 in ≤500 steps on a single 32-molecule batch with all losses live (including L_seg).

- [ ] **Task 18.** `scripts/train_v2.py`: adapted from V1 `train.py`. Accepts V2 loss components' lambdas, V2-specific flags (`--sinkhorn-eps`, `--sinkhorn-iters`, `--warmstart-epochs`, etc.). Reuses `Trainer` class as much as possible.

- [ ] **Task 19.** Launch "Micro V2 Phase 6": 1 epoch × 3 V2 caches, batch 32, LR 3e-4 cosine → 1.5e-5, all V2 losses ON. ETA ~70 min. Success: no NaN, clean monotone train_loss descent over the epoch.

### Tier 5: Evaluation

- [ ] **Task 20.** Run `scripts/evaluate_peak_match.py` (V1's evaluator) against the V2 checkpoint on the held-out Red cache (`STB03-064D-02L58270w05-433H09d`). Report F1. Success criterion: F1 > 0.5 (loose — V1 hit 0.917 after a proper Micro Phase 6 with physics ON; V2 with a totally new backbone + loss composition after 1 epoch on 3 caches should produce SOMETHING coherent).

- [ ] **Task 21.** Run `scripts/visualize_predictions.py` with the V2 checkpoint. Visually inspect: peaks should land near references (like V1 did), plus the segmentation head should correctly identify fold regions if any holdout molecules are structured.

### Final

- [ ] **Task 22.** Write `docs/peer_reviews/2026-04-21-round5-v2-smoke.md` summarizing:
  - A2a gradient sanity test results (pass/fail per phase)
  - Overfit-gate convergence curve (probe / bp / vel / seg per 50 steps)
  - Micro V2 Phase 6 training curve
  - Holdout F1 vs V1 Micro F1
  - Any anomalies (Sinkhorn NaN events, FlashAttention fallback, class-imbalance issues)

- [ ] **Task 23.** Commit everything, push branch, open Deep Think round-6 review request.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hand-rolled Sinkhorn has a subtle NaN bug under mask conditions we didn't anticipate | Medium | Task 12 has explicit NaN-safety unit test; A2a gradient sanity catches it before we waste training time |
| FlashAttention-2 fallback to math backend on A4500 due to some dtype issue | Low | Task 11 explicitly wraps in `sdpa_kernel([FLASH_ATTENTION])` with `raise_on_failure`; we'll know immediately |
| PixelShuffle1d einops implementation has an off-by-one on odd-length sequences | Low | Task 8 unit test covers this |
| V2 overfit gate converges slower than V1 (new losses are less well-tuned) | High | Strong smoke criterion is "converges AT ALL in 500 steps," not "converges as fast as V1." If we hit 500 steps without probe < 0.5, bump to 1000 steps; log the decision |
| Class-weight hack (uniform [1, 10, 10, 5, 15, 30]) is too far off, L_segmentation dominates or underfits | Medium | If L_seg is off by >10× compared to other losses, immediately recompute from cache stats and rerun |
| Micro V2 Phase 6 val_loss climbs (overfitting to 3 caches of augmented-schema data) | Low | Compare to V1 Micro's val=0.26 baseline; if V2 val > 0.50 that's a red flag to investigate before scaling |
| Training is slow because batched Sinkhorn is 100MB per step cost-matrix materialization on A4500 | Medium | V2 spec §6.10 #5 notes this; if tight, stratified-subsample source distribution to T_sub ≤ 20k per molecule |

---

## Success criteria — Strong Smoke passed if all hold

1. All tests in `tests/test_data/`, `tests/test_model/`, `tests/test_losses/` green (incl. new V2 test files)
2. Gradient sanity test (Task 16) produces finite gradients for heatmap+velocity+seg at all 3 phases (warmstart, mid-fade, post-peel)
3. Overfit-one-batch gate (Task 17) drives probe loss < 0.1 within 500 steps on a single batch
4. Micro V2 Phase 6 (Task 19) trains 1 epoch without NaN, clean monotone train_loss descent, val_loss NOT > 2× V1 Micro baseline (0.26)
5. Holdout F1 (Task 20) > 0.5 on Red holdout — orders of magnitude higher than V1 v1 pre-sprint (0.033) and meaningfully above "model produces outputs but hasn't learned anything"

If all 5 hold: V2 smoke is passed, ready to proceed to full implementation (ablations, 30-cache regen, cloud production run).

If any fail: diagnose that specific failure, document, decide whether to continue or loop Deep Think round-6.
