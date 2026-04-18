# V1 Training Plan — 48-Hour Design

**Date:** 2026-04-18
**Branch of origin:** `perf/criterion-amp-fixes` (GPU perf + AMP stability fixes)
**Budget:** 48 hours wall-clock on an NVIDIA RTX A4500 (20 GB VRAM) workstation
**Goal priority (user-ranked):** A (trained model) > C (scale to 30 runs) > B (recipe) > D (diagnostics)

---

## 1. Goal and Success Criteria

Produce a trained V1 T2D U-Net model that detects probe peaks on E. coli
translocation waveforms and evaluate it both quantitatively and visually.
The target artifacts at the 48-h mark:

- A **multi-run model** trained on 27 of the 30 E. coli runs, with 3 runs
  (one per color) held out for final evaluation.
- A **single-run model** trained on the first preprocessed run, evaluated
  on its own held-out val split, produced as a stepping stone and
  fallback artifact.
- **Quantitative evaluation:** peak-match F1 / precision / recall at
  ±50-sample tolerance against wfmproc's `warmstart_probe_centers_samples`,
  reported per-molecule and aggregated.
- **Visual evaluation:** 20-molecule grid PNG showing waveform, predicted
  heatmap, predicted peak positions, and reference peak positions, for a
  30-second gut check.

**Working definition of "it works":** F1 meaningfully above random
baseline (> 0.3 at minimum; > 0.5 is success, > 0.7 is strong) AND
visible peaky structure in the heatmap grid — **both** required, not
either-or. A model that scores well on F1 but has a flat heatmap is
failing in a way the scalar metric can hide, which is exactly what
tonight's attempt revealed.

---

## 2. Why the Previous Attempt Failed

Tonight's run (`--lr 3e-4`, 10 epochs, single-run cache) produced these
pathological losses:

| Epoch | blend | probe | bp | vel | count |
|-------|-------|-------|------|------|-------|
| 3 | 1.0 | 0.050 | 517,575 | 9,840 | 0.99 |
| 5 | 0.0 | 0.994 | 178,985 | 10,320 | 4.82 |
| 10 | 0.0 | 0.998 | 33,481 | 3,820 | 0.54 |

**Structural problem (1):** Once warmstart fade completes at epoch 5, the
probe loss becomes pure `peakiness_regularizer` in
[combined.py:173-176](../../src/mongoose/losses/combined.py). Peakiness is
`(1 - max_pool(heatmap, W)).mean()`, bounded in [0, 1]. When the heatmap
is near zero everywhere, `max_pool` is tiny, `1 - max_pool ≈ 1`, and the
gradient w.r.t. heatmap values is `-1/W` — a tiny pressure to increase
each value. Meanwhile `bp_loss` (tens of thousands) has a huge gradient
directly wired to `cumulative_bp`, and the Adam optimizer routes most
updates there. Net: the heatmap collapses to flat because no other
gradient is pushing it toward peaky.

**Structural problem (2):** Loss components differ by 4-5 orders of
magnitude at the default `lambda_* = 1.0`. `bp = 33,000` dominates
`probe = 1.0`, `count = 0.5`, and `vel = 4,000` at the gradient level.
Effectively the model trains only on `bp`; everything else is noise.

---

## 3. Loss Recipe Changes (the aggressive code changes)

Two targeted changes to [combined.py](../../src/mongoose/losses/combined.py),
both with new unit tests.

### 3.1 Blend floor

Add `min_blend: float = 0.1` to `CombinedLoss.__init__`. Modify
`set_epoch` so `self._warmstart_blend = max(min_blend, computed_blend)`.

Effect: focal supervision against wfmproc Gaussian targets never drops
below 10% weight. Keeps the heatmap pulled toward peaky locations for
the entire run, preventing the tonight-observed collapse.

Cost: a small permanent bias toward wfmproc's labeling. Acceptable
because wfmproc is currently our only peak-position supervision signal;
the bias is the floor of what the model could learn from data it has
access to.

### 3.2 Running-mean loss-scale normalization

Add `_scale_ema` state to `CombinedLoss` tracking a per-component
exponential moving average of unweighted loss magnitude (probe, bp,
vel, count). On forward, divide each raw loss by its scale (detached,
gradient does not flow through the divisor) before applying `λ`.

EMA decay: 0.99. Initialization: first batch sets the scale. Per-batch
order of operations: (1) compute raw losses, (2) divide each by the
**previous-batch** EMA (so the current gradient uses a stable denominator),
(3) update the EMA with the newly-observed raw loss for use by the next
batch. Floor scale at `1e-6` to avoid division-by-zero if a loss term is
legitimately zero early.

Effect: `λ_* = 1.0` now means "this component contributes equal gradient
magnitude." Retuning by loss term reduces to intuitive scalar choice.

Cost: Small amount of extra state to serialize in the checkpoint
(scheduler state already opts in to pickling, so EMA floats go in
cleanly). Slight warmup oddness on the very first batch since the EMA
hasn't stabilized, but the first batch won't dominate training anyway.

### 3.3 Tests

- `test_combined_blend_floor`: set `min_blend=0.2`, step to epoch 100,
  assert `_warmstart_blend == 0.2`.
- `test_combined_scale_normalization_stabilizes`: feed a batch with
  known-imbalanced raw losses, step 50 batches, assert each scaled
  component in `details` is ≈ 1.0 within tolerance.
- Regression test: existing `CombinedLoss` behavior with
  `min_blend=0.0, scale_normalize=False` stays numerically identical to
  what tonight's code produced.

---

## 4. Evaluation Infrastructure

Two new scripts, each small and independently testable.

### 4.1 `scripts/evaluate_peak_match.py`

**Inputs:** model checkpoint, list of cache dirs, output JSON path,
optional `--run-ids` filter, optional `--max-molecules`.

**Logic:**
1. Load checkpoint, build model on GPU, set eval mode.
2. For each molecule in the given cache(s):
   a. Forward pass to get `pred_heatmap`, `raw_velocity`.
   b. Extract peaks via existing `extract_peak_indices`.
   c. Load reference peaks from `warmstart_probe_centers_samples` in the
      manifest.
   d. Match predicted peaks to references with greedy bipartite matching
      at ±50-sample tolerance. Iterate predicted peaks in descending
      order of heatmap confidence; each predicted peak claims its
      closest **unmatched** reference within tolerance, or counts as a
      false positive if none exists. Remaining unclaimed references
      are false negatives.
3. Aggregate: per-molecule TP / FP / FN, compute precision / recall /
   F1 per molecule, then mean + median across molecules. Per-run
   breakdown for multi-run evaluation.

**Output:** JSON with `{ "overall": {...}, "per_run": {...}, "per_molecule": [...] }`
plus a stdout summary table.

**Unit tests:** synthetic waveform with known peaks, assert matching
logic handles: perfect match, off-by-25-samples match, off-by-100-sample
miss, extra predicted peak (FP), missing predicted peak (FN), two
predicted peaks near one reference (one TP + one FP).

### 4.2 `scripts/visualize_predictions.py`

**Inputs:** model checkpoint, cache dir, `--n-molecules` (default 20),
`--seed` (default 42), output PNG path.

**Logic:**
1. Load checkpoint, pick N molecules at random from the cache (fixed
   seed for reproducibility).
2. For each molecule: forward pass, extract peaks.
3. Render a `(n_rows, 2)` matplotlib figure: left column = waveform
   overlaid with heatmap (twin axis), right column = predicted peak
   positions (red `|` markers) + reference peak positions (blue `|`).
4. Save as single PNG.

**Unit tests:** matplotlib doesn't need heavy testing; one smoke test
that the script runs end-to-end on a tiny cache fixture and produces a
non-empty PNG file.

---

## 5. Execution Plan

### Timeline

| Phase | Work | Est. wall | GPU |
|-------|------|-----------|-----|
| 1 | Code: loss changes + evaluate_peak_match + visualize_predictions + tests | 3 h | no |
| 2a | Smoke: new recipe, 10 epochs, single-run cache | 4 h | yes |
| 2b | Evaluate + visualize run 2a; decide go/tune | 0.5 h | brief |
| 2c | (Conditional) smoke run 2 with one adjustment | 4 h | yes |
| 3 | Preprocess remaining 29 runs (parallel with other GPU work) | 20 min | no |
| 4 | Long single-run train: 35 epochs, winning recipe | 15 h | yes |
| 5 | Evaluate single-run model (F1 + viz) | 0.5 h | brief |
| 6 | Multi-run train: 2 epochs on 27 runs | 20-22 h | yes |
| 7 | Evaluate multi-run model on 3 held-out test runs + viz | 1 h | brief |
| 8 | Write summary (what worked / what didn't, F1 numbers, viz links) | 0.5 h | no |

**Cumulative:** ~44-49 h, with 2-4 h of explicit slack.

### Test Holdout Policy (Multi-Run)

Deterministic by filename sort. Hold out the **last** run in alphabetical
order from each color (Black / Blue / Red). Using last-in-sort instead of
first-in-sort guarantees the held-out runs are stable regardless of which
runs the single-run training used — neither of the two preprocessed runs
from the session so far (`STB03-060A-02L58270w05-202G16j`,
`STB03-060A-02L58270w05-433B23e`) is the alphabetical last, so they cannot
land in the test set by accident.

The 27 remaining runs form the training set. The 3 held-out test runs
are only ever touched by `evaluate_peak_match.py` and
`visualize_predictions.py`. The exact test run IDs are resolved at the
start of Phase 3 by listing each color directory and picking the final
entry, and they are pinned in the final summary report.

### Go/No-Go Gates

- **After Phase 2a:** F1 on the single-run val split must be ≥ 0.3. If
  not, **stop**, do one adjustment (warmstart length, loss-scale cap,
  LR), do Phase 2c, re-check. If still < 0.3 after Phase 2c, the design
  assumptions are wrong and we escalate rather than continue burning GPU.
- **After Phase 5:** F1 on single-run should ideally be ≥ 0.5 for Phase
  6 to be worth launching. If 0.3-0.5, launch Phase 6 but with reduced
  expectations. If < 0.3, use Phase 6's 20 h budget to iterate further
  on single-run recipe instead.
- **During Phase 4 / 6 long runs:** per-epoch checkpoints enable aborting
  if any of: val_loss exploding, F1 trending down after an early peak,
  GPU OOM, NaN observed. Stop, triage, resume from last good checkpoint
  or restart.

### Config Choices

Inheriting tonight's working config and adjusting only where evidence
points:

- `batch_size`: 32 (measured stable at 99% VRAM; safe margin by dropping
  to 24 if a run OOMs).
- `lr`: 3e-4 (worked tonight; may tune to 1e-4 if overtraining signs show).
- `min_lr`: `lr / 20` = 1.5e-5 (not 1e-6; tonight's cosine-to-1e-6 froze
  last epochs).
- `warmstart_epochs`: 8 (up from 5) with `warmstart_fade_epochs`: 4 and
  `min_blend`: 0.1. Gentler ramp keeps focal supervision engaged longer.
- `save_every`: 5 for long runs (Phase 4, 6) to avoid 15+ GB of
  checkpoints. Every epoch for short smokes (Phase 2).
- `use_amp`: True (bf16 + fp32 criterion from the perf work).

---

## 6. Risks and Mitigations

**Risk 1: The loss-scale normalization EMA oscillates and never stabilizes.**
Mitigation: log per-epoch EMA values in the training summary. If we see
wild swings, fall back to static `lambda` overrides (Approach 2-moderate
from the brainstorm) as a Phase 2c adjustment.

**Risk 2: Phase 4 recipe works for 55k single-run but doesn't scale to
1.5M multi-run.** Could happen if the loss scales learned on single-run
don't match multi-run statistics (different runs have different peak
densities, velocities, etc.). Mitigation: the running-mean
normalization re-stabilizes on new data; the first batch of Phase 6
will be used to reset the EMA. Also, Phase 2c or Phase 4 can
include a quick sanity check where we run forward on molecules from a
different color run and confirm the loss magnitudes are similar.

**Risk 3: Multi-run preprocessing hits a snag on one of the other 29
runs (missing file, TDB parse failure).** Tonight's preprocess worked
on two runs clean; all 30 use the same pipeline. Mitigation: run the
preprocessing early (Phase 3) so failures surface while there's still
GPU time to work around them. Skip bad runs rather than halt; 25 good
runs is still a multi-run demonstration.

**Risk 4: Phase 6 VRAM spikes because of larger per-batch molecule
variety / longer sequences in some runs.** Mitigation: if OOM, drop
batch size to 24 and resume from last checkpoint. If OOM persists,
drop to 16 — throughput takes a ~30% hit, still finishes within budget.

**Risk 5: Evaluation logic has a bug and F1 reports wrong numbers.**
Mitigation: unit tests on known-good synthetic cases before running on
real data. Additionally, the visual grid in every evaluation gives a
human-checkable sanity read — if F1 reports 0.8 but the visualizations
show flat heatmaps, we know the metric is lying.

---

## 7. Deliverables at 48 h

1. Single-run model checkpoint + F1 report + 20-molecule visual grid.
2. Multi-run model checkpoint + per-held-out-run F1 breakdown + 20-molecule
   visual grids on each of the 3 held-out runs.
3. The 3 test run IDs, pinned in the final summary so they can be
   excluded from any retraining.
4. Training logs (per-epoch losses + EMA traces) for both runs.
5. Short written summary: what worked, what didn't, recommended next steps.
6. Code changes committed on a clean branch with passing tests (199
   existing + new ones for the code added in Section 3-4).

---

## 8. Out of Scope

- **`docs/ablation_protocol.md` diagnostics (Priority D).** Not running
  the `lambda_vel=0` ablation or peak-count discrepancy study in this
  48 h.
- **Training-recipe robustness study (Priority B).** Not doing multiple
  seeds, not benchmarking the recipe's variance. The recipe either
  works well enough on this run to produce a usable model or it
  doesn't; that's the only question being asked.
- **`scripts/batch_preprocess.py` / `scripts/build_manifest.py`
  infrastructure.** We will invoke `scripts/preprocess.py` 29 more times
  by shell scripting in Phase 3. Writing proper batch-preprocessing
  tooling is deferred to a follow-up.
- **Cloud / multi-GPU training.** Single A4500, single workstation.
- **Any UI / visualization beyond a static PNG grid.**
- **Peak localization in bp-space.** Evaluation operates in sample
  space against wfmproc's `warmstart_probe_centers_samples`. Full
  bp-space evaluation against the reference genome would require the
  velocity head to be good, which we have no evidence of yet.
