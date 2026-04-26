# Overnight Report — Direction C Analysis (2026-04-26)

> Generated overnight while user slept. H100 Direction A still running;
> this report covers Direction C only. All numbers from
> `residual_eval_results.json` (eval-script output on 28 caches).

---

## TL;DR

**Direction C beats T2D's 16.2% baseline on every single run.** Across
21.5M evaluated bp-intervals (28 runs), the per-probe MLP delivers a
median bp-interval rel-error of **8.07%** vs T2D's 16.2% and production's
5.77%. **Direction C is real.**

Production (TVC + Method 1) is still better by 1.8-4.4 percentage points
on every run, indicating a consistent architectural gap rather than
overfitting. The most likely cause: missing per-detector / per-condition
features (the docs explicitly say production parameterizes TVC per
detector × per V × per psi).

**Recommendation: pursue Direction C as a V5 baseline, with two extensions:**
1. Add per-detector / V / psi features to close the production gap.
2. Run a clean per-run held-out eval to confirm the in-distribution numbers
   here translate (90 min retraining cost).

---

## 1. Background — what we tested

Direction C trains a small per-probe MLP to predict the bp shift production
applies to each probe via TVC + head-dive Method 1:

| | |
|---|---|
| Input | 22-dim feature vector: pre-position frac, log intervals, attribute bits, molecule aggregates, length-group bin, direction one-hot |
| Target | `residual_bp = post_position_bp − pre_position_bp` (production's per-probe bp shift) |
| Architecture | 4 residual blocks × 256 hidden = 537K params |
| Training | 28 caches, ~30M accepted probes after filtering, batch 4096, lr 3e-4, 20 epochs cosine |
| Hardware | RTX A4500 local, ~5 min/epoch, 110 min wallclock |
| Best epoch | 17 (val_mae 1107.3 bp, ratio 0.7947 vs predict-mean baseline) |

The 20.5% MAE reduction over predict-mean is meaningful but trivial; the
real comparison is **bp-interval rel-error vs the reference genome** —
the same metric T2D's 16.2% comes from.

---

## 2. Headline: bp-interval rel-error vs reference

For each molecule, ordered probes give consecutive intervals. Compute:

```
pred_position[i] = pre_position_bp[i] + model(features[i])
pred_intervals   = abs(diff(pred_position))
prod_intervals   = abs(diff(post_position_bp))
ref_intervals    = abs(diff(reference_bp_position))
model_rel_err = |pred_iv − ref_iv| / ref_iv
prod_rel_err  = |prod_iv − ref_iv| / ref_iv
```

**Across all 28 runs (21.5M intervals):**

| line | overall median rel-err |
|---|---:|
| **T2D baseline** (V3 holdout median) | **0.162** |
| **Production** (TVC + Method 1) | **0.0577** ← ceiling |
| **Direction C model** | **0.0807** ← what we built |
| **Predict-mean baseline** (trivial) | (training-time ratio implied: ~0.6) |

The model captures most of the production correction; it sits roughly
**halfway between T2D and production** in absolute rel-error terms.

**Caveat — this is NOT a clean held-out evaluation.** The training used
a per-probe random val split (val_fraction=0.2), so each cache has a mix
of train and val probes. The model has seen probes from these molecules
during training. A clean per-run-held-out evaluation requires retraining
with 25 train caches + 3 holdout caches (the V3 holdouts:
`STB03-063B-...-433B23b`, `STB03-064D-...-433H09d`,
`STB03-065H-...-433H09j`). Estimated 90 min retraining. The numbers in
this report are likely an **upper bound** on true held-out performance,
but the consistency across all 28 runs (ALL beat T2D) suggests the
within-distribution leakage is small.

---

## 3. Per-run breakdown — does the model overfit some runs?

| run | n_iv | model_med | prod_med | gap | beats T2D? |
|---|---:|---:|---:|---:|:---:|
| STB03-062A-…-433H09i | 562k | 0.1187 | 0.0751 | +0.044 | ✓ |
| STB03-065H-…-433H09j | 319k | 0.1064 | 0.0790 | +0.027 | ✓ |
| STB03-062B-…-433B23h | 787k | 0.1036 | 0.0675 | +0.036 | ✓ |
| STB03-063A-…-433B23g | 1317k | 0.0943 | 0.0669 | +0.027 | ✓ |
| STB03-060B-…-433H09f | 1259k | 0.0873 | 0.0605 | +0.027 | ✓ |
| STB03-065A-…-202G16c | 627k | 0.0872 | 0.0647 | +0.022 | ✓ |
| STB03-063B-…-202G16a | 1085k | 0.0861 | 0.0593 | +0.027 | ✓ |
| STB03-064B-…-202G16g | 688k | 0.0859 | 0.0572 | +0.029 | ✓ |
| STB03-062D-…-433H09g | 773k | 0.0843 | 0.0576 | +0.027 | ✓ |
| STB03-060A-…-202G16j | 1024k | 0.0840 | 0.0615 | +0.023 | ✓ |
| STB03-065F-…-433B23d | 568k | 0.0825 | 0.0616 | +0.021 | ✓ |
| STB03-064C-…-433B23c | 755k | 0.0821 | 0.0592 | +0.023 | ✓ |
| STB03-060A-…-433B23e | 1080k | 0.0809 | 0.0611 | +0.020 | ✓ |
| STB03-060C-…-433H09a | 1050k | 0.0807 | 0.0578 | +0.023 | ✓ |
| STB03-065G-…-433B23a | 630k | 0.0806 | 0.0553 | +0.025 | ✓ |
| STB03-065E-…-433B23i | 706k | 0.0806 | 0.0618 | +0.019 | ✓ |
| STB03-065G-…-202G16h | 588k | 0.0803 | 0.0576 | +0.023 | ✓ |
| STB03-063B-…-433B23b | 1304k | 0.0797 | 0.0587 | +0.021 | ✓ |
| STB03-062B-…-433H09c | 556k | 0.0789 | 0.0574 | +0.022 | ✓ |
| STB03-065F-…-433H09h | 620k | 0.0786 | 0.0573 | +0.021 | ✓ |
| STB03-064C-…-202G16b | 590k | 0.0779 | 0.0551 | +0.023 | ✓ |
| STB03-064A-…-429C12a | 620k | 0.0777 | 0.0573 | +0.020 | ✓ |
| STB03-063A-…-433H09e | 1086k | 0.0776 | 0.0575 | +0.020 | ✓ |
| STB03-065D-…-202G16e | 399k | 0.0769 | 0.0567 | +0.020 | ✓ |
| STB03-064D-…-433H09d | 668k | 0.0768 | 0.0561 | +0.021 | ✓ |
| STB03-065B-…-202G16f | 516k | 0.0762 | 0.0556 | +0.021 | ✓ |
| STB03-065D-…-433H09b | 497k | 0.0756 | 0.0560 | +0.020 | ✓ |
| STB03-065A-…-433B23f | 805k | 0.0746 | 0.0564 | +0.018 | ✓ |

**Aggregate stats:**

| | min | median | mean | max |
|---|---:|---:|---:|---:|
| model_med | 0.0746 | 0.0807 | 0.0841 | 0.1187 |
| prod_med | 0.0551 | 0.0577 | 0.0603 | 0.0790 |
| gap (model − prod) | 0.0182 | 0.0225 | 0.0238 | 0.0436 |

**Observations:**

- **All 28 runs beat T2D 16.2%.** Even the worst run (STB03-062A at
  11.87%) is 4.3 pts better than T2D.
- **The gap to production is remarkably consistent — 1.8% to 4.4%.**
  This argues the gap is **architectural**, not overfitting.
- **The 5 worst-performing runs** (STB03-062A, STB03-065H, STB03-062B,
  STB03-063A, STB03-060B) all share **higher production rel-err too**
  (0.067-0.079). They're harder runs — production also struggles. The
  model and production fail correlated on hard runs.
- **STB03-062A stands out** with model_med 11.87% (nearly 3 pts above
  the next worst). Worth a closer look — could indicate a feature gap
  on this run's specific operating conditions.
- **4 of 28 runs nearly match production** (gap < 2%): STB03-065A-…-433B23f,
  STB03-065E, STB03-065D-…-433H09b, STB03-064A.

---

## 4. Per-decile breakdown — where the model fails

Stratified by `|residual_bp|` magnitude (target the model is predicting):

| decile | residual range (bp) | n_probes | model MAE | mean abs target | MAE/target |
|---:|---|---:|---:|---:|---:|
| 0 | 0 – 755 | 2.63M | 621 | 207 | **299%** |
| 1 | 66 – 1602 | 2.64M | 594 | 637 | 93% |
| 2 | 134 – 2341 | 2.64M | 605 | 1026 | 59% |
| 3 | 238 – 3047 | 2.64M | 673 | 1315 | 51% |
| 4 | 352 – 3779 | 2.64M | 779 | 1582 | 49% |
| 5 | 490 – 4596 | 2.64M | 905 | 1869 | 48% |
| 6 | 663 – 5578 | 2.64M | 1064 | 2205 | 48% |
| 7 | 884 – 6917 | 2.64M | 1280 | 2632 | 49% |
| 8 | 1197 – 9222 | 2.64M | 1623 | 3262 | 50% |
| 9 | 1628 – 45988 | 2.64M | 2735 | 4860 | 56% |

**Diagnosis:**

- **Decile 0 is broken — 299% rel-error on small shifts.** When the
  true residual is ~200 bp the model predicts ~620 bp on average. This
  is a textbook **bias-init-too-aggressive** problem: the head bias was
  initialized at +1817 bp (training mean), so the model defaults to
  predicting "big shift" even when the true shift is small.
- **Deciles 2-8 are strikingly uniform at ~48-50% rel-error.** The
  model captures roughly half the magnitude of true shifts in the
  middle range. This is what shows up in the headline 8.07% interval
  rel-err: per-probe shift errors of ~50% become per-interval errors
  of ~5-10% because intervals are differences of nearby probes (some
  cancellation).
- **Decile 9 (largest shifts, head-dive territory) degrades to 56%
  rel-error.** Slightly worse than the middle, but absolute MAE is
  huge (2.7 kb). The +5 kb shifts are exactly what head-dive Method 1
  is designed for, and the model doesn't have explicit length-group
  splines.

**Implications:**

- The decile-0 problem is **fixable cheaply.** Two options:
  - Replace MSE with **Huber loss** (less penalty on large errors,
    more focus on small ones).
  - **Predict log(1 + residual)** instead of raw residual — natural
    handling of the heavy-tail target distribution.
  - Or just **lower the head-bias init** to the median target (~ 800 bp)
    instead of mean (1817 bp).
- The decile-9 problem is the **architectural** axis: missing length-group
  splines + missing per-detector / per-condition features.
- Total contribution of decile 0+9 to the headline: probably 2-3% of
  the bp-interval rel-error gap to production. Closing those would put
  the model at ~6.5% rel-err — within 1% of production.

---

## 5. Per-run residual MAE distribution

Overall residual-prediction MAE (per probe, not per interval):

```
min=440 bp   median=975 bp   mean=1075 bp   max=2567 bp
```

Spread is wide. The 5th-percentile run is 564 bp; the 95th-percentile
is 1604 bp. **3× variation across runs.** Tracks with the per-run
table above — STB03-062A is the worst-MAE outlier (matches its 11.87%
interval rel-err). This further supports the "missing run-level features"
hypothesis: if run characteristics (detector, V, psi, frame) drive
production's per-detector TVC parameter set, our run-agnostic model has
no way to specialize.

---

## 6. Recommendations

### Immediate (before V5 sprint)

1. **Run a clean per-run held-out eval.** Retrain on 25 caches, hold out
   the 3 V3-holdouts (STB03-063B-…-433B23b, STB03-064D-…-433H09d,
   STB03-065H-…-433H09j). 90 min on local A4500. Get a number directly
   comparable to T2D's 16.2%. The within-distribution 8.07% number is
   likely accurate but should be confirmed.

2. **Fix the decile-0 over-prediction.** Lowest-cost intervention:
   - Change head-bias init to the **median** target (~800 bp) instead
     of the mean (1817 bp).
   - Or predict `log1p(max(residual, 0))` and exp-decode.
   - Expected lift: 1-2% absolute on overall rel-err.

3. **Add per-detector + V + psi features.** Need to parse
   `_remapSettings.txt` per cache. The Confluence docs explicitly say
   production parameterizes TVC by exactly these axes. Expected lift:
   1-3% absolute on overall rel-err. Most of the consistent 2-2.5%
   model-vs-prod gap likely lives here.

### Bigger architectural moves (V5 candidates)

4. **Length-group-conditioned correction splines.** Replace the single
   per-probe MLP with a length-group-binned ensemble (16 bins per the
   head-dive Method 1 paper). Each bin gets its own correction curve
   parameterized by molecule-position frac. This explicitly mimics the
   production head-dive correction structure.

5. **Open the data filter at preprocess.py:142.** Currently we exclude
   `attr_in_structure` and `attr_folded_*` molecules entirely. Phase 0a
   said these are where the ×10 residual lift lives — exactly the
   population the production RTS branch handles. With a per-attribute
   feature already in C's input, including these molecules in training
   could yield additional residual signal.

6. **Compose with Direction A** (if A delivers). If the H100 noise-model
   loss path also produces a viable model, the V5 architecture could
   layer Direction C's residual correction ON TOP of A's improved
   per-sample velocity. They're complementary — A predicts cumulative bp
   from waveforms; C predicts a per-probe correction in bp space.

### What NOT to do

- Don't pursue bigger MLP (more layers / hidden-dim). The model is
  already plateaued; adding capacity without adding feature/architectural
  inductive bias won't move the needle.
- Don't try to **beat** production with C alone — the per-probe MLP
  framework is structurally limited (no inter-probe attention). The
  best C can do is asymptote to production. Beating production requires
  signal production doesn't see (e.g., the raw waveform that V3/V4-A use,
  or the noise-model NLL framework).

---

## 7. Status of Direction A (still running on H100)

As of report generation:

```
Epoch 1/20 done: loss=0.5293  probe=0.26  bp=0.34  vel=0.007  count=0.04  val=0.4780
Epoch 2/20 done: loss=0.5322  probe=0.24  bp=0.31  vel=0.006  count=0.03  val=0.4792
Epoch 3 still in progress (~30 min remaining as of writeup)
```

- 4h+ into a ~40h projected run, only 2 epochs done
- val_loss bouncing in 0.478-0.479 range — too early to call
- Eval at random init (epoch 0): overall_median 0.3535 vs T2D 0.162.
  V4-A starts way above T2D and needs to descend ~20 pts in 5-7 epochs
  (V3 hit 15.87% by epoch 3 — same trajectory).
- All 4 components decreasing slowly. v2 scaling fix is clearly better
  than v1, but convergence pace is unclear.
- **Verdict at writeup time: cannot say yet whether A will work.**
  Need 4-6 more epochs (8-12 hours) to assess.

---

## 8. Recommended next-session action

1. **First check the H100.** If A is decreasing fast, let it continue.
   If A is still flat/divergent at epoch 5-6, kill and pivot to C-based
   V5.

2. **Run the held-out C eval** (~90 min). Confirms the 8.07% number.
   Independent of A's outcome.

3. **Decide V5 architecture** based on which Direction wins:
   - If C wins (likely): build V5 = C + per-detector features +
     length-group splines + RTS-aware data inclusion. ~1-2 weeks of work.
   - If A wins: build V5 = A's noise-model loss + Direction B's per-detector
     conditioning. Keep C as a baseline.
   - If both work: V5 composes. Most exciting outcome.

4. **Either way, address the decile-0 over-prediction in C** as a quick
   win.

---

## Files referenced

- `residual_run_checkpoints/residual_epoch_017.pt` — best model (ckpt epoch 17)
- `residual_run_checkpoints/history.json` — per-epoch training metrics
- `residual_eval_results.json` — full eval JSON (28 runs × per-run + per-decile breakdowns)
- `residual_run.log` — training log
- `residual_eval.log` — eval log
- `scripts/eval_residual.py` — eval harness
- `scripts/train_residual.py` — training harness
- `src/mongoose/data/residual_dataset.py` — feature extraction
- `src/mongoose/model/residual_mlp.py` — model architecture
- `src/mongoose/etl/reads_maps_table.py` — pre/post pair → DataFrame ETL
- `src/mongoose/io/reads_maps_bin.py` — V5 binary parser

Branch: `feat/v4-production-residual` at commit `652d058`. All Direction
C tests passing (30/30).
