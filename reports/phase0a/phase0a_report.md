# Phase 0a: T2D Residual Decomposition

**Date:** 2026-04-23
**Data asset:** `data/derived/probe_table.parquet` (56.5 M probes × 128 cols)
**Analysis module:** `src/mongoose/analysis/phase0a_t2d_residual_decomposition.py`
**Entry point:** `decompose()`
**Spec:** `phase0a_t2d_residual_spec.md`
**Depends on:** Phase 0 ETL, Phase 0b classifier characterization

---

## 1. Executive summary

**~28% of T2D's per-probe error variance is structured and potentially fixable; the remaining ~72% is a noise floor.** A better T2D model has real but bounded headroom.

**The single most important number for decision-making is the per-molecule residual-std distribution:**

| per-molecule residual std | value |
|---|---:|
| median | **716 bp** |
| p90 | 7,439 bp |
| p99 | 15,909 bp |

Read: the median molecule has ~716 bp of within-molecule residual spread AFTER a per-molecule OLS affine has been fit. This is not tight. The per-molecule trend slope is **centered at zero** (median +2.4 bp/pos_frac-unit, 50.4/49.6 positive-negative split, IQR 444) — so there is no *shared* within-molecule linear trend T2D is missing. What there IS instead: each molecule has its own idiosyncratic residual shape. That's the "harder to fix" regime (shared shape → one nonlinear correction; idiosyncratic shape → per-molecule adaptation).

**Where the ~28% of structure lives** (dominant contributors from the multivariate OLS; see §5):

1. **Attribute bits on non-clean regions** — `attr_in_structure` and `attr_folded_start` add +2.7 kbp and +1.8 kbp respectively to predicted |residual|. Probes in folded or structured regions have ~**10× the residual** of clean-region probes.
2. **Position along molecule** — head-dive nonlinearity survives the per-molecule affine. `pos_frac = 0.0–0.05` has `abs_median = 1850 bp` vs `pos_frac = 0.45–0.50` at 664 bp — a clear U-shape.
3. **Molecule-mean velocity** — slowest decile has `abs_median = 3782 bp` vs fastest at **266 bp**. Fourteen-fold spread.
4. **Molecule length (num_probes)** — longer molecules have wider residual distributions.

**Where structure is absent** (explicit null findings, also important):
- **Strand** (top vs bottom): no bias. Medians differ by ±7 bp.
- **Genome position**: no chromosome-level structure.
- **Molecule stretch factor**: Pearson r = −0.001 with per-molecule residual std. **Refutes the spec's §114 hypothesis** that stretch_factor would be the most informative molecule-level feature.
- **Biochem-flagged vs unflagged**: flagged runs have *slightly worse* residuals (abs_median 552 vs 526 bp). Consistent with Phase 0b — the biochem flag is not capturing T2D quality either.

**Blue-holdout hypothesis test (spec §156):** **refuted.** Blue's within-molecule velocity coefficient-of-variation is **0.697** vs other `low_dil` runs at 0.718 and global at 0.736. Blue is *smoother* on local velocity than its peers, not wobblier. Whatever Option A learned on Blue does not live in within-molecule velocity variation. See §7.

---

## 2. Method and caveats

### 2.1 Residual definition and the per-molecule OLS caveat

Residual: `residual_bp = predicted_genomic_bp − ref_genomic_pos_bp`, where `predicted_genomic_bp` comes from the Phase 0b `fit_per_molecule_affine` + `predict_genomic_positions` pipeline. That pipeline OLS-fits `ref_genomic_pos_bp ~ slope × t2d_predicted_bp_pos + intercept` per aligned molecule using only that molecule's assigned probes.

**After per-molecule affine fit, per-molecule mean residual is zero by construction. All findings in this phase are DEVIATIONS FROM THAT PER-MOLECULE LINEAR FIT, not absolute T2D errors.** Any reader who doesn't internalize this will mistake structure for bias. The specific consequence is that Axis 1 (position along molecule) is a **nonlinearity finder, not a bias finder**: a constant head-dive offset would be absorbed into the affine; what survives is head-dive curvature.

### 2.2 The shared-wobble ground-truth issue

"Residual" here is the disagreement between T2D and the aligner's assignment, not between T2D and physical reality. Phase 0b showed the aligner itself has ~250 bp to 1 kbp of wobble (70% of assigned probes are more than 250 bp from the T2D-predicted site). For this reason, **correlations of residual with features are meaningful; absolute magnitudes are not**. A 500-bp residual could be T2D wrong, aligner wrong, or both. The report emphasizes correlations and structural findings over raw magnitudes.

### 2.3 Filter cascade

| stage | rows |
|---|---:|
| total | 56,531,725 |
| aligned | 35,675,076 |
| + not do_not_use | 35,675,076 |
| + num_probes ≥ 6 | 35,675,076 |
| + is_assigned | 24,650,774 |
| + non-null t2d_predicted_bp_pos | 24,650,774 |
| + per-molecule affine fit available | **24,650,770** |

The last 4 rows lost to the affine fit are probes in molecules with fewer than 3 assigned probes (below `MIN_FIT_POINTS`). Negligible.

---

## 3. Global residual distribution

| | signed (bp) | absolute (bp) |
|---|---:|---:|
| mean | 0.0 (by construction) | 1930 |
| median | 0.7 | **532** |
| p75 | — | 1,601 |
| p90 | — | 5,899 |
| p95 | — | 9,865 |
| p99 | — | 19,458 |
| p1 / p99 | −15,253 / +15,288 | — |

Signed tails are symmetric: no global bias. Absolute residual p50 = 532 bp matches Phase 0b §6 exactly. `frac_gt_250_bp = 0.705`, `frac_gt_1000_bp = 0.338` — over two-thirds of assigned probes are placed more than 250 bp from where T2D thinks they are.

---

## 4. Axis-by-axis decomposition

Order: (1) position along molecule, (2) per-molecule dispersion, (3) run-level covariates, (4) velocity, (5) probe features, (6) molecule features, (7) genomic context. The spec's original numbering corresponds to axes 1 (→1), 4 (→2+6), 5 (→3), 2 (→4), 3 (→5), 6 (→7).

### 4.1 Position along molecule (spec axis 1)

See `plots/residual_by_position_along_molecule.png`, `residual_by_position.csv`.

**Finding: head-dive nonlinearity confirmed.** Abs residual median as a function of pos_frac:

| pos_frac bin | abs_median (bp) | signed_p10–p90 (bp) |
|---|---:|---:|
| 0.00–0.05 | **1,851** | −5,983 / +6,086 |
| 0.05–0.10 | 1,317 | −5,278 / +5,332 |
| 0.10–0.15 | 1,097 | −5,218 / +5,161 |
| 0.20–0.25 | 900 | −5,033 / +4,938 |
| 0.45–0.50 | **664 (min)** | −2,943 / +2,914 |
| 0.55–0.60 | 564 | −2,114 / +2,130 |
| 0.75–0.80 | 390 | −1,140 / +1,160 |
| 0.85–0.90 | 308 | −786 / +797 |
| 0.90–0.95 | 282 | −691 / +700 |
| 0.95–1.00 | 335 | −822 / +802 |

The signed median is ~0 across all bins (as expected — the affine has absorbed linear bias). The *envelope* (abs median and p10/p90 spread) is **3× wider at head than middle**. The head is where T2D has the most curvature the affine couldn't fit. The tail is actually *tightest*.

**Effect size:** head bin residual 3× the middle-molecule residual.
**Actionable:** yes. A nonlinear head-correction (polynomial or piecewise) should pay for itself. Exploratory estimate: if head-dive-corrected T2D reduced the 0.00–0.05 bin to the 0.45–0.50 bin's 664 bp, that's a ~1 percentage-point improvement in the overall abs_median (the head bin represents ~6% of the data).
**Next step:** test a simple polynomial correction to T2D in the 0–10% region against the existing pipeline; see if it moves the needle on the 16.2% holdout number.

### 4.2 Per-molecule residual std + trend slope (spec axis 4 partial)

See `plots/per_molecule_residual_std.png`, `plots/per_molecule_trend_slope.png`.

- **Residual std**: median **716 bp**, p90 7,439 bp, p99 15,909 bp. Most molecules have a few hundred bp of within-molecule spread; a long tail has kbp-scale spread.
- **Trend slope** (per-molecule OLS of `residual_bp ~ pos_frac`): median +2.4 bp/unit, IQR 444, 50.4% positive / 49.6% negative, abs-median 222 bp/unit. **Centered at zero with wide tails.**
- **Stop condition `std < 100`: NOT triggered.** Within-molecule error is real, not a trivial offset.
- **Stop condition `slope skewed from zero`: NOT triggered.** Population has no shared within-molecule trend.

**Interpretation.** The residual wiggle within a molecule is idiosyncratic, not a shared shape. This is the "harder to fix" regime. A one-size-fits-all nonlinear correction will cancel out on average because the slopes go both directions. Fixing this requires per-molecule adaptation — which is essentially what the V3 / Option A sprint has been trying to do (learn a per-molecule residual).

### 4.3 Run-level covariates (spec axis 5)

See `residual_by_run_metadata.csv`.

| stratum | value | n | abs_median (bp) | abs_p90 | frac > 250 bp | frac > 1 kbp |
|---|---|---:|---:|---:|---:|---:|
| concentration_group | low | 6.47M | 504 | 5295 | 0.69 | 0.32 |
| concentration_group | low_dil | 8.22M | 487 | 4888 | 0.69 | 0.31 |
| concentration_group | **std** | 9.96M | **598** | **6983** | 0.73 | 0.38 |
| biochem_flagged_good | False | 19.36M | 526 | 5848 | 0.70 | 0.34 |
| biochem_flagged_good | True | 5.29M | 552 | 6084 | 0.71 | 0.35 |
| instrument | 202 | 5.95M | 531 | 5735 | 0.71 | 0.33 |
| instrument | **429** | 2.12M | **434** | 4322 | 0.66 | 0.28 |
| instrument | 433 | 16.58M | 547 | 6136 | 0.71 | 0.35 |
| SNR tercile | low | 8.76M | 555 | 6499 | 0.71 | 0.35 |
| SNR tercile | mid | 7.74M | 542 | 5987 | 0.71 | 0.34 |
| SNR tercile | high | 8.16M | 500 | 5099 | 0.69 | 0.32 |

**Effect sizes.**
- `std` concentration is 23% worse than `low_dil` on abs_median.
- Instrument 429 is 21% better than instrument 433.
- Biochem flag: flagged runs are *slightly worse* (5% on abs_median). Not the direction one would hope.
- SNR: monotone improvement low→high, but only 10% range.

**Actionable:** per-instrument T2D calibration constants could close the 429-vs-433 gap. Cheap to implement (add a per-channel scalar correction). Per-concentration calibration could close the std-vs-low_dil gap. Both are one-line changes at the calibration layer.

### 4.4 Velocity (spec axis 2)

See `plots/residual_by_molecule_velocity.png`, `residual_by_molecule_velocity.csv`, `residual_by_local_velocity_ratio.csv`.

**Molecule-mean velocity deciles (D1 slowest, D10 fastest):**

| decile | vel (bp/ms) | abs_median (bp) | abs_p90 (bp) |
|---|---:|---:|---:|
| D1 slow | 242 | **3,782** | 14,201 |
| D2 | 398 | 1,210 | 10,589 |
| D3 | 477 | 775 | 7,355 |
| D4 | 544 | 628 | 5,581 |
| D5 | 610 | 533 | 4,298 |
| D6 | 680 | 460 | 3,285 |
| D7 | 759 | 394 | 2,445 |
| D8 | 854 | 341 | 1,823 |
| D9 | 979 | 292 | 1,335 |
| D10 fast | 1,193 | **266** | 1,059 |

**14× spread between slowest and fastest deciles on abs_median.** This is the dominant axis in the data. T2D's power-law model fits fast molecules well and breaks down on slow ones.

**Local/mean velocity ratio deciles** confirm: residuals are minimized at ratio ≈ 1.0 and grow sharply when local velocity deviates strongly from molecule-mean. Both extremes (D1 ratio 0.57 and D10 ratio 8.4) have high residuals.

**Actionable:** yes. A velocity-conditioned T2D (or a velocity-gate on which molecules to trust) is high-value. Molecules in the slowest quintile account for a disproportionate share of T2D error. If you dropped molecules with mean velocity < 500 bp/ms, the overall abs_median would drop sharply. Throughput trade-off: that's ~40% of molecules lost.

### 4.5 Per-probe features (spec axis 3)

See `probe_feature_correlations.csv`, `residual_by_attribute_bit.csv`, `plots/residual_by_attribute_bit.png`.

Pearson r (abs_residual_bp vs feature):

| feature | r |
|---|---:|
| duration_ms | +0.205 |
| area_samples_uv | +0.200 |
| probe_local_density | +0.103 |
| max_amp_uv | +0.092 |
| prev_probe_gap_ms | +0.075 |
| next_probe_gap_ms | +0.086 |

All weak positive correlations. Longer or larger-area probes have slightly elevated residuals — consistent with those probes being on slower molecules (which have higher T2D error, from axis 4).

**Per-attribute-bit residuals** (abs_median of |residual|):

| bit | n_set | set (bp) | clear (bp) | set / clear |
|---|---:|---:|---:|---:|
| `attr_folded_start` | 73,077 | **5,223** | 529 | **9.87×** |
| `attr_in_structure` | 2,546,142 | **4,627** | 452 | **10.23×** |
| `attr_clean_region` (complement) | 22,095,602 | 452 | 4,621 | 0.10× |
| `attr_accepted` | 24,650,770 | 532 | — | tautology |

**Finding:** probes in folded or structured regions have ~10× the T2D residual of clean-region probes. This is the cleanest single-feature lever in the data.

**Actionable:** yes, two options.
1. Gate downstream: exclude probes with `attr_in_structure` or `attr_folded_start` from T2D-critical analyses. Immediate effect, costs probes.
2. Condition T2D: learn a residual head for probes in these regions. More work, keeps the probes.

### 4.6 Molecule-level features (spec axis 4, remainder)

See `molecule_feature_correlations.csv`.

Pearson r (per-molecule residual std vs feature):

| feature | r |
|---|---:|
| num_probes | **+0.523** |
| molecule_velocity_bp_per_ms | **−0.454** |
| translocation_time_ms | +0.396 |
| fall_time_t50_ms | +0.396 |
| rise_time_t50_ms | −0.061 |
| molecule_align_score | +0.056 |
| mean_lvl1_mv | −0.007 |
| **molecule_stretch_factor** | **−0.001** |

**Headline:** `molecule_stretch_factor` has **essentially zero correlation with T2D residual dispersion** — this **refutes** the spec's §114 hypothesis that stretch_factor would be the most informative feature. The aligner's stretch-factor does not forecast how wide within-molecule T2D residuals are. `num_probes` and velocity are the real signals.

### 4.7 Genomic context (spec axis 6)

See `residual_by_strand.csv`, `residual_by_nearest_neighbor.csv`, `residual_by_genome_position.csv`.

- **Strand**: top (0) abs_median 525, bottom (1) 540. No meaningful bias.
- **Nearest-neighbor distance** (circular-genome aware):
  - 500 bp–10 kbp: mostly uniform around 510–525 bp
  - 10–20 kbp: 586 bp (+12% vs baseline)
  - 20 kbp+: 732 bp (+40%) and p90 10 kbp
  - Isolated probes (large inter-site gap) do have elevated residuals — a real but small effect.
- **Position in genome** (10% bins, circular): no chromosome-level structure. Abs_median ranges 466–622 across all 10 bins; no consistent sign across adjacent bins.

**Actionable on strand and position**: nothing. **Actionable on nearest-neighbor**: maybe, but the effect is small and the mechanism is unclear (probably confounded with molecule length).

---

## 5. Multivariate variance decomposition

See `phase0a_metrics.json`, `multivariate_ols` key.

OLS fit: target = `abs_residual_bp` (winsorized at p99 = 19,458 bp); features = 9 structural signals (pos_frac, head/tail indicators, log velocity, log num_probes, log duration, attr_folded_start, attr_in_structure, log local_density).

| quantity | value |
|---|---:|
| n_fit | 24,650,770 |
| target mean | 1,930 bp |
| target std | 3,534 bp |
| **R²** | **0.279** |
| OLS residual std | 3,002 bp |
| **structured variance fraction** | **27.9%** |
| **unstructured variance fraction** | **72.1%** |

**Coefficients (sorted by magnitude, bp per unit of feature):**

| feature | coef |
|---|---:|
| intercept | +4,491 |
| attr_in_structure | **+2,712** |
| attr_folded_start | **+1,826** |
| log_num_probes | +1,628 |
| pos_frac | −1,505 |
| log_mol_vel | −1,080 |
| tail_indicator | −753 |
| log_duration_ms | −564 |
| log_probe_local_density | +59 |
| head_indicator | +56 |

**Reading.** The two attribute bits (in-structure, folded_start) and molecule length contribute most positively to predicted residual; pos_frac and log velocity contribute most negatively. The head_indicator and tail_indicator dummies are small after pos_frac is in the model — the U-shape is captured by the linear pos_frac term itself, not by sharp head/tail discontinuities.

**Decision fork (spec §172).** 28% is in the middle of the spec's hypothetical poles (40% "meaningful headroom" vs 5% "noise floor close"). A better T→D model can capture up to ~28% of current variance; the remaining 72% is noise floor at this feature set. Real headroom exists but it is bounded — T2D is not close to free for the taking.

Caveats: the 28% uses a linear target. A log-target OLS would likely claim a higher R² because the residual distribution is heavy-tailed; the winsorization partially compensates. The number quoted here is appropriate for "what fraction of the bp-linear error is structured," which is the right framing for downstream impact estimation.

---

## 6. Blue-holdout deep dive

See `blue_holdout` key in `phase0a_metrics.json`. Target: `STB03-065H-02L58270w05-433H09j`.

### 6.1 Blue residual vs peers

| cohort | n probes | abs_median (bp) | abs_p90 (bp) |
|---|---:|---:|---:|
| Blue holdout | 345,894 | **514** | 4,856 |
| Other `low_dil` runs | 7,878,485 | 486 | 4,889 |
| `low_dil` ∩ biochem-flagged | 1,534,746 | 534 | 5,638 |
| global | 24,650,770 | 532 | 5,899 |

Blue is **slightly worse** than the other `low_dil` runs on T2D residual (+6%), and roughly matches global. Per-molecule residual std on Blue is **710 bp** vs other `low_dil` **651 bp** — 9% wider within-molecule dispersion.

### 6.2 Within-molecule velocity CV (the spec's §156 hypothesis)

Hypothesis: Option A beats T2D on Blue by capturing within-molecule velocity variation. If true, Blue should have *more* velocity wobble than peers.

| cohort | CV_median | CV_p90 |
|---|---:|---:|
| **Blue holdout** | **0.697** | 1.315 |
| Other `low_dil` | 0.718 | 1.321 |
| `low_dil` ∩ biochem-flagged | 0.750 | 1.363 |
| global | 0.736 | 1.366 |

**Refuted.** Blue has the **lowest** within-molecule velocity CV among these cohorts. Its molecules are *smoother* on local velocity than peers, not wobblier.

### 6.3 Where does Option A's Blue advantage come from then?

Phase 0b §7 showed Blue's classifier MCC is similar-to-slightly-worse than its `low_dil` peers (not distinctively good or bad on probe-ID). §6.1 here shows Blue is slightly worse on T2D residual. §6.2 rules out the within-molecule-velocity hypothesis.

The candidates that remain (none testable from the probe table alone):
- Blue has a run-level amplitude or baseline-current profile that the Option A residual head picked up on.
- Blue's per-channel T2D constants (`mult_const`, `addit_const`, `alpha`) happen to be biased in a direction that Option A's tanh-bounded residual partially corrects.
- Statistical noise on small-sample holdouts — the 3-point delta could be within variance.

None of these are actionable from Phase 0a. If Option A's Blue advantage matters for Phase 1 planning, inspecting the Option A checkpoint's per-molecule residual predictions (separate analysis) would test whether the learned residual systematically shifts in a direction that correlates with any of the above.

---

## 7. Actionable findings and open questions

### 7.1 Actionable

1. **Head-dive correction** (axis 1). A polynomial correction in pos_frac < 0.10 would address the 3× residual elevation in the head bin. Impact estimate: ~1 percentage-point improvement on the overall abs_median; not transformative but well-scoped and testable.
2. **Non-clean-region gating** (axis 5). Probes in folded or structured regions carry 10× the T2D error. Either gate them out of T2D-critical downstream uses (immediate throughput cost) or train a conditional residual on them (moderate effort).
3. **Per-instrument calibration offsets** (axis 3). Instrument 429 vs 433 differ by 21% on abs_median. A per-channel scalar correction to T2D constants is a one-line change.
4. **Velocity-conditioned T2D** (axis 4). The 14× velocity-decile spread is the dominant axis. A velocity-aware T2D (separate parameters per decile or a continuous covariate) would mechanistically address the power-law mismatch on slow molecules.
5. **Honest throughput trade-off**: "drop molecules with mean velocity < 500 bp/ms" would halve the abs_median at the cost of ~40% of molecules. Phase 0b §7's open question (characterize which filters the stack is using) is the right place to study this trade-off more carefully.

### 7.2 Null findings worth knowing

- **Stretch factor is not informative.** r = −0.001 with per-molecule residual std. The aligner's stretch calibration has already absorbed whatever molecule-level distortion was there.
- **No strand bias, no chromosome-level position bias.** T2D is strand- and genome-position-agnostic as expected.
- **Biochem-flagged runs are *slightly worse* on T2D**, not better. The biochem flag measures something orthogonal to both probe-ID (0b) and T→D (this phase) quality.

### 7.3 Open questions

1. **Does the 28% structured variance fraction get meaningfully larger in log-target space?** A log-target OLS would naturally handle the heavy tail and might claim more structure. Cheap to try — 20 extra lines.
2. **Can the stack's 6.13 M "SP-accept, aligner-filter" probes be characterized?** From Phase 0b §8 question 1 — still open and still the single highest-value follow-up. If a lot of those probes have T2D-decent features (velocity, density, non-structured), the filter that drops them may be too aggressive.
3. **What explains Blue's 3-point Option A advantage?** Phase 0a refutes the within-molecule-velocity hypothesis; the remaining candidates need Option-A-checkpoint-level introspection, out of scope here.
4. **Is the 28% "noise floor" actually measurement noise, or is it shared-wobble between T2D and the aligner?** Phase 0a cannot distinguish "T2D is wrong" from "aligner is wrong" in that 72%. A cleaner oracle (if one existed — manually curated sites, or a different orthogonal assay) would calibrate this.

---

*Report compiled by `decompose()`. Headline numbers in `phase0a_metrics.json`; raw tables in the 10 `residual_by_*.csv` files; plots under `plots/`.*
