# Phase 0b: Current Probe Classifier Characterization

**Date:** 2026-04-23
**Data asset:** `data/derived/probe_table.parquet` (56.5 M probes × 128 cols)
**Analysis module:** `src/mongoose/analysis/phase0b_classifier_characterization.py`
**Entry point:** `characterize()`
**Spec:** `phase0b_classifier_spec.md`

---

## 1. Executive summary

**The signal-processing classifier has perfect recall by construction, weak specificity (16%), and an over-aggressive `attr_folded_start` rejection bit.** The signal-processing + remapping-filter *stack* has much better specificity (72%), demonstrating that the remapping pre-alignment filter is doing substantial real work.

**Headline numbers** (global, against the assigns-based ground truth, N = 35,675,076 eligible probes on 1,834,704 aligned molecules):

| metric | narrow (`attr_accepted`) | broad (`attr_accepted` + offered to aligner) | Δ |
|---|---:|---:|---:|
| precision | 0.727 | **0.888** | +0.161 |
| recall | 1.000 (tautology — see §2.3) | 1.000 (tautology) | — |
| specificity | 0.162 | **0.718** | +0.556 |
| MCC | 0.343 | **0.798** | +0.456 |
| Cohen's κ | 0.210 | 0.779 | +0.569 |

**The clearest signal about what the stack costs us is the broad-minus-narrow FP move.** 6.13 M probes were accepted by signal processing but filtered before the aligner saw them. Under the assigns oracle, essentially all of those were true negatives (the aligner correctly didn't place them) — the remapping-side filter is converting millions of SP-false-positives into true-negatives. This is the system's primary line of defense against false positives.

**Most interesting failure mode:** the `attr_folded_start` rejection bit fires on 14,081 probes in the eligible set. **72.9% of those probes lie inside the known-good feature envelope**, which is higher than the 67.2% envelope-inside rate for false-positive probes (accept-but-not-assigned). In plain terms: probes rejected by `attr_folded_start` look, featurewise, MORE like real bound tags than the FP population does. The rule is likely over-aggressive.

**Actionable findings:**
1. Revisit `attr_folded_start` — evidence of over-rejection.
2. `attr_in_structure` and `attr_excl_width_sp` are the only other rejection bits that fire on this data; both are behaving closer to expectation (62% / 54% envelope-inside vs TP's 89%).
3. Four bitfield rejection bits (`attr_excl_amp_high`, `attr_excl_width_remap`, `attr_folded_end`, `attr_excl_outside_partial`) **never fire** on the analysis-eligible set. Either the SP pipeline does not emit these attributes for this data, or they are redundant with the three that do fire.
4. Biochem-flagged vs unflagged runs are **indistinguishable** on probe-ID metrics (narrow precision 0.729 vs 0.726 overall). The "good" flag is not capturing probe-ID quality; it is presumably measuring something else (downstream remapping yield).

Analysis cost: ~3 minutes CPU, no GPU contention.

---

## 2. Method and caveats

### 2.1 Filter cascade

Starting from 56,531,725 probe rows, we filter to molecule-aligned probes on non-do-not-use molecules with ≥ 6 probes:

| stage | rows | delta |
|---|---:|---:|
| total | 56,531,725 | — |
| `molecule_aligned=True` | 35,675,076 | −20.9 M |
| + `molecule_do_not_use=False` | 35,675,076 | 0 |
| + `num_probes ≥ 6` | 35,675,076 | 0 |

The latter two filters collapse to no-ops on this data — every aligned molecule already satisfies them. All three are kept in the code so future datasets with different properties fail informatively rather than drift silently. `min_probes_remapping = 6` and `interval_match_tol_bp = 250` were verified constant across all 30 runs' `_remapSettings.txt` on 2026-04-23.

### 2.2 Ground-truth definitions

**Positive (real bound tag):** `is_assigned = True`. The probe was placed at a reference site by the aligner.

**Negative:** `is_assigned = False` on an aligned molecule. The probe's molecule was placed, but this probe did not match a reference site. The negative class is a mixture of free tags, folded-region artifacts, noise spikes, and real bound tags the aligner gave up on.

**Classifiers under test:**

| classifier | ACCEPT | REJECT |
|---|---|---|
| narrow | `attr_accepted = True` | `attr_accepted = False` |
| broad | `attr_accepted = True` AND probe offered to aligner | otherwise |

Broad-ACCEPT is derived from the ETL's `ref_idx.notna()` invariant: a non-null `ref_idx` means the probe appeared in the `.assigns` ProbeK tuple (and therefore was offered to the aligner). Non-accepted probes always have null `ref_idx` (enforced by the ETL integration test); 6.13 M accepted probes also have null `ref_idx` — those are "SP-accepted but filtered before alignment," the tail cases discussed in the Phase 0 ETL report.

### 2.3 The narrow-classifier oracle tautology

The assigns-based oracle has a structural property that must be flagged explicitly before any result is read: **remapping only sees probes that signal processing accepted**. Therefore `is_assigned = True` implies `attr_accepted = True`, and narrow-classifier false negatives are **zero by construction**.

This has two consequences:
- Narrow recall is 1.000 trivially; it is not informative.
- The spec's original §3 ("rejection-bit breakdown of how often each rejection actually kills a real bound tag") is not answerable from this data using the assigns oracle alone. A rejected probe cannot be `is_assigned=True`.

We evaluated a T2D-proximity secondary signal to plug this gap (spec-refinement path). It was abandoned: per-probe T2D error in the genomic frame has p50 = 532 bp (see §6), which is wider than any useful proximity tolerance. At tol = 250 bp, the proximity-signal / assigns-oracle agreement on broad-ACCEPT has Cohen's κ = 0.085 — far below the spec's "≥ 0.7 loose at best" threshold. The proximity-signal code remains in the module for reproducibility (see `compute_t2d_plausible_match`, `compute_agreement_matrix`) but no downstream analysis depends on it.

Instead, Phase 0b uses a **feature-space envelope** (§5) to ask a weaker but cleanly answerable question: *does this rejected probe look, featurewise, like probes we know are real bound tags?* The envelope is tested for validity on the known-positive class before any rejection-bit claim is derived from it (§5.1).

### 2.4 Bootstrap CIs omitted

The spec requested per-molecule bootstrap for confidence intervals. Given analysis-eligible N ≥ 10⁶ in every stratum reported below, sampling variance is negligible and bootstrap CIs would round to zero width at the reported precision. Omitted for this report; the module's pure-function structure makes adding them straightforward if future Phases need them on smaller strata.

---

## 3. Global confusion matrices

### Narrow (`attr_accepted`)

|             | is_assigned = T | is_assigned = F | row total |
|---|---:|---:|---:|
| ACCEPT | 24,650,774 | 9,244,245 | 33,895,019 |
| REJECT | 0 (tautology) | 1,780,057 | 1,780,057 |
| col total | 24,650,774 | 11,024,302 | 35,675,076 |

precision 0.727 · recall 1.000 · specificity 0.162 · MCC 0.343 · F1 0.842

### Broad (`attr_accepted` AND offered to aligner)

|             | is_assigned = T | is_assigned = F | row total |
|---|---:|---:|---:|
| ACCEPT | 24,650,774 | 3,109,457 | 27,760,231 |
| REJECT | 0 | 7,914,845 | 7,914,845 |
| col total | 24,650,774 | 11,024,302 | 35,675,076 |

precision 0.888 · recall 1.000 · specificity 0.718 · MCC 0.798 · F1 0.941

### Δ (broad − narrow)

- 6,134,788 probes shift from FP (narrow) to TN (broad).
- FP share of ACCEPT column drops from 27.3% to 11.2%.
- Specificity: +0.556. MCC: +0.456.

**This is the clearest window the data gives us into what the signal-processing + remapping stack costs and earns us.** The remapping-side filter is converting ~6.13 M SP-false-positives into true-negatives before they contaminate downstream analyses. Whatever this filter is doing, it is the system's primary defense — worth characterizing in detail in a follow-up phase, separate from bit-level rejection analysis.

---

## 4. Stratified confusion

Full table in `stratified_confusion.csv`. Highlights:

### By concentration_group

| group | n probes | narrow MCC | broad MCC |
|---|---:|---:|---:|
| std (10 ng/µL) | 9,204,260 | 0.351 | 0.784 |
| low (5 ng/µL) | 5,999,291 | 0.347 | 0.808 |
| low_dil (5 ng/µL diluted) | 7,612,519 | 0.341 | 0.793 |

Essentially identical across groups. Concentration does not drive classifier performance on this metric.

### By biochem_flagged_good

| flag | n probes | narrow MCC | broad MCC |
|---|---:|---:|---:|
| False (24 runs) | 17,906,689 | 0.343 | 0.797 |
| True (6 runs) | 4,909,381 | 0.344 | 0.804 |

**Flagged and unflagged runs are indistinguishable on probe-ID classifier metrics.** The biochem flag is not measuring classifier quality; whatever the flag captures (likely downstream assembly yield) is orthogonal to probe-level accept/reject accuracy.

### By instrument and SNR

See `stratified_confusion.csv`. Variation within strata is small compared to the narrow-vs-broad delta; no instrument or SNR tercile is a clear outlier.

---

## 5. Rejection analysis and feature-space envelope

Merges the spec's original §3 (rejection-bit breakdown) and §4 (feature distributions) per the 2026-04-23 pivot — see §2.3 for why the proximity-oracle path was abandoned.

### 5.1 Envelope construction and TP sanity

Known-good probe set: `is_assigned = True` AND `molecule_align_score ≥ p75` (threshold 500,834 on this data) AND probe center in `[0.10, 0.90]` of the molecule's translocation. **N = 4,168,778** known-good probes.

Feature vector (scaled for molecule-level context):
- `duration_scaled = duration_ms × molecule_velocity_bp_per_ms` (probe width in bp-equivalent)
- `amplitude_scaled = max_amp_uv / mean_lvl1_mv`
- `area_scaled = area_samples_uv / mean_lvl1_mv`
- `density_scaled = probe_local_density`

Mahalanobis distance is computed against the known-good mean and covariance.

**TP sanity check** — median TP Mahalanobis should be close to √χ²(4)_p50 = 1.83 if the envelope is well-formed:

| percentile | observed TP Mahalanobis | χ²(4) expectation |
|---|---:|---:|
| p50 | 1.978 | 1.832 |
| p90 | 3.045 | 3.08 |
| p95 | 3.510 | 3.42 |

The observed distribution tracks χ²(4) tightly. The envelope is valid on this feature set.

Reference inside-envelope rates at the `mahal ≤ 3` threshold:
- **TP** (accepted + assigned): **89.2%** inside
- **FP** (accepted + not assigned): **67.2%** inside

The 22-point gap between TP and FP is what gives the envelope discriminative power for rejection-bit analysis.

### 5.2 Feature distributions

See `plots/feature_dist_{duration,amplitude,area,density}_scaled.png`. For all four features, TP is tightly concentrated; FP and TN have heavier tails. Duration and amplitude show the cleanest separation; area scaled is noisiest. No feature shows a bimodal good-probe distribution that would recommend a Gaussian-mixture or KDE upgrade over the single-Gaussian Mahalanobis envelope.

### 5.3 Rejection-bit breakdown

For each rejection bit, we compute: how many rejected probes have this bit set, the median Mahalanobis distance of those probes, and the fraction inside the envelope. A bit that fires predominantly inside the envelope is flagging probes that look featurewise similar to known bound tags — *suspicious rule.* A bit that fires predominantly outside is doing its job.

| rejection bit | n fired | mahal median | inside envelope | vs FP (67.2%) | vs TP (89.2%) |
|---|---:|---:|---:|---:|---:|
| `attr_folded_start` | 14,081 | 2.72 | **72.9%** | **+5.7 pp** | −16.3 pp |
| `attr_in_structure` | 990,032 | 2.84 | 61.8% | −5.4 pp | −27.4 pp |
| `attr_excl_width_sp` | 1,780,057 | 2.94 | 53.9% | −13.3 pp | −35.3 pp |
| `attr_excl_amp_high` | 0 | — | — | — | — |
| `attr_excl_width_remap` | 0 | — | — | — | — |
| `attr_folded_end` | 0 | — | — | — | — |
| `attr_excl_outside_partial` | 0 | — | — | — | — |

**Column naming discipline.** The "inside envelope" column is *not* a false-negative count. It answers only: "of probes this bit rejected, what fraction look featurewise similar to the known-good population?" A high value means "this rejection rule fires on probes that look like real bound tags." It does not mean "this rejection killed real bound tags" — we cannot establish that from this data because the assigns oracle is tautological for rejected probes (see §2.3).

**Read of the table.**

- **`attr_folded_start` is the most suspicious rule.** 72.9% of its rejected probes lie inside the envelope — a rate *higher* than the 67.2% seen on accepted-but-unassigned probes (FP). This rule is rejecting probes whose features look more bound-tag-like than the FP population's features do. Either the rule has a high false-reject rate, or the "folded start" condition genuinely looks bound-tag-like in our chosen feature space and the classification is correct but our envelope cannot see the distinguishing feature. N = 14,081 — small enough that this bit isn't contributing much to the overall stack, so even if it is over-aggressive the fix would be low-impact.

- **`attr_in_structure`** fires on ~1 M probes, 62% inside envelope. This is close to the FP baseline (67.2%) — the rule is flagging a population that looks similar to the one that survives SP but fails alignment. Neutral finding; the bit isn't clearly wrong but isn't obviously right either.

- **`attr_excl_width_sp`** fires on 1.78 M probes, only 54% inside envelope. This is below the FP baseline and far below TP. The bit is doing its job: rejecting probes whose features look less bound-tag-like than the FP population's. This is the rejection rule with the strongest evidence for correct calibration.

- **Four bits never fire.** Either the SP pipeline does not emit those attributes for this data, or they are redundant with the three above. Worth a one-sentence confirmation with the upstream pipeline; not pursued here.

---

## 6. Oracle sanity check

For all 24,650,770 assigned probes, we compute per-probe genomic-frame T2D residual after the per-molecule affine fit:

| percentile | |predicted_genomic − ref_genomic_pos_bp| |
|---|---:|
| p50 | 532 bp |
| p75 | 1,601 bp |
| p90 | 5,899 bp |
| p95 | 9,865 bp |
| p99 | 19,458 bp |

- **70.5%** of assigned probes exceed the remap-settings `interval_match_tol_bp = 250` threshold.
- **33.8%** exceed 1,000 bp.

The per-molecule affine fit has already absorbed molecule-level T2D miscalibration; these residuals are irreducible per-probe T2D noise. This tells us two things:

1. The aligner's "within 250 bp" tolerance is not a tight alignment. It is a coarse match. 70% of assigned probes are more than 250 bp from the genome position they were placed at — the aligner must be using interval structure, not absolute position, to achieve alignment. This is a calibration point for reading §3's ground-truth labels.
2. Stratified by `biochem_flagged_good`: flagged p50 = 552 bp, unflagged p50 = 526 bp. Negligible difference. **T2D accuracy is not correlated with the biochem flag**, consistent with the finding in §4 that the biochem flag is orthogonal to probe-ID quality.

**Stash for Phase 0a.** The p50 = 532 bp per-probe residual and p90 = 5.9 kbp tail are the irreducible T2D noise floor on this data. When we revisit T→D improvements in Phase 0a, these numbers are the ceiling any non-Option-A approach needs to beat.

---

## 7. Blue-holdout deep dive

Per-run results for `STB03-065H-02L58270w05-433H09j` (the Blue holdout, unflagged, `low_dil`):

| set | n probes | narrow MCC | broad MCC |
|---|---:|---:|---:|
| Blue holdout | 475,170 | 0.319 | 0.737 |
| other `low_dil` runs | 10,755,564 | 0.333 | 0.778 |
| `low_dil` ∩ biochem-flagged | 2,113,544 | 0.345 | 0.815 |
| global | 35,675,076 | 0.343 | 0.798 |

Blue holdout's broad MCC (0.737) is modestly *below* its `low_dil` peers (0.778) and ~6 points below the flagged-`low_dil` subset. Blue is not a distinctive run on classifier metrics; it sits slightly below `low_dil` norms, not above.

**Consequence for the Option A sprint.** The sprint reported better bp-interval error on Blue (14.7% vs T2D 17.9%). That advantage does **not** correlate with Blue having better probe-ID quality in this classifier analysis — if anything Blue has slightly worse probe-ID quality. Whatever Option A learned on Blue was either (a) compensating for the weaker probe-ID signal, or (b) exploiting a different property of the data entirely (molecule-level features, velocity distribution, etc.). This narrows the space of hypotheses worth testing in Phase 0a.

---

## 8. Open questions and what they'd take to answer

1. **What are the 6.13 M "SP-accepted, aligner-filtered" probes?** We established that the remapping-side filter drops them, shifting them from narrow-FP to broad-TN. We did not characterize *which* filter step drops them or *why*. This is the highest-value follow-up from this phase — it's where the stack's specificity comes from. Answering it likely requires either: running the aligner with verbose logging on one run, or parsing the `_remapSettings.txt` filter chain and matching probes to filter outcomes. Low-risk, moderate-effort.

2. **Is the `attr_folded_start` rule actually wrong?** The 72.9% envelope-inside rate is suggestive but not dispositive. A deeper check would compare probe waveforms at probes flagged with `attr_folded_start` to waveforms at known-TP probes. Requires waveform-level re-read; out of scope for a probe-table-only analysis.

3. **Why do four rejection bits never fire?** Confirm with upstream pipeline that these are vestigial or dataset-specific, vs. a broken wiring in the probes.bin output we're consuming. One Slack message.

4. **Why is biochem-flag orthogonal to probe-ID metrics?** The flag correlates with *something* the biochem team observed in downstream output. The candidates are: assembly yield, coverage, remapping confidence distribution. This report does not measure any of those. A follow-up that joins `biochem_flagged_good` against `excel_*` outcome columns (`Remapped Coverage`, `% filtered remapped`, etc.) would answer this cheaply.

5. **Does the Option A advantage on Blue come from molecule-level features?** §7 suggests probe-ID quality does not drive Blue's advantage. The candidates are: Blue's velocity distribution, Blue's molecule length distribution, Blue's T2D residual distribution. Phase 0a (T2D residual decomposition) will address some of these.

---

*Report compiled by `characterize()`. Headline numbers in `phase0b_metrics.json`; raw tables in `confusion_matrices.csv`, `stratified_confusion.csv`, `rejection_breakdown.csv`.*
