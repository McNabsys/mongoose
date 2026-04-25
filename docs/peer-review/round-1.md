# Deep Think Peer-Review Prompt — Draft (2026-04-18)

Paste the content below between the `---BEGIN PROMPT---` / `---END PROMPT---` markers into Gemini Deep Think. ~1800 words, comprehensive one-shot.

---BEGIN PROMPT---

I need an independent ML peer review of a training recipe that isn't converging. Please read the full context below, then answer the specific questions at the end. My goal: identify whether this is a recipe flaw (fixable with loss/hyperparameter surgery), an architecture flaw (fixable with a head/init change), or a fundamentally wrong framing that needs deeper rework.

## PROBLEM DOMAIN

Nanopore translocation waveform analysis. Each input is a 1D time-series of ionic current signal (~1800-10,000 samples, 40 kHz sample rate) from a single DNA molecule passing through a nanopore. The DNA has been labeled with molecular "probes" at known reference positions in base-pair coordinates. The model must detect WHERE in the time-series each probe passed the pore (in sample-index coordinates), then map those detected events onto the reference genome's probe coordinates via a learned velocity profile.

Ground truth comes from an older classical signal processing pipeline ("wfmproc"). That pipeline provides per-molecule:
- `warmstart_probe_centers_samples`: array of sample indices where wfmproc detected probes (typically 5-20 per molecule).
- `reference_bp_positions`: the bp coordinates those probes correspond to on the reference genome.
- `n_ref_probes`: count of reference probes (ground truth peak count).

We believe wfmproc has detection limitations (false negatives, cluster merging) and we want the neural model to exceed it. But for training we treat wfmproc's peak positions as labels.

## ARCHITECTURE

"T2D U-Net" — a 1D U-Net with three output heads:

1. **Probe head** → `probe_logits` (raw logits, `[B, T]`) + `probe = sigmoid(probe_logits)`. Should produce a soft heatmap with sharp peaks at each probe's sample index.
2. **Velocity head** → `raw_velocity = softplus(v)` (strictly positive bp/sample velocity). Expected to be smooth — DNA moves through the pore at a roughly constant but noisy rate.
3. **Cumulative bp** = `cumsum(raw_velocity * mask)` — running integral of velocity giving predicted bp coordinate at each sample.

At inference, peak positions in sample-space are extracted via velocity-adaptive non-maximum suppression on the probe heatmap with confidence threshold 0.3. Those indices, evaluated against the cumulative_bp function, give predicted bp coordinates, which are compared to reference bp coordinates to score the model.

## LOSS FORMULATION (current, tuned)

Four components, each divided by a hardcoded scale divisor to balance gradients, then weighted by a lambda schedule:

```
L_total = scaled_probe + λ_bp·scaled_bp + λ_vel·scaled_vel + λ_count·scaled_count
```

Where:

**L_probe (probe head):** blend between a supervised term and a self-supervised term, scheduled by training epoch:
```
L_probe = blend · L_BCE + (1 - blend) · L_peakiness
```
- `L_BCE = weighted_BCE_with_logits(probe_logits, warmstart_gaussian_target)` with weight `(1 + probe_pos_weight · target)` at each sample. `probe_pos_weight=50`, so at peak centers (target≈1) samples are 51× weighted vs negatives. BCE-on-logits chosen for numerical stability vs applying sigmoid then MSE (sigmoid saturation killed gradients in earlier attempts).
- `warmstart_gaussian_target` is a sparse `[T]` tensor with Gaussian bumps of amplitude 1.0 at each wfmproc-detected peak center, sigma ≈ `max(1.5, duration_samples/6)`, so each Gaussian spans ~5 samples. ~99% of target samples are 0.
- `L_peakiness = mean(1 - max_pool(heatmap, window=20))`. Self-supervised. Encourages at least one near-1.0 value in every sliding window. Introduced as a regularizer for post-warmstart training.
- `blend` follows a schedule: 1.0 during `warmstart_epochs - warmstart_fade_epochs`, linear fade to 0 over `warmstart_fade_epochs`, floored at `min_blend` thereafter. Tested values: `warmstart_epochs=8, warmstart_fade_epochs=4, min_blend=0.1`.

**L_bp (velocity-integrated localization):** soft-DTW distance between (`cumulative_bp` evaluated at detected peak indices) and (`reference_bp_positions`), span-normalized. Critical: **the peak-index extraction is `@torch.no_grad`**, so this loss's gradient flows only to `cumulative_bp`, NOT to the heatmap. If no peaks cross the NMS threshold, this loss is 0 and contributes no gradient.

**L_vel:** MSE between `raw_velocity` sampled at detected peak indices and target velocities derived from peak widths via FWHM estimation. Same property as L_bp: `@torch.no_grad` peak extraction, so gradient flows only to the velocity head, not the heatmap.

**L_count:** smooth L1 between `sum(heatmap)` and `n_ref_probes`, divided by `n_ref_probes`.

**Schedule during warmstart:** `λ_bp = λ_vel = λ_count = 0.5 × base_lambda` (ramps to 1.0 after warmstart). We currently use `base_lambda = 1.0` for all three.

**Scale divisors (for gradient balance):** `scale_probe=1`, `scale_bp=300000`, `scale_vel=5000`, `scale_count=50` — calibrated to make each scaled component ~1.0 at epoch 1 given observed raw magnitudes.

**Probe head initialization:** final conv bias = −3.0 so at init the heatmap = sigmoid(−3) ≈ 0.05 everywhere. Chosen to match the ~0.5% peak-sample fraction (sparse-detection init trick à la RetinaNet).

## DATA

Single preprocessed cache: 55,038 molecules from one experimental run. 80/20 train/val split. Future multi-run work would add 29 more caches; this session is on the single run.

## RUN RESULTS (10 epochs, ~4 hours wall)

**Training loss (blend schedule in column 2, all scaled values):**

| Ep | blend | probe | bp | vel | count | val_loss | raw_bp | raw_vel | raw_count |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 1.12 | 1.06 | 5.49 | 0.79 | 6.61 | 318,367 | 27,473 | 39.48 |
| 2 | 1.0 | 1.10 | 0.50 | 2.96 | 0.77 | 2.52 | 149,509 | 14,819 | 38.36 |
| 3 | 1.0 | 1.14 | 0.35 | 2.31 | 0.70 | **1.90** best | 105,426 | 11,525 | 35.19 |
| 4 | 1.0 | 1.17 | 0.64 | 1.94 | 0.65 | 2.28 | 192,446 | 9,689 | 32.46 |
| 5 | 0.75 | 1.23 | 0.38 | 2.08 | 0.47 | 4.14 | 114,325 | 10,404 | 23.34 |
| 6 | 0.50 | 1.25 | **4.74** | 3.12 | 0.30 | 6.17 | 1,423,421 | 15,619 | 15.06 |
| 7 | 0.25 | 1.21 | 0.32 | 1.62 | 0.13 | 3.52 | 96,361 | 8,118 | 6.74 |
| 8 | 0.10 | 1.13 | 0.02 | 0.20 | 0.04 | 2.25 | 5,821 | 982 | 2.11 |
| 9 | 0.10 | 1.14 | 0.002 | 0.02 | 0.03 | 4.81 | 615 | 124 | 1.71 |
| 10 | 0.10 | 1.14 | 0.01 | 0.02 | 0.03 | 2.62 | 3,457 | 87 | 1.68 |

**Peak-match F1 (tolerance=50 samples vs wfmproc ground-truth peak centers, over the full 55k-mol val split):**

- **Best model (epoch 3):** F1 = 0.037. Precision = 0.23, Recall = 0.02.
- **Final model (epoch 10):** F1 = 0.000. Precision = 0.001, Recall = 0.

**Heatmap state diagnostic (over 200 validation molecules):**

- Epoch 3: mean max-amplitude per molecule = 0.38, 86% of molecules reach 0.3+ somewhere. Detected peaks at NMS threshold 0.3: 188 total (≈ 1 per molecule). At threshold 0.1: 29,599 (≈ 150 per molecule — noise-like).
- Epoch 10: mean max-amplitude per molecule = 0.047 — exactly the sigmoid(-3) init prior. Detected peaks at any threshold: 0. **The model relaxed back to its initialization.**

## WHAT THE DIAGNOSTIC TELLS US

Three regimes:

1. **Warmstart (epochs 1-4, blend=1.0):** BCE lifts heatmap amplitudes to ~0.4 mean-max, but not sharp or correctly-positioned. val_loss drops 6.61 → 1.90 then plateaus.
2. **Fade (epochs 5-7):** As peakiness takes over, bp/vel briefly destabilize (raw_bp spike to 1.4M at epoch 6), then the model finds a lower-loss solution by shrinking amplitudes.
3. **Post-fade (epochs 8-10, blend=0.1):** Heatmap returns to the init prior (~0.05 everywhere). L_bp and L_vel go silent because no peaks cross threshold 0.3. Peakiness is minimized at ~0.95 (high but saturated, weak gradient). Count loss raw=1.7 (heatmap sum near zero matches small n_ref). Training has found a stable local minimum where the model does nothing and all losses are small.

## RECIPE EVOLUTION — PRIOR ATTEMPTS

In rough chronological order; each fixed something but exposed the next problem:

1. **Original focal loss, α=0.25:** heatmap collapsed to flat zero. Focal down-weights positives with α<0.5, de-emphasizing the 8 peak samples vs 1800 negatives.
2. **`min_blend=0.1` with focal:** same collapse. Focal signal stays weak even with blend floor.
3. **Neutralize count loss (`scale_count=1e6`):** heatmap went to uniform ~0.4 everywhere. Count was the only term demanding sparsity, once gone no term differentiates localized peaks from uniform amplitude.
4. **MSE instead of focal:** heatmap collapsed to zero via sigmoid saturation. Predictions reached ~0.0003 at peak centers where target=1.0; sigmoid' at that point ≈ 0.0003, so MSE gradient × sigmoid Jacobian ≈ vanishing.
5. **Positive-weighted MSE with pos_weight=50/500:** still saturated — same sigmoid gradient issue.
6. **BCE-with-logits + pos_weight=50 + init-bias −3:** stable numerics, but see current-run results above. Model reaches uniform mid-amplitude rather than sparse peaks, then collapses back to init under peakiness.

## MY HYPOTHESIS (what I think is going on)

The loss system has multiple degenerate local minima because the terms interact:

- **L_probe** alone prefers uniform target-matching (BCE pulls pred toward a soft Gaussian, which on average is ~0.04 target so "predict 0.04 everywhere" has low BCE).
- **L_peakiness** alone prefers uniform high (max-pool everywhere = 1.0 → loss = 0).
- **L_count** prefers `sum = n_ref` (achievable with any amplitude distribution).
- **L_bp and L_vel** are gradient-silent when no peaks cross NMS threshold — they don't vote on the heatmap shape.

The only term that actively demands "sharp peaks at specific sample positions" is L_probe (via the sparse Gaussian target in BCE). But BCE's gradient is averaged over 1800 samples, most of which are ~zero on both sides, producing tiny total gradient signal even with `pos_weight=50`. The model is not getting strong enough localization pressure.

## QUESTIONS (please answer each)

**Q1 — Loss design:** Is this loss recipe fundamentally broken, or is it mostly right and we need one more term / fix? If mostly right, what do you think the minimum-viable additional term would be to force sparse-localized peaks? (MSE-at-peak-centers only? Soft-IoU on heatmap? Dice loss? Weighted focal with α=0.99?)

**Q2 — Sparse-peak detection techniques:** For 1D sparse peak detection with soft Gaussian targets on noisy time-series, what's the current SOTA or most reliable approach? Examples in the literature you'd point to. Are we missing a standard trick — e.g., CenterNet-style focal with offset regression, or seq-to-seq with learnable anchors, or peak-picking-as-classification with objectness scores?

**Q3 — Gradient flow through NMS:** Our design makes L_bp and L_vel silent (gradient-wise) when no peaks are detected, creating a dead zone. Is this an architectural mistake that we should fix by making L_bp/L_vel produce heatmap gradient directly (e.g., by applying DTW against a soft-argmax rather than a discrete peak selection)? Or is the sparsity-via-NMS common practice and the fix is elsewhere?

**Q4 — Peakiness regularizer:** We have `L_peakiness = mean(1 - max_pool(heatmap, window))`. My suspicion is this term is actively harmful — it rewards uniform amplitude, not localized peaks, and its gradient is weak on flat surfaces. Should we drop it entirely? Replace with what?

**Q5 — Training schedule:** Should we abandon the "warmstart fade" concept in favor of keeping BCE (or equivalent supervised term) at full weight forever, ramping lambda_bp/vel/count later? The fade seems to actively destabilize the model (epoch 6 spike, post-fade collapse). Or is the fade the right idea but our peakiness regularizer is the wrong thing to fade IN to?

Finally, if you think the whole framing is wrong (e.g., "treat this as seq-to-seq output of peak positions with an RNN decoder" or "use a transformer encoder with CTC loss"), please say so — I'd rather burn the recipe and start over than grind on a dead end.

---END PROMPT---

## Notes for the human

- Fill in any project-specific facts before submitting if needed (the prompt is mostly self-contained).
- The recipe-evolution list is deliberately blunt about "what failed" — Deep Think does better when you're honest about failure modes rather than pitching a polished story.
- Q5 is where I'd most want an outside opinion; the schedule has felt like the least-principled choice throughout.
- If Deep Think asks for the architecture code or a specific waveform image, the `docs/plans/2026-04-18-v1-training-plan-design.md` file in the repo has additional detail you can paste.
