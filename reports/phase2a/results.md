# Phase 2a Results — 2026-04-18 17:40

**Config:** 10 epochs, batch 32, lr 3e-4, warmstart 8/4, min_blend 0.1,
scale_bp=300k, scale_vel=5k, scale_count=50, probe_pos_weight=50,
BCE-on-logits, probe_head bias init -3.

**Data:** single-run cache (`STB03-060A-02L58270w05-433B23e`), 55,038 molecules, 80/20 split.

## Per-epoch training log

| Epoch | blend | lr | probe | bp | vel | count | val | raw_bp | raw_vel | raw_count |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 2.93e-4 | 1.12 | 1.06 | 5.49 | 0.79 | 6.61 | 318,367 | 27,473 | 39.48 |
| 2 | 1.00 | 2.73e-4 | 1.10 | 0.50 | 2.96 | 0.77 | **2.52** | 149,509 | 14,819 | 38.36 |
| 3 | 1.00 | 2.41e-4 | 1.14 | 0.35 | 2.31 | 0.70 | **1.90** *(best)* | 105,426 | 11,525 | 35.19 |
| 4 | 1.00 | 2.02e-4 | 1.17 | 0.64 | 1.94 | 0.65 | 2.28 | 192,446 | 9,689 | 32.46 |
| 5 | 0.75 | 1.57e-4 | 1.23 | 0.38 | 2.08 | 0.47 | 4.14 | 114,325 | 10,404 | 23.34 |
| 6 | 0.50 | 1.13e-4 | 1.25 | 4.74 | 3.12 | 0.30 | 6.17 | **1,423,421** | 15,619 | 15.06 |
| 7 | 0.25 | 7.4e-5 | 1.21 | 0.32 | 1.62 | 0.13 | 3.52 | 96,361 | 8,118 | 6.74 |
| 8 | 0.10 | 4.2e-5 | 1.13 | 0.02 | 0.20 | 0.04 | 2.25 | 5,821 | 982 | 2.11 |
| 9 | 0.10 | 2.2e-5 | 1.14 | 0.002 | 0.02 | 0.03 | 4.81 | 615 | 124 | 1.71 |
| 10 | 0.10 | 1.5e-5 | 1.14 | 0.01 | 0.02 | 0.03 | 2.62 | 3,457 | 87 | 1.68 |

## Peak-Match F1 evaluation (tolerance=50 samples, NMS threshold=0.3)

**Best model (epoch 3, val_loss=1.90):**
- Overall sum-of-counts: P=0.378 R=0.017 F1=0.033
- Per-molecule mean: P=0.227 R=0.021 **F1=0.037** (n=55,038)

**Final epoch 10:**
- Overall sum-of-counts: P=0.364 R=0.000 F1=0.000
- Per-molecule mean: P=0.001 R=0.000 **F1=0.000** (n=55,038)

**Threshold of 0.3 on F1 not met.** Autonomous Phase 4 launch aborted per plan.

## Diagnostic (heatmap amplitudes over first 200 molecules)

**Best model (epoch 3):**
- mean_max_per_mol = 0.38, median_max = 0.41, 86% of molecules cross 0.3 at some sample
- detected peaks @ thr=0.1: **29,599**, @ thr=0.3: **188**, references: 2,732
- Model over-predicts by 10× at threshold=0.1 (noisy-high) and under-predicts by 14× at threshold=0.3 (one stray per molecule)

**Final model (epoch 10):**
- mean_max_per_mol = 0.047 (≈ init-bias prior sigmoid(-3) = 0.047 exactly)
- detected peaks @ thr=0.1: 0, @ thr=0.3: 0
- **Model collapsed back to its initialization-prior everywhere.** Peak-detection ability lost when the peakiness regularizer took over post-warmstart.

## Diagnosis

The recipe goes through three regimes:

1. **Full warmstart (epochs 1-4, blend=1.0):** BCE-on-logits pushes heatmap toward Gaussian targets. Gets SOMETHING (mean max ~0.4) but still uniform-ish. val_loss drops 6.61 → 1.90 then plateaus at 2.28.
2. **Fade (epochs 5-7, blend 0.75 → 0.50 → 0.25):** Peakiness regularizer starts contributing. At epoch 6, raw_bp SPIKES to 1.4M — the fade transition destabilizes the model, bp/vel temporarily go haywire.
3. **Post-fade (epochs 8-10, blend=0.1):** Peakiness + weak focal floor. Model shrinks heatmap toward zero. raw_bp drops to ~3k (no peaks → DTW sees nothing), count=1.7 (sum near zero), and the heatmap returns to the sigmoid(-3)=0.047 init prior.

The peakiness regularizer doesn't have the gradient signal to maintain localized peaks — it only knows "mean max-pool should be high," and the model prefers "predict low everywhere, sum → 0, all terms small" as the local minimum.

## What this rules out

- **Recipe convergence on single run:** nope, at least not with this loss combination.
- **More epochs fixing it:** the trajectory is clear — post-warmstart decay is structural.
- **Longer warmstart:** might help BCE ground more, but the post-fade collapse is the real issue.

## What we've learned

1. **The probe loss needs supervision that survives warmstart fade.** min_blend=0.1 isn't enough to anchor shape.
2. **Peakiness regularizer is pathological** on sparse-target data when heatmap can be driven to zero (no localized peaks → nothing to max-pool → "uniform zero" satisfies it at ~1.0 which is still better than scattered mid-amplitude).
3. **Soft-DTW + velocity losses require peak amplitudes above NMS threshold** to backprop anything. Once heatmap < 0.3 everywhere, bp/vel go silent and can't push the heatmap up.
4. **With this architecture (sigmoid head), losses that operate on probabilities have saturation traps.** BCE-on-logits helped but doesn't solve the fundamental redundancy problem between loss terms.

## Paths not yet tried

- Keep focal/BCE at weight >= 0.5 post-fade (not a floor — a hard minimum large enough to matter).
- Replace peakiness with something gradient-friendly on flat heatmaps (e.g., MSE-to-warmstart throughout training, not just warmstart).
- Auxiliary loss targeting peak amplitude directly at known centers (explicit `peak_amplitude_loss(pred[centers], 1.0)`).
- Higher `probe_pos_weight` (500+ not 50) to make peak samples structurally dominant.
- Curriculum learning: stage 1 = probe localization only, stage 2 = add bp/vel once peaks are found.

## Artifacts

- `phase2a_train.log` — full log
- `phase2a_checkpoints/` — all 10 epoch checkpoints + best_model
- `phase2a_eval_best.json`, `phase2a_eval_final.json` — F1 / P / R breakdowns
- `phase2a_viz_best.png`, `phase2a_viz_final.png` — 20-molecule grids
