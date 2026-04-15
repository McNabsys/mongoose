# T2D U-Net Architecture Design -- V2 (Rearchitectured)

**Status:** Supersedes `2026-04-13-t2d-unet-design.md`. Executed on branch `v1-rearchitecture`. Original V1 preserved on `main`.

**Project Mongoose** -- Deep learning replacement for the legacy physics-based Time-to-Distance model. V2 of this design removes the dependency on wfmproc probe-level labels and makes probe detection an end-to-end learned task.

## What Changed From V1 (Original)

| Area | V1 Original | V1 Rearchitecture |
|------|-------------|-------------------|
| Input boundary | TDB + probes.bin | TDB only |
| Probe centers (temporal) | wfmproc-labeled | Learned from raw signal |
| Probe durations | wfmproc-measured | Measured from heatmap FWHM |
| Level-1 amplitude | From probes.bin | Computed from TDB backbone samples |
| L_bp matching | 1:1 via wfmproc probe_indices | Soft-DTW between detected peaks and reference probes |
| L_velocity supervision | Sparse at wfmproc centers | Dense at detected peaks, target = 511/heatmap_FWHM |
| L_probe supervision | Focal loss on wfmproc Gaussians (whole training) | Warmstart phase (3-5 epochs), then count + peakiness |
| New loss | -- | L_count (sum(heatmap) ~= N_reference_probes) |
| Architecture | Same | Same |

## Problem Statement (unchanged from V1)

The legacy T2D model (`L = C * (t + offset)^alpha`) uses a single-parameter viscous drag formula per channel to convert temporal nanochannel signals to spatial (base pair) coordinates. Empirical analysis of 31,468 clean E. coli molecules shows trailing-end velocity is 3.5x faster than mid-molecule velocity, with a head-dive acceleration burst at the leading edge. The nonlinear fluid dynamics (TVC, head dive, channel drift, concentration polarization) exceed what static parametric equations can capture.

## Architectural Scope (V1 Rearchitecture)

**In scope for V1:**
- Learn probe temporal positions from raw TDB waveform
- Learn probe durations implicitly from heatmap shape
- Learn velocity field and spatial coordinates via Softplus + cumsum
- Replace legacy T2D entirely for molecules that can be trained with reference anchoring

**Inputs the pipeline still consumes:**
- TDB file (raw waveform per molecule, rise/fall edges, structural flag)
- `.assigns` file (only to identify reference interval + direction for each remapped molecule)
- Reference genome FASTA / `_referenceMap.txt` (BssSI site positions)
- Run metadata (applied bias, pressure, timestamps for FiLM conditioning)

**Inputs the pipeline no longer consumes (as training labels):**
- probes.bin probe centers
- probes.bin probe durations
- probes.bin probe amplitudes
- probes.bin attribute flags
- probes.bin mean_lvl1 (computed from raw waveform instead)

**Wfmproc exception:** probes.bin probe centers are used ONLY during the 3-5 epoch warmstart phase to bootstrap the probe head. They are discarded from training for the remainder of the run.

**Out of scope (unchanged from V1 original):**
- Molecule detection from continuous TDB streams (TDB already does this at the block level)
- On-instrument inference
- Cross-organism generalization

## Ground Truth Construction (V2)

### Reference-Anchored Probe Positions

For each molecule that the aligner mapped to E. coli K-12 MG1655 with top-5% alignment confidence:
1. From `.assigns`: get `ref_start`, `ref_end`, and direction
2. From `_referenceMap.txt`: look up all BssSI site positions in `[ref_start, ref_end]`
3. This gives `reference_bp_positions`: the array of known probe bp coordinates for this molecule (size N_ref, ordered by direction)
4. `reference_bp_positions` is the spatial GT for this molecule

### Level-1 Estimation (From Raw TDB)

For each molecule block in the TDB:
1. Identify backbone samples: indices between `RiseConvEnd Index` and `FallConvMin Index`
2. Compute level-1 as the median voltage of these samples (robust to tag dips)
3. Subtract a small tag-dip correction if needed (trimmed mean of top 60% of values)
4. This is `mean_lvl1_from_tdb` -- used for waveform normalization

### No More wfmproc Probe Centers in GT

The old `MoleculeGT.probe_sample_indices` is dropped. Probe centers are produced dynamically at training time by the model's own probe head (via NMS on the current heatmap).

### Warmstart Labels (Temporary)

For the first 3-5 epochs only:
- wfmproc probe centers and durations are loaded separately as warmstart labels
- Used to build Gaussian heatmap targets for L_probe during warmstart
- Discarded after warmstart -- the cache can be deleted at that point

## Loss Functions (V2)

```
L_total = L_probe_warmstart(t) + λ_bp * L_bp_softDTW + λ_vel * L_velocity_dense + λ_count * L_count
```

### L_probe_warmstart(t)

Time-dependent loss that transitions during training:

**During warmstart (epochs 1-5):**
- Focal loss against Gaussian heatmap targets built from wfmproc probe centers (same as V1 original)
- Dynamic sigma = max(1.5, duration_samples / 6)
- Focal gamma=2, alpha=0.25

**After warmstart (epochs 6+):**
- L1 regularizer on (1 - max_over_local_window(heatmap)) -- encourages at least one sharp peak per window
- NO wfmproc supervision
- L_count (described below) provides the quantitative constraint

**Transition:** Linear blend over epochs 4-6 so the switch is not abrupt.

### L_bp_softDTW (Primary Spatial Signal)

For remapped molecules only:
1. Extract peak positions from current heatmap via NMS: `detected_peaks = NMS(heatmap, velocity_pred)`
2. Sample the predicted cumulative bp curve at detected peaks: `pred_bp_at_peaks = pred_cumulative[detected_peaks]`
3. Compute soft-DTW alignment between `pred_bp_at_peaks` and `reference_bp_positions`:
   - Cost function: relative-position distance (both sequences normalized to [0, 1] by their max)
   - Gamma: 0.1 (smoothing parameter)
4. Extract aligned pairs: `(pred_bp_i, ref_bp_i)` where `ref_bp_i` is the reference probe the DTW matched to detected peak `i`
5. Compute inter-pair deltas on both sides: `pred_deltas = diff(pred_bp_at_aligned)`, `gt_deltas = abs(diff(ref_bp_at_aligned))`
6. Huber loss (delta=500bp) on deltas, divided by mean GT delta for normalization

Soft-DTW handles missing peaks (FN) and extra peaks (FP) natively via its soft minimum. No hard matching required.

### L_velocity_dense

At each detected peak in the current heatmap:
1. NMS gives integer peak indices
2. Measure FWHM of each peak: find samples where `heatmap > 0.5 * peak_value`, count them, convert to ms via sample rate
3. Compute target velocity: `target_v = 511 bp / FWHM_ms`
4. `target_v` is converted to bp/sample using `SAMPLE_PERIOD_MS`
5. L2 loss between `pred_velocity[peak_index]` and `target_v`
6. **STOP-GRADIENT on target_v** -- the target is derived from the heatmap but the gradient does not flow back through the duration measurement

This is "dense" in the sense that every detected peak contributes, unlike the original sparse version which only evaluated at wfmproc-labeled positions.

### L_count (Bootstrap Mechanism)

For remapped molecules only:
- Target: `N_ref = len(reference_bp_positions)` -- the number of reference probes in the mapped interval
- Prediction: `sum(heatmap * mask)` -- continuous count proxy
- Loss: smooth L1 between prediction and target, scaled by `1/N_ref` for normalization

This loss gives the heatmap a direct incentive to produce the right number of peaks. Without it, L_probe post-warmstart has no quantitative signal and can drift to zero or maximum.

## Loss Weighting Schedule

Epoch 1-3 (warmstart full): lambda_bp=0.5, lambda_vel=0.5, lambda_count=0.5, focal loss active
Epoch 4-5 (warmstart fade): lambda_bp=1.0, lambda_vel=1.0, lambda_count=1.0, focal loss decaying linearly
Epoch 6+ (post-warmstart): lambda_bp=1.0, lambda_vel=1.0, lambda_count=1.0, focal loss off, peakiness regularizer on

## Architecture (Unchanged from V1 Original)

All model structure unchanged:
- 5-level encoder (channels 32 -> 64 -> 128 -> 256 -> 512, kernel 7)
- Hybrid bottleneck (dilated cascade [1,2,4,8] + MHSA)
- 5-level symmetric decoder with skip connections to full resolution
- Bifurcated heads: probe head (kernel 7), velocity head (kernel 31)
- FiLM conditioning: 6-value vector at encoder L0 and bottleneck
- Softplus on velocity, cumsum for integration (parameter-free monotonic spatial output)

See `2026-04-13-t2d-unet-design.md` for the full architecture details. Only the training signal has changed.

## Data Pipeline Changes

### Preprocessing

Rewrite `preprocess.py` to:
1. Compute `mean_lvl1_from_tdb` from backbone samples (median estimator)
2. Cache reference-derived `reference_bp_positions` (from aligner + FASTA lookup)
3. Cache wfmproc probe centers SEPARATELY as warmstart-only labels (in a distinct key)
4. Drop caching of wfmproc probe durations, amplitudes, attributes

### Dataset

Rewrite `CachedMoleculeDataset.__getitem__` to return:
- `waveform`: level-1 normalized using `mean_lvl1_from_tdb`
- `reference_bp_positions`: the reference-derived probe bp coordinates
- `warmstart_heatmap`: Gaussian target built from wfmproc labels (only loaded if warmstart active)
- `warmstart_valid`: bool indicating whether warmstart labels exist for this molecule
- `conditioning`: 6-value FiLM vector (unchanged)
- `n_ref_probes`: count for L_count loss
- `mask`: valid-sample mask

Does NOT return wfmproc probe_sample_indices or gt_deltas in the old form.

## Success Criteria (V1 Rearchitecture)

Primary:
1. **Probe recall on remapped held-out die D08:** >= 95% of reference BssSI sites are detected by the model within +/- 50 bp tolerance
2. **Inter-probe MAE on D08:** Beats legacy T2D by at least 5-10% with tight bootstrap confidence interval
3. **Precision on detected peaks:** Low FP rate (< 10% of detected peaks have no reference match within tolerance)

Secondary:
4. Lambda-vel ablation shows contribution of velocity physics constraint
5. Warmstart ablation (skip warmstart entirely) shows whether bootstrap is necessary
6. Rescued-molecule yield on previously-unmappable molecules > 5%

## Training Risks and Mitigations

1. **Heatmap collapse during post-warmstart transition.** Mitigation: peakiness regularizer, L_count constraint, gradual transition over epochs 4-6.

2. **Soft-DTW failure on very short or very long molecules.** Mitigation: skip L_bp for molecules with < 4 detected peaks or > 100 detected peaks. Add a fallback to Hungarian matching if soft-DTW gradient is unstable (we'll monitor early training).

3. **Count loss underconstraints peak position.** Mitigation: soft-DTW cost function includes a position prior (peaks should be roughly evenly distributed given expected mean velocity). If peaks cluster in wrong region, L_velocity at those peaks will show high loss (measured widths won't match required velocities).

4. **Loss balance tuning.** Warmstart with lower lambda values (0.5) to let probe head stabilize before spatial and velocity losses pull too hard. Post-warmstart lambdas at 1.0.

## Fallback Plan

If V1 rearchitecture shows instability after 2 training iterations:
- Extend warmstart phase from 5 to 15 epochs
- Keep wfmproc labels active throughout (full V1 original behavior)
- Accept the wfmproc dependency and ship V1 original as a partial result
- V1 rearchitecture work becomes V1.5

The original V1 design on `main` branch is the fallback. No work is lost.
