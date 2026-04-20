# Peer-review request: multi-task peak-detection U-Net stuck at ~0% recall despite numerically stable training

## DRAFT — do not send yet; review first

---

## 1. Problem

We are training a 1-D U-Net to detect probe-binding events in nanopore
translocation waveforms. Each molecule is a variable-length 1-D signal (raw
ionic current; T ≈ 10k–30k samples) passing through a nanopore. At known
reference positions along the DNA molecule, chemical probes have been
attached; when the probe passes through the pore it briefly perturbs the
current. The training target is the per-sample location of those probe
events.

Ground-truth labels are built from a separate reference-alignment pipeline
(`wfmproc`) that produces, per molecule:
- A list of reference probe positions in base pairs (typically ~14 probes
  per molecule; 771,855 total reference probes across 55,038 eval molecules).
- Per-probe estimated duration in samples (used for target Gaussian σ).

A single training waveform is ~10k–30k samples long and contains ~5–40
peaks, so the target is **very sparse** (~0.5–1.0% of samples are inside a
probe Gaussian).

---

## 2. Architecture

Standard 5-level 1-D U-Net (see code below, abbreviated):

```
Encoder:  5 levels, channels [32, 64, 128, 256, 512], kernel=7,
          2× ResBlock per level, MaxPool1d(2) between levels
Bottleneck: 4× dilated ResBlocks (d=1,2,4,8) + 4-head self-attention + LayerNorm
Decoder:  mirror of encoder with skip concatenation
FiLM conditioning: 6-dim physical observables injected at level 0 and bottleneck

Heads (shared decoder features, [B, 32, T]):
  probe_head:    2×ResBlock + Conv1d(32→1, 1)  →  probe_logits, sigmoid → probe ∈ [0,1]
  velocity_head: 2×ResBlock(kernel=31) + Conv1d → Softplus → raw_velocity > 0
  cumulative_bp: torch.cumsum(raw_velocity * mask) along time
```

Probe head final-conv bias is initialized to **-3.0** so sigmoid(-3) ≈ 0.047 ≈
the average target-peak-sample fraction (standard sparse-detection init, à la
RetinaNet focal-loss paper).

### 2a. Target heatmap construction

```python
def build_probe_heatmap(length, probe_centers_samples, probe_durations_samples):
    heatmap = np.zeros(length, dtype=np.float32)
    for center, duration in zip(probe_centers_samples, probe_durations_samples):
        sigma = max(1.5, float(duration) / 6.0)
        lo = max(0, int(center - 4 * sigma))
        hi = min(length, int(center + 4 * sigma) + 1)
        if lo >= hi:
            continue
        x = np.arange(lo, hi, dtype=np.float32)
        gaussian = np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)
        heatmap[lo:hi] = np.maximum(heatmap[lo:hi], gaussian)  # max-combine, not sum
    return heatmap
```

Typical probe duration is ~10–30 samples → σ ≈ 2–5 samples → per-peak support
~8–40 samples out of T=10k–30k. Peak targets are exactly 1.0 at their
centers; between-peak regions are exactly 0.0.

---

## 3. Loss

Four components. Both the blend between probe-supervision modes and the
λ-scale are epoch-scheduled.

**L_probe** — two modes blended by `w ∈ [min_blend, 1]`:
1. **Warmstart (w > 0)**: positive-weighted BCE-with-logits vs the Gaussian
   heatmap target from §2a. Per-molecule, with mask `mask_b` flagging valid
   (non-padded) samples:
   ```python
   mask_f_b   = mask_b.to(pred_h_b.dtype)
   target_b   = warmstart_heatmap[b].to(pred_h_b.dtype)
   weight_b   = 1.0 + self.probe_pos_weight * target_b        # pos_weight = 50
   logits_b   = pred_heatmap_logits[b]

   bce_elem   = F.binary_cross_entropy_with_logits(
                   logits_b, target_b, reduction="none")      # [T]
   weighted_err = bce_elem * weight_b * mask_f_b

   denom      = (weight_b * mask_f_b).sum().clamp(min=1.0)
   probe_loss = weighted_err.sum() / denom                    # scalar

   L_probe_bce = w * probe_loss
   ```
   Then averaged across the batch (per-molecule mean). Note: because
   `target_b` is near-zero almost everywhere, `weight_b ≈ 1` almost
   everywhere, and `denom ≈ T + pos_weight * sum(target)`, dominated by the
   first term when T ≫ n_positives (as here).

2. **Self-supervised (w < 1)**: peakiness regularizer
   ```python
   def peakiness_regularizer(heatmap: torch.Tensor, window: int = 20):
       half = window // 2
       padded = F.pad(
           heatmap.unsqueeze(0).unsqueeze(0),
           (half, window - half - 1), mode="replicate")
       max_pooled = F.max_pool1d(padded, kernel_size=window, stride=1)
       max_pooled = max_pooled.squeeze(0).squeeze(0)[: heatmap.shape[0]]
       return (1.0 - max_pooled).mean()
   ```

   `L_probe = L_probe_bce + (1 - w) * peakiness`

**L_bp** — soft-DTW between **detected** peak bp-positions and reference
bp-positions, zero-anchored and span-normalized. The detection-and-gate
structure per molecule is:

```python
peak_indices = extract_peak_indices(
    pred_h_b, raw_v_b,
    threshold=self.nms_threshold,        # 0.3
    tag_width_bp=self.tag_width_bp,
)

if peak_indices.numel() >= 2 and ref_bp.numel() >= 2:
    pred_bp_at_peaks = pred_bp_b[peak_indices]
    pred_norm = pred_bp_at_peaks - pred_bp_at_peaks[0]
    ref_norm  = (ref_bp_f - ref_bp_f[0]).abs()
    span      = (ref_bp_f[-1] - ref_bp_f[0]).abs().clamp(min=1.0)
    dtw       = soft_dtw(pred_norm, ref_norm, gamma=self.softdtw_gamma)
    bp_terms.append(dtw / span)

    # L_vel is computed in the same branch — same gate
    ...
# else: molecule contributes 0 to both L_bp and L_vel
```

`extract_peak_indices` is a forward-pass peak finder (threshold at 0.3 +
width-based NMS from `tag_width_bp / raw_velocity`). Because
`pred_heatmap = sigmoid(logits)` and logits are initialized around −3, at
init `pred_h_b ≈ 0.047` everywhere, **no molecule produces ≥ 2 peaks, and
both bp and vel losses contribute exactly zero gradient for the entire
early-training phase.**

**L_vel** — MSE at detected peak positions against velocity targets derived
from peak FWHM and known tag width. Same no-detection gate as L_bp.

**L_count** — **Always on. No detection gate.**

```python
def count_loss(heatmap, target_count, mask=None):
    if mask is not None:
        heatmap = heatmap * mask.to(heatmap.dtype)
    predicted  = heatmap.sum()
    target     = torch.tensor(float(target_count),
                              device=heatmap.device, dtype=heatmap.dtype)
    raw_loss   = F.smooth_l1_loss(predicted, target, reduction="mean")
    denominator = max(float(target_count), 1.0)
    return raw_loss / denominator
```

At init: `predicted ≈ 0.047 × T ≈ 0.047 × 15000 ≈ 700`, `target ≈ 14`,
`error ≈ 686`, smooth-L1 in linear regime → `raw_loss ≈ 686 − 0.5 ≈ 685`,
normalized → ~49. Divided by `scale_count = 50` → ~1, but the gradient
w.r.t. each heatmap sample is `+1 / 14 / 50 ≈ +0.00143` (scaled), uniform
across all T samples, pulling every sample's probability down.

**Total:**
```
scaled_probe = L_probe / scale_probe          (scale_probe = 1)
scaled_bp    = L_bp    / scale_bp             (scale_bp    = 300000)
scaled_vel   = L_vel   / scale_vel            (scale_vel   = 5000)
scaled_count = L_count / scale_count          (scale_count = 50)

L_total = scaled_probe
        + current_λ_bp    * scaled_bp
        + current_λ_vel   * scaled_vel
        + current_λ_count * scaled_count
```

`current_λ` schedule: 0.5× target during warmstart (8 epochs) ramping to 1.0×
after. `w` schedule: full for 4 epochs, linearly fading 1 → 0 over 4 epochs,
then clamped to min_blend = 0.1:

```python
def set_epoch(self, epoch: int) -> None:
    full_epochs = max(self.warmstart_epochs - self.warmstart_fade_epochs, 0)
    if   self.warmstart_epochs <= 0:   w = 0.0
    elif epoch < full_epochs:          w = 1.0
    elif epoch < self.warmstart_epochs:
        frac = (epoch - full_epochs + 1) / max(self.warmstart_fade_epochs, 1)
        w    = max(0.0, 1.0 - frac)
    else:                              w = 0.0
    self._warmstart_blend = max(w, self.min_blend)

    # Lambda scale: 0.5× during warmstart, ramping to 1.0× after.
    if   self.warmstart_epochs <= 0:        scale = 1.0
    elif epoch < self.warmstart_epochs:     scale = 0.5 + 0.5 * (epoch / self.warmstart_epochs)
    else:                                   scale = 1.0

    self.current_lambda_bp    = self.lambda_bp    * scale
    self.current_lambda_vel   = self.lambda_vel   * scale
    self.current_lambda_count = self.lambda_count * scale
```

---

## 4. Recipe evolution (what we tried and why we moved on)

| Attempt | Outcome |
|---|---|
| Focal loss on probe head (γ=2, α=0.25), default init | Heatmap collapsed to ~flat 0. Focal's α=0.25 down-weights positives and mean-reduction over padded length dilutes the sparse signal. |
| min_blend=0.1 keeping focal alive post-warmstart | Still collapsed. Focal gradient too weak vs bp/count. |
| Push λ_count to 1e-6 to "neutralize" the count pull-down | Heatmap went uniform ~0.4 everywhere. No localization — count-pressure-free equilibrium. |
| Replace focal with dense MSE vs Gaussian target | Sigmoid saturated to ~0. At logits ≈ −6, sigmoid′ ≈ 0.0025 → vanishing gradient. |
| Positive-weighted MSE vs Gaussian (pos_weight on target) | Still saturated — same sigmoid-gradient collapse at init. |
| **BCE-with-logits + pos_weight = 50 + init-bias = −3** (current) | **Numerically stable, no NaN, no collapse. But probe loss remains ~flat across all 10 training epochs (see §5).** |

All experiments used the same architecture, same optimizer (AdamW), same
data, same augmentation, same schedule. Only the probe loss formulation
changed.

---

## 5. Phase 2a results — 10 epochs, batch 32, single 11-run cache

Command: `train.py --epochs 10 --batch-size 32 --lr 3e-4
--warmstart-epochs 8 --warmstart-fade-epochs 4 --min-blend 0.1
--scale-bp 300000 --scale-vel 5000 --scale-count 50 --scale-probe 1.0
--probe-pos-weight 50 --min-lr 1.5e-5`

Per-epoch (scaled components reported; `raw` = before /scale):

```
Ep | train | val  | blend | raw_probe | raw_bp   | raw_vel | raw_count | lr
 1 |  4.79 | 6.61 | 1.000 |    1.12   |  318367  |  27473  |   39.48   | 2.93e-4
 2 |  3.48 | 2.52 | 1.000 |    1.10   |  149509  |  14819  |   38.36   | 2.73e-4
 3 |  3.24 | 1.90 | 1.000 |    1.14   |  105426  |  11525  |   35.19   | 2.41e-4   ← best val
 4 |  3.39 | 2.28 | 1.000 |    1.17   |  192446  |   9689  |   32.46   | 2.02e-4
 5 |  3.43 | 4.14 | 0.750 |    1.23   |  114325  |  10404  |   23.34   | 1.57e-4
 6 |  7.89 | 6.17 | 0.500 |    1.25   | 1423421  |  15619  |   15.06   | 1.13e-4   ← raw_bp blow-up
 7 |  3.03 | 3.52 | 0.250 |    1.21   |   96361  |   8118  |    6.74   | 7.4e-5
 8 |  1.37 | 2.25 | 0.100 |    1.13   |    5821  |    982  |    2.11   | 4.2e-5
 9 |  1.20 | 4.81 | 0.100 |    1.14   |     615  |    124  |    1.71   | 2.2e-5
10 |  1.20 | 2.62 | 0.100 |    1.14   |    3457  |     87  |    1.68   | 1.5e-5
```

**Observations:**

- `raw_probe` is essentially flat at 1.10–1.25 across all 10 epochs, even
  during epochs 1–4 when `blend=1.0` and BCE is the dominant probe-head
  teaching signal.
- `raw_bp` and `raw_vel` and `raw_count` all drop by orders of magnitude
  (318K → 3.5K, 27K → 87, 39 → 1.7).
- "best_model.pt" (val_loss minimum) selected at epoch 3, but that selection
  is driven by the other three components; the probe head never learned.
- No NaN, no divergence, no collapse. Just: the probe head did not move.

**Evaluation of best_model.pt on the same 11-run cache (validation-set
molecules), peak-match F1 @ tolerance=50 samples:**

```
Overall (sum-of-counts):  TP=13,375  FP=22,043  FN=771,855
                          P=0.378   R=0.017   F1=0.033
Per-molecule mean:        P=0.227   R=0.021   F1=0.037
                          n_molecules=55,038
```

**The model predicts ~0 peaks. Recall is 1.7%.** When it does emit anything
above threshold, precision is okay (38%), but it emits almost nothing —
consistent with the heatmap being stuck at ~sigmoid(−3) ≈ 0.047, below the
0.3 detection threshold.

Earlier overfit-one-batch gate (300 steps on a single batch) also failed to
produce sparse peak localization, though `train_loss` dropped as expected.

---

## 6. What we think is going on (our best guesses — please push back)

**Hypothesis A: L_count fights L_probe.** Count loss applies a uniform
downward gradient of roughly `sign(predicted_sum - target) / target_count`
on every heatmap sample, regardless of whether it's at a peak or not. At
init, `predicted_sum ≈ 0.047 * 30000 ≈ 1400` while `target ≈ 14`, so the
count-loss gradient at every sample is `+1 / 14 / 50 ≈ +0.0014` (scaled)
pulling every sample down. BCE at peak samples is larger (≈0.015 per peak
sample at init), but is concentrated on ~1% of samples while count-pressure
is everywhere.

**Hypothesis B: L_bp and L_vel are gated off until peaks exist, but peaks
can't form because L_probe alone isn't strong enough.** A chicken-and-egg
failure: threshold-based detection in the forward pass means bp/vel
supervision waits for the probe head to already work.

**Hypothesis C: Probe-loss normalization dilutes the positive signal.**
The BCE denominator is `sum(weight * mask)` which is dominated by the 99%
of samples where target ≈ 0 and weight = 1. So even with pos_weight=50,
the effective per-sample loss magnitude is close to the unweighted mean.
We may want to switch to a "sum-then-divide-by-n_positives" formulation
(mean over positives only), or a `reduction='sum'` form, or something
structurally different.

**Hypothesis D: Per-molecule averaging over variable-length sequences.**
Each molecule's probe loss is computed independently then averaged across
the batch. Long molecules have proportionally more negatives, which shifts
their mean further toward the negative-class BCE value. We don't currently
re-weight by sequence length or by positive-count.

**Hypothesis E: The whole multi-task setup is too coupled for the early
epochs.** Maybe the right move is to train probe-only (drop L_bp, L_vel,
L_count) until the heatmap has peaks, then turn the others on.

---

## 7. Questions for you

1. **Is hypothesis A correct?** Does uniform count pressure realistically
   suppress sparse-peak emergence at init, given the numbers above? If so,
   what's the standard trick — delay L_count until after probe-onset is
   detected, or reformulate count as a soft-penalty-only-above-target, or
   structurally something else?

2. **BCE normalization for sparse dense-regression:** Is the
   weighted-mean-with-weight-sum denominator structurally wrong for this
   problem? What formulations work in practice for dense sparse-target
   tasks in TTS/speech (e.g., MFA-style forced alignment targets)?

3. **Detection-gated auxiliary losses:** bp/vel losses give zero gradient
   until peaks appear. Is this a well-known failure mode, and is the right
   answer (a) remove the detection gate and use a soft differentiable
   surrogate, (b) bootstrap-stage training (probe-only first), or (c)
   something else?

4. **Schedule:** The probe-BCE blend linearly fades to self-supervised
   peakiness regularizer over 4 epochs. Given probe_raw didn't converge
   during the BCE-dominant phase, is fading at all the right move? Should
   warmstart be maintained until a recall threshold is hit?

5. **Is anything fundamentally mis-designed at the architecture level?**
   Single-head sigmoid heatmap on a shared-feature U-Net for sparse
   1-D-sequence peak detection — is there a better-documented family of
   approaches we're missing (e.g., encoder-decoder with explicit per-peak
   token prediction, set-prediction, CTC, keypoint-style heatmap +
   sub-sample offset regression)?

---

## 8. What would most help

A ranked list of likely causes with concrete proposed fixes, especially for
(1) the count-loss interaction and (2) the BCE normalization, is the most
valuable feedback. Second-most valuable: any literature pointers for
sparse-peak dense-regression from related fields (nanopore basecalling
alignment, speech forced alignment, ECG beat detection, time-series event
detection). We have prior art in mind for focal loss, CenterNet-style
keypoint heatmaps, and DETR-style set prediction, but if you know a
well-characterized approach for this specific setup — sparse 1-D peak
detection with auxiliary physical-consistency losses — please name it.

---

*Architecture code, loss code, and full training log available on
request. Prior attempts summarized in §4 represent roughly two weeks of
iteration.*
