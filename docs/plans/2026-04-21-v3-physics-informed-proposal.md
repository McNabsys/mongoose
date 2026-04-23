# V3: Physics-Informed Probe-Interval Model — Proposal for Peer Review

**Author:** Claude (with project lead Jon McBee, Nabsys)
**Date:** 2026-04-21
**Status:** Draft for Deep Think peer review — not yet approved for implementation
**Supersedes:** V2 design (2026-04-20-v2-design.md) — proposing V2 be killed, see §1

---

## 1. Why V2 is being killed

V1 and V2 are both supervised against remapping-derived ground truth (`.assigns` files from the aligner + wfmproc's `probes.bin`). The network learns to mimic the remapping pipeline. This has a hard ceiling: **the model can never be more accurate than its teacher**, and it cannot generalize beyond the regime where remapping itself works.

That is the wrong objective.

The actual end goal is:

> **Input:** a TDB file (raw 32 kHz current waveforms).
> **Output:** for each molecule, a list of inter-probe distances in basepairs.
> **Downstream:** an assembler consumes these interval lists and reconstructs the target genome. After assembly, structural-variant (SV) calling.

No remapping in the inference pipeline. Ideally, no remapping in the training pipeline either. The model should learn the physics of nanopore translocation directly from the current signal, anchored by known physical constants.

**V1 has value as a remapping emulator and as an oracle for validating V3 predictions.** Finish the V1 cloud run; then stop that line of development.

V2's additional architectural firepower (Transformer U-Net, Sinkhorn-OT position loss, segmentation head) is irrelevant to the actual goal. It is a better mimic of remapping. Kill it.

---

## 2. The physical anchor that makes V3 possible

Every probe in the Nabsys system is designed to be **exactly 511 bp wide**. This is a hard, controlled, unchanging physical constant. It gives us the following identity for the per-sample translocation velocity $v(t)$:

$$\int_{t_i^{\text{enter}}}^{t_i^{\text{exit}}} v(t)\, dt = 511 \text{ bp} \quad \forall \text{ probe } i$$

Applied across thousands of probes in thousands of molecules, this is dense self-supervision on the velocity field. No aligner output required.

Inter-probe distance then follows trivially:

$$d(i, i+1) = \int_{t_i^{\text{exit}}}^{t_{i+1}^{\text{enter}}} v(t)\, dt$$

The entire V3 design is organized around making this identity trainable and the integrals numerically well-behaved.

---

## 3. Architecture

### 3.1 Overall dataflow

```
raw current I(t) at 32 kHz
        │
        ▼
  normalization (per-molecule z-score over valid mask — unchanged from V1)
        │
        ▼
  shared 1-D CNN/Transformer backbone (reuse V1's U-Net trunk or V2's)
        │
        ├─► probe-event head  ──► P(probe | t), per-sample logit
        │
        └─► velocity head     ──► v̂(t), per-sample bp/s, constrained v̂ ∈ [v_min, v_max]

  post-processing (non-learned):
    1. extract discrete probe events {(t_i^enter, t_i^exit)} from the probe heatmap
    2. cumulative bp: B̂(t) = ∫₀ᵗ v̂ dτ
    3. per-probe transit: Δᵢ = B̂(t_i^exit) − B̂(t_i^enter)
    4. inter-probe interval: dᵢ = B̂(t_{i+1}^enter) − B̂(t_i^exit)
```

The backbone can be inherited wholesale from V1 (proven stable) or upgraded to V2's 8x-Res Transformer U-Net if capacity proves limiting. **My recommendation: inherit V1's backbone. Architectural complexity is not the bottleneck — the supervision regime is.**

### 3.2 Velocity-head parameterization

Output $v̂(t) = v_{\min} + (v_{\max} - v_{\min}) \cdot \sigma(\text{logit}(t))$

where $v_{\min}$ and $v_{\max}$ are known physical bounds (from prior characterization of the system). This guarantees positivity and physical plausibility without soft penalties. The sigmoid is per-sample.

### 3.3 Probe-event head

Same CenterNet-style peak heatmap as V1: per-sample logit, focal BCE loss in pre-training, hysteresis-threshold decode at inference to produce probe events as intervals, not just centers. Width of the event is needed for the 511-bp integral limits.

---

## 4. Losses

### 4.1 The core loss: L_511

For each detected or labeled probe event $i$ with interval $[t_i^{\text{enter}}, t_i^{\text{exit}}]$:

$$\mathcal{L}_{511} = \frac{1}{N_{\text{probes}}} \sum_i \left( \int_{t_i^{\text{enter}}}^{t_i^{\text{exit}}} v̂(t)\, dt - 511 \right)^2$$

**Implementation note:** the integral is a sum over samples weighted by $\Delta t = 1/32000$ s. The gradient flows through every sample inside the probe interval — hundreds of samples per probe at typical translocation speeds. This is a dense gradient.

### 4.2 Smoothness prior: L_smooth

$$\mathcal{L}_{\text{smooth}} = \frac{1}{T} \sum_t \left( v̂(t+1) - v̂(t) \right)^2$$

Velocity should vary continuously. Without this, the model can place arbitrary spikes between probes to satisfy L_511 trivially (e.g., zero velocity everywhere except sharp Dirac spikes during probes).

### 4.3 Probe-event supervision: L_probe

**Bootstrap phase:** use wfmproc's `probes.bin` as weak labels. This is signal-processing output (current-shape matching), not remapping-derived. Arguably within the "no remapping" constraint.

**Full V3 phase:** replace with self-supervised probe detection:
- Contrastive loss on the stereotyped probe-spike template
- Or: probe detector distilled from a wfmproc-pretrained initialization, then frozen; V3 re-learns it only via backprop through L_511

### 4.4 Optional: L_length (when applicable)

If the molecule's total length is known a priori (e.g., for a controlled E. coli run):

$$\mathcal{L}_{\text{length}} = \left( \int_0^{T_{\text{mol}}} v̂(t)\, dt - L_{\text{known}} \right)^2$$

This is a single global constraint per molecule. Weak but useful for validation-on-the-fly.

### 4.5 Optional: L_current_velocity coupling

Nanopore physics predicts that deeper current blockade corresponds to slower translocation (probe resistance). A soft monotonicity prior:

$$\mathcal{L}_{\text{phys}} = \sum_t \max(0, v̂(t) - v̂(t') )^2 \quad \text{whenever } I(t) < I(t') \text{ within probe interior}$$

Applied only within probe events, where the coupling is strongest. **I am least confident about this loss.** It may hurt more than it helps if the physics assumption is wrong in edge cases. Disable by default; enable in ablation.

### 4.6 Total loss

$$\mathcal{L} = \lambda_{511} \mathcal{L}_{511} + \lambda_s \mathcal{L}_{\text{smooth}} + \lambda_p \mathcal{L}_{\text{probe}} + \lambda_L \mathcal{L}_{\text{length}}$$

with $\lambda_{511}$ dominant (the whole program lives or dies on L_511). Recommended starting weights:
- $\lambda_{511} = 1.0$
- $\lambda_s = 0.01$ (tune so smooth-term magnitude ≈ 1% of L_511 at init)
- $\lambda_p = 0.1$ during bootstrap; 0.0 once self-supervised
- $\lambda_L = 0.1$ when available, 0 otherwise

---

## 5. Training regime

### 5.1 Bootstrap stage (V1→V3 transfer)

Initialize from V1's best checkpoint. V1 already learned a useful probe detector and a velocity head that is approximately right in the regions where remapping was. Freeze the backbone for 1–2 epochs, train only the V3 losses on the existing caches. This tests whether the 511-bp constraint is consistent with V1's learned representation.

### 5.2 Full-training stage

Unfreeze backbone. Train with L_511 + L_smooth + L_probe (weak) for ~30 epochs. No `.assigns`-derived labels enter any loss. Validation metric is interval-level MAE against remapping, but remapping is **never** in the loss.

### 5.3 Self-supervised probe detection (stretch goal)

Replace weak wfmproc probe labels with self-supervised detection:
- Masked-reconstruction on current waveform → representation that discriminates probe from baseline
- K-means / GMM on the representation → two clusters, one of which is the probe class
- Fine-tune detection head to predict the self-supervised cluster

Only tackle this after the supervised-bootstrap V3 is working.

---

## 6. Pilot study (before full commit)

Before investing in a full V3 build, run a 1–2 week feasibility pilot. The question the pilot answers: **"can a velocity field trained only with the 511-bp integral constraint plus smoothness recover inter-probe distances accurately enough for assembly?"**

### 6.1 Pilot design

1. **Inputs:** existing 30 caches, but strip all `.assigns`-derived labels except for *validation*. Probe positions (event intervals, not bp coordinates) come from wfmproc's `probes.bin` directly — this is signal-processing output, not remapping.
2. **Model:** tiny. V1 backbone, probe head frozen to wfmproc, velocity head trainable.
3. **Loss:** L_511 + L_smooth only.
4. **Training:** ~5 epochs on 4 caches. Fast iteration.
5. **Evaluation:** held-out cache. For each predicted inter-probe interval $d̂_i$, compare to the remapping-derived $d_i$. Report:
   - Median absolute relative error: median $|d̂_i - d_i| / d_i$
   - Distribution of errors (so we see tails, not just central tendency)
   - Correlation between predicted and remapped intervals

### 6.2 Pilot success criteria

| Outcome | Median rel. err. | Decision |
|---------|-----------------:|----------|
| Excellent | < 2% | commit to V3, skip architectural upgrades |
| Good | 2–5% | commit to V3, consider adding L_length and L_phys |
| Marginal | 5–10% | V3 concept viable but needs richer priors or better backbone — investigate before committing |
| Poor | > 10% | 511-bp constraint alone is insufficient. Need fundamentally more signal (extra inputs, molecule-length constraint, multi-molecule coupling). Redesign. |

### 6.3 Why this matters

If the pilot fails, we learn cheaply that velocity is genuinely underdetermined by current alone — a fundamental physics result. That would change the program: we'd need extra sensing (dual-channel signal, voltage modulation, etc.) or accept that the model produces *relative* intervals only (sufficient for assembly after normalization, but no absolute bp).

If the pilot succeeds, V3 is derisked and the full implementation is months of engineering, not research.

---

## 7. Risks

### 7.1 Velocity underdetermined in inter-probe gaps

The 511-bp constraint pins $v̂$ *during* probes. Between probes, only smoothness constrains it. If current features don't encode local translocation rate, the inter-probe integral is a free parameter and intervals are recoverable only up to a per-molecule scaling. **Mitigation:** L_length when available; multi-molecule consistency (v̂ as a function of current features should be universal across molecules of the same chemistry).

### 7.2 Probe detection errors propagate

A false-positive probe contributes a spurious 511-bp constraint that distorts the velocity field. A false negative removes a constraint. **Mitigation:** hysteresis-threshold decode with conservative thresholds; downweight L_511 contributions by detector confidence.

### 7.3 Probe width variability

511 bp is the *design*. Actual probe transit may have some bp-width jitter due to conformational effects. If that jitter is ±1 bp, fine. If it's ±10 bp, the constraint becomes noisy. **Open empirical question for pilot.** If needed, relax to a Gaussian: $\int v̂ dt \sim \mathcal{N}(511, \sigma^2)$ and marginalize.

### 7.4 Physics-informed training instability

Self-supervised losses like L_511 have trivial global minima (e.g., set $v̂ = 0$ everywhere and rely on probe-detection failures). **Mitigation:** smoothness prior; velocity lower bound via sigmoid parameterization; curriculum from V1 weights rather than random init.

### 7.5 The 511-bp constant assumes clean probe boundaries

"Probe enter" and "probe exit" are only well-defined if the current signature cleanly separates probe from baseline. Overlap regions, translocation pauses, or probe-probe proximity could violate this. **Mitigation:** pilot probes in isolation first; then tackle edge cases.

### 7.6 Probe detection dependency

If we have to fall back to wfmproc for probe events in production, we haven't fully escaped the classical pipeline. **Acceptable short-term**, but the self-supervised stretch goal (§5.3) exists precisely to close this gap.

---

## 8. What this proposal does NOT address

- **Strand orientation / probe identity.** V3 produces intervals but doesn't distinguish probe-A from probe-B. The Nabsys chemistry does (probes have type-specific signatures). Integrating probe-typing is a V3.1 concern.
- **Molecules with very few probes (N ≤ 2).** L_511 provides one constraint per probe; sparse molecules have weak supervision signal. These may need to be excluded or handled with L_length.
- **Unknown total molecule length.** Current design assumes L_known is available when used. For unknown samples, this loss is disabled and the model operates on relative intervals only.
- **Assembly/SV-calling integration.** Downstream — out of scope for this proposal. But the output format (interval list per molecule) is chosen to match what the assembler consumes.
- **End-to-end with probe typing.** If we eventually need probe identity, the probe head grows extra classes. Not in scope here.
- **Training-data requirements.** This proposal assumes the existing 30 E. coli caches are sufficient for the pilot. A full V3 production model may need diverse targets (not just E. coli) to learn a universal velocity–current mapping. Out of scope for the pilot; a question for post-pilot planning.

---

## 9. Open questions for Deep Think

1. **Is L_511 + L_smooth well-posed?** Is there a provably unique velocity field that satisfies the integral constraint on probes and the smoothness prior between them, given realistic current data? Or is it fundamentally degenerate (e.g., up to scaling)?

2. **What additional physics priors should be in the initial V3 loss set?** I proposed L_current_velocity (monotone in current) but flagged low confidence. Are there stronger/safer physical couplings I'm missing — ionic current theory, drift-diffusion, polymer physics?

3. **Is the wfmproc probe-detection dependency a real violation of the "no classical pipeline" spirit, or acceptable bootstrap?** The proposal treats it as acceptable because wfmproc probe detection is pure signal processing, not remapping. But is there an argument it contaminates the learning signal?

4. **Pilot design critique.** Is the 5-cache, 5-epoch pilot enough signal to distinguish "V3 works" from "V3 marginally works"? Or do we need a larger, more systematic ablation?

5. **Bootstrap vs. from-scratch.** Is initializing V3 from V1's weights a good idea (warm start, proven backbone) or a bad idea (contamination from remapping-supervised features)? The latter is a real concern — V1's velocity head is already biased toward remapping's answer.

6. **Cross-molecule consistency.** The proposal assumes $v̂$ is a universal function of local current features across molecules. If inter-molecule variance is large (pore aging, contamination, temperature drift), we need conditioning. How should we detect whether this matters before paying for it?

7. **Alternative architectures.** Is there a physically-motivated backbone that's better than generic U-Net/Transformer for this — e.g., a neural-ODE style velocity integrator, or an explicit Kalman-filter-like smoother over probe events? I defaulted to keeping V1's backbone to minimize architectural risk. Is that the wrong default?

8. **Assembly-downstream consequences.** If V3 intervals have, say, 3% median error but heavy tails at 20%, is that worse for assembly than V1's ~1% median with thinner tails? Assembly tolerances for interval error are something I don't have a great model of. Is interval MAE even the right pilot metric?

---

## 10. Decision requested

Deep Think is asked to:
1. Validate or challenge the physics reasoning in §2.
2. Critique the loss design in §4, particularly the completeness of the constraint set.
3. Evaluate whether the pilot study in §6 is sufficient to go/no-go the full V3 build.
4. Flag any fundamental objection I haven't identified.
5. Recommend any targeted literature — particularly on physics-informed neural networks applied to translocation signals, or on integral-constraint-based self-supervision.

Output I want back: a written review in the format used for prior peer-review rounds, with explicit recommendations at the end (proceed as-proposed / proceed with modifications / do not proceed / need more evidence).

---

*End of proposal.*
