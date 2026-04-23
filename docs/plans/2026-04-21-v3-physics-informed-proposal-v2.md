# V3: Physics-Informed Probe-Interval Model — Proposal v2 (post-review)

**Author:** Claude (with project lead Jon McBee, Nabsys)
**Date:** 2026-04-21 (v2)
**Status:** Draft for Deep Think round-6 peer review
**Supersedes:** `2026-04-21-v3-physics-informed-proposal.md` (v1)
**Basis:** Incorporates Deep Think round-5 review (`Physics-Informed Nanopore Signal Processing.md`) plus a follow-up exchange on question 3 (probe-detection ceiling).

---

## Changelog vs. v1

Six substantive changes. Review these first; the rest of the document is edited to match.

1. **§3.2 — Macro/micro velocity decomposition.** Flat per-sample velocity head replaced by $v̂(t) = v_{\text{macro}}(t) \cdot v_{\text{micro}}(t)$. This was the fix for the variational degeneracy Deep Think flagged (piecewise-linear interpolation trivially satisfies L_511 + L_smooth).
2. **§3.3 — Mandatory metadata conditioning.** FiLM/AdaLN over the continuous metadata vector (pore voltage, temperature, molecule length estimate, pore-age proxy). A universal $v(I)$ across molecules is physically impossible; conditioning is not optional.
3. **§4 — L_current_velocity killed.** Physically wrong: drag is dominated by viscous force on the untranslocated tail outside the pore, not by the volume of the probe inside it.
4. **§4.4 — L_length upgraded from optional to mandatory-when-available.** Required to anchor the global integration budget in the face of the new v_macro parameterization.
5. **§5.1 — Cold-start the velocity head.** V1's velocity head is contaminated by remapping-supervised soft-DTW training; warm-starting it would leak the biases we're trying to kill. Backbone + probe head may be warm-started; velocity head is random-init.
6. **§5.3 — Probe detection is now trainable end-to-end with L_511 gradient, and a separate V3.1 track spins out self-supervised probe detection.** This resolves the ceiling concern ("are we capped at wfmproc's ability?") — the probe head is a warm start, not a freeze. V3.1 replaces wfmproc entirely.

Also: §6 pilot success criteria now specify p95/p99 tail thresholds, not just median. Evaluation moved off full-remapping MAE to a golden-subset methodology. §9 adds a new block of follow-up questions specifically about probe-head failure modes under L_511 refinement.

---

## 1. Why V2 is being killed

V1 and V2 are both supervised against remapping-derived ground truth (`.assigns` from the aligner + `probes.bin` bp-positions). The network learns to mimic the remapping pipeline. Hard ceiling: the model can never exceed its teacher, and cannot generalize beyond the regime where remapping works.

That is the wrong objective.

The actual end goal is:

> **Input:** a TDB file (raw 32 kHz current waveforms).
> **Output:** for each molecule, a list of inter-probe distances in basepairs.
> **Downstream:** assembler consumes interval lists and reconstructs the target genome; SV-calling follows assembly.

No remapping in the inference pipeline. Ideally no remapping-derived *positional* supervision in training either. V3 learns nanopore translocation physics directly from current, anchored by known physical constants.

V1 has value as a remapping emulator and as an oracle for validating V3 predictions. Finish the V1 cloud run, then stop that line.

V2's additional firepower (Transformer U-Net, Sinkhorn-OT, segmentation head) is architectural polish aimed at the wrong objective. Kill it.

---

## 2. The physical anchor

Every Nabsys probe is designed at **exactly 511 bp**. This is a hard, controlled, unchanging physical constant. It gives the per-sample translocation velocity $v(t)$ an integral identity:

$$\int_{t_i^{\text{enter}}}^{t_i^{\text{exit}}} v(t)\, dt = 511 \text{ bp} \quad \forall \text{ probe } i$$

Applied across thousands of probes in thousands of molecules, this is dense self-supervision on the velocity field. No aligner output required.

Inter-probe distance follows:

$$d(i, i+1) = \int_{t_i^{\text{exit}}}^{t_{i+1}^{\text{enter}}} v(t)\, dt$$

V3 is organized around making this identity trainable and the integrals numerically well-behaved — **and, crucially, around preventing the network from trivially satisfying the identity by ignoring the inter-probe signal.** (See §3.2.)

---

## 3. Architecture

### 3.1 Dataflow

```
raw current I(t) at 32 kHz
        │
        ▼
  per-molecule z-score normalization (unchanged from V1 collate)
        │
        ▼
  shared 1-D CNN U-Net backbone (inherited from V1)
        │       ↑ FiLM/AdaLN conditioning from metadata vector (see §3.3)
        │
        ├─► probe-event head   ──► P(probe | t), per-sample logit  (trainable, warm-started from V1)
        │
        └─► velocity head       ──► v̂(t) = v_macro(t) · v_micro(t)  (cold-started)

  post-processing (non-learned):
    1. extract discrete probe events {(t_i^enter, t_i^exit)} from the probe heatmap
    2. cumulative bp: B̂(t) = ∫₀ᵗ v̂ dτ
    3. per-probe transit: Δᵢ = B̂(t_i^exit) − B̂(t_i^enter)              [used in L_511]
    4. inter-probe interval: dᵢ = B̂(t_{i+1}^enter) − B̂(t_i^exit)        [primary output]
```

Backbone inherited from V1 to limit architectural risk. State-space models (Mamba) are the V3.1 architectural candidate — natively accumulate hidden state, which mirrors the physics of cumulative position — but not for the pilot.

### 3.2 Velocity head — macro/micro decomposition

**Why decomposed.** Deep Think proved (via calculus of variations) that a flat per-sample velocity head with L_511 + L_smooth is degenerate: the energy-minimal solution is piecewise-linear interpolation between probe-anchored values, which ignores $I(t)$ in the inter-probe gaps. A translationally-invariant CNN cannot model global deceleration (Stokes drag $t^{-1/3}$) from local receptive fields alone.

**Fix.** Parameterize velocity as the product of a slow global baseline and a fast local modifier:

$$v̂(t) = v_{\text{macro}}(t; \theta_{\text{mol}}) \cdot v_{\text{micro}}(t; I(t), \text{context})$$

where:

- $v_{\text{macro}}(t; \theta_{\text{mol}})$ is a smooth, global, per-molecule baseline velocity. Candidate parameterizations:
  - **Parametric**: $v_{\text{macro}}(t) = A (t + t_0)^{-\alpha}$ where $(A, t_0, \alpha)$ are predicted per-molecule from the metadata vector (voltage, estimated molecule length, pore-age proxy) via a small MLP. Physical: matches Stokes-drag power law.
  - **Nonparametric**: a low-capacity MLP fed positional encoding $t/T_{\text{mol}}$ and metadata, output smoothed across sample axis.
  - Pilot decision: start with **parametric power-law** for interpretability and physical grounding. Fall back to nonparametric if power-law fails empirically.
- $v_{\text{micro}}(t)$ is a per-sample CNN output passed through sigmoid and scaled to $[0.1, 2.0]$. Modulates the macro baseline up or down based on local current features. Has no knowledge of absolute $t$.

This parameterization is **mathematically forced** to learn the global power-law deceleration in $v_{\text{macro}}$ (because L_length and the overall budget of probe integrals cannot be satisfied with a constant baseline) and the local current-velocity coupling in $v_{\text{micro}}$ (because L_511 on individual probes sees high-frequency structure).

### 3.3 Metadata conditioning

Continuous metadata vector $\mathbf{m}$ per molecule, minimum:
- applied pore voltage
- estimated molecule length (from current-trace duration × nominal velocity — coarse)
- temperature
- pore-age proxy (seconds since pore reset, or flow-cell index)
- baseline open-pore current (before translocation)

Injected as **FiLM at each U-Net stage** (scale + shift of intermediate features) and as input to the $v_{\text{macro}}$ MLP.

Without this, the model treats every molecule as drawn from the same physical regime. It isn't. Pore aging, voltage, and temperature dominate inter-molecule velocity variance.

### 3.4 Probe-event head

CenterNet-style peak heatmap (per-sample logit), **trainable end-to-end** — not frozen. Hysteresis-threshold decode at inference produces probe events as intervals, not just centers (width is required for the L_511 integration limits).

Warm-started from V1's probe head (already learned to approximate probe events). L_511 gradient flows back into probe-head parameters: a false-positive detection in a noise region contributes a large L_511 residual (no 511 bp there, no velocity can integrate to it), pushing the detector to suppress it.

This is the mechanism by which the network **can exceed wfmproc's detection accuracy**. See §5.3 for the full argument and the V3.1 follow-up track.

---

## 4. Losses

### 4.1 L_511 — the core integral identity

$$\mathcal{L}_{511} = \frac{1}{N_{\text{probes}}} \sum_i \left( \int_{t_i^{\text{enter}}}^{t_i^{\text{exit}}} v̂(t)\, dt - 511 \right)^2$$

Integral is a sample-weighted sum over samples inside each probe interval. Gradient flows through $v̂$ (both $v_{\text{macro}}$ and $v_{\text{micro}}$) during the probe, and through the probe-head logits that define $(t_i^{\text{enter}}, t_i^{\text{exit}})$ via a differentiable boundary-softening scheme (sigmoid mask over interval endpoints).

### 4.2 L_smooth

$$\mathcal{L}_{\text{smooth}} = \frac{1}{T} \sum_t \left( v̂(t+1) - v̂(t) \right)^2$$

Applied to the composed $v̂$, not to $v_{\text{macro}}$ alone. Kept as a weak regularizer; its role shrinks now that the macro/micro decomposition carries most of the smoothness burden.

### 4.3 L_length — mandatory when molecule length is known

$$\mathcal{L}_{\text{length}} = \left( \int_0^{T_{\text{mol}}} v̂(t)\, dt - L_{\text{known}} \right)^2$$

Anchors the global integration budget. **Required** to prevent $v_{\text{macro}}$ from drifting into a degenerate scaling. For the E. coli pilot, $L_{\text{known}}$ is available for each molecule at rough accuracy; this is sufficient.

For future targets where molecule length is not known per-sample, a per-target distributional prior can be substituted ($\int v̂ dt$ matches a known distribution of lengths), but that's out of pilot scope.

### 4.4 L_probe — supervision for the probe head during bootstrap

Per-sample focal BCE on probe-head logits, using wfmproc's `probes.bin` event labels.

**Crucially, this is a warm-start regularizer, not a ceiling.** The loss is weighted $\lambda_p$ which **decays to zero across training** (curriculum: $\lambda_p = 0.1 \to 0$ over the first ~10 epochs). After decay, probe detection is refined only by L_511's backprop pressure. This lets the network diverge from wfmproc where the physics says wfmproc was wrong.

### 4.5 L_current_velocity — KILLED

Previously proposed as a monotone prior (deeper current → slower velocity). Deep Think's physics: **drag is dominated by viscous force on the untranslocated tail outside the pore**, not by the probe volume inside it. Deep blockade ≠ high drag. Removing.

### 4.6 Total loss

$$\mathcal{L} = \lambda_{511} \mathcal{L}_{511} + \lambda_s \mathcal{L}_{\text{smooth}} + \lambda_L \mathcal{L}_{\text{length}} + \lambda_p(t) \mathcal{L}_{\text{probe}}$$

Recommended starting weights:
- $\lambda_{511} = 1.0$
- $\lambda_s = 0.001$ (weaker than v1; macro/micro carries most smoothness)
- $\lambda_L = 0.5$ (mandatory; heavy weight to anchor global integration)
- $\lambda_p(t=0) = 0.1 \to 0$ over 10 epochs

---

## 5. Training regime

### 5.1 Warm-start policy

- **Backbone**: load from V1 best checkpoint. Features are cheap to transfer and uncontroversial.
- **Probe head**: load from V1. Trainable. Initial probe detection is wfmproc-quality; refines under L_511 gradient.
- **Velocity head ($v_{\text{macro}}$ and $v_{\text{micro}}$): cold-start, random init.** V1's velocity head was trained by soft-DTW against remapped `.assigns` files. Loading it would leak remapping biases into the very component that is meant to replace them.
- **Metadata-FiLM modulators**: cold-start.

### 5.2 Training sequence

1. Freeze backbone. Train only velocity head + FiLM + probe head for 2 epochs with full loss. Lets the new components catch up.
2. Unfreeze backbone. Train full network for ~30 epochs.
3. At epoch 10, $\lambda_p \to 0$: probe detection now depends only on L_511 gradient.

### 5.3 Probe detection: warm-start, then refine, then replace

This is where the "wfmproc ceiling" concern is resolved. Three stages:

**Stage A — Pilot (weeks 1–2).** Probe head warm-started from V1/wfmproc labels. Trainable. L_511 refines it. By end of pilot, measure:
- Does the refined probe head outperform frozen wfmproc on a hand-labeled validation slice?
- Do inter-probe intervals hit target accuracy?

If yes, the ceiling concern is empirically resolved — the network exceeded its bootstrap.

**Stage B — V3.0 production.** Continue refine-through-L_511 approach. Weight $\lambda_p$ on probe supervision decays to zero; only L_511's backprop trains probe detection after warm-up. Ship V3.0 when assembly metrics are acceptable.

**Stage C — V3.1 (separate research track, post-pilot).** Fully self-supervised probe detection. Three candidate approaches:
1. **Contrastive / masked-reconstruction** on the raw waveform. Learn an embedding that discriminates stereotyped probe events from baseline without labels. Sparse, repeating events are strong contrastive signal.
2. **Anomaly autoencoder** on open-pore baseline. Probes emerge as reconstruction residuals. Simpler but noisier — pore gating and bubbles will also fire.
3. **Joint cold-start** with L_511 providing the only signal to the probe head. Chicken-and-egg; flagged as likely to fail but worth a controlled attempt.

V3.1 removes wfmproc from the training pipeline entirely. Benchmark: V3.1 probe detector vs. wfmproc vs. V3.0's L_511-refined detector, on held-out hand-labeled data.

**Why not do V3.1 in the pilot.** The pilot's purpose is to answer *one* question: does L_511 + macro/micro velocity work? Conflating it with "can we learn probe detection from scratch" is two risky hypotheses at once. If the joint pilot fails, we don't know which one broke. Separate them; derisk L_511 first.

---

## 6. Pilot study

Same goal as v1: cheap test of whether L_511 (plus the new macro/micro decomposition and mandatory L_length) can recover inter-probe distances accurately enough for assembly.

### 6.1 Design

1. **Inputs:** existing 30 E. coli caches. All `.assigns`-derived positional labels removed from training. Probe events come from `probes.bin` (signal-processing output of wfmproc, no reference genome).
2. **Model:** V1 backbone + V1 probe head warm-start + **cold-start velocity head** with macro/micro split + metadata FiLM.
3. **Loss:** L_511 + L_smooth + L_length + decaying L_probe (weights per §4.6).
4. **Training:** 10 epochs on 4 training caches. Fast iteration.
5. **Evaluation:** held-out cache, on a **Golden Subset** of molecules where wfmproc-aligner had high confidence (e.g., >95% of probes assigned, residual < threshold). This avoids the trap of penalizing V3 when V3 is right and remapping is wrong.
6. **Secondary evaluation (if time permits):** de novo assembly metrics on V3 intervals — contig N50, misassembly count, reference recovery rate — on a single small assembly task.

### 6.2 Success criteria

Two dimensions: central tendency and tails. Assembly tolerates fuzz; it chokes on heavy tails because a single bad interval creates a false edge in the layout graph and shatters contigs.

| Outcome | Median rel. err. | **p95 rel. err.** | **p99 rel. err.** | Decision |
|---|---:|---:|---:|---|
| Excellent | < 2% | < 5% | < 10% | commit to V3.0 production build |
| Good | 2–5% | < 10% | < 15% | commit; consider architectural upgrades (Mamba, stronger conditioning) |
| Marginal | 5–10% | 10–20% | 15–30% | concept viable, needs richer priors before commit |
| Poor | > 10% | > 20% | — | L_511 + macro/micro + L_length is insufficient. Fundamental rethink. |

p95 is the go/no-go gate, not the median. Assembly cares about the tail.

### 6.3 Secondary pilot measurements

- **Probe-head refinement check (for the ceiling concern):** at pilot end, compare the trained probe head to frozen wfmproc on a hand-labeled slice (~50 molecules, hand-curated). Report precision/recall delta. Positive delta = ceiling concern empirically resolved. Negative delta = investigate; probe head may be regressing under L_511 pressure.
- **Macro-velocity fit check:** inspect learned $v_{\text{macro}}$ curves across molecules. Do they show the power-law shape? Is the learned $\alpha$ exponent near the Stokes value (~1/3)? If not, the parametric form may be wrong.
- **Cross-molecule consistency:** plot $v_{\text{micro}}$ vs. $I(t)$ across many molecules. Should collapse to a universal relationship (with conditioning). If it doesn't, FiLM is insufficient.

---

## 7. Risks

### 7.1 Variational degeneracy (the old #7.1)

**Status: resolved by macro/micro decomposition.** Previously the dominant risk. Now the architecture itself prevents the trivial solution.

### 7.2 Probe-head drift under L_511 refinement (NEW)

L_511 pressures the probe head to suppress detections that don't integrate to 511. **But it also has perverse incentives:**
- Dropping ALL probes trivially minimizes L_511 (zero residual, vacuous sum). The decaying L_probe warm-start prevents this early but not forever.
- Shrinking detected probe widths toward zero also trivially minimizes the integral (but then $\Delta t \to 0$ and L_length resists).
- Jointly moving probe boundaries and velocity can find pathological local minima.

**Mitigations:**
- Keep $\lambda_p > 0$ even after "decay" — decay to small nonzero value, not literal zero.
- Add minimum-probe-count prior: expected $N_{\text{probes}}$ per unit molecule length, penalize deviations.
- Monitor probe-count trajectory in training logs. If the count collapses, intervene.
- Compare to frozen wfmproc detections every N epochs as a sanity check.

### 7.3 Probe width variability

511 bp is the *design*. Actual probe transit may have bp-width jitter from conformational effects. If ±1 bp, irrelevant. If ±10 bp, L_511 becomes noisy. **Open empirical question resolvable in the pilot** by regression: $\text{std}(\int v̂ \, dt | \text{probe})$ across well-aligned probes.

If needed, relax to $\int v̂ dt \sim \mathcal{N}(511, \sigma^2)$ and either marginalize analytically or treat $\sigma$ as a learned scalar.

### 7.4 Cross-molecule variability

Pore aging, voltage, temperature. **Addressed by FiLM conditioning** (§3.3). Residual risk: the metadata vector may miss relevant state (e.g., contamination events). Monitor via cross-molecule consistency plots (§6.3).

### 7.5 Macro-velocity parametric form

Power-law is a physical guess, not a certainty. If actual velocity profiles don't fit, $v_{\text{macro}}$ will force a bad baseline that $v_{\text{micro}}$ cannot fully correct. **Mitigation:** nonparametric backup ($v_{\text{macro}}$ as a low-capacity MLP). Decision in pilot.

### 7.6 Golden Subset selection bias

Evaluating on high-confidence-remapping molecules could systematically be the "easy" ones and overstate performance. **Mitigation:** also report on the full validation set; differences between golden and full give a measure of the selection effect.

---

## 8. What this proposal does NOT address

- Strand orientation / probe identity (probe-typing) — V3.2 concern.
- Molecules with very few probes (N ≤ 2) — L_511 is weak; rely on L_length or exclude.
- Unknown total molecule length — L_length disabled; model produces relative intervals only, acceptable for assembly after per-molecule normalization.
- Assembly / SV-calling integration — out of scope.
- Training-data diversity beyond E. coli — post-pilot planning.
- The V3.1 fully-self-supervised probe-detection track is **described** (§5.3) but its detailed research plan is out of scope for this document.

---

## 9. Open questions for Deep Think (round 6)

### 9.A New / follow-up on the probe-detection ceiling (Q3 from v1)

**Q3.1** — Given that we make the probe head trainable end-to-end with L_511 gradient flowing into it, do you see any failure mode where the network drifts *worse* than its wfmproc warm-start? Specifically:

- Can L_511 push the probe head into mode collapse (dropping all probes)?
- Can L_511 push the probe head into pathological shrinking of detected widths to trivially satisfy the integral?
- Is the proposed mitigation (decaying $\lambda_p$ to small nonzero + minimum-probe-count prior + training-time monitoring) sufficient, or are there other adversarial routes through the loss landscape we should guard against?

**Q3.2** — Is there a principled schedule for $\lambda_p$ decay that's better than a linear ramp? E.g., a schedule that floors at a level proportional to the disagreement between wfmproc and the L_511-refined detector, so the warm-start regularizer strengthens only where the refinement is untrustworthy.

**Q3.3** — For the V3.1 fully-self-supervised track: of the three candidate approaches (contrastive, anomaly-AE, joint cold-start), which do you consider most likely to succeed? Is there a fourth I haven't considered? Specifically, is there a way to use L_511 itself as a self-supervised signal for probe *detection* — i.e., the network hypothesizes probe locations, computes L_511, and refines the hypothesis — with no wfmproc bootstrap at all?

**Q3.4** — How would you design a benchmark that distinguishes "probe head learned to exceed wfmproc" from "probe head is just overfitting to the L_511 training set"? Hand-labeled validation is the obvious answer, but at what scale does it become statistically meaningful for tail events?

### 9.B Carried over from v1 (restated against the v2 design)

**Q1.** Is L_511 + L_smooth + L_length with macro/micro decomposition provably well-posed? v1's Q1 is answered (no — pre-decomposition, it was degenerate). Is the post-decomposition version free of similar pathologies? Are there residual degeneracies we should guard against (e.g., gauge symmetries where $v_{\text{macro}} \to k \cdot v_{\text{macro}}$ and $v_{\text{micro}} \to k^{-1} \cdot v_{\text{micro}}$ leaves $\hat v$ invariant but could cause training instability)?

**Q2.** With L_current_velocity killed, are there other physical priors worth adding — e.g., ionic-current-to-conformation coupling, polymer-physics constraints on tension propagation during probe transit, drift-diffusion dynamics inside the pore?

**Q4.** Pilot design critique. Is 10 epochs × 4 caches × held-out golden subset enough signal? Does the golden-subset methodology address the circularity concern (V3 right / remapping wrong) without introducing new selection biases?

**Q5.** Macro-velocity parametric form. Is Stokes-drag power-law the right first guess? Or should we start from a fully nonparametric $v_{\text{macro}}$ and let the data tell us the form?

**Q6.** FiLM conditioning. Are the five metadata channels (voltage, length, temperature, pore-age, baseline current) the right set? Are there measurements we're not currently capturing that we should be?

**Q7.** Architecture. V3.1 candidate is Mamba / State-Space Model. Is that the right bet, or is there a stronger physically-motivated architecture (e.g., an explicit neural-ODE integrator over the velocity field, hard-coding the $\int v \, dt$ structure into the forward pass)?

**Q8.** Assembly-downstream consequences. Is p95 rel. err. < 10% a defensible threshold, or should we move directly to assembly metrics (N50, misassembly count) in the pilot?

---

## 10. Decision requested

Deep Think is asked to:
1. Evaluate whether the macro/micro velocity decomposition actually escapes the variational degeneracy, or merely hides it.
2. Critique the probe-head drift risk analysis (§7.2) and the follow-up questions in 9.A.
3. Sign off on the pilot's modified design (golden-subset evaluation, p95/p99 thresholds) as sufficient to go/no-go V3.0.
4. Flag any new fundamental objection introduced by the v2 changes.
5. Identify any references in the physics-informed neural network or nanopore-physics literature that would sharpen specific design choices.

**Output requested:** a written review in the format used for prior rounds, with an explicit verdict at the end (proceed as-proposed / proceed with modifications / do not proceed / need more evidence) and, if modifications, a minimal set of pre-pilot changes.

---

*End of proposal v2.*
