# Project Mongoose: Rich-Slide Deck Outline (v2)

**Purpose:** Authoritative outline for a NEW deck where the slides themselves convey content at the depth of presenter talking points. This replaces the prior outline approach where slides carried terse bullets and depth lived in speaker notes.

**Audience:** Internal Nabsys technical review. Mix of signal processing engineers, bioinformaticians, and leadership. Slides should be readable as a self-contained document if circulated without a presenter.

**Format:** 12 slides total. Each slide is dense but structured -- think McKinsey-style briefing slides where each slide is a short memo with a clear hierarchy.

**Slide layout convention:**
- Each slide has a declarative title (one assertion)
- A one-sentence thesis line directly under the title
- 3-5 labeled body sections (small caps headers, paragraph-style content beneath)
- One supporting visual element per slide (chart, diagram, or callout)
- Footer with section number and slide title for navigation

**Body text size:** 10-11pt. Slides will be denser than typical decks; that is intentional and matches the user's request. Reading time per slide: 60-90 seconds silent, 4-6 minutes with presenter.

---

## CRITICAL GUARDRAILS for Cowork

This is a NEW deck. Do not modify, extend, or reuse the existing `Project_Mongoose_Review.pptx` file. Build a new file, e.g. `Project_Mongoose_Review_v2.pptx`.

Do NOT include any of these claims (carryovers from earlier Gemini deck that are inaccurate or aspirational):
- Zero-shot human deployment / "without a single retraining epoch"
- Endogenous quality canary as a delivered feature
- "Stalling instrument throughput"
- Full pipeline replacement
- Human DNA framing for V1 (E. coli only)

Use these terms consistently:
- **T2D U-Net** (not "Kinematic U-Net")
- **time-to-distance conversion** (not "DNA kinematics")
- **wfmproc** (lowercase) for the legacy signal processing
- ***E. coli*** K-12 MG1655 (italicize on first reference)
- **BssSI** (E. coli runs use BssSI only)
- **base pairs** / **bp** lowercase

---

## Deck Structure (12 slides)

---

### Slide 1: Title

**Title:** Project Mongoose: A Physics-Informed Deep Learning Model for Time-to-Distance Conversion

**Subtitle:** Replacing the parametric drag model in the Nabsys signal processing pipeline

**Footer:** Internal Review · V1 Architecture · April 2026

**Visual:** Raw TDB voltage trace transitioning into a base-pair coordinate axis. The legacy parametric equation `L = C · (t + offset)^α` shown as a dashed curve diverging from the true signal.

---

### Slide 2: The T2D transform is the accuracy ceiling, not the detector

**Thesis:** The Nabsys detector produces clean, high-bandwidth electrical signatures. The limitation is in the time-to-distance math that converts those signatures into base-pair coordinates -- and the limitation is fundamental, not a tuning problem.

**SECTION 1 -- THE CURRENT MODEL**

The current T2D conversion is a parametric equation: `L = C · (t + offset)^α`, fitted with three parameters per channel across tens of thousands of channels in the fleet. It is a single-mode drag approximation derived from idealized polymer physics. Every downstream step in the analysis pipeline -- remapping, structural variant calling, coverage estimation -- inherits whatever error this conversion introduces. Improving T2D accuracy compounds across the pipeline.

**SECTION 2 -- WHAT THE PARAMETRIC MODEL CANNOT REPRESENT**

Three physical effects appear in essentially every molecule we record. The three-parameter model averages all of them into noise:

- **Head dive.** As the leading edge of the DNA enters the channel, it experiences a brief acceleration burst before drag from the unbound trailing portion sets in. The parametric model assumes monotonically changing velocity from leading to trailing edge.
- **Tag velocity compensation (TVC).** When dense tag clusters pass through the detection volume, their bulky physical footprint creates additional local hydrodynamic drag, producing localized deceleration. The model has no representation of position-dependent drag.
- **Channel drift within a run.** Over 12-24 hour runs, NaOH channel maintenance, clogging, and concentration polarization shift effective velocity. A single fitted alpha cannot represent both start-of-run and end-of-run kinematics.

**SECTION 3 -- THE EMPIRICAL SIGNATURE**

On 31,468 clean *E. coli* molecules from existing data, we measured trailing-end velocity at ~1,487 kbp/s and mid-molecule velocity at ~341 kbp/s -- a 3.5x ratio. The parametric model represents this as a smooth single-exponent curve. The leading-edge head-dive burst shows up directly as a velocity peak the parametric model cannot reproduce.

**Visual:** Median tag duration (left axis, inverted) vs. position along molecule. Three regions annotated with the measured velocities. Overlay: the smooth `t^α` curve, clearly diverging from the empirical signal in the head-dive region and at the trailing whip.

---

### Slide 3: Replace the parametric equation with a learned velocity field

**Thesis:** The proposal is to replace the three-parameter equation with a 1D U-Net that reads the raw 40 kHz voltage trace per molecule and outputs an instantaneous velocity field. Deterministic integration produces the spatial coordinate.

**SECTION 1 -- INPUT**

A single molecule's raw Int16 waveform from the TDB file, normalized by its level-1 backbone amplitude. The level-1 normalization neutralizes the dominant cross-channel geometric variance, so a molecule from a 30nm channel and a molecule from a 32nm channel produce mathematically similar normalized waveforms. The network learns one universal waveform shape rather than one shape per detector.

**SECTION 2 -- OUTPUTS (TWO HEADS)**

The model produces two outputs at the same 40 kHz resolution as the input.

- **Probe heatmap** -- per-sample probability that a tag is centered at that position. Replaces wfmproc's peak-picker.
- **Cumulative base pairs** -- per-sample spatial coordinate measured from molecule start. The inter-probe distances we care about downstream are differences between cumulative bp values at consecutive probe positions. Replaces the parametric T2D equation.

**SECTION 3 -- SCOPE BOUNDARY (CRITICAL)**

The model does **not** replace molecule detection, level-1 estimation, or quality flagging -- those remain wfmproc's responsibility. V1 takes the per-molecule waveform that wfmproc already produces and replaces exactly the T2D step. This is a deliberate scope choice for failure attribution. If V1 fails, we want to know that the T2D math was the problem, not that molecule detection broke simultaneously. V2+ extends scope toward a unified model.

**SECTION 4 -- WHY A 1D U-NET**

The task is sequence-to-sequence with the same temporal resolution at input and output -- exactly the problem class U-Nets were designed for. The encoder-decoder structure with skip connections captures the two frequency scales we need simultaneously: the macro-velocity context that depends on total remaining drag (deep path through the bottleneck) and the localized decelerations from tag-induced drag (skip connections preserve high-frequency detail). About 15M parameters, small by modern ML standards.

**Visual:** Pipeline diagram. Top row: TDB stream -> wfmproc (existing, in gray) -> per-molecule waveform. Bottom row: per-molecule waveform -> T2D U-Net (highlighted) -> probe heatmap + cumulative bp. The boundary between wfmproc and T2D U-Net marked with a vertical scope-line.

---

### Slide 4: Predict velocity, integrate deterministically

**Thesis:** What the network outputs matters more than how big it is. A model that directly predicts cumulative base pairs is brittle. A model that predicts instantaneous velocity, then integrates deterministically, is robust by mathematical construction.

**SECTION 1 -- WHY DIRECT BP PREDICTION FAILS**

Cumulative bp is a non-stationary, monotonically-increasing quantity. 1D convolutional networks are translation-equivariant -- excellent at extracting local stationary properties, terrible at producing globally-consistent accumulators. A network trained to directly output cumulative bp produces noisy predictions that violate monotonicity. It can output bp positions that decrease from one sample to the next, implying DNA traveling backward in time.

**SECTION 2 -- THE ALTERNATIVE (THREE STAGES)**

- **Stage 1.** Predict raw velocity (bp/sample). A local stationary property -- exactly what 1D convolutions extract well.
- **Stage 2.** Apply Softplus activation (smooth approximation of ReLU). Strictly positive for any input. Velocity is always greater than zero, meaning DNA always moves forward.
- **Stage 3.** Apply `torch.cumsum` -- a parameter-free, differentiable cumulative sum. Given per-sample velocity, returns the running total in bp.

**SECTION 3 -- WHAT THIS GUARANTEES BY CONSTRUCTION**

The output is monotonically non-decreasing as a structural property of the architecture, not as a learned soft constraint. The model physically cannot predict DNA traveling backward. This is what "physics-informed" means in practice: we baked the physical constraint into the architecture so the network has one fewer thing to learn from data.

**SECTION 4 -- THE BONUS PROPERTY**

The cumulative sum is itself a perfect low-pass filter. Any high-frequency noise in the velocity prediction washes out in the integral. The network can make noisy local velocity predictions (which CNNs naturally do) and still produce a smooth, physically plausible cumulative bp curve. This is why we don't need to explicitly regularize the velocity output.

**Visual:** Three-stage pipeline diagram. Left: noisy raw velocity tensor (mixed positive/negative spiky values). Middle: Softplus curve with shaded "always positive" region. Right: smooth, monotone cumulative bp curve. Arrows between stages.

---

### Slide 5: Ground truth is anchored to the *E. coli* reference genome

**Thesis:** Training labels produced by the legacy T2D would teach the network to replicate legacy biases. The reference genome is the only source of spatial truth that does not come from the model we are trying to replace. We anchor the loss there using shift-invariant inter-probe deltas.

**SECTION 1 -- THE TRAP**

A naive approach would use probes.bin as the source of truth. But probes.bin's spatial coordinates were produced by the legacy T2D. Training a neural network to match them teaches it to replicate the legacy model's biases. We would have spent thousands of dollars of compute to build, at best, a slightly faster version of the equation we already have. At worst, we would bake in legacy T2D's errors and call it progress.

**SECTION 2 -- THE SOLUTION**

There is exactly one source of spatial truth that does not come from legacy T2D: the *E. coli* reference genome. *E. coli* K-12 MG1655 is a known 4,641,652 bp sequence. BssSI recognition sites occur at 810 specific known positions. When the Nabsys aligner successfully maps a molecule to a stretch of the reference, it identifies which specific reference probe each detected tag corresponds to. Each matched reference probe has an exact bp coordinate from the FASTA file -- a coordinate that does not depend on any signal processing step.

**SECTION 3 -- SHIFT-INVARIANT INTER-PROBE DELTAS**

We do not train on absolute positions. We train on differences between consecutive probes: `GT_deltas = abs(diff(reference_bp_positions))`. If the reference says probe A is at position 500,000 and probe B is at position 512,500, the training label for that interval is exactly 12,500 bp. That number is unassailable.

Differencing buys two additional benefits. **First**, taking absolute value handles forward and reverse molecule orientations natively -- distance is always positive. **Second**, the loss becomes shift-invariant to the "blind overhang" problem. Physical molecule ends do not land on restriction sites, and the aligner estimates those tail lengths using legacy T2D. By training only on inter-probe deltas, we don't care where the molecule starts or ends -- we only care that distances between visible probes match the reference.

**SECTION 4 -- THE COMPLETE LOSS PIPELINE**

The model outputs cumulative bp. We sample that output at the sample indices of the detected probes. We take the differences. We compare those predicted differences to the reference deltas with a Huber loss. The gradient flows back through `cumsum`, through the velocity predictions, all the way back through the encoder. The reference genome -- not the legacy model -- drives the learning signal.

**Visual:** Three-row stack. Top: molecule waveform with detected probes marked. Middle: *E. coli* K-12 reference slice with known bp positions marked, lines connecting matched probes between the two rows. Bottom: the `abs(diff)` operation producing the GT_deltas array as a sequence of bp values.

---

### Slide 6: The 511 bp tag as a self-calibrating velocimeter

**Thesis:** Every detected tag occupies a constant 511 bp of DNA. Its temporal width in the signal is therefore a direct measurement of local velocity at that point on the molecule -- a physics constraint we can use to supervise the model's velocity prediction directly, not just through the bp integral.

**SECTION 1 -- THE PHYSICAL INVARIANT**

Every tag that binds to a nick site occupies exactly 511 base pairs of DNA. This is a chemical constant of the labeling chemistry, not a fitted parameter. It is the same for every tag, on every molecule, on every channel, on every instrument. Variance is sub-bp.

**SECTION 2 -- WHAT THIS MAKES POSSIBLE**

As a tag passes through the detector, it occupies 511 bp of DNA. If the DNA is translocating at velocity v, the tag takes `511 / v` seconds to pass. We measure this duration directly in the signal -- it is the temporal width of the voltage deflection. From this:

`local_velocity = 511 bp / tag_duration_in_ms`

Every detected tag is a speedometer reading. Not an estimate -- a direct measurement, anchored in the physical dimensions of the molecule.

**SECTION 3 -- EMPIRICAL VALIDATION**

On 31,468 clean *E. coli* molecules from existing data, the tag-width measurement directly confirms the biophysics:

- **Leading edge** (first 10%): median ~430 kbp/s
- **Middle** (40-60%): median ~341 kbp/s -- the slowest, where remaining drag is maximum
- **Trailing edge** (last 10%): median ~1,487 kbp/s -- 3.5x faster than the middle, near terminal velocity

These numbers were not fitted. They are direct measurements from the signal, enabled by knowing every tag is 511 bp wide. Both the leading-edge head-dive burst and the trailing-end velocity peak are visible.

**SECTION 4 -- HOW THE MODEL USES IT**

A third loss term -- `L_velocity` -- compares the model's raw velocity prediction at each detected probe position against the physics-consistent value of `511 / measured_tag_duration`. Sparse supervision (only at tag positions, ~15-50 points per molecule) that pulls local velocity toward physical consistency.

`L_velocity` is auxiliary. The reference-anchored `L_bp` is the dominant signal; `L_velocity` exists to keep local velocity predictions physically consistent so the network is not free to make up arbitrary velocity curves as long as the integral happens to land right. We test this with a planned ablation: if removing `L_velocity` does not hurt MAE, the reference anchor is doing all the work and `L_velocity` can be dropped.

**Visual:** Plot of median tag duration vs. position along molecule, with three velocity regimes annotated as colored bands. Inset: tiny diagram showing `velocity = 511 bp / duration_ms` formula with example numbers.

---

### Slide 7: Architecture overview -- 5-level encoder, hybrid bottleneck, bifurcated decoder

**Thesis:** The T2D U-Net is a 1D U-Net of approximately 15M parameters with three targeted modifications, each motivated by a specific failure mode the architecture would otherwise hit. Small by modern ML standards -- the problem requires the right architectural inductive biases, not raw parameter count.

**SECTION 1 -- ENCODER (5 LEVELS, STANDARD)**

Five levels of 2x downsampling. Channels: 32 → 64 → 128 → 256 → 512. Kernel size 7 throughout. Each level has two residual blocks. Standard U-Net structure. After all five levels, a 4,000-sample input molecule is compressed to 125 bottleneck tokens.

**SECTION 2 -- BOTTLENECK (HYBRID -- DEPARTURE #1)**

The first significant departure from a vanilla U-Net. Two mechanisms expand the effective receptive field essentially for free at this compressed scale:

- **Dilated convolution cascade** with rates 1, 2, 4, 8 (kernel 7). At 32x downsampling, each dilated conv sees a wider span without adding parameters. Cumulative theoretical RF exceeds 4,500 input samples -- enough to span our worst-case probe deserts (covered on slide 8).
- **Single 1D Multi-Head Self-Attention layer** (4 heads). Gives every bottleneck token a global view of the entire molecule. Computationally trivial at 125-625 tokens.

**SECTION 3 -- DECODER (BIFURCATED -- DEPARTURE #2)**

Symmetric 5-level decoder with skip connections all the way back to full 1x resolution. The departure: rather than a single output head, the decoder bifurcates at full resolution into two parallel heads with different kernel sizes. **Probe head** (kernel 7) for sharp localization. **Velocity head** (kernel 31) for smooth integration. Detail on slide 9.

**SECTION 4 -- FILM CONDITIONING (DEPARTURE #3)**

A 6-value continuous physics vector modulates features at encoder level 0 and at the bottleneck. The values are macro-physics signals that cannot be recovered from the normalized waveform alone:

1. **Absolute pre-event baseline voltage** -- channel state proxy
2. **log(molecule duration)** -- macro-scale drag prior
3. **log(inter-event interval)** -- concentration polarization recovery
4. **Time-in-run fraction** -- Joule heating drift
5. **Applied bias voltage** -- electric field drives velocity
6. **Applied pressure** -- pneumatic back-pressure

All measurable per-molecule without legacy T2D. We deliberately do **not** feed the network a categorical channel ID. Categorical IDs fail zero-shot on new dies; continuous physical observables let the model learn generalizable physics that transfer to unseen hardware.

**Visual:** Full T2D U-Net architecture diagram. 5-level encoder on left descending, dilated cascade + MHSA bottleneck at the bottom, 5-level decoder on right ascending with skip connection arrows, bifurcation into two heads at the top right. FiLM injection arrows at L0 and bottleneck, with the 6-value vector listed in a sidebar.

---

### Slide 8: Receptive field must span the worst-case probe desert

**Thesis:** Between BssSI tag clusters, *E. coli* molecules contain regions with no probes at all. The model must predict smooth, physically plausible velocity through 4,000+ consecutive samples where the only signal is the flat normalized backbone. The bottleneck design solves this -- a vanilla U-Net would not.

**SECTION 1 -- THE WORST-CASE MATH**

Typical inter-tag gaps are a few thousand bp. Worst-case gaps reach 15-20 kbp -- particularly when wfmproc misses a tag and effectively merges two adjacent gaps. A 20 kbp desert at slow local velocity (200 kbp/s) spans 100 milliseconds, which at 40 kHz is **4,000 samples**. The model must predict 4,000 consecutive samples of plausible velocity from baseline alone, with no probe anchors to pin the velocity to.

**SECTION 2 -- WHY A NAIVE U-NET FAILS**

A standard 5-level U-Net with kernel 7 has a theoretical receptive field around 1,000-2,000 samples. This sounds adequate but isn't, because the **effective** receptive field is Gaussian-weighted -- samples at the extreme edges of the theoretical window contribute essentially zero gradient to the central prediction. If the flanking probes are at the outer edges of the theoretical RF, they are functionally invisible. In the middle of a desert, the model has nothing to anchor to and defaults to predicting an average-case velocity. That default error compounds through the integration and corrupts the spatial output across the entire desert.

**SECTION 3 -- THE SOLUTION**

Two complementary mechanisms in the bottleneck:

- **Dilated cascade** (rates 1, 2, 4, 8). Pushes theoretical RF beyond 4,500 input samples. Costs zero parameters.
- **Single MHSA layer**. At 125-625 bottleneck tokens, attention is computationally trivial and gives every token a true global view. Dilated cascade handles mid-range structure; attention handles macro-scale anchoring.

**SECTION 4 -- WHY NOT JUST GO DEEPER**

Adding a 6th encoder level (64x downsampling) would expand RF but destroy temporal resolution. The 511 bp tag width is only ~14 samples at the trailing edge; at 64x downsampling, the tag-induced velocity dips would alias away. Bottleneck-level RF expansion via dilation and attention is essentially free in compute and adds no parameters. Encoder-depth expansion would cost us resolution we cannot afford to lose.

**Visual:** Horizontal molecule-length bar with two probe clusters at the ends and a 20 kbp desert in the middle. Overlay: a Gaussian-weighted ERF curve centered in the desert, with shaded regions showing where flanking probes fall. Annotation: "naive U-Net ERF" (probes outside fat center) vs "T2D U-Net ERF" (probes inside fat center).

---

### Slide 9: Two heads, two frequency regimes -- full resolution for both

**Thesis:** Probe detection and velocity prediction have opposite frequency requirements. They share the encoder, decoder, and skip connections but diverge in the final stage with different kernel sizes.

**SECTION 1 -- PROBE HEAD (HIGH FREQUENCY)**

Tags must be localized to within 1-2 samples to support downstream inference. The heatmap must be sharp -- essentially a narrow Gaussian at each tag center. Wide kernels would smooth this out and corrupt sub-sample precision. Probe head: two residual blocks at kernel size 7, then a 1×1 convolution to a single-channel heatmap, then sigmoid.

**SECTION 2 -- VELOCITY HEAD (LOW FREQUENCY -- WITH ONE EXCEPTION)**

Velocity changes smoothly over long timescales as drag redistributes during translocation. Sharp kernels would produce high-frequency noise in the velocity output, which corrupts the cumulative integral. We want wide kernels that act as learned low-pass filters. Velocity head: two residual blocks at kernel size 31.

**The exception.** Bulky tags physically plug the channel and produce localized cliff-edge decelerations -- tag-induced velocity dips that span only a few samples. The velocity head must preserve these. This is why the velocity branch operates at full 1x resolution. Downsampling to 8x would compress a trailing-edge 511 bp tag (~14 samples) to under 2 tokens; the tag-induced dip would become unrepresentable, and upsampling back to 1x would smear it across 32 samples -- producing gradient conflict with the sparse `L_velocity` supervision at exactly those sample indices.

**SECTION 3 -- WHERE THEY SHARE AND WHERE THEY SPLIT**

Both heads share the entire encoder, all skip connections, and the decoder up through full 1x resolution. They split into parallel paths only at the final stage, with different kernel sizes per path. This means the heads can specialize at the output without forcing the rest of the network to compromise.

**SECTION 4 -- HOW THE OUTPUTS COMBINE FOR INFERENCE**

Inference takes the probe heatmap, runs velocity-adaptive non-maximum suppression to extract integer peak indices, then refines them to sub-sample positions via parabolic interpolation on the heatmap values. For each refined probe position, the bp coordinate is read off the cumulative bp curve via linear interpolation. The result is a list of (probe_bp, confidence) pairs -- exactly what downstream remapping consumes.

**Visual:** Side-by-side narrow vs. wide kernel illustration -- a sharp single peak (k=7) and a smoothed Gaussian (k=31). Below: a small inset showing the bifurcation point in the architecture, with shared trunk in gray and the two parallel head paths in color.

---

### Slide 10: Training data and three-part loss

**Thesis:** 30 *E. coli* runs across 3 instruments and 5 dies, held out by die for cross-hardware generalization testing. Three-part loss combining one primary supervision signal (reference-anchored spatial distance) with two auxiliary constraints (detection and velocity consistency).

**SECTION 1 -- DATASET FUNNEL**

30 existing *E. coli* BssSI runs from production data collection. ~3.2M total molecules → ~1.8M clean (not structured, folded, or rejected) → 600K-800K pass all filters (remapped to reference, ≥ 8 matched probes). Reference: *E. coli* K-12 MG1655 (4,641,652 bp, ~810 BssSI sites). 900 GB of raw TDB compresses to ~5 GB of preprocessed cache (waveforms + ground truth + conditioning).

**SECTION 2 -- HELD OUT BY DIE, NOT BY MOLECULE**

Train: dies H09, B23, G16 across instruments 202 and 433 (~24 runs). Validation: die D08 (separate instrument). Test: die C12 (separate instrument). The model never sees molecules from D08 or C12 during training. Splitting by molecule would test only intra-die generalization (trivially easy because all molecules share hardware quirks). Splitting by die forces the real test: does the model work on a new detector without retraining?

**SECTION 3 -- THE THREE-PART LOSS**

`L_total = L_probe + λ_bp · L_bp + λ_vel · L_velocity`

- **L_probe** -- focal loss on the heatmap. Down-weights easy negatives, focuses gradient on ambiguous tag boundaries. Dynamic Gaussian sigma tied to measured tag duration.
- **L_bp** -- sparse Huber (δ = 500 bp) on inter-probe deltas from the reference genome. Primary supervision signal. Robust to outliers.
- **L_velocity** -- sparse L2 at each tag against `511 / duration`. Physics-consistency constraint; auxiliary signal. Subject to the planned ablation.

**SECTION 4 -- WARMUP, AUGMENTATIONS, AND THE PHYSICS WE RESPECT**

Linear warmup of `λ_bp` and `λ_vel` from 0 to 1 over epochs 1-5. End-to-end multi-task training from epoch 1 -- no sequential phasing (which would cause encoder feature starvation). Augmentations: Gaussian noise scaled to baseline RMS, ±5% amplitude scaling, uniform time-stretching (factor 0.9-1.1, physics-consistent: scale velocity targets by 1/k while bp targets stay fixed, since time × velocity = distance). **Not used: time-flipping**. Entry and exit hydrodynamics are asymmetric; flipping would place the trailing whip at t=0, violating the physics we are trying to learn.

**Visual:** Two-column layout. Left: dataset funnel diagram (3.2M → 1.8M → 600K) on top, train/val/test die assignment table below. Right: loss formula with three components broken into labeled boxes, plus a small warmup schedule sparkline.

---

### Slide 11: Evaluation -- two metrics, plus diagnostics

**Thesis:** V1 success requires beating legacy T2D on both a pure ML metric (accuracy) and a downstream business metric (rescued molecules). We do not gate against legacy T2D output -- the goal is to succeed on molecules legacy gets wrong.

**SECTION 1 -- PHASE 1: ML METRIC**

On molecules from the held-out die D08, compute inter-probe intervals from both our model and from legacy T2D. Compare both to *E. coli* reference distances. Report median absolute error in bp, plus error distribution (histogram, tail statistics, per-interval-size breakdown).

We do not compare our model to legacy T2D directly -- we compare both against the reference. If they disagree, we want to know which is closer to truth, not just which one they agree with each other. Success target: at least 5-10% MAE reduction with tight bootstrap confidence interval. Tighter variance is also a positive signal.

**SECTION 2 -- PHASE 2: BUSINESS METRIC (RESCUED MOLECULES)**

Today, a substantial fraction of molecules fail to remap. Some failures are fundamentally unusable data; some, we hypothesize, are because legacy T2D distorts probe spacing badly enough that the aligner cannot find a match. The business metric: take molecules the aligner rejected when using legacy T2D coordinates, re-process through our model, feed the new coordinates back to the aligner, measure how many now successfully map. A 5% rescue rate represents a 5% effective throughput increase achieved purely through software.

**SECTION 3 -- WHAT WE EXPLICITLY DO NOT DO**

We do not gate the model's output against legacy T2D output. The whole point of this project is that the new model should succeed on molecules where legacy T2D gets the answer wrong. Using legacy as a quality gate would reject the new model's best contributions as anomalies. We gate against absolute physical limits (e.g., reject if predicted macro-velocity exceeds theoretical maximum electrophoretic speed), not against agreement with the prior model.

**SECTION 4 -- DIAGNOSTICS BUILT INTO EVALUATION**

Two metrics that are not pass-fail but inform the V2 scope decision:

- **Peak-count discrepancy** -- compare model-detected peaks per molecule to wfmproc-matched reference probes. Mean above zero suggests the model is detecting tags wfmproc missed; below zero suggests we are constrained by weak supervision from wfmproc labels.
- **Lambda-vel ablation** -- train a parallel model with `L_velocity` disabled. If MAE is equivalent, the reference anchor is doing all the work and we can drop the wfmproc-duration dependency. If full loss wins, `L_velocity` is contributing despite weak supervision.

**Visual:** Two-column layout. Left: side-by-side MAE histogram for T2D U-Net (shifted left) vs legacy T2D. Right: bar chart showing legacy mappable molecule count + a green "rescue" increment on top. Footer band shows the two diagnostic metric icons.

---

### Slide 12: Status, scope, and ask

**Thesis:** Software is complete and tested (124 passing tests). The blocker is cloud compute. V1 is a deliberately narrow, ~$75 experiment with a binary success criterion to inform the V2 decision.

**SECTION 1 -- STATUS: SOFTWARE COMPLETE**

The entire stack from binary parsers through evaluation is implemented and tested. **Built:** TDB binary reader, probes.bin parser, assignment parser, reference-map parser, transform parser; ground truth builder with shift-invariant deltas; full 1D U-Net with FiLM and bifurcated decoder; three-component loss with warmup scheduler; preprocessing pipeline (900 GB → 5 GB); training loop with mixed precision and checkpoint-based recovery; inference pipeline with velocity-adaptive NMS and sub-sample interpolation; evaluation comparing against legacy T2D with peak-count and ablation diagnostics. **Pending:** cloud training run, evaluation on held-out die D08, V2 scope decision. Software is green; the blocker is compute.

**SECTION 2 -- V1 SCOPE (DELIBERATELY NARROW)**

**In scope:** time-to-distance conversion (the core claim), probe detection as a secondary head trained on wfmproc labels with reference-genome anchor as corrective signal, cross-die generalization via level-1 normalization and FiLM on physical observables.

**Out of scope:** molecule detection and chunking (kept as wfmproc responsibility), self-supervised probe detection independent of wfmproc, on-instrument inference and real-time deployment, cross-organism generalization (human DNA -- requires new data collection and re-training).

**Why narrow scope:** failure attribution. A broad V1 that fails has ambiguous failure modes -- could be detection, integration, conditioning. A narrow V1 either beats legacy on MAE or it doesn't. Clean signal for the V2 decision.

**SECTION 3 -- THE CLOUD TRAINING ASK**

**Compute:** NVIDIA L4 GPU on GCP at spot pricing (~$0.21/hour), single g2-standard-4 instance, 24 GB VRAM (comfortably fits batch 8-16 with mixed precision). ~130 hours total compute across the baseline V1 run, the L_velocity ablation, and 1-2 hyperparameter sweep iterations.

**Storage:** preprocess locally, transfer only the 5 GB cache. Minimal cloud storage cost.

**Total:** ~$75 for the full development cycle. Spot-preemptible with full checkpoint-based recovery -- preemptions are harmless.

**Timeline:** Week 1: local preprocessing, sanity checks, GCP setup. Weeks 2-3: V1 training, evaluation on D08, ablation. Week 4: analysis, V2 decision, final report.

**SECTION 4 -- THE DECISION FRAMING**

This is not a fund-the-research ask -- the research is done. This is a $75 experiment with a binary success criterion to produce the data that tells us whether V2 is worth pursuing. Even a failed V1 produces useful information through the diagnostic metrics: we will know whether the problem is supervision quality, data quantity, model capacity, or physics approximation, and which V2 direction to invest in next.

**Visual:** Three-column layout. Left: status checklist with green/yellow markers. Middle: V1 in/out scope table. Right: large "$75" callout with timeline gantt below it.

---

## Visual Style Guidance for Cowork

- **Color palette:** Match the existing v1 deck (clean, muted teal/green accents on light background). Don't reinvent the visual identity.
- **Typography:** Slide titles 28-32pt. Thesis line 14-16pt italic or color-accent. Section headers (small caps): 11-12pt. Body text: 10-11pt. Minimum 10pt -- do not go smaller.
- **Section labels (small caps):** Use as visual anchors so dense text remains scannable. Each section label sits above 2-4 sentences of body text.
- **Avoid bullet lists where prose works better.** This deck is intentionally text-rich. Use bullets only for true enumerations (list of three effects, list of FiLM values, etc.). Prefer paragraph-style content for explanations.
- **Visuals are illustrative, not load-bearing.** Each slide has one supporting visual element, but the substance is in the text. Don't over-design the visuals.
- **Slide footer:** Section number + slide title (e.g., "Section 2 · The Approach · Predict velocity, integrate deterministically · Slide 4 of 12"). Helps with navigation when the deck is read asynchronously.
- **Page numbers:** Include them. This deck may be circulated as a PDF.

## Output File

Name the new file `Project_Mongoose_Review_v2.pptx` (or similar; do **not** overwrite the existing v1 deck).

16:9 aspect ratio. PowerPoint format.
