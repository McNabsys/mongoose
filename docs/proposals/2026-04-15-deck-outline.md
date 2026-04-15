# Project Mongoose: Deck Outline (for Cowork)

**Purpose:** Internal review deck for Project Mongoose, the deep learning replacement for the legacy T2D (time-to-distance) model. This outline is authoritative for technical content; Gemini's prior deck can be used only for visual style reference.

**Audience:** Internal Nabsys technical review. Mix of signal processing engineers, bioinformaticians, and leadership. Assume familiarity with the Nabsys detector and the Lunch & Learn reference (Oliver 2025-05-28).

**Target length:** 14-16 slides.

**Tone:** Honest about scope. This is a V1 scientific experiment, not a finished product. No marketing superlatives. No aspirational capabilities presented as facts.

---

## Critical Guardrails for Cowork

Do **NOT** include any of the following claims (these appeared in the prior Gemini deck and are inaccurate or aspirational):

- Zero-shot human deployment. The E. coli model has not been tested on human DNA.
- "Without a single retraining epoch." Not demonstrated.
- "Endogenous quality canary" as a delivered feature. Not implemented.
- "Stalling instrument throughput." The legacy T2D works; it has accuracy limits.
- Claims that the model replaces the entire signal processing pipeline. V1 replaces only the T2D step.
- Human DNA framing (50 kbp mega-deserts). V1 trains on E. coli where typical deserts are ~15 kbp.

Use the exact terminology specified below. In particular, call the architecture the **T2D U-Net**, not "Kinematic U-Net".

---

## Deck Structure

### Section 1 -- The Problem (3 slides)

---

**Slide 1: Title**

- **Title:** Project Mongoose: A Physics-Informed Deep Learning Model for Time-to-Distance Conversion
- **Subtitle:** Replacing the parametric drag model in the Nabsys signal processing pipeline
- **Footer tags:** Internal Review - V1 Architecture - April 2026
- **Visual:** Raw TDB voltage trace transitioning into a base-pair coordinate axis, with the current T2D equation `L = C * t^alpha` shown as a dashed line that diverges from the true signal

---

**Slide 2: The T2D transform is the accuracy ceiling, not the detector**

- **Assertion:** The detector produces clean, high-bandwidth electrical signatures. The limitation is in the time-to-distance math that converts these signals into base-pair coordinates.
- **Bullets:**
  - Current T2D model: `L = C * (t + offset)^alpha`, fitted per channel
  - 3 tunable parameters per channel; tens of thousands of channels across the fleet
  - Every downstream step (remapping, structural variant calling, coverage estimation) inherits T2D error
  - Empirical observation: trailing-end velocity is ~3.5x faster than mid-molecule velocity; leading-end shows a head-dive burst the parametric model cannot represent
- **Visual:** Measured probe width vs. molecule position plot from the M1_probeWidths data, annotated to show where the parametric model diverges from observation
- **Speaker note:** This frames the problem accurately -- we are not claiming throughput is stalled; we are claiming accuracy is capped.

---

**Slide 3: Why the parametric model has a physics ceiling**

- **Assertion:** The legacy model assumes a single drag mode with three fitted parameters. Real nanochannel physics includes phenomena no three-parameter curve can capture.
- **Three effects the parametric model cannot represent:**
  1. **Head dive:** Leading-edge DNA enters with a brief acceleration burst before drag sets in
  2. **Tag velocity compensation (TVC):** Dense tag clusters alter local drag, producing localized deceleration
  3. **Channel drift within a run:** NaOH etching, clogging, and concentration polarization shift the effective velocity over 12-24 hour runs
- **Visual:** Three small panels, one per effect. Head dive: zoom on leading edge showing fast->slow transition. TVC: zoom on cluster showing local dip. Drift: macro plot of velocity shifting over run duration.
- **Speaker note:** These are not edge cases. They are present in essentially every molecule. The parametric model averages them out and absorbs the residual into noise.

---

### Section 2 -- The Approach (4 slides)

---

**Slide 4: A 1D U-Net that learns velocity directly from raw waveforms**

- **Assertion:** We replace the parametric drag equation with a neural network that reads the raw 40 kHz voltage trace and outputs an instantaneous velocity field. Integration of that velocity field produces the spatial coordinate.
- **Inputs:** Per-molecule raw Int16 waveform, normalized by level-1 backbone amplitude
- **Outputs (two heads):**
  - Probe heatmap (per-sample probability of tag center)
  - Cumulative base pairs (per-sample spatial coordinate)
- **What is not changed:** Molecule detection and level-1 estimation are provided by the existing wfmproc pipeline. V1 replaces only the T2D step.
- **Visual:** Waveform in, U-Net diagram (abstracted), two output curves. Clearly label the scope boundary between wfmproc and the new model.

---

**Slide 5: Predict velocity, integrate deterministically**

- **Assertion:** The most important architectural decision is what the network outputs. Predicting cumulative base pairs directly would be brittle. Predicting velocity with a forced-positive activation and integrating deterministically is robust.
- **The architectural primitive:**
  - Network output: local velocity in bp/sample
  - Softplus activation ensures strictly positive (DNA always moves forward)
  - `torch.cumsum` deterministically converts velocity to cumulative bp
  - No learnable weights in the integration step
- **Why this matters:** 1D CNNs are naturally good at local stationary properties (instantaneous velocity) and naturally bad at non-stationary accumulators (cumulative distance). This decomposition plays to the network's strengths.
- **Side effect:** Monotonicity of the spatial output is guaranteed by construction, not learned. The model cannot output DNA traveling backward in time.
- **Visual:** Three stages -- raw velocity tensor, softplus activation curve, cumsum producing monotone spatial curve. Label the cumsum box as "parameter-free."

---

**Slide 6: The chicken-and-egg problem and how we solved it**

- **Assertion:** Training a replacement for the legacy T2D is contaminated if the training labels were themselves produced by the legacy T2D. We solved this by anchoring ground truth to the E. coli reference genome, not to legacy T2D output.
- **The problem:** Legacy T2D decides where each probe "is" in bp space. Training a model to match those positions teaches it to replicate the legacy model's biases.
- **The solution -- shift-invariant inter-probe deltas:**
  - For each molecule mapped to E. coli K-12 MG1655, we know the exact reference base-pair positions of each matched probe
  - We train on `GT_deltas = abs(diff(reference_bp_positions))` -- the true physical distance between consecutive probes
  - The model predicts cumulative bp; we compute `pred_deltas = diff(pred_cumulative[probe_indices])` and compare to GT_deltas with a Huber loss
  - Taking differences makes the loss shift-invariant to unknown molecule end positions (the "blind overhang" problem)
- **Why this matters:** The reference genome is the only unassailable source of spatial truth. Anchoring there is what makes V1 more than a clone of the legacy model.
- **Visual:** Molecule with probes above, reference genome slice with known bp positions below, arrows connecting matched probes. Overlay "abs(diff)" operation producing the GT delta array.

---

**Slide 7: The 511 bp tag as a self-calibrating velocimeter**

- **Assertion:** Every detected tag is a physical ruler. Because each tag occupies a constant 511 bp of DNA, its temporal width in the signal is a direct measurement of local velocity at that point on the molecule.
- **The physics:** `local_velocity = 511 bp / tag_temporal_width`
- **Empirical validation (from our clean E. coli molecules):**
  - Leading edge (first 10% of molecule): ~430 kbp/s
  - Middle (40-60%): ~341 kbp/s (slowest - maximum remaining drag)
  - Trailing edge (last 10%): ~1,487 kbp/s (fastest - minimum drag)
- **How the model uses this:** An auxiliary loss `L_velocity` teaches the model to predict local velocity consistent with the 511 bp / tag_width relationship at every detected probe position. The reference-anchored `L_bp` is the dominant supervision signal; `L_velocity` is the physics-consistency constraint.
- **Visual:** Plot of median tag duration vs. position along molecule (from M1_probeWidths-style data), annotated with the three velocity regimes.

---

### Section 3 -- The Architecture (3 slides)

---

**Slide 8: The T2D U-Net: 5-level encoder, hybrid bottleneck, bifurcated decoder**

- **Assertion:** The architecture is a 1D U-Net (~15M parameters) with three key departures from a vanilla U-Net, each motivated by a specific failure mode we would otherwise hit.
- **Encoder:** 5 levels, 2x downsampling per level, channels [32, 64, 128, 256, 512], kernel size 7.
- **Bottleneck:** Dilated convolution cascade `[1, 2, 4, 8]` plus 1D Multi-Head Self-Attention (4 heads). Handles probe deserts via receptive field expansion + global context.
- **Decoder:** Symmetric 5 levels with skip connections to full resolution (1x). Bifurcates at full resolution into two heads (sharp probe detection vs. smooth velocity).
- **FiLM conditioning:** 6-value physical observable vector modulates features at encoder level 0 and at the bottleneck.
- **Visual:** Complete U-Net diagram showing all 5 encoder levels, the dilated/MHSA bottleneck, 5 decoder levels, bifurcation at 1x, and FiLM injection arrows from a conditioning block.

---

**Slide 9: Receptive field must span the biggest probe desert**

- **Assertion:** Between clusters of tags, E. coli molecules contain long "deserts" with no probes. The model must still predict physically plausible velocity through these gaps. The effective receptive field must be wide enough that the flanking probes on either side land in the heavily weighted center of the receptive field.
- **The worst-case math:**
  - A false-negative missed tag merges two ~12 kbp gaps into a ~20 kbp gap
  - At slow local velocity (~200 kbp/s), that is ~4,000 samples
  - Theoretical RF must exceed this; effective RF (ERF) is Gaussian-weighted, so target ~1.5x the gap
- **The solution (at the bottleneck, where sequences are compressed 32x):**
  - Dilated convolution cascade (rates 1, 2, 4, 8) expands theoretical RF beyond 4,500 samples
  - 1D self-attention adds global context essentially for free at 125-625 bottleneck tokens
- **Visual:** Desert illustration with probes on either side, receptive field curve showing Gaussian-weighted ERF, annotations for dilated cascade expansion and MHSA global link.

---

**Slide 10: Two heads, two frequency regimes, full resolution for both**

- **Assertion:** Probe detection and velocity prediction have opposite temporal requirements. They must share the encoder but diverge in the decoder's last stage.
- **Probe head (kernel size 7):** High-frequency, localizes tags to within 1-2 samples. Outputs a Gaussian-like heatmap at full 40 kHz resolution.
- **Velocity head (kernel size 31):** Wide kernels act as learned low-pass filters on the velocity output. Must preserve localized cliff-edge decelerations caused by bulky tags plugging the channel. Runs at full 1x resolution (not downsampled) to avoid aliasing the 511 bp tag-induced drag signature.
- **FiLM conditioning vector (6 values):**
  - Absolute pre-event baseline (buffer conductivity + pore geometry proxy)
  - log(molecule duration in samples) (macro-scale drag prior)
  - log(inter-event interval + 1) (pore rested vs. exhausted from concentration polarization)
  - Time-in-run fraction (Joule heating, viscosity drift)
  - Applied bias voltage (electric field drives velocity)
  - Applied pressure (pneumatic back-pressure)
- **Visual:** Side-by-side narrow vs. wide kernel illustration, then the 6-value FiLM vector feeding into an MLP that outputs gamma/beta modulation at two injection points.

---

### Section 4 -- Training and Evaluation (3 slides)

---

**Slide 11: Training data -- 30 E. coli runs, held-out by die**

- **Assertion:** V1 trains on 30 E. coli BssSI runs across 3 instruments and 5 dies. The test set is held out by die, not by molecule, to force out-of-distribution generalization at the hardware level.
- **Dataset:**
  - ~3.2M total molecules, ~1.8M clean (not structured, folded, or rejected)
  - ~600K-800K molecules pass all filters (remapped to E. coli reference, >=8 matched probes)
  - Reference: E. coli K-12 MG1655 (4,641,652 bp, ~810 BssSI probe sites)
- **Split (held out by die):**
  - Train: H09, B23, G16 (~24 runs, 3 dies, 2 instruments)
  - Validation: D08 (separate instrument)
  - Test: C12 (separate instrument)
- **Preprocessing:** 900 GB of raw TDB files compresses to ~5 GB of clean molecule waveforms + ground truth + conditioning vectors
- **Visual:** Table of runs by instrument and die, colored by train/val/test split. Small barchart showing total-to-usable molecule funnel.

---

**Slide 12: Three-part loss, anchored to physics and to reference**

- **Assertion:** The total loss combines one primary supervision signal (reference-anchored spatial distance) with two auxiliary constraints (detection and velocity consistency). All three are tuned to operate at compatible scales.
- **The three components:**
  - `L_probe` (focal loss) on the probe heatmap, with dynamic Gaussian sigma tied to measured tag duration
  - `L_bp` (sparse Huber loss, delta=500 bp) on inter-probe deltas from the reference genome
  - `L_velocity` (sparse L2 loss) at each tag position against `511 bp / tag_duration`
- **Loss combination:** `L_total = L_probe + lambda_bp * L_bp + lambda_vel * L_velocity`
- **Warmup:** `lambda_bp` and `lambda_vel` ramp linearly from 0 to target over epochs 1-5. Single curriculum, end-to-end from epoch 1; no sequential training phases.
- **Augmentations:** Gaussian noise, amplitude scaling, uniform time-stretching (simulates voltage/viscosity drift). No time-flipping -- entry and exit hydrodynamics are asymmetric.
- **Visual:** Three loss boxes on the left feeding into the total loss. Small inset showing the warmup schedule. Augmentation examples across the bottom.

---

**Slide 13: Evaluation -- two metrics, not one**

- **Assertion:** V1 success requires beating legacy T2D on both a pure ML metric and a downstream business metric. We explicitly avoid using legacy T2D output as a quality gate, because the goal is to rescue molecules legacy T2D gets wrong.
- **ML metric (Phase 1):** Median absolute error on inter-probe intervals, held-out die D08, measured against E. coli reference genome distances. Computed for both the T2D U-Net and legacy T2D on the same molecules.
- **Business metric (Phase 2):** Feed molecules that legacy T2D failed to map into the Nabsys aligner using T2D U-Net output. Measure the additional yield (rescued molecule fraction). A 5% rescue rate on previously unmappable molecules is a meaningful throughput increase.
- **Diagnostics added to evaluation:**
  - Peak-count discrepancy: does the model detect more/fewer probes than wfmproc matched?
  - Lambda-vel ablation: does training without `L_velocity` reach equivalent MAE? (Tests our wfmproc-duration dependency.)
- **Visual:** Two-column comparison: left column "ML metric" with a MAE histogram overlay, right column "business metric" with a rescued-molecule yield bar.

---

### Section 5 -- Status, Scope, and Ask (2-3 slides)

---

**Slide 14: Current status -- full pipeline built, training pending**

- **Assertion:** The entire software stack from binary parsers through evaluation is implemented and tested. What remains is cloud compute to run the actual training.
- **What is built (124 passing tests):**
  - TDB binary reader, probes.bin parser, assignment parser, reference map parser, transform parser
  - Ground truth builder with shift-invariant deltas
  - 1D U-Net model with FiLM conditioning and bifurcated decoder
  - Focal + sparse Huber + sparse L2 losses with warmup scheduler
  - Preprocessing pipeline (900 GB -> 5 GB cache)
  - Training loop with mixed precision and checkpointing
  - Inference pipeline with velocity-adaptive NMS and sub-sample interpolation
  - Evaluation comparing against legacy T2D with peak-count diagnostics
- **What is pending:** Cloud training run, evaluation against held-out die, V1 vs legacy comparison, decision on V2 scope.
- **Visual:** Component status grid, color-coded green for complete, yellow for pending. Optionally a test-count badge.

---

**Slide 15: V1 scope is deliberately narrow**

- **Assertion:** V1 proves that learned velocity integration beats parametric T2D. It does not attempt to replace the full signal processing pipeline. This scope choice is about failure attribution, not technical limitation.
- **In scope for V1:**
  - Time-to-distance conversion (the core claim)
  - Probe detection as a secondary head (trained on wfmproc labels, with reference-genome anchor correcting small errors)
  - Cross-die generalization via level-1 normalization and FiLM on physical observables
- **Out of scope for V1 (future work):**
  - Molecule detection and chunking (kept as wfmproc responsibility)
  - Self-supervised probe detection independent of wfmproc labels
  - On-instrument inference and real-time deployment
  - Cross-organism generalization (human DNA deployment)
- **Why narrow scope:** If a broad V1 fails, failure is ambiguous (detection? integration? which?). A narrow V1 either beats legacy on MAE or it does not. Clean signal for the V2 decision.
- **Visual:** Two-column "V1 / V2+" table mapping each pipeline stage to its V1 ownership and future roadmap.

---

**Slide 16: Cloud training ask**

- **Assertion:** V1 training requires a single GPU instance on GCP. Estimated total cost is ~$75 for a full development cycle including hyperparameter sweeps and the V1 vs V1-no-velocity ablation.
- **Compute profile:**
  - NVIDIA L4 (24 GB VRAM) on spot pricing (~$0.21/hour)
  - Single g2-standard-4 instance
  - ~130 hours total compute across multiple training runs
- **Storage profile:**
  - Preprocess locally (convert 900 GB TDB to 5 GB cache)
  - Transfer only 5 GB to cloud
  - Small data disk sufficient
- **Timeline:**
  - Week 1: Local preprocessing, sanity checks, GCP setup
  - Weeks 2-3: V1 training + evaluation + ablation
  - Week 4: V2 scope decision, final report
- **Total budget request:** Approximately $75, spot-preemptible with full checkpoint-based recovery built in.
- **Visual:** Simple cost/timeline gantt or breakdown table.

---

## Style Notes for Cowork

**Visual language:**
- Follow the aesthetic of the prior Gemini deck: clean, technical diagrams, muted green/teal accent color, light background
- Retain the visual structure of panels with "The Problem / The Solution / The Result" where it naturally applies
- DNA helix motifs are fine for Section 1 but should not clutter technical architecture slides

**Title style:**
- Assertive, declarative titles (not topic labels)
- Example good: "Predict velocity, integrate deterministically"
- Example bad: "Architecture"

**Terminology to use consistently:**
- T2D U-Net (not Kinematic U-Net)
- Time-to-distance (T2D) conversion (not "DNA kinematics")
- Physics-informed neural network (PINN) -- use sparingly
- wfmproc (for the legacy signal processing)
- E. coli K-12 MG1655 on first reference, E. coli thereafter
- BssSI (not BspQI -- the E. coli runs use only BssSI)
- Base pairs (bp) -- lowercase in units, standard scientific style
- Level-1 normalization (hyphenated)

**Numbers to reference:**
- 511 bp tag width (the physical invariant)
- ~810 BssSI probe sites on E. coli
- 40 kHz sample rate (25 us period)
- ~15M model parameters
- 5-level encoder, 32x downsampling at bottleneck
- 6-value FiLM conditioning vector
- 30 runs, 3 instruments, 5 dies in training data

**Things to never say:**
- "Zero-shot"
- "Universal"
- "Hardware-agnostic"
- "Endogenous quality canary"
- "Without retraining"
- "Hallucinating backward-traveling DNA" (colorful but overly casual)
- "Mega-deserts" (informal)
- "Stalling throughput"
