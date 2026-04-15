# Project Mongoose: Deck Talking Points

**Purpose:** Presenter notes for Slides 4-16 (Approach, Architecture, Training/Eval, Status and Ask). Companion document to `2026-04-15-deck-outline.md`. Each slide gets opening transition, core content in presenter voice, anticipated questions, and timing guidance.

**Slides 1-3 (Problem Setup)** are not covered here -- they are introductory and do not require the same level of presenter scripting.

---

## Section 2 -- The Approach

---

### Slide 4: A 1D U-Net that learns velocity directly from raw waveforms

**Opening (transition from Section 1):**

> "Slides 1 through 3 established that the legacy T2D model has a physics ceiling -- three fitted parameters per channel can't capture head dive, tag velocity compensation, or channel drift. That's the problem. The approach we're proposing is to replace those three parameters with a 1D U-Net that learns the same transform directly from raw voltage data."

**What the model does (inputs and outputs):**

> "The model is a sequence-to-sequence network. It reads in one input -- the raw 40 kHz voltage trace for a single molecule, normalized by its level-1 backbone amplitude -- and produces two outputs at the same 40 kHz resolution."

> "The first output is a probe heatmap: at each sample in the waveform, the model emits a probability that a tag is centered at that point. This replaces what wfmproc's peak-picker does today."

> "The second output is cumulative base pairs: at each sample, the model emits the spatial coordinate in bp measured from the molecule's start. This replaces what the legacy T2D equation does today. The inter-probe distances we actually care about downstream are just the differences between the cumulative bp values at consecutive probe positions."

**The scope boundary (critical point -- do not rush):**

> "I want to be explicit about scope, because this is the most common question. The model does not replace wfmproc's molecule detection. It does not replace level-1 estimation. It does not replace the rising/falling edge convolutions that chunk the raw TDB stream into per-molecule windows. All of that stays as it is today."

> "The V1 model takes the per-molecule waveform that wfmproc already produces and replaces exactly the T2D step. This is a deliberate scope choice, not a technical limitation. If V1 fails, I want to know that T2D math was the problem, not that we simultaneously broke molecule detection."

> "We have a V2 roadmap that extends scope toward a unified model. But V1 has to prove the T2D claim first."

**Why a U-Net specifically:**

> "A few words on why a U-Net is the right architectural family. This is fundamentally a sequence-to-sequence task -- input and output have the same temporal resolution, both are 1D. U-Nets were designed for this class of problem."

> "The encoder-decoder structure matters because velocity prediction has two frequency scales simultaneously. Global macro-velocity -- how fast is the molecule translocating overall -- depends on the total remaining drag, which requires seeing a long temporal window. Localized decelerations from tag-induced drag happen over a few samples. The U-Net's deep path captures the macro context; the skip connections preserve the high-frequency detail. You get both, which is hard to do with a pure CNN or a pure transformer."

**The physics-informed qualifier:**

> "We call this a physics-informed neural network rather than just a neural network because we've baked several physical constraints directly into the architecture -- not learned from data, but imposed structurally. Those are coming up in slides 5 through 7."

> "The short version: the model cannot output DNA traveling backward in time. It cannot predict spatial coordinates that decrease with time. Those are architectural guarantees. The network is free to learn everything else."

**Setting up what's next:**

> "The next three slides unpack the three design decisions that make this work. Slide 5 is about what the network actually outputs, which is the most consequential architectural choice. Slide 6 is about how we avoid training the new model to replicate the old model's biases. Slide 7 is about the physical invariant we discovered in the signal that gives us an extra supervision channel."

**Anticipated questions:**

- **"Why not just a transformer?"** -> "A transformer on 4,000-20,000 tokens per molecule is feasible, but the bottleneck of this problem is the receptive field across probe deserts, which a U-Net handles efficiently with hierarchical downsampling. We do use self-attention -- but only at the bottleneck layer, where it's cheap."
- **"Why one molecule at a time, not the full TDB stream?"** -> "That's the V2 question. V1 takes per-molecule chunks because that's what wfmproc already produces. A streaming model that ingests full TDB is a larger scope change."
- **"What's 'level-1 normalized' doing for us?"** -> "That's coming up on slide 7 in more detail. Short answer: it neutralizes most of the cross-channel geometric variance so the network learns one universal waveform shape regardless of which die the molecule came from."
- **"How big is the model?"** -> "About 15 million parameters, which is small by modern ML standards. Fits comfortably on a single L4 GPU."

**Timing:** 3-4 minutes. The scope boundary paragraph is the most important content.

---

### Slide 5: Predict velocity, integrate deterministically

**Opening:**

> "This slide covers what I think is the single most important architectural decision in this project. It's subtle, but it's what separates a model that works from one that doesn't. The question is: what should the network actually output?"

**The naive approach and why it fails:**

> "The obvious choice is to have the network directly output cumulative base pairs -- that's what we want, so why not predict it? The problem is that cumulative bp is a non-stationary, monotonically increasing quantity. A 1D convolutional network is fundamentally translation-equivariant; it's excellent at extracting local, stationary properties, and it's terrible at producing a globally-consistent accumulator."

> "If you train a network to directly output cumulative bp, you get noisy predictions that don't respect monotonicity. The network might output bp positions that decrease from one sample to the next -- which is physically impossible. That would mean the DNA traveled backward in time."

**The alternative and why it works:**

> "Instead, we have the network predict instantaneous velocity at each sample -- how many base pairs of DNA pass through the detector per sample at that moment. This is a local, stationary property. It's exactly what 1D convolutions are good at extracting."

> "Then we stack two operations on top of the velocity output. First, a Softplus activation. Softplus is like ReLU but smooth -- it's strictly positive for any input. This guarantees the predicted velocity is always greater than zero, which means the DNA is always moving forward. Second, a parameter-free cumulative sum -- PyTorch's torch.cumsum operator. Given the per-sample velocity, it returns the running total."

**What this guarantees by construction:**

> "The output is monotonically non-decreasing, by mathematical construction. Not by training. Not as a soft constraint. As a structural property of the architecture. The model physically cannot predict DNA traveling backward. This is what it means to be physics-informed -- we've baked the physical constraint that DNA moves forward into the architecture itself, so the network has one fewer thing to learn."

> "There's a second benefit: the cumulative sum is itself a perfect low-pass filter. Any high-frequency noise in the velocity prediction washes out in the integral. So the network can make noisy local velocity predictions -- which is what CNNs naturally do -- and still produce a smooth, physically plausible cumulative bp curve."

**The subtle consequence:**

> "This decomposition has a specific implication for how we compute the loss. If the ground truth is inter-probe distance -- so many bp between probe A and probe B -- we don't compare the raw velocity against anything. We take the model's cumulative output, sample it at the two probe positions, subtract, and compare that difference to the reference distance. That's coming up on slide 6."

**Anticipated questions:**

- **"Why Softplus and not ReLU?"** -> "Softplus is smooth everywhere. ReLU has a hard zero that kills gradients. For velocity, we want the network to learn that velocity is very small, not exactly zero, between tags. Softplus lets the gradient flow even at small velocities."
- **"Could you predict velocity in physical units -- bp per second?"** -> "Yes, and that's what it is under the hood. We convert between bp/sample and bp/second using the known 40 kHz sample rate."
- **"Does the cumsum operator work with backpropagation?"** -> "Yes, it's fully differentiable. Gradients flow through it cleanly, just with a triangular Jacobian. PyTorch handles it natively."

**Timing:** 3-4 minutes. The key line to deliver slowly is "The output is monotonically non-decreasing by mathematical construction."

---

### Slide 6: Ground truth is anchored to the E. coli reference genome

**Opening:**

> "This slide addresses what was the hardest design question in this whole project -- and it's a question that, if answered wrong, would make this entire effort scientifically meaningless."

**The chicken-and-egg problem:**

> "Think about what we're trying to build: a replacement for the legacy T2D model. Now think about where our training labels would come from in the naive approach. If we use probes.bin as our source of truth -- those probe positions are produced by the legacy T2D. If we use remapping output to say 'this molecule is 100 kbp long' -- that length came from legacy T2D."

> "If we train a neural network to match those labels, we've trained it to replicate the legacy model's biases. We've spent thousands of dollars of compute to build something that is, at best, a slightly faster version of the equation we already have. At worst, we've baked in legacy T2D's errors and called it progress."

> "This is the chicken-and-egg problem: the only available labels come from the thing we're trying to replace."

**The solution -- reference genome as source of truth:**

> "There is exactly one source of spatial ground truth that does not come from legacy T2D: the E. coli reference genome itself. E. coli K-12 MG1655 is a known 4,641,652 base pair sequence. BssSI recognition sites -- where our tags bind -- occur at 810 specific, known positions on that genome."

> "When the Nabsys aligner successfully maps a molecule to a stretch of the reference genome, it's telling us which specific reference probe each detected tag corresponds to. And for each matched reference probe, we know its exact base-pair coordinate. That coordinate comes from the FASTA file, not from any signal processing step."

**Shift-invariant deltas:**

> "We don't train on absolute positions. We train on the differences between consecutive probes -- `abs(diff(reference_bp_positions))`. If the reference says probe A is at position 500,000 and probe B is at position 512,500, the training label for that interval is exactly 12,500 bp. That number is unassailable."

> "Taking differences does two other important things. First, it handles forward and reverse molecule orientations natively. A molecule entering head-first reads the reference in forward order; a molecule entering tail-first reads it in reverse. The absolute value of the difference is the same either way -- distance is always positive."

> "Second, it makes the loss shift-invariant to the 'blind overhang' problem. The physical ends of the molecule -- the bare DNA between the molecule's tip and its first nick site -- don't land on restriction sites. The aligner estimates those tail lengths using legacy T2D. If we anchored our loss to absolute positions, we would inherit that estimate. By training only on inter-probe deltas, we don't care where the molecule starts or ends -- we only care that the distances between the probes we can see match the reference."

**The full loss pipeline:**

> "Putting it together: the model outputs cumulative bp. We sample that output at the sample indices of the detected probes. We take the differences. We compare those predicted differences to the reference deltas with a Huber loss. The gradient flows back through the cumsum, through the velocity predictions, all the way back through the encoder."

> "The reference genome, not the legacy model, is what drives the learning signal."

**Why this is the most important slide in the deck:**

> "Without this design decision, V1 would be an expensive exercise in replicating what we already have. With it, V1 is genuinely capable of learning something the legacy model cannot -- because the reference genome is a better teacher than the legacy model."

**Anticipated questions:**

- **"What about molecules that don't map?"** -> "Those don't contribute to the spatial loss. We can still use them for the probe detection loss, which doesn't require reference coordinates. But for L_bp, only remapped molecules contribute."
- **"What if the aligner is wrong about which reference probe a tag corresponds to?"** -> "We filter to high-confidence alignments only. The E. coli probe pattern is spatially distinctive enough that mismatches are rare when alignment scores are high."
- **"Why not use human DNA?"** -> "Human reference has billions of possible probe patterns, making unique identification of a 100 kbp fragment difficult. E. coli is 4.6 Mbp with a single unique sequence -- the reference mapping is essentially unambiguous. Training on the simpler organism first is the correct sequencing."

**Timing:** 4-5 minutes. Do not rush the chicken-and-egg explanation -- this is where skeptical technical audience members will evaluate whether this project is scientifically serious.

---

### Slide 7: The 511 bp tag as a self-calibrating velocimeter

**Opening:**

> "The reference-anchored ground truth we just discussed gives the model its primary supervision signal. This slide is about a secondary physical constraint we discovered in the signal itself -- something that lets us supervise the model's velocity prediction directly, not just through the bp integral."

**The physical invariant:**

> "Every tag that binds to a nick site occupies exactly 511 base pairs of DNA. That's a chemical constant, not a fitted parameter. It's the same for every tag, on every molecule, on every channel, on every instrument."

> "Now consider what happens as a tag passes through the detector. The tag occupies 511 bp of DNA. If the DNA is translocating at velocity v, the tag takes `511 / v` seconds to pass. We measure this duration directly in the signal -- it's the temporal width of the voltage deflection when the tag goes through. From this, we can compute instantaneous velocity at every tag position: `velocity = 511 bp / tag_duration_in_ms`."

> "Every detected tag is a speedometer reading. Not an estimate -- a direct measurement, anchored in the physical dimensions of the molecule."

**Empirical validation:**

> "We ran this analysis on 31,468 clean E. coli molecules from our existing data. The tag-width measurement produces a velocity profile that directly confirms the biophysics we've been talking about."

> "At the leading edge of the molecule -- first 10% -- median velocity is about 430 kbp/s. At the middle of the molecule, velocity drops to 341 kbp/s. At the trailing edge -- last 10% -- velocity jumps to 1,487 kbp/s."

> "The trailing end is 3.5 times faster than the middle. This matches the viscous drag physics exactly: the trailing end has no remaining DNA to drag, so it translocates at terminal velocity. The middle has the maximum remaining drag. The leading edge shows a small velocity burst that we attribute to the head-dive effect described in the legacy T2D literature."

> "These are not numbers we fitted. These are direct measurements from the signal, enabled by knowing that every tag is 511 bp wide."

**How the model uses this:**

> "We turn this into a third loss term: L_velocity. At each detected probe position, we take the model's velocity prediction -- the raw output before the Softplus and cumsum -- and compare it against `511 / measured_tag_duration`. An L2 loss pulls the predicted velocity toward the physics-consistent value."

> "This is a sparse loss -- it's evaluated only at the tag positions, maybe 15 to 50 points per molecule. But it gives the model direct supervision on the quantity it's actually predicting, not just the integral of that quantity."

**The role of this loss:**

> "L_velocity is an auxiliary constraint, not the primary signal. L_bp -- the reference-anchored inter-probe distance loss -- is the dominant gradient that makes the spatial output correct. L_velocity exists to keep the local velocity predictions physically consistent with the measured tag widths, so the network isn't free to make up any velocity curve it wants as long as the integral happens to land right."

> "We'll run an ablation during training with L_velocity turned off. If the model performs equivalently without it, we know the reference anchor is doing all the work and L_velocity can be dropped. If performance drops, L_velocity is pulling its weight. Either way, the ablation tells us something useful."

**Anticipated questions:**

- **"Is 511 bp exactly constant across tags?"** -> "Yes, within sub-bp variance. The tag chemistry is deterministic -- the same enzyme cuts the same recognition sequence and the same probe complex binds. Biological variance is negligible compared to temporal measurement precision."
- **"What about cluster sites where multiple tags are within 50 bp?"** -> "Good point. Clustered tags merge into a single wider peak in the signal. We exclude those from L_velocity -- the duration there reflects the combined cluster, not a single 511 bp tag."
- **"Does the duration measurement come from wfmproc?"** -> "Yes, and this is a legitimate dependency we've flagged. If wfmproc's logistic-fit duration has systematic error, L_velocity inherits it. The reference-anchored L_bp is what corrects for this. It's also why the ablation matters."

**Timing:** 3-4 minutes. The 3.5x trailing/leading velocity ratio is a great moment to let land -- it's empirical proof that the physics we're modeling is real.

---

## Section 3 -- The Architecture

---

### Slide 8: The T2D U-Net: 5-level encoder, hybrid bottleneck, bifurcated decoder

**Opening:**

> "This slide is the architectural overview. Each component here corresponds to a specific failure mode we would hit without it. I'll walk through each, then slides 9 and 10 go deeper on the two most consequential design choices."

**The high-level structure:**

> "The T2D U-Net is a standard 1D U-Net shape with three targeted modifications. Start with the encoder: 5 levels of 2x downsampling, channel counts going 32, 64, 128, 256, 512. Each level has two residual blocks with kernel size 7. Standard stuff."

> "Then the bottleneck. This is where we depart from a vanilla U-Net, and slide 9 is dedicated to why."

> "Then a symmetric 5-level decoder with skip connections, bringing the representation back to full 40 kHz resolution. But rather than a single output head, the decoder bifurcates at full resolution into two heads with different kernel sizes -- one sharp, one smooth. Slide 10 covers that."

**The parameter count and why:**

> "About 15 million parameters total. For reference, small image U-Nets are typically 10-30 million, large ones are hundreds of millions. We're firmly in the small regime."

> "This is deliberate. The problem does not require a huge model. Each molecule is a relatively simple 1D sequence. What matters is the architectural inductive biases -- the encoder-decoder shape, the skip connections, the dilated bottleneck, the cumsum integration -- not raw parameter count. A larger model would overfit faster and train slower without producing better predictions."

**FiLM conditioning:**

> "One more element to point out here: FiLM conditioning. At two points in the network -- encoder level 0 and the bottleneck -- we inject a 6-value vector of measured physical properties. The vector includes things like the absolute pre-event baseline voltage, the log duration of the molecule, where in the run the molecule appeared. This vector is processed by a small MLP that produces per-channel scale and shift parameters."

> "The purpose is to let the network adapt its behavior based on macro-physics that can't be recovered from the normalized waveform alone. If the fluid has warmed up over a long run and viscosity has dropped, that affects baseline velocity. The FiLM vector tells the network about that shift. Slide 10 goes into the specifics of what's in the vector and why."

**The overall design philosophy:**

> "I want to emphasize that this architecture is not exotic. Every element -- residual blocks, dilated convolutions, self-attention, FiLM conditioning -- is standard and well-understood in the ML literature. We have combined them in a way that matches the specific structure of this problem. There is no novel ML research happening here; what's novel is the problem formulation and the physics-informed constraints."

**Anticipated questions:**

- **"Why 5 encoder levels specifically?"** -> "Each level halves the sequence length. Five levels gives us 32x downsampling at the bottleneck. For a typical 4,000-sample molecule, that's 125 tokens at the bottleneck -- short enough to apply self-attention cheaply. Six levels would make the bottleneck too short to encode meaningful global structure; four levels would make self-attention expensive."
- **"Why kernel size 7 throughout?"** -> "It's a good default for 1D signals at 40 kHz. Large enough to see local structure, small enough not to blur sharp features."
- **"Does inference time matter?"** -> "Yes, for real-time deployment. V1 inference per-molecule on a single GPU is sub-millisecond. A full run's worth of molecules (say, 80,000 molecules) takes about a minute. That's acceptable -- we can run the T2D U-Net as a post-processing step on a batch of finished wfmproc output without holding up the instrument."

**Timing:** 3-4 minutes. This is mostly navigation -- setting up slides 9 and 10. Don't linger on the parameter count discussion.

---

### Slide 9: The receptive field must span the biggest probe desert

**Opening:**

> "The most common failure mode for a naive 1D U-Net on this problem is bridging probe deserts. This slide explains the problem and how our bottleneck design solves it."

**The desert problem:**

> "Between clusters of BssSI sites, E. coli has regions with no probes at all. Typical gaps are a few thousand bp. The worst case gaps reach 15,000 bp. And there's a compounding factor: if wfmproc misses a tag -- a false negative -- that effectively merges two adjacent gaps into one. So we need to design for 20 kbp deserts."

> "Now convert that to samples. A 20 kbp desert at leading-edge velocity of 200 kbp/s spans 100 milliseconds, which at 40 kHz is 4,000 samples. The model must predict smooth, physically plausible velocity through 4,000 consecutive samples where the only signal is the flat normalized backbone."

**Why a naive U-Net fails:**

> "A standard 5-level U-Net with kernel size 7 has a theoretical receptive field around 1,000-2,000 samples. Sounds like it would work, but it doesn't. The key distinction is between theoretical receptive field and effective receptive field."

> "The effective receptive field is Gaussian-weighted -- the samples at the extreme edges of the theoretical window contribute essentially zero gradient to the central prediction. If the flanking probes are at the outer edges of the theoretical receptive field, they're functionally invisible. In the middle of the desert, the model has nothing to anchor to, and it defaults to predicting an average-case velocity. That default error compounds through the integration and corrupts the spatial output for the entire desert."

**The solution, part one -- dilated convolution cascade:**

> "The first fix is in the bottleneck. We stack residual blocks with dilation rates 1, 2, 4, and 8, all using kernel size 7. Each dilated convolution sees a wider span without adding parameters. At the bottleneck -- where sequences are compressed 32x -- the cumulative theoretical receptive field of this cascade exceeds 4,500 input samples. That's beyond our worst-case desert."

**The solution, part two -- self-attention:**

> "The dilated cascade expands local context. We also add a single multi-head self-attention layer at the bottleneck. This gives every token a global view of the entire molecule. For a 4,000-sample molecule, that's 125 bottleneck tokens; for a 20,000-sample molecule, 625 tokens. Self-attention on that many tokens is computationally trivial -- a few milliseconds per molecule."

> "The combination is complementary. The dilated cascade handles mid-range structural context -- what's happening within a few thousand samples. Self-attention handles the true global context -- is this molecule translocating fast overall, and where are the macro-anchors. Together, the model can comfortably interpolate velocity through any desert we've seen in the data."

**Why not just add more encoder levels:**

> "The obvious alternative is just going deeper -- 6 or 7 encoder levels, more downsampling. We considered this and rejected it. At 64x or 128x downsampling, the model loses the temporal resolution needed to represent the 511 bp tag width, which is only about 14 samples at the trailing edge. The tag-induced velocity dips would alias away at deep downsampling."

> "Expanding the receptive field at the bottleneck through dilation and attention is essentially free -- the bottleneck is short and cheap. Expanding through additional downsampling would cost us resolution where we can't afford to lose it."

**Anticipated questions:**

- **"What if there's a 30 kbp desert somewhere we haven't seen?"** -> "Possible. The architecture will still attempt to interpolate, and the error in that region will grow. We've designed for 20 kbp worst case because that's what we see in E. coli data with realistic false-negative rates. Human DNA has larger deserts -- that's a V2 scope issue."
- **"Why only one self-attention layer, not a stack?"** -> "One is enough to let every token see every other token. Adding more doesn't add more context. And the bottleneck is deliberately simple so we don't overfit to idiosyncrasies of the training data."

**Timing:** 3-4 minutes. The "theoretical vs effective receptive field" distinction is the key insight on this slide -- spend time on it.

---

### Slide 10: Two heads, two frequency regimes, full resolution for both

**Opening:**

> "The last architectural decision is how the decoder ends. Our two outputs have opposite frequency requirements. A shared output path would force a compromise that hurts both."

**The two requirements:**

> "Probe detection is a high-frequency task. We need to localize tag centers to within 1-2 samples to support downstream inference. The heatmap has to be sharp -- essentially a narrow Gaussian at each tag. This requires kernels that preserve fast local transitions."

> "Velocity prediction is the opposite. Velocity changes smoothly over long timescales as drag redistributes during translocation. Sharp kernels on the velocity output would produce high-frequency noise that corrupts the cumulative integral. We want wide kernels that act as learned low-pass filters."

**The bifurcation:**

> "The solution is to share the encoder, decoder, and skip connections -- everything up to the point where the representation returns to full 1x resolution -- and then split into two parallel heads."

> "The probe head uses two residual blocks with kernel size 7. Standard width, preserves high-frequency detail, produces a sigmoid-activated heatmap."

> "The velocity head uses two residual blocks with kernel size 31. These wide kernels are learned low-pass filters. They smooth out random electrical noise but, crucially, because they operate at full 1x resolution, they faithfully reproduce the localized velocity dips caused by bulky tags physically plugging the channel."

**Why full resolution for both:**

> "We considered downsampling the velocity branch -- running it at 8x instead of 1x would save compute and smooth the output for free. We rejected this because of aliasing. At 8x downsampling, a trailing-edge 511 bp tag is compressed to under 2 tokens. The tag-induced velocity dip becomes unrepresentable. Upsampling smears that dip across 32 input samples, which produces a gradient conflict with the sparse L_velocity supervision at exactly those sample indices."

> "Full resolution costs compute, but it's the only way to make the velocity branch's learned smoothing compatible with the tag-width physics supervision."

**The FiLM conditioning vector, in detail:**

> "I promised on slide 8 to come back to the FiLM conditioning vector. It has six values, each chosen for a specific physical reason."

> "Value 1: the absolute pre-event baseline voltage, measured from the samples just before the molecule entered the channel. This is our proxy for buffer conductivity and effective pore diameter. Level-1 normalization erases this information in the waveform; FiLM restores it as a conditioning signal."

> "Value 2: log of the molecule's duration in samples. This gives the network a macro-scale prior on drag. A longer molecule experiences more total drag than a shorter one because there's more remaining mass for most of the translocation."

> "Value 3: log of the inter-event interval -- how long since the last molecule passed this channel. Nanochannels suffer from concentration polarization; a channel needs milliseconds to recover ion balance after a translocation. A channel that just saw a molecule ten milliseconds ago is 'exhausted' compared to one that's been idle for a second."

> "Value 4: time-in-run as a fraction from 0 to 1. Over a 24-hour run, Joule heating warms the buffer, viscosity drops, and velocity systematically shifts upward. This feature teaches the model about thermodynamic drift."

> "Value 5: applied bias voltage. Velocity is linearly proportional to electric field. If different protocols use different bias, the model needs to know."

> "Value 6: applied pressure. Some protocols add pneumatic back-pressure that modulates velocity. Same reason."

> "The vector is processed by a small MLP that produces scale and shift parameters for feature modulation at encoder level 0 and at the bottleneck. The network learns to adjust its convolutional features based on the macro physics of the current molecule."

**Why no categorical channel ID:**

> "The obvious alternative is to just feed the model a channel identifier and let it memorize each die's quirks. We explicitly rejected that. Categorical channel IDs fail zero-shot on new dies, new instruments, new wafers. Feeding continuous physical observables instead lets the model learn generalizable physics that transfer to unseen hardware."

**Anticipated questions:**

- **"Do all six FiLM values come from the raw TDB?"** -> "Baseline and molecule duration come from the TDB and probes.bin. Time-in-run and inter-event interval come from timestamps. Bias and pressure come from run metadata. None require legacy T2D."
- **"What if a value is missing for an older run?"** -> "We default to a sensible value. The network learns to be insensitive to missing conditioning on the handful of runs where metadata is incomplete."

**Timing:** 4-5 minutes. The FiLM vector discussion is long but worth the time -- it's what enables cross-hardware generalization.

---

## Section 4 -- Training and Evaluation

---

### Slide 11: Training data -- 30 E. coli runs, held out by die

**Opening:**

> "The training data is 30 existing E. coli BssSI runs. We have not collected new data for this project; we're using standard production runs that were collected for other purposes and are available in the existing file shares."

**The dataset funnel:**

> "Total molecules across the 30 runs: about 3.2 million. After quality filtering -- excluding structured, folded, or rejected molecules -- about 1.8 million remain. Of those, the ones that successfully remap to the E. coli reference and have at least 8 matched probes come to roughly 600,000 to 800,000 usable training molecules."

> "That's enough for training. Modern 1D U-Nets converge reliably on datasets in the 100,000 to 1 million range. We're comfortably within that."

**The hardware diversity:**

> "The 30 runs span 3 instruments -- Uno 202, 429, and 433 -- and 5 distinct detector dies: H09, B23, G16, C12, and D08. That diversity matters because we want to test whether the model generalizes across manufacturing variance in the detectors, not just across molecules within a single die."

**The split -- held out by die, not by molecule:**

> "This is the most important methodological choice on this slide. We split the data by die, not by molecule."

> "If we split by molecule, we would be training on molecules from die X and testing on other molecules from die X. That's trivially easy -- all the molecules share the same hardware quirks. You would get a misleadingly good test score that wouldn't predict real-world performance on a new die."

> "Instead, we hold out die D08 for validation and die C12 for testing. The training set contains no molecules from either die. The test set forces the model to generalize to hardware it has never seen during training. This is the real question we're asking: does this model work on a new detector without retraining?"

**Preprocessing and cloud cost:**

> "The raw TDB files for 30 runs total about 900 GB. That's the full continuous voltage traces for every channel. The vast majority of that data is between-molecule baseline -- empty signal. We have a preprocessing step that extracts just the per-molecule waveforms plus the ground truth and conditioning vectors, and compresses the whole thing to about 5 GB."

> "That 180x compression is what makes this feasible to train in the cloud cheaply. We preprocess locally where the raw TDB files already live, then transfer only the compact cache to GCP. The full training run fits on a single L4 GPU instance."

**Anticipated questions:**

- **"Is 5 dies enough diversity?"** -> "It's what we have. The real test is whether held-out die performance matches training die performance. If yes, the FiLM conditioning and level-1 normalization are doing their job. If no, we need more dies and possibly need to revisit the conditioning strategy. The V1 result will tell us."
- **"Could we synthetically augment to simulate more hardware variance?"** -> "Possible but risky. Synthetic augmentation only captures variance we can model. The whole point of empirical ML here is to capture variance we can't model. More real data from more dies would be better than synthetic data. That's part of the V2 roadmap if V1 indicates insufficient generalization."
- **"Are there any known data quality issues in the 30 runs?"** -> "We filter molecules through the same quality flags that wfmproc uses -- structured, folded, do-not-use. A few runs have higher rejection rates; we include them because diversity of run conditions helps the FiLM conditioning learn."

**Timing:** 3-4 minutes. The held-out-by-die argument is the key content -- this is what proves the evaluation is rigorous.

---

### Slide 12: Three-part loss, anchored to physics and to reference

**Opening:**

> "This slide covers the loss function, the warmup schedule, and the augmentations. Each choice has a specific reason tied to the physics or the data."

**The three loss components:**

> "The total loss is a weighted sum of three components."

> "L_probe is a focal loss on the probe detection heatmap. Focal loss was originally developed for dense object detection where most pixels are negative. Our situation is analogous -- most samples in the waveform are not at a probe center. Focal loss down-weights easy negatives and focuses gradient on hard cases: ambiguous tag boundaries and missed detections. We use standard parameters, gamma 2 and alpha 0.25."

> "The heatmap target is a soft Gaussian at each probe position, with the Gaussian sigma tied to the measured tag duration. A tag that's wider in the signal gets a wider Gaussian target. This prevents the model from being penalized for producing a slightly wider output at genuinely slower-velocity tags."

> "L_bp is the sparse Huber loss on inter-probe deltas that we discussed on slide 6. Huber loss is robust to outliers -- it behaves like L2 for small errors and L1 for large ones, controlled by a threshold delta. We set delta to 500 bp, which is a physically meaningful scale -- smaller than our smallest intervals of interest, larger than typical measurement noise. The scalar loss is then divided by the mean ground-truth interval size to normalize across molecules with different characteristic interval scales."

> "L_velocity is the sparse L2 loss on the raw velocity predictions at tag positions, discussed on slide 7. It enforces that the model's local velocity is consistent with the 511 bp tag width physics."

**The combination and warmup:**

> "The total loss is L_total equals L_probe plus lambda_bp times L_bp plus lambda_vel times L_velocity. Lambda_bp and lambda_vel are target weights of 1.0 after normalization."

> "Critically, we don't use a sequential curriculum -- we don't train probe detection first and then switch on the spatial losses later. Sequential curricula cause feature starvation: the encoder optimizes for whichever task is active and then has to relearn when other tasks turn on. Instead, we use a linear warmup of lambda_bp and lambda_vel from zero to target over the first 3-5 epochs, while L_probe is active from epoch 1. Everything trains end-to-end from the start."

**Augmentations:**

> "We use three data augmentations. Gaussian noise injection scaled to the baseline RMS -- simulates electrical noise variation. Amplitude scaling plus-or-minus 5% -- simulates residual imperfection in level-1 normalization."

> "The third augmentation is uniform time-stretching by a factor between 0.9 and 1.1. This simulates run-to-run voltage or viscosity drift that makes molecules translocate faster or slower on different runs. If we stretch the waveform by factor k, we also divide the velocity targets by k -- but the spatial targets are unchanged, because time times velocity equals distance. This is a physics-consistent augmentation."

**The augmentation we do NOT use:**

> "We explicitly do not use time-flipping -- reading the waveform backward. This would be a standard augmentation for many sequence tasks, but it's physically wrong for our problem. The microfluidic geometry of the channel is asymmetric: the entry funnel produces head-dive acceleration, the exit produces trailing-edge whip. Flipping the signal would place the trailing whip at t equals zero, which violates the physics. We'd be training the model on impossible data."

**Anticipated questions:**

- **"Why not dynamic lambda tuning during training?"** -> "Tried it conceptually, rejected it. The normalized losses are designed to operate at compatible scales with fixed unit weights. Dynamic tuning would add a hyperparameter we don't need."
- **"Do we have any data augmentation for the conditioning vector?"** -> "No. The conditioning values come from physical measurements; perturbing them would train the model on wrong physics. Augmentation is only on the waveform."

**Timing:** 3-4 minutes. The no-time-flipping point is a good technical detail to land -- shows respect for the physics.

---

### Slide 13: Evaluation -- two metrics, not one

**Opening:**

> "How we measure success matters as much as how we train. We use two distinct metrics that answer two different questions."

**The ML metric -- accuracy:**

> "The first metric is the straightforward ML evaluation: on molecules from the held-out die D08, we compute inter-probe intervals from both our model and from legacy T2D, and compare both to the E. coli reference genome distances. We report median absolute error in base pairs."

> "We explicitly avoid comparing our model to legacy T2D directly. We compare both against the reference genome. This is important because if the two models disagree, we want to know which one is closer to the truth, not just which one they agree with each other."

> "Success on this metric looks like: our model has lower MAE than legacy T2D, with a statistically meaningful margin. And ideally, our model has tighter variance -- fewer large outliers."

**The business metric -- rescued molecules:**

> "The second metric answers a different question: does this model actually help the downstream pipeline?"

> "Today, a substantial fraction of molecules fail to map to the reference genome. Some of those failures are because the molecule itself is unusable -- too short, too fragmented, too noisy. Some, we hypothesize, are because legacy T2D distorts the probe spacing so badly that the aligner cannot find a match."

> "The business metric is: take molecules that the aligner rejected when using legacy T2D coordinates. Run them through our model. Feed the new spatial coordinates back to the aligner. How many of those previously-rejected molecules now successfully map?"

> "If we recover even 5% of rejected molecules, that's a 5% effective throughput increase for the instrument, purely through software. That's a concrete, measurable business outcome."

**Quality gating -- what we don't do:**

> "One design choice worth flagging: we do not gate our model's output against legacy T2D output. Some might suggest: if our model's prediction differs wildly from legacy's prediction, flag it as suspicious. We reject that approach."

> "The whole point of this project is that our model should succeed on molecules where legacy T2D gets the answer wrong. Using legacy as a quality gate would reject our model's best contributions as anomalies. Instead, we gate against absolute physical limits -- if the model predicts a macro-velocity faster than the theoretical maximum electrophoretic speed, that's a hallucination and we flag it. Disagreement with legacy T2D is the goal, not a failure mode."

**Diagnostic metrics -- answering the wfmproc-dependency question:**

> "We've added two diagnostic metrics to the evaluation pipeline that aren't pass-fail, but help us understand where the model's contribution is coming from."

> "First: peak-count discrepancy. For each molecule, compare the number of probes our model detects versus the number of probes wfmproc's peak-picker matched to the reference. A mean discrepancy above zero means our model is detecting more peaks than wfmproc labeled -- possibly finding tags that wfmproc missed. A mean below zero means we're constrained by the weak supervision."

> "Second: lambda-vel ablation. We train two versions of the model -- one with L_velocity at full weight, one with L_velocity disabled. Compare test MAE. If they perform equivalently, L_velocity's contribution is negligible and the reference-genome anchor is doing all the work. If the full model is meaningfully better, L_velocity is pulling its weight despite its weak supervision from wfmproc duration measurements."

> "These diagnostics inform the V2 scope decision: should we invest in removing more wfmproc dependencies, or is V1 already getting everything useful out of the data?"

**Anticipated questions:**

- **"How will you report the comparison?"** -> "For each molecule, compute MAE. Produce histograms of MAE distribution for both models side by side. Report median, mean, and tail statistics. If our model's distribution is shifted left (lower errors), we win. We'll also report per-interval-size error -- does the win hold across small, medium, and large intervals, or is it concentrated somewhere?"
- **"What's the statistical threshold for declaring victory?"** -> "We'll report the MAE reduction as a percentage and compute a confidence interval via bootstrap over molecules. A 5-10% reduction with tight confidence would be meaningful; anything less than 5% would raise questions about whether the engineering effort is justified relative to improvements to the legacy parametric model."

**Timing:** 4-5 minutes. The "rescued molecule yield" framing is what makes this project valuable in dollars, not just in science. Land that point.

---

## Section 5 -- Status, Scope, and Ask

---

### Slide 14: Current status -- full pipeline built, training pending

**Opening:**

> "This slide is a status check. The entire software stack for V1 is built and tested. What remains is cloud compute to run the actual training."

**The inventory of what's complete:**

> "I won't read every line, but the categories are: binary file parsers for TDB, probes.bin, and the remap output formats. Ground truth construction with the shift-invariant inter-probe delta logic. The complete 1D U-Net model with FiLM conditioning and bifurcated decoder. All three loss functions with the warmup scheduler. Training loop with mixed-precision support and checkpoint-based recovery from spot instance preemption. Preprocessing pipeline that compresses 900 GB of raw TDB to 5 GB of training cache. Inference pipeline with velocity-adaptive non-maximum suppression. Evaluation scripts including the peak-count discrepancy and ablation diagnostics."

> "All of that is implemented. 124 unit tests passing. The smoke test on synthetic data runs end-to-end without errors."

**What remains:**

> "Three things remain. First, local preprocessing of the 30 real E. coli runs -- converts the 900 GB of raw TDB to the 5 GB cache. This is an overnight job. Second, cloud setup and data transfer -- a few hours of work. Third, the actual training runs, which is where the compute budget comes in."

**Risk mitigation already built in:**

> "I want to point out two pieces of risk mitigation that are already in the implementation. First, checkpointing every 5 epochs. If a spot instance gets preempted mid-training, we resume from the last checkpoint automatically. This lets us use spot pricing -- about 70% cheaper than on-demand -- without risking lost work."

> "Second, the sanity-check script. Before running preprocessing on all 30 runs, we run it on a single run and verify the cached molecule count is in the expected range, the ground truth arrays have consistent shapes, and the cached dataset loads back correctly. Exits with distinct non-zero codes for different failure types so we can diagnose issues before they become batch problems."

**The subtext:**

> "What this status slide is actually saying: the scientific and engineering work is done. The budget ask on slide 16 is purely for compute to run an experiment. This is not a fund-the-research ask; it's a run-the-experiment ask."

**Anticipated questions:**

- **"What's the test coverage like?"** -> "124 tests, covering every binary parser, every loss function, every model output invariant, and the inference pipeline. Full suite runs in about 30 seconds on CPU."
- **"Has the code been reviewed?"** -> "Reviewed end-to-end during development using a structured subagent-driven approach. Two-stage review -- spec compliance then code quality -- after each major component. Every commit corresponds to a reviewed, tested unit of work."

**Timing:** 2-3 minutes. This is a credibility slide; don't over-sell it, just list facts.

---

### Slide 15: V1 scope is deliberately narrow

**Opening:**

> "Before the budget ask, I want to be explicit about what V1 is and is not. This is where the scope discipline lives."

**What V1 is:**

> "V1 is a scientific experiment with three questions:"

> "One: does learned velocity integration beat the parametric drag equation on inter-probe accuracy? Two: does the reference-genome-anchored training strategy avoid replicating legacy T2D biases? Three: does FiLM conditioning on physical observables generalize to a held-out detector die?"

> "If all three answers are yes, V1 succeeds. If any answer is no, V1 has told us something useful about what needs to change."

**What V1 is not:**

> "V1 is not a replacement for wfmproc. It replaces exactly the T2D step in the signal processing pipeline. Everything else -- molecule detection, level-1 estimation, quality flags -- remains the responsibility of wfmproc."

> "V1 is not validated on human DNA. The training data is E. coli only. Any claim about cross-organism generalization is speculative until V2 tests it."

> "V1 is not a production deployment. A successful V1 result would be the green light to build the production integration path, not the deployment itself."

**Why narrow scope matters:**

> "The reason for scope discipline is failure attribution. Consider a broader V1 that also replaces molecule detection, also replaces level-1 estimation, also does streaming inference. If that broad V1 doesn't beat legacy T2D, the failure could be in any of four places. Debugging means rebuilding the whole pipeline to isolate the problem."

> "A narrow V1 either beats legacy on MAE or it doesn't. That's a clean signal. If it succeeds, the narrow V1 is also the minimum viable contribution -- we've proven the core claim and can then expand scope in V2 with confidence that the foundation works."

**The V2+ roadmap:**

> "Here is the progression we're imagining after V1. Not commitments -- just the direction."

> "V1.5 would be self-supervised probe detection: remove the dependency on wfmproc's peak-picker labels by using the reference-genome probe count as a supervision signal. This would be addressable in a few weeks if V1 succeeds."

> "V2 would add molecule detection as a third output head. Unified model ingests raw TDB streams, identifies molecule events, and predicts probe positions and bp coordinates in one pass. Bigger scope change, probably months of work."

> "V3 would be on-instrument inference with real-time deployment. Requires ONNX conversion, efficient streaming inference, and instrument firmware changes. Cross-team effort. Far future."

**The one acknowledged V1 limitation:**

> "I want to be transparent about one limitation we've accepted for V1. Training labels for probe positions come from wfmproc. If wfmproc misses a tag, our training target has no Gaussian there, and we teach the model to also miss it. If wfmproc merges two closely-spaced tags into one peak, we teach the model to also merge them."

> "Our mitigation is the reference-genome anchor on L_bp. If the spatial distance between two tags is wrong because of cluster merging, the reference-genome loss will pull the model toward correcting that. But we can't escape the supervision entirely. The peak-count discrepancy diagnostic I mentioned on slide 13 will tell us quantitatively how limited we are by this."

**Anticipated questions:**

- **"Why not do V1.5 first -- remove wfmproc probe labels before training?"** -> "Because V1.5 adds complexity and a new failure mode to the training process. If V1 fails, we want to know why without that extra variable. If V1 succeeds, V1.5 is an obvious and low-cost follow-up."
- **"What if V1 fails?"** -> "We have a well-defined debugging sequence: first check the peak-count diagnostic to see if we're supervision-limited, then check the velocity-loss ablation to see if physics supervision is helping, then inspect per-molecule error patterns to see if failures cluster on specific hardware or specific velocity regimes. Each of those pointers toward a different V2 direction."

**Timing:** 4-5 minutes. This slide is where skeptical reviewers are most likely to push. Don't rush it.

---

### Slide 16: Cloud training ask

**Opening:**

> "The ask is straightforward. Approximately $75 on GCP, spread over 4 weeks, for the full V1 development cycle. Here's the breakdown."

**The compute profile:**

> "One NVIDIA L4 GPU instance on GCP. L4 has 24 GB of VRAM, which comfortably fits our model at batch size 8 to 16 with mixed precision. L4 is the cost-optimal choice for 1D convolutional workloads -- A100s are overkill for this model size."

> "Spot-preemptible pricing at approximately $0.21 per hour. On-demand is about 3.5 times more expensive; we use spot because our checkpointing infrastructure makes preemption harmless."

> "Estimated training time for a single full run: 60 to 100 hours. We budget for multiple runs -- the baseline V1, the L_velocity ablation for diagnostics, one or two hyperparameter sweep iterations. Total compute: approximately 130 hours."

**The storage profile:**

> "The preprocessing step runs locally on machines that already have network access to the raw TDB files. We transfer only the 5 GB compact cache to GCP, not the 900 GB of raw data. Small persistent disk, minimal storage cost."

**The budget breakdown:**

> "Compute: 130 hours at L4 spot pricing -- about $27. VM base instance cost -- about $10. Boot and data disk -- about $30 over two months. Network egress -- negligible. Total approximately $75."

**The timeline:**

> "Week 1: local preprocessing of the 30 runs, sanity checks, GCP environment setup, initial data transfer, and a local smoke test on a small preprocessed sample. Weeks 2 and 3: V1 training and evaluation on held-out die D08, plus the L_velocity ablation run for diagnostic purposes. Week 4: analysis, final report, V2 scope decision."

**The decision framing:**

> "I'm not asking for a commitment to V2. I'm asking for compute to run V1 and produce the data that tells us whether V2 is worth pursuing. A $75 experiment with a binary success criterion -- does it beat legacy T2D on held-out data? -- is a small investment for a high-leverage answer."

**What success enables:**

> "If V1 succeeds on the ML metric and the business metric, the path forward is: a one-week sprint to build the production inference CLI, then an A/B evaluation against the legacy pipeline on a few recent runs, then a gradual rollout."

> "If V1 fails, the diagnostics tell us where to invest the next iteration. Even a failed V1 produces useful information -- we will know whether the problem is supervision quality, data quantity, model capacity, or physics approximation."

**Closing:**

> "I'm ready to proceed as soon as approval is in place. I can start local preprocessing this week and have the sanity-check results ready before we commit any cloud spend."

**Anticipated questions:**

- **"Is $75 a hard number?"** -> "It's the expected total. With spot preemption risk and additional debugging cycles, worst-case ceiling is probably $200. Still trivial for the experimental value."
- **"What if we want to train on more GPUs for faster iteration?"** -> "A single L4 is sufficient for this model size. Multi-GPU training would add engineering complexity for limited speedup. If V2 scales up substantially, multi-GPU becomes relevant; not for V1."
- **"Can this run on on-premises hardware instead of GCP?"** -> "Yes, if Nabsys has available GPU capacity. We'd need a GPU with 16-24 GB VRAM and a recent PyTorch installation. The code is portable."

**Timing:** 3-4 minutes. End with the closing line -- hands on the table, ready to start.

---

## Overall Timing Summary

| Section | Slides | Target Duration |
|---------|--------|-----------------|
| Problem setup | 1-3 | 6-8 minutes |
| The Approach | 4-7 | 13-16 minutes |
| The Architecture | 8-10 | 10-13 minutes |
| Training and Evaluation | 11-13 | 10-12 minutes |
| Status, Scope, Ask | 14-16 | 9-11 minutes |
| **Total** | **16** | **48-60 minutes** |

Leave 10-15 minutes for Q&A. A 60-minute slot is appropriate. A 30-minute slot would require compressing the architecture section (slides 8-10) to a single overview slide.
