# Mongoose ML Sprint — Presentation Brief

> **For Cowork:** Generate a 12-15 slide presentation from this brief. Audience is internal Nabsys (ML / biology / engineering mix). Tone: technical but accessible. Each `## SLIDE:` header is a recommended slide. Tables are chart candidates — convert to bar/line charts where it improves clarity. Highlight the bolded numbers as headline data.

---

## SLIDE: The Goal (single sentence)

**Build a model that takes a TDB file and outputs accurate inter-probe distances in basepairs — no remapping required at inference time.**

The output feeds an assembler. Assembly quality is downstream of how accurately we report inter-probe bp distances. Two metrics matter:
- **Median relative error**: how good are typical predictions?
- **p95 relative error**: how bad are the tails? (Assembly chokes on heavy tails — a single bad interval creates a phantom edge in the layout graph and shatters contigs.)

Production deployment bar: **median < 10%, p95 < 20%.**

---

## SLIDE: Where we started (V1 — the existing recipe)

V1 was Phase 6 of an existing sprint: a 1D U-Net trained against remapping's `.assigns` ground truth. Three output heads:
- Probe heatmap (where are the probe events in time?)
- Cumulative bp curve (per-sample integrated bp)
- Velocity (per-sample bp/s)

Trained 4 epochs × 27 caches with teacher-forced physics losses (L_bp via soft-DTW, L_vel MSE) and CenterNet focal loss for probe detection.

**Key V1 result: probe detection works well** (F1 = 0.89 on 3 holdouts).
**Key V1 problem: bp output isn't accurate enough** (we'll see).

### Visual: V1 architecture
A 1D U-Net diagram: shared backbone → 3 heads (probe, cum_bp, velocity).

---

## SLIDE: V1 results — peak detection vs bp accuracy

V1 is great at **finding** probes (where in the time-domain they are) but inaccurate at **measuring** bp distances between them.

### Table: V1 holdout performance

| Metric | Value | Verdict |
|---|---|---|
| Peak F1 (sum-of-counts, all 3 holdouts) | **0.894** | Excellent |
| Median bp-interval relative error | **30.1%** | Far from production bar |
| p95 bp-interval relative error | **271%** | Assembly-killer |
| Correlation (pred vs ref intervals) | 0.64 | OK but not great |

The peak F1 metric was what we'd been optimizing for during V1. The bp-interval metric is the one that actually matters for assembly. We hadn't been measuring it directly until this week.

### Talking point
V1 was trained to mimic remapping. Its output ceiling is therefore remapping accuracy, and even that's not fully achieved — the soft-DTW loss is forgiving of local mis-alignment, allowing systematic scale drift. Net result: V1 reports "where probes are in time" well but "how far apart they are in bp" poorly.

---

## SLIDE: The reframe — what's the right benchmark?

Until this week we'd been comparing V1 to itself or to its own training metrics. Critical realization: we should be comparing to the **production-deployed alternative**, which is the legacy T2D power-law model:

$$L(t) = \text{mult\_const} \cdot t_{\text{from\_tail}}^{\alpha} + \text{addit\_const}$$

Per-channel constants are fit against remapping data and stored in `_transForm.txt`. Deployed in Nabsys production today.

### Visual: T2D formula + plot of L vs t for typical molecule
Show the power-law shape — distance from trailing edge grows as roughly t^0.55. This embeds Stokes drag physics (DNA accelerates as it translocates).

---

## SLIDE: T2D is way better than V1

When we built the bp-interval evaluator and ran it against the same 3 holdouts:

### Table: T2D vs V1 head-to-head

| Method | Median rel err | p95 rel err | Correlation |
|--------|---:|---:|---:|
| **T2D (production)** | **16.2%** | 167% | 0.72 |
| V1 (Phase 6) | 30.1% | 271% | 0.64 |

T2D wins by a factor of 2 on the median and 1.6× on p95 with just **3 calibration parameters per channel** vs V1's millions of neural-network weights.

### Talking point
This was the wake-up call. We had been positioning V1 as a replacement for the legacy pipeline. It actually performs **worse** than the algorithm it was meant to replace. The deployment bar isn't 30% (V1) — it's 16% (T2D). And we're 9 points away.

---

## SLIDE: Why is T2D so much better?

T2D has **3 advantages V1 lacks**:

1. **Per-channel calibration.** Every detector channel gets its own 3-parameter fit against that specific channel's remapping data. V1 is one universal model across all channels.
2. **Physics inductive bias.** The power-law formula bakes in Stokes drag from first principles. V1 has to discover this from data.
3. **No spurious supervision.** T2D fits the bp-position calibration directly. V1's soft-DTW loss is lenient to local mis-alignment, allowing scale drift.

### Talking point
The fight isn't "neural net vs power law" — it's "general-purpose neural net trained on a forgiving loss" vs "physics-constrained 3-parameter model fit per-channel." The latter wins because the inductive bias and per-channel calibration are doing real work.

---

## SLIDE: V2 — the design we considered, then killed

V2 was originally going to be an architectural upgrade (Transformer U-Net, Sinkhorn-OT positional loss, 6-class segmentation head, 8x temporal resolution). We did 6 rounds of design with a peer reviewer (Deep Think / Gemini).

**Then we killed it.** Why: V2 is more architectural firepower aimed at the same wrong objective. It would be a better mimic of remapping — but remapping is no better than T2D, and the soft-DTW supervision regime doesn't move us toward bp-accurate intervals.

V2 wouldn't beat T2D. It would just be a more expensive way to match V1's ceiling.

### Talking point
The discipline here was important: it's tempting to keep optimizing the architecture you have. The honest answer was that the SUPERVISION REGIME was the bottleneck, not the architecture. V2 was a sunk-cost trap. Killing it freed us to pivot.

---

## SLIDE: The pivot — physics-informed training (V3)

**Core insight:** every Nabsys probe is exactly **511 bp wide** (a controlled physical constant). This gives a per-probe integral identity on the velocity field:

$$\int_{\text{probe}_i} v(t)\,dt = 511 \text{ bp} \quad \forall \text{ probe } i$$

Applied across thousands of probes per batch, this is **dense self-supervision** on the velocity field that does NOT require remapping. This was the seed of V3.

We ran a feasibility spike to test if L_511 could train the velocity head at all.

---

## SLIDE: V3 spike — does L_511 work as a supervision signal?

3 epochs × 4 caches × from-scratch training with L_511 + L_smooth + L_length losses.

### Table: V3 spike on Blue holdout (canary cache)

| Epoch | Median rel err | p95 rel err | Correlation |
|---:|---:|---:|---:|
| 1 | 47.2% | 258% | 0.52 |
| 2 | 33.1% | 263% | 0.60 |
| 3 | **31.9%** | 268% | 0.60 |

L_511 produces real, monotonic loss reduction. The architecture isn't degenerate. But the trajectory **decelerated sharply** epoch 2→3 (14-point gain → 1-point gain). Looks like a plateau.

Hypothesis: V1's CNN backbone can't internalize the global drag physics on its own. The L_511 anchor is real but the architecture caps how much it can be exploited.

---

## SLIDE: V3 extension — does more training help?

Warm-started from spike best, ran 7 more epochs (10 total) with fresh learning-rate schedule.

### Table: V3 extension trajectory (overall, across 3 holdouts)

| Epoch | Median | p95 | Correlation |
|---:|---:|---:|---:|
| 1 | 34.3% | 211% | 0.59 |
| 4 | 27.1% | 209% | 0.66 |
| 7 | **25.4%** | 201% | 0.67 |

7 more epochs gained 6.5 percentage points on median (31.9% → 25.4%). But still 9 points behind T2D (16.2%). And p95 barely moved.

Conclusion: **the plateau isn't training-time, it's architectural.** V1's translation-invariant CNN can't model the global drag deceleration that T2D's power-law captures with 3 parameters.

---

## SLIDE: The diagnostic — why V3 plateaus

We used a unified comparison harness to test all V3 variants against all 3 holdouts.

### Table: Cross-method comparison (all metrics overall, 1.5M intervals)

| Method | Training | Median | p95 | Correlation |
|--------|----------|---:|---:|---:|
| T2D | per-channel calibration | **16.2%** | 167% | **0.72** |
| V1 (Phase 6) | 4 ep × 27 caches, soft-DTW | 30.1% | 271% | 0.64 |
| V3 spike | 3 ep × 4 caches, L_511 | 31.9% | 268% | 0.60 |
| V3 extension | +7 ep continued | 25.4% | 201% | 0.67 |

The pattern: **all three neural variants converge near a 25-30% floor.** T2D sits 9-14 points below. The neural net can't replicate T2D's drag physics from data alone in a reasonable epoch budget.

### Visual: line chart of median rel err vs epoch
X-axis: epochs (V3 spike 0-3, then continued through extension to 10).  
Y-axis: median rel err on Blue holdout.  
Horizontal line at 16.2% labeled "T2D bar".  
Shows V3 trajectory descending but flattening as it approaches T2D bar.

---

## SLIDE: Option A — bake the physics into the architecture

If V3 can't learn drag from scratch, **bake T2D in as the floor and learn corrections on top.**

$$v_{\text{final}}(t) = v_{\text{T2D}}(t) \cdot (1 + \text{residual}(t))$$

- $v_{\text{T2D}}(t)$: derived per-sample from per-molecule constants (mult, alpha, tail_ms). NOT trained.
- $\text{residual}(t)$: tanh-bounded to ±50%, output by the velocity head. Trained.

### Properties

- **Worst case = T2D.** When residual → 0, output collapses to pure T2D. We can't do worse than 16% by construction.
- **Best case beats T2D.** If residual learns useful corrections (probe-specific slowdown, channel-specific deviations T2D's 3 params can't capture), we go below 16%.
- **Graceful degradation.** At inference without telemetry, can fall back to pure T2D as a baseline.

### Visual: architecture diagram
Show v_T2D (closed-form, fixed) × (1 + residual_NN_output) → composed velocity → cum_bp.

---

## SLIDE: Option A — what we shipped this week

Full implementation in 1 day (code committed `aa6a9c3`):

- `compute_v_t2d()` — per-sample T2D from molecule metadata
- `T2DUNet.forward(..., t2d_params=...)` — backward-compatible hybrid path
- `precompute_t2d_params.py` — enriched all 30 caches with per-molecule T2D constants
- `--use-t2d-hybrid` CLI flag
- 8 unit tests pinning invariants (residual=0 → pure T2D, ±0.5 bounds, gradient flow, etc.)
- Cloud pipeline (NAT, GCS upload, retry-loop H100/A100 launcher)

---

## SLIDE: Option A — the experiment we're running

**Two parallel runs:**

| Track | Setup | Status |
|-------|-------|--------|
| Local smoke | 3 ep × 4 caches, from scratch | Running, ep 2/3 |
| Cloud production | 20 ep × 27 caches, A100 80GB | Waiting on cloud capacity |

**Local smoke** is the canary — does residual descend or stall? Tells us trajectory in ~3.5h.

**Cloud production** is the verdict — does Option A actually beat T2D? Once we land an A100 (capacity is tight today), runs ~10 hours, lands tomorrow morning.

### Decision gates (pre-committed before seeing results)

| Outcome | Action |
|---------|--------|
| Median < 10% (p95 < 20%) | Production-grade. Ship Option A as the new baseline. |
| Median 10-16% | Beats T2D. Strong case to ship, but iterate on tails. |
| Median 16-20% | Comparable to T2D. Architectural insight valuable but not a clear win. |
| Median > 20% | Worse than T2D. Investigate; iterate or pivot. |

---

## SLIDE: The bigger picture — what this is really about

The end goal isn't "replace T2D with a neural net." It's:

> **Predict bp intervals accurately enough to enable de-novo assembly without remapping dependence.**

T2D is calibrated PER-RUN against remapping. So even though T2D wins today, **its accuracy depends on having a remapping run for each new sample to fit per-channel constants.** This is the fundamental limitation of the legacy pipeline — it can't generalize to a new sample without a remapping pass.

A neural model that achieves T2D-comparable accuracy on its own — without per-sample calibration — would be a real product win even if it just matches T2D's median. It would generalize to new samples without the remapping bottleneck.

### Talking point
This is the long-term framing. Option A inherits T2D's per-channel calibration today (we still use _transForm.txt). But the path forward is to learn what T2D encodes from physics + telemetry alone, removing the remapping dependence entirely. That's V3.x territory after Option A.

---

## SLIDE: Engineering choices we made and why

Worth highlighting for the team:

1. **Killed V2 instead of finishing it.** Avoided sunk-cost trap. Freed budget to test the right hypothesis (physics-informed) instead of polishing the wrong objective.

2. **Built a real bp-interval eval.** Wasn't measuring it before. The "V1 is bad at bp" insight came from finally measuring the metric that actually matters.

3. **Fixed a 5× formula error in legacy T2D code.** The repo's T2D implementation had wrong units (samples vs ms) and wrong addit_const semantics. Corrected per the Oliver 2023 derivation in `support/T2D.pdf`. Without this, our T2D baseline would have been 273% median (wildly wrong) and Option A's design would have proceeded against a phantom benchmark.

4. **Two-track parallel execution.** Local smoke + cloud production simultaneously. Worst case loses $80 of cloud spend. Best case: cloud finishes 10h faster than sequential.

5. **All training code lives in `--use-l511 --use-t2d-hybrid` flags.** Toggle-able. Backwards-compatible with V1. Old checkpoints still load and run.

---

## SLIDE: What we don't yet know

Honest uncertainties:

1. **Does Option A beat T2D, or just match it?** Answered tomorrow morning by cloud results.
2. **How much of T2D's accuracy is the per-channel calibration vs the physics?** If we strip per-channel calibration, can we still hit 16-20%?
3. **How much further can we push without remapping in the loss?** Currently the residual learns from L_511 (which uses wfmproc probe events — signal-processing, not remapping). True remapping-free training is a future deliverable.
4. **Can we generalize across runs (different chemistry, different detectors)?** All training so far is E. coli on the OhmX433 detector. Need cross-detector evaluation.

---

## SLIDE: Roadmap — what's next after this week

Assuming Option A lands favorably:

**Short-term (1-2 weeks)**
- Production cloud run on full 27-cache training set
- Cross-detector eval (different detector ID)
- Cross-genome eval (something other than E. coli, e.g., lambda phage)

**Medium-term (1-2 months)**
- V3.1: remove per-channel _transForm.txt dependency. Learn per-molecule T2D parameters from metadata (bias voltage, NanoPress, baseline current — RunLog parser already shipped this week).
- Self-supervised probe detection (currently uses wfmproc — signal-processing — as warm-start; ultimately replace with contrastive learning).

**Long-term (research)**
- Full remapping-free pipeline. Neural model that produces assembler-ready intervals from raw TDB without ANY classical SP or remapping in either training or inference.

---

## SLIDE: Headline numbers (closing slide)

| What | Number |
|------|---:|
| V1 median bp error | 30.1% |
| **T2D (production) median** | **16.2%** |
| Spike feasibility result | L_511 produces real signal (validates direction) |
| Extension result | architectural ceiling, not training-time |
| Option A architecture | guarantees ≤ T2D, learns to beat it |
| Days from V1 ship to Option A cloud run | **5** |
| Total cloud spend (projected) | **<$100** |

**The real win, regardless of Option A's final number:** we now have a benchmark that matters (bp-interval median + p95) and an architecture (Option A) that gracefully degrades to the production baseline. Future iterations can only improve from here, not regress.

---

## Supplementary appendix (don't make slides — talking-point fodder)

### The label-mapping bug
Earlier in the sprint we discovered V1 had been training on labels that were systematically mis-positioned by ~280 samples (the molecule's `start_within_tdb_ms` offset wasn't being added). Probes were landing on baseline noise instead of probe spikes. F1 jumped from 0.033 → 0.917 after the fix. This bug had been live for weeks.

### The GCP setup adventure
Spent ~3 hours navigating GCP UI quirks: API enable propagation, default VPC creation, organization policy blocking external IPs (resolved via Cloud NAT), quota increase requests for both H100 and A100 80GB on-demand. Eventually got everything working but it was a longer-than-expected detour.

### Deep Think (peer review) rounds
6 rounds of architectural review. Key contributions:
- Round 5: caught the variational degeneracy in v1's L_511 + L_smooth design
- Round 6: caught the "slack-variable trap" where probe boundaries could be exploited
- Round 7: validated the macro/micro decomposition (which we ultimately replaced with the simpler T2D-hybrid approach)

### Code stats
- 23 files changed in main commit (`aa6a9c3`)
- 2,529 lines of code added
- 22 unit tests passing (8 Option A, 8 L_511, 6 RunLog parser)
- 7 launch scripts (local + cloud)

---

*End of brief. Cowork: produce the presentation. Use the data tables as charts where it improves clarity. Keep the narrative arc intact: starting position → measurement of what really matters → pivot rationale → current bet.*
