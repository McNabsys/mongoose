# Peer-review round 2: we found the ACTUAL bug, now how do we train?

## DRAFT — review before sending

---

## TL;DR

Your previous review was correct about everything mechanical (count-vs-probe
contradiction, NMS graph break, BCE dilution). We implemented your fixes and
Deep Think's earlier plan: CenterNet focal loss, physics-off smoke, count
disabled. None of it worked. Probe plateaued at 2.18 after 300 steps. Your
"reward-hacking" diagnosis of the step-100 velocity blowup was also spot on.

**But the REAL bug was upstream of all that: our warmstart labels were
systematically mis-aligned with the waveform.** `probe.center_ms` from
Nabsys `probes.bin` is measured from the molecule's translocation start
(`start_within_tdb_ms`), not waveform sample 0 — and the TDB samples at
32 kHz, not 40 kHz as our code assumed. Both bugs silently compound, producing
labels that land mostly on flat baseline instead of the real upward probe
peaks.

Once fixed and the cache was regenerated, the overfit-one-batch gate passed
cleanly: probe loss dropped from 6.19 → 0.066 in 300 steps, monotone decrease
from step 50 onward. Every label now lands dead-center on a visible peak.

So our model was never the problem. Our loss was never the problem. We were
training the U-Net to place peaks at positions where the waveform had no
feature at all. Given that, it's remarkable the original BCE recipe got
recall 1.7% instead of 0%.

---

## 1. What we did since your last review (chronological)

1. **Implemented your exact fix plan.** Added `centernet_focal_loss`
   (pos_inds with threshold 0.99, α=2, β=4, per-molecule normalization by
   `num_pos.clamp(min=1.0)`). Wired it into `CombinedLoss`. Removed
   `probe_pos_weight`, `focal_gamma`, `focal_alpha` from the config. Ran
   the overfit-one-batch gate. All unit tests green, BF16 autocast
   stable, no NaN.

2. **Overfit gate failed:** probe plateaued at 2.18. Expected.
   Ran diagnostic sweep (your suggestion):
   - Run 1: Physics off, default CenterNet. Probe=1.80.
   - Run 2: Physics off, relaxed α=1, β=2, pos_threshold=0.8. Probe=2.22 (worse).
   - Run 3: Same as Run 2 with LR 3e-3. Diverged.

   You noted 2-D vs 1-D geometry and wondered if lowering β would help.
   Turned out the opposite — β=2 made `(1-target)^β` larger at small
   target values, so the halo penalty went *up*, not down. I had that
   math backwards.

3. **Run 4 sweep:** isolated variables. Run 4b (default geometry,
   LR=1e-3, 500 steps) was best, reached probe=1.36 with a "pred=1"
   viz. But that single predicted peak was a false positive in the
   zero-padded region past the end of the molecule — a mask-leak in
   the viz code, not a real detection.

4. **Then we noticed:** the blue reference lines in our viz clustered
   in a dead region of the waveform. No real physical spikes under
   them. The domain expert (project lead, ex-Nabsys engineer) said
   categorically: *wfmproc does not label flat regions.*

5. **Found the mapping bug by bisecting through the code:**
   - `src/mongoose/data/ground_truth.py` hardcoded `SAMPLE_RATE_HZ = 40_000`.
     TDB header says 32,000.
   - Line 144: `centers.append(int(round(probe.center_ms / SAMPLE_PERIOD_MS)))`
     — conversion ignored `mol.start_within_tdb_ms`, which is the molecule's
     translocation-start offset (typically 5–15 ms / 160–480 samples).

6. **Confirmed by inspection of in-house tool** (`src/mongoose/tools/probe_viz/viewer.py:89`):
   ```python
   t = pm.start_within_tdb_ms + probe.center_ms
   ```
   That's the correct formula.

7. **Fixed the bug.** Required `sample_rate_hz` kwarg in `build_molecule_gt`,
   threaded from `TdbHeader.sample_rate` through `preprocess.preprocess_run`.
   Added a TDD test pinning the formula with synthetic inputs calibrated
   to a known real molecule (uid=13577: expected samples 667 / 1659 for
   first and last probes). Test passes.

8. **Regenerated the cache for all 30 runs.** Took 10 minutes end-to-end.
   Each run produces 20k–65k cached molecules; 1.25M total across Black
   (8 runs), Blue (13 runs), Red (9 runs).

9. **Re-ran the overfit gate with the fixed cache.**
   - Original recipe (physics on, LR=3e-4):
     ```
     step 1:    probe=6.19
     step 150:  probe=0.30
     step 300:  probe=0.70  ← regression! step-100 reward-hacking event
                              exactly as you predicted: bp/vel blowup
                              around the moment peaks cross the NMS threshold
     ```
   - Physics off (`--lambda-bp 0 --lambda-vel 0`):
     ```
     step 1:    probe=6.19
     step 50:   probe=0.63
     step 100:  probe=0.39
     step 150:  probe=0.27
     step 200:  probe=0.20
     step 250:  probe=0.13
     step 300:  probe=0.066  ← GATE PASSED (<0.1 target)
     ```
   Monotone descent every checkpoint, no reward hacking without the
   physics losses.

The model is ready to train.

---

## 2. Current Recipe (proven to work on single-batch overfit)

- **Architecture:** 1-D U-Net, 5 encoder levels (32→64→128→256→512),
  kernel=7, dilated bottleneck (d=1,2,4,8), 4-head self-attention,
  FiLM conditioning at level-0 and bottleneck. 3 heads: probe logits,
  velocity (softplus), cumulative_bp (cumsum of velocity). Probe-head
  bias = -3.0 init for sparse prior.
- **Loss:** CenterNet focal on probe head (α=2, β=4, pos_threshold=0.99,
  normalized by num_pos per molecule). L_bp (soft-DTW) and L_vel
  (MSE-at-detected-peaks) exist but are zeroed via `--lambda-bp 0
  --lambda-vel 0`. L_count (smooth-L1 on sum(heatmap) vs n_probes)
  effectively disabled via `--scale-count 1e9`.
- **Schedule:** blend=1.0 throughout (`--min-blend 1.0
  --warmstart-epochs N --warmstart-fade-epochs 0` with N ≥ epochs),
  LR=3e-4, AdamW, bfloat16 autocast.
- **Data:** 30 runs × ~40k molecules each = 1.25M cached molecules.
  Waveforms sampled at 32 kHz, normalized by `mean_lvl1_from_tdb`
  (though the normalization scale itself looks suspect — amplitudes
  come out at ~1e-4 post-normalize, not ~1 as typical for "fraction
  of open-channel current." We haven't chased this yet.)

---

## 3. Open architectural question: how to use the physics losses

The physics branches (L_bp soft-DTW and L_vel MSE) are necessary for the
final task — recovering base-pair positions from probe peaks needs the
velocity head to learn a real velocity profile. But the current
formulation evaluates them at NMS-detected peak positions, which means:

- They can't train until the probe head produces peaks above threshold.
- The NMS op is non-differentiable, so L_bp/L_vel gradient flows only
  into the velocity head, not the probe head (you called this out in
  round 1).
- Once peaks DO cross threshold, the velocity head receives a burst of
  garbage gradient from its random init, which destabilizes the shared
  encoder and the probe head squashes outputs back below threshold —
  the "reward-hacking" oscillation we see at steps 100–300 of the
  physics-on gate.

**Two options for Phase 4 (the real 35-epoch training run):**

**Option A: Stage the loss.** Train with physics OFF for first N epochs
(probe head learns to place peaks; velocity head gets zero gradient,
stays near init). Once probe head is stable, fade physics losses in.
No code changes. Risk: velocity head wakes up cold and has to
catch up on already-frozen encoder features.

**Option B: Teacher-forced L_bp and L_vel.** Evaluate them at
GROUND-TRUTH reference indices (`probe_centers_samples`) instead of
predicted peaks during training. This bypasses the NMS gate entirely,
gives the velocity head supervision from epoch 1 step 1, and keeps the
probe head in its own lane via the CenterNet loss. Requires plumbing
`probe_centers_samples` through the loss signature (dataset →
collate → `CombinedLoss.__call__`). Non-trivial but a real architectural
improvement and what you recommended in round 1 for exactly these
reasons.

I now lean strongly toward **B**. With the label mapping fixed and the
CenterNet loss proven to work, teacher forcing the physics losses gives
us supervision at all three heads from step 1 with NO coupling bugs.
Current "A" plan leaves a half-trained encoder ossified against random
velocity-head outputs.

---

## 4. Open data question: Phase 2a smoke (single cache) vs straight to
Phase 4 (multi-run)

The original plan called for Phase 2a = 10 epochs on one cache
(smoke), then Phase 4 = 35 epochs on one cache (real training), then
Phase 6 = 35 epochs on all runs with holdouts.

**But with the label fix, the single-cache smoke feels redundant** —
we'd just re-prove the overfit gate passes at scale. It would still
leave us with no generalization signal (training and val from the same
run share the same nanopore detector, same reagent lot, same chip
history).

**Proposed alternative:** go straight to multi-run, shorter epoch count,
held-out split from day 1.
- Train on 27 runs (9 Black + 12 Blue + 6 Red roughly).
- Hold out 3 (1 per color, alphabetically-last per color for
  reproducibility).
- 5 epochs instead of 35 (smoke-test the multi-run pipeline, not a
  production model).
- ETA: ~3–4 hours on our hardware.
- F1 on held-out runs gives a REAL generalization number vs what
  Phase 2a F1 ever could.

If that works: scale up to 35 epochs on the same split for the
production model. If not: go back to single-cache Phase 2a as a
debugging step.

---

## 5. My recommendation

**Do B and the multi-run smoke in one shot:**

1. Implement teacher-forced L_bp and L_vel. Plumb
   `probe_centers_samples` through to `CombinedLoss`. Write unit test
   that verifies gradient flows into the velocity head from epoch 1
   step 1 regardless of probe head output (currently it doesn't).
2. Launch a 5-epoch multi-run training on 27 runs with:
   - CenterNet probe loss, pos_threshold=0.99, α=2, β=4
   - Teacher-forced L_bp and L_vel (real contributions, not zeroed)
   - blend=1.0 throughout
   - LR=3e-4
   - 3 held-out runs (1 per color)
3. Evaluate peak-match F1 on held-out runs.
4. If F1 > ~0.3, run the full 35-epoch production training on the
   same split.

**Why this order:** the overfit gate already proves the probe head works.
What we don't know is whether (i) the velocity head can learn anything
useful on real data and (ii) the model generalizes across runs. Those
are the real questions, and they need the physics losses live AND
held-out data.

**Risk I'm aware of:** teacher forcing is a real code change (not a CLI
flag tweak), so it could introduce new bugs. But it's what you
recommended originally and I think we'd have saved days if we'd done
it then.

---

## 6. Questions for you

1. **Teacher forcing or stage-the-loss?** I think B for the reasons
   above — your call in round 1 was B, does the label-bug story change
   anything? If yes, why?

2. **Skip Phase 2a smoke?** Does the argument for going straight to
   multi-run hold up, or is there a category of bug you'd want the
   single-cache 10-epoch run to surface first?

3. **Holdout strategy.** 1 run per color works as a smoke holdout.
   For a real generalization test, should we hold out more
   aggressively — e.g., all runs from a specific chip lot
   (reagent batches are encoded in the run_id)?

4. **The waveform normalization scale.** Post-normalize amplitudes are
   ~1e-4, not ~1. Does that feel like a smoking gun for you, or
   acceptable as long as the model's being trained on a consistent
   signal?

5. **Anything we're missing.** The label-mapping bug sat in our
   repo for two weeks undetected because the labels were internally
   consistent — they just happened to land on baseline. Is there a
   systematic way we should be catching bugs like this (invariant
   tests, visualization-on-each-PR gates, etc.) before the next one
   costs us another weekend?

---

## Artifacts for reference

- `docs/plans/2026-04-18-centernet-focal-loss-rescue.md` — the
  implementation plan we executed.
- `fix_verified_uid13577.png` — before/after visual: every probe label
  now lands dead-center on a visible upward peak.
- `overfit_gate_FIXED_physics_off.log` — the 6.19 → 0.066 convergence
  trace.
- Git history on branch `claude/peaceful-rubin-bfb7a9`, commit
  `26f84ec` is the mapping fix.
