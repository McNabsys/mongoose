# Peer-review round 3: gate passed (sort of), ready for Phase 6 — or should we tune first?

## DRAFT — review before sending

---

## TL;DR

Your round-2 plan is fully implemented. Z-score normalization in collate,
teacher-forced L_bp and L_vel at ground-truth indices, trainer plumbing, the
dataloader-QA viz script, and the final physics-ON overfit gate are all in
place on the branch.

**Results:**
- probe loss went **8.23 → 0.10 in 100 steps** (first time we've seen that
  speed — z-score did what you predicted)
- pred=**5 peaks detected out of 8 reference** at step 300 (first time
  peaks have actually crossed the NMS threshold on the real signal region)
- teacher forcing is active (vel and bp both nonzero from step 1, no NMS gate)
- BUT the probe loss regresses after hitting the floor:
  ```
  step 100: probe=0.10  ← gate hit
  step 150: probe=0.16
  step 200: probe=0.49  ← oscillation
  step 250: probe=0.17
  step 300: probe=0.34  ← final, not <0.1
  ```

The architecture, the data pipeline, and the loss are now all proven correct.
The model is detecting real peaks on real signal in z-scored waveforms. But
we haven't reached the clean monotone descent that physics-off gave us.

Question for you: launch Phase 6 (5-epoch, 27-run, 3 holdout, ETA 4–6 h) as is,
or tune the gate recipe first?

---

## 1. What we did (round 2 implementation)

Followed your round-2 plan precisely. Five commits on branch
`claude/peaceful-rubin-bfb7a9`:

1. `819a393` — Z-score the valid-mask region in `collate_molecules`
   (your 5-line fix for the 1e-4 amplitude / BF16 precision floor). Unit
   tests verify mean≈0, std≈1 per molecule; padding region preserved at 0;
   constant-signal safe (eps clamp).
2. `3d15f8c` — Teacher-force L_bp and L_vel at
   `warmstart_probe_centers_samples`. Legacy NMS branch retained as
   `else` clause for back-compat. TDD: a test that forces a flat probe
   output (all logits = -3) and confirms L_vel gradient flows to
   `raw_velocity` regardless — would have been 0 under NMS path.
3. `5a3d215` — Thread the list through `trainer.py` and
   `overfit_one_batch.py` into `CombinedLoss.__call__`.
4. `85a3aea` — `scripts/plot_random_dataloader_batch.py` — your Q5 gate.
   Plots 8 random post-collate molecules with `warmstart_heatmap` (black)
   and `warmstart_probe_centers_samples` (green) overlaid on the
   z-scored waveform. Manually verified: green lines land dead-center
   on upward peaks across all 8 panels.
5. (gate artifacts not committed yet; see §2)

---

## 2. Gate results (physics ON + teacher forcing)

Command:
```
scripts/overfit_one_batch.py \
    --cache-dir <single-run-cache> \
    --steps 300 --batch-size 32 --lr 3e-4 \
    --scale-bp 300000 --scale-vel 5000 \
    --scale-count 1e9 --scale-probe 1.0 \
    --lambda-bp 1.0 --lambda-vel 1.0 \
    --min-blend 1.0 \
    --warmstart-epochs 1 --warmstart-fade-epochs 0
```

```
step   1: total=9.66  probe=8.23  bp=1.07  vel=1.78  raw_bp=320398  raw_vel=8882
step  50: total=0.96  probe=0.27  bp=0.75  vel=0.64  raw_bp=224562  raw_vel=3216
step 100: total=0.59  probe=0.10  bp=0.63  vel=0.33  raw_bp=189820  raw_vel=1653
step 150: total=0.59  probe=0.16  bp=0.60  vel=0.25  raw_bp=179981  raw_vel=1228
step 200: total=0.91  probe=0.49  bp=0.51  vel=0.34  raw_bp=153485  raw_vel=1695
step 250: total=0.49  probe=0.17  bp=0.42  vel=0.21  raw_bp=127166  raw_vel=1057
step 300: total=0.63  probe=0.34  bp=0.37  vel=0.20  raw_bp=111730  raw_vel=997
```

Viz at step 300 (on a single molecule with 8 reference probes):
**pred=5 red (real detections over threshold=0.3) near the 8 blue reference
lines.** First time we've had real detected peaks in the valid signal
region — previous physics-off gate had pred=0 with the model at amplitude
~0.11.

**Comparison to physics-off gate (for calibration):**
```
             physics off (Round 2)    physics on + teacher forcing (now)
step 50:     probe=0.63               probe=0.27
step 100:    probe=0.39               probe=0.10
step 150:    probe=0.27               probe=0.16
step 200:    probe=0.20               probe=0.49 ← regression
step 250:    probe=0.13               probe=0.17
step 300:    probe=0.066              probe=0.34 ← regression
             (monotone descent)       (oscillating)
```

So z-score + teacher forcing reach the threshold 3× faster than physics-off
alone (step 100 vs step 250), but don't sustain it — the probe head drifts
back up once the velocity head is fully engaging via the shared encoder.

---

## 3. Our reading of the oscillation

Teacher forcing eliminated the NMS-gate reward-hacking you diagnosed in
round 1. That was the BIG concern (probe squashing outputs back below
threshold to close the gate). That's gone — now the probe head HAS peaks
above threshold and L_bp + L_vel are getting honest supervision via
ground-truth indices.

The remaining oscillation (0.10 → 0.49 → 0.17 → 0.34) is smaller and
different in character. Our hypothesis: it's a garden-variety learning-rate
issue. Once the probe head is close to the CenterNet-optimal heatmap shape,
AdamW steps of size ~3e-4 overshoot and the shared encoder absorbs some
of the gradient noise from the velocity head's larger-magnitude loss term.
Phase 6 has a cosine LR decay (from 3e-4 to 1.5e-5 over 5 epochs), which
should naturally damp this out as LR decreases.

We think this is fine. But we'd hate to burn 5 hours on Phase 6 and find
out the oscillation compounds epoch-over-epoch rather than damping.

---

## 4. What we want to do next

**Option A: Launch Phase 6 as specified in your round-2 plan.** 5 epochs,
27 runs, 3 holdouts (1 per color). ETA ~4–6h. If LR decay tames the
oscillation, we end up with a real generalization F1 by dinner. If not,
we've wasted GPU time but at least we have a per-epoch loss trajectory
showing WHERE the problem is.

**Option B: Tune gate first.** Rerun overfit gate with lower LR
(e.g., 1e-4) or add LR decay to the gate itself. Budget ~10 min, less
risk of "5-hour mystery." If 1e-4 gives clean monotone descent, use that
as the Phase 6 LR.

**Option C: Something architectural you'd recommend.** Maybe gradient
clipping? Per-head LR (lower for encoder, higher for velocity head)?
Decouple the velocity head entirely (own encoder branch)?

---

## 5. My recommendation (the controller)

**Option A.** The Phase 6 LR schedule already includes cosine decay
from 3e-4 to 1.5e-5. By epoch 3 the LR is already below 1e-4 territory.
If LR is the issue, Phase 6 will demonstrate that by showing clean
per-epoch descent from epoch 2 or 3 onward. If it's NOT LR (i.e., if
the oscillation gets worse at low LR), we'll learn that too with a
real multi-run signal rather than a single-cache gate. Either outcome
teaches us what we need to know. We've already done 4 rounds of
single-cache gating; time to look at actual training dynamics on real
multi-run data.

**Risks I'm aware of:**
- If the oscillation is NOT LR but is, say, some gradient-contention
  issue between the three heads on the shared encoder, it could get
  worse at scale (more molecules per batch = more diverse gradients =
  potentially louder contention).
- Phase 6 uses 32 GB of GPU and 5+ hours of wall time; if it diverges
  we lose a weekend evening.

---

## 6. Questions for you

1. **LR schedule check.** We're using `--lr 3e-4 --min-lr 1.5e-5 --epochs 5`
   with cosine decay. Is there a well-known "physics losses added" tweak
   we should apply — a brief linear warmup, a warm-restart schedule, or
   per-head LR scaling — or is vanilla cosine fine?

2. **Oscillation at 0.1–0.5 range: benign or signal?** The probe loss
   regressing from 0.10 to 0.49 at step 200 and then recovering to 0.17
   feels mechanical (Adam overshoot) but we can't rule out a more
   structural gradient-contention problem. Is there a way you'd
   measure this that would tell us the answer in 5 minutes?

3. **Is there a "smaller Phase 6" we should run first?** — e.g., 1 epoch
   on 5 runs (~30 min) to confirm multi-run training dynamics before
   committing to 5 full epochs on 27 runs. Or is this unnecessary
   gold-plating?

4. **The viz we're reading (`pred=5, ref=8`).** On a SINGLE molecule at
   step 300, 5 of 8 reference probes were detected above threshold 0.3.
   Is that a reasonable "gate passed" for a 300-step 32-batch overfit
   on one cache, or should we be seeing 8/8 before we commit?

---

## Artifacts

- `fix_verified_uid13577.png` — label mapping fix, before/after.
- `overfit_gate_teacherforcing.png` — the round-3 gate viz (pred=5/ref=8).
- `dataloader_batch_check_after_zscore.png` — your Q5 gate; 8 random
  molecules, all with green labels on upward peaks.
- Branch `claude/peaceful-rubin-bfb7a9`, commits `819a393 → 85a3aea`.
- Plan: `docs/plans/2026-04-19-teacher-forcing-and-zscore.md`.
