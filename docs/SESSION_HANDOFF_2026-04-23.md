# Session Handoff — 2026-04-23 09:10 ET

> **For post-compaction Claude:** read this top-to-bottom before doing anything. This captures everything from the V3 sprint that won't be in the compacted history. After reading, status check the running tasks per §3 then ask the user what they want next.

---

## 1. The 30-second context

We're trying to build a model that takes raw TDB nanopore waveforms and outputs accurate inter-probe **basepair distances** for downstream genome assembly. Production benchmark is **legacy T2D power-law** at **16.2% median bp interval error** on 3 holdouts. V1 (CNN trained against remapping with soft-DTW) hits **30.1% median** — worse than production. We pivoted to V3 (physics-informed, L_511 constraint = ∫v dt over each probe must equal exactly 511 bp) which spike+extension hit **25.4% median** — closer to T2D but still 9 points worse.

This week's iteration is **Option A (T2D-hybrid)**: `v_final = v_T2D × (1 + tanh-bounded residual)`. Architecturally guarantees ≤ T2D as the floor. The bet: residual learns local current-driven corrections T2D's smooth power-law misses.

**Status:** Option A's local 3-epoch smoke matched T2D exactly (16.2%) — graceful-degradation worked but residual didn't beat T2D. Per-cache: BETTER on Blue (14.7% vs 17.9%), slightly worse on Black/Red. Suggests there's signal but it's inconsistent.

**Currently running:** local 10×5 long run with two ceiling-break upgrades layered on (probe-aware velocity head + mixed supervision via confidence-gated alignment loss).

---

## 2. What is actually running RIGHT NOW

### a) Local Option A long run — task `b9wi3k5qg`
- **Recipe:** 10 caches × 5 epochs, from-scratch, `--use-l511 --use-t2d-hybrid --probe-aware-velocity --lambda-align 1.0 --align-min-confidence 0.7`
- **Started:** 2026-04-22 ~19:30 ET
- **Pace:** ~7h per epoch (slower than projected — 10-cache scaling is worse than expected)
- **Latest checkpoints:**
  - `option_a_long_checkpoints/checkpoint_epoch_000.pt` (saved 02:24 ET; bp_raw=33472, val=0.365)
  - `option_a_long_checkpoints/checkpoint_epoch_001.pt` (saved ~09:00 ET; bp_raw=32775, val=0.375)
- **ETA for completion:** epoch 5 ≈ 6 AM Friday 2026-04-24
- **What to watch:** the per-probe RMSE trajectory. Smoke was flat (183→180→179 bp). Long is so far also flat (183→181 bp at ep1→ep2). If ep3-5 stay flat, both architectural upgrades didn't help and we have evidence that this configuration is at the T2D ceiling regardless of recipe.

### b) Cloud H100 retry loop — task `bf0ptoik8`
- Looping launcher (`scripts/cloud/launch_loop.sh`) re-fires `launch.sh` every 30s
- Default 60-round budget per cycle, restarts forever
- **Latest:** Round 52/60 of cycle ~3 (so ~150 cycles total). **Zero capacity hits in 18+ hours.**
- Tries 16 zones across us-central1, us-east5, us-east4, us-west4, europe-west4
- Will exit if any `mongoose-*` instance exists (so won't double-provision when A100 lands)

### c) Cloud A100 retry loop — task `b1fyy8f4o`
- Same setup, A100 80GB, 7 zones (us-central1 + us-east5)
- **Latest:** Round 54/60. **Zero capacity hits in 18+ hours.**
- Same auto-deconflict.

### Summary state

```
LOCAL GPU:    busy with Option A long run (epoch 2/5 done, ~22h to finish)
CLOUD GCP:    nothing landing. Both H100 + A100 loops grinding.
PIVOT READY:  Lambda Labs walkthrough doc in this thread, scripts/cloud/lambda_bootstrap.sh ready
```

---

## 3. What user wants you to do post-compaction

**Add wandb to the trainer.** They want experiment tracking with a web dashboard. Spec:

- Install: `pip install wandb` (already in venv? check)
- Add `wandb.init(project="mongoose", name=...)` in `Trainer.__init__` or top of `fit()`
- Log per-epoch: train_loss, val_loss, probe, bp, vel, count, all `*_raw` versions, lr, blend, n_align_active (already in details dict from L511Loss)
- Log hyperparams: dump `config` to wandb at init
- Log git commit hash
- Log GPU info on init
- Make wandb **optional** via config flag (`use_wandb: bool = False`) so it doesn't break runs without API key
- Set WANDB_API_KEY via env var documented in CLI help
- Project name: `mongoose-v3` or `mongoose`

Approx 30-50 lines of code. Test with synthetic.

After wandb is in: add it to `scripts/cloud/lambda_bootstrap.sh` so the eventual cloud run streams to wandb dashboard for free.

---

## 4. Status of core diagnostic data (THE table)

| Method | Holdouts overall median | p95 | Correlation | Notes |
|--------|------------------------:|----:|------------:|-------|
| **T2D (production)** | **16.2%** | 167% | 0.72 | per-channel calibrated, the bar to beat |
| V1 (Phase 6) | 30.1% | 271% | 0.64 | universal model, soft-DTW, 4ep × 27 caches |
| V3 spike (3ep × 4 caches) | 31.9% | 268% | 0.60 | from-scratch L_511 |
| V3 extension (10ep × 4 caches total) | 25.4% | 201% | 0.67 | spike + 7ep more, still descending |
| **Option A smoke (3ep × 4 caches)** | **16.2%** | **188%** | **0.72** | matches T2D exactly; per-cache: Blue 14.7% (BETTER), Black 19.1% (worse), Red 15.0% (slightly worse) |
| Option A long (running) | TBD | TBD | TBD | epochs 1-2 show bp_raw still flat at ~180 bp RMSE; smoke pattern continuing despite architectural upgrades |

**The Blue holdout improvement (14.7% vs T2D 17.9%) is the most interesting datapoint of the sprint.** Suggests Option A's residual CAN learn corrections in some regimes; just not consistently across all caches in 3 epochs. Long run + cloud will tell us if 5 / 40 epochs improves consistency.

---

## 5. Code that landed this sprint (key files)

**Committed in `aa6a9c3`:**
- `src/mongoose/losses/l511.py` — L511Loss with probe + L_511 + L_smooth + L_length terms
- `src/mongoose/io/run_log.py` — RunLog parser (telemetry: bias, NanoPress, baselines)
- `src/mongoose/model/unet.py` — `compute_v_t2d()` + `T2DUNet.forward(t2d_params=...)`
- `scripts/precompute_t2d_params.py` — enriches caches with per-molecule T2D constants
- `scripts/evaluate_bp_intervals.py`, `compare_bp_intervals.py` — bp-domain eval + comparison harness
- `scripts/cloud/{setup,launch,vm_startup,teardown,launch_loop}.sh` — GCP cloud pipeline
- `scripts/cloud/lambda_bootstrap.sh` — Lambda Labs equivalent (uploaded to GCS)
- `tests/test_*` — 22 tests passing across L_511, T2D-hybrid, run_log, legacy_t2d

**Committed in `207304e`:**
- `src/mongoose/inference/legacy_t2d.py` — corrected T2D formula (was 5× off due to unit bug)
- `tests/test_inference/test_legacy_t2d.py` — pinned correctness to hand-verified molecule

**NOT YET COMMITTED (this morning's work):**
- `src/mongoose/model/unet.py` — added `probe_aware_velocity` flag + arch
- `src/mongoose/losses/l511.py` — added `lambda_align` + `align_min_confidence` for mixed supervision
- `src/mongoose/training/{config,cli,trainer}.py` — flag plumbing
- `tests/test_model/test_t2d_hybrid.py` — 3 new tests for probe-aware path
- `scripts/launch_option_a_long.sh` — local 10×5 launcher
- `scripts/cloud/lambda_bootstrap.sh` — Lambda walkthrough
- `scripts/zero_velocity_head.py` — utility to zero vel head before warm-start
- `docs/presentation_brief_2026-04-22.md` — Cowork presentation brief
- `phase6_holdout_eval/*` — eval outputs and plots

You should commit these before doing wandb work. They're stable and tested.

---

## 6. Critical user preferences

1. **Honest framing over diplomatic.** User explicitly asked for "no bullshit" multiple times. Surface uncertainty, don't hedge wins prematurely.
2. **Don't auto-launch big things without explicit approval.** User has been explicit that I should ask before kicking off major training runs (cloud especially).
3. **One in_progress todo at a time.** They like clean todo lists.
4. **Save state to git frequently.** They want commits to mark milestones.
5. **GPU contention warning:** I screwed this up overnight before — running the comparison harness while a training run was using the GPU caused a 4.5h slowdown to one epoch. **Never run GPU workloads concurrently.** CPU-only stuff (parsing logs, computing T2D values) is fine.
6. **User's preferred working dir:** `C:\git\mongoose\.claude\worktrees\peaceful-rubin-bfb7a9` (this worktree).
7. **Python invocation pattern:** `PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src C:/git/mongoose/.venv/Scripts/python.exe` for any script invocation.
8. **User goes by Jon McBee.** Currently driving / working at home + work via TeamViewer.

---

## 7. GCP infrastructure state

- **Project:** `project-mongoose-494111`
- **GCS bucket:** `gs://project-mongoose-494111-mongoose-training` — currently has source tarball, init_checkpoint.pt (zero'd vel head), 30 enriched caches (~12.5 GB total), `lambda_bootstrap.sh`. **MAY BE PUBLIC-READ** if user followed Lambda walkthrough Step 2.
- **Default VPC:** created
- **Cloud NAT:** `nat-config` on `nat-router` in us-central1
- **Quotas approved:**
  - `PREEMPTIBLE_NVIDIA_H100_GPUS` = 128 (us-central1) — already had it from the start, was misleading UI display
  - `GPUS_PER_GPU_FAMILY: NVIDIA_H100` = 2 (us-central1, on-demand) — user requested
  - `NVIDIA_A100_80GB_GPUS` = 2 (us-central1, on-demand)
  - `NVIDIA_A100_80GB_GPUS` = 2 (us-east5, on-demand)
  - `A2_CPUS` = 192 (more than enough)
- **No active VMs.** All capacity attempts have failed.

---

## 8. Open decisions / pending items

1. **Lambda Labs path:** I walked the user through it. They were considering pivoting (cost ~$60). They asked about wandb instead. **If user pivots to Lambda after wandb is added**, the Lambda bootstrap should also include wandb env var injection.

2. **Cloud loops:** still grinding. User can leave them indefinitely (no cost until landing). If user wants to stop them, use TaskStop on `bf0ptoik8` and `b1fyy8f4o`.

3. **Local long run:** runs until ~6 AM Friday unless user wants to kill. **Don't run anything else on local GPU while it's running.**

4. **Mixed supervision lambda tuning:** we set `lambda_align=1.0` and `align_min_confidence=0.7` as initial guesses. If long run shows alignment loss isn't helping, may need to tune.

5. **SkyPilot:** discussed but not adopted. User's view: claude+wandb is sufficient for current scale. Revisit if multi-cloud or team growth.

6. **Future-proofing for V3.x:** RunLog telemetry is already parseable (`src/mongoose/io/run_log.py` + tests). Would add per-molecule bias/NanoPress/baseline as conditioning channels. Not done yet.

---

## 9. Where to find recent eval artifacts

```
phase6_holdout_eval/
├── training_curves.png              ← latest plot, 3 stacked subplots
├── peak_match.json                  ← V1 peak F1
├── bp_intervals.json                ← V1 bp eval
├── t2d_intervals.json               ← T2D baseline eval
├── comparison_spike.json            ← V3 spike vs T2D
├── comparison_ext.json              ← V3 extension vs T2D
├── comparison_option_a_smoke_v2.json ← Option A smoke vs T2D (FIXED eval)
├── viz_STB03-063B.png               ← prediction grid viz
├── viz_STB03-064D.png
└── viz_STB03-065H.png
```

Note `comparison_option_a_smoke.json` (without _v2) is the BROKEN version where the eval script didn't pass `t2d_params`. Don't use it. The fix is in `compare_bp_intervals.py` — checks `config.use_t2d_hybrid` and passes `t2d_params` accordingly.

---

## 10. The wandb integration ask, in one sentence

**Add optional wandb logging to `Trainer` so all per-epoch metrics + hyperparams + git commit hash stream to wandb.ai for live dashboards, opt-in via `--use-wandb` flag (default off so existing runs aren't broken), document `WANDB_API_KEY` env var.** Then bake it into `lambda_bootstrap.sh` for the Lambda run.

---

*End of handoff. Read top-to-bottom. Status-check the running tasks via §2, then ask the user.*
