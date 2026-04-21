# Session handoff — 2026-04-21 mid-morning

**Purpose:** let a fresh session pick up from here without reading the full
prior conversation. Written just before a context-compact. Assume no prior
context in the reader. Read this top-to-bottom first, then dive into the docs
it references.

---

## 1. What's actively happening right now

**Phase 6 V1 production smoke is running on the local A4500 GPU.**

- Task ID: `butd2s60v` (launched 2026-04-19 13:54 via `scripts/launch_phase6.sh`)
- Config: 4 epochs, 27 training caches (3 holdouts: 1 per color — see §2), batch 32, LR 3e-4 cosine → 1.5e-5, CenterNet focal probe + teacher-forced physics (L_bp + L_vel), L_count disabled (scale_count=1e9), min_blend=1.0, bf16 AMP
- Checkpoints: `phase6_checkpoints/` (in the worktree) — epochs 0, 1, 2 saved; best_model.pt selected by val_loss
- Status at time of this writeup:
  - Epoch 1 done 2026-04-20 01:45 (train=0.29, val=0.31, probe_raw=0.23, bp_raw=15406, vel_raw=354)
  - Epoch 2 done 2026-04-20 13:28 (train=0.23, val=0.22, probe_raw=0.19, bp_raw=8393, vel_raw=232)
  - Epoch 3 done 2026-04-21 01:08 (train=0.21, val=0.21, probe_raw=0.17, bp_raw=5008, vel_raw=194)
  - Epoch 4 running — ETA ~12:50 today based on ~11h40m-per-epoch pace
- GPU: 100% util, 20.1 GB VRAM steady, 63°C

**Training trajectory is healthy.** val_loss dropped 32% ep1→ep2, plateau-bent to 2% drop ep2→ep3 (val_loss is dominated ~80% by `probe_raw`, which is approaching its asymptote). But `bp_raw` is still dropping ~40% per epoch — the velocity head is still sharpening. Epoch 4 should give us a final bp_raw around 3500-3800 and our 3-holdout F1 numbers.

**Do not touch the worktree while training is running.** The PID 99732 Python process is writing to `phase6_checkpoints/` continuously.

---

## 2. Where we are in the 48-hour sprint (what's been done)

The sprint goal: fix V1's recipe, launch a credible production training run. As of right now we have:

### 2a. Code and data fixes (all committed on branch `claude/peaceful-rubin-bfb7a9`, pushed to origin)

**The critical bug:** probe labels were in the wrong sample indices.
- Root cause: `src/mongoose/data/ground_truth.py` hardcoded `SAMPLE_RATE_HZ = 40_000` but TDB is 32 kHz, AND the sample-index formula ignored `mol.start_within_tdb_ms` (the molecule-start offset within the TDB block's waveform).
- Fix: `26f84ec` — take `sample_rate_hz` from the TDB header, include the offset: `(start_within_tdb_ms + center_ms) * sample_rate_hz / 1000`.
- Effect: Phase 2a v1 had F1=0.033 (recall 1.7%) because labels were on baseline noise. After fix: Micro Phase 6 F1=0.917 (recall 98.8%) on a never-seen-before holdout, after ONLY 1 epoch on 3 caches.

**Other commits in order:**
- CenterNet focal loss (Zhou et al. 2019) replaces BCE probe loss — normalizes by num_positives per molecule, sequence-length invariant
- Z-score per-molecule waveform normalization in `collate_molecules` — fixes 1e-4 amplitude problem that was silently breaking BF16 precision floor and PyTorch's Kaiming init assumptions
- Teacher-forced L_bp and L_vel at ground-truth indices — bypasses the non-differentiable NMS graph break that caused "reward-hacking" oscillations
- `scripts/plot_random_dataloader_batch.py` — mandatory post-data-pipeline viz; Deep Think's Q5 recommendation
- `scripts/visualize_predictions.py` fix — was silently feeding raw (unscaled) waveform to model; caught during Micro Phase 6 F1 verification

**Phase 6 recipe (current training, what works):**
```
CenterNet focal probe loss (pos_threshold=0.99, α=2, β=4, num_pos-normalized)
+ L_bp: teacher-forced soft-DTW at ground-truth reference indices
+ L_vel: teacher-forced MSE at ground-truth indices (FWHM from warmstart heatmap)
+ L_count: disabled via scale_count=1e9
+ z-score per-molecule collate normalization
+ min_blend=1.0 (warmstart active throughout)
+ grad_clip_norm=1.0
```

### 2b. Holdouts (same across Micro Phase 6 and full Phase 6)

Alphabetically-last per color:
- Black: `STB03-063B-02L58270w05-433B23b`
- Blue: `STB03-065H-02L58270w05-433H09j`
- Red: `STB03-064D-02L58270w05-433H09d` ← also the Micro Phase 6 holdout (F1=0.917 there after 1 epoch)

### 2c. 30 caches available (all regenerated with corrected labels)

Located at `E. coli/cache/<run_id>/` — 8 Black + 13 Blue + 9 Red = 30 runs, ~1.25M total molecules. Regenerated via `scripts/regen_all_caches.sh` after the label-mapping fix. Old broken caches archived to `E. coli/cache_broken_labels/`.

### 2d. All sprint docs consolidated on the branch (committed + pushed)

```
docs/
├── SESSION_STATE.md                              ← pre-sprint handoff (historical)
├── SESSION_HANDOFF_2026-04-21.md                 ← THIS FILE
├── plans/
│   ├── 2026-04-18-centernet-focal-loss-rescue.md ← V1 round-1 plan (executed)
│   ├── 2026-04-19-teacher-forcing-and-zscore.md   ← V1 round-2 plan (executed)
│   ├── 2026-04-20-cloud-production-training.md   ← GCP spec (H100 primary, A100 fallback)
│   ├── 2026-04-20-v2-design.md                   ← V2 full design (6 Deep Think rounds)
│   └── 2026-04-21-v2-smoke-implementation.md     ← V2 Strong-Smoke plan for TODAY (23 tasks)
└── peer_reviews/
    ├── 2026-04-18-round1-prompt.md
    ├── 2026-04-18-round2-zscore-teacherforcing.md
    ├── 2026-04-19-round3-overfit-gate.md
    └── 2026-04-19-round4-microphase6.md
```

Pushed to `origin/claude/peaceful-rubin-bfb7a9` — clone from any machine, git pull, have everything.

---

## 3. What's happening NEXT (in order)

### 3.1 When Phase 6 finishes (~12:50 today)

1. Tail `phase6_train.log` to capture the final `Epoch 4/4` summary line
2. Run eval on all 3 holdouts:
   ```bash
   cd C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9
   for holdout in STB03-063B-02L58270w05-433B23b STB03-065H-02L58270w05-433H09j STB03-064D-02L58270w05-433H09d; do
       PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
       C:/git/mongoose/.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
           --checkpoint phase6_checkpoints/best_model.pt \
           --cache-dir "C:/git/mongoose/E. coli/cache/$holdout" \
           --output "phase6_eval_${holdout}.json"
   done
   ```
3. Run viz on each holdout:
   ```bash
   for holdout in STB03-063B-02L58270w05-433B23b STB03-065H-02L58270w05-433H09j STB03-064D-02L58270w05-433H09d; do
       PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
       C:/git/mongoose/.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
           --checkpoint phase6_checkpoints/best_model.pt \
           --cache-dir "C:/git/mongoose/E. coli/cache/$holdout" \
           --output "phase6_viz_${holdout}.png" --n-molecules 5 --seed 42
   done
   ```
4. Commit artifacts: `git add phase6_train.log phase6_eval_*.json phase6_viz_*.png`, commit, push
5. Write `docs/peer_reviews/2026-04-21-round5-v1-phase6.md` with per-epoch trajectory + per-holdout F1 + comparison to Micro Phase 6

### 3.2 Final code review of V1 sprint

Dispatch a code-reviewer subagent over the full branch diff. Catch any sloppy-during-sprint bits before merging. Specifically:
- Any dead code / commented-out debugging
- Any docstring staleness (esp. in `combined.py` where we removed L_velocity-era params)
- Any unit-test coverage gaps
- Fix-forward whatever comes up in one or two commits

### 3.3 Cleanup branches and worktrees (REQUIRED before V2 work starts)

The user was emphatic: "before I can build V2, and after the current run finishes we MUST clean up our branches and worktrees." This is step-3 not step-5.

Sequence:
```bash
# In main repo — bring sprint branch into training/v1-recipe-design
cd C:/git/mongoose
git fetch origin
git checkout training/v1-recipe-design
git merge claude/peaceful-rubin-bfb7a9       # fast-forward merge (no conflicts expected)
git push origin training/v1-recipe-design    # push to GitHub for cloud access

# Remove the worktree
git worktree remove .claude/worktrees/peaceful-rubin-bfb7a9

# Delete the feature branch
git branch -d claude/peaceful-rubin-bfb7a9
git push origin --delete claude/peaceful-rubin-bfb7a9
```

**Decision point to confirm with user first:** GitHub's default branch is `claude/angry-sinoussi`. Is `training/v1-recipe-design` the true main? Or should we open a PR for team review? User prefers PR route for narrative-diff view and CI (mentioned previously).

### 3.4 Two parallel tracks after cleanup

**Track A — Cloud production (~2-3 hours):**
- `docs/plans/2026-04-20-cloud-production-training.md` has the full spec
- TL;DR: a3-highgpu-1g preemptible (H100 80GB primary) / a2-ultragpu-1g (A100 80GB fallback). A100 40GB is NOT enough — Windows Task Manager showed 37.5 GB of "shared GPU memory" being used (PyTorch spilling to PCIe-mapped system RAM because peak working set is >40 GB)
- Pre-flight code changes: `torch.cuda.empty_cache()` between train/val, Dockerfile, requirements-lock, auto-resume shell loop wrapper for spot preemption
- Cost estimate: ~$75-155 for a 35-epoch run

**Track B — V2 Strong Smoke (~6-8 hours):**
- `docs/plans/2026-04-21-v2-smoke-implementation.md` has the full 23-task plan (5 tiers)
- New worktree: `.claude/worktrees/v2-smoke/` on new branch `v2-smoke-test` branched off `training/v1-recipe-design` after the merge in §3.3
- Scope: implement V2 backbone + 5 losses, pass A2a gradient sanity (free ablation), overfit-one-batch convergence, 1-epoch Micro V2 Phase 6 on 3 V2-regenerated caches, holdout F1 > 0.5
- Risks called out in the plan: hand-rolled Sinkhorn NaN traps (use torch.where pattern, NOT naive -inf masking), FlashAttention fallback (verify A4500 Ampere support), class-weight hack (use uniform [1,10,10,5,15,30] for smoke, recompute later)

---

## 4. V2 design — what's locked and why

**`docs/plans/2026-04-20-v2-design.md`** is the authoritative spec. 6 rounds of Deep Think review. Do not modify without a round-7 review.

V2's three goals:
1. Wfmproc-independent peak localization (drop wfmproc as positional supervision, keep it as warmstart scaffolding only, fade to zero by epoch 5)
2. Probe-vs-non-probe discrimination (new 6-class segmentation head)
3. Structured/folded molecule coverage (stop dropping them at preprocess, teach model to identify defect regions and mask peak losses accordingly)

V2's five loss terms (V1's L_velocity is REMOVED):
- L_bp: bidirectional translation-invariant Sinkhorn-OT (batched, NaN-safe padded-row handling)
- L_count: dynamic target synchronized to anchor schedule
- L_segmentation: 6-class CE with class weights from cache statistics
- L_probe_anchor: wide-boxcar positive + narrow-negative anchors, faded to zero over 5 epochs
- L_heatmap_sharpness: negative L2 norm under L1 constraint (post-peel concentration regularizer)

V2's architecture: 8x-Resolution Transformer U-Net (RoPE + AdaLN + FlashAttention-2, 8 blocks at 8x downsample, 12-value FiLM metadata conditioning).

**Critical V2 implementation gotchas from Deep Think round-6:**
- `nn.PixelShuffle` is hardcoded for 4D tensors, so we need einops `rearrange(x, 'b (c r) t -> b c (t r)', r=2)` — the V2 spec documents this
- Sinkhorn `logsumexp([-inf, ...]) = -inf` whose gradient is softmax([-inf,...]) = NaN, which poisons weights on the first backward. Must use `torch.where` to neutralize padded rows BEFORE the reduction, restore -inf on output
- `(max_pool(m) - m)²` sharpness formulation is MINIMIZED by flat heatmaps. Use negative L2 norm instead
- L_velocity target of `511 bp / FWHM_ms` breaks in V2 because the V2 heatmap is decoupled from physical tag width. DROP L_velocity entirely
- Full batching discipline: NO `for b in range(B)` loops anywhere in V2's loss path

---

## 5. Where the key files and numbers live

**Git state (branch `claude/peaceful-rubin-bfb7a9`, pushed to origin):**
- Latest commit: `fac22e0` "docs: V2 Strong-Smoke implementation plan"
- Previous: `79d2cfe` "docs: V2 design spec", `0a82574` "docs: SESSION_STATE.md snapshot", `9d479a1` "docs: consolidate sprint artifacts"
- Label mapping fix: `26f84ec` "fix(gt): correct probe sample mapping"
- CenterNet focal: `27f48ee`, teacher forcing: `3d15f8c`, z-score: `819a393`

**Current run artifacts (NOT committed until Phase 6 finishes):**
- `phase6_checkpoints/best_model.pt` (~507 MB, excluded from git)
- `phase6_checkpoints/checkpoint_epoch_{000,001,002}.pt` (each ~507 MB)
- `phase6_train.log` (to commit after epoch 4)
- `phase6_checkpoints/tb/events.out.tfevents.*` (TensorBoard data)

**Working directory:** `C:\git\mongoose\.claude\worktrees\peaceful-rubin-bfb7a9\`

**Python path pattern (always prepend for any script invocation):**
```bash
PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src \
C:/git/mongoose/.venv/Scripts/python.exe <script>
```

---

## 6. User preferences and conversation patterns to carry forward

- The user is the project lead. Domain expertise: ex-Nabsys engineer, deep nanopore-physics / probe-chemistry / data-format knowledge.
- The user wants their own voice heard in writeups. When asked to draft something, include explicit "my recommendation" sections and push back on suggestions rather than hedging.
- The user uses "Deep Think" (Gemini) as an external peer reviewer. Writeups for Deep Think go in `docs/peer_reviews/` and should be self-contained, include concrete numbers, and ask 3-5 focused questions. We've done 4 rounds so far (one per major design iteration), one more expected after Phase 6.
- When Phase 6 finishes, expect a Deep Think round-5 request.
- User is fine going past original timeline estimates if val_loss is still descending (said so explicitly about the 42h budget becoming 48h).
- User cares about correctness more than speed. Said so: "I'd rather spend a weekend chasing a subtle bug than ship something that silently produces wrong answers."
- User is NOT a classical ML researcher. Their ML intuition comes from this specific dataset. Explain concepts in terms of the physics + the data pipeline, not in terms of ML jargon unless specifically asked.

---

## 7. Strategic discussions this session that don't live elsewhere

### Aligner / remapping dependency (recent conversation topic)

The current training pipeline depends on 3 Nabsys tools upstream:
1. **wfmproc** — detects probe events in the waveform (V2 partially replaces this: wfmproc's output becomes warmstart scaffolding that fades to zero after epoch 5)
2. **Aligner** (produces `.assigns` file) — matches observed probe sequences to reference probe positions (V2 still depends on this; "wfmproc independence" ≠ "aligner independence")
3. **Reference map** — precomputed list of all probe binding sites on the E. coli genome (neither V1 nor V2 plans to replace this; it's a fixed artifact of probe chemistry design)

Strategies for reducing aligner dependence (from conversation):
- **S1:** Reimplement the aligner in-house. ~4-6 weeks 1 engineer. Full audit access.
- **S2:** Iterative self-improvement — use V2's outputs to correct aligner errors, retrain. Natural semi-supervised loop.
- **S3:** End-to-end with our own inference-time alignment. Training still uses aligner; deployment doesn't.
- **S4:** Joint probe detection + alignment (truly end-to-end differentiable). Research scope.
- **S5:** Physics simulation + fine-tuning (basecaller-style). 6-12 months, largest impact, largest risk.

Recommended next step for V3: S1 + S3 in parallel, then layer S2 on top. S4 and S5 are research bets for later.

### F1 interpretation / what we've actually achieved (for future context)

- V1 pre-sprint (broken labels): F1 = 0.033, recall = 0.017
- Micro Phase 6 (1 epoch, 3 caches, corrected labels): F1 = 0.917, recall = 0.988
- Phase 6 ep1 (1 epoch, 27 caches): val_loss = 0.31, probe_raw = 0.23 → peak amplitudes ~0.75
- Phase 6 ep3 (3 epochs, 27 caches): val_loss = 0.21, probe_raw = 0.17 → peak amplitudes ~0.83
- Phase 6 ep4: projected val_loss ~0.20, bp_raw ~3500-3800

The 27× F1 improvement from broken-labels → fixed was the single biggest win of the sprint. Everything else — CenterNet focal, z-score, teacher forcing — was lesser contribution that was masked by the label bug until we found it.

### V1 vs V2 positioning (team-meeting summary I wrote for user earlier)

**V1 replaces wfmproc's peak detection. V2 replaces wfmproc entirely, including the data filtering, the structure handling, and the "what counts as a real probe" decision.**

- V1: conservative, well-validated (Phase 6 numbers pending), production-ready path. Retains wfmproc dependency for peak-center supervision.
- V2: ambitious, requires full implementation (the 23-task plan in `docs/plans/2026-04-21-v2-smoke-implementation.md`). Segmentation head enables structured-molecule rescue. Sinkhorn-OT fixes V1's NMS graph break. ~30% of Nabsys's data becomes usable (structured/folded molecules).

---

## 8. One-sentence status

**Phase 6 is in epoch 4 of 4, numbers are excellent, ETA ~12:50 today, then cleanup-then-two-parallel-tracks (cloud V1 production on H100 / Strong V2 smoke on this desktop).**
