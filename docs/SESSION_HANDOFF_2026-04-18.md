# Session State — 2026-04-18 PM handoff

**Purpose:** Enable a fresh Claude Code session to pick up this project mid-flight with no prior conversation context. Read this end-to-end before doing anything.

---

## 1. One-Paragraph Context

This is a ML training project for a V1 T2D U-Net peak-detection model on E. coli nanopore translocation waveforms. We spent today (a) fixing GPU perf (~600× speedup), (b) fixing AMP-induced NaN divergence, (c) replacing focal probe loss with BCE-on-logits, (d) adding a probe-head bias init for sparse-target training. Phase 2a (10-epoch single-run smoke) is currently running as the validation that the tuned recipe works.

---

## 2. Current Training Process — COMPLETED (17:40)

- **Phase 2a smoke — DONE.** All 10 epochs completed cleanly. No NaN. See `PHASE2A_RESULTS.md` for full analysis.
- **F1 = 0.037 on best model (epoch 3), 0.000 on final (epoch 10). Below 0.3 gate.**
- **Phase 4 was NOT auto-launched** per user instruction. Model collapsed back to init-bias prior (sigmoid(-3)=0.047) post-warmstart-fade.
- **Log:** `phase2a_train.log`. **Checkpoints:** `phase2a_checkpoints/`. **Eval:** `phase2a_eval_best.json`, `phase2a_eval_final.json`. **Viz:** `phase2a_viz_best.png`, `phase2a_viz_final.png`.

**Summary of what happened:**

Three regimes during the 10 epochs:
1. Warmstart (epochs 1-4, blend=1.0): BCE lifts mean-max amplitude to 0.38, val_loss 6.61 → 1.90 (best).
2. Fade (epochs 5-7): peakiness taking over destabilizes the model; raw_bp spike to 1.4M at epoch 6.
3. Post-fade (epochs 8-10, blend=0.1): heatmap collapses back to init prior. All losses "solved" by doing nothing.

Core diagnosis in PHASE2A_RESULTS.md. Deep Think peer-review prompt drafted in DEEP_THINK_PROMPT.md.

---

## 3. Running Background Processes (likely to die on session close)

| Process | Purpose | Survives session close? |
|---|---|---|
| Phase 2a Python train loop | Main work | **Likely NO** — children of CLI |
| TensorBoard server (port 6006) | Dashboard | **Likely NO** — same reason |
| Any background shell tasks | | NO |

**If Phase 2a died: checkpoint at `phase2a_checkpoints/checkpoint_epoch_N.pt` survives. Resume from there (the Trainer auto-loads latest checkpoint in `_maybe_load_checkpoint`).**

### Relaunch commands

```bash
# Check if Phase 2a is still running
tasklist 2>/dev/null | grep python.exe
tail -10 phase2a_train.log
ls -la phase2a_checkpoints/

# If training died, resume from checkpoint (trainer auto-loads latest)
# Same command as original launch; it will pick up from the last saved epoch:
.venv/Scripts/python.exe -u scripts/train.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --epochs 10 --batch-size 32 --lr 3e-4 \
    --warmstart-epochs 8 --warmstart-fade-epochs 4 \
    --min-blend 0.1 --scale-bp 300000 --scale-vel 5000 \
    --scale-count 50 --scale-probe 1.0 --probe-pos-weight 50 \
    --min-lr 1.5e-5 --checkpoint-dir phase2a_checkpoints --save-every 1 \
    2>&1 | tee -a phase2a_train.log
# Note: -a on tee (append, not overwrite) to preserve prior log

# Restart TensorBoard if needed (points at repo root; picks up all run dirs)
.venv/Scripts/python.exe -m tensorboard.main --logdir=. --port=6006 --bind_all &
```

---

## 4. Decision Gates (IMPORTANT — follow these)

### After Phase 2a completes:

**Run evaluation + visualization:**
```bash
.venv/Scripts/python.exe -u scripts/evaluate_peak_match.py \
    --checkpoint phase2a_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2a_eval.json

.venv/Scripts/python.exe -u scripts/visualize_predictions.py \
    --checkpoint phase2a_checkpoints/best_model.pt \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --output phase2a_viz.png --n-molecules 20 --seed 42
```

**Read F1 from JSON:**
```bash
.venv/Scripts/python.exe -c "import json; d=json.load(open('phase2a_eval.json')); o=d['overall']; m=o['per_molecule_mean']; print(f'F1_sum={o[\"f1\"]:.3f}  F1_mol_mean={m[\"f1\"]:.3f}  P={o[\"precision\"]:.3f} R={o[\"recall\"]:.3f}  n={m[\"n_molecules\"]}')"
```

### Gating rules (USER APPROVAL CHANGED THIS SESSION):

- **Phase 4 does NOT auto-launch.** User wants to approve after reviewing Phase 2a results + manual context compaction. New session should **report Phase 2a outcome and wait for user instruction**, not launch Phase 4.
- **Phase 6 never auto-launches.** Always awaits user approval.

### If the user has returned and approves Phase 4:

```bash
rm -rf phase4_single_run 2>/dev/null
.venv/Scripts/python.exe -u scripts/train.py \
    --cache-dir "E. coli/cache/STB03-060A-02L58270w05-433B23e" \
    --epochs 35 --batch-size 32 --lr 3e-4 \
    --warmstart-epochs 8 --warmstart-fade-epochs 4 \
    --min-blend 0.1 --scale-bp 300000 --scale-vel 5000 \
    --scale-count 50 --scale-probe 1.0 --probe-pos-weight 50 \
    --min-lr 1.5e-5 --checkpoint-dir phase4_single_run --save-every 5 \
    2>&1 | tee phase4_train.log
```

~15 hours. Uses Phase 2a-tuned recipe, 35 epochs, more conservative save-every.

---

## 5. Git State

- **Branch:** `training/v1-recipe-design`
- **HEAD:** `d45e9fb`
- **Upstream base:** `perf/criterion-amp-fixes` (perf + AMP work, separately committed)
- **Not pushed.** All work is local.
- **Recent commits (newest first):**
  - `d45e9fb` fix(preprocess): fall back to Remapping/AllCh when Remapped/AllCh is absent
  - `af24d2a` feat(training): TensorBoard scalar logging per epoch
  - `b1fe202` feat(model,cli): probe-head bias init + expose probe_pos_weight via CLI
  - `01b9bec` feat(loss): replace focal probe loss with positive-weighted BCE-on-logits
  - `f792e37` feat(training): expose --min-lr; log raw loss magnitudes alongside scaled
  - (see `git log --oneline -15` for more)

All 221 tests pass on this branch.

---

## 6. Dataset State

**Preprocessed caches in `E. coli/cache/` — 11 of 30 total:**

| Color | Cached | Total | Missing |
|---|---|---|---|
| Black | 8 | 8 | 0 |
| Blue  | 3 | 13 | 10 |
| Red   | 0 | 9 | 9 |

**Why missing:** 22 of the original 30 runs had their remap output as `Remapping/AllCh/`; a code fix (commit `d45e9fb`) picks those up now — 3 recovered. The remaining 19 Red+Blue runs have TDB files still in `.7z` archives; they need **upstream unzip + remap pipeline work by the user**, which they plan to do tonight in parallel with Phase 4.

**Test holdout design for Phase 6 (MULTI-RUN training):**
- Original spec: 1 held-out per color (Black/Blue/Red).
- Current reality: no Red caches → can't hold out Red.
- **Propose:** hold out alphabetically-last 1 Black + 1 Blue as test runs. Train on remaining 9. If user unzips more runs by Phase 6 launch, we re-derive holdouts from the larger set.

---

## 7. Recipe Config (what's tuned for Phase 2a/4)

```
batch_size = 32
lr = 3e-4
min_lr = 1.5e-5  (was 1e-6 default; cosine schedule was frozen last 2 epochs)
warmstart_epochs = 8
warmstart_fade_epochs = 4
min_blend = 0.1  (focal-vs-peakiness blend never falls below 0.1)
scale_probe = 1.0
scale_bp = 300000  (calibrated to observed raw_bp magnitude)
scale_vel = 5000
scale_count = 50
probe_pos_weight = 50  (positive-sample emphasis for BCE)
```

**Probe loss** = positive-weighted BCE-with-logits against wfmproc Gaussian targets (not focal; not MSE; not plain BCE).

**Probe head init:** `probe_head[-1].bias = -3.0` (sigmoid prior ≈ 0.05, matches sparse-target fraction).

---

## 8. Recipe Evolution Summary (for context if the new session needs to reason about what's going on)

| Attempt | What | Result |
|---|---|---|
| Original focal, α=0.25 | Default from V1 design | Heatmap collapsed to flat-zero. Focal down-weights positives in sparse-target regime. |
| min_blend=0.1 with focal | Keep focal active post-warmstart | Still collapsed. Focal too weak against bp. |
| Count-loss neutralize (scale 1e6) | Remove mass pressure | Heatmap uniform ~0.4 everywhere. No localization. |
| MSE instead of focal | Dense regression | Sigmoid-saturated to near-zero. Gradient died at sigmoid. |
| Positive-weighted MSE | Emphasize peak samples | Still saturated — same sigmoid gradient issue. |
| **BCE-with-logits + pos_weight=50 + init-bias -3** | **Numerically stable + sparse-target prior** | **Phase 2a progressing without NaN or collapse.** |

**What's still uncertain:** whether sparse peak localization emerges over 10 epochs of real data (it didn't in 300-step gate on 1 batch). Data variety + longer training may help.

---

## 9. Dashboards / Monitoring

- **TensorBoard:** http://localhost:6006/ (if server survived). Scans repo root. Phase 2a does NOT emit TB data (trainer hadn't been instrumented when it launched). **Phase 4 and beyond DO emit TB data** via `SummaryWriter(checkpoint_dir/tb)`.
- **Text log:** `phase2a_train.log` has per-epoch loss summaries. Similar files for phase4, phase6 once those run.

---

## 10. Exact Paths

```
Repo root:              C:/git/mongoose
Branch:                 training/v1-recipe-design
Python:                 .venv/Scripts/python.exe (Python 3.12 + torch 2.11+cu128)
Single-run cache:       E. coli/cache/STB03-060A-02L58270w05-433B23e
All caches:             E. coli/cache/
Phase 2a log:           phase2a_train.log
Phase 2a checkpoints:   phase2a_checkpoints/
Design doc:             docs/plans/2026-04-18-v1-training-plan-design.md
Implementation plan:    docs/plans/2026-04-18-v1-training-plan-implementation.md
Eval script:            scripts/evaluate_peak_match.py
Viz script:             scripts/visualize_predictions.py
Overfit gate:           scripts/overfit_one_batch.py
```

---

## 11. Pending Tasks for This Handoff

A fresh session picking up from here should, in order:

1. **Read this whole file.**
2. **Check Phase 2a status:** `tasklist | grep python.exe`, `tail phase2a_train.log`, `ls phase2a_checkpoints/`.
3. **If Phase 2a died:** relaunch per section 3. It auto-resumes from last checkpoint.
4. **If Phase 2a still running:** report current epoch + metrics to the user; wait for completion.
5. **When Phase 2a completes:** run eval + viz (section 4), report F1, DO NOT launch Phase 4 automatically.
6. **Restart TensorBoard if it died** (section 3) — user wants the dashboard available.
7. **When user approves Phase 4:** launch per section 4's command.
8. **Write the Deep Think peer-review prompt** — user asked for one after Phase 2a results are in. See section 12.
9. **Phase 6 stays queued.** Never auto-launch.

---

## 12. Deep Think Peer-Review Prompt — TO BE DRAFTED after Phase 2a completes

User asked for a single comprehensive prompt to paste into Gemini Deep Think for peer review. Constraints: user has 10 prompts per period; this one must be one-shot comprehensive. Should include:

- Crisp problem statement (nanopore translocation → probe detection)
- Architecture summary (T2D U-Net, 3 heads: probe logits / cumulative_bp / raw_velocity)
- Loss formulation (BCE-on-logits + soft-DTW + smooth_l1 count + peakiness; lambda scale schedule)
- Recipe evolution from Section 8
- **Phase 2a actual epoch-by-epoch loss curves + final F1** (fill in from results)
- 3-5 focused questions where outside ML eyes would help

Draft this AFTER Phase 2a + eval, using actual numbers. Template outline is in the Recipe Evolution in Section 8.

---

## 13. What NOT to do

- **Do not auto-launch Phase 4.** User's preference was made explicit mid-afternoon.
- **Do not auto-launch Phase 6.** Always needs approval.
- **Do not modify `src/mongoose/losses/combined.py` or `src/mongoose/training/trainer.py`** while anything is training. The running Python has already imported them; edits don't retroactively apply, but editing them invites confusion.
- **Do not force-push or rewrite history.** Nothing is pushed; preserve local commits.
- **Do not commit to memory or `~/.claude/projects/...`** without user approval. If unsure, ask first.

---

## 14. Anticipated User Prompts When They Return

The user will likely ask one of:
- "How did Phase 2a go?" → F1 number + loss trend + viz link.
- "Let's launch Phase 4." → Use command in section 4.
- "Show me the Deep Think prompt." → Hand over drafted prompt (section 12).
- "What's the multi-run plan?" → 11 caches today; may be more by tomorrow; hold out last-alphabetical per color (Black + Blue; no Red yet).

---

Last updated: 2026-04-18 by Claude (Opus 4.7). Training session: `77a4add5-b18a-4c5f-9445-06f618794692`.
