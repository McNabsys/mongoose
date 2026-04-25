# Session Handoff — 2026-04-25

> **For post-compaction Claude:** read top-to-bottom. This carries everything the
> two-day Phase 0 + cloud-ablation session won't preserve through compaction.
> Live state and pending decisions are in §1; everything else is reference.

---

## 1. Critical live state — pending Jon's decision

### 1.1 Training is RUNNING on a Lambda H100

A 20-epoch Option A ablation (`--lambda-align` dropped, otherwise identical to
the local long run) has been training on a Lambda Cloud H100 since
**2026-04-24 20:43:36 UTC**. It is still running at the time of writing
(2026-04-25 ~12:47 UTC).

| handle | value |
|---|---|
| VM | `ubuntu@192.222.52.84` |
| SSH key | `ssh/peaceful-rubin.pem` (in worktree, `chmod`-fixed via `icacls`) |
| Per-epoch time | ~2 hours |
| Recipe | 27 caches × 20 epochs, batch 32, lr 3e-4, num_workers 8, no `--lambda-align` |
| tmux session `train` | the training loop |
| tmux session `eval` | per-epoch holdout eval watcher (Python) |
| Eval script on VM | `/home/ubuntu/eval_watcher.py` |
| Source on VM | `/home/ubuntu/mongoose/` (extracted tarball + venv) |
| Caches on VM | `/home/ubuntu/caches/` (12 GB, 30 dirs) |
| Checkpoints | `/home/ubuntu/checkpoints/checkpoint_epoch_NNN.pt` (485 MB each) |
| Train log | `/home/ubuntu/train.log` |
| Eval log | `/home/ubuntu/eval.log` |
| Training wandb run | `wbwav0fw` — `option-a-no-align-20ep` in `mcbee-nabsys/mongoose-v3` |
| Eval wandb run | `7niiw564` — `option-a-no-align-20ep-eval` in same project |

Cost meter: **~$48 spent at 2026-04-25 12:47 UTC; $72 remaining if the run
completes 20 epochs.**

### 1.2 The result so far is decisive enough that Jon was leaning toward kill

Per-epoch holdout overall medians (T2D baseline = 16.2%):

| epoch | overall median | warmstart blend | vs T2D |
|---:|---:|---:|---:|
| 3 | **15.87%** | 1.000 (full) | **−0.33 pts (V3 wins!)** |
| 4 | 16.93% | 0.500 (fading) | +0.73 |
| 5 | 17.71% | 0.050 (off) | +1.51 |
| 6 | 17.23% | 0.050 | +1.03 |
| 7 | 17.84% | 0.050 | +1.64 |

Read: **V3 peaked at epoch 3 while warmstart was forcing the heatmap, beat T2D
by 0.33 pts. The moment warmstart faded, V3 regressed past T2D.** The probe
detector improved (probe loss 0.23 → 0.01) but the bp/velocity prediction
didn't translate to holdout bp-interval improvement. Same conclusion as the
local long run, confirmed at higher resolution: **probe-aware-velocity +
T2D-hybrid is the wrong architectural lever.** Dropping `--lambda-align` did
not change the answer.

### 1.3 What needs to happen next (decisions Jon owes the new session)

1. **Kill the Lambda VM run? Yes / No / let it finish epoch 10 first.** I was
   recommending kill at the point of writing this handoff. If you let it run
   to completion, the trajectory is overwhelmingly likely to stay flat-to-
   worsening through epoch 20 (24h, $72 from now).

2. **Before killing, scp `best_model.pt`** (currently the epoch-3 weights —
   that's the version that beat T2D — though best_val_loss tracks val, not
   holdout, so this might be a different epoch by now). Worth grabbing for
   later inspection / paper-figure replication. ~485 MB.

3. **Tear down the Lambda VM after** (Lambda dashboard, terminate instance).

4. **Active background watcher** task `b5q0oomo3` is polling for
   `checkpoint_epoch_009.pt`. Will fire when epoch 10 saves (Sat ~10:54 AM
   EDT). Either let it complete naturally or stop it via TaskStop — does not
   affect the training itself.

---

## 2. What we accomplished this session

Five substantial work blocks since 2026-04-23, in order:

### 2.1 Phase 0 ETL — unified probe table (committed)
- `src/mongoose/etl/` — schema, parsers, builder, orchestrator
- 56,531,725 probe rows × 128 columns, 2.6 GB merged parquet
- Per-run shards at `data/derived/probe_table/<run_id>.parquet`
- Manifest at `data/derived/probe_table_manifest.json`
- Five commits: parsers/schema → per-run builder → join-rule fix → orchestrator
  → tests
- Phase 0 spec: `docs/plans/phase0_etl_spec.md` (the markdown was originally
  in `Downloads/`, copied via system-reminder during planning)

### 2.2 Phase 0b — classifier characterization (committed)
- `src/mongoose/analysis/phase0b_classifier_characterization.py`
- Reports at `reports/phase0b/` (markdown + JSON + CSVs + plots)
- Headline: narrow MCC 0.343, broad MCC **0.798**. Remapping-side filter is
  the stack's primary specificity defense (6.13 M FP→TN shift)
- `attr_folded_start` flagged as over-aggressive
- Pivoted from T2D-proximity oracle → Mahalanobis envelope mid-build (kappa
  was 0.17, way below the spec's 0.7 threshold)
- 15 unit tests, all green

### 2.3 Phase 0a — T2D residual decomposition (committed)
- `src/mongoose/analysis/phase0a_t2d_residual_decomposition.py`
- Reports at `reports/phase0a/`
- Headline: **27.9% structured variance, 72.1% noise floor.** A better T→D
  could reduce error by up to ~28%; bigger wins probably aren't there.
- Strongest axes: velocity (14× spread D1→D10), `attr_in_structure` /
  `attr_folded_start` (~10× residual lift), head-dive nonlinearity
- Refuted spec hypothesis: `molecule_stretch_factor` r ≈ 0 (spec said it'd be
  the most informative; it's not)
- Refuted Blue-holdout hypothesis: Blue's local-velocity CV (0.697) is *lower*
  than peers, not higher — Option A's Blue advantage is not from velocity
  variation
- 8 unit tests, all green

### 2.4 Eval harness improvements (uncommitted, in worktree)
- `scripts/stratified_interval_eval.py` — V3 vs T2D stratified by
  attr-bit / velocity decile / position-along-molecule. Found:
  - Cache filter drops all `attr_in_structure` molecules → bit stratification
    is vacuous on the cache. The 10×-residual population isn't visible to V3
    eval.
  - V3 wins by 5–6 pts in head bin (0–10%), loses 3–5 pts across mid-and-tail.
  - V3 loses at every velocity decile (worst on slow molecules — opposite of
    what probe-aware-velocity head was designed for).
- `scripts/plot_tag_intervals_comparison.py` — single-molecule 3-row sparkline
- `scripts/plot_tag_intervals_gallery.py` — 6×3 grid (18 panels)
- `scripts/plot_tag_interval_bars.py` — interval-length bar chart
- `scripts/build_molecule_library.py` + `src/mongoose/analysis/molecule_viewer.py`
  — produced 100-molecule × 3-holdout deep view with HTML index at
  `phase6_holdout_eval/molecule_library/index.html`
- `scripts/compare_bp_intervals.py` patched to load `probe_aware_velocity`
  from saved config (or it would crash on Option A checkpoints)
- `src/mongoose/training/cli.py` patched to add `--num-workers` flag

### 2.5 Lambda H100 ablation (in flight, see §1)
- Pre-eval baseline: local Option A long run finished at **18.5% median,
  T2D = 16.2%** (V3 lost by 2.3 pts). Blue narrowly won at 16.9% vs 17.9%.
- Cloud ablation (this session's work): see §1.2.

---

## 3. The next direction Jon flagged

**Jon has Nabsys signal-processing docs to share next session.** The
hypothesis driving this: Phase 0a + 0b said the structured variance V3 could
go after lives in (a) head-dive nonlinearity, (b) non-clean-region probes, (c)
slow-molecule velocity calibration. The current architecture (probe-aware
velocity + T2D-hybrid + L_511) hasn't captured any of these. Nabsys's docs
about the production signal-processing pipeline may show *what they already
do* in those regions — and surface architectural patterns we should adopt.

Topics worth looking for in those docs:
- How Nabsys handles head-dive in production (per-channel calibration? a
  position-aware velocity model? a head-region exclusion zone?)
- The TVC (translocation velocity compensation) algorithm details
- Probe-width filtering logic — the `expected_min_probe_width_factor` etc.
  came from `_remapSettings`, but I never saw a derivation
- Any feature-engineered signals beyond what's in `probes.bin` (e.g., raw-
  current statistics, spectral features)

The Phase 0 ETL knows how to ingest these signals if they're in any of the
existing files (probes.bin, .assigns, M1_probeWidths, etc.). New signals would
need ETL extension.

---

## 4. Repository state — cleanup is needed

### 4.1 Git status as of handoff write

Branch: `claude/peaceful-rubin-bfb7a9` (worktree). Trunk per `origin/HEAD` is
`claude/angry-sinoussi`. Jon is conceptualizing a "main" — clarify whether
that's `claude/angry-sinoussi` or whether he wants a `main` branch created.

**Uncommitted but worth keeping** (working changes):
- `M .gitignore` — added `ssh/`, `*.pem`, `*.key`
- `M scripts/compare_bp_intervals.py` — `probe_aware_velocity` config-load fix
- `M src/mongoose/training/cli.py` — `--num-workers` flag
- `?? scripts/build_molecule_library.py`
- `?? scripts/plot_tag_interval_bars.py`
- `?? scripts/plot_tag_intervals_comparison.py`
- `?? scripts/plot_tag_intervals_gallery.py`
- `?? scripts/stratified_interval_eval.py`
- `?? src/mongoose/analysis/molecule_viewer.py` (it's there, not yet listed in
  git status output we captured)
- `?? phase6_holdout_eval/comparison_option_a_long.json`
- `?? phase6_holdout_eval/stratified_option_a_long.json`
- `?? data/derived/probe_table_manifest.json` already committed; the
  per-shard parquets (~3 GB) and the merged parquet (2.6 GB) are NOT
  committed (large binary, deliberately)

**Untracked junk that should be `.gitignore`'d** (training artifacts):
- `*_checkpoints/` directories at root (`l511_spike_checkpoints/`,
  `option_a_long_checkpoints/`, `phase6_checkpoints/`,
  `micro_phase6_checkpoints/`, etc.)
- `*.log` at root (`overfit_run*.log`, `phase6_train.log`,
  `option_a_long_train.log`, etc.)
- `*.png` at root (`overfit_run*.png`, `Black_STB03-*.png`,
  `label_diagnostic.png`, etc.)
- `phase6_holdout_eval/molecule_library/` (300 generated PNGs + HTML; ~50 MB)
- `phase6_holdout_eval/molecule_library_smoke/` (test-run leftover)

**Worktree size: ~21 GB** (mostly checkpoints + parquet shards + cached molecule
library). Git ignores most of it via existing `.gitignore` (`checkpoints/`,
`__pycache__/`, etc.) — but the root-level training artifacts predate the
ignore rules and are still in the workspace.

### 4.2 Cleanup plan (for the new session to execute)

1. **Stage and commit the uncommitted-but-keep code:**
   - `.gitignore` (the security hardening)
   - `scripts/compare_bp_intervals.py` (probe_aware_velocity load fix)
   - `src/mongoose/training/cli.py` (num_workers flag)
   - `scripts/{build_molecule_library, plot_tag_intervals_*, plot_tag_interval_bars, stratified_interval_eval}.py`
   - `src/mongoose/analysis/molecule_viewer.py`
   - `phase6_holdout_eval/{comparison,stratified}_option_a_long.json` (eval
     artifacts; small JSON, useful provenance)

2. **Extend `.gitignore`** to cover the messy untracked artifacts:
   ```
   # Training artifacts (root-level)
   *_checkpoints/
   *_train.log
   overfit_*.log
   overfit_*.png
   label_*.png
   fix_verified_*.png
   Black_*.png
   phase6_holdout_eval/molecule_library*/
   ```

3. **Consider deleting (NOT committing) the large local artifacts** since they
   can be regenerated from raw data if needed:
   - `data/derived/probe_table/` (3 GB shards) — regenerable via `build_probe_table.py`
   - `data/derived/probe_table.parquet` (2.6 GB merged) — regenerable
   - `phase6_holdout_eval/molecule_library/` (~50 MB) — regenerable via
     `build_molecule_library.py`
   - `*_checkpoints/` directories — old training artifacts, no longer needed

   **Don't delete without asking.** Some checkpoints (e.g.,
   `option_a_long_checkpoints/best_model.pt`) might still have value.

4. **Roll up to main:** clarify with Jon what "main" means
   (`claude/angry-sinoussi`? a fresh `main` branch?), then either:
   - Open a PR from `claude/peaceful-rubin-bfb7a9` → main; let GitHub handle it
   - Or fast-forward / rebase locally and push

5. **Don't kill the Lambda VM until §1.3 decisions are made.** That's a Jon
   call.

### 4.3 Things that should survive any cleanup

- All committed source / tests / docs
- `docs/SESSION_HANDOFF_2026-04-25.md` (this file — gets you here)
- `docs/SESSION_HANDOFF_2026-04-23.md` (previous one)
- `data/derived/probe_table_manifest.json` (small, already committed)
- Reports under `reports/phase0a/`, `reports/phase0b/` (already committed)

---

## 5. Operational reminders Jon emphasized this session

1. **Honest framing over diplomatic.** Surface uncertainty, don't hedge wins
   prematurely. Repeated multiple times.
2. **Stop and surface structural findings before building around them.**
   Worked twice this session — the narrow-classifier oracle tautology
   (Phase 0b) and the T2D-proximity calibration failure (also Phase 0b)
   would have produced subtly miscalibrated reports if we'd built around
   them silently.
3. **One in_progress todo at a time.** Keeps todo lists clean.
4. **GPU contention rule** (locally): no GPU work concurrent with a running
   local training. Cloud is fine.
5. **Don't auto-launch big things without explicit approval.**
6. **Save state to git frequently** at meaningful milestones.
7. **Working dir:** `C:\git\mongoose\.claude\worktrees\peaceful-rubin-bfb7a9`.
8. **Python:** `C:/git/mongoose/.venv/Scripts/python.exe` with
   `PYTHONPATH=C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9/src`.
9. Jon goes by **Jon McBee**. Currently on PowerShell, not bash. SSH is
   Windows-native OpenSSH — NTFS ACLs matter (use `icacls`, not `chmod`).

---

## 6. Prompt for the new session

Copy this to start a fresh Claude Code session:

> Welcome back. Read `docs/SESSION_HANDOFF_2026-04-25.md` top-to-bottom before
> doing anything. It captures everything from this week's Phase 0 / 0a / 0b /
> Lambda H100 work that won't be in the compacted history.
>
> Three things I want from you, in this order:
>
> 1. **Decide the Lambda VM run.** Per the handoff §1.2, V3 peaked at epoch 3
>    (15.87% beating T2D's 16.2%) while warmstart was forcing the heatmap,
>    then regressed once warmstart faded. The conclusion is consistent with
>    the local long run: probe-aware-velocity + T2D-hybrid + L_511 isn't the
>    right architectural lever. **Recommend kill or finish.** If kill: scp
>    `best_model.pt` down first (it's the V3-beat-T2D snapshot — useful for
>    later inspection), then terminate the Lambda VM.
>
> 2. **Clean up the worktree.** Per handoff §4.2:
>    - Commit the uncommitted-but-keep changes (gitignore hardening,
>      compare_bp_intervals.py fix, cli.py `--num-workers`, the new scripts
>      under `scripts/`, the analysis module, the eval JSONs).
>    - Extend `.gitignore` to cover root-level training artifacts.
>    - Ask me before deleting any checkpoint directories or the `molecule_library`.
>    - "Roll up to main" — but ask me first whether "main" means `claude/angry-sinoussi`
>      (current `origin/HEAD`) or whether I want a fresh `main` branch. Open a PR
>      vs. fast-forward — propose the cleanest option.
>
> 3. **Receive the Nabsys signal-processing docs I'm about to drop.** Per
>    handoff §3, our next architectural move depends on understanding what the
>    production pipeline already does about head-dive, non-clean-region probes,
>    and slow-molecule velocity calibration — the three axes Phase 0a flagged
>    as the dominant structured error sources. Read the docs end-to-end, surface
>    any patterns / signals that aren't yet in the Phase 0 ETL or the V3 model,
>    and propose 2–3 concrete architectural directions to evaluate next. **Don't
>    start coding anything yet** — propose first, wait for my approval.
>
> Honest framing > diplomatic. Surface ambiguity. Stop and ask before doing
> anything that destroys state (cloud VM, large files, branch operations).

---

*End of handoff. Read top-to-bottom. Execute §1 first, §4 second, §3 last.*
