# Desktop Session Playbook -- Local Preprocessing + Smoke Test

**Goal for tonight's session:** Get the repo running on your desktop, validate one run with the sanity-check script, batch preprocess all 30 runs, and run a local smoke test on real data.

**Estimated time:** 2-4 hours wall clock, mostly waiting on preprocessing.

---

## Part 1 -- Pre-Session Setup (do this before starting Claude)

### 1. Clone the repo

Pick a path. **Recommendation: use `C:\git\mongoose`** -- this matches your laptop and keeps the project memory continuous between machines.

```bash
mkdir -p C:/git
cd C:/git
git clone <your-repo-url> mongoose
cd mongoose
```

If you don't have a remote yet (it was a local-only repo on the laptop), bundle from the laptop:

```bash
# On laptop:
cd C:/git/mongoose
git bundle create mongoose.bundle --all

# Copy mongoose.bundle to desktop, then on desktop:
cd C:/git
git clone mongoose.bundle mongoose
cd mongoose
git remote remove origin   # optional, removes the bundle reference
```

### 2. Verify the recent commits made it across

```bash
git log --oneline -10
git branch -a
```

You should see two branches:
- `main` -- original V1 (complete, 124 tests passing, preserved as fallback)
- `v1-rearchitecture` -- V1 pivoted to self-supervised probe detection (2026-04-15)

**Check out the rearchitecture branch before proceeding:**

```bash
git checkout v1-rearchitecture
```

The code on this branch is partially in-progress -- only the design doc and implementation plan are committed. Task R1-R7 code changes still need to be executed (via subagents during the session). See `docs/plans/2026-04-15-t2d-unet-implementation-v2.md` for the task list.

### 3. (Optional but recommended) Copy memory from laptop

The project memory lives at `C:\Users\mcbee\.claude\projects\C--git-mongoose\memory\`. If your user profile syncs across machines (OneDrive, etc.), it's already there. If not, copy that directory manually.

This contains the design decisions, scope tradeoffs, and architectural reasoning we worked through. Without it, the new Claude session starts cold.

### 4. Set up Python environment

```bash
# Verify Python (3.12+)
python --version

# Install dependencies (CPU torch first -- we'll upgrade to CUDA after smoke test)
pip install torch numpy
pip install -e ".[dev]"
```

### 5. Run the test suite

```bash
pytest -q
```

Expected: 124 tests pass in ~30 seconds. If any fail, that's the first thing to debug -- something didn't transfer cleanly.

### 6. Confirm where the data lives

Note the absolute paths to:
- The TDB files
- The probes.bin files (from signal processing output)
- The probeassignment.assigns files (from remap output)
- The referenceMap.txt (one file, shared across all 30 runs)

Have these paths handy when you start the Claude session.

### 7. Confirm GPU availability (optional, can also do this in-session)

```bash
nvidia-smi
```

Note the GPU model and VRAM. You'll want at least 8 GB; more is better.

---

## Part 2 -- Starting the Claude Session

Open Claude Code in `C:/git/mongoose`. **Verify you're on branch `v1-rearchitecture`** (`git status` should show it). Paste this as your first message:

> I'm continuing work on Project Mongoose from a fresh session on my desktop. I'm on branch `v1-rearchitecture` which contains the V1 pivot to self-supervised probe detection (soft-DTW L_bp, dense L_velocity, warmstart phase). The original V1 is preserved on `main` as a fallback.
>
> Memory should be loaded. In particular, read `project_v1_rearchitecture.md` for the pivot context.
>
> Status: I've cloned the repo, installed dependencies, and the test suite on main passes (124 tests). This branch has the new design doc and implementation plan committed but the Task R1-R7 code changes are not yet implemented. I have all 30 E. coli runs downloaded locally. Data layout is `<paste your actual layout here>`. GPU available: `<paste your nvidia-smi output here>`.
>
> Tonight's goals, in order:
> 1. Execute Tasks R1-R7 from `docs/plans/2026-04-15-t2d-unet-implementation-v2.md` via subagents (level-1 estimator, soft-DTW, count/peakiness losses, ground truth revision, loss composition, dataset integration, smoke test). Use subagent-driven-development for this.
> 2. Once R7 passes on synthetic data, build `scripts/batch_preprocess.py` and `scripts/build_manifest.py` (these don't exist yet)
> 3. Sanity-check preprocessing on one real run (use STB03-064B-02L58270w05-202G16g if available)
> 4. Review summary stats with me before batch-processing all 30
> 5. Batch preprocess all 30 runs
> 6. Local smoke test on real preprocessed data
> 7. If smoke test passes, install CUDA PyTorch and re-run with `--amp` to verify GPU training works
>
> Step 1 is the biggest block. Please start by reading the implementation plan, summarize the R1-R7 task structure, and dispatch the first subagent. If there are ambiguities in the plan, ask me to resolve them first.

---

## Part 3 -- Expected Session Flow

### Phase 0: Execute V1 rearchitecture code changes (60-120 minutes)

Claude will run the subagent-driven-development workflow over tasks R1-R7 from the implementation plan. Each task is small (15-20 minutes per subagent dispatch, with spec/quality review between). Total: 1-2 hours.

At the end, `pytest -q` should report ~140+ passing tests (124 existing + new tests from R1-R5) and the synthetic smoke test should complete 3 epochs with warmstart transitioning mid-run.

### Phase 1: Sanity-check (15-30 minutes)

Claude will help you construct the sanity-check command for one run. Expected outcome:

- Exits with code 0
- Reports ~15K-25K cached molecules
- Prints summary stats (translocation time distribution, level-1 distribution, fwd/rev split, matched probe counts)

If exit code is non-zero, the code tells us where:
- 1 = preprocessing crashed
- 2 = cache files missing
- 3 = molecule counts out of expected range
- 4 = data integrity check failed
- 5 = CachedMoleculeDataset can't load the cache

Most likely failure mode: TDB header parsing if the file format is slightly different from spec. Claude will walk you through diagnosing.

### Phase 2: Review stats (5-10 minutes)

Look for these red flags before batch processing:
- Forward/reverse direction ratio is wildly skewed (>90% one direction)
- Median translocation time is < 30ms or > 500ms (likely an issue with TDB parsing)
- Median mean level-1 is < 0.1 mV or > 5 mV (suggests amplitude scaling problem)
- Matched probes per molecule median is < 5 (filtering may be wrong)

If anything looks weird, stop and diagnose before processing more runs.

### Phase 3: Build batch preprocessing + manifest (15-20 minutes)

Claude will dispatch a subagent to add two scripts:
- `scripts/batch_preprocess.py` -- iterates over all runs, calls preprocess_run for each, reports summary
- `scripts/build_manifest.py` -- scans cache directories, assigns each run to train/val/test based on die, writes JSON manifests

Both will get unit tests added.

### Phase 4: Batch preprocessing (60-180 minutes, mostly unattended)

Run the batch script. Each run takes 2-5 minutes depending on TDB file size. Total: ~1-3 hours for all 30 runs.

You can let this run in the background while you do other things. Output:
```
cache/
  STB03-064B-02L58270w05-202G16g/
    manifest.json
    waveforms.bin
    offsets.npy
    conditioning.npy
    molecules.pkl
  STB03-063A-02L58270w05-433H09e/
    ...
  ...
manifests/
  train.json
  val.json
  test.json
```

Total cache size: ~5-8 GB.

### Phase 5: Local smoke test (10-20 minutes)

```bash
python scripts/train.py --epochs 2 --batch-size 2 --no-amp
```

Expected:
- Loads the cached dataset successfully
- First epoch completes without errors
- Loss decreases from epoch 1 to epoch 2 (or at least doesn't explode to NaN)
- Checkpoint saves to `checkpoints/`

This is plumbing verification, not a scientific result. Two epochs on a subset is not enough to converge.

### Phase 6: GPU training (15-30 minutes)

Install CUDA PyTorch:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Re-run smoke test with mixed precision:

```bash
python scripts/train.py --epochs 2 --batch-size 8 --save-every 1
```

If your desktop GPU is bigger than the A2000 (8 GB), you can push batch size higher. RTX 3090/4090 (24 GB) can handle batch size 16-32 comfortably.

---

## Part 4 -- What "Done for Tonight" Looks Like

You're in good shape if at end of session:

- All 30 runs preprocessed into `cache/` directory
- Train/val/test manifests written to `manifests/`
- Smoke test ran end-to-end with real data, no NaN losses
- GPU training verified working (if you got that far)
- Any data weirdness flagged for follow-up

You're **not** trying to get a converged model tonight. That's the cloud training run. Tonight is about validating that everything works on real data.

---

## Part 5 -- If Things Go Wrong

**Tests fail on fresh setup:** Likely a path issue with the conftest fixtures or a missing dependency. Re-run `pip install -e ".[dev]"` and check Python version.

**Sanity-check exits with code 1 (preprocessing crashed):** Most likely the TDB header parsing trips on a field count discrepancy. Claude can help diagnose by reading the TDB file's actual byte layout against the spec.

**Sanity-check exits with code 3 (molecule counts off):** Either the run has unusually high rejection rates (look at the ratios -- total vs clean vs remapped) or the filtering thresholds need adjustment. Tell Claude the actual numbers.

**Smoke test produces NaN losses immediately:** Almost always a numerical issue with the level-1 normalization (zero or near-zero mean_lvl1 for some molecule). Claude will help add a guard.

**GPU OOM at batch size 8:** Drop to batch size 4, or wait for the cloud run with a bigger GPU.

**You hit something not covered here:** Tell Claude in the session, paste the error, and we'll work through it.

---

## Quick Reference Card

### Key paths
- Repo: `C:\git\mongoose`
- Memory: `C:\Users\mcbee\.claude\projects\C--git-mongoose\memory\`
- Data: `<your local data path>`
- Cache: `cache/` (auto-created during preprocessing)
- Manifests: `manifests/` (auto-created)
- Checkpoints: `checkpoints/` (auto-created during training)

### Key commands
```bash
# Setup
pytest -q                                              # verify tests pass

# Sanity check one run
python scripts/sanity_check_preprocess.py --tdb ... --probes-bin ... --assigns ... --reference-map ... --run-id ...

# Smoke test (after preprocessing)
python scripts/train.py --epochs 2 --batch-size 2 --no-amp

# GPU training
python scripts/train.py --epochs 2 --batch-size 8 --save-every 1

# Compare runs (after multiple training runs exist)
python scripts/compare_runs.py --results r1.json r2.json --labels baseline ablation
```

### Key reference docs in repo
- `docs/plans/2026-04-13-t2d-unet-design.md` -- architecture spec
- `docs/plans/2026-04-13-t2d-unet-implementation.md` -- 12-task implementation plan (already complete)
- `docs/proposals/2026-04-15-deck-outline-v2-rich.md` -- presentation outline
- `docs/ablation_protocol.md` -- how to run the lambda_vel ablation post-training
