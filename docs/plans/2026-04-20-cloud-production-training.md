# Cloud Production Training Spec (GCP)

**Context.** Phase 6 local smoke (4 epochs × 27 caches, RTX A4500 20 GB VRAM) is running at ~12 hours per epoch. For a full V1 production 35-epoch run on all caches, local hardware is infeasible (~15 days). This document specs a cloud setup that can finish a 35-epoch run in ~1-3 days for ~$75-155.

## Workload profile (empirical)

Measured from Phase 6 run (all 27 training caches, batch 32, BF16 AMP, teacher-forced physics + CenterNet probe):

| Metric | Value |
|---|---|
| GPU VRAM (active, dedicated) | 20.0 GB (saturated) |
| GPU VRAM (spilled to shared system RAM via PCIe) | 37.5 GB (slow — degrades throughput) |
| **Peak working set (dedicated + referenced-shared)** | **~40-58 GB** |
| System RAM | 91 GB / 128 GB peak |
| GPU utilization | 100% |
| Per-epoch wall time | ~12 h on RTX A4500 (Ampere, 20 GB) |
| Per-epoch wall time projection on A100 40 GB | would OOM; don't use |
| Checkpoint size | 507 MB / epoch |
| Total cache size | ~14 GB across 30 cache dirs |

**Implication:** need ≥80 GB VRAM. This rules out the A100 40 GB (a2-highgpu-1g). Leaves A100 80 GB (a2-ultragpu-1g) or H100 80 GB (a3-highgpu-1g).

## Recommended spec: a3-highgpu-1g (H100 80 GB)

```yaml
# gcloud compute instance spec
machine_type: a3-highgpu-1g
  vcpus: 26
  ram_gb: 234
  gpus:
    - type: nvidia-h100-80gb
      count: 1
      memory_gb: 80
boot_disk:
  type: pd-ssd
  size_gb: 300          # OS + Python env + caches (14 GB) + checkpoints + headroom
image_family: common-cu128     # Deep Learning VM, PyTorch 2.x, CUDA 12.8
region: prefer us-central1, fall back to us-west4, europe-west4
provisioning:
  spot: true             # 70-80% cheaper than on-demand; survive eviction via
                         # checkpoint-resume wrapper (see below)
```

## Why H100 over A100 80 GB

| | A100 80 GB (a2-ultragpu-1g) | H100 80 GB (a3-highgpu-1g) |
|---|---|---|
| Price (spot, us-central1, April 2026 ballpark) | ~$2.20/hr | ~$3.00/hr |
| BF16 TFLOPS | 312 | 989 (~3×) |
| Memory bandwidth | 2.0 TB/s | 3.35 TB/s |
| Expected 35-epoch wall time (projected) | ~48-70h | ~25-35h (2-3× faster) |
| Total cost (spot pricing × projected hours) | ~$105-155 | ~$75-105 |
| VRAM | 80 GB | 80 GB |

H100 is both faster AND cheaper per completed-run. Only caveat is availability: H100 spot instances are heavily contested. If `gcloud compute instances create` returns a capacity error, try other regions (us-west4, europe-west4, asia-southeast1) or fall back to A100 80 GB.

## Budget summary

| Scenario | Hours | Cost (spot) |
|---|---|---|
| 4-epoch Phase 6 replay on cloud (sanity check) | ~4h | **~$12-15** |
| 35-epoch full production | ~25-35h | **~$75-105** |
| 35-epoch on A100 80 GB (fallback) | ~48-70h | **~$105-155** |
| Egress: download final checkpoint (500 MB) | — | **~$0.04** |
| Persistent disk: 300 GB PD-SSD for a week | — | **~$10** |

**Total all-in, end-to-end, H100 production:** **~$85-115.**

## Why not multi-GPU (a3-highgpu-8g, etc.)

- Current `src/mongoose/training/trainer.py` is single-device. No DistributedDataParallel wrapper.
- Adding DDP is 2-4 hours of work and needs validation. Single H100 at ~30h for 35 epochs is already fast enough; not worth the additional code risk for a first production run.
- If future scaling is needed (e.g., multi-experiment hyperparameter sweep), revisit multi-GPU then.

## Preemption strategy (spot instances)

Spot GPUs can be evicted at any time. Trainer already checkpoints per epoch (`--save-every 1`) and `_maybe_load_checkpoint()` in `trainer.py` auto-resumes from the latest checkpoint in `--checkpoint-dir`. We just need an outer shell loop:

```bash
#!/usr/bin/env bash
# scripts/cloud_auto_resume_train.sh — called inside the cloud VM
# Loops training until the requested epoch count is reached. Each
# invocation either runs to completion or gets preempted; the next
# invocation resumes from the last saved checkpoint.
set -u
MAX_ATTEMPTS=50
for i in $(seq 1 $MAX_ATTEMPTS); do
    bash scripts/launch_phase6.sh   # or a cloud-specific launcher
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Training completed successfully (attempt $i)."
        break
    fi
    echo "Attempt $i exited with code $exit_code (likely preemption or crash). Retrying in 60s."
    sleep 60
done
```

This can run directly on the spot VM — on preemption, GCP will schedule a new instance when capacity returns, and the auto-resume wrapper picks up where it left off.

## Data handoff strategy

### Upload caches to GCS (one-time)

```bash
# on local Windows machine
gsutil -m cp -r "E. coli/cache" gs://<your-bucket>/ecoli_cache/
# ~14 GB, takes ~10-20 min at 100 Mbps
```

### Download on cloud instance (once, at startup)

```bash
# on cloud VM
gsutil -m cp -r gs://<your-bucket>/ecoli_cache/ /data/ecoli_cache/
# VM downloads within GCP network — very fast, ~1-2 min
# Persist on local SSD for memmap I/O speed — do NOT run training
# against GCS FUSE, it will be 10-100x slower.
```

### Upload checkpoints back (optional)

```bash
# on cloud VM, after training
gsutil -m cp -r phase6_checkpoints/ gs://<your-bucket>/phase6_prod_checkpoints/
```

## Pre-flight code changes (do before first cloud run)

1. **Dockerfile** pinning Python 3.12 + torch 2.11 + CUDA 12.8 + project dependencies.
   - Build with `docker build -t mongoose-training:v1 .`
   - Push to Google Artifact Registry (or use Deep Learning VM image directly and `pip install -e .` the repo)

2. **Requirements lock** — `pip freeze > requirements-lock.txt` so the cloud
   has exact versions. Especially important for PyTorch / CUDA compatibility.

3. **Memory cleanup** (before running anything significant):
   - Add `torch.cuda.empty_cache()` at train → val boundary and at the end
     of each epoch in `trainer.py`.
   - Verify val pass uses `torch.no_grad()` and doesn't retain the graph.
   - This might bring peak VRAM from ~58 GB down to ~25-30 GB, at which
     point A100 40 GB becomes viable (cheaper).

4. **Cloud launch script** `scripts/launch_cloud.sh` (new):
   - Resolves environment variables (data path, checkpoint dir, W&B key)
   - Kicks off `scripts/launch_phase6.sh` (or a cloud variant with
     `--epochs 35` and appropriate holdouts)
   - Optionally: streams logs to GCS or Cloud Logging

5. **Verify cache portability Windows → Linux.** One quick test: copy
   `E. coli/cache/STB03-060A-02L58270w05-433B23e/` to a Linux box
   (or WSL), `pip install -e .`, run:
   ```python
   from mongoose.data.cached_dataset import CachedMoleculeDataset
   ds = CachedMoleculeDataset([Path("STB03-060A-02L58270w05-433B23e")], augment=False)
   item = ds[0]
   assert item["waveform"].shape[-1] > 0
   ```
   If that works, the cache is portable (numpy handles endianness; int16
   waveforms + pickled dicts + JSON + .npy are all platform-neutral).

## Alternative: Vertex AI Custom Training

If you'd rather skip VM lifecycle management entirely, **Vertex AI Custom Training** takes a Docker image + training command and runs it as a managed job. Slightly more expensive (~15% overhead) but no instance provisioning / SSH / preemption-wrapper complexity. Good for "submit + walk away" workflows.

```bash
# Rough Vertex AI command
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=mongoose-prod \
    --config=vertex_job_config.yaml
```

## Action list (when ready to execute)

1. [ ] Memory cleanup PR on `claude/peaceful-rubin-bfb7a9` (1-2h):
       `torch.cuda.empty_cache()` between train/val, verify `torch.no_grad()`.
2. [ ] Dockerfile + requirements lock + push to Artifact Registry.
3. [ ] `gsutil cp` caches to GCS bucket.
4. [ ] `gcloud compute instances create` a3-highgpu-1g spot in whichever
       region has capacity. Fall back to A100 80 GB if H100 unavailable.
5. [ ] SSH to instance, `git clone` branch `claude/peaceful-rubin-bfb7a9`,
       pull caches from GCS, kick off `scripts/cloud_auto_resume_train.sh`
       with `--epochs 35` and 3 per-color holdouts.
6. [ ] Monitor via TensorBoard (SSH tunnel: `ssh -L 6006:localhost:6006
       <instance>`).
7. [ ] When done: eval on 3 holdouts, pull artifacts to local.
8. [ ] Shut down instance (don't leak spot-GPU billing).
