#!/usr/bin/env bash
# One-time GCP setup for Option A cloud training.
#
# Idempotent — safe to re-run. Creates:
#   - GCS bucket  for caches, source tarball, and run artifacts
#   - Cloud NAT   so internal-IP-only VMs can pip install etc.
#   - Source tarball uploaded to bucket
#   - All 30 enriched caches uploaded (14 GB, ~10-20 min first time)
#   - Warm-start checkpoint uploaded (if present locally)
#
# Run this ONCE per project. Re-running is cheap (skips existing files).
#
# Usage:
#   bash scripts/cloud/setup.sh [bucket_suffix]
#
# bucket_suffix defaults to 'mongoose-training'. Bucket name becomes
# gs://<project-id>-<suffix>.
set -eu

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: no active gcloud project. Run 'gcloud config set project <id>'." >&2
    exit 1
fi

SUFFIX="${1:-mongoose-training}"
BUCKET="gs://${PROJECT_ID}-${SUFFIX}"
REGION="us-central1"
LOCAL_ROOT="C:/git/mongoose"
CACHE_ROOT="$LOCAL_ROOT/E. coli/cache"
WORKTREE="$LOCAL_ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

echo "=== Using bucket: $BUCKET (region $REGION, project $PROJECT_ID)"

# ---- 1. GCS bucket ----
if gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
    echo "Bucket already exists."
else
    echo "Creating bucket $BUCKET ..."
    gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access
fi

# ---- 2. Cloud NAT for outbound internet from internal-IP VMs ----
if gcloud compute routers describe nat-router --region="$REGION" >/dev/null 2>&1; then
    echo "Cloud Router 'nat-router' already exists."
else
    echo "Creating Cloud Router ..."
    gcloud compute routers create nat-router \
        --network=default --region="$REGION"
fi

if gcloud compute routers nats describe nat-config --router=nat-router --region="$REGION" >/dev/null 2>&1; then
    echo "Cloud NAT 'nat-config' already exists."
else
    echo "Creating Cloud NAT ..."
    gcloud compute routers nats create nat-config \
        --router=nat-router --region="$REGION" \
        --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips
fi

# ---- 3. Upload source tarball ----
echo "Packaging source tarball ..."
cd "$WORKTREE"
TARBALL="/tmp/mongoose-src.tar.gz"
# Package only the things the VM needs to run training.
tar -czf "$TARBALL" \
    src \
    scripts \
    pyproject.toml \
    2>/dev/null
echo "  tarball size: $(du -h "$TARBALL" | awk '{print $1}')"
gcloud storage cp "$TARBALL" "$BUCKET/mongoose-src.tar.gz"

# ---- 4. Upload enriched caches ----
echo "Uploading caches (14 GB, ~10-20 min first time) ..."
cd "$LOCAL_ROOT"
# Only upload caches that have been enriched with t2d_params.npy.
n_enriched=0
n_skipped=0
for cache_dir in "$CACHE_ROOT"/*/; do
    run_id=$(basename "$cache_dir")
    if [ -f "$cache_dir/t2d_params.npy" ]; then
        # Use gcloud storage cp with -r and --no-clobber (won't overwrite existing).
        echo "  $run_id"
        gcloud storage cp -r --no-clobber "$cache_dir" "$BUCKET/caches/" 2>&1 | grep -E "ERROR|Copying" | tail -3 || true
        n_enriched=$((n_enriched + 1))
    else
        echo "  SKIP $run_id (no t2d_params.npy; run precompute_all_caches_t2d.sh first)"
        n_skipped=$((n_skipped + 1))
    fi
done
echo "  uploaded: $n_enriched caches (skipped: $n_skipped)"

# ---- 5. Upload warm-start checkpoint (if present) ----
INIT_CKPT=""
if [ -f "$WORKTREE/l511_spike_ext_checkpoints/best_model.pt" ]; then
    INIT_CKPT="$WORKTREE/l511_spike_ext_checkpoints/best_model.pt"
elif [ -f "$WORKTREE/option_a_local_checkpoints/best_model.pt" ]; then
    INIT_CKPT="$WORKTREE/option_a_local_checkpoints/best_model.pt"
elif [ -f "$WORKTREE/l511_spike_checkpoints/best_model.pt" ]; then
    INIT_CKPT="$WORKTREE/l511_spike_checkpoints/best_model.pt"
fi
if [ -n "$INIT_CKPT" ]; then
    echo "Uploading warm-start checkpoint: $INIT_CKPT"
    gcloud storage cp "$INIT_CKPT" "$BUCKET/init_checkpoint.pt"
else
    echo "No warm-start checkpoint found; cloud run will start from scratch."
fi

echo ""
echo "=== Setup complete ==="
echo "Bucket:  $BUCKET"
echo "NAT:     nat-config on nat-router in $REGION"
echo "Source:  $BUCKET/mongoose-src.tar.gz"
echo "Caches:  $BUCKET/caches/"
echo "Init:    $BUCKET/init_checkpoint.pt (if present)"
echo ""
echo "Next step: bash scripts/cloud/launch.sh"
