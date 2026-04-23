#!/usr/bin/env bash
# VM startup script — runs automatically when the spot VM boots.
#
# Pulls source + caches from GCS, installs project deps, launches
# training, uploads artifacts to GCS as it goes. Designed for a fresh
# Deep Learning VM (has torch/CUDA preinstalled).
#
# Metadata contract (passed via gcloud --metadata):
#   BUCKET        : GCS bucket base (gs://...)
#   EPOCHS        : training epochs (e.g., 20)
#   EXTRA_FLAGS   : additional train.py flags (space-separated)
#
# The training command uses --use-l511 --use-t2d-hybrid by default.
set -u
exec > /var/log/mongoose-startup.log 2>&1
set -x

# ---- 1. Pull metadata ----
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H="Metadata-Flavor: Google"
BUCKET=$(curl -sf -H "$H" "$META/BUCKET" || echo "")
EPOCHS=$(curl -sf -H "$H" "$META/EPOCHS" || echo "20")
EXTRA_FLAGS=$(curl -sf -H "$H" "$META/EXTRA_FLAGS" || echo "")

if [ -z "$BUCKET" ]; then
    echo "FATAL: BUCKET metadata not set" >&2
    exit 1
fi
echo "BUCKET=$BUCKET  EPOCHS=$EPOCHS  EXTRA_FLAGS=$EXTRA_FLAGS"

# ---- 2. Wait for GPU + CUDA to be ready ----
# DL VM images sometimes need a moment after boot for the driver to settle.
for i in 1 2 3 4 5; do
    if nvidia-smi >/dev/null 2>&1; then
        break
    fi
    echo "Waiting for GPU ($i/5) ..."
    sleep 10
done
nvidia-smi

# ---- 3. Workspace ----
mkdir -p /data
cd /data

# ---- 4. Pull source tarball + unpack ----
gcloud storage cp "$BUCKET/mongoose-src.tar.gz" /data/mongoose-src.tar.gz
mkdir -p /data/mongoose
tar -xzf /data/mongoose-src.tar.gz -C /data/mongoose

# ---- 5. Pull caches ----
gcloud storage cp -r "$BUCKET/caches" /data/
# Reorganize: gcloud storage cp -r creates /data/caches/STB03-.../...
# Training expects a list of per-run cache dirs. We'll enumerate them in the launch.

# ---- 6. Pull init checkpoint (if present) ----
INIT_ARG=""
if gcloud storage ls "$BUCKET/init_checkpoint.pt" >/dev/null 2>&1; then
    gcloud storage cp "$BUCKET/init_checkpoint.pt" /data/init.pt
    INIT_ARG="--init-from /data/init.pt"
fi

# ---- 7. Install project ----
cd /data/mongoose
# DL VM has torch etc.; just install our package in editable mode.
pip install -e . 2>&1 | tail -5

# ---- 8. Build cache-dir args. 3 holdouts are excluded per the project
#        convention (1 per color) — same as launch_phase6.sh.
CACHE_ARGS=()
for d in /data/caches/*/; do
    run=$(basename "$d")
    case "$run" in
        STB03-063B-02L58270w05-433B23b|STB03-065H-02L58270w05-433H09j|STB03-064D-02L58270w05-433H09d)
            echo "HOLDOUT (excluding from training): $run" >&2
            ;;
        *)
            if [ -f "$d/t2d_params.npy" ]; then
                CACHE_ARGS+=( --cache-dir "$d" )
            else
                echo "SKIP $run (no t2d_params.npy)" >&2
            fi
            ;;
    esac
done
echo "training caches: $(( ${#CACHE_ARGS[@]} / 2 ))"

# ---- 9. Launch training ----
# Stream logs to stdout (captured above) AND to GCS every time a
# checkpoint is saved. We accomplish the latter with a background
# watcher process that gsutil-cp's checkpoint_epoch_*.pt every 60s.
mkdir -p /data/checkpoints
(
    while sleep 120; do
        gcloud storage cp -r --no-clobber /data/checkpoints/*.pt "$BUCKET/run_artifacts/" 2>/dev/null || true
    done
) &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT

PYTHONPATH=/data/mongoose/src PYTHONIOENCODING=utf-8 python -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs "$EPOCHS" --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --use-l511 --use-t2d-hybrid \
    --lambda-511 1.0 --lambda-smooth 0.001 --lambda-length 0.5 \
    --warmstart-epochs 5 --warmstart-fade-epochs 2 \
    --min-blend 0.05 \
    $INIT_ARG $EXTRA_FLAGS \
    --checkpoint-dir /data/checkpoints --save-every 1 \
    2>&1 | tee /var/log/mongoose-train.log

# ---- 10. Final upload of all checkpoints + logs ----
gcloud storage cp -r /data/checkpoints "$BUCKET/run_artifacts/" || true
gcloud storage cp /var/log/mongoose-train.log "$BUCKET/run_artifacts/train.log" || true
gcloud storage cp /var/log/mongoose-startup.log "$BUCKET/run_artifacts/startup.log" || true

echo "=== Training complete ==="

# ---- 11. Self-delete so we don't leak billing if user forgets ----
# The Compute Engine default service account has Editor role which
# includes compute.instances.delete on its own VM. If this fails (e.g.
# scope or permissions issue), the VM stays up and the user can manually
# delete via the console or scripts/cloud/teardown.sh.
INSTANCE_NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
echo "Self-deleting VM $INSTANCE_NAME in $ZONE in 60 seconds (gives time to inspect logs) ..."
gcloud storage cp /var/log/mongoose-startup.log "$BUCKET/run_artifacts/startup_pre_delete.log" || true
sleep 60
gcloud --quiet compute instances delete "$INSTANCE_NAME" --zone="$ZONE" 2>&1 || \
    echo "WARNING: self-delete failed. Manually delete with: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
