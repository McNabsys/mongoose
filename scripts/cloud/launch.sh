#!/usr/bin/env bash
# Launch an Option A training run on H100 spot. Retries zones until one
# accepts (H100 spot capacity is thin and comes/goes minute-by-minute).
#
# Usage:
#   bash scripts/cloud/launch.sh [epochs] [extra_flags]
#
#   epochs:       default 20
#   extra_flags:  extra flags for train.py, quoted (e.g., '"--max-molecules 1000"')
#
# Requires scripts/cloud/setup.sh to have been run already.
set -u

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
SUFFIX="${SUFFIX:-mongoose-training}"
BUCKET="gs://${PROJECT_ID}-${SUFFIX}"
EPOCHS="${1:-20}"
EXTRA_FLAGS="${2:-}"

INSTANCE_NAME="mongoose-option-a-$(date +%Y%m%d-%H%M)"
WORKTREE="C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9"
STARTUP_SCRIPT="$WORKTREE/scripts/cloud/vm_startup.sh"

# DL VM image family. As of 2026-04, 'common-cu124' is the current
# stable CUDA 12.4 image family with PyTorch preinstalled. Override via
# IMAGE_FAMILY env var if that's rolled forward.
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu124}"

# Zones to try, ordered by typical H100 availability. Extend as needed.
ZONES=(
    us-central1-a us-central1-b us-central1-c us-central1-f
    us-east5-a us-east5-b us-east5-c
    us-east4-a us-east4-b us-east4-c
    us-west4-a us-west4-b us-west4-c
    europe-west4-a europe-west4-b europe-west4-c
)

MAX_ROUNDS=20
SLEEP_BETWEEN_ROUNDS=60

# Reusable create command as a function — differs only by zone.
try_create() {
    local ZONE="$1"
    echo "[$(date +%H:%M:%S)] Trying zone $ZONE ..."
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type=a3-highgpu-1g \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --image-family="$IMAGE_FAMILY" \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=300GB \
        --boot-disk-type=pd-ssd \
        --no-address \
        --metadata="BUCKET=$BUCKET,EPOCHS=$EPOCHS,EXTRA_FLAGS=$EXTRA_FLAGS,enable-osconfig=TRUE" \
        --metadata-from-file="startup-script=$STARTUP_SCRIPT" \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        2>&1
}

# Main retry loop.
for round in $(seq 1 $MAX_ROUNDS); do
    for zone in "${ZONES[@]}"; do
        OUTPUT=$(try_create "$zone")
        if echo "$OUTPUT" | grep -q "Created"; then
            echo ""
            echo "=== VM created in $zone ==="
            echo "Instance: $INSTANCE_NAME"
            echo ""
            echo "Tail startup log via IAP (VM has no external IP):"
            echo "  gcloud compute ssh $INSTANCE_NAME --zone=$zone --tunnel-through-iap --command 'sudo tail -f /var/log/mongoose-startup.log'"
            echo ""
            echo "Monitor checkpoints (upload every 2 min to $BUCKET/run_artifacts/):"
            echo "  gcloud storage ls $BUCKET/run_artifacts/"
            echo ""
            echo "Kill VM when done:"
            echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$zone --quiet"
            exit 0
        fi
        # Only print non-capacity errors, which probably mean we need
        # to fix something rather than retry. Capacity errors are normal.
        echo "$OUTPUT" | grep -E "ERROR" | grep -v "ZONE_RESOURCE_POOL_EXHAUSTED" | grep -v "QUOTA_EXCEEDED" || true
    done
    echo "[$(date +%H:%M:%S)] Round $round/$MAX_ROUNDS — no zone had capacity. Sleeping ${SLEEP_BETWEEN_ROUNDS}s."
    sleep $SLEEP_BETWEEN_ROUNDS
done

echo "ERROR: exhausted $MAX_ROUNDS retry rounds, no H100 spot capacity anywhere." >&2
exit 1
