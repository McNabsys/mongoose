#!/usr/bin/env bash
# Clean up cloud resources when done. Does NOT delete the GCS bucket
# (checkpoints and logs are there) — that's a manual decision.
#
# Usage:
#   bash scripts/cloud/teardown.sh [instance_name]
#
# If instance_name omitted, lists any mongoose-* instances and prompts.
set -u

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
INSTANCE_NAME="${1:-}"

if [ -z "$INSTANCE_NAME" ]; then
    echo "Active mongoose-* instances:"
    gcloud compute instances list --filter="name ~ ^mongoose-" \
        --format="table(name,zone,status,machineType.basename())"
    echo ""
    read -p "Instance name to delete (empty = skip): " INSTANCE_NAME
    if [ -z "$INSTANCE_NAME" ]; then
        echo "Skipped."
        exit 0
    fi
fi

ZONE=$(gcloud compute instances list --filter="name=$INSTANCE_NAME" \
    --format="value(zone)" | head -1)

if [ -z "$ZONE" ]; then
    echo "Instance not found: $INSTANCE_NAME"
    exit 1
fi

ZONE=$(basename "$ZONE")
echo "Deleting $INSTANCE_NAME (zone: $ZONE) ..."
gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
echo "Done."

echo ""
echo "Artifacts remain at gs://${PROJECT_ID}-mongoose-training/run_artifacts/"
echo "To delete bucket entirely: gcloud storage rm -r gs://${PROJECT_ID}-mongoose-training/"
