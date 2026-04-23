#!/usr/bin/env bash
# Loop launch.sh forever until it lands a VM. Stops automatically if any
# mongoose-* instance already exists (so we don't double-provision when
# the parallel H100/A100 loop also lands one).
#
# Usage:
#   bash scripts/cloud/launch_loop.sh [args-for-launch.sh]
#
# Pass through env vars (SPOT, MACHINE_TYPE, ZONES) and positional args
# (epochs, extra_flags) exactly as you would to launch.sh.
set -u

WORKTREE="C:/git/mongoose/.claude/worktrees/peaceful-rubin-bfb7a9"
SLEEP_BETWEEN_CYCLES=30

cycle=0
while true; do
    cycle=$((cycle + 1))
    # Stop if a mongoose-* instance already exists (parallel launcher won).
    if gcloud compute instances list --filter="name ~ ^mongoose-" --format="value(name)" 2>/dev/null | head -1 | grep -q .; then
        echo "[$(date +%H:%M:%S)] Detected existing mongoose-* instance. Stopping loop."
        exit 0
    fi
    echo "[$(date +%H:%M:%S)] === Cycle $cycle starting ==="
    bash "$WORKTREE/scripts/cloud/launch.sh" "$@"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Cycle $cycle SUCCESS, exiting loop."
        exit 0
    fi
    echo "[$(date +%H:%M:%S)] Cycle $cycle exhausted (rc=$rc). Sleeping ${SLEEP_BETWEEN_CYCLES}s before next cycle."
    sleep $SLEEP_BETWEEN_CYCLES
done
