#!/usr/bin/env bash
# Option A long local run with TWO architectural upgrades:
#   1. Probe-aware velocity head — vel head sees the probe heatmap as an
#      extra input channel (lets it learn probe-state-conditioned corrections)
#   2. Mixed supervision — for high-confidence remapped molecules, add a
#      direct L1 alignment loss on cum_bp at probe centers vs reference_bp
#
# 10 caches × 5 epochs from-scratch. ~25h on local GPU.
# Hedge against cloud capacity being unavailable today.
set -u
ROOT="C:/git/mongoose"
CACHE_ROOT="$ROOT/E. coli/cache"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
cd "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

# 10 caches (first 10 non-holdout alphabetically). Mix of colors.
CACHE_ARGS=(
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-202G16j"
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-433B23e"
    --cache-dir "$CACHE_ROOT/STB03-060B-02L58270w05-433H09f"
    --cache-dir "$CACHE_ROOT/STB03-060C-02L58270w05-433H09a"
    --cache-dir "$CACHE_ROOT/STB03-062A-02L58270w05-433H09i"
    --cache-dir "$CACHE_ROOT/STB03-062B-02L58270w05-433B23h"
    --cache-dir "$CACHE_ROOT/STB03-062B-02L58270w05-433H09c"
    --cache-dir "$CACHE_ROOT/STB03-062D-02L58270w05-433H09g"
    --cache-dir "$CACHE_ROOT/STB03-063A-02L58270w05-433B23g"
    --cache-dir "$CACHE_ROOT/STB03-063A-02L58270w05-433H09e"
)

PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs 5 --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --use-l511 --use-t2d-hybrid \
    --probe-aware-velocity \
    --lambda-511 1.0 --lambda-smooth 0.001 --lambda-length 0.5 \
    --lambda-align 1.0 --align-min-confidence 0.7 \
    --warmstart-epochs 5 --warmstart-fade-epochs 2 \
    --min-blend 0.05 \
    --checkpoint-dir option_a_long_checkpoints --save-every 1
