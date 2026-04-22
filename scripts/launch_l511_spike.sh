#!/usr/bin/env bash
# L_511 Feasibility Spike (V3 phase 1):
#   3 epochs × 4 training caches, swap CombinedLoss for L511Loss.
#   From-scratch training. Answers: does the 511-bp integral identity
#   contain enough signal to train a velocity head at all?
#
# Decision gates on bp-interval eval (Blue holdout):
#   median_rel < 10% and p95_rel < 20%  --> L_511 viable, proceed to V3 design
#   median_rel 10-30%                   --> marginal, investigate before commit
#   median_rel > 30%                    --> L_511 insufficient, rethink
set -u
ROOT="C:/git/mongoose"
CACHE_ROOT="$ROOT/E. coli/cache"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
cd "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

# 4 training caches (non-holdout), one per color where available.
CACHE_ARGS=(
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-202G16j"
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-433B23e"
    --cache-dir "$CACHE_ROOT/STB03-060B-02L58270w05-433H09f"
    --cache-dir "$CACHE_ROOT/STB03-060C-02L58270w05-433H09a"
)

PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs 3 --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --use-l511 \
    --lambda-511 1.0 --lambda-smooth 0.001 --lambda-length 0.5 \
    --warmstart-epochs 3 --warmstart-fade-epochs 0 \
    --min-blend 0.05 \
    --checkpoint-dir l511_spike_checkpoints --save-every 1
