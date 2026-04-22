#!/usr/bin/env bash
# L_511 overnight extension — warm-start from the 3-epoch spike's best model,
# train another 7 epochs (fresh cosine schedule), answer whether the spike's
# 31.9% Blue-holdout plateau is training-time or architectural.
set -u
ROOT="C:/git/mongoose"
CACHE_ROOT="$ROOT/E. coli/cache"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
cd "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

CACHE_ARGS=(
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-202G16j"
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-433B23e"
    --cache-dir "$CACHE_ROOT/STB03-060B-02L58270w05-433H09f"
    --cache-dir "$CACHE_ROOT/STB03-060C-02L58270w05-433H09a"
)

PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs 7 --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --use-l511 \
    --lambda-511 1.0 --lambda-smooth 0.001 --lambda-length 0.5 \
    --warmstart-epochs 3 --warmstart-fade-epochs 0 \
    --min-blend 0.05 \
    --init-from l511_spike_checkpoints/best_model.pt \
    --checkpoint-dir l511_spike_ext_checkpoints --save-every 1
