#!/usr/bin/env bash
# Phase 6: 4 epochs × 27 training caches (3 holdouts: 1 per color).
# ETA ~37.6h. Launch with this script to guarantee proper path quoting
# across the "E. coli" space in cache paths.
set -u
ROOT="C:/git/mongoose"
CACHE_ROOT="$ROOT/E. coli/cache"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
cd "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

# Build the --cache-dir arg list as a bash ARRAY to preserve quoting.
CACHE_ARGS=()
for d in "$CACHE_ROOT"/*/; do
    run=$(basename "$d")
    case "$run" in
        STB03-063B-02L58270w05-433B23b|STB03-065H-02L58270w05-433H09j|STB03-064D-02L58270w05-433H09d)
            echo "HOLDOUT: $run" >&2
            ;;
        *)
            CACHE_ARGS+=( --cache-dir "$CACHE_ROOT/$run" )
            ;;
    esac
done
echo "training caches: $(( ${#CACHE_ARGS[@]} / 2 ))" >&2

PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs 4 --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --warmstart-epochs 4 --warmstart-fade-epochs 0 \
    --min-blend 1.0 \
    --scale-bp 300000 --scale-vel 5000 --scale-count 1e9 --scale-probe 1.0 \
    --checkpoint-dir phase6_checkpoints --save-every 1
