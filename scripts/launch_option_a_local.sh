#!/usr/bin/env bash
# Option A local smoke: 3 epochs × 4 caches, T2D-hybrid velocity head
# warm-started from the L_511 extension's best model. Same caches as the
# spike/extension so numbers are apples-to-apples with V3-spike and V3-ext.
#
# Decision gate: if median rel err on Blue holdout beats T2D's 16.2%,
# Option A is the new baseline and we promote to a cloud production run.
set -u
ROOT="C:/git/mongoose"
CACHE_ROOT="$ROOT/E. coli/cache"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
cd "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

# Verify enrichment has run on the training caches.
for run in STB03-060A-02L58270w05-202G16j STB03-060A-02L58270w05-433B23e \
           STB03-060B-02L58270w05-433H09f STB03-060C-02L58270w05-433H09a; do
    if [ ! -f "$CACHE_ROOT/$run/t2d_params.npy" ]; then
        echo "ERROR: $run missing t2d_params.npy" >&2
        echo "Run scripts/precompute_all_caches_t2d.sh first." >&2
        exit 1
    fi
done

CACHE_ARGS=(
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-202G16j"
    --cache-dir "$CACHE_ROOT/STB03-060A-02L58270w05-433B23e"
    --cache-dir "$CACHE_ROOT/STB03-060B-02L58270w05-433H09f"
    --cache-dir "$CACHE_ROOT/STB03-060C-02L58270w05-433H09a"
)

# Warm-start from the L_511 extension's best model (if it exists), else
# from the L_511 spike's best, else fresh. Velocity head logits are
# reinterpreted under the new hybrid rules so old weights just serve
# as a reasonable initialization — the tanh-bounded residual starts
# at a small magnitude regardless.
INIT_FROM=""
if [ -f "l511_spike_ext_checkpoints/best_model.pt" ]; then
    INIT_FROM="--init-from l511_spike_ext_checkpoints/best_model.pt"
elif [ -f "l511_spike_checkpoints/best_model.pt" ]; then
    INIT_FROM="--init-from l511_spike_checkpoints/best_model.pt"
fi

PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" -u scripts/train.py \
    "${CACHE_ARGS[@]}" \
    --epochs 3 --batch-size 32 --lr 3e-4 --min-lr 1.5e-5 \
    --use-l511 --use-t2d-hybrid \
    --lambda-511 1.0 --lambda-smooth 0.001 --lambda-length 0.5 \
    --warmstart-epochs 3 --warmstart-fade-epochs 0 \
    --min-blend 0.05 \
    $INIT_FROM \
    --checkpoint-dir option_a_local_checkpoints --save-every 1
