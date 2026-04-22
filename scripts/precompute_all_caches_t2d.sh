#!/usr/bin/env bash
# Enrich every cache directory with t2d_params.npy for Option A training.
#
# For each cache dir, locates the matching _transForm.txt in the
# sibling Remapped/AllCh directory of the original run, then invokes
# scripts/precompute_t2d_params.py. Missing transform files are
# reported but not fatal — those caches just won't be usable for
# hybrid training.
set -u
ROOT="C:/git/mongoose"
CACHE_ROOT="$ROOT/E. coli/cache"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
cd "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9"

n_done=0
n_skip=0
for cache_dir in "$CACHE_ROOT"/*/; do
    run_id=$(basename "$cache_dir")
    # Run_id has the form STB03-xxxx-02L58270w05-NNNNNNN. The transform
    # file is named <run_id>_transForm.txt and lives in the color's
    # date-stamped Remapped/AllCh directory. Find it by glob.
    transform=$(find "$ROOT/E. coli" -name "${run_id}_transForm.txt" 2>/dev/null | head -1)
    if [ -z "$transform" ]; then
        echo "SKIP: $run_id (no _transForm.txt found)" >&2
        n_skip=$((n_skip + 1))
        continue
    fi
    echo "=== $run_id ===" >&2
    PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" -u scripts/precompute_t2d_params.py \
        --cache-dir "$cache_dir" \
        --transform-file "$transform" 2>&1 | tail -4
    n_done=$((n_done + 1))
done

echo ""
echo "Enriched: $n_done caches"
if [ $n_skip -gt 0 ]; then
    echo "Skipped:  $n_skip caches (no transform file)"
fi
