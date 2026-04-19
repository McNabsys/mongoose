#!/usr/bin/env bash
# Regenerate all 30 run caches with the corrected probe-sample mapping.
# Writes to E. coli/cache/<run_id>/ and tees per-run logs into cache_regen_logs/.
set -u
ROOT="C:/git/mongoose"
DATA="$ROOT/E. coli"
OUT="$DATA/cache"
LOGS="$ROOT/cache_regen_logs"
PYBIN="$ROOT/.venv/Scripts/python.exe"
PYTHONPATH="$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/src"
mkdir -p "$LOGS"

for color in Black Blue Red; do
  for run_dir in "$DATA/$color"/*/; do
    run_id=$(basename "$run_dir")
    date_dir=$(ls -d "$run_dir"20*/ 2>/dev/null | head -1)
    if [ -z "$date_dir" ]; then
      echo "SKIP $run_id: no date dir" | tee -a "$LOGS/_index.log"
      continue
    fi
    date_dir="${date_dir%/}"
    out_sub="$OUT/$run_id"
    if [ -d "$out_sub" ] && [ -f "$out_sub/manifest.json" ]; then
      echo "SKIP $run_id: already exists at $out_sub" | tee -a "$LOGS/_index.log"
      continue
    fi
    echo "=== START $color/$run_id at $(date +%H:%M:%S) ===" | tee -a "$LOGS/_index.log"
    PYTHONPATH="$PYTHONPATH" PYTHONIOENCODING=utf-8 "$PYBIN" "$ROOT/.claude/worktrees/peaceful-rubin-bfb7a9/scripts/preprocess.py" \
      --run-id "$run_id" --run-dir "$date_dir" --output "$OUT" \
      2>&1 | tee "$LOGS/$run_id.log"
    status=${PIPESTATUS[0]}
    echo "=== END   $color/$run_id at $(date +%H:%M:%S) exit=$status ===" | tee -a "$LOGS/_index.log"
  done
done
echo "ALL RUNS DONE at $(date +%H:%M:%S)" | tee -a "$LOGS/_index.log"
