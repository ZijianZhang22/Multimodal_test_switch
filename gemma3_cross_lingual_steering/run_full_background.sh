#!/usr/bin/env bash
# Full *local synthetic dataset* validation + test (not full upstream CVBench).
# Starts a single detached job. The test set is evaluated only after validation.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

for pidfile in logs/run.pid logs/full.pid; do
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Another steering launcher is still running (PID $old_pid; file $pidfile)"
      echo "Do not launch two GPU experiments simultaneously."
      exit 1
    fi
  fi
done

log="logs/full_$(date +%Y%m%d_%H%M%S).log"

# Inherited env such as LAYERS=8,16 ALPHAS=5,15 MASK=both is honored
# by both stages. Do not recreate the corpus or rerun extraction unless
# REBUILD_DATA=1 or FORCE_EXTRACT=1 was explicitly supplied.
nohup bash -c '
set -euo pipefail
cd "$1"
echo "=== FULL CURRENT DATASET: VAL (LIMIT=0) ==="
LIMIT=0 SPLIT=val bash run.sh
echo "=== FULL CURRENT DATASET: TEST (LIMIT=0) ==="
LIMIT=0 SPLIT=test INSTALL=0 FORCE_EXTRACT=0 REBUILD_DATA=0 bash run.sh
echo "=== FULL VAL + TEST COMPLETE ==="
' _ "$PWD" > "$log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > logs/full.pid
printf '%s\n' "$log" > logs/full_latest_log_path.txt
echo "Background job PID: $pid"
echo "Log: $log"
echo "Watch: cd gemma3_cross_lingual_steering && tail -f \"$log\""
echo "Outputs: outputs/gemma3/analysis_val/ and analysis_test/"
