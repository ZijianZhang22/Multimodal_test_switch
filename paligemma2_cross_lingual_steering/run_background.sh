#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for f in logs/latest.pid ../gemma3_cross_lingual_steering/cvbench_extension/logs/cvbench.pid; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    st="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ "$st" == Z* ]] || { echo "Experiment PID $pid already running ($f)"; exit 1; }
  fi
done
log="logs/paligemma2_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONUNBUFFERED=1 bash run.sh > "$log" 2>&1 < /dev/null &
echo $! > logs/latest.pid
echo "$log" > logs/latest_log_path.txt
echo "Started PID $(cat logs/latest.pid)"
echo "Log $log"
echo 'Monitor: tail -f "$(cat logs/latest_log_path.txt)"'
