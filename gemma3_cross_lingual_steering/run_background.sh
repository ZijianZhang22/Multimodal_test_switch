#!/usr/bin/env bash
# One command to launch detached in RunPod terminal.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [[ -f logs/run.pid ]]; then
  old="$(cat logs/run.pid)"
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "Experiment launcher is still running: PID $old"
    exit 1
  fi
fi
log="logs/run_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONUNBUFFERED=1 bash run.sh > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > logs/run.pid
printf '%s\n' "$log" > logs/latest_log_path.txt
printf 'Gemma3 job PID=%s\nLog=%s\n' "$pid" "$log"
echo "Inspect: tail -f $log"
