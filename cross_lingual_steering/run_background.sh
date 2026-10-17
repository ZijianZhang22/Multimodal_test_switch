#!/usr/bin/env bash
# Start the steering experiment in the background on RunPod.
# Run from anywhere: bash cross_lingual_steering/run_background.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

pidfile="logs/run.pid"
if [[ -f "$pidfile" ]]; then
  oldpid="$(cat "$pidfile")"
  if [[ "$oldpid" =~ ^[0-9]+$ ]] && kill -0 "$oldpid" 2>/dev/null; then
    echo "An experiment launcher is already running (PID $oldpid)."
    echo "Check: tail -f $(cat logs/latest_log_path.txt 2>/dev/null || echo logs/run.log)"
    exit 1
  fi
fi

log="logs/run_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONUNBUFFERED=1 bash run.sh > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
printf '%s\n' "$log" > logs/latest_log_path.txt
printf 'Started background run: PID %s\nLog: %s\n' "$pid" "$log"
printf 'Monitor: tail -f %s\n' "$log"
printf 'Status: ps -fp %s\n' "$pid"
