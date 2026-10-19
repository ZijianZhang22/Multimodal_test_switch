#!/usr/bin/env bash
# Start run_parity.sh detached (survives SSH disconnects). Same environment variables.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [[ -f logs/parity.pid ]] && kill -0 "$(cat logs/parity.pid)" 2>/dev/null; then
  echo "Already running: PID $(cat logs/parity.pid)"; exit 1
fi
log="logs/parity_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONUNBUFFERED=1 bash run_parity.sh > "$log" 2>&1 < /dev/null &
echo $! > logs/parity.pid
echo "$log" > logs/latest_log_path.txt
echo "Started PID $(cat logs/parity.pid); log: $PWD/$log"
echo "Monitor: tail -f \"$PWD/$log\""
