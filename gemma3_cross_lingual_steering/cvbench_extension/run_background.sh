#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for f in logs/cvbench.pid ../logs/run.pid ../logs/full.pid; do
  if [[ -f "$f" ]]; then
    p="$(cat "$f")"
    if [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; then
      echo "An experiment is running (PID $p, file $f); don't launch another GPU job" >&2
      exit 1
    fi
  fi
done
log="logs/run_$(date +%Y%m%d_%H%M%S).log"
nohup bash run.sh >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >logs/cvbench.pid
printf '%s\n' "$log" >logs/latest_log_path.txt
echo "Started PID $pid | log $log"
echo "Watch: tail -f $log"
