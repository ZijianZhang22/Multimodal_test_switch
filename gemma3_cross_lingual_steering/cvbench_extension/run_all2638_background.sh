#!/usr/bin/env bash
# Full official CV-Bench coverage: local validation (60/task) then all remaining test.
# Three prompt languages (EN/ZH/SW) x four CV-Bench tasks x four concept vectors.
# Reuses unmodified upstream Gemma3 Mean Shift; no hyperparameter tuning on test.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# Do not collide with an active smoke / pilot / full GPU run.
for f in logs/cvbench.pid logs/all2638.pid ../logs/run.pid ../logs/full.pid; do
  [[ -f "$f" ]] || continue
  p="$(cat "$f")"
  if [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; then
    state="$(ps -o stat= -p "$p" 2>/dev/null || true)"
    if [[ "$state" != Z* ]]; then
      echo "Existing experiment still active: PID $p ($f). Finish it before starting another." >&2
      exit 1
    fi
  fi
done

# New directory to preserve all existing smoke/partial predictions and vectors.
export DATA="${DATA:-outputs/gemma3_cvbench_all2638}"
export N_VAL="${N_VAL:-60}"
export N_TEST=0
export LIMIT=0
export LAYERS="${LAYERS:-8,16}"
export ALPHAS="${ALPHAS:-0.1}"
export ALL_FEATURES=1
export NORMALIZED="${NORMALIZED:-0}"
export MASK="${MASK:-both}"
export SEED="${SEED:-42}"
export INSTALL="${INSTALL:-1}"
export FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

# Both stages deliberately use the SAME predetermined settings.
log="logs/all2638_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c '
  set -euo pipefail
  cd "$1"
  echo "=== FULL CV-BENCH: PREPARE + VAL ==="
  echo "Dataset=$DATA | per-task val=$N_VAL | test=all remaining"
  echo "Layers=$LAYERS | Alphas=$ALPHAS | Features=all four | Languages=en,zh,sw"
  SPLIT=val bash ./run.sh
  echo "=== FULL CV-BENCH: HELD-OUT TEST (same fixed settings) ==="
  SPLIT=test INSTALL=0 FORCE_EXTRACT=0 bash ./run.sh
  echo "=== FULL ORIGINAL CV-BENCH EVALUATION COMPLETE ==="
' _ "$PWD" > "$log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > logs/all2638.pid
printf '%s\n' "$log" > logs/all2638_latest_log_path.txt
echo "Started full CV-Bench job PID $pid"
echo "Log: $log"
echo "Monitor: cd gemma3_cross_lingual_steering/cvbench_extension && tail -f \"$log\""
echo "Results: $DATA/analysis_val/ and $DATA/analysis_test/"
