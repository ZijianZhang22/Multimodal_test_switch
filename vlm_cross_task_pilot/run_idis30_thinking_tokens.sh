#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python idis30_thinking_tokens.py \
  --predictions "${PREDICTIONS:-idis30/results/predictions.jsonl}" \
  --out-dir "${OUT_DIR:-idis30/results/thinking_token_analysis}" \
  --bootstrap "${BOOTSTRAP:-3000}"
