#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OUT="${OUT:-idis_math50_new}"
export COUNT="${COUNT:-50}"
export SEED="${SEED:-1043}"
export VARIANT="${VARIANT:-irrelevant}"
export N_DISTRACTORS="${N_DISTRACTORS:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12288}"
export EXCLUDE_IDS_FILE="${EXCLUDE_IDS_FILE:-idis_math10/pilot10_exclusions.json}"
echo "50 new problems (excluding all versions of the first 10)."
bash idis_math10/run_runpod.sh
