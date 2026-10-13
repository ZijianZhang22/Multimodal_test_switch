#!/usr/bin/env bash
set -euo pipefail

# InternVL3.5-14B cross-family replication of the 500-problem
# single-visual logical-role experiment.
#
# Conditions on the same 500 matched 4-hop problems:
#   TTTT, VTTT, TVTT, TTVT, TTTV
#
# Total: 2,500 evaluations.
#
# Usage:
#   ./run_internvl_3t1v_500.sh

cd "$(dirname "$0")"

# Avoid the common Hugging Face transfer-package mismatch on fresh pods.
unset HF_HUB_ENABLE_HF_TRANSFER || true
export HF_HUB_ENABLE_HF_TRANSFER=0

export MODEL_FAMILY=internvl
export MODEL="${MODEL:-OpenGVLab/InternVL3_5-14B}"
export N_PROBLEMS="${N_PROBLEMS:-500}"
export SEED="${SEED:-20261014}"
export LABEL="${LABEL:-internvl35_14b_3t1v_500}"

exec ./run_3t1v_500.sh
