#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
LOG="logs/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "ERROR line $LINENO. See $LOG. Rerun the same command after fixing the error; completed stages are reused."' ERR
command -v nvidia-smi >/dev/null || { echo 'No NVIDIA GPU detected. Start a GPU RunPod.'; exit 1; }
nvidia-smi
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,14), "Use Python 3.10-3.13"'
VENV="${MRRE_VENV:-/workspace/mrre-venv}"
if [ ! -x "$VENV/bin/python" ]; then "$PYTHON" -m venv --system-site-packages "$VENV"; fi
P="$VENV/bin/python"
"$P" -m pip install -r requirements.txt
"$P" -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see CUDA. Use a RunPod PyTorch CUDA template (a CPU-only torch build will not work)."; print("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0))'
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TOKENIZERS_PARALLELISM=false
"$P" runpod_pipeline.py "$@"
