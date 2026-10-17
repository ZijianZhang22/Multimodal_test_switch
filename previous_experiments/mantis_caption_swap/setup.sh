#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U "datasets>=3" "transformers>=4.57,<6" accelerate pillow safetensors huggingface_hub
mkdir -p data results logs
