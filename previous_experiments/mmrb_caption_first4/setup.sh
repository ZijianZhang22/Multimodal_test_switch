#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U "datasets>=3" "transformers>=4.57,<6" accelerate pillow safetensors huggingface_hub
mkdir -p data results logs
python - <<'PY'
from huggingface_hub import list_repo_files
files=list_repo_files("HarrytheOrange/MMRB",repo_type="dataset")
print("Official dataset files:")
for f in files:
    if f.lower().endswith((".zip",".json",".parquet")): print(" ",f)
PY
