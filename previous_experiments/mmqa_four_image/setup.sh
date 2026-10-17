#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U 'transformers>=4.57,<6' accelerate safetensors pillow huggingface_hub
mkdir -p data
if [ ! -f data/multimodalqa/dataset/MMQA_dev.jsonl.gz ]; then
  git clone --depth=1 https://github.com/allenai/multimodalqa.git data/multimodalqa
fi
if [ ! -s data/images.zip ]; then
  wget -c 'https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip' -O data/images.zip
fi
