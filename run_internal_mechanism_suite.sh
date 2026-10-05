#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Multimodal_test_switch

echo "=== Round 6: 1) Extract paired hidden states + exact intermediate-state labels ==="
python extract_internal_probe_states.py \
  --data data_role_position/role_position_examples.jsonl \
  --results results_role_position/qwen3vl_role_position_full.jsonl \
  --out probe_data/internal_probe_states_round6.pt \
  --roles 1,2,3,4 \
  --max_pairs_per_role_group 40 \
  --layers 0,8,16,20,24,28,32,35

echo "=== Round 6: 2) Train exact-state and paired-delta probes ==="
python train_internal_probes.py \
  --states probe_data/internal_probe_states_round6.pt \
  --out_dir analysis_probes_round6

echo "=== Optional causal control: 3) Run fact-level patching ==="
python run_fact_level_patching.py \
  --data data_role_position/role_position_examples.jsonl \
  --results results_role_position/qwen3vl_role_position_full.jsonl \
  --out results_fact_patching/fact_level_patching.jsonl \
  --roles 1,4 \
  --max_failures_per_role 30 \
  --layers 20:32:2

echo "=== Optional causal control: 4) Analyze fact-level patching ==="
python analyze_fact_level_patching.py \
  --results results_fact_patching/fact_level_patching.jsonl \
  --out_dir analysis_fact_patching

echo "=== DONE ==="
echo "Round-6 probe results: analysis_probes_round6/"
echo "Fact-patching results: analysis_fact_patching/"
