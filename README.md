# Multimodal Test Switch

This repository now studies **reasoning-role-dependent modality effects** in
multi-step multimodal reasoning.

The project began with a switching-cost hypothesis, but the first two rounds
showed a more interesting pattern: switch count itself was not significant,
while the effect of Vision changed strongly across reasoning steps.

See \`RESEARCH_PLAN.md\` for the updated scientific framing.

## Current finding from Round 2

Using the same 100 latent 4-hop spatial problems and all \(2^4=16\) Text/Vision
carrier assignments:

- Step 1: Vision - Text ≈ **+4.38 points**
- Step 2: ≈ **-1.38 points**
- Step 3: ≈ **-7.88 points**
- Step 4: ≈ **-10.13 points**
- switch count was not significant after controlling for step position

The next question is whether this is a **logical reasoning-role effect** or
merely a **physical prompt-position effect**.

---

# Round 3

Round 3 has two controls:

1. **Logical role x physical presentation position**
2. **Single-fact perception control**

## 1. Update the repo on RunPod

\`\`\`bash
cd /workspace/Multimodal_test_switch
git pull
pip install -r requirements.txt
\`\`\`

## 2. Generate the Round-3 datasets

This reuses the exact same latent problems and images from \`data16\`.

\`\`\`bash
python generate_role_position_dataset.py \
  --source_dir data16 \
  --out_dir data_role_position \
  --order_set balanced4
\`\`\`

For 100 latent problems, the default balanced pilot creates:

- **2000 role-position examples**
  - 4 balanced physical presentation orders
  - 4 single-visual logical roles per order
  - 1 matched all-text baseline per order
- **800 perception-control examples**
  - 4 facts × Text/Vision × 100 problems

The four presentation orders are balanced so every logical role appears exactly
once in every physical position.

## 3. Smoke test the role-position experiment

\`\`\`bash
python run_qwen3vl.py \
  --data data_role_position/role_position_examples.jsonl \
  --out results_role_position/smoke_role_position.jsonl \
  --limit 20
\`\`\`

Then run the full pilot:

\`\`\`bash
python run_qwen3vl.py \
  --data data_role_position/role_position_examples.jsonl \
  --out results_role_position/qwen3vl_role_position.jsonl
\`\`\`

Analyze:

\`\`\`bash
python analyze_role_position.py \
  --results results_role_position/qwen3vl_role_position.jsonl \
  --out_dir analysis_role_position
\`\`\`

Key outputs:

- \`cell_accuracy.csv\`
- \`accuracy_by_logical_role.csv\`
- \`accuracy_by_presentation_position.csv\`
- \`matched_delta_by_logical_role.csv\`
- \`matched_delta_by_presentation_position.csv\`
- \`clustered_logit_role_plus_physical_position.csv\`
- two role × physical-position heatmaps

The most important quantity is the matched delta:

\`\`\`text
single-visual accuracy - matched all-text accuracy
\`\`\`

for the same latent problem and the exact same presentation order.

## 4. Run the single-fact perception control

\`\`\`bash
python run_qwen3vl.py \
  --data data_role_position/perception_examples.jsonl \
  --out results_role_position/qwen3vl_perception.jsonl
\`\`\`

Analyze:

\`\`\`bash
python analyze_perception.py \
  --results results_role_position/qwen3vl_perception.jsonl \
  --out_dir analysis_perception
\`\`\`

This checks whether the visual diagram can be read correctly in isolation.

## How to interpret Round 3

### Strong result

If isolated visual perception is high **and** the late-logical-role penalty
remains after physical presentation position is controlled:

> the effect is not just basic perception and not just prompt order.

That would support a **reasoning-role-dependent multimodal integration effect**
and justify mechanism work.

### Order-bias result

If the penalty follows physical presentation position instead of logical role:

> the phenomenon is mainly a prompt/order effect.

This is still useful, but overlaps more strongly with existing order-bias work.

### Perception result

If the diagrams are already much worse than text in the single-fact test:

> improve/calibrate visual realization before making a multimodal integration
> claim.

## Stronger follow-up

If the balanced-4 pilot survives, run all 24 presentation permutations:

\`\`\`bash
python generate_role_position_dataset.py \
  --source_dir data16 \
  --out_dir data_role_position24 \
  --order_set all24
\`\`\`

With 100 problems this creates 12,000 role-position examples, so use it only
after the cheaper balanced pilot gives a clear signal.

## Previous experiments

The original 16-schedule generator and analysis remain available:

\`\`\`bash
python generate_switching_dataset.py \
  --out_dir data16 \
  --n_problems 100 \
  --seed 42

python run_qwen3vl.py \
  --data data16/examples.jsonl \
  --out results16/qwen3vl_results.jsonl

python analyze_results.py \
  --results results16/qwen3vl_results.jsonl \
  --out_dir analysis16
\`\`\`


---

# Round 5: Internal Mechanism Diagnostics

Behavioral controls now motivate direct internal-state tests. The current
mechanism hypothesis is:

> Visual facts are perceived correctly, but late logical-role evidence may be
> less successfully written into / integrated with the evolving reasoning
> state.

The repo now includes:

- \`mechanism_utils.py\` — token-boundary and latent-label helpers
- \`extract_internal_probe_states.py\` — collect matched Text/Vision hidden states
- \`train_internal_probes.py\` — grouped linear probes
- \`run_fact_level_patching.py\` — local fact-boundary causal patching
- \`analyze_fact_level_patching.py\` — rescue-rate + matched-control analysis
- \`run_internal_mechanism_suite.sh\` — run the full internal suite

## What the probes test

At the **target fact boundary**:

- can we decode the semantic fact direction?
- can we decode whether the carrier was Text or Vision?
- can we predict whether the Visual run will eventually succeed or fail?

At the **final prompt state**:

- is the target fact still decodable?
- is the final answer decodable?
- does modality identity persist?
- when does the hidden state begin to predict failure?

All probe train/test splits are grouped by \`problem_id\` to avoid leakage from
matched variants of the same latent problem.

## What fact-level patching tests

For a failing single-Visual run, patch only the residual state at the end of the
target visual fact. Compare four donors:

1. \`matched_fact\` — same problem / same logical fact / all-Text baseline
2. \`same_example_other_fact\` — same problem but wrong logical fact
3. \`unrelated_same_answer\` — different problem with the same answer
4. \`unrelated_other_answer\` — different problem and different answer

The desired causal signature is:

\`matched_fact rescue >> controls\`

especially for late logical roles in the mid/late decoder window.

## Run everything

\`\`\`bash
cd /workspace/Multimodal_test_switch
git pull
pip install -r requirements.txt
chmod +x run_internal_mechanism_suite.sh
./run_internal_mechanism_suite.sh
\`\`\`

Main outputs:

\`\`\`text
analysis_probes/probe_results.csv
analysis_probes/best_probe_layer_summary.csv

analysis_fact_patching/rescue_by_role_layer_control.csv
analysis_fact_patching/best_layer_by_role_control.csv
analysis_fact_patching/matched_vs_controls_paired_tests.csv
\`\`\`

Recommended interpretation:

- local fact-direction probe high, but final-state fact/answer probe drops for
  late Visual failures -> integration / state-update bottleneck
- modality probe stays high late -> incomplete modality-invariant alignment
- Visual-failure probe becomes predictive in a specific layer range -> candidate
  failure-formation window
- matched fact patch rescues much more than controls -> fact-specific causal
  evidence for an integration bottleneck
