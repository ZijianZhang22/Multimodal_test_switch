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


---

# Round 6: Exact Intermediate-State Probes

Round 5 changed the mechanism hypothesis.

Observed so far:

- single-fact Text/Vision perception is essentially perfect
- target fact direction remains strongly linearly decodable
- the final prompt state can still retain the target fact
- local fact-boundary patching does **not** selectively rescue errors
- coarse global final-state patching can rescue some failures

Therefore the current priority is to distinguish:

1. **relational composition / state-update failure**
2. **downstream decision / readout failure**

## What changed in Round 6

### 1. Exact accumulated reasoning-state labels

For every latent problem, the code now stores:

\`\`\`text
s1 = f1
s2 = f1 + f2
s3 = f1 + f2 + f3
s4 = f1 + f2 + f3 + f4
\`\`\`

Each state is represented by its exact displacement:

\`\`\`text
stateK_x
stateK_y
stateK_coord
\`\`\`

This is richer than the final direction answer. For example, \`(1,0)\` and
\`(3,0)\` are both EAST as an answer, but they are different internal reasoning
states.

### 2. Exact-state probes

\`train_internal_probes.py\` now probes, layer by layer:

- \`state1_x/state1_y\`
- \`state2_x/state2_y\`
- \`state3_x/state3_y\`
- \`state4_x/state4_y\`

at the final prompt state.

The most important diagnostic is the target-role state:

- Role 1 visual run -> inspect \`s1\`
- Role 2 visual run -> inspect \`s2\`
- Role 3 visual run -> inspect \`s3\`
- Role 4 visual run -> inspect \`s4\`

If the target visual fact is decodable but the corresponding accumulated state
degrades, that supports a **state-update / relational-composition failure**.

If the exact accumulated state remains highly decodable but the generated
answer is wrong, that shifts evidence toward a **decision/readout failure**.

### 3. Paired V-minus-T failure probe

The outcome probe now also uses:

\`\`\`text
delta_h = h_visual - h_text
\`\`\`

for the exact matched latent problem and presentation order.

This control is important because a raw Visual-only outcome probe can partially
learn static problem difficulty.  The paired delta asks whether the
**modality-induced representational change itself** predicts failure.

## Run Round 6

\`\`\`bash
cd /workspace/Multimodal_test_switch
git pull
pip install -r requirements.txt

python extract_internal_probe_states.py \
  --data data_role_position/role_position_examples.jsonl \
  --results results_role_position/qwen3vl_role_position_full.jsonl \
  --out probe_data/internal_probe_states_round6.pt \
  --roles 1,2,3,4 \
  --max_pairs_per_role_group 40 \
  --layers 0,8,16,20,24,28,32,35

python train_internal_probes.py \
  --states probe_data/internal_probe_states_round6.pt \
  --out_dir analysis_probes_round6
\`\`\`

Or run the whole mechanism suite:

\`\`\`bash
chmod +x run_internal_mechanism_suite.sh
./run_internal_mechanism_suite.sh
\`\`\`

## New Round-6 outputs

\`\`\`text
analysis_probes_round6/probe_results_round6.csv
analysis_probes_round6/best_probe_layer_summary_round6.csv
analysis_probes_round6/target_role_state_probe_curves.csv
analysis_probes_round6/outcome_probe_raw_vs_delta.csv
analysis_probes_round6/probe_*.png
\`\`\`

The two most important files are:

\`\`\`text
target_role_state_probe_curves.csv
outcome_probe_raw_vs_delta.csv
\`\`\`


---

# Next Stage: 32B + Variable-Hop Depth Scaling

The repository has now been checked for the next paper stage.

## Code that is now generalized

- \`run_qwen3vl.py\`
  - Qwen3-VL-8B / 32B
  - arbitrary evidence count
  - safe \`--resume\`
  - model/dtype/device options

- \`mechanism_utils.py\`
  - arbitrary hop count \`H\`
  - evidence-boundary lookup for \`Evidence 1 ... Evidence H\`
  - dynamic \`state1 ... stateH\`
  - percentage layer specs such as
    \`0%,25%,50%,65%,75%,90%,100%\`

- \`extract_internal_probe_states.py\`
  - arbitrary hop count
  - \`--roles all\`
  - 8B/32B
  - normalized layer-depth metadata

- \`train_internal_probes.py\`
  - dynamic \`state1 ... stateH\` probing

## New scripts

### 1. Variable-hop dataset

\`\`\`bash
python generate_depth_scaling_dataset.py \
  --out_dir data_depth_scaling \
  --hops 4,6,8,12 \
  --n_problems_per_hop 200 \
  --order_mode identity
\`\`\`

Each latent problem receives:

\`\`\`text
all-Text
V@role1
V@role2
...
V@roleH
\`\`\`

so the number of interventions is O(H), not 2^H.

For a stronger position control use \`--order_mode reverse_pair\`, or use
\`--order_mode balanced_cyclic\` for full role-position balance. The latter is
much more expensive.

### 2. 8B depth run

\`\`\`bash
chmod +x run_depth_scaling_model.sh

MODEL=Qwen/Qwen3-VL-8B-Instruct \
LABEL=qwen3vl8b \
./run_depth_scaling_model.sh
\`\`\`

### 3. 32B depth run

\`\`\`bash
MODEL=Qwen/Qwen3-VL-32B-Instruct \
LABEL=qwen3vl32b \
DTYPE=bf16 \
./run_depth_scaling_model.sh
\`\`\`

The runner uses \`--resume\`, so an interrupted long run can continue safely.

### 4. Analyze model scale × reasoning depth

\`\`\`bash
python analyze_depth_scaling.py \
  --results \
    results_depth_scaling/qwen3vl8b.jsonl \
    results_depth_scaling/qwen3vl32b.jsonl \
  --out_dir analysis_depth_scaling
\`\`\`

Main outputs:

\`\`\`text
all_text_calibration.csv
matched_delta_by_role_depth.csv
matched_delta_by_depth_bucket.csv
clustered_depth_trend_regression.csv
matched_examples.csv
depth_scaling_<model>.png
\`\`\`

The key variable is normalized logical depth \`(k-1)/(H-1)\`, and the primary
estimand is single-Visual correctness minus matched all-Text correctness.

## Immediate mechanism control

\`analyze_matched_state_decoding.py\` directly compares Text and Vision
representations of the SAME accumulated reasoning state using the SAME grouped
train/test splits.

It reports both:

- discrete x/y state classification
- continuous Ridge x/y regression (R² and MAE)

\`\`\`bash
python analyze_matched_state_decoding.py \
  --states probe_data/internal_probe_states_round6.pt \
  --out_dir analysis_matched_state_round6
\`\`\`

Important outputs:

\`\`\`text
matched_state_compact_classification.csv
matched_state_compact_r2.csv
matched_state_classification_gap.csv
matched_state_r2_gap.csv
matched_state_mae_gap.csv
\`\`\`

This is the highest-priority mechanism control before making a strong
state-composition claim.

## Recommended execution order

1. Run matched Text-vs-Vision state decoding on the existing 8B states.
2. Replicate the original 4-hop role/position result with 32B.
3. Calibrate all-Text accuracy for 4/6/8/12 hops.
4. Run single-Visual depth scaling only where the baseline is not at floor.
5. Compare 8B vs 32B role-depth curves.
6. Only then spend compute on long-chain mechanism probes.
