# Revised Research Plan: Reasoning-Role-Dependent Multimodal Utilization

**Status date:** 2026-10-05

## 1. Current project framing

The original project asked whether repeated Vision/Text switching creates an
independent multimodal reasoning cost.

That hypothesis is no longer the main story.

The 4-hop experiments show that **switch count itself is not significant**,
while the effect of Vision depends strongly on the **logical reasoning role**
occupied by the visual evidence.

The current core question is:

> **When and why does visual evidence become less usable as its logical role
> moves later in a reasoning chain?**

The working phenomenon name is:

> **reasoning-role-dependent multimodal utilization failure**

This framing is intentionally narrower than the generic claim that a model can
"see a fact but fail to use it." Recent work such as *Seeing, Saying, but Not
Using* (arXiv:2610.02876) already establishes an availability-utilization gap
for spatial state. Our contribution must instead isolate **logical role** as a
controlled causal variable.

---

## 2. What is already established

### 2.1 Switching cost is not the main effect

Round 1 and Round 2 do not show a monotonic decline with more V↔T switches.

In the all-16 factorial analysis:

- Vision at logical Step 1: about **+4.38 points**
- Step 2: about **-1.38 points**
- Step 3: about **-7.88 points**
- Step 4: about **-10.13 points**
- switch count: not significant after controlling for step position

So the project should not be presented as a switching-cost paper.

### 2.2 Logical role is stronger than physical prompt position

Round 3 shuffled the physical presentation order while keeping the latent
reasoning chain fixed.

After controlling for physical position:

- Role 2 vs Role 1: OR ≈ 0.80, p ≈ 0.032
- Role 3 vs Role 1: OR ≈ 0.61, p ≈ 4.45e-6
- Role 4 vs Role 1: OR ≈ 0.49, p ≈ 2.5e-6

Physical position 2/3 were not significant; physical position 4 had a smaller
negative effect.

Current behavioral conclusion:

> **logical-role effect >> ordinary presentation-position effect**

### 2.3 Basic perception is not the explanation

The single-fact control gives 100% accuracy for both Text and Vision across all
four latent steps.

Therefore the late-role penalty is not well explained by the visual relation
being unreadable.

### 2.4 Local fact loss is not the main mechanism

Current mechanism evidence:

- target fact direction remains strongly linearly decodable
- final-prompt state can retain the visual fact
- global Text/Vision cosine similarity remains high
- matched fact-boundary activation patching does not selectively rescue errors
  relative to other-fact / unrelated controls

This weakens a simple "late visual fact is lost" explanation.

### 2.5 Distributed trajectory divergence is currently the strongest mechanism signal

For matched Text/Visual variants of the same problem/order, the paired hidden
state difference

`delta_h = h_visual - h_text`

strongly predicts final failure for late roles.

Examples from Round 6:

- Role 3: balanced failure-prediction accuracy reaches about 0.95
- Role 4: about 0.89 in late layers

This suggests that the modality substitution changes the distributed reasoning
trajectory in a way that is highly predictive of failure.

---

## 3. Current unresolved mechanism question

The main unresolved distinction is:

1. **relational composition / state-update degradation**
2. **downstream decision / readout failure**

Round 6 probes exact accumulated states

`s_k = f_1 + ... + f_k = (x_k, y_k)`

and shows useful signals, especially for Role 3, but raw state decoding cannot
yet be compared fairly across roles because the class support differs with k.

The mechanism claim therefore still needs one key matched control.

---

## 4. Phase A — Immediate mechanism control (highest priority)

### 4.1 Matched Text-vs-Vision state decoding

For the exact same latent problem, presentation order, logical role, and target
state, compare:

- matched all-Text run
- single-Visual run

Probe the same `s_k` in both conditions.

Primary quantity:

`decode_gap_k = Decode_Vision(s_k) - Decode_Text(s_k)`

This avoids comparing raw probe accuracy across different logical roles.

### 4.2 Continuous state regression

In addition to discrete x/y classification, use linear/ridge regression for the
exact cumulative coordinates:

- x_k
- y_k

Report:

- R²
- MAE
- matched Vision-minus-Text degradation

This reduces sensitivity to different class counts at different depths.

### 4.3 Multi-seed robustness

Use at least 5 grouped train/test splits by `problem_id`.

Report mean ± std or bootstrap confidence intervals.

### 4.4 Decision rule

If the target visual fact remains decodable but matched Vision `s_k` decoding
drops relative to Text, call the mechanism:

> **role-dependent state-composition / state-update degradation**

If `s_k` remains comparably decodable in Text and Vision but the final answer
still fails, shift the mechanism framing toward:

> **decision/readout utilization failure**

---

## 5. Phase B — Qwen3-VL 8B -> 32B scale replication

Use the exact same 4-hop Round-3 dataset first.

Do not change model size and task structure at the same time.

Primary quantity:

`Delta_k = Acc(single-V at logical role k) - Acc(matched all-Text)`

Compare:

- `Delta_k^(8B)`
- `Delta_k^(32B)`

Questions:

- Does scale reduce the late-role gap?
- Does scale merely move the failure onset later?
- Does the role curve remain qualitatively unchanged?

### Layer comparison

Do not reuse 8B absolute layer indices for 32B.

Use normalized decoder depth such as:

- 0%
- 25%
- 50%
- 65%
- 75%
- 90%
- final layer

---

## 6. Phase C — Variable-hop depth scaling

Target hop lengths:

- 4
- 6
- 8
- 12

### 6.1 Calibration first

For each H, first run all-Text.

Prefer hop ranges where all-Text accuracy is roughly 65-95%, avoiding obvious
floor/ceiling regimes.

### 6.2 Long-chain intervention design

Do not enumerate all `2^H` schedules.

Use:

- all-Text baseline
- one single-Visual replacement at every logical role

Cost becomes O(H).

### 6.3 Normalized logical depth

Define:

`r = (k - 1) / (H - 1)`

Analyze Vision replacement as a function of relative reasoning depth.

The main scaling question becomes:

> Does the visual penalty become more negative with normalized logical depth,
> and does longer reasoning amplify that trend?

### 6.4 Physical-order control

For representative early/middle/late roles, continue to balance or shuffle
physical presentation order so depth effects cannot be reduced to recency.

---

## 7. Phase D — Cross-family and cross-domain generalization

Only after the 8B/32B within-family replication is stable:

### Second model family

Add one architecture-different VLM.

The purpose is replication, not another large mechanism sweep.

### Second reasoning domain

Preferred first extension:

> graph/path reasoning

Requirements:

- explicit `f_1 ... f_H`
- exact intermediate states
- Text/Vision twin realization for every fact
- matched single-Visual role intervention

The second domain tests whether the role-dependent effect is specific to spatial
relations or generalizes to another compositional structure.

---

## 8. Phase E — Component localization and mitigation

Only do this if Phase A produces a stable mechanism window.

Possible next steps:

- Attention vs MLP contribution
- component/head ablation
- state-explicit prompting
- intermediate-state supervision
- readout-specific intervention if the bottleneck is downstream

Self-routing / oracle-routing is no longer the default mitigation because
switching/routing lag is no longer the strongest explanation.

---

## 9. Statistical analysis

### Behavioral model

A useful mixed model is:

`Correct ~ VisionReplacement * NormalizedDepth * HopLength * ModelScale + PhysicalPosition + (1 | latent_problem)`

Also report matched bootstrap / McNemar comparisons.

### Mechanism model

Prioritize:

- matched Text-vs-Vision decode gap
- paired `delta_h = h_V - h_T`
- relative layer depth
- grouped splits by latent problem

Do not interpret raw probe differences across logical roles without controlling
for target-state complexity.

---

## 10. Code status

### Completed for the next stage

`run_qwen3vl.py` supports:

- Qwen3-VL-8B and Qwen3-VL-32B
- arbitrary evidence count
- model labels and dtype/device options
- safe `--resume` for long runs

`generate_depth_scaling_dataset.py` now supports:

- arbitrary hop counts such as 4/6/8/12
- all-Text + single-Visual-at-each-role interventions
- identity, reverse-pair, or fully balanced cyclic presentation orders

`mechanism_utils.py` now supports:

- arbitrary hop count H
- dynamic evidence-boundary lookup
- dynamic `state1 ... stateH`
- normalized layer selections such as
  `0%,25%,50%,65%,75%,90%,100%`

`extract_internal_probe_states.py` and `train_internal_probes.py` are now
variable-hop compatible.

`analyze_depth_scaling.py` now analyzes:

- all-Text calibration
- matched single-Visual penalty
- normalized logical depth
- hop-length interaction
- 8B-vs-32B model-scale comparison

`analyze_matched_state_decoding.py` implements the immediate mechanism control:

- same matched latent problems
- same grouped train/test split for Text and Vision
- x/y classification
- Ridge x/y regression
- five-seed Text-vs-Vision decode gaps

### Immediate execution priority

1. Run `analyze_matched_state_decoding.py` on the existing Round-6 8B state file.
2. Replicate the 4-hop role/position experiment with Qwen3-VL-32B.
3. Generate 4/6/8/12-hop calibration data.
4. Run all-Text calibration first and drop hop lengths that enter a floor regime.
5. Run single-Visual depth scaling on the retained hop lengths.
6. Compare 8B and 32B role-depth curves.
7. Only after a stable behavioral result, run long-hop mechanism probes.

### Still not implemented / still needed for the paper

- second VLM family
- second reasoning domain
- adapter from an existing real benchmark into the RoleSwap format
- final mitigation experiment after the mechanism is resolved

---

## 11. Updated paper structure

1. **Introduction** — reasoning-role-dependent visual usability
2. **Controlled benchmark** — matched Text/Vision fact carriers and role intervention
3. **Behavioral finding** — role effect; switch-count null result
4. **Controls** — physical position, perception, matched all-Text
5. **Mechanism** — fact availability, local patching null, paired trajectory shift, matched state decoding
6. **Scale and depth** — 8B vs 32B; 4/6/8/12 hops
7. **Generalization** — second family + second domain
8. **Related work / discussion**

---

## 12. Minimum evidence for a paper

### Already satisfied

- controlled paired 4-hop benchmark
- all-16 carrier decomposition
- logical-role vs physical-position control
- isolated perception calibration
- initial mechanistic probing and causal patching

### Still needed for a stronger submission

- matched Text-vs-Vision state decoding with robust statistics
- 32B replication
- longer-hop replication
- ideally a second model family
- ideally a second reasoning domain

The main-conference version should not rely only on one model and one synthetic
spatial domain.

---

## 13. Current one-sentence pitch

> **We show that the usability of otherwise correctly perceived visual evidence
> depends systematically on the logical role it occupies within a reasoning
> chain, even after controlling for physical prompt position, and we study how
> this role dependence emerges internally and scales with reasoning depth and
> model size.**
