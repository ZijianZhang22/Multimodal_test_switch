# Updated Research Plan: Reasoning-Role-Dependent Modality Effects

## Where the project stands

Round 1 started from a switching-cost hypothesis:

> Does repeatedly switching between Text and Vision hurt multi-hop reasoning?

Round 2 ran all 16 Text/Vision carrier assignments for the same 4-hop latent
problems. The result changed the direction of the project.

The main Round-2 pattern was:

- Step 1: Vision - Text = **+4.38 points**
- Step 2: Vision - Text = **-1.38 points**
- Step 3: Vision - Text = **-7.88 points**
- Step 4: Vision - Text = **-10.13 points**
- switch count was not significant after controlling for the four positions
  (\`p ≈ 0.285\`)

So the project should no longer be framed as "switching cost."

## New core question

> **Does the effect of a modality depend on the logical reasoning role at which
> that evidence is needed, rather than simply on its physical position in the
> prompt?**

The key distinction is:

- **logical reasoning role**: which edge/fact in the latent chain the evidence
  represents (\`f1, f2, f3, f4\`)
- **physical presentation position**: where that evidence appears in the actual
  input sequence (1st, 2nd, 3rd, 4th)

The current Round-2 experiment confounds these because logical Step 1 was also
presented first, Step 4 was presented last, etc.

## Round 3A: Role x Presentation-Position Decoupling

Keep the latent chain fixed but reorder the four evidence items.

For example, the logical chain stays:

\`f1 -> f2 -> f3 -> f4\`

but the prompt may present:

\`f4, f1, f3, f2\`

Exactly one logical fact is rendered as Vision; the other three are Text.

For every presentation order, also run a matched all-Text baseline.

Primary matched effect:

\`delta(role, position) = single-visual accuracy - matched all-text accuracy\`

where the problem and physical presentation order are identical.

### Cheap pilot

Use four balanced presentation orders. They are selected so that each logical
role appears once at each physical position.

With 100 latent problems:

- 4 presentation orders
- 4 single-visual logical roles per order
- 1 matched all-text baseline per order

Total: \`100 x 4 x 5 = 2000\` examples.

### Stronger follow-up

Use all 24 permutations after the cheap pilot if the effect survives.

Total: \`100 x 24 x 5 = 12000\` examples.

## Round 3B: Single-Fact Perception Control

Before claiming an integration failure, test whether the model can simply read
the visual relation.

For every latent fact, ask its pairwise relation directly:

- Text realization
- Vision realization

With 100 problems:

\`100 x 4 facts x 2 modalities = 800\` examples.

Interpretation:

- **High visual single-fact accuracy + late visual chain penalty**
  -> evidence for an integration/reasoning bottleneck.
- **Low visual single-fact accuracy**
  -> the current effect may mainly reflect perception/rendering difficulty and
  must be calibrated before a stronger mechanism claim.

## Main hypotheses

### H1: Reasoning-role hypothesis

The visual penalty remains tied to late logical roles even when those facts are
moved to early physical prompt positions.

This is the most interesting outcome.

### H2: Physical-position hypothesis

The penalty follows where the evidence is physically placed in the prompt.

Then the project is closer to known modality/order bias and should be reframed
accordingly.

### H3: Perception hypothesis

Vision is already much worse on the isolated pairwise relation.

Then the 4-hop effect is not yet evidence for multimodal integration failure.

### H4: Integration hypothesis

Vision is accurate in isolation, but becomes harmful for certain logical roles
inside the chain.

Then the next phase should investigate representation alignment and integration
with layer-wise probes and causal patching.

## Go / No-Go decision

Continue to mechanism work if:

1. visual perception is reasonably strong in the single-fact control, and
2. logical-role effects remain after physical presentation order is controlled.

If the effect is only physical-position bias, the result is still useful but
has more overlap with existing order-bias work.

If the effect is mostly explained by poor visual perception, improve/calibrate
the visual realization before drawing multimodal reasoning conclusions.

## Mechanism phase after Round 3

Only after Round 3 supports the role-dependent effect:

1. compare text vs visual fact representations layer by layer
2. test whether visual information becomes modality-agnostic later than text
3. patch the matched Text representation into the failing Visual run
4. identify whether recovery happens at perception, alignment, integration, or
   reasoning layers
5. then test whether a small pre-alignment/routing intervention reduces the
   late-role visual penalty

## Paper-level framing if the result survives

Avoid:

> Image-first is better than text-first.

Avoid:

> More modality switches hurt reasoning.

Prefer:

> **Prior work shows that modality order matters. We ask whether apparent order
> effects are actually caused by physical presentation order or by which
> logical reasoning roles are assigned to each modality.**

A possible title:

**When Should Evidence Be Visual? Disentangling Reasoning Role and Presentation
Position in Multimodal Multi-Step Reasoning**
