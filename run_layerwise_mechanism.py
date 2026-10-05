#!/usr/bin/env python3
"""
Round 4: layer-wise mechanism pilot.

This script does two things on matched Round-3 pairs:

1) Representation diagnostic
   Compare the final prompt-token hidden state for:
      matched all-Text baseline
      vs. the same example with exactly one logical fact rendered as Vision

   We measure layer-wise cosine similarity for both:
      - SUCCESS pairs: Text correct, Vision correct
      - FAILURE pairs: Text correct, Vision wrong

2) Coarse causal patching
   For FAILURE pairs only, replace the Vision run's final prompt-token residual
   stream after selected decoder layers with the matched Text baseline state,
   then continue generation.

If patching at a layer rescues the wrong Vision answer, that layer is evidence
for where the modality-induced failure has become causally represented in the
global reasoning state.

Important:
This is FINAL-PROMPT-TOKEN residual-stream patching. It is a coarse causal
localization experiment, not yet fact-token-level patching.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from run_qwen3vl import build_messages, parse_label


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_roles(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_results_path(path):
    if path.exists():
        return path

    fallback = Path("results_role_position/qwen3vl_role_position.jsonl")
    if fallback.exists():
        print(f"Results file {path} not found; using fallback {fallback}")
        return fallback

    raise FileNotFoundError(path)


def get_text_layers(model):
    candidates = [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
    ]

    for name, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                print(f"Using decoder layers from: {name}")
                return layers
        except Exception:
            pass

    raise AttributeError(
        "Could not locate text decoder layers. "
        "Expected Qwen3-VL-style model.model.language_model.layers."
    )


def prepare_inputs(processor, example, data_dir, device):
    messages = build_messages(example, data_dir)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


def collect_last_token_states(model, inputs):
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "Model returned no hidden_states. "
            "Check Transformers/Qwen3-VL compatibility."
        )

    # hidden_states[0] is embeddings; hidden_states[i+1] is after decoder layer i.
    states = [
        h[0, -1, :].detach().float().cpu()
        for h in hidden_states[1:]
    ]

    del outputs
    return states


def decode_generated(processor, generated, prompt_len):
    new_ids = generated[:, prompt_len:]
    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, parse_label(text)


def generate_with_patch(
    model,
    processor,
    layers,
    visual_inputs,
    baseline_state,
    layer_idx,
    max_new_tokens,
):
    prompt_len = int(visual_inputs["input_ids"].shape[-1])
    layer = layers[layer_idx]

    def hook_fn(module, module_inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Patch only the initial prefill pass, not one-token decode steps.
        if (
            torch.is_tensor(hidden)
            and hidden.ndim == 3
            and hidden.shape[1] == prompt_len
        ):
            patched = hidden.clone()
            patched[:, -1, :] = baseline_state.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return output

    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **visual_inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
    finally:
        handle.remove()

    return decode_generated(processor, generated, prompt_len)


def build_pairs(dataset_rows, result_rows, roles):
    dataset_by_id = {x["example_id"]: x for x in dataset_rows}
    result_by_id = {x["example_id"]: x for x in result_rows}

    baseline_by_key = {}
    for ex in dataset_rows:
        if ex.get("condition") == "text_baseline":
            baseline_by_key[(ex["problem_id"], ex["order_id"])] = ex

    pairs = []
    for ex in dataset_rows:
        if ex.get("condition") != "single_visual":
            continue

        role = int(ex["visual_logical_step"])
        if role not in roles:
            continue

        vis_res = result_by_id.get(ex["example_id"])
        baseline = baseline_by_key.get((ex["problem_id"], ex["order_id"]))
        if vis_res is None or baseline is None:
            continue

        base_res = result_by_id.get(baseline["example_id"])
        if base_res is None:
            continue

        # We only interpret the intervention when the matched all-text run is correct.
        if not bool(base_res["correct"]):
            continue

        group = "failure" if not bool(vis_res["correct"]) else "success"

        pairs.append({
            "role": role,
            "group": group,
            "problem_id": ex["problem_id"],
            "order_id": ex["order_id"],
            "baseline_example": baseline,
            "visual_example": ex,
            "baseline_result": base_res,
            "visual_result": vis_res,
        })

    return pairs


def balanced_select(pairs, roles, max_pairs_per_group):
    selected = []
    for role in roles:
        for group in ["failure", "success"]:
            bucket = [
                p for p in pairs
                if p["role"] == role and p["group"] == group
            ]
            bucket.sort(key=lambda x: (x["problem_id"], x["order_id"]))
            picked = bucket[:max_pairs_per_group]
            print(
                f"role={role} group={group}: "
                f"available={len(bucket)}, selected={len(picked)}"
            )
            selected.extend(picked)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_role_position/role_position_examples.jsonl"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "results_role_position/qwen3vl_role_position_full.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_mechanism/layerwise_mechanism.jsonl"),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--roles",
        default="1,4",
        help="Logical roles to compare. Default: early role 1 vs late role 4.",
    )
    parser.add_argument(
        "--max_pairs_per_group",
        type=int,
        default=12,
        help="Per role, select up to N failure and N success pairs.",
    )
    parser.add_argument(
        "--layer_stride",
        type=int,
        default=4,
        help="Patch every Nth decoder layer, plus the final layer.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    args.results = choose_results_path(args.results)
    roles = parse_roles(args.roles)
    data_dir = args.data.parent
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows = load_jsonl(args.data)
    result_rows = load_jsonl(args.results)
    pairs = build_pairs(dataset_rows, result_rows, roles)
    selected = balanced_select(
        pairs,
        roles=roles,
        max_pairs_per_group=args.max_pairs_per_group,
    )

    if not selected:
        raise RuntimeError("No eligible matched pairs found.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    layers = get_text_layers(model)
    n_layers = len(layers)
    patch_layers = list(range(0, n_layers, args.layer_stride))
    if (n_layers - 1) not in patch_layers:
        patch_layers.append(n_layers - 1)

    print(f"Number of decoder layers: {n_layers}")
    print("Patch layers:", patch_layers)

    with args.out.open("w", encoding="utf-8") as out_file:
        for pair in tqdm(selected, desc="matched pairs"):
            baseline_ex = pair["baseline_example"]
            visual_ex = pair["visual_example"]

            baseline_inputs = prepare_inputs(
                processor, baseline_ex, data_dir, model.device
            )
            visual_inputs = prepare_inputs(
                processor, visual_ex, data_dir, model.device
            )

            baseline_states = collect_last_token_states(
                model, baseline_inputs
            )
            visual_states = collect_last_token_states(
                model, visual_inputs
            )

            if len(baseline_states) != n_layers:
                raise RuntimeError(
                    f"Expected {n_layers} baseline states, "
                    f"got {len(baseline_states)}"
                )
            if len(visual_states) != n_layers:
                raise RuntimeError(
                    f"Expected {n_layers} visual states, "
                    f"got {len(visual_states)}"
                )

            patch_results = {}
            if pair["group"] == "failure":
                for layer_idx in patch_layers:
                    raw, pred = generate_with_patch(
                        model=model,
                        processor=processor,
                        layers=layers,
                        visual_inputs=visual_inputs,
                        baseline_state=baseline_states[layer_idx],
                        layer_idx=layer_idx,
                        max_new_tokens=args.max_new_tokens,
                    )
                    patch_results[layer_idx] = {
                        "patched_raw_output": raw,
                        "patched_prediction": pred,
                        "patched_correct": bool(
                            pred == visual_ex["answer"]
                        ),
                    }

            for layer_idx in range(n_layers):
                b = baseline_states[layer_idx]
                v = visual_states[layer_idx]

                cosine = F.cosine_similarity(
                    b.unsqueeze(0),
                    v.unsqueeze(0),
                    dim=-1,
                ).item()

                b_norm = F.normalize(b, dim=0)
                v_norm = F.normalize(v, dim=0)
                normalized_l2 = torch.norm(
                    b_norm - v_norm,
                    p=2,
                ).item()

                patch = patch_results.get(layer_idx, {})

                record = {
                    "problem_id": pair["problem_id"],
                    "order_id": pair["order_id"],
                    "logical_role": pair["role"],
                    "physical_position": visual_ex[
                        "visual_presentation_position"
                    ],
                    "group": pair["group"],
                    "answer": visual_ex["answer"],
                    "baseline_example_id": baseline_ex["example_id"],
                    "visual_example_id": visual_ex["example_id"],
                    "baseline_prediction": pair["baseline_result"].get(
                        "prediction"
                    ),
                    "visual_prediction": pair["visual_result"].get(
                        "prediction"
                    ),
                    "baseline_input_tokens": int(
                        baseline_inputs["input_ids"].shape[-1]
                    ),
                    "visual_input_tokens": int(
                        visual_inputs["input_ids"].shape[-1]
                    ),
                    "layer": layer_idx,
                    "cosine_similarity": cosine,
                    "normalized_l2": normalized_l2,
                    "patched_tested": layer_idx in patch_results,
                    "patched_prediction": patch.get(
                        "patched_prediction"
                    ),
                    "patched_correct": patch.get("patched_correct"),
                    "patched_raw_output": patch.get(
                        "patched_raw_output"
                    ),
                }
                out_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                out_file.flush()

            del baseline_inputs, visual_inputs
            del baseline_states, visual_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("Saved:", args.out)
    print(
        "Interpretation reminder: this is coarse final-token residual patching, "
        "not fact-token-level causal tracing."
    )


if __name__ == "__main__":
    main()
