#!/usr/bin/env python3
"""
Round 5A: extract paired internal states for linear-probe diagnostics.

For matched all-Text vs single-Visual examples we save, at selected decoder
layers:
  1) the hidden state at the END OF THE TARGET FACT
  2) the hidden state at the FINAL PROMPT TOKEN

This supports four mechanistic questions:
  - Fact encoding: is the semantic direction decodable at the fact boundary?
  - Intermediate-state composition: are the exact accumulated states
    s_1, s_2, s_3, s_4 = cumulative (x,y) displacements decodable?
  - Modality persistence: can a probe still tell Text vs Vision at each layer?
  - Failure prediction: can internal states predict whether the Visual run
    succeeds or fails?

Splits should be done by problem_id, not by individual examples, to avoid
paired-data leakage.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mechanism_utils import (
    build_latent_labels,
    get_evidence_boundary_positions,
    get_text_layers,
    load_jsonl,
    parse_layer_spec,
    prepare_inputs,
)


def parse_roles(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_results_path(path):
    if path.exists():
        return path
    for fallback in [
        Path("results_role_position/qwen3vl_role_position_full.jsonl"),
        Path("results_role_position/qwen3vl_role_position.jsonl"),
    ]:
        if fallback.exists():
            print(f"Results file {path} not found; using {fallback}")
            return fallback
    raise FileNotFoundError(path)


def build_pairs(dataset_rows, result_rows, roles, max_pairs_per_role_group):
    result_by_id = {x["example_id"]: x for x in result_rows}
    baseline_by_key = {
        (x["problem_id"], x["order_id"]): x
        for x in dataset_rows
        if x.get("condition") == "text_baseline"
    }

    buckets = {}
    for role in roles:
        buckets[(role, "success")] = []
        buckets[(role, "failure")] = []

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
        if base_res is None or not bool(base_res["correct"]):
            continue

        group = "success" if bool(vis_res["correct"]) else "failure"
        buckets[(role, group)].append((baseline, ex, base_res, vis_res))

    selected = []
    for key, bucket in buckets.items():
        bucket.sort(key=lambda x: (x[1]["problem_id"], x[1]["order_id"]))
        picked = bucket[:max_pairs_per_role_group]
        print(
            f"role={key[0]} group={key[1]}: "
            f"available={len(bucket)} selected={len(picked)}"
        )
        selected.extend([
            {
                "role": key[0],
                "group": key[1],
                "baseline": b,
                "visual": v,
                "baseline_result": br,
                "visual_result": vr,
            }
            for b, v, br, vr in picked
        ])
    return selected


def collect_states(model, processor, example, data_dir, layers_to_keep):
    inputs = prepare_inputs(processor, example, data_dir, model.device)
    boundaries = get_evidence_boundary_positions(inputs, processor, example)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model returned no hidden_states.")

    states = {}
    for layer in layers_to_keep:
        h = hidden_states[layer + 1][0]
        states[layer] = {
            "final": h[boundaries["final"]].detach().float().cpu(),
            "logical": {
                logical: h[token_pos].detach().float().cpu()
                for logical, token_pos in boundaries["logical"].items()
            },
        }

    token_count = int(inputs["input_ids"].shape[-1])

    del outputs, inputs
    return states, boundaries, token_count


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
        default=Path("results_role_position/qwen3vl_role_position_full.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("probe_data/internal_probe_states.pt"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--roles",
        default="1,2,3,4",
        help="Logical roles to include.",
    )
    parser.add_argument(
        "--max_pairs_per_role_group",
        type=int,
        default=40,
        help="Max success and failure pairs per logical role.",
    )
    parser.add_argument(
        "--layers",
        default="0,8,16,20,24,28,32,35",
        help='Layer spec, e.g. "20:32", "20:32:2", "0,8,16,24,28,32,35", or "all".',
    )
    args = parser.parse_args()

    args.results = choose_results_path(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    roles = parse_roles(args.roles)

    dataset_rows = load_jsonl(args.data)
    result_rows = load_jsonl(args.results)
    pairs = build_pairs(
        dataset_rows,
        result_rows,
        roles,
        args.max_pairs_per_role_group,
    )
    if not pairs:
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

    layer_path, layers = get_text_layers(model)
    n_layers = len(layers)
    layers_to_keep = parse_layer_spec(args.layers, n_layers)
    print("Decoder path:", layer_path)
    print("Selected layers:", layers_to_keep)

    records = []
    data_dir = args.data.parent

    for pair in tqdm(pairs, desc="paired state extraction"):
        role = int(pair["role"])
        baseline = pair["baseline"]
        visual = pair["visual"]

        baseline_states, baseline_bounds, baseline_tokens = collect_states(
            model, processor, baseline, data_dir, layers_to_keep
        )
        visual_states, visual_bounds, visual_tokens = collect_states(
            model, processor, visual, data_dir, layers_to_keep
        )

        labels = build_latent_labels(visual)

        for variant, example, states, bounds, token_count, result in [
            (
                "T",
                baseline,
                baseline_states,
                baseline_bounds,
                baseline_tokens,
                pair["baseline_result"],
            ),
            (
                "V",
                visual,
                visual_states,
                visual_bounds,
                visual_tokens,
                pair["visual_result"],
            ),
        ]:
            for layer in layers_to_keep:
                target_state = states[layer]["logical"][role]
                final_state = states[layer]["final"]

                common = {
                    "problem_id": example["problem_id"],
                    "order_id": example["order_id"],
                    "logical_role": role,
                    "physical_position": int(
                        visual["visual_presentation_position"]
                    ),
                    "outcome_group": pair["group"],
                    "variant": variant,
                    "correct": bool(result["correct"]),
                    "layer": int(layer),
                    "fact_direction": labels[f"fact{role}"],
                    "prefix_state": labels[f"prefix{role}"],
                    "suffix_state": labels[f"suffix{role}"],

                    # Exact intermediate reasoning states s_1 ... s_4.
                    # Store every prefix coordinate on every record so the
                    # downstream probe can ask whether the FINAL prompt state
                    # still contains each intermediate computation.
                    "state1_x": labels["state1_x"],
                    "state1_y": labels["state1_y"],
                    "state1_coord": labels["state1_coord"],
                    "state2_x": labels["state2_x"],
                    "state2_y": labels["state2_y"],
                    "state2_coord": labels["state2_coord"],
                    "state3_x": labels["state3_x"],
                    "state3_y": labels["state3_y"],
                    "state3_coord": labels["state3_coord"],
                    "state4_x": labels["state4_x"],
                    "state4_y": labels["state4_y"],
                    "state4_coord": labels["state4_coord"],

                    "answer": labels["answer"],
                    "input_tokens": token_count,
                }

                records.append({
                    **common,
                    "site": "target_boundary",
                    "feature": target_state.to(torch.float16),
                })
                records.append({
                    **common,
                    "site": "final_prompt",
                    "feature": final_state.to(torch.float16),
                })

        del baseline_states, visual_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    payload = {
        "model": args.model,
        "n_layers": n_layers,
        "layers": layers_to_keep,
        "roles": roles,
        "records": records,
        "notes": (
            "target_boundary = residual stream at end of target logical fact; "
            "final_prompt = residual stream at final prompt token before generation; "
            "stateK_x/stateK_y/stateK_coord = exact cumulative reasoning state "
            "s_K = f_1 + ... + f_K."
        ),
    }
    torch.save(payload, args.out)
    print(f"Saved {len(records)} state records to {args.out}")


if __name__ == "__main__":
    main()
