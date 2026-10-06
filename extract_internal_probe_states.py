#!/usr/bin/env python3
"""
Extract paired internal states for matched all-Text vs single-Visual examples.

Works for arbitrary hop count H and both Qwen3-VL-8B / Qwen3-VL-32B.

For each matched pair, save at selected decoder layers:
  1) hidden state at the end of the target logical fact
  2) hidden state at the final prompt token

The output stores exact accumulated reasoning states s_1...s_H, allowing later
matched Text-vs-Vision state decoding and paired V-T trajectory analysis.
"""

import argparse
import gc
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


def parse_roles(text, dataset_rows):
    raw = str(text).strip().lower()
    if raw == "all":
        roles = sorted({
            int(x["visual_logical_step"])
            for x in dataset_rows
            if x.get("condition") == "single_visual"
            and x.get("visual_logical_step") is not None
        })
        return roles
    return sorted({
        int(x.strip()) for x in str(text).split(",") if x.strip()
    })


def resolve_dtype(name):
    name = name.lower()
    if name == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_pairs(dataset_rows, result_rows, roles, max_pairs_per_role_group):
    result_by_id = {x["example_id"]: x for x in result_rows}
    baseline_by_key = {
        (
            x["problem_id"],
            x.get("order_id", "o00"),
            int(x.get("hop_count", len(x["facts"]))),
        ): x
        for x in dataset_rows
        if x.get("condition") == "text_baseline"
    }

    buckets = {
        (role, group): []
        for role in roles
        for group in ["success", "failure"]
    }

    for ex in dataset_rows:
        if ex.get("condition") != "single_visual":
            continue
        role = int(ex["visual_logical_step"])
        if role not in roles:
            continue

        key = (
            ex["problem_id"],
            ex.get("order_id", "o00"),
            int(ex.get("hop_count", len(ex["facts"]))),
        )
        baseline = baseline_by_key.get(key)
        vis_res = result_by_id.get(ex["example_id"])
        if baseline is None or vis_res is None:
            continue

        base_res = result_by_id.get(baseline["example_id"])
        if base_res is None or not bool(base_res["correct"]):
            continue

        group = "success" if bool(vis_res["correct"]) else "failure"
        buckets[(role, group)].append((baseline, ex, base_res, vis_res))

    selected = []
    for key, bucket in buckets.items():
        bucket.sort(
            key=lambda x: (
                x[1]["problem_id"],
                x[1].get("order_id", "o00"),
            )
        )
        picked = bucket[:max_pairs_per_role_group]
        print(
            f"role={key[0]} group={key[1]}: "
            f"available={len(bucket)} selected={len(picked)}"
        )
        for baseline, visual, base_res, vis_res in picked:
            selected.append({
                "role": key[0],
                "group": key[1],
                "baseline": baseline,
                "visual": visual,
                "baseline_result": base_res,
                "visual_result": vis_res,
            })
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("probe_data/internal_probe_states.pt"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_label", default=None)
    parser.add_argument(
        "--roles",
        default="all",
        help='Comma-separated logical roles or "all".',
    )
    parser.add_argument(
        "--max_pairs_per_role_group",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--layers",
        default="0%,25%,50%,65%,75%,90%,100%",
        help=(
            'Absolute indices/ranges or normalized depth percentages. '
            'Percentages are recommended for 8B-vs-32B comparisons.'
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument(
        "--dtype",
        choices=[
            "auto", "bf16", "bfloat16", "fp16",
            "float16", "fp32", "float32",
        ],
        default="auto",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows = load_jsonl(args.data)
    result_rows = load_jsonl(args.results)
    roles = parse_roles(args.roles, dataset_rows)

    pairs = build_pairs(
        dataset_rows,
        result_rows,
        roles,
        args.max_pairs_per_role_group,
    )
    if not pairs:
        raise RuntimeError("No eligible matched pairs found.")

    print("Loading processor:", args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading model:", args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    layer_path, layers = get_text_layers(model)
    n_layers = len(layers)
    layers_to_keep = parse_layer_spec(args.layers, n_layers)
    print("Decoder path:", layer_path)
    print("Number of decoder layers:", n_layers)
    print("Selected layers:", layers_to_keep)

    records = []
    data_dir = args.data.parent
    model_label = args.model_label or args.model.split("/")[-1]

    for pair in tqdm(pairs, desc="paired state extraction"):
        role = int(pair["role"])
        baseline = pair["baseline"]
        visual = pair["visual"]
        hop_count = int(visual.get("hop_count", len(visual["facts"])))

        baseline_states, _, baseline_tokens = collect_states(
            model, processor, baseline, data_dir, layers_to_keep
        )
        visual_states, _, visual_tokens = collect_states(
            model, processor, visual, data_dir, layers_to_keep
        )

        labels = build_latent_labels(visual)

        for variant, example, states, token_count, result in [
            (
                "T", baseline, baseline_states, baseline_tokens,
                pair["baseline_result"],
            ),
            (
                "V", visual, visual_states, visual_tokens,
                pair["visual_result"],
            ),
        ]:
            for layer in layers_to_keep:
                common = {
                    "problem_id": example["problem_id"],
                    "order_id": example.get("order_id", "o00"),
                    "hop_count": hop_count,
                    "logical_role": role,
                    "normalized_logical_depth": (
                        0.0 if hop_count == 1
                        else (role - 1) / (hop_count - 1)
                    ),
                    "physical_position": int(
                        visual["visual_presentation_position"]
                    ),
                    "outcome_group": pair["group"],
                    "variant": variant,
                    "correct": bool(result["correct"]),
                    "layer": int(layer),
                    "relative_layer_depth": (
                        0.0 if n_layers == 1
                        else layer / (n_layers - 1)
                    ),
                    "fact_direction": labels[f"fact{role}"],
                    "prefix_state": labels[f"prefix{role}"],
                    "suffix_state": labels[f"suffix{role}"],
                    "answer": labels["answer"],
                    "input_tokens": token_count,
                    "model": args.model,
                    "model_label": model_label,
                }

                for k in range(1, hop_count + 1):
                    common[f"state{k}_x"] = labels[f"state{k}_x"]
                    common[f"state{k}_y"] = labels[f"state{k}_y"]
                    common[f"state{k}_coord"] = labels[f"state{k}_coord"]

                records.append({
                    **common,
                    "site": "target_boundary",
                    "feature": states[layer]["logical"][role].to(torch.float16),
                })
                records.append({
                    **common,
                    "site": "final_prompt",
                    "feature": states[layer]["final"].to(torch.float16),
                })

        del baseline_states, visual_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    payload = {
        "model": args.model,
        "model_label": model_label,
        "n_layers": n_layers,
        "layers": layers_to_keep,
        "roles": roles,
        "records": records,
        "notes": (
            "Arbitrary-hop paired Text/Vision state extraction. "
            "stateK_x/y are exact cumulative states s_K=f_1+...+f_K."
        ),
    }
    torch.save(payload, args.out)
    print(f"Saved {len(records)} state records to {args.out}")


if __name__ == "__main__":
    main()
