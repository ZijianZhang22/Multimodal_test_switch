#!/usr/bin/env python3
"""
Round 5C: fact-level causal patching with matched controls.

Main intervention:
  In a single-Visual FAILURE run, patch the residual-stream state at the END OF
  THE TARGET FACT using the matched all-Text baseline's state for the SAME
  logical fact, then continue generation.

Controls:
  - matched_fact: same problem, same order, same logical fact (main test)
  - same_example_other_fact: same Text baseline, wrong logical fact
  - unrelated_same_answer: different problem, same answer and same order_id
  - unrelated_other_answer: different problem, different answer, same order_id

The patch is applied only during the prefill pass and only at one evidence
boundary token. This is substantially more local than the previous final-prompt
state patch.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mechanism_utils import (
    get_evidence_boundary_positions,
    get_text_layers,
    load_jsonl,
    parse_layer_spec,
    prepare_inputs,
)
from run_qwen3vl import parse_label


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


def decoder_output_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def collect_boundary_states(
    model,
    processor,
    example,
    data_dir,
    layers_to_keep,
    logical_roles,
):
    inputs = prepare_inputs(processor, example, data_dir, model.device)
    boundaries = get_evidence_boundary_positions(inputs, processor, example)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    states = {}
    for layer in layers_to_keep:
        h = outputs.hidden_states[layer + 1][0]
        states[layer] = {
            role: h[boundaries["logical"][role]].detach().float().cpu()
            for role in logical_roles
        }

    del outputs, inputs
    return states


def generate_with_boundary_patch(
    model,
    processor,
    layers,
    visual_inputs,
    target_token_pos,
    donor_state,
    layer_idx,
    answer,
    max_new_tokens,
):
    prompt_len = int(visual_inputs["input_ids"].shape[-1])
    layer = layers[layer_idx]

    def hook_fn(module, module_inputs, output):
        hidden = decoder_output_hidden(output)
        if (
            torch.is_tensor(hidden)
            and hidden.ndim == 3
            and hidden.shape[1] == prompt_len
        ):
            patched = hidden.clone()
            patched[:, target_token_pos, :] = donor_state.to(
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

    new_ids = generated[:, prompt_len:]
    raw = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    pred = parse_label(raw)
    return raw, pred, bool(pred == answer)


def build_failure_pairs(dataset_rows, result_rows, roles):
    result_by_id = {x["example_id"]: x for x in result_rows}
    baseline_by_key = {
        (x["problem_id"], x["order_id"]): x
        for x in dataset_rows
        if x.get("condition") == "text_baseline"
    }

    pairs = []
    for ex in dataset_rows:
        if ex.get("condition") != "single_visual":
            continue
        role = int(ex["visual_logical_step"])
        if role not in roles:
            continue

        vis_res = result_by_id.get(ex["example_id"])
        base = baseline_by_key.get((ex["problem_id"], ex["order_id"]))
        if vis_res is None or base is None:
            continue
        base_res = result_by_id.get(base["example_id"])
        if base_res is None:
            continue

        if bool(base_res["correct"]) and not bool(vis_res["correct"]):
            pairs.append({
                "role": role,
                "visual": ex,
                "baseline": base,
                "visual_result": vis_res,
                "baseline_result": base_res,
            })

    pairs.sort(
        key=lambda x: (
            x["role"],
            x["visual"]["problem_id"],
            x["visual"]["order_id"],
        )
    )
    return pairs


def index_correct_baselines(dataset_rows, result_rows):
    result_by_id = {x["example_id"]: x for x in result_rows}
    baselines = []
    for ex in dataset_rows:
        if ex.get("condition") != "text_baseline":
            continue
        res = result_by_id.get(ex["example_id"])
        if res is not None and bool(res["correct"]):
            baselines.append(ex)
    baselines.sort(key=lambda x: (x["order_id"], x["problem_id"]))
    return baselines


def choose_unrelated_donor(baselines, target, same_answer):
    candidates = []
    for ex in baselines:
        if ex["problem_id"] == target["problem_id"]:
            continue
        if ex["order_id"] != target["order_id"]:
            continue
        answer_match = ex["answer"] == target["answer"]
        if answer_match == same_answer:
            candidates.append(ex)
    return candidates[0] if candidates else None


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
        default=Path("results_fact_patching/fact_level_patching.jsonl"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--roles",
        default="1,4",
        help="Default early-vs-late comparison.",
    )
    parser.add_argument(
        "--max_failures_per_role",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--layers",
        default="20:32:2",
        help='Dense causal window. Use "20:32" for every layer.',
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    args.results = choose_results_path(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    roles = parse_roles(args.roles)

    dataset_rows = load_jsonl(args.data)
    result_rows = load_jsonl(args.results)
    failure_pairs = build_failure_pairs(dataset_rows, result_rows, roles)
    correct_baselines = index_correct_baselines(dataset_rows, result_rows)

    selected = []
    for role in roles:
        bucket = [x for x in failure_pairs if x["role"] == role]
        picked = bucket[:args.max_failures_per_role]
        print(
            f"role={role}: failures available={len(bucket)} "
            f"selected={len(picked)}"
        )
        selected.extend(picked)

    if not selected:
        raise RuntimeError("No eligible failure pairs found.")

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
    patch_layers = parse_layer_spec(args.layers, n_layers)
    print("Decoder path:", layer_path)
    print("Patch layers:", patch_layers)

    data_dir = args.data.parent

    with args.out.open("w", encoding="utf-8") as out_file:
        for pair in tqdm(selected, desc="fact-level patching"):
            role = int(pair["role"])
            visual = pair["visual"]
            baseline = pair["baseline"]
            other_role = 2 if role == 1 else role - 1

            same_answer_donor = choose_unrelated_donor(
                correct_baselines,
                baseline,
                same_answer=True,
            )
            other_answer_donor = choose_unrelated_donor(
                correct_baselines,
                baseline,
                same_answer=False,
            )

            needed_roles = sorted({role, other_role})
            matched_states = collect_boundary_states(
                model,
                processor,
                baseline,
                data_dir,
                patch_layers,
                needed_roles,
            )

            same_answer_states = None
            if same_answer_donor is not None:
                same_answer_states = collect_boundary_states(
                    model,
                    processor,
                    same_answer_donor,
                    data_dir,
                    patch_layers,
                    [role],
                )

            other_answer_states = None
            if other_answer_donor is not None:
                other_answer_states = collect_boundary_states(
                    model,
                    processor,
                    other_answer_donor,
                    data_dir,
                    patch_layers,
                    [role],
                )

            visual_inputs = prepare_inputs(
                processor,
                visual,
                data_dir,
                model.device,
            )
            visual_bounds = get_evidence_boundary_positions(
                visual_inputs,
                processor,
                visual,
            )
            target_token_pos = visual_bounds["logical"][role]

            for layer_idx in patch_layers:
                controls = {
                    "matched_fact": (
                        matched_states[layer_idx][role],
                        baseline,
                        role,
                    ),
                    "same_example_other_fact": (
                        matched_states[layer_idx][other_role],
                        baseline,
                        other_role,
                    ),
                }

                if same_answer_states is not None:
                    controls["unrelated_same_answer"] = (
                        same_answer_states[layer_idx][role],
                        same_answer_donor,
                        role,
                    )

                if other_answer_states is not None:
                    controls["unrelated_other_answer"] = (
                        other_answer_states[layer_idx][role],
                        other_answer_donor,
                        role,
                    )

                for control_name, (
                    donor_state,
                    donor_example,
                    donor_role,
                ) in controls.items():
                    raw, pred, correct = generate_with_boundary_patch(
                        model=model,
                        processor=processor,
                        layers=layers,
                        visual_inputs=visual_inputs,
                        target_token_pos=target_token_pos,
                        donor_state=donor_state,
                        layer_idx=layer_idx,
                        answer=visual["answer"],
                        max_new_tokens=args.max_new_tokens,
                    )

                    record = {
                        "problem_id": visual["problem_id"],
                        "order_id": visual["order_id"],
                        "logical_role": role,
                        "physical_position": int(
                            visual["visual_presentation_position"]
                        ),
                        "visual_example_id": visual["example_id"],
                        "baseline_example_id": baseline["example_id"],
                        "original_visual_prediction": pair[
                            "visual_result"
                        ].get("prediction"),
                        "answer": visual["answer"],
                        "layer": int(layer_idx),
                        "target_token_pos": int(target_token_pos),
                        "control": control_name,
                        "donor_problem_id": donor_example["problem_id"],
                        "donor_example_id": donor_example["example_id"],
                        "donor_logical_role": int(donor_role),
                        "donor_answer": donor_example["answer"],
                        "patched_prediction": pred,
                        "patched_correct": correct,
                        "patched_raw_output": raw,
                    }
                    out_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    out_file.flush()

            del matched_states, visual_inputs
            if same_answer_states is not None:
                del same_answer_states
            if other_answer_states is not None:
                del other_answer_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
