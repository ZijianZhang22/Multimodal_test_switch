#!/usr/bin/env python3
"""
Run Qwen3-VL on variable-hop multimodal reasoning datasets.

Compatible with:
- legacy 4-hop schedule data
- Round-3 role/position data
- single-fact perception controls
- depth-scaling data with arbitrary hop count
- Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct

The runner preserves experiment metadata in the result JSONL and supports
safe resume, which is especially useful for 32B runs.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


LABELS = [
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
    "NORTH", "SOUTH", "EAST", "WEST", "SAME",
]


def parse_label(text):
    normalized = text.upper().strip()

    # Short-CoT mode is instructed to end with "FINAL: <LABEL>".
    # Prefer that explicit final answer so direction words mentioned in the
    # reasoning do not get mistaken for the prediction.
    final_match = re.search(
        r"FINAL\s*:\s*(NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|NORTH|SOUTH|EAST|WEST|SAME)\b",
        normalized,
    )
    if final_match:
        return final_match.group(1)

    # Backward-compatible direct-answer parsing.
    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", normalized):
            return label
    return None


def add_fact_content(content, fact, data_dir, display_index):
    if fact["source"] == "T":
        content.append({
            "type": "text",
            "text": f"\nEvidence {display_index}: {fact['text']}",
        })
    else:
        image_path = (data_dir / fact["image"]).resolve()
        content.append({
            "type": "text",
            "text": f"\nEvidence {display_index}:",
        })
        content.append({
            "type": "image",
            "image": str(image_path),
        })


def build_messages(example, data_dir, reasoning_mode="direct"):
    task_type = example.get("task_type", "chain")

    if task_type == "perception":
        content = [{
            "type": "text",
            "text": (
                "You will receive one spatial relation as either text or a diagram. "
                "Read the relation and answer the question.\n\n"
                "Answer with exactly ONE label from: "
                "NORTH, SOUTH, EAST, WEST."
            ),
        }]

        fact = example["facts"][0]
        add_fact_content(content, fact, data_dir, display_index=1)
        content.append({
            "type": "text",
            "text": f"\nQuestion: {example['question']}",
        })
        return [{"role": "user", "content": content}]

    n_facts = len(example["facts"])

    if reasoning_mode == "direct":
        answer_instruction = (
            "Answer with exactly ONE label from: "
            "NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, "
            "SOUTHEAST, SOUTHWEST, SAME."
        )
    else:
        answer_instruction = (
            "Use a very short scratchpad to solve the chain. "
            "Keep the reasoning to at most TWO short lines and under about 40 words. "
            "Then end with exactly: FINAL: <LABEL>, where <LABEL> is one of "
            "NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, "
            "SOUTHEAST, SOUTHWEST, SAME."
        )

    content = [{
        "type": "text",
        "text": (
            f"You will receive {n_facts} pieces of evidence. "
            "Some evidence is written as text and some is shown as a diagram. "
            "Each diagram shows only a spatial relation between two labeled nodes. "
            "The evidence items may not be presented in reasoning-chain order. "
            "Use the entity labels and all relevant facts to solve the final question.\n\n"
            + answer_instruction
        ),
    }]

    for display_index, fact in enumerate(example["facts"], start=1):
        # Use physical presentation index, not logical-step ID.
        # This prevents shuffled examples from leaking the latent chain order.
        add_fact_content(content, fact, data_dir, display_index=display_index)

    content.append({
        "type": "text",
        "text": f"\nQuestion: {example['question']}",
    })

    return [{"role": "user", "content": content}]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def copy_metadata(example):
    skip = {"facts", "question", "answer"}
    return {k: v for k, v in example.items() if k not in skip}


def existing_example_ids(path):
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("example_id") is not None:
                ids.add(row["example_id"])
    return ids


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/examples.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/qwen3vl_results.jsonl"),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help=(
            "Examples: Qwen/Qwen3-VL-8B-Instruct or "
            "Qwen/Qwen3-VL-32B-Instruct."
        ),
    )
    parser.add_argument(
        "--model_label",
        default=None,
        help="Short label stored in results, e.g. qwen3vl8b or qwen3vl32b.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Passed to from_pretrained; "auto" is recommended for 32B.',
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Optional comma-separated condition filter, e.g. "
            "text_baseline or text_baseline,single_visual."
        ),
    )
    parser.add_argument(
        "--hops",
        default=None,
        help="Optional comma-separated hop-count filter, e.g. 4,6,8.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Useful for a quick smoke test after filtering.",
    )
    parser.add_argument(
        "--reasoning_mode",
        choices=["direct", "short_cot"],
        default="direct",
        help=(
            "direct: output one label only; short_cot: allow at most two short "
            "reasoning lines and require FINAL: <LABEL>."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            "Generation cap. Defaults to 16 for direct mode and 48 for "
            "short_cot mode."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing output and skip example_ids already present. "
            "Recommended for long 32B runs."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data.parent
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generation_max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (48 if args.reasoning_mode == "short_cot" else 16)
    )
    print(
        "Reasoning mode:",
        args.reasoning_mode,
        "| max_new_tokens:",
        generation_max_new_tokens,
    )

    print("Loading processor:", args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading model:", args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    examples = list(load_jsonl(args.data))

    if args.conditions:
        wanted_conditions = {
            x.strip()
            for x in args.conditions.split(",")
            if x.strip()
        }
        examples = [
            ex for ex in examples
            if ex.get("condition") in wanted_conditions
        ]
        print("Condition filter:", sorted(wanted_conditions))

    if args.hops:
        wanted_hops = {
            int(x.strip())
            for x in args.hops.split(",")
            if x.strip()
        }
        examples = [
            ex for ex in examples
            if int(ex.get("hop_count", len(ex.get("facts", []))))
            in wanted_hops
        ]
        print("Hop filter:", sorted(wanted_hops))

    if args.limit is not None:
        examples = examples[:args.limit]

    done_ids = existing_example_ids(args.out) if args.resume else set()
    if done_ids:
        before = len(examples)
        examples = [
            ex for ex in examples
            if ex.get("example_id") not in done_ids
        ]
        print(
            f"Resume: found {len(done_ids)} existing example_ids; "
            f"{before} -> {len(examples)} examples remaining."
        )

    mode = "a" if args.resume else "w"
    model_label = args.model_label or args.model.split("/")[-1]

    with args.out.open(mode, encoding="utf-8") as output_file:
        for example in tqdm(examples):
            messages = build_messages(
                example,
                data_dir,
                reasoning_mode=args.reasoning_mode,
            )

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs.pop("token_type_ids", None)
            input_token_count = int(inputs["input_ids"].shape[-1])

            n_text_tokens = None
            n_visual_tokens = None
            if "mm_token_type_ids" in inputs:
                mm = inputs["mm_token_type_ids"]
                n_text_tokens = int((mm == 0).sum().item())
                n_visual_tokens = int((mm == 1).sum().item())

            inputs = inputs.to(model.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=generation_max_new_tokens,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = time.perf_counter() - start

            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = generated[:, prompt_len:]

            output_text = processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            prediction = parse_label(output_text)

            record = {
                **copy_metadata(example),
                "model": args.model,
                "model_label": model_label,
                "answer": example["answer"],
                "prediction": prediction,
                "correct": bool(prediction == example["answer"]),
                "raw_output": output_text,
                "reasoning_mode": args.reasoning_mode,
                "generation_max_new_tokens": generation_max_new_tokens,
                "input_tokens": input_token_count,
                "text_tokens": n_text_tokens,
                "visual_tokens": n_visual_tokens,
                "output_tokens": int(new_ids.shape[-1]),
                "latency_sec": latency,
            }

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
