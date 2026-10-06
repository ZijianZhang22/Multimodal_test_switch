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


def build_messages(example, data_dir):
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
    content = [{
        "type": "text",
        "text": (
            f"You will receive {n_facts} pieces of evidence. "
            "Some evidence is written as text and some is shown as a diagram. "
            "Each diagram shows only a spatial relation between two labeled nodes. "
            "The evidence items may not be presented in reasoning-chain order. "
            "Use the entity labels and all relevant facts to solve the final question.\n\n"
            "Answer with exactly ONE label from: "
            "NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, "
            "SOUTHEAST, SOUTHWEST, SAME."
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
        "--limit",
        type=int,
        default=None,
        help="Useful for a quick smoke test.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
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
            messages = build_messages(example, data_dir)

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
                    max_new_tokens=args.max_new_tokens,
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
