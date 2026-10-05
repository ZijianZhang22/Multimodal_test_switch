#!/usr/bin/env python3
"""
Run the switching-cost pilot with Qwen3-VL-8B-Instruct.
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


def build_messages(example, data_dir):
    content = [
        {
            "type": "text",
            "text": (
                "You will receive four pieces of evidence in order. "
                "Some evidence is written as text and some is shown as a diagram. "
                "Each diagram shows only a spatial relation between two labeled nodes. "
                "Use all four facts to answer the final question.\n\n"
                "Answer with exactly ONE label from: "
                "NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, "
                "SOUTHEAST, SOUTHWEST, SAME."
            ),
        }
    ]

    for fact in example["facts"]:
        step = fact["step"]

        if fact["source"] == "T":
            content.append({
                "type": "text",
                "text": f"\nEvidence {step}: {fact['text']}",
            })
        else:
            image_path = (data_dir / fact["image"]).resolve()

            content.append({
                "type": "text",
                "text": f"\nEvidence {step}:",
            })
            content.append({
                "type": "image",
                "image": str(image_path),
            })

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
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Useful for a quick smoke test, e.g. --limit 30",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    data_dir = args.data.parent
    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    examples = list(load_jsonl(args.data))
    if args.limit is not None:
        examples = examples[:args.limit]

    with args.out.open("w", encoding="utf-8") as output_file:
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
                "example_id": example["example_id"],
                "problem_id": example["problem_id"],
                "schedule": example["schedule"],
                "switch_count": example["switch_count"],
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

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
