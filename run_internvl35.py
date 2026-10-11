#!/usr/bin/env python3
"""
Run InternVL3.5 on the same spatial-reasoning JSONL used by the Qwen3-VL experiments.

Primary use: cross-family replication of the 4-hop logical-role effect with
OpenGVLab/InternVL3_5-14B using exactly the same latent problems and visual
diagrams.

The runner supports arbitrary Text/Vision schedules, but the recommended first
replication is TTTT vs V@1 vs V@4.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


LABELS = [
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
    "NORTH", "SOUTH", "EAST", "WEST", "SAME",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_label(text):
    normalized = text.upper().strip()
    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", normalized):
            return label
    return None


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_file, input_size=448):
    # Our synthetic diagrams are square and simple. A single 448x448 tile is
    # sufficient and avoids unnecessary dynamic tiling / extra visual tokens.
    image = Image.open(image_file).convert("RGB")
    return torch.stack([build_transform(input_size)(image)])


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def copy_metadata(example):
    skip = {"facts", "question", "answer"}
    return {k: v for k, v in example.items() if k not in skip}


def build_question(example):
    n_facts = len(example["facts"])
    parts = [
        (
            f"You will receive {n_facts} pieces of evidence. "
            "Some evidence is written as text and some is shown as a diagram. "
            "Each diagram shows only a spatial relation between two labeled nodes. "
            "The evidence items may not be presented in reasoning-chain order. "
            "Use the entity labels and all relevant facts to solve the final question.\n\n"
            "Answer with exactly ONE label from: "
            "NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, "
            "SOUTHEAST, SOUTHWEST, SAME."
        )
    ]

    image_paths = []
    for display_index, fact in enumerate(example["facts"], start=1):
        if fact["source"] == "T":
            parts.append(f"\nEvidence {display_index}: {fact['text']}")
        else:
            # InternVL's chat interface uses one <image> placeholder per image.
            parts.append(f"\nEvidence {display_index}: <image>")
            image_paths.append(fact["image"])

    parts.append(f"\nQuestion: {example['question']}")
    return "".join(parts), image_paths


def resolve_dtype(name):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_4hop_trace/depth_scaling_examples.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_internvl14b/internvl35_14b_4hop_direct.jsonl"),
    )
    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL3_5-14B",
    )
    parser.add_argument(
        "--model_label",
        default="internvl35_14b",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--device_map",
        default="none",
        choices=["none", "auto"],
        help=(
            "Use 'none' for a single sufficiently large GPU (recommended for 14B on 48GB). "
            "Use 'auto' only when Accelerate sharding is actually needed."
        ),
    )
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--visual_roles", default=None)
    parser.add_argument(
        "--schedules",
        default=None,
        help="Optional comma-separated exact modality schedules, e.g. TTTT,VTTT,TTTV.",
    )
    parser.add_argument("--hops", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Optional bitsandbytes fallback if the 16-bit model does not fit.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use FlashAttention if it is installed; off by default for portability.",
    )
    args = parser.parse_args()

    data_dir = args.data.parent
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )

    print("Loading model:", args.model)
    load_kwargs = dict(
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=args.use_flash_attn,
    )
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        if args.device_map == "none":
            load_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(args.model, **load_kwargs).eval()
    if args.device_map == "none" and not args.load_in_8bit:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the single-GPU InternVL run.")
        model = model.cuda()

    examples = list(load_jsonl(args.data))

    if args.conditions:
        wanted = {x.strip() for x in args.conditions.split(",") if x.strip()}
        examples = [ex for ex in examples if ex.get("condition") in wanted]
        print("Condition filter:", sorted(wanted))

    if args.hops:
        wanted_hops = {int(x.strip()) for x in args.hops.split(",") if x.strip()}
        examples = [
            ex for ex in examples
            if int(ex.get("hop_count", len(ex.get("facts", [])))) in wanted_hops
        ]
        print("Hop filter:", sorted(wanted_hops))

    if args.schedules:
        wanted_schedules = {
            x.strip().upper() for x in args.schedules.split(",") if x.strip()
        }
        def _schedule(ex):
            return ex.get("schedule") or ex.get("logical_schedule") or ex.get("presentation_schedule")
        examples = [ex for ex in examples if _schedule(ex) in wanted_schedules]
        print("Schedule filter:", sorted(wanted_schedules))

    if args.visual_roles:
        wanted_roles = {
            int(x.strip()) for x in args.visual_roles.split(",") if x.strip()
        }
        examples = [
            ex for ex in examples
            if ex.get("condition") != "single_visual"
            or int(ex.get("visual_logical_step")) in wanted_roles
        ]
        print("Visual-role filter:", sorted(wanted_roles))

    if args.limit is not None:
        examples = examples[:args.limit]
        print("Limit:", args.limit)

    done_ids = existing_example_ids(args.out) if args.resume else set()
    if done_ids:
        before = len(examples)
        examples = [ex for ex in examples if ex.get("example_id") not in done_ids]
        print(f"Resume: {before} -> {len(examples)} examples remaining.")

    mode = "a" if args.resume else "w"
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }

    with args.out.open(mode, encoding="utf-8") as output_file:
        for example in tqdm(examples):
            question, image_rels = build_question(example)

            pixel_values = None
            num_patches_list = None
            if image_rels:
                image_tensors = []
                num_patches_list = []
                for image_rel in image_rels:
                    image_path = (data_dir / image_rel).resolve()
                    pv = load_image(image_path)
                    image_tensors.append(pv)
                    num_patches_list.append(int(pv.shape[0]))
                pixel_values = torch.cat(image_tensors, dim=0).to(
                    dtype=resolve_dtype(args.dtype),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                chat_kwargs = dict(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    history=None,
                    return_history=False,
                )
                if num_patches_list is not None and len(num_patches_list) > 1:
                    chat_kwargs["num_patches_list"] = num_patches_list

                response = model.chat(**chat_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            if not isinstance(response, str):
                response = str(response)
            response = response.strip()
            prediction = parse_label(response)

            input_tokens = len(
                tokenizer(question, add_special_tokens=False).input_ids
            )
            output_tokens = len(
                tokenizer(response, add_special_tokens=False).input_ids
            )

            record = {
                **copy_metadata(example),
                "model": args.model,
                "model_label": args.model_label,
                "answer": example["answer"],
                "prediction": prediction,
                "correct": bool(prediction == example["answer"]),
                "raw_output": response,
                "reasoning_mode": "direct",
                "generation_max_new_tokens": args.max_new_tokens,
                "input_tokens": input_tokens,
                "text_tokens": None,
                "visual_tokens": None,
                "output_tokens": output_tokens,
                "latency_sec": latency,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
