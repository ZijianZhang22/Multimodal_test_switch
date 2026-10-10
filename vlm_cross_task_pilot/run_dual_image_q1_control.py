#!/usr/bin/env python3
"""Paired text-Q1 vs second-image-Q1 control, WITH the original Idis image retained.

Both conditions use the same Idis image and exact Q2 from the existing pairs.jsonl:
  joint_text       : [Idis image] + written "17 + 23?" + original MATH-500 Q2
  joint_image_q1   : [Idis image, arithmetic PNG] + visual Q1 + identical Q2

No benchmark resampling, no modification of the original Idis image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from run_text_q1_control import make_prompt as original_text_q1_prompt

ROOT = Path(__file__).resolve().parent
DEFAULT_Q1_IMAGE = ROOT / "assets" / "q1_text_17_plus_23.png"
DEFAULT_PAIRS = ROOT / "pilot_math20" / "data" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "pilot_math20_dual_image_q1" / "results" / "predictions.jsonl"
CONDITIONS = ("joint_text", "joint_image_q1")
OLD_Q1_LINE = "Question 1 (use ONLY text; do not use the image): What is 17 + 23?"
NEW_Q1_LINE = ("Question 1 (use ONLY the SECOND image, not the FIRST image): "
               "Read and solve the arithmetic expression in the second image.")


def make_prompt(q2: str, condition: str) -> str:
    """Preserve the previous text-Q1 control prompt exactly."""
    prompt = original_text_q1_prompt(q2)
    if condition == "joint_text":
        return prompt
    if condition == "joint_image_q1":
        if prompt.count(OLD_Q1_LINE) != 1:
            raise ValueError("The upstream text-Q1 prompt changed; update OLD_Q1_LINE before comparing.")
        return prompt.replace(OLD_Q1_LINE, NEW_Q1_LINE)
    raise ValueError(f"Unknown condition: {condition}")


def load_pairs(path: Path, variant: str, limit: int = 0) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pair manifest: {path}. Generate it on RunPod first.")
    records = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    seen = set()
    selected = []
    for row in records:
        if variant not in row.get("images", {}):
            continue
        assert row.get("q2_question") and row.get("q2_gold") is not None, row
        key = (row["id"], row["q2_id"])
        if key in seen:
            raise ValueError(f"Duplicate image/text pair in manifest: {key}")
        seen.add(key)
        selected.append(row)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No samples containing image variant {variant!r} found in {path}")
    return selected


def make_messages(idis_image: Path, equation_image: Path, prompt: str, condition: str):
    """Order is ALWAYS Idis (image 1), arithmetic image (image 2), prompt."""
    from PIL import Image
    original = Image.open(idis_image).convert("RGB").resize((512, 512))
    content = [{"type": "image", "image": original}]
    if condition == "joint_image_q1":
        # Keep the arithmetic image crisp; Qwen processor controls patch resizing.
        arithmetic = Image.open(equation_image).convert("RGB")
        content.append({"type": "image", "image": arithmetic})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def encode(processor, messages, device):
    from qwen_vl_utils import process_vision_info
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return processor(text=[chat], images=images, videos=videos,
                     padding=True, return_tensors="pt").to(device)


def key_of(r: dict) -> tuple:
    return (r["id"], r["q2_id"], r["variant"], r["condition"], int(r["rep"]))


def existing_records(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            key = key_of(rec)
            if key in done:
                raise ValueError(f"Repeated prediction in {path}: {key}")
            done[key] = rec
    return done


def import_previous_text(args, rows: list[dict], done: dict) -> int:
    """Optionally reuse the older single-Idis-image text-Q1 outputs."""
    previous = args.reuse_text_file
    if not previous:
        return 0
    previous = Path(previous).expanduser().resolve()
    if not previous.is_file():
        raise FileNotFoundError(f"Older text-Q1 predictions not found: {previous}")
    manifest = {(row["id"], row["q2_id"]): row for row in rows}
    imported = 0
    for line in previous.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        if old.get("condition") != "joint_text" or old.get("variant") != args.variant:
            continue
        pair_key = (old.get("id"), old.get("q2_id"))
        if pair_key not in manifest or int(old.get("rep", 0)) >= args.samples:
            continue
        row = manifest[pair_key]
        if str(old.get("q2_gold")) != str(row["q2_gold"]):
            raise ValueError(f"Q2 gold answer mismatch for {pair_key}")
        if old.get("model") != args.model:
            raise ValueError(f"Model mismatch for previous record {pair_key}")
        if old.get("temperature") != args.temperature or old.get("top_p") != args.top_p:
            raise ValueError(f"Sampling mismatch for previous record {pair_key}")
        rec = dict(old)
        rec["q1_gold"] = "40"
        rec["q2_question"] = row["q2_question"]
        rec["idis_image"] = row["images"][args.variant]
        rec["arithmetic_image"] = None
        rec["n_images"] = 1
        rec["prompt"] = make_prompt(row["q2_question"], "joint_text")
        rec["source"] = str(previous)
        rec["comparison_mode"] = "dual_image_q1"
        key = key_of(rec)
        if key not in done:
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[key] = rec
            imported += 1
    return imported


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--variant", default="conflicting", help="Idis variant in pairs.jsonl")
    ap.add_argument("--q1-image", type=Path, default=DEFAULT_Q1_IMAGE)
    ap.add_argument("--conditions", default="joint_text,joint_image_q1")
    ap.add_argument("--limit", type=int, default=0, help="0 means all existing pairs (20 recommended)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--reuse-text-file", type=Path, help="Optional old pilot_math20/results/text_q1_predictions.jsonl")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Validate input pairs/prompts without loading a GPU model")
    args = ap.parse_args()
    args.out = args.out.expanduser().resolve()
    args.q1_image = args.q1_image.expanduser().resolve()
    conditions = tuple(s.strip() for s in args.conditions.split(",") if s.strip())
    if not conditions or len(set(conditions)) != len(conditions) or any(x not in CONDITIONS for x in conditions):
        ap.error("--conditions must contain joint_text and/or joint_image_q1")
    if args.samples < 1 or args.limit < 0:
        ap.error("--samples must be >=1, --limit must be >=0")
    if args.score_only:
        from score_dual_image_q1_control import score
        score(args.out)
        return
    rows = load_pairs(args.pairs_file.expanduser().resolve(), args.variant, args.limit)
    if "joint_image_q1" in conditions and not args.q1_image.is_file():
        raise FileNotFoundError(f"Arithmetic image not found: {args.q1_image}")
    for row in rows:
        original = Path(row["images"][args.variant]).expanduser()
        if not original.is_file():
            raise FileNotFoundError(f"Idis image not found: {original} (copy/download the images on this pod)")
    print(f"Paired manifest: {len(rows)} fixed Q2 items, Idis variant={args.variant}")
    if args.dry_run:
        first = rows[0]
        print("Q2:", first["q2_id"], first["q2_question"][:160])
        for condition in conditions:
            prompt = make_prompt(first["q2_question"], condition)
            messages = make_messages(Path(first["images"][args.variant]), args.q1_image, prompt, condition)
            print(f"\n{condition}: image count={sum(p['type']=='image' for p in messages[0]['content'])}\n{prompt[:600]}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_records(args.out)
    imported = import_previous_text(args, rows, done)
    if imported:
        print(f"Reused {imported} existing text-Q1 predictions (no GPU calls for them).")
    tasks = []
    for row in rows:
        for rep in range(args.samples):
            for cond in conditions:
                task_key = (row["id"], row["q2_id"], args.variant, cond, rep)
                if task_key not in done:
                    tasks.append((row, cond, rep))
    print(f"Pending generations: {len(tasks)}", flush=True)
    if tasks:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU required for model generation; use --dry-run or --score-only for local checks")
        processor = AutoProcessor.from_pretrained(args.model)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto"
        ).eval()
        device = next(model.parameters()).device
        for idx, (row, condition, rep) in enumerate(tasks, 1):
            # Same initial random seed per paired condition; divergence after sampling is expected.
            stable = int(hashlib.sha256(f'{row["id"]}|{row["q2_id"]}|{rep}'.encode()).hexdigest()[:8], 16)
            seed = (args.seed + stable) % (2**31)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            prompt = make_prompt(row["q2_question"], condition)
            idis_image = Path(row["images"][args.variant]).expanduser()
            messages = make_messages(idis_image, args.q1_image, prompt, condition)
            inputs = encode(processor, messages, device)
            sampling = args.temperature > 0
            with torch.inference_mode():
                sequence = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=sampling,
                    **({"temperature": args.temperature, "top_p": args.top_p} if sampling else {}),
                )
            continuation = sequence[0][inputs["input_ids"].shape[1]:]
            prediction = processor.decode(continuation, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            record = {
                "id": row["id"], "q2_id": row["q2_id"], "variant": args.variant,
                "condition": condition, "rep": rep, "q1_gold": "40",
                "q2_question": row["q2_question"], "q2_gold": row["q2_gold"],
                "q2_benchmark": row["q2_benchmark"], "q2_level": row.get("q2_level"),
                "original_visual_label": row["q1_label"],
                "idis_image": str(idis_image),
                "arithmetic_image": str(args.q1_image) if condition == "joint_image_q1" else None,
                "n_images": 2 if condition == "joint_image_q1" else 1,
                "prompt": prompt,
                "pred": prediction,
                "num_new_tokens": int(len(continuation)),
                "hit_max_token": bool(len(continuation) >= args.max_new_tokens),
                "model": args.model, "temperature": args.temperature, "top_p": args.top_p,
                "seed": seed, "comparison_mode": "dual_image_q1",
            }
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(tasks)}] {row['id']} {condition} rep={rep} generated_tokens={len(continuation)}", flush=True)
    from score_dual_image_q1_control import score
    score(args.out)
    print("FINISHED:", args.out)


if __name__ == "__main__":
    main()
