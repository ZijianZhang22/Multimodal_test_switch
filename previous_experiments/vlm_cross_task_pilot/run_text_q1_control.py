#!/usr/bin/env python3
"""Add text-Q1 + math-Q2 generations using the existing paired MATH-500 images.

Run from vlm_cross_task_pilot; reuses run.inputs_for() for identical image
preprocessing and chat formatting. Stores output separately, resumes safely.
"""
import argparse
import json
import random
from pathlib import Path

from run import inputs_for


def make_prompt(q2):
    return (
        "Answer both INDEPENDENT questions, in the specified order.\n"
        "Question 1 (use ONLY text; do not use the image): What is 17 + 23?\n"
        f"Question 2 (use ONLY Question 2 text): {q2}\n"
        "Reason step by step as needed. End with <q1>ANSWER</q1> <q2>FINAL_ANSWER</q2>."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="pilot_math20/data/pairs.jsonl")
    parser.add_argument("--out", default="pilot_math20/results/text_q1_predictions.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--variant", default="conflicting")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(s) for s in Path(args.pairs).read_text(encoding="utf-8").splitlines() if s.strip()]
    rows = [r for r in rows if args.variant in r["images"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["id"], rec["variant"], rec["condition"], rec["rep"]))

    todo = [r for r in rows if (r["id"], args.variant, "joint_text", 0) not in done]
    print(f"Total tasks {len(rows)} already done {len(rows) - len(todo)}", flush=True)
    if not todo:
        print("COMPLETE:", out, flush=True)
        return

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    device = next(model.parameters()).device

    for idx, row in enumerate(rows, 1):
        key = (row["id"], args.variant, "joint_text", 0)
        if key in done:
            continue
        prompt = make_prompt(row["q2_question"])
        inputs = inputs_for(processor, row["images"][args.variant], prompt, device)
        sampling = args.temperature > 0
        with torch.inference_mode():
            sequence = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=sampling,
                **({"temperature": args.temperature, "top_p": args.top_p} if sampling else {}),
            )
        generated = sequence[0][inputs["input_ids"].shape[1]:]
        prediction = processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        record = {
            "id": row["id"], "variant": args.variant, "condition": "joint_text", "rep": 0,
            "q1_gold": "40", "q1_visual_gold": row["q1_label"],
            "q2_gold": row["q2_gold"], "q2_id": row["q2_id"],
            "q2_benchmark": row["q2_benchmark"], "q2_level": row.get("q2_level"),
            "pred": prediction, "num_new_tokens": len(generated),
            "hit_max_token": bool(len(generated) >= args.max_new_tokens),
            "model": args.model, "temperature": args.temperature, "top_p": args.top_p,
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{idx}/{len(rows)} {row['id']} {args.variant} joint_text rep=0 tokens={len(generated)}", flush=True)
    print("COMPLETE:", out, flush=True)


if __name__ == "__main__":
    main()
