#!/usr/bin/env python3
"""Zero-shot PaliGemma2 baseline before expensive language-steering A/B sweeps.

Identical to run_ab.py with --shots 0, source=none, and parity common.build_prompt.
Saves every generation incrementally; resumes safely after interruption.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paligemma2_cross_lingual_steering" / "parity"))
from common import MODEL_ID, build_prompt, jsonl_read, score  # noqa: E402
from checkpoint_eval import evaluate_resumable  # noqa: E402

HEADERS = ["id", "task", "split", "lang", "source", "mask", "layer", "alpha",
           "response", "normalized", "valid", "correct", "tokens_edited"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Baseline CSV path")
    p.add_argument("--lang", default="zh", choices=["zh", "en", "sw"])
    p.add_argument("--limit-test", type=int, default=0, help="Per task; 0=all 600")
    p.add_argument("--max-new-tokens", type=int, default=20)
    args = p.parse_args()
    if args.limit_test < 0 or args.max_new_tokens < 1:
        p.error("--limit-test must be >= 0 and --max-new-tokens >= 1")
    data = args.data.expanduser().resolve()
    rows = [s for s in jsonl_read(data / "visual_eval.jsonl")
            if s["split"] == "test"]
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    samples = [s for task in sorted(by_task)
               for s in (by_task[task][:args.limit_test]
                         if args.limit_test else by_task[task])]
    if not samples:
        raise RuntimeError("No test samples found")
    print(f"Zero-shot {args.lang} baseline: {len(samples)} held-out questions.",
          flush=True)

    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).eval().to("cuda")

    def run_one(sample, missing):
        assert missing == [("none", "baseline", -1, 0.0)]
        with Image.open(data / sample["image"]) as im:
            image = im.copy()
        # NO English demonstrations; exact same prompt as parity baseline.
        prompt = build_prompt(sample, args.lang, "native")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = inputs.to(torch.bfloat16).to("cuda")
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        answer = processor.decode(
            out[0][input_length:], skip_special_tokens=True).strip()
        scored = score(answer, sample["answer"], sample["choices"],
                       sample["choices_i18n"][args.lang], args.lang)
        yield dict(id=sample["id"], task=sample["task"], split="test",
                   lang=args.lang, source="none", mask="baseline",
                   layer=-1, alpha=0.0, response=answer, tokens_edited=0,
                   **scored)

    results = evaluate_resumable(
        samples, [("none", "baseline", -1, 0.0)],
        args.out.expanduser().resolve(), HEADERS,
        run_one, f"zeroshot-{args.lang}")
    groups = defaultdict(list)
    for row in results:
        groups[row["task"]].append(row)
    print("\nZero-shot baseline results:", flush=True)
    for task, task_rows in sorted(groups.items()):
        n = len(task_rows)
        accuracy = sum(int(r["correct"]) for r in task_rows) / n
        validity = sum(int(r["valid"]) for r in task_rows) / n
        empty = sum(not r["response"].strip() for r in task_rows)
        print(f"  {task:10s}: n={n:3d} acc={accuracy:.3%} "
              f"valid={validity:.3%} empty={empty}", flush=True)
    n = len(results)
    acc = sum(int(r["correct"]) for r in results) / n
    print(f"  TOTAL: n={n}, accuracy={acc:.3%} "
          f"({sum(int(r['correct']) for r in results)} correct)", flush=True)


if __name__ == "__main__":
    main()
