#!/usr/bin/env python3
"""Qwen3-VL paired MME-CoT inference, continuing in chunks until natural EOS.

Important: --chunk-tokens is NOT a limit on the total response length.
When a chunk fills without EOS we append its token IDs and generate again.
Generation ends only at the model's EOS, an exception, or user interruption.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from prompts import build_prompt


def model_eos_ids(model, tokenizer):
    values = []
    for item in (getattr(model.generation_config, "eos_token_id", None),
                 getattr(model.config, "eos_token_id", None),
                 getattr(tokenizer, "eos_token_id", None)):
        if isinstance(item, (list, tuple, set)):
            values.extend(int(x) for x in item if x is not None)
        elif item is not None:
            values.append(int(item))
    # The Qwen chat turn-end token must also terminate a completed answer.
    turn_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(turn_end, int) and turn_end >= 0:
        values.append(turn_end)
    return sorted(set(values))


def prepare_inputs(processor, row, prompt, device):
    content = []
    for image_path in row["image_paths"]:
        with Image.open(image_path) as im:
            content.append({"type": "image", "image": im.convert("RGB").copy()})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def generate_until_eos(model, processor, inputs, chunk_tokens, temperature, top_p):
    """Generate without a user-imposed total output token cap.

    Recomputes the full prompt every chunk (simple, robust resume strategy).
    Subject to finite hardware/model context; those limits are NOT silent truncation.
    """
    tokenizer = processor.tokenizer
    eos = set(model_eos_ids(model, tokenizer))
    if not eos:
        raise RuntimeError("Cannot resolve EOS token IDs; refusing unbounded generation.")
    all_generated = []
    chunks = 0
    prompt_tokens = int(inputs["input_ids"].shape[1])
    work = dict(inputs)
    started = time.monotonic()
    with torch.inference_mode():
        while True:
            chunks += 1
            generation_kwargs = {
                "max_new_tokens": chunk_tokens,  # internal allocation, not total cap
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": list(eos),
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output = model.generate(**work, **generation_kwargs)
            past_length = int(work["input_ids"].shape[1])
            new_ids = output[0, past_length:].detach().cpu().tolist()
            if not new_ids:
                raise RuntimeError("generate() returned 0 new tokens without EOS")
            all_generated.extend(int(t) for t in new_ids)
            if any(token_id in eos for token_id in new_ids):
                return {
                    "raw_output": tokenizer.decode(
                        all_generated, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    ),
                    "text_output": tokenizer.decode(
                        all_generated, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ),
                    "generation_tokens": len(all_generated),
                    "prompt_tokens": prompt_tokens,
                    "chunks": chunks,
                    "seconds": round(time.monotonic() - started, 2),
                    "stop_reason": "model_eos",
                }
            print(
                f"    chunk {chunks} complete: {len(all_generated)} output tokens, "
                "no EOS — continuing (not truncated)",
                flush=True,
            )
            # Re-use the unchanged visual tensors and append exact generated IDs.
            work["input_ids"] = output
            work["attention_mask"] = torch.ones_like(output, dtype=torch.long)
            del output
            # There is deliberately NO overall max token count here.


def thought_stats(raw, tokenizer):
    """Thinking tokens measured by re-tokenizing the model's text inside think tags."""
    if "<think>" not in raw or "</think>" not in raw:
        return None, False
    thinking = raw.split("<think>", 1)[1].split("</think>", 1)[0]
    return len(tokenizer.encode(thinking, add_special_tokens=False)), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot_data/pairs.jsonl")
    ap.add_argument("--out", default="pilot_results/predictions.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--conditions", default="baseline,plus_arithmetic")
    ap.add_argument("--chunk-tokens", type=int, default=1024,
                    help="Per-call chunk size. NOT a total output limit; auto-continues.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.chunk_tokens < 1:
        ap.error("--chunk-tokens must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("Qwen3-VL-8B-Thinking requires a CUDA GPU. Please run on RunPod.")

    examples = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if not examples:
        raise SystemExit("No examples. Run prepare.py first.")
    conditions = [s.strip() for s in args.conditions.split(",") if s.strip()]
    if any(c not in ("baseline", "plus_arithmetic") for c in conditions):
        ap.error("Only baseline and plus_arithmetic are supported")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("stop_reason") == "model_eos":
                done.add((str(rec["id"]), rec["condition"]))
    print("Loading model:", args.model, flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    count = len(examples) * len(conditions)
    print(f"{count} cases, {len(done)} finished; no overall output token cap.", flush=True)
    index = 0
    for row in examples:
        for condition in conditions:
            index += 1
            key = (str(row["id"]), condition)
            if key in done:
                print(f"[{index}/{count}] SKIP completed {key}", flush=True)
                continue
            prompt = build_prompt(row, condition)
            print(f"[{index}/{count}] {row['category']}/{row['id']} {condition}", flush=True)
            per_pair_seed = int.from_bytes(
                hashlib.sha256(f"{args.seed}/{row['id']}".encode()).digest()[:4], "big"
            )
            torch.manual_seed(per_pair_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(per_pair_seed)
            record = {
                "id": str(row["id"]),
                "category": row["category"],
                "subcategory": row.get("subcategory", ""),
                "question_type": row["question_type"],
                "gold_answer": row["answer"],
                "question": row["question"],
                "options": row.get("options", []),
                "condition": condition,
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": per_pair_seed,
                "prompt": prompt,
                "image_paths": row["image_paths"],
            }
            try:
                inputs = prepare_inputs(processor, row, prompt, model.device)
                record.update(generate_until_eos(
                    model, processor, inputs,
                    args.chunk_tokens, args.temperature, args.top_p
                ))
                thought_length, present = thought_stats(record["raw_output"], processor.tokenizer)
                record["thinking_tokens"] = thought_length
                record["has_thinking_tags"] = present
            except KeyboardInterrupt:
                print("Interrupted. Previously finished examples are saved.", flush=True)
                raise
            except Exception as e:
                record["stop_reason"] = "generation_error"
                record["error"] = f"{type(e).__name__}: {e}"
                print("ERROR:", record["error"], flush=True)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "    result:", record["stop_reason"],
                "generated_tokens:", record.get("generation_tokens"),
                "thinking_tokens:", record.get("thinking_tokens"),
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("Inference saved:", output_path)


if __name__ == "__main__":
    main()
