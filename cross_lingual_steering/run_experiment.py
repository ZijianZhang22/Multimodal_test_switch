#!/usr/bin/env python3
"""Extract bilingual textual Mean Shift vectors and test cross-lingual VLM steering.

Original method inspiration: https://github.com/Woodygan/TextSteersVision
This adaptation extracts vectors from text-only forwards of the SAME VLM.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from model_utils import (DEFAULT_MODEL, get_layers, load_model, predict,
                         read_jsonl, canonical_answer, text_target_masks)


def int_list(s):
    return [int(v) for v in s.split(",") if v.strip()]


def float_list(s):
    return [float(v) for v in s.split(",") if v.strip()]


def extract(model, processor, data_dir, layers, vectors_path):
    decoder = get_layers(model, layers)
    tokenizer = processor.tokenizer
    results = {}
    for concept in ("counting", "spatial"):
        results[concept] = {str(l): {} for l in layers}
        for lang in ("en", "zh"):
            rows = read_jsonl(data_dir / "corpus" / f"{concept}_{lang}.jsonl")
            # Each bucket stores target sum/count and control sum/count.
            sums = {l: {bucket: {"t_sum": None, "c_sum": None, "nt": 0, "nc": 0}
                        for bucket in ("all", "a", "b")} for l in layers}
            masks = {"t": [], "c": [], "bucket": "a"}

            def add(bucket, layer, t, c):
                rec = sums[layer][bucket]
                for key, chunk in (("t_sum", t), ("c_sum", c)):
                    vec = chunk.sum(dim=0).float().cpu()
                    rec[key] = vec if rec[key] is None else rec[key] + vec
                rec["nt"] += t.shape[0]
                rec["nc"] += c.shape[0]

            def make_hook(layer):
                def hook(_module, _inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    # One text sample, no image, no padding.
                    if h.shape[0] != 1:
                        raise RuntimeError("Expected one text sentence per forward pass")
                    t = h[0, masks["t"], :].detach()
                    c = h[0, masks["c"], :].detach()
                    for bucket in ("all", masks["bucket"]):
                        add(bucket, layer, t, c)
                return hook

            handles = [decoder[l].register_forward_hook(make_hook(l)) for l in layers]
            try:
                for row in tqdm(rows, desc=f"Extract {concept}/{lang}"):
                    encoded, tpos, cpos = text_target_masks(
                        tokenizer, row["sentence"], row["target"])
                    masks["t"], masks["c"] = tpos, cpos
                    masks["bucket"] = "a" if row["id"] % 2 == 0 else "b"
                    inputs = encoded.to(model.device)
                    with torch.inference_mode():
                        model(**inputs, use_cache=False)
            finally:
                for h in handles:
                    h.remove()

            for l in layers:
                for bucket in ("all", "a", "b"):
                    s = sums[l][bucket]
                    if not s["nt"] or not s["nc"]:
                        raise RuntimeError(f"Empty token group: {concept}/{lang}/{bucket}/layer{l}")
                    v = s["t_sum"] / s["nt"] - s["c_sum"] / s["nc"]
                    name = lang if bucket == "all" else f"{lang}_{bucket}"
                    results[concept][str(l)][name] = v
                    print(f"{concept} {name} layer {l}: norm={v.norm().item():.4f}")

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"method": "text_mean_shift_same_vlm", "model": model.config._name_or_path,
                "layers": layers, "vectors": results}, vectors_path)
    print(f"Saved: {vectors_path}")


def evaluate(model, processor, data_dir, vectors_path, layers, alphas,
             split, limit, mask_type, max_new_tokens, out_file, seed):
    obj = torch.load(vectors_path, map_location="cpu", weights_only=True)
    if obj["model"] != model.config._name_or_path:
        raise ValueError(f"Vector model {obj['model']} differs from current VLM "
                         f"{model.config._name_or_path}; never mix hidden spaces.")
    available = obj["vectors"]
    rows = [x for x in read_jsonl(data_dir / "visual_eval.jsonl") if x["split"] == split]
    # Limit is PER concept, with the same examples for all interventions/languages.
    rows = [r for task in ("counting", "spatial")
            for r in [x for x in rows if x["task"] == task][:limit if limit else None]]
    if not rows:
        raise ValueError(f"No samples found for split={split}")

    get_layers(model, layers)
    for l in layers:
        for concept in ("counting", "spatial"):
            if str(l) not in available[concept]:
                raise ValueError(f"Layer {l} missing from {concept} vectors; re-run extraction")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    # Random vectors are deterministic and matched on layer + concept. Unit normalized by the hook.
    random_vectors = {(task, l): torch.randn(
        available[task][str(l)]["en"].shape, generator=rng)
        for task in ("counting", "spatial") for l in layers}

    fields = ["id", "split", "task", "prompt_lang", "vector_lang", "condition",
              "layer", "alpha", "mask_type", "answer", "prediction", "parsed", "correct"]
    start = time.monotonic()
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in tqdm(rows, desc=f"Evaluate {split}"):
            task = sample["task"]
            image_path = data_dir / sample["image"]
            for prompt_lang in ("en", "zh"):
                question = sample["questions"][prompt_lang]
                configs = [("none", "baseline", -1, 0.0, None)]
                for l in layers:
                    for alpha in alphas:
                        if alpha == 0:
                            continue  # baseline is computed once, not redundantly.
                        for vlang in ("en", "zh"):
                            configs.append((vlang, "steering", l, alpha,
                                            available[task][str(l)][vlang]))
                        configs.append(("random", "random_control", l, alpha,
                                        random_vectors[(task, l)]))
                for vlang, condition, layer, alpha, vector in configs:
                    prediction = predict(
                        model, processor, image_path, question,
                        layer=layer if vector is not None else None, vector=vector,
                        alpha=alpha, mask_type=mask_type, max_new_tokens=max_new_tokens,
                    )
                    parsed = canonical_answer(prediction, task)
                    rec = dict(id=sample["id"], split=split, task=task,
                               prompt_lang=prompt_lang, vector_lang=vlang,
                               condition=condition, layer=layer, alpha=alpha,
                               mask_type=mask_type, answer=sample["answer"],
                               prediction=prediction, parsed=parsed,
                               correct=int(parsed == sample["answer"]))
                    writer.writerow(rec)
                    f.flush()
    print(f"Saved {out_file} in {time.monotonic() - start:.1f}s")


def main():
    p = argparse.ArgumentParser(description="Bilingual textual steering for Qwen3-VL")
    p.add_argument("--mode", required=True, choices=["extract", "evaluate", "all"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default="outputs/pilot")
    p.add_argument("--vectors", default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--layers", default="6,12,18", help="Zero-based decoder layer ids")
    p.add_argument("--alphas", default="3,8", help="Steering strengths, unit-normalized vectors")
    p.add_argument("--mask-type", choices=["image", "text", "both"], default="both")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--limit", type=int, default=0, help="Limit per task (0=all)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    data_dir = Path(a.data)
    vector_path = Path(a.vectors or (data_dir / "vectors.pt"))
    result_path = Path(a.results or (data_dir / f"results_{a.split}.csv"))
    layers, alphas = int_list(a.layers), float_list(a.alphas)
    if not layers or not alphas or any(not math.isfinite(x) or x < 0 for x in alphas):
        p.error("Specify valid layers and finite non-negative steering strengths")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, processor = load_model(a.model)
    if a.mode in ("extract", "all"):
        extract(model, processor, data_dir, layers, vector_path)
    if a.mode in ("evaluate", "all"):
        evaluate(model, processor, data_dir, vector_path, layers, alphas, a.split,
                 a.limit, a.mask_type, a.max_new_tokens, result_path, a.seed)


if __name__ == "__main__":
    main()
