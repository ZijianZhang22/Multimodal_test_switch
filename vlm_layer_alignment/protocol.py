"""PaliGemma2 multilingual CV-Bench protocol shared by frozen alignment & LoRA.

Data: local train/validation/test partition of user's 840 CV-Bench prompts.
No test examples in representation fitting, gradient updates, or alpha selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ID = "google/paligemma2-3b-mix-448"
PROMPT_PREFIX = "answer en Answer the multiple choice question by only responding the letter of the correct answer. "
LANGS = ("en", "zh", "sw")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_prompt(row, language):
    return PROMPT_PREFIX + row["questions"][language]


def parse_letter(value):
    value = str(value).strip()
    for pattern in (r"^\s*\(?([A-Fa-f])\)?(?=\W|$)",
                    r"^(?:answer|option|choice|jibu|答案|选项|the answer is)\s*[:：]?\s*\(?([A-Fa-f])\)?(?=\W|$)"):
        m = re.search(pattern, value, re.I)
        if m:
            return m.group(1).upper()
    return ""


def prepare(input_jsonl, out_dir, val_fraction=0.2, seed=42):
    """Divide only OLD local 'val' into train/new-val. Preserve old test untouched.

    Require unique image file across all splits; test partition never repurposed.
    All question translations and fixed answer letters are copied unchanged.
    """
    src = Path(input_jsonl).resolve()
    out = Path(out_dir).resolve()
    rows = load_jsonl(src)
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    if not rows:
        raise ValueError("No data")
    srcroot = src.parent
    for r in rows:
        if not {"id", "task", "split", "image", "answer", "questions"} <= set(r):
            raise ValueError(f"Missing essential fields in {r.get('id')}")
        if any(lang not in r["questions"] for lang in LANGS):
            raise ValueError("Need en, zh, sw questions")
        if not re.fullmatch("[A-F]", str(r["answer"])):
            raise ValueError(f"Invalid answer label {r['id']}")
        f = Path(r["image"])
        if not f.is_absolute():
            f = (srcroot / f).resolve()
        if not f.is_file():
            raise FileNotFoundError(f)
        r["image"] = str(f)
    if len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("Duplicate CVBench sample IDs")
    if {r["split"] for r in rows} - {"val", "test"}:
        raise ValueError("Expected upstream local val/test; use a clean existing CVBench dataset")
    # Exact byte-identical images may have DIFFERENT filenames across tasks.
    # Group by image SHA256. If public-test rows collide with upstream local-val,
    # EXCLUDE those val rows (rather than using known test images for training).
    hashes = {r["image"]: file_hash(r["image"]) for r in rows}
    test_hashes = {hashes[r["image"]] for r in rows if r["split"] == "test"}
    overlapping_val_ids = [r["id"] for r in rows if r["split"] == "val"
                           and hashes[r["image"]] in test_hashes]
    rows = [r for r in rows if r["split"] == "test" or
            hashes[r["image"]] not in test_hashes]
    chosen_val = set()
    rng = random.Random(seed)
    for task in sorted(set(r["task"] for r in rows)):
        candidates = sorted({hashes[r["image"]] for r in rows
                             if r["split"] == "val" and r["task"] == task})
        rng.shuffle(candidates)
        if len(candidates) < 5:
            raise ValueError(f"Too few upstream val images in task {task}")
        chosen_val.update(candidates[:max(1, round(len(candidates) * val_fraction))])
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        new = "test" if r["split"] == "test" else ("val" if hashes[r["image"]] in chosen_val else "train")
        splits[new].append({**r, "split": new})
    split_hashes = {key: {hashes[r["image"]] for r in group} for key, group in splits.items()}
    if any(split_hashes[a] & split_hashes[b] for a,b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Same image content occurs across train/val/test")
    for key, group in splits.items():
        if not group:
            raise ValueError(f"Empty split {key}")
        write_jsonl(out / f"{key}.jsonl", group)
    ids = {k: {r["image"] for r in v} for k, v in splits.items()}
    if any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Image ID leakage")
    manifest = {"source_jsonl": str(src), "sha256": file_hash(src),
                "seed": seed, "val_fraction_of_upstream_val": val_fraction,
                "model": MODEL_ID,
                "note": "Original public CV-Bench test partition was locally split; NOT official independent benchmark training.",
                "excluded_local_val_ids_due_to_exact_test_image_collision": overlapping_val_ids,
                "counts": {k: len(v) for k, v in splits.items()},
                "image_counts": {k: len(ids[k]) for k in splits},
                "translation_verified": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared", manifest["counts"], "at", out, flush=True)
    return splits


def load_splits(data_dir):
    d = Path(data_dir)
    result = {k: load_jsonl(d / f"{k}.jsonl") for k in ("train", "val", "test")}
    for k in result:
        if not result[k]:
            raise ValueError(f"Empty {k} split")
    keys = [{r["image"] for r in result[k]} for k in result]
    if len(set.union(*keys)) != sum(map(len, keys)):
        raise ValueError("Images overlap across train/validation/test")
    return result


def load_model(model_id=MODEL_ID, dtype="bf16", peft_adapter=None):
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for PaliGemma2; tests use a separate CPU mock")
    dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dt, attn_implementation="sdpa",
        low_cpu_mem_usage=True).to("cuda").eval()
    if peft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, peft_adapter).eval()
    return processor, model


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_layers(model):
    root = base_model(model)
    # transformers==4.51.3: PaliGemmaForConditionalGeneration.language_model.model.layers
    obj = root.language_model.model.layers
    if not obj or not hasattr(obj[0], "register_forward_hook"):
        raise RuntimeError("Cannot locate PaliGemma2 decoder layers")
    return obj


def multimodal_inputs(processor, image_path, prompt, device="cuda", suffix=None):
    from PIL import Image
    with Image.open(image_path) as im:
        image = im.convert("RGB")
        kwargs = dict(images=image, text=prompt, return_tensors="pt")
        if suffix is not None:
            kwargs["suffix"] = suffix
        batch = processor(**kwargs)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def prompt_last_index(batch):
    """Last non-image PROMPT token, excluding any fine-tuning suffix."""
    import torch
    ids = batch["input_ids"]
    attention = batch.get("attention_mask", torch.ones_like(ids)).bool()
    if "token_type_ids" in batch:
        attention &= batch["token_type_ids"].eq(0)
    if attention.shape[0] != 1:
        raise ValueError("Implement only batch_size=1 for exact token alignment")
    valid = attention[0].nonzero(as_tuple=False).view(-1)
    if valid.numel() == 0:
        raise ValueError("No valid prompt token")
    return int(valid[-1].item())


def decoder_hook(module, inputs, output, modify):
    """Modify decoder hidden tensor while retaining all other tuple values."""
    tensor = output[0] if isinstance(output, tuple) else output
    altered = modify(tensor)
    if isinstance(output, tuple):
        return (altered,) + output[1:]
    return altered


def generate_answer(model, processor, row, language, layer=None, intervention=None, max_new_tokens=8):
    """Generate once with a hook on prompt prefill ONLY; never mutate cached decode steps."""
    import torch
    batch = multimodal_inputs(processor, row["image"], get_prompt(row, language))
    index = prompt_last_index(batch)
    handle = None
    if intervention is not None:
        if layer is None:
            raise ValueError("Specify a decoder layer for the intervention")
        layers = decoder_layers(model)
        if not 0 <= layer < len(layers):
            raise ValueError("Layer out of range")
        def hook(mod, args, output):
            def patch(h):
                if h.shape[1] <= index:
                    return h  # cached autoregressive decoding, not initial prompt
                result = h.clone()
                result[:, index, :] = intervention(result[:, index, :])
                return result
            return decoder_hook(mod, args, output, patch)
        handle = layers[layer].register_forward_hook(hook)
    try:
        model.eval()
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        response = processor.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if handle is not None:
            handle.remove()
    return response


def extract_hidden(model, processor, row, language, layers):
    import torch
    batch = multimodal_inputs(processor, row["image"], get_prompt(row, language))
    pos = prompt_last_index(batch)
    captured = {}
    handles = []
    modules = decoder_layers(model)
    for l in layers:
        if not 0 <= l < len(modules):
            raise ValueError(f"Decoder layer {l} is invalid (n={len(modules)})")
        def add(layer):
            def hook(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer] = h[0, pos, :].detach().float().cpu().numpy()
            return hook
        handles.append(modules[l].register_forward_hook(add(l)))
    try:
        with torch.inference_mode():
            model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError("Missing layer activations")
    return captured


def evaluate(model, processor, rows, language, method, out_csv, layer=None, intervention=None, max_new_tokens=8):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ("id", "task", "language", "method", "layer", "answer", "prediction", "parsed", "correct", "valid")
    done = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if not done:
            wr.writeheader()
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            response = generate_answer(model, processor, row, language,
                                       layer=layer, intervention=intervention,
                                       max_new_tokens=max_new_tokens)
            parsed = parse_letter(response)
            wr.writerow({"id": row["id"], "task": row["task"], "language": language,
                         "method": method, "layer": -1 if layer is None else layer,
                         "answer": row["answer"], "prediction": response,
                         "parsed": parsed, "correct": int(parsed == row["answer"]), "valid": int(bool(parsed))})
            f.flush()
    with out_csv.open(newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    actual = {r["id"] for r in rows}
    results = [r for r in results if r["id"] in actual]
    if len(results) != len(actual):
        raise RuntimeError("Incomplete evaluation")
    return {"n": len(results), "accuracy": sum(int(r["correct"]) for r in results) / len(results),
            "valid_rate": sum(int(r["valid"]) for r in results) / len(results),
            "per_task": {t: sum(int(r["correct"]) for r in results if r["task"] == t) /
                         sum(r["task"] == t for r in results)
                         for t in sorted({r["task"] for r in results})}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Existing visual_eval.jsonl, with en/zh/sw prompts")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    a = p.parse_args()
    prepare(a.source, a.out, a.val_fraction, a.seed)


if __name__ == "__main__":
    main()