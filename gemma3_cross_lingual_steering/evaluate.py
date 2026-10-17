#!/usr/bin/env python3
"""2x2 EN/ZH steering through upstream Gemma3MeanShiftManipulator (unchanged).

Instantiate separate wrappers for each language and layer, sharing model weights,
so the upstream set_config() vector cache cannot apply stale language vectors.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from extract import MODEL_ID, REV, corpus_fingerprint

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / ".upstream"


def canonical(text, task):
    s = text.lower().strip()
    if task == "counting":
        digits = re.search(r"(?<!\d)([0-5])(?!\d)", s)
        if digits:
            return digits.group(1)
        words = {"zero": "0", "one": "1", "two": "2", "three": "3",
                 "four": "4", "five": "5", "零": "0", "一": "1",
                 "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}
        for word, number in words.items():
            exists = re.search(r"\b" + re.escape(word) + r"\b", s) if word.isascii() else word in s
            if exists:
                return number
    if task == "spatial":
        for name, terms in (("left", ["left", "左"]), ("right", ["right", "右"]),
                            ("above", ["above", "top", "上"]),
                            ("below", ["below", "bottom", "下"])):
            if any(re.search(r"\b" + t + r"\b", s) if t.isascii() else t in s
                   for t in terms):
                return name
    return ""


def run(data, split, layers, alphas, mask, limit, max_tokens):
    if not (UPSTREAM / "models" / "gemma3_meanshift_model.py").exists():
        raise RuntimeError("Upstream checkout missing: run bash run.sh")
    # Invoke author-supplied implementation rather than a second hook.
    sys.path.insert(0, str(UPSTREAM))
    from models.gemma3_meanshift_model import Gemma3MeanShiftManipulator, MeanShiftConfig
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from transformers import __version__ as transformers_version

    manifest = json.loads((data / "vectors" / "manifest.json").read_text())
    if (manifest.get("model") != MODEL_ID
            or manifest.get("upstream_revision") != REV
            or manifest.get("transformers") != transformers_version
            or manifest.get("corpus_sha256") != corpus_fingerprint(data)):
        raise RuntimeError("Vector manifest mismatch; run FORCE_EXTRACT=1 bash run.sh")
    if not set(layers).issubset(set(manifest["layers"])):
        raise RuntimeError(f"Requested layers {layers}; vectors cover {manifest['layers']}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gemma3 evaluation requires CUDA")

    raw = [json.loads(x) for x in (data / "visual_eval.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for task in ("counting", "spatial"):
        matches = [r for r in raw if r["split"] == split and r["task"] == task]
        rows.extend(matches[:limit] if limit > 0 else matches)
    if not rows:
        raise RuntimeError(f"No images for split={split}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval().to("cuda")
    n_layers = len(model.language_model.model.layers)
    if any(l < 0 or l >= n_layers for l in layers):
        raise RuntimeError(f"Gemma3 has {n_layers} decoder layers; requested {layers}")

    wrappers = {}
    for task in ("counting", "spatial"):
        for layer in layers:
            for lang in ("en", "zh"):
                for alpha in alphas:
                    path = data / "vectors" / task / lang / f"layer_{layer}_feature.pt"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    config = MeanShiftConfig(feature_paths={layer: str(path)},
                                             manipulation_values={layer: alpha},
                                             normalize_features=True)
                    wrappers[(task, layer, lang, alpha)] = Gemma3MeanShiftManipulator(
                        model=model, processor=processor, model_name=MODEL_ID,
                        device="cuda", config=config, debug=False)

    fields = ["id", "split", "task", "prompt_lang", "vector_lang",
              "layer", "alpha", "mask", "prediction", "parsed", "answer", "correct"]
    destination = data / f"predictions_{split}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(rows, desc=f"Original Gemma3 steering {split}"):
            with Image.open(data / row["image"]) as source:
                image = source.convert("RGB")
            for prompt_lang in ("en", "zh"):
                first = True
                for layer in layers:
                    for alpha in alphas:
                        for vector_lang in ("en", "zh"):
                            manipulator = wrappers[(row["task"], layer, vector_lang, alpha)]
                            generated = manipulator.generate_with_manipulation(
                                image=image,
                                text_prompt=row["questions"][prompt_lang],
                                masks=mask,
                                max_new_tokens=max_tokens,
                                do_sample=False,
                                output_baseline=first,
                                save_distribution_plot=False)
                            if first:
                                baseline = generated["original_text"] or ""
                                parsed = canonical(baseline, row["task"])
                                writer.writerow(dict(id=row["id"], split=split,
                                    task=row["task"], prompt_lang=prompt_lang,
                                    vector_lang="none", layer=-1, alpha=0, mask=mask,
                                    prediction=baseline, parsed=parsed, answer=row["answer"],
                                    correct=int(parsed == row["answer"])))
                                first = False
                            prediction = generated["manipulated_text"] or ""
                            parsed = canonical(prediction, row["task"])
                            writer.writerow(dict(id=row["id"], split=split,
                                task=row["task"], prompt_lang=prompt_lang,
                                vector_lang=vector_lang, layer=layer, alpha=alpha,
                                mask=mask, prediction=prediction, parsed=parsed,
                                answer=row["answer"],
                                correct=int(parsed == row["answer"])))
                            f.flush()
    print(f"Saved {destination}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/gemma3")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--layers", default="8,16")
    p.add_argument("--alphas", default="5")
    p.add_argument("--mask", choices=["image_token", "text_token", "both"], default="both")
    p.add_argument("--limit", type=int, default=4, help="Images per task; 0=all")
    p.add_argument("--max-tokens", type=int, default=24)
    a = p.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    if not layers or not alphas or any(x <= 0 for x in alphas):
        p.error("Specify layers and positive alphas")
    run(Path(a.data), a.split, layers, alphas, a.mask, a.limit, a.max_tokens)


if __name__ == "__main__":
    main()
