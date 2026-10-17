#!/usr/bin/env python3
"""Text-only Mean Shift vectors using Gemma3 language decoder.

Same formula and pooling convention as upstream get_features_mean_shift.py:
  mean(highlighted non-BOS token activations) - mean(other non-BOS tokens)
Differences: offset-based target matching for Chinese; Gemma3 VLM language
backbone rather than external Gemma2 text checkpoint. Zero-based layer IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

MODEL_ID = "google/gemma-3-4b-it"
ROOT = Path(__file__).resolve().parent
REV = "ce1c0f0cffed9b859901ce347f9d2b869a512178"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def corpus_fingerprint(data):
    sha = hashlib.sha256()
    for concept in ("counting", "spatial"):
        for lang in ("en", "zh"):
            sha.update((data / "corpus" / f"{concept}_{lang}.jsonl").read_bytes())
    return sha.hexdigest()


def locate_tokens(tokenizer, sentence, target):
    if sentence.count(target) != 1:
        raise ValueError(f"Target must occur once: {target!r} in {sentence!r}")
    batch = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
    offsets = batch.pop("offset_mapping")[0].tolist()
    s = sentence.index(target)
    e = s + len(target)
    ids = batch["input_ids"][0]
    # Original code excludes BOS from both target/control pools.
    valid = [i for i, (start, end) in enumerate(offsets)
             if end > start and not (i == 0 and ids[i].item() == tokenizer.bos_token_id)]
    target_positions = [i for i in valid
                        if offsets[i][0] < e and offsets[i][1] > s]
    control_positions = [i for i in valid if i not in set(target_positions)]
    if not target_positions or not control_positions:
        raise RuntimeError(f"Empty target/control token group: {sentence!r}")
    return batch, target_positions, control_positions


def mean_shift_from_hidden(rows, tokenizer, model, layers):
    # Stream totals in float64 to avoid BF16 roundoff / huge stored activations.
    totals = {l: {"target": None, "control": None, "nt": 0, "nc": 0}
              for l in layers}
    for row in tqdm(rows, desc="Extract mean shift"):
        inputs, target, control = locate_tokens(tokenizer, row["sentence"], row["target"])
        with torch.inference_mode():
            output = model.language_model(**inputs.to(model.device),
                                          output_hidden_states=True, use_cache=False)
        states = output.hidden_states
        if states is None:
            raise RuntimeError("No hidden_states returned by Gemma3 text decoder")
        for layer in layers:
            # hidden_states[0] is embeddings, hidden_states[L+1] is decoder layer L output
            h = states[layer + 1][0].detach().to(dtype=torch.float32, device="cpu")
            t = h[target].double().sum(0)
            c = h[control].double().sum(0)
            entry = totals[layer]
            entry["target"] = t if entry["target"] is None else entry["target"] + t
            entry["control"] = c if entry["control"] is None else entry["control"] + c
            entry["nt"] += len(target)
            entry["nc"] += len(control)
        del output, states
    return {l: (v["target"]/v["nt"] - v["control"]/v["nc"]).float()
            for l, v in totals.items()}


def extract(data, layers, overwrite=False):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to extract Gemma3-4B vectors")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval().to("cuda")
    n_layers = len(model.language_model.model.layers)
    if not layers or not all(0 <= x < n_layers for x in layers):
        raise ValueError(f"layers must be zero-based from 0 to {n_layers-1}")
    for concept in ("counting", "spatial"):
        for lang in ("en", "zh"):
            dest = data / "vectors" / concept / lang
            dest.mkdir(parents=True, exist_ok=True)
            rows = read_jsonl(data / "corpus" / f"{concept}_{lang}.jsonl")
            vectors = mean_shift_from_hidden(rows, processor.tokenizer, model, layers)
            for layer in layers:
                path = dest / f"layer_{layer}_feature.pt"
                if path.exists() and not overwrite:
                    raise FileExistsError(f"Refusing to overwrite vectors: {path}")
                torch.save(vectors[layer], path)
                print(f"{concept}/{lang} layer={layer}: "
                      f"norm={vectors[layer].norm().item():.4f}")
    manifest = {"model": MODEL_ID, "transformers": __import__("transformers").__version__,
                "upstream_revision": REV, "layers": layers,
                "corpus_sha256": corpus_fingerprint(data),
                "method": "TextSteersVision mean-shift; Gemma3 language decoder"}
    (data / "vectors" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved vectors/manifest.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/gemma3")
    p.add_argument("--layers", default="8,16")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    extract(Path(a.data), [int(x) for x in a.layers.split(",")], overwrite=a.overwrite)


if __name__ == "__main__":
    main()
