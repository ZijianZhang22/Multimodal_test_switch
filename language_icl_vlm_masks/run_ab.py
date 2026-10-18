#!/usr/bin/env python3
"""A/B experiment: Kirtane-style vectors from Gemma2 vs PaliGemma2 decoder.

A: google/gemma-2-2b (pre-multimodal) -> PaliGemma2
B: language decoder of google/paligemma2-3b-mix-448 -> PaliGemma2

For BOTH extraction sources: identical parallel text sample IDs, concept-free
target-minus-English vectors, and the SAME PaliGemma2 evaluation checkpoint.
Each source × modality selects (layer, alpha) on disjoint validation data.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paligemma2_cross_lingual_steering" / "parity"))
from common import MODEL_ID, TEXT_ID, build_prompt, jsonl_read, score  # noqa: E402
from core import (partition, sample_pairs, hash_pairs, extract_vectors,
                  token_masks, steer_decoder, demos_text)  # noqa: E402
from checkpoint_eval import evaluate_resumable  # noqa: E402

HEADERS = ["id", "task", "split", "lang", "source", "mask", "layer", "alpha",
           "response", "normalized", "valid", "correct", "tokens_edited"]


def parse_ints(value):
    numbers = []
    for entry in value.split(","):
        if "-" in entry:
            lo, hi = map(int, entry.split("-", 1))
            numbers.extend(range(lo, hi + 1))
        elif entry.strip():
            numbers.append(int(entry))
    result = sorted(set(numbers))
    if not result or min(result) < 0:
        raise ValueError("Invalid layer list")
    return result


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def select_samples(rows, per_task):
    grouped = defaultdict(list)
    for sample in rows:
        grouped[sample["task"]].append(sample)
    return [s for task in sorted(grouped)
            for s in (grouped[task][:per_task] if per_task else grouped[task])]


def compose_vlm_prompt(sample, lang, demos):
    """Keep the PaliGemma task prefix at position zero in ICL prompts.

    Original paper-parity single-question prompt:
        answer zh <instruction> <question>
    For ICL, insert English demonstrations after the prefix/instruction and
    before the target question, never before 'answer zh'.
    """
    baseline = build_prompt(sample, lang, "native")
    if not demos:
        return baseline
    question = sample["prompts"][lang]
    if not question or not baseline.endswith(question):
        raise RuntimeError("PaliGemma prompt unexpectedly changed; cannot place demos")
    prompt = baseline[:-len(question)] + "\n" + demos + question
    if not prompt.startswith(f"answer {lang} "):
        raise RuntimeError("PaliGemma task prefix must be at the beginning")
    return prompt


def compare_vectors(a, b, layers, filename):
    """Float32 cosine, norm, norm ratio for exactly corresponding decoder layers."""
    rows = []
    for layer in layers:
        left = a[layer].float()
        right = b[layer].float()
        if left.shape != right.shape:
            raise RuntimeError("Backbone and VLM vector hidden sizes differ")
        norm_a = float(left.norm())
        norm_b = float(right.norm())
        cosine = float(torch.nn.functional.cosine_similarity(
            left[None, :], right[None, :], dim=-1)[0])
        rows.append({"layer": layer, "base_norm": norm_a, "vlm_norm": norm_b,
                     "vlm_to_base_norm_ratio": norm_b / max(norm_a, 1e-12),
                     "cosine_similarity": cosine})
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def compare_tokenizers(tok_a, tok_b, pairs):
    """Audit source/target tokenization without assuming identical token IDs.

    Cross-checkpoint language steering transfers *activation-space* vectors,
    not token IDs. The PaliGemma tokenizer may use different BOS/special-token
    handling and added multimodal tokens. Encode each paired text with its own
    tokenizer; differences are a measured confound, not an execution error.
    """
    report = {"checked": 0, "full_token_ids_equal": 0,
              "content_token_ids_equal": 0,
              "source_token_count": 0, "vlm_token_count": 0,
              "source_roundtrip_changed": 0, "vlm_roundtrip_changed": 0,
              "examples": []}
    for item in pairs[:min(8, len(pairs))]:
        for passage in item:
            aa = tok_a(passage, add_special_tokens=True)["input_ids"]
            bb = tok_b(passage, add_special_tokens=True)["input_ids"]
            content_a = tok_a(passage, add_special_tokens=False)["input_ids"]
            content_b = tok_b(passage, add_special_tokens=False)["input_ids"]
            report["checked"] += 1
            report["full_token_ids_equal"] += int(aa == bb)
            report["content_token_ids_equal"] += int(content_a == content_b)
            report["source_token_count"] += len(aa)
            report["vlm_token_count"] += len(bb)
            dec_a = tok_a.decode(content_a, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
            dec_b = tok_b.decode(content_b, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
            report["source_roundtrip_changed"] += int(dec_a != passage)
            report["vlm_roundtrip_changed"] += int(dec_b != passage)
            if aa != bb and len(report["examples"]) < 3:
                report["examples"].append({
                    "text_sha256": hashlib.sha256(passage.encode()).hexdigest(),
                    "source_tokens": len(aa), "vlm_tokens": len(bb),
                    "source_prefix_ids": aa[:8], "vlm_prefix_ids": bb[:8]})
    return report


def summarize(rows, sources, modalities):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["id"], r["lang"])][(r["source"], r["mask"])] = r
    results = []
    for source in sources:
        for mask in modalities:
            paired = [(g[("none", "baseline")], g[(source, mask)])
                      for g in grouped.values()
                      if ("none", "baseline") in g and (source, mask) in g]
            if not paired:
                continue
            n = len(paired)
            fixes = sum(not int(base["correct"]) and int(steer["correct"])
                        for base, steer in paired)
            breaks = sum(int(base["correct"]) and not int(steer["correct"])
                         for base, steer in paired)
            results.append(dict(source=source, mask=mask, n=n,
                                baseline_accuracy=sum(int(b["correct"]) for b, _ in paired) / n,
                                steering_accuracy=sum(int(s["correct"]) for _, s in paired) / n,
                                delta_pp=100 * (fixes - breaks) / n,
                                fixed=fixes, broken=breaks,
                                changed_correctness=(fixes + breaks) / n))
    return results


def write_csv(path, rows, headers):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Existing prepared parity CV-Bench dataset directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lang", choices=["zh", "sw"], default="zh")
    parser.add_argument("--layers", default="8,12,16")
    parser.add_argument("--alphas", default="0.5,1.0,2.0")
    parser.add_argument("--masks", default="text,image,both")
    parser.add_argument("--sources", default="base,vlm", help="Keep base,vlm for full A/B")
    parser.add_argument("--extract-limit", type=int, default=12,
                        help="Extraction anchor count, 0=all")
    parser.add_argument("--limit-val", type=int, default=3, help="Per task; 0=all")
    parser.add_argument("--limit-test", type=int, default=3, help="Per task; 0=all")
    parser.add_argument("--shots", type=int, default=6, help="English task demonstrations")
    parser.add_argument("--k", type=int, default=6, help="QA per vector extraction block")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--skip-test", action="store_true",
                        help="Validation only; no held-out test inspection")
    parser.add_argument("--resume-test", action="store_true",
                        help="Only evaluate if test.csv does not already exist")
    args = parser.parse_args()

    layers = parse_ints(args.layers)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    masks = [x.strip() for x in args.masks.split(",") if x.strip()]
    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    if (not alphas or any(a <= 0 for a in alphas) or
            not masks or len(masks) != len(set(masks)) or
            not set(masks) <= {"text", "image", "both"} or
            not sources or len(sources) != len(set(sources)) or
            not set(sources) <= {"base", "vlm"} or
            min(args.k, args.shots, args.extract_limit, args.limit_val,
                args.limit_test, args.max_new_tokens) < 0 or args.k == 0 or
            args.max_new_tokens == 0):
        parser.error("Invalid hyperparameters or masks/sources")

    data = args.data.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not (data / "dataset_manifest.json").exists():
        raise FileNotFoundError("Prepared parity dataset required; see README_AB.md")
    rows = list(jsonl_read(data / "visual_eval.jsonl"))
    splits = partition(rows)
    selected_val = select_samples(splits["val"], args.limit_val)
    selected_test = select_samples(splits["test"], args.limit_test)
    pairs = sample_pairs(splits["compute"], args.lang, args.k,
                         args.extract_limit, args.seed)
    dataset_hash = hashlib.sha256((data / "visual_eval.jsonl").read_bytes()).hexdigest()
    pair_hash = hash_pairs(pairs)
    config = {"dataset_sha256": dataset_hash, "paired_text_sha256": pair_hash,
              "model_base": TEXT_ID, "model_vlm": MODEL_ID,
              "layers": layers, "alphas": alphas, "masks": masks, "sources": sources,
              "lang": args.lang, "seed": args.seed, "k": args.k, "shots": args.shots,
              "extract_limit": args.extract_limit, "limit_val": args.limit_val,
              "limit_test": args.limit_test, "max_new_tokens": args.max_new_tokens,
              "method_version": "kirtane_vlm_ab_v2_prefix_first"}
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Settings/corpus changed: choose a NEW output directory.")
    atomic_json(config_path, config)
    print("Extraction pairs:", len(pairs), "Validation:", len(selected_val),
          "Held-out test:", len(selected_test), flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    # Experiment A: extract on independent original Gemma2, then FREE memory.
    vectors = {}
    if "base" in sources:
        p = out / "vectors_base.pt"
        if p.exists():
            vectors["base"] = torch.load(p, map_location="cpu", weights_only=True)
        else:
            print("A: extracting original text-backbone vectors", flush=True)
            model_base = AutoModelForCausalLM.from_pretrained(
                TEXT_ID, torch_dtype=torch.bfloat16).eval().to("cuda")
            tokenizer_base = AutoTokenizer.from_pretrained(TEXT_ID)
            if max(layers) >= len(model_base.model.layers):
                raise RuntimeError("Requested layer does not exist in original Gemma2")
            vectors["base"] = extract_vectors_base(model_base, tokenizer_base, pairs, layers)
            torch.save(vectors["base"], p)
            del model_base, tokenizer_base
            gc.collect()
            torch.cuda.empty_cache()

    # Same final VLM checkpoint for B extraction and both A/B interventions.
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    vlm = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).eval().to("cuda")
    if max(layers) >= len(vlm.language_model.model.layers):
        raise RuntimeError("Requested layer does not exist in VLM decoder")
    tok = processor.tokenizer
    if "base" in sources:
        # Same paired texts, different checkpoint-native tokenizations.
        # Record discrepancies; there is no requirement for matching token IDs.
        tok_base = AutoTokenizer.from_pretrained(TEXT_ID)
        audit = compare_tokenizers(tok_base, tok, pairs)
        atomic_json(out / "tokenization_audit.json", audit)
        if audit["full_token_ids_equal"] != audit["checked"]:
            print("NOTE: tokenizer token IDs differ across checkpoints; "
                  "vector extraction uses each checkpoint's native tokenizer. "
                  "See tokenization_audit.json.", flush=True)
        del tok_base
    if "vlm" in sources:
        p = out / "vectors_vlm.pt"
        if p.exists():
            vectors["vlm"] = torch.load(p, map_location="cpu", weights_only=True)
        else:
            print("B: extracting VLM decoder text-only vectors", flush=True)
            vectors["vlm"] = extract_vectors(vlm, tok, pairs, layers, "cuda")
            torch.save(vectors["vlm"], p)
    for source in sources:
        for layer in layers:
            vector = vectors[source][layer]
            if vector.ndim != 1 or vector.numel() != vlm.config.text_config.hidden_size:
                raise RuntimeError(f"Vector shape incompatible with VLM ({source}, L{layer})")
    if {"base", "vlm"} <= set(sources):
        report = compare_vectors(vectors["base"], vectors["vlm"], layers,
                                 out / "vector_cosine_norms.csv")
        print("Layer-wise vector cosine/norm:", report, flush=True)

    def generate(image, prompt, source, modality, layer, alpha):
        inputs = processor(text=prompt, images=image,
                           return_tensors="pt").to(torch.bfloat16).to("cuda")
        input_length = inputs["input_ids"].shape[1]
        if source == "none":
            with torch.inference_mode():
                result = vlm.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                      do_sample=False)
            return processor.decode(result[0][input_length:],
                                    skip_special_tokens=True).strip(), 0
        masks_all = token_masks(
            inputs["input_ids"], vlm.config.image_token_index,
            [vlm.config.pad_token_id, vlm.config.bos_token_id,
             vlm.config.eos_token_id])
        selected = masks_all[modality]
        if not selected.any().item():
            raise RuntimeError(f"No {modality} tokens to steer")
        with torch.inference_mode(), steer_decoder(
                vlm, layer, vectors[source][layer], alpha, selected) as edits:
            result = vlm.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False)
        if edits["calls"] != 1:
            raise RuntimeError(f"Unexpected prefill steering count: {edits}")
        return processor.decode(result[0][input_length:],
                                skip_special_tokens=True).strip(), edits["tokens"]

    def evaluate(samples, split_name, grid, output_file):
        """Durably append each generated condition and resume missing cells."""
        configs = [("none", "baseline", -1, 0.0)] + grid

        def sample_rows(sample, missing):
            # Avoid loading images and prompts for fully completed samples.
            with Image.open(data / sample["image"]) as original:
                image = original.copy()
            demos = demos_text(splits["compute"], sample["task"], args.shots)
            prompt = compose_vlm_prompt(sample, args.lang, demos)
            for source, mask, layer, alpha in missing:
                answer, edited = generate(image, prompt, source, mask, layer, alpha)
                scored = score(answer, sample["answer"], sample["choices"],
                               sample["choices_i18n"][args.lang], args.lang)
                yield (source, mask, layer, alpha), dict(
                    id=sample["id"], task=sample["task"], split=split_name,
                    lang=args.lang, source=source, mask=mask, layer=layer,
                    alpha=alpha, response=answer, tokens_edited=edited, **scored)

        return evaluate_resumable(samples, configs, output_file, HEADERS,
                                  sample_rows, split_name)

    chosen_path = out / "selected_validation.json"
    if chosen_path.exists():
        chosen = json.loads(chosen_path.read_text())
        if not (out / "validation.csv").exists():
            raise RuntimeError("Selection exists without a completed validation.csv")
    else:
        grid = [(s, m, l, a) for s in sources for m in masks
                for l in layers for a in alphas]
        val_rows = evaluate(selected_val, "validation", grid, out / "validation.csv")
        chosen = {}
        for source in sources:
            chosen[source] = {}
            for mask in masks:
                options = [(l, a) for l in layers for a in alphas]
                def score_config(setting):
                    subset = [r for r in val_rows if r["source"] == source and
                              r["mask"] == mask and (r["layer"], r["alpha"]) == setting]
                    if len(subset) != len(selected_val):
                        raise RuntimeError("Incomplete validation cells")
                    return (sum(int(r["correct"]) for r in subset) / len(subset),
                            -setting[1], -setting[0])
                layer, alpha = max(options, key=score_config)
                chosen[source][mask] = dict(layer=layer, alpha=alpha)
        atomic_json(chosen_path, chosen)
    print("Frozen validation selection:", chosen, flush=True)

    if args.skip_test:
        print("Validation complete; held-out test not evaluated.", flush=True)
        return
    test_file = out / "test.csv"
    if test_file.exists() and not args.resume_test:
        print("Existing test.csv found; validating/reusing saved predictions.",
              flush=True)
    grid = [(source, mask, int(chosen[source][mask]["layer"]),
             float(chosen[source][mask]["alpha"])) for source in sources for mask in masks]
    test_rows = evaluate(selected_test, "test", grid, test_file)
    summary = summarize(test_rows, sources, masks)
    write_csv(out / "summary.csv", summary,
              ["source", "mask", "n", "baseline_accuracy", "steering_accuracy",
               "delta_pp", "fixed", "broken", "changed_correctness"])
    lines = ["# Kirtane Language Steering → PaliGemma2: A/B results", "",
             "A = original Gemma2-2B; B = PaliGemma2 decoder.", "",
             "| Source | Mask | n | Baseline | Steered | Δ pp | Fixed | Broken |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("| {source} | {mask} | {n} | {baseline_accuracy:.3f} | "
                     "{steering_accuracy:.3f} | {delta_pp:+.2f} | {fixed} | {broken} |"
                     .format(**r))
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


@torch.inference_mode()
def extract_vectors_base(model, tokenizer, pairs, layers):
    """Same mathematical extractor as core.extract_vectors, for bare Gemma2."""
    totals = {int(l): None for l in layers}
    if not pairs:
        raise ValueError("No paired examples")
    for src, tgt in pairs:
        pair_states = []
        for passage in (src, tgt):
            inputs = tokenizer(passage, return_tensors="pt")
            if inputs["input_ids"].shape[-1] > 2048:
                raise ValueError("QA extraction text exceeds 2048-token budget")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            hstates = model(**inputs, output_hidden_states=True,
                            use_cache=False).hidden_states
            attn = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
            pooled = {}
            for layer in layers:
                h = hstates[layer + 1].float()
                mask = attn.unsqueeze(-1).to(h.dtype)
                pooled[layer] = ((h * mask).sum(1) /
                                 mask.sum(1).clamp_min(1))[0].cpu()
            pair_states.append(pooled)
        for layer in layers:
            delta = pair_states[1][layer] - pair_states[0][layer]
            totals[layer] = delta if totals[layer] is None else totals[layer] + delta
    return {layer: (state / len(pairs)).contiguous()
            for layer, state in totals.items()}


if __name__ == "__main__":
    main()
