"""Kirtane & Huang (2026) language vector extraction for VLM mask ablations.

Unlike Gan et al. concept Mean Shift, these are parallel Q&A language
differences, extracted from the VLM's own language decoder.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import random
from collections import defaultdict
import torch


def partition(samples):
    """Split each task's tune examples in half: extraction vs validation.

    Pre-existing test samples are never used for extraction or selection.
    """
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(s)
    result = {"compute": [], "val": [], "test": []}
    for task in sorted(by_task):
        rows = by_task[task]
        for i, s in enumerate(x for x in rows if x["split"] == "tune"):
            result["compute" if i % 2 == 0 else "val"].append(s)
        result["test"].extend(x for x in rows if x["split"] == "test")
    ids = [s["id"] for v in result.values() for s in v]
    assert len(ids) == len(set(ids)), "Overlapping sample IDs"
    return result


def qa_text(s, lang):
    """Same paired sample ID and correct choice; translated question/choices."""
    prompt = s["prompts"][lang].strip()
    answer = str(s["answer"]).strip()
    if lang == "zh":
        return f"问题：{prompt}\n答案：{answer}\n"
    if lang == "sw":
        return f"Swali: {prompt}\nJibu: {answer}\n"
    return f"Question: {prompt}\nAnswer: {answer}\n"


def sample_pairs(compute, lang, k=6, max_samples=0, seed=42):
    """Match authors' multi-QA extraction: anchor i plus k-1 random pairs."""
    if k < 1 or not compute:
        raise ValueError("Extraction requires compute rows and k>=1")
    rows = sorted(compute, key=lambda s: s["id"])
    anchors = rows[:max_samples] if max_samples else rows
    rng = random.Random(seed)
    out = []
    for anchor in anchors:
        pool = [s for s in rows if s["id"] != anchor["id"]]
        chosen = [anchor] + [rng.choice(pool or [anchor]) for _ in range(k - 1)]
        out.append(("\n".join(qa_text(s, "en") for s in chosen),
                    "\n".join(qa_text(s, lang) for s in chosen)))
    return out


def hash_pairs(pairs):
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest()


@torch.inference_mode()
def extract_vectors(model, tokenizer, pairs, layers, device, max_input=2048):
    """Mean over token positions then mean(target minus English) over samples.

    A forward hook on decoder layer L observes output hidden_states[L+1].
    Thus use L+1 here to ensure extraction and injection layers match.
    """
    sums = {int(l): None for l in layers}
    if not pairs:
        raise ValueError("Empty extraction pairs")
    for en, target in pairs:
        features = []
        for passage in (en, target):
            inputs = tokenizer(passage, return_tensors="pt")
            if inputs["input_ids"].shape[-1] > max_input:
                raise ValueError("Extraction exceeds max_input; no silent truncation")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hs = model.language_model(**inputs, output_hidden_states=True,
                                      use_cache=False).hidden_states
            mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
            pooled = {}
            for l in layers:
                h = hs[int(l) + 1].float()
                m = mask.unsqueeze(-1).to(h.dtype)
                pooled[int(l)] = ((h * m).sum(1) / m.sum(1).clamp_min(1))[0].cpu()
            features.append(pooled)
        for l in layers:
            l = int(l)
            delta = features[1][l] - features[0][l]
            sums[l] = delta if sums[l] is None else sums[l] + delta
    return {l: (total / len(pairs)).contiguous() for l, total in sums.items()}


def token_masks(input_ids, image_token_id, special_ids):
    """Mask decoder positions AFTER vision-to-language projection, not pixels.

    Mirrors original PaliGemma manipulator's exemption of last text token.
    """
    if input_ids.ndim != 2:
        raise ValueError("Expect [batch, tokens]")
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token in special_ids:
        if token is not None:
            special |= input_ids == int(token)
    image = input_ids == int(image_token_id)
    text = ~image & ~special
    for b in range(input_ids.shape[0]):
        valid = (~special[b]).nonzero(as_tuple=False).flatten()
        if len(valid):
            text[b, valid[-1]] = False
    return {"text": text, "image": image, "both": image | text}


@contextlib.contextmanager
def steer_decoder(model, layer, vector, alpha, mask):
    """Modify layer output on prefill only; cached token-by-token decode is untouched."""
    if mask.ndim != 2 or not mask.any().item():
        raise ValueError("Invalid/empty steering mask")
    layer = int(layer)
    vector = vector.detach().float()
    edits = {"calls": 0, "tokens": 0}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] != mask.shape[1]:
            return output
        if h.shape[0] != mask.shape[0] or h.shape[2] != vector.numel():
            raise RuntimeError("Activation shape does not match mask/vector")
        active = mask.to(h.device).unsqueeze(-1)
        shift = vector.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
        edited = h + float(alpha) * active.to(h.dtype) * shift
        edits["calls"] += 1
        edits["tokens"] += int(mask.sum().item())
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    handle = model.language_model.model.layers[layer].register_forward_hook(hook)
    try:
        yield edits
    finally:
        handle.remove()


def demos_text(compute, task, n=6):
    if not n:
        return ""
    rows = [s for s in compute if s["task"] == task]
    if len(rows) < n:
        raise ValueError(f"Not enough {task} examples for {n} demonstrations")
    return "English examples:\n" + "\n\n".join(
        f"Question: {s['prompts']['en']}\nAnswer: {s['answer']}"
        for s in rows[:n]) + "\n\n"


def paired_metrics(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["lang"], row["id"])][row["mask"]] = row
    result = {}
    for lang in sorted({r["lang"] for r in rows}):
        for mask in ("text", "image", "both"):
            pairs = [(g["baseline"], g[mask]) for (lg, _), g in grouped.items()
                     if lg == lang and "baseline" in g and mask in g]
            if not pairs:
                continue
            n = len(pairs)
            gain = sum(int(b["correct"]) == 0 and int(s["correct"]) == 1 for b, s in pairs)
            loss = sum(int(b["correct"]) == 1 and int(s["correct"]) == 0 for b, s in pairs)
            result[(lang, mask)] = {"n": n, "baseline_accuracy": sum(int(b["correct"]) for b, _ in pairs) / n,
                                    "steered_accuracy": sum(int(s["correct"]) for _, s in pairs) / n,
                                    "delta_pp": 100 * (gain - loss) / n, "fixed": gain,
                                    "broken": loss, "discordance": (gain + loss) / n}
    return result
