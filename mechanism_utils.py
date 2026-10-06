#!/usr/bin/env python3
"""Shared utilities for internal-state and causal-mechanism experiments."""

import json

import torch

from run_qwen3vl import build_messages


DIRECTION_VECTORS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def vector_to_answer(x, y):
    if x == 0 and y == 0:
        return "SAME"
    vertical = "NORTH" if y < 0 else "SOUTH" if y > 0 else ""
    horizontal = "EAST" if x > 0 else "WEST" if x < 0 else ""
    return vertical + horizontal if vertical and horizontal else vertical or horizontal


def get_text_layers(model):
    candidates = [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return name, layers
        except Exception:
            pass
    raise AttributeError(
        "Could not locate text decoder layers. "
        "Expected a Qwen3-VL-style language-model layer stack."
    )


def prepare_inputs(processor, example, data_dir, device=None):
    messages = build_messages(example, data_dir)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    if device is not None:
        inputs = inputs.to(device)
    return inputs


def _find_subsequence(sequence, pattern, start=0):
    if not pattern:
        return None
    end = len(sequence) - len(pattern) + 1
    for i in range(start, end):
        if sequence[i:i + len(pattern)] == pattern:
            return i
    return None


def _marker_ids(tokenizer, marker):
    variants = [marker, "\n" + marker, " " + marker]
    outputs = []
    for text in variants:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids:
            outputs.append(ids)
    return outputs


def find_marker_start(input_ids, tokenizer, marker, start=0):
    best = None
    for ids in _marker_ids(tokenizer, marker):
        pos = _find_subsequence(input_ids, ids, start=start)
        if pos is not None and (best is None or pos < best):
            best = pos
    return best


def infer_hop_count(example):
    if example.get("hop_count") is not None:
        return int(example["hop_count"])
    return len(example["facts"])


def get_evidence_boundary_positions(inputs, processor, example):
    """
    Return token positions at the end of each physical evidence item and the
    final prompt position for arbitrary hop count H.

    The runner labels evidence as "Evidence 1:" ... "Evidence H:". We locate
    those markers in the tokenized multimodal prompt, then map physical
    positions back to latent logical roles.
    """
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokenizer = processor.tokenizer
    hop_count = infer_hop_count(example)

    starts = {}
    cursor = 0
    for physical in range(1, hop_count + 1):
        marker = f"Evidence {physical}:"
        pos = find_marker_start(ids, tokenizer, marker, start=cursor)
        if pos is None:
            raise RuntimeError(
                f"Could not locate tokenized marker {marker!r}. "
                "The prompt format may have changed."
            )
        starts[physical] = pos
        cursor = pos + 1

    question_start = find_marker_start(ids, tokenizer, "Question:", start=cursor)
    if question_start is None:
        raise RuntimeError("Could not locate Question: marker in tokenized prompt.")

    physical_boundaries = {}
    for physical in range(1, hop_count):
        physical_boundaries[physical] = starts[physical + 1] - 1
    physical_boundaries[hop_count] = question_start - 1

    logical_boundaries = {}
    logical_to_physical = {}
    for physical, fact in enumerate(example["facts"], start=1):
        logical = int(fact.get("logical_step", fact.get("step", physical)))
        logical_boundaries[logical] = physical_boundaries[physical]
        logical_to_physical[logical] = physical

    return {
        "physical": physical_boundaries,
        "logical": logical_boundaries,
        "logical_to_physical": logical_to_physical,
        "final": len(ids) - 1,
        "hop_count": hop_count,
    }


def logical_fact_map(example):
    result = {}
    for physical, fact in enumerate(example["facts"], start=1):
        logical = int(fact.get("logical_step", fact.get("step", physical)))
        result[logical] = fact
    return result


def build_latent_labels(example):
    """
    Build semantic labels for arbitrary hop count H.

    For every logical role k:
      factK        : direction of edge f_k
      sourceK      : T or V carrier
      prefixK      : coarse direction of f_1 + ... + f_k
      suffixK      : coarse direction of f_k + ... + f_H
      stateK_x/y   : exact cumulative displacement
      stateK_coord : exact cumulative coordinate string
    """
    facts = logical_fact_map(example)
    hop_count = len(facts)

    expected = list(range(1, hop_count + 1))
    if sorted(facts) != expected:
        raise ValueError(
            f"Logical steps must be contiguous 1..H; got {sorted(facts)}"
        )

    labels = {"hop_count": hop_count}
    for k in expected:
        labels[f"fact{k}"] = facts[k]["direction"]
        labels[f"source{k}"] = facts[k]["source"]

    x = 0
    y = 0
    for k in expected:
        dx, dy = DIRECTION_VECTORS[facts[k]["direction"]]
        x += dx
        y += dy
        labels[f"prefix{k}"] = vector_to_answer(x, y)
        labels[f"state{k}_x"] = int(x)
        labels[f"state{k}_y"] = int(y)
        labels[f"state{k}_coord"] = f"{x},{y}"

    x = 0
    y = 0
    suffix = {}
    for k in range(hop_count, 0, -1):
        dx, dy = DIRECTION_VECTORS[facts[k]["direction"]]
        x += dx
        y += dy
        suffix[k] = vector_to_answer(x, y)
    for k in expected:
        labels[f"suffix{k}"] = suffix[k]

    labels["answer"] = example["answer"]
    return labels


def parse_layer_spec(text, n_layers):
    """
    Supported forms:
      "20:32"                      inclusive absolute range
      "20:32:2"                    absolute range with stride
      "0,4,8,12"                   absolute indices
      "0%,25%,50%,65%,75%,90%,100%" normalized model depth
      "all"

    Percentage 0% maps to layer 0 and 100% maps to the final decoder layer.
    """
    raw = str(text).strip()
    lower = raw.lower()

    if lower == "all":
        return list(range(n_layers))

    if "%" in raw:
        values = []
        for item in raw.split(","):
            item = item.strip()
            if not item.endswith("%"):
                raise ValueError(
                    "When using percentage layer specs, every item must end in %."
                )
            pct = float(item[:-1])
            if pct < 0 or pct > 100:
                raise ValueError(f"Percentage outside [0,100]: {pct}")
            idx = round((pct / 100.0) * (n_layers - 1))
            values.append(int(idx))
        return sorted(set(values))

    if ":" in raw:
        parts = [int(x) for x in raw.split(":") if x != ""]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid layer spec: {raw}")
        layers = list(range(start, stop + 1, step))
    else:
        layers = [int(x.strip()) for x in raw.split(",") if x.strip()]

    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(
            f"Layer indices out of range for {n_layers} layers: {bad}"
        )
    return sorted(set(layers))
