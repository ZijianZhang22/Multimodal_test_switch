"""Qwen3-VL layer hooks and scoring helpers.

Layers use ZERO-based indices into model.model.language_model.layers.
Steering modifies decoder-layer OUTPUTS. All interventions are prefill-only.
"""
import contextlib
import re
from pathlib import Path

import torch
from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def read_jsonl(path):
    import json
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(model_id=DEFAULT_MODEL):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot needs a CUDA GPU. Run it on RunPod/A100 or similar.")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    if not hasattr(model.model, "language_model") or not hasattr(model.model.language_model, "layers"):
        raise RuntimeError("Unexpected Transformers architecture: missing Qwen3-VL language decoder layers")
    return model, processor


def get_layers(model, layers):
    decoders = model.model.language_model.layers
    for layer in layers:
        if not 0 <= layer < len(decoders):
            raise ValueError(f"Invalid zero-based layer {layer}; expected 0..{len(decoders)-1}")
    return decoders


def text_target_masks(tokenizer, sentence, target):
    """Locate target by character span; handle multi-token Chinese/English targets."""
    start = sentence.find(target)
    if start < 0 or sentence.find(target, start + len(target)) >= 0:
        raise ValueError(f"Target must occur exactly once: {target!r} in {sentence!r}")
    encoded = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    end = start + len(target)
    target_pos = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    control_pos = [i for i, (a, b) in enumerate(offsets)
                   if b > a and not (a < end and b > start)]
    if not target_pos or not control_pos:
        raise ValueError(f"Unable to produce both token groups for {sentence!r}")
    return encoded, target_pos, control_pos


def token_mask(input_ids, processor, mask_type, image_token_id):
    """Image masks include all expanded Qwen <|image_pad|> tokens."""
    ids = input_ids[0]
    img = ids.eq(image_token_id)
    special = torch.zeros_like(img, dtype=torch.bool)
    for sid in processor.tokenizer.all_special_ids:
        if sid != image_token_id:
            special |= ids.eq(sid)
    text = ~special & ~img
    if mask_type == "image":
        mask = img
    elif mask_type == "text":
        mask = text
    elif mask_type == "both":
        mask = img | text
    else:
        raise ValueError(mask_type)
    if not bool(mask.any().item()):
        raise ValueError(f"No matching {mask_type} positions in the encoded prompt")
    return mask


@contextlib.contextmanager
def steer_layer(model, layer, vector, alpha, mask):
    """Insert additive direction on selected prefill positions only.

    Output may be Tensor or tuple (different Transformers versions).
    A shape check skips autoregressive decode steps; no persistent model edits.
    """
    if alpha == 0 or vector is None:
        yield
        return

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.ndim != 3 or h.shape[1] != mask.numel():
            return output
        norm = vector.float().norm().clamp_min(1e-12)
        addition = (vector.float() / norm * float(alpha)).to(device=h.device, dtype=h.dtype)
        h2 = h.clone()
        positions = mask.to(h.device)
        h2[:, positions, :] += addition
        if isinstance(output, tuple):
            return (h2,) + output[1:]
        return h2

    handle = get_layers(model, [layer])[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def predict(model, processor, image_path, question, layer=None, vector=None,
            alpha=0.0, mask_type="both", max_new_tokens=16):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}]}]
    batch = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    batch.pop("token_type_ids", None)
    batch = batch.to(model.device)
    image_token_id = model.config.image_token_id
    mask = token_mask(batch["input_ids"], processor, mask_type, image_token_id)
    with torch.inference_mode(), steer_layer(model, layer, vector, alpha, mask):
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
    gen = out[0, batch["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def canonical_answer(text, task):
    s = text.strip().lower()
    if task == "counting":
        numbers = re.findall(r"\d+", s)
        if numbers:
            return numbers[0]
        words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "一": "1", "二": "2", "三": "3", "四": "4", "五": "5"}
        for word, digit in words.items():
            if re.search(r"\b" + re.escape(word) + r"\b", s) if word.isascii() else word in s:
                return digit
    elif task == "spatial":
        for en, variants in (("left", ["left", "左"]), ("right", ["right", "右"]),
                             ("above", ["above", "top", "上"]), ("below", ["below", "bottom", "下"])):
            if any((re.search(r"\b" + x + r"\b", s) if x.isascii() else x in s)
                   for x in variants):
                return en
    return ""
