#!/usr/bin/env python3
"""MRRE-inspired language steering + upstream Mean Shift concept steering.

Language direction (EN−LANG) is extracted from paired last prompt tokens;
norm-preserving hook edits last-position decoder states on prefill and decode.
This adapts MRRE stage 1, not a reproduction of its reasoning-trace setup.
The first-letter-token logit margin is only a proxy for response confidence.
"""
from __future__ import annotations

import contextlib
import sys
import torch
from common import MODEL_ID, UPSTREAM, build_prompt, score


def decoder_layers(model):
    return model.language_model.model.layers


def steer_last(h, v, alpha):
    out = h.clone()
    old = h[:, -1, :].float()
    new = old + alpha * v.to(device=h.device, dtype=torch.float32)
    new = new * (old.norm(dim=-1, keepdim=True) /
                 new.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    out[:, -1, :] = new.to(h.dtype)
    return out


@contextlib.contextmanager
def lang_steering(model, spec):
    handles = []
    try:
        for layer, vector, alpha in spec:
            def hook(module, inputs, output, vector=vector, alpha=alpha):
                if isinstance(output, tuple):
                    return (steer_last(output[0], vector, alpha),) + tuple(output[1:])
                return steer_last(output, vector, alpha)
            handles.append(decoder_layers(model)[layer].register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def capture_last(model, layers):
    states, handles = {}, []
    try:
        for layer in layers:
            def hook(module, inputs, output, layer=layer):
                h = output[0] if isinstance(output, tuple) else output
                states[layer] = h[0, -1].detach().float().cpu()
            handles.append(decoder_layers(model)[layer].register_forward_hook(hook))
        yield states
    finally:
        for h in handles:
            h.remove()


class Runner:
    def __init__(self, data, max_new_tokens=20, mask="both"):
        sys.path.insert(0, str(UPSTREAM))
        from models.paligemma_meanshift_model import MeanShiftConfig, PaliGemmaMeanShiftManipulator
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        self.MSC, self.MSM = MeanShiftConfig, PaliGemmaMeanShiftManipulator
        self.processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16).eval().to("cuda")
        self.data, self.max_new_tokens, self.mask = data, max_new_tokens, mask
        self.letter_ids = []
        for letter in "ABCDEF":
            ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Letter {letter} is not one token: {ids}")
            self.letter_ids.append(ids[0])
        self.wrappers = {}

    def inputs(self, image, prompt):
        return self.processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to("cuda")

    def concept_manipulator(self, c):
        if not c:
            return None
        layer = int(c["layer"])
        path = self.data / "vectors" / c["variant"] / c["concept"] / f"layer_{layer}_feature.pt"
        key = (c["variant"], c["concept"], layer)
        cfg = self.MSC(feature_paths={layer: str(path)},
                       manipulation_values=float(c["alpha"]), normalize_features=False)
        if key not in self.wrappers:
            self.wrappers[key] = self.MSM(
                model=self.model, processor=self.processor, model_name=MODEL_ID,
                device="cuda", config=cfg, debug=False)
        else:
            self.wrappers[key].config = cfg
        return self.wrappers[key]

    @torch.inference_mode()
    def margin(self, image, prompt, sample, lang_spec, concept):
        enc = self.inputs(image, prompt)
        manip = self.concept_manipulator(concept)
        with lang_steering(self.model, lang_spec):
            if manip is not None:
                manip.remove_hooks()
                manip.image_mask = manip.create_image_mask(enc["input_ids"], masks=self.mask)
                manip._register_manipulation_hooks()
            try:
                logits = self.model(**enc).logits[0, -1].float()
            finally:
                if manip is not None:
                    manip.remove_hooks()
                    manip.image_mask = None
        n = len(sample["choices"])
        opt = logits[self.letter_ids[:n]]
        gold = ord(sample["answer"].strip("()").strip().upper()[0]) - 65
        if not 0 <= gold < n or n < 2:
            raise ValueError(f"Bad correct choice: {sample['answer']} / {n}")
        other = torch.cat((opt[:gold], opt[gold + 1:]))
        best = int(opt.argmax())
        return {"margin": float(opt[gold] - other.max()),
                "p_gold": float(torch.softmax(opt, 0)[gold]),
                "argmax_letter": chr(65 + best), "argmax_correct": int(best == gold)}

    @torch.inference_mode()
    def generate(self, image, prompt, lang_spec, concept):
        manip = self.concept_manipulator(concept)
        with lang_steering(self.model, lang_spec):
            if manip is None:
                enc = self.inputs(image, prompt)
                n = enc["input_ids"].shape[-1]
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
                return self.processor.decode(out[0][n:], skip_special_tokens=True).strip()
            result = manip.generate_with_manipulation(
                image=image, text_prompt=prompt, max_new_tokens=self.max_new_tokens,
                masks=self.mask, save_distribution_plot=False,
                do_sample=False, output_baseline=False)
            return (result.get("manipulated_text") or "").strip()

    def run(self, image, sample, lang, style, lang_spec, concept, do_generate=True):
        prompt = build_prompt(sample, lang, style)
        row = self.margin(image, prompt, sample, lang_spec, concept)
        if do_generate:
            ans = self.generate(image, prompt, lang_spec, concept)
            row.update(response=ans, **score(ans, sample["answer"],
                       sample["choices"], sample["choices_i18n"][lang], lang))
        return row
