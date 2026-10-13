#!/usr/bin/env python3
"""Offline smoke tests for prompt pairing and scoring (no GPU, no dataset needed)."""
from prompts import ARITHMETIC_SUFFIX, build_prompt
from score import build_rows, build_summary, extract_tag, score_target

mock = {
    "id": "test-1", "question": "What does the graph imply?",
    "options": ["A. Increasing", "B. Decreasing"],
}
a = build_prompt(mock, "baseline")
b = build_prompt(mock, "plus_arithmetic")
assert b.startswith(a)
assert b[len(a):] == ARITHMETIC_SUFFIX
assert "1 + 1" not in a and "1 + 1" in b
assert score_target("B", "B", mock["options"]) == (True, "multiple_choice")
assert score_target("B. Decreasing", "B", mock["options"]) == (True, "multiple_choice")
assert score_target("A", "B", mock["options"]) == (False, "multiple_choice")
assert extract_tag("<target_answer>A</target_answer>", "target_answer") == "A"
records = []
for condition, target, thinking, total in [
    ("baseline", "B", 200, 240),
    ("plus_arithmetic", "B", 120, 175),
]:
    records.append({
        "id": "test-1", "category": "Logic", "subcategory": "",
        "condition": condition, "stop_reason": "model_eos",
        "gold_answer": "B", "options": mock["options"],
        "text_output": f"<target_answer>{target}</target_answer> "
                       + ("<arithmetic_answer>2</arithmetic_answer>" if condition != "baseline" else ""),
        "thinking_tokens": thinking, "generation_tokens": total,
    })
scored, pairs = build_rows(records)
assert len(scored) == 2
assert len(pairs) == 1
assert pairs[0]["compression_pct"] == 40.0
assert pairs[0]["arithmetic_correct"] is True
summary = build_summary(pairs)
assert next(x for x in summary if x["category"] == "ALL")["mc_baseline_accuracy"] == 1.0

from io import BytesIO
import base64
from PIL import Image
from prepare import to_pil

_buffer = BytesIO()
Image.new("RGB", (3, 2), (255, 0, 0)).save(_buffer, format="JPEG")
_raw_image = _buffer.getvalue()
_encoded_image = base64.b64encode(_raw_image).decode("ascii")
for _payload in (
    _encoded_image,
    "data:image/jpeg;base64," + _encoded_image,
    _raw_image,
    {"bytes": _raw_image},
):
    _image = to_pil(_payload)
    assert _image.size == (3, 2) and _image.mode == "RGB"


# Qwen3-VL resume regression: the new sequence and M-RoPE token types
# MUST stay aligned after >1024 generated tokens, without altering visual inputs.
import torch
from resume_utils import extend_multimodal_inputs, thought_stats

_original_work = {
    "input_ids": torch.tensor([[11, 20, 21, 22]], dtype=torch.long),
    "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
    "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]], dtype=torch.long),
    "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
    "pixel_values": torch.zeros(2, 3),
}
_first = torch.cat([_original_work["input_ids"], torch.full((1, 1024), 33)], dim=1)
_after_first = extend_multimodal_inputs(_original_work, _first)
assert _after_first["input_ids"].shape == (1, 1028)
assert _after_first["mm_token_type_ids"].shape == (1, 1028)
assert _after_first["attention_mask"].shape == (1, 1028)
assert torch.equal(_after_first["mm_token_type_ids"][:, :4], _original_work["mm_token_type_ids"])
assert torch.all(_after_first["mm_token_type_ids"][:, 4:] == 0)
assert _after_first["pixel_values"] is _original_work["pixel_values"]
assert _original_work["input_ids"].shape == (1, 4)  # no mutation

_second = torch.cat([_first, torch.full((1, 750), 34)], dim=1)
_after_second = extend_multimodal_inputs(_after_first, _second)
assert _after_second["mm_token_type_ids"].shape == (1, 1778)

class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

# Actual Qwen3-VL-Thinking chat template places '<think>' in the PROMPT.
# Generated text itself can therefore contain only the closing marker.
assert thought_stats("Reasoning a bit more ... </think>Final", _FakeTokenizer()) == (5, True)
assert thought_stats("<think>Step one step two</think>Final", _FakeTokenizer()) == (4, True)

print("OFFLINE SMOKE TEST PASSED")
