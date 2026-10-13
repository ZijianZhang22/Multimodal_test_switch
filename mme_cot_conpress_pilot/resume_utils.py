"""Utilities for safely re-prefilling Qwen3-VL after a generation chunk.

The processor returns mm_token_type_ids alongside input_ids for multimodal
M-RoPE. When model.generate() returns longer input_ids, those modality ids
MUST be extended by text-token (0) markers to the same sequence length.
"""
import torch


def extend_multimodal_inputs(work, completed):
    """Append generated text to multimodal input tensors without losing image state.

    Args:
        work: Last model.generate() kwargs, with input_ids, attention_mask,
              image features, image_grid_thw, and mm_token_type_ids.
        completed: model.generate() return [batch_size, total_tokens] including
                   the previous input prefix and the new generated chunk.
    """
    old_tokens = work["input_ids"]
    if completed.ndim != 2 or old_tokens.ndim != 2:
        raise ValueError("Expected 2D input_ids, batch x tokens")
    if completed.shape[0] != old_tokens.shape[0]:
        raise ValueError("Batch size changed during continuation")
    if not torch.equal(completed[:, : old_tokens.shape[1]], old_tokens):
        raise ValueError("Generated sequence does not preserve the original input prefix")
    increment = completed.shape[1] - old_tokens.shape[1]
    if increment <= 0:
        raise ValueError("No tokens generated in the previous chunk")

    mask = work["attention_mask"]
    if mask.shape != old_tokens.shape:
        raise ValueError(
            f"attention_mask {tuple(mask.shape)} vs input_ids {tuple(old_tokens.shape)}"
        )
    appended_mask = torch.ones(
        (old_tokens.shape[0], increment), dtype=mask.dtype, device=mask.device
    )
    new_work = dict(work)
    new_work["input_ids"] = completed
    new_work["attention_mask"] = torch.cat([mask, appended_mask], dim=1)

    if "mm_token_type_ids" in work:
        types = work["mm_token_type_ids"]
        if types.shape != old_tokens.shape:
            raise ValueError(
                f"mm_token_type_ids {tuple(types.shape)} vs input_ids {tuple(old_tokens.shape)}"
            )
        text_types = torch.zeros(
            (old_tokens.shape[0], increment), dtype=types.dtype, device=types.device
        )
        new_work["mm_token_type_ids"] = torch.cat([types, text_types], dim=1)
    elif work.get("image_grid_thw") is not None:
        raise ValueError(
            "Missing mm_token_type_ids for image input; Qwen3-VL M-RoPE cannot continue."
        )

    if "position_ids" in work:
        # M-RoPE position IDs should be recomputed from full input_ids and
        # matching mm_token_type_ids by the model, not manually extended.
        new_work.pop("position_ids")

    return new_work


def thought_stats(raw, tokenizer):
    """Count approximate generated CoT tokens, accounting for Qwen's prompt.

    Qwen3-VL-Thinking's add_generation_prompt=True ALREADY inserts '<think>\\n'
    in the input prompt. Therefore the generated output often starts INSIDE the
    reasoning block and contains only '</think>' (no opening '<think>').
    """
    stop = raw.find("</think>")
    if stop < 0:
        return None, False
    open_at = raw.find("<think>")
    begin = open_at + len("<think>") if 0 <= open_at < stop else 0
    thought_text = raw[begin:stop]
    return len(tokenizer.encode(thought_text, add_special_tokens=False)), True
