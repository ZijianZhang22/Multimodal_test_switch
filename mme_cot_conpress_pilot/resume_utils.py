"""Helpers for Qwen3-VL-Thinking inference and token accounting.

This experiment intentionally does NOT resume model.generate() in chunks:
separate multimodal generation calls can misalign M-RoPE input masks.
"""


def model_context_budget(model, prompt_tokens):
    """Return (architectural_context, available_output_tokens).

    Returns architecture-dependent capacity. The configured generation cap,
    if any, is applied separately and cannot exceed this available capacity.
    """
    config = model.config
    text_config = getattr(config, "text_config", None)
    limit = getattr(text_config, "max_position_embeddings", None)
    if limit is None:
        limit = getattr(config, "max_position_embeddings", None)
    if not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("Cannot determine model's architectural context limit")
    available = limit - int(prompt_tokens)
    if available <= 0:
        raise RuntimeError(
            f"Input consumes {prompt_tokens} tokens, meeting/exceeding "
            f"model context limit {limit}"
        )
    return limit, available


def effective_generation_limit(model, prompt_tokens, requested_max_new_tokens):
    """Return (context_window, permitted_generation_tokens).

    Keep the model's architectural context boundary authoritative; a
    practical configurable cap avoids accidentally using hundreds of
    thousands of output tokens on a long-running, non-terminating trace.
    """
    if not isinstance(requested_max_new_tokens, int) or requested_max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be a positive integer")
    context_window, available = model_context_budget(model, prompt_tokens)
    return context_window, min(available, requested_max_new_tokens)


def completion_status(generated_ids, eos_ids, effective_max_new_tokens):
    """Distinguish real natural termination from length-capped outputs."""
    if not generated_ids:
        raise ValueError("Model returned no output tokens")
    if any(int(t) in eos_ids for t in generated_ids):
        return "model_eos"
    if len(generated_ids) >= effective_max_new_tokens:
        return "token_limit"
    return "generation_error"


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
