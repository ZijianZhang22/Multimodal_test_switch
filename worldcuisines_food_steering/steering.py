"""Explicit decoder-output intervention, scoped to prompt prefill only."""
from contextlib import contextmanager
import torch


def layers_of(model):
    # Supported layout: transformers==4.51.3 PaliGemma/Gemma2.
    language_model = getattr(model, 'language_model', model)
    return language_model.model.layers


def token_mask(input_ids, image_id, special_ids, mode):
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    for idx in special_ids:
        valid &= input_ids != idx
    if mode == 'image':
        return input_ids == image_id
    if mode == 'text':
        return valid & (input_ids != image_id)
    if mode == 'both':
        return valid | (input_ids == image_id)
    raise ValueError(mode)


@contextmanager
def intervene(model, layer, vector, alpha, mask):
    """Unnormalized vector; same-norm random controls; remove hook even on errors.

    Only the first (full prompt) forward pass is modified. Cached decode tokens
    are unmodified. Different from the upstream wrapper's token exclusions.
    """
    if alpha == 0:
        yield
        return
    block = layers_of(model)[layer]
    used = False

    def hook(module, args, output):
        nonlocal used
        if used:
            return output
        used = True
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[:2] != mask.shape or vector.shape != hidden.shape[-1:]:
            raise ValueError('Steering mask/vector shape mismatch')
        shifted = hidden + alpha * mask.to(hidden.device).unsqueeze(-1) * vector.to(hidden)
        return (shifted,) + output[1:] if isinstance(output, tuple) else shifted

    handle = block.register_forward_hook(hook)
    try:
        yield
        if not used:
            raise RuntimeError('Steering hook was never called')
    finally:
        handle.remove()


def random_control(vector, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    r = torch.randn(vector.shape, generator=generator)
    return r * (vector.float().norm() / r.norm())
