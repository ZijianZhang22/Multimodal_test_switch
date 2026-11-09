"""CLIP text and image towers as the English (E) and image (I) encoders.

Both towers come from ONE CLIP checkpoint, so E and I already share an embedding space: CLIP replaces the weak
self-trained English->image bridge. The low-resource encoder (L) is NOT CLIP and must stay monolingual.

Embeddings are CLIP's projected features (the vectors CLIP compares with cosine similarity), not L2-normalized here;
the downstream PCA / Rosetta code centers and normalizes them.
"""

import torch

DEFAULT_CLIP = "openai/clip-vit-base-patch32"
CLIP_MAX_TOKENS = 77


def projected(output):
    """transformers 4.x returns the projected tensor; 5.x returns an output whose pooler_output is that tensor."""
    if torch.is_tensor(output):
        return output
    for name in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    raise TypeError(f"Unexpected CLIP feature output: {type(output).__name__}")


def load_clip(name, revision, device):
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

    model = CLIPModel.from_pretrained(name, revision=revision).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    processor = AutoImageProcessor.from_pretrained(name, revision=revision)
    return model, tokenizer, processor


@torch.inference_mode()
def encode_texts(model, tokenizer, texts, device):
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=CLIP_MAX_TOKENS, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    return projected(model.get_text_features(**inputs)).float().cpu().numpy()


@torch.inference_mode()
def encode_images(model, processor, images, device):
    inputs = processor(images=list(images), return_tensors="pt")
    return projected(model.get_image_features(pixel_values=inputs["pixel_values"].to(device))).float().cpu().numpy()
