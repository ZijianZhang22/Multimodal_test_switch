"""Text-only Gemma2 food anchors -> pooled target minus context activations."""
import argparse
import json
from pathlib import Path
import torch
from core import TEXT_MODEL, VLM_MODEL, digest, write_json
from steering import layers_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='outputs/smoke')
    p.add_argument('--concepts', default=str(Path(__file__).with_name('concepts.json')))
    p.add_argument('--layers', default='8,16')
    p.add_argument('--revision', default='main')
    a = p.parse_args()
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM, __version__
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for model extraction')
    root = Path(a.data) / 'vectors'
    if root.exists():
        raise FileExistsError('Vector directory exists; use a new DATA directory')
    anchors = json.loads(Path(a.concepts).read_text())
    if not anchors:
        raise ValueError('Empty concept corpus')
    layers = sorted(set(map(int, a.layers.split(','))))
    revision = HfApi().model_info(TEXT_MODEL, revision=a.revision).sha
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL, revision=revision,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
    if not layers or min(layers) < 0 or max(layers) >= len(layers_of(model)):
        raise ValueError('Invalid zero-based decoder layer')
    sums = {l: {'target': 0, 'context': 0, 'nt': 0, 'nc': 0} for l in layers}
    for row in anchors:
        sentence, target = row['sentence'], row['target']
        if not target or sentence.count(target) != 1:
            raise ValueError('Each target must occur exactly once: ' + sentence)
        batch = tok(sentence, return_tensors='pt', return_offsets_mapping=True)
        offsets = batch.pop('offset_mapping')[0].tolist()
        start, end = sentence.index(target), sentence.index(target) + len(target)
        valid = [i for i, (x, y) in enumerate(offsets) if y > x]
        positive = [i for i in valid if offsets[i][0] < end and offsets[i][1] > start]
        context = [i for i in valid if i not in positive]
        if not positive or not context:
            raise ValueError('Empty target or context token span')
        handles = []
        def collect(layer):
            def hook(module, args, output):
                h = (output[0] if isinstance(output, tuple) else output)[0].detach().float()
                s = sums[layer]
                s['target'] += h[positive].double().sum(0).cpu()
                s['context'] += h[context].double().sum(0).cpu()
                s['nt'] += len(positive)
                s['nc'] += len(context)
            return hook
        try:
            for layer in layers:
                handles.append(layers_of(model)[layer].register_forward_hook(collect(layer)))
            with torch.inference_mode():
                # Decoder only: avoid allocating vocabulary logits during extraction.
                model.model(**batch.to('cuda'), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    root.mkdir(parents=True)
    norms = {}
    for layer, s in sums.items():
        v = (s['target'] / s['nt'] - s['context'] / s['nc']).float()
        if not torch.isfinite(v).all() or v.norm() == 0:
            raise ValueError('Invalid extracted vector')
        torch.save(v, root / f'layer_{layer}.pt')
        norms[str(layer)] = v.norm().item()
    write_json(root / 'manifest.json', dict(text_model=TEXT_MODEL, text_revision=revision,
        vlm=VLM_MODEL, layers=layers, anchors=anchors, corpus_sha256=digest(anchors),
        norms=norms, transformers=__version__, torch=torch.__version__,
        method='Pooled target minus pooled non-target, non-special tokens; raw decoder outputs; no normalization'))


if __name__ == '__main__':
    main()
