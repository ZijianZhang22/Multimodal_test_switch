"""Greedy MCQ generation with baseline, prompting, +/- food and random controls."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from core import VLM_MODEL, digest, make_prompt, parse_answer, read_jsonl, write_json
from steering import intervene, layers_of, random_control, token_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='outputs/smoke')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--layers', default='8,16')
    p.add_argument('--alphas', default='0.1,0.3')
    p.add_argument('--mask', choices=['image', 'text', 'both'], default='both')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-tokens', type=int, default=16)
    p.add_argument('--revision', default='main')
    p.add_argument('--selection', type=Path, help='Required on test: frozen validation selection.json')
    a = p.parse_args()
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, __version__
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for evaluation')
    if a.limit < 0 or a.max_tokens < 1:
        raise ValueError('Invalid limit/max-tokens')
    torch.manual_seed(42)
    root = Path(a.data)
    manifest = json.loads((root / 'vectors/manifest.json').read_text())
    dataset = json.loads((root / 'dataset_manifest.json').read_text())
    if hashlib.sha256((root / 'samples.jsonl').read_bytes()).hexdigest() != dataset['samples_sha256']:
        raise ValueError('Dataset changed since preparation')
    if manifest['vlm'] != VLM_MODEL or manifest['transformers'] != __version__:
        raise ValueError('Vector model/transformers mismatch')
    layers = sorted(set(map(int, a.layers.split(','))))
    alphas = sorted(set(map(float, a.alphas.split(','))))
    import math
    if not alphas or any(not math.isfinite(x) or x <= 0 for x in alphas):
        raise ValueError('Supply positive finite alpha magnitudes')
    revision = HfApi().model_info(VLM_MODEL, revision=a.revision).sha
    if a.split == 'test':
        if not a.selection:
            raise ValueError('Test requires a validation-frozen --selection')
        selection = json.loads(a.selection.read_text())
        if selection['dataset_sha256'] != dataset['samples_sha256'] or selection['vector_manifest_sha256'] != digest(manifest):
            raise ValueError('Selection belongs to a different dataset/vector')
        if selection['vlm_revision'] != revision or selection['max_tokens'] != a.max_tokens:
            raise ValueError('Test must use validation model revision and max-tokens')
        layers, alphas, a.mask = [selection['layer']], [selection['alpha']], selection['mask']
    if not set(layers) <= set(manifest['layers']):
        raise ValueError('Extract requested layers first')
    vectors = {l: torch.load(root / f'vectors/layer_{l}.pt', map_location='cpu', weights_only=True) for l in layers}
    vector_hashes = {str(l): hashlib.sha256((root / f'vectors/layer_{l}.pt').read_bytes()).hexdigest() for l in layers}
    if a.split == 'test' and vector_hashes[str(layers[0])] != selection['vector_sha256']:
        raise ValueError('Selected vector file changed')
    samples = [r for r in read_jsonl(root / 'samples.jsonl') if r['split'] == a.split]
    samples = samples[:a.limit] if a.limit else samples
    if not samples or len({r['id'] for r in samples}) != len(samples):
        raise ValueError('Empty cohort or duplicate question IDs')
    settings = dict(split=a.split, layers=layers, alphas=alphas, mask=a.mask, limit=a.limit,
        max_tokens=a.max_tokens, vlm=VLM_MODEL, vlm_revision=revision,
        dataset_sha256=dataset['samples_sha256'], vector_manifest_sha256=digest(manifest),
        vector_hashes=vector_hashes, transformers=__version__, torch=torch.__version__,
        sample_ids=[r['id'] for r in samples], protocol='prefill-only-v1; raw-vector; greedy; strict parse')
    output = root / f'predictions_{a.split}.jsonl'
    settings_path = root / f'settings_{a.split}.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing run settings differ; use a new DATA directory')
    if output.exists() and not settings_path.exists():
        raise ValueError('Predictions have no provenance manifest')
    write_json(settings_path, settings)
    existing = read_jsonl(output) if output.exists() else []
    key = lambda r: (r['id'], r['condition'], r['layer'], r['alpha'], r['random_seed'])
    done = {key(r) for r in existing}
    processor = PaliGemmaProcessor.from_pretrained(VLM_MODEL, revision=revision)
    model = PaliGemmaForConditionalGeneration.from_pretrained(VLM_MODEL, revision=revision,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
    if min(layers) < 0 or max(layers) >= len(layers_of(model)):
        raise ValueError('Invalid layer')
    conditions = [('baseline', -1, 0., -1, None), ('prompt', -1, 0., -1, None)]
    for layer in layers:
        for alpha in alphas:
            conditions.extend([('food', layer, alpha, -1, vectors[layer]),
                               ('negative', layer, -alpha, -1, vectors[layer])])
            for seed in (0, 1, 2):
                conditions.append(('random', layer, alpha, seed, random_control(vectors[layer], seed)))
    with output.open('a') as f:
        for i, row in enumerate(samples):
            image_path = root / row['image']
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError('Image bytes changed: ' + str(image_path))
            with Image.open(image_path) as source:
                im = source.convert('RGB')
            for condition, layer, alpha, seed, vector in conditions:
                record = dict(id=row['id'], food_id=row['food_id'], split=a.split,
                    condition=condition, layer=layer, alpha=alpha, random_seed=seed)
                if key(record) in done:
                    continue
                prompt = make_prompt(row, condition == 'prompt')
                batch = processor(images=im, text=prompt, return_tensors='pt').to('cuda')
                batch['pixel_values'] = batch['pixel_values'].to(model.dtype)
                # Exclude image ID from special IDs; it is included explicitly by token_mask.
                image_id = model.config.image_token_index
                special = [x for x in processor.tokenizer.all_special_ids if x != image_id]
                mask = token_mask(batch['input_ids'], image_id, special, a.mask)
                if alpha and not mask.any():
                    raise ValueError('Empty intervention mask')
                with torch.inference_mode(), intervene(model, layer, vector, alpha, mask):
                    generated = model.generate(**batch, max_new_tokens=a.max_tokens,
                        do_sample=False, use_cache=True)
                raw = processor.decode(generated[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                prediction = parse_answer(raw, row['options'])
                record.update(prompt=prompt, raw=raw, prediction=prediction, answer=row['answer'],
                              correct=int(prediction == row['answer']), valid=int(bool(prediction)))
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                done.add(key(record))
            print(f'{a.split}: {i + 1}/{len(samples)}', flush=True)


if __name__ == '__main__':
    main()
