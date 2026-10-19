#!/usr/bin/env python3
"""Run a job list on PaliGemma2-3B with the author's unmodified PaliGemmaMeanShiftManipulator.

A job = {task, prompt_lang, style, concept, variant, layer, alpha}. Baselines are generated
once per (sample, prompt_lang, style). Model loading, prompt, mask, generation settings and
scoring follow upstream cvbench_grid_search.py / utils.evaluate_cvbench:
  bf16, default attention, prefix + instruction + prompt, masks='both', greedy, 20 new tokens,
  response .strip(), normalize_answer().
Output is an append-only, resumable CSV.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from common import FIELDS, MODEL_ID, UPSTREAM, build_prompt, jsonl_read, score
JOB_KEYS = ('task', 'prompt_lang', 'style', 'concept', 'variant', 'layer', 'alpha')


def row_key(r):
    return (str(r['id']), str(r['prompt_lang']), str(r['style']), str(r['concept']),
            str(r['variant']), str(int(r['layer'])), f"{float(r['alpha']):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--split', required=True, choices=['tune', 'test'])
    p.add_argument('--jobs', required=True, type=Path, help='JSON list of jobs')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--limit', type=int, default=0, help='per task, for smoke tests')
    p.add_argument('--mask', default='both', choices=['image_token', 'text_token', 'both'])
    p.add_argument('--max-new-tokens', type=int, default=20)
    a = p.parse_args()

    jobs = json.loads(a.jobs.read_text())
    for j in jobs:
        j['layer'] = int(j['layer'])
        j['alpha'] = float(j['alpha'])
    settings = {'split': a.split, 'mask': a.mask, 'max_new_tokens': a.max_new_tokens, 'limit': a.limit,
                'jobs_sha256': hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(),
                'vectors': json.loads((a.data / 'vectors' / 'manifest.json').read_text()),
                'dataset': json.loads((a.data / 'dataset_manifest.json').read_text())}
    spath = a.out.with_suffix('.settings.json')
    if a.out.exists() and spath.exists() and json.loads(spath.read_text()) != settings:
        raise SystemExit(f'{a.out} was produced with different settings/jobs; use another --out')
    spath.write_text(json.dumps(settings, indent=2, ensure_ascii=False))

    samples = [s for s in jsonl_read(a.data / 'visual_eval.jsonl') if s['split'] == a.split]
    if a.limit:
        by = {}
        for s in samples:
            by.setdefault(s['task'], []).append(s)
        samples = [s for v in by.values() for s in v[:a.limit]]
    jobs_by_task = {}
    for j in jobs:
        jobs_by_task.setdefault(j['task'], []).append(j)
    samples = [s for s in samples if s['task'] in jobs_by_task]

    done = set()
    if a.out.exists():
        with a.out.open(encoding='utf-8', newline='') as f:
            done = {row_key(r) for r in csv.DictReader(f)}
    total = sum(len(jobs_by_task[s['task']]) for s in samples)
    print(f'{len(samples)} samples, {total} steered generations planned, {len(done)} rows already done', flush=True)

    sys.path.insert(0, str(UPSTREAM))
    from models.paligemma_meanshift_model import MeanShiftConfig, PaliGemmaMeanShiftManipulator
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval().to('cuda')
    print('attn_implementation =', model.config._attn_implementation, flush=True)
    wrappers = {}

    def manipulator(j):
        path = a.data / 'vectors' / j['variant'] / j['concept'] / f"layer_{j['layer']}_feature.pt"
        k = (j['variant'], j['concept'], j['layer'])
        cfg = MeanShiftConfig(feature_paths={j['layer']: str(path)},
                              manipulation_values=j['alpha'], normalize_features=False)
        if k not in wrappers:
            wrappers[k] = PaliGemmaMeanShiftManipulator(model=model, processor=processor, model_name=MODEL_ID,
                                                        device='cuda', config=cfg, debug=False)
        else:
            wrappers[k].config = cfg          # same vector & layer; only alpha changes
        return wrappers[k]

    def baseline(image, prompt):              # identical to the manipulator's "original" branch
        inputs = processor(text=prompt, images=image, return_tensors='pt').to(torch.bfloat16).to('cuda')
        n = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        return processor.decode(out[0][n:], skip_special_tokens=True).strip()

    new_file = not a.out.exists()
    with a.out.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        bar = tqdm(total=total, desc=f'{a.split}')
        for s in samples:
            sjobs = jobs_by_task[s['task']]
            todo = [j for j in sjobs if row_key({'id': s['id'], **j}) not in done]
            needed_base = {(j['prompt_lang'], j['style']) for j in sjobs}
            base_todo = [pl for pl in needed_base if row_key({'id': s['id'], 'prompt_lang': pl[0], 'style': pl[1],
                                                              'concept': 'none', 'variant': 'none', 'layer': -1, 'alpha': 0}) not in done]
            bar.update(len(sjobs) - len(todo))
            if not todo and not base_todo:
                continue
            with Image.open(a.data / s['image']) as im:
                image = im.copy()

            def emit(j, text):
                lang = j['prompt_lang']
                sc = score(text, s['answer'], s['choices'], s['choices_i18n'][lang], lang)
                row = {'id': s['id'], 'task': s['task'], 'split': s['split'], 'prompt_lang': lang, 'style': j['style'],
                       'concept': j['concept'], 'variant': j['variant'], 'layer': j['layer'], 'alpha': j['alpha'],
                       'response': text, **sc}
                w.writerow(row)
                done.add(row_key(row))

            for lang, style in sorted(base_todo):
                emit({'prompt_lang': lang, 'style': style, 'concept': 'none', 'variant': 'none', 'layer': -1, 'alpha': 0.0},
                     baseline(image, build_prompt(s, lang, style)))
            for j in todo:
                res = manipulator(j).generate_with_manipulation(
                    image=image, text_prompt=build_prompt(s, j['prompt_lang'], j['style']),
                    max_new_tokens=a.max_new_tokens, masks=a.mask, save_distribution_plot=False,
                    do_sample=False, output_baseline=False)
                emit(j, (res.get('manipulated_text') or '').strip())
                bar.update(1)
            f.flush()
        bar.close()
    print(f'Saved {a.out}', flush=True)


if __name__ == '__main__':
    main()
