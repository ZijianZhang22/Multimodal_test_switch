#!/usr/bin/env python3
"""Mean Shift vectors from the text-only google/gemma-2-2b, aligned with upstream.

* EN : upstream get_features_mean_shift.mean_shift_probing, executed from the pinned
       upstream source file itself (AST-extracted, so it is byte-identical code).
* ZH/SW : same pooling math; target tokens located by tokenizer character offsets
       (upstream's space-prefixed token-ID matching cannot handle Chinese/Swahili spans).
       As a check, EN is also extracted with the offset method and the cosine to the
       upstream EN vector is reported per layer.
* Model loaded exactly like upstream main(): fp32, default attention implementation.
* All 26 layers are saved, so any layer of the grid can be used.

Derived variants (to separate "direction" from "magnitude"):
  <lang>_normen : the <lang> vector rescaled to the EN vector norm at the same layer/concept
  random_s<k>   : a seeded Gaussian direction rescaled to the EN norm (control)
"""
import argparse
import ast
import csv
import json
import typing
import zlib
from pathlib import Path

import torch
from tqdm import tqdm

from common import CONCEPTS, LANGS, N_TEXT_LAYERS, REV, TEXT_ID, UPSTREAM


def upstream_mean_shift():
    src_path = UPSTREAM / 'get_features' / 'get_features_mean_shift.py'
    src = src_path.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'mean_shift_probing')
    ns = {'torch': torch, 'tqdm': tqdm, **{k: getattr(typing, k) for k in ('List', 'Dict', 'Set', 'Tuple', 'Optional')}}
    exec(compile(ast.get_source_segment(src, fn), str(src_path), 'exec'), ns)
    return ns['mean_shift_probing']


def offset_mean_shift(model, tok, rows):
    """Upstream pooling (all target tokens vs all remaining non-BOS tokens), offset-aligned."""
    tgt_states, ctl_states = {}, {}
    for row in tqdm(rows, desc='offset mean shift'):
        sentence, target = row['sentence'], row['target']
        if sentence.count(target) != 1:
            raise ValueError((sentence, target))
        enc = tok(sentence, return_tensors='pt', return_offsets_mapping=True)
        offsets = enc.pop('offset_mapping')[0].tolist()
        start = sentence.index(target)
        end = start + len(target)
        valid = [i for i, (a, b) in enumerate(offsets) if b > a and i != 0]      # drop BOS
        tgt = [i for i in valid if offsets[i][0] < end and offsets[i][1] > start]
        ctl = [i for i in valid if i not in tgt]
        if not tgt or not ctl:
            raise RuntimeError(f'missing target/control: {sentence}')
        with torch.no_grad():
            hs = model(**enc.to(model.device), output_hidden_states=True).hidden_states
        for layer in range(len(hs) - 1):
            h = hs[layer + 1][0]
            tgt_states.setdefault(layer, []).append(h[tgt])
            ctl_states.setdefault(layer, []).append(h[ctl])
    return {l: (torch.cat(tgt_states[l]).mean(0) - torch.cat(ctl_states[l]).mean(0)).cpu() for l in tgt_states}


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float()[None], b.float()[None])[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--random-seeds', default='0')
    a = p.parse_args()
    data = a.data
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    tok = AutoTokenizer.from_pretrained(TEXT_ID)
    model = AutoModelForCausalLM.from_pretrained(TEXT_ID).to('cuda')       # fp32, default attn (= upstream)
    n_layers = len(model.model.layers)
    assert n_layers == N_TEXT_LAYERS, n_layers
    ms_upstream = upstream_mean_shift()

    vec = {}
    sanity = []
    for concept in CONCEPTS:
        for lang in LANGS:
            rows = [json.loads(x) for x in (data / 'corpus' / f'{concept}_{lang}.jsonl').read_text(encoding='utf-8').splitlines() if x]
            if lang == 'en':
                res = ms_upstream(model, tok, [{'sentence': r['sentence'], 'target': r['target']} for r in rows])
                res = {int(l): v.detach().float().cpu() for l, v in res.items()}
                off = offset_mean_shift(model, tok, rows)
                for l in range(n_layers):
                    sanity.append({'concept': concept, 'layer': l, 'cos_upstream_vs_offset': cos(res[l], off[l])})
            else:
                res = {l: v.float() for l, v in offset_mean_shift(model, tok, rows).items()}
            for l, v in res.items():
                vec[(lang, concept, l)] = v

    seeds = [int(s) for s in a.random_seeds.split(',') if s != '']
    norms, saved = [], {}

    def save(variant, concept, layer, v):
        d = data / 'vectors' / variant / concept
        d.mkdir(parents=True, exist_ok=True)
        torch.save(v.contiguous(), d / f'layer_{layer}_feature.pt')
        saved.setdefault(variant, set()).add(concept)

    for concept in CONCEPTS:
        for l in range(n_layers):
            en_norm = vec[('en', concept, l)].norm()
            for lang in LANGS:
                v = vec[(lang, concept, l)]
                save(lang, concept, l, v)
                if lang != 'en':
                    save(f'{lang}_normen', concept, l, v * (en_norm / v.norm()))
                norms.append({'concept': concept, 'layer': l, 'lang': lang, 'norm': float(v.norm())})
            for s in seeds:
                g = torch.Generator().manual_seed(1000003 * s + 7919 * l + zlib.crc32(concept.encode()))
                r = torch.randn(vec[('en', concept, l)].shape, generator=g)
                save(f'random_s{s}', concept, l, r * (en_norm / r.norm()))

    geom = []
    for l in range(n_layers):
        for concept in CONCEPTS:
            for i, x in enumerate(LANGS):
                for y in LANGS[i + 1:]:
                    geom.append({'layer': l, 'kind': 'cross_lang', 'concept_a': concept, 'concept_b': concept,
                                 'lang_a': x, 'lang_b': y, 'cosine': cos(vec[(x, concept, l)], vec[(y, concept, l)])})
        for lang in LANGS:
            cs = list(CONCEPTS)
            for i, x in enumerate(cs):
                for y in cs[i + 1:]:
                    geom.append({'layer': l, 'kind': 'cross_concept', 'concept_a': x, 'concept_b': y,
                                 'lang_a': lang, 'lang_b': lang, 'cosine': cos(vec[(lang, x, l)], vec[(lang, y, l)])})
    for name, rows in (('vector_norms.csv', norms), ('vector_geometry.csv', geom), ('en_extraction_sanity.csv', sanity)):
        with (data / 'vectors' / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    worst = min(r['cos_upstream_vs_offset'] for r in sanity)
    manifest = {'text_model': TEXT_ID, 'dtype': 'float32', 'attn_implementation': model.config._attn_implementation,
                'transformers': __version__, 'upstream_revision': REV, 'layers': list(range(n_layers)),
                'variants': {k: sorted(v) for k, v in saved.items()},
                'en_method': 'upstream mean_shift_probing (AST-extracted from pinned source)',
                'zh_sw_method': 'upstream pooling, tokenizer-offset target alignment',
                'min_cos_en_upstream_vs_offset': worst}
    (data / 'vectors' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if worst < 0.99:
        print(f'WARNING: EN upstream vs offset alignment cosine as low as {worst:.4f}; inspect en_extraction_sanity.csv', flush=True)


if __name__ == '__main__':
    main()
