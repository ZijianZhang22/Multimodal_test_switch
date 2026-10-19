#!/usr/bin/env python3
"""Build the upstream-aligned CV-Bench cohort + EN/ZH/SW prompts + anchor corpus.

Split  : identical to upstream utils.load_cvbench_dataset(subtask, split_type, test_size=150, seed=42).
Prompt : EN = CV-Bench `prompt` field verbatim (what upstream feeds the model).
         ZH/SW = same structure, question/choices via deterministic templates + curated lexicon
         (reproduces all 840 GPT-curated rows exactly; see test_parity.py).
Tune   : the first N_TUNE rows of upstream `train` per task (0 = the whole train split, as upstream).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from common import (CONCEPTS, PROTOCOL_VERSION, LANGS, PKG, TASKS, UPSTREAM, Translator, jsonl_write,
                    render_prompt, split_prompt, upstream_split)

SHARED = PKG.parent / 'gemma3_cross_lingual_steering'


def build_anchors(out: Path):
    sys.path.insert(0, str(UPSTREAM / 'get_features'))
    sys.path.insert(0, str(SHARED))
    sys.path.insert(0, str(SHARED / 'cvbench_extension'))
    from taxonomy_sentences import generate_taxonomy_sentences
    from make_data import ZH
    from anchors import ZH_EXTRA, SW
    for concept, taxonomy in CONCEPTS.items():
        sets = {'en': generate_taxonomy_sentences(taxonomy),
                'zh': ZH[concept] if concept in ZH else ZH_EXTRA[concept],
                'sw': SW[concept]}
        for lang, pairs in sets.items():
            if len(pairs) != 20:
                raise ValueError(f'{concept}/{lang}: expected 20 anchors')
            if lang != 'en':
                for s, t in pairs:
                    if s.count(t) != 1:
                        raise ValueError(f'Ambiguous target {lang}/{concept}: {s!r}/{t!r}')
            jsonl_write(out / 'corpus' / f'{concept}_{lang}.jsonl',
                        [{'id': i, 'sentence': s, 'target': t} for i, (s, t) in enumerate(pairs)])


def make_fallback(model_id):
    """Lazy NLLB, used ONLY for out-of-lexicon terms or unknown templates."""
    state = {}

    def fallback(texts, lang):
        if 'tr' not in state:
            import importlib.util
            sys.path.insert(0, str(SHARED))
            sys.path.insert(0, str(SHARED / 'cvbench_extension'))
            spec = importlib.util.spec_from_file_location('shared_prepare', SHARED / 'cvbench_extension' / 'prepare.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            state['tr'] = mod.NLLBTranslator(model_id)
        return state['tr'].translate(texts, lang)
    return fallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--test-size', type=int, default=150)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-tune', type=int, default=150, help='per task; 0 = full upstream train split')
    p.add_argument('--strict-lexicon', action='store_true', help='fail instead of NLLB for unknown terms')
    p.add_argument('--mt-model', default='facebook/nllb-200-distilled-600M')
    a = p.parse_args()
    data = a.data
    data.mkdir(parents=True, exist_ok=True)
    build_anchors(data)

    from datasets import load_dataset
    ds = load_dataset('nyu-visionx/CV-Bench', split='test')
    tasks_col = [str(t) for t in ds['task']]
    tr = Translator(None if a.strict_lexicon else make_fallback(a.mt_model))
    samples, audit = [], []
    for task in TASKS:
        rows = [i for i, t in enumerate(tasks_col) if t.lower() == task]   # upstream: dataset order
        train_pos, test_pos = upstream_split(len(rows), a.test_size, a.seed)
        tune_pos = train_pos if a.n_tune == 0 else train_pos[:a.n_tune]
        role = {**{q: 'train' for q in train_pos}, **{q: 'test' for q in test_pos}}
        keep = sorted(set(tune_pos) | set(test_pos))
        print(f'{task}: {len(rows)} rows | upstream train {len(train_pos)} / test {len(test_pos)} | tune {len(tune_pos)}', flush=True)
        for pos in keep:
            row = ds[rows[pos]]
            idx = int(row['idx'])
            choices = [str(c) for c in row['choices']]
            question, prompt_en = str(row['question']), str(row['prompt'])
            image_rel = f'images/{task}_{idx:06d}.png'
            dest = data / image_rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                row['image'].save(dest)            # lossless; mode kept as upstream passes it
            connector = split_prompt(prompt_en, question, choices)
            prompts, choices_i18n, provenance = {'en': prompt_en}, {'en': choices}, {}
            for lang in ('zh', 'sw'):
                q_l, src_q = tr.question(task, question, lang)
                ch = [tr.choice(task, c, lang) for c in choices]
                choices_i18n[lang] = [c for c, _ in ch]
                prompts[lang] = render_prompt(q_l, choices_i18n[lang],
                                              connector if connector is not None else ' Select from the following choices.\n',
                                              lang)
                provenance[lang] = {'question': src_q, 'choices': sorted({s for _, s in ch}),
                                    'prompt_structure': 'cvbench' if connector is not None else 'rebuilt'}
            split = 'test' if role[pos] == 'test' else 'tune'
            samples.append({'id': f'cv_{idx}', 'idx': idx, 'task': task, 'split': split,
                            'upstream_split': role[pos], 'image': image_rel, 'source': row.get('source'),
                            'answer': str(row['answer']), 'question': question, 'choices': choices,
                            'choices_i18n': choices_i18n, 'prompts': prompts, 'provenance': provenance})
            audit.append({'id': f'cv_{idx}', 'task': task, 'split': split, 'answer': row['answer'],
                          'zh_source': provenance['zh']['question'], 'sw_source': provenance['sw']['question'],
                          **{f'prompt_{l}': prompts[l] for l in LANGS}})
    jsonl_write(data / 'visual_eval.jsonl', samples)
    with (data / 'translation_audit.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    (data / 'missing_terms.txt').write_text('\n'.join(sorted(tr.missing_terms)) + '\n', encoding='utf-8')
    (data / 'untemplated_questions.txt').write_text('\n'.join(sorted(tr.untemplated)) + '\n', encoding='utf-8')
    manifest = {'protocol_version': PROTOCOL_VERSION,
                'source_dataset': 'nyu-visionx/CV-Bench', 'original_split': 'test',
                'split_protocol': 'upstream utils.load_cvbench_dataset (pandas sample frac=1, random_state=seed; test=last test_size)',
                'seed': a.seed, 'test_size': a.test_size, 'n_tune': a.n_tune,
                'counts': dict(Counter(f"{s['task']}_{s['split']}" for s in samples)),
                'translation': 'templates + curated lexicon; NLLB only for missing terms/templates',
                'missing_terms': len(tr.missing_terms), 'untemplated_questions': len(tr.untemplated),
                'prompt_provenance': dict(Counter(f"{l}:{s['provenance'][l]['question']}" for s in samples for l in ('zh', 'sw'))),
                'translation_verified_by_native_speakers': False}
    (data / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if tr.missing_terms:
        print(f'NOTE: {len(tr.missing_terms)} object terms were machine-translated; review {data}/missing_terms.txt', flush=True)


if __name__ == '__main__':
    main()
