#!/usr/bin/env python3
"""Prepare a paired Idis-perception x GSM8K/MATH-500 pilot, downloading only selected images."""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

IDIS = 'Vail-2000/Idis'
MATH500 = 'HuggingFaceH4/MATH-500'
CLASS_MAP = {'wheeled vehicle':'vehicle', 'musical instrument':'instrument'}
BLOCKED_MATH = ('[asy]', '\\includegraphics', '[asy]', 'shown below', 'diagram below', 'the figure', 'in the figure', 'as shown', 'the graph below')


def listing(api, path):
    return list(api.list_repo_tree(repo_id=IDIS, repo_type='dataset', path_in_repo=path, recursive=False))


def is_folder(entry):
    return entry.__class__.__name__.lower().endswith('folder') or str(type(entry)).lower().find('repofolder') >= 0


def dirs(api, path):
    return sorted(x.path for x in listing(api, path) if is_folder(x))


def files(api, path):
    return sorted(x.path for x in listing(api, path) if not is_folder(x) and str(x.path).lower().endswith(('.png', '.jpg', '.jpeg')))


def load_questions(kind, seed, count, min_level):
    rng = random.Random(seed + 101)
    if kind == 'gsm8k':
        from datasets import load_dataset
        ds = load_dataset('openai/gsm8k', 'main', split='test')
        pool = []
        for i, row in enumerate(ds):
            if '####' not in row['answer']:
                continue
            gold = row['answer'].split('####')[-1].strip().replace(',', '')
            if not re.fullmatch(r'-?\d+(?:\.\d+)?', gold):
                continue
            pool.append({'id':f'gsm8k-{i}', 'question':row['question'], 'gold':gold, 'benchmark':'gsm8k'})
    else:
        path = hf_hub_download(repo_id=MATH500, repo_type='dataset', filename='test.jsonl')
        pool = []
        for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
            x = json.loads(line)
            question = x['problem']
            if int(x.get('level', 0)) < min_level or any(s in question.lower() for s in BLOCKED_MATH):
                continue
            pool.append({'id':x.get('unique_id', f'math500-{i}'), 'question':question,
                         'gold':str(x['answer']).strip(), 'benchmark':'math500',
                         'level':x.get('level'), 'subject':x.get('subject')})
    rng.shuffle(pool)
    if len(pool) < count:
        raise RuntimeError(f'Not enough usable {kind} items: {len(pool)} < {count}')
    return pool[:count]


def find_subtree(api, root):
    """Try official published layout and legacy tree layout."""
    candidates = [root + '/Idis-perception/visual_distractor', root + '/Idis-perception',
                  'Idis-perception/visual_distractor', 'Idis-perception']
    for p in candidates:
        p = p.strip('/')
        try:
            children = dirs(api, p)
        except Exception:
            continue
        if p.endswith('Idis-perception'):
            v = [s for s in children if s.rsplit('/', 1)[-1] == 'visual_distractor']
            if v:
                return v[0], dirs(api, v[0])
        if children and any(re.match(r'^\d\d_', c.rsplit('/',1)[-1]) for c in children):
            return p, children
    raise RuntimeError('Could not find Idis visual_distractor/<class> on Hugging Face; inspect dataset repo tree.')


def collect_for_class(api, class_path, n_distractors, variants):
    """Return filename stem -> {variant:path}, requiring all variants for every selected stem."""
    numeric = dirs(api, class_path)
    countdir = next((d for d in numeric if d.rsplit('/',1)[-1] in (str(n_distractors), f'n{n_distractors}')), None)
    if not countdir:
        return {}
    folders = {d.rsplit('/',1)[-1]: d for d in dirs(api, countdir)}
    lookup = {}
    for v in variants:
        if v not in folders:
            return {}
        lookup[v] = {Path(p).stem:p for p in files(api, folders[v])}
    stems = set.intersection(*(set(lookup[v]) for v in variants))
    return {stem: {v:lookup[v][stem] for v in variants} for stem in sorted(stems)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pilot_data')
    ap.add_argument('--benchmark', choices=['gsm8k','math500'], default='gsm8k')
    ap.add_argument('--pairs', type=int, default=8, help='Number of UNIQUE visual originals; each variant reuses the same text Q2')
    ap.add_argument('--classes', type=int, default=2, help='Number of target classes to sample')
    ap.add_argument('--variants', default='conflicting,irrelevant')
    ap.add_argument('--n-distractors', type=int, default=4)
    ap.add_argument('--math-min-level', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--original-root', default='', help='Optional ImageNet-9 original/val dir, with 00_dog/...JPEG folders')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    variants = [v.strip() for v in args.variants.split(',')]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    path_root, classes = find_subtree(api, '')
    chosen = [c for c in classes if re.match(r'^\d\d_',c.rsplit('/',1)[-1])]
    if not chosen:
        raise RuntimeError('No class dirs found below ' + path_root)
    chosen = chosen[:args.classes]
    per_class = (args.pairs + len(chosen)-1)//len(chosen)
    selected = []
    for cpath in chosen:
        options = list(collect_for_class(api, cpath, args.n_distractors, variants).items())
        rng.shuffle(options)
        for stem, variant_paths in options[:per_class]:
            class_dir = cpath.rsplit('/',1)[-1]
            full_class = class_dir.split('_',1)[1]
            label = CLASS_MAP.get(full_class, full_class)
            selected.append({'id':class_dir+'/'+stem, 'class_dir':class_dir, 'label':label,
                             'stem':stem, 'variant_paths':variant_paths})
    if len(selected) < args.pairs:
        raise RuntimeError(f'Only found {len(selected)} images for requested {args.pairs}; reduce --pairs or increase --classes')
    selected = selected[:args.pairs]
    questions = load_questions(args.benchmark, args.seed, len(selected), args.math_min_level)
    rows = []
    for j, (s,q) in enumerate(zip(selected, questions)):
        variants_local = {}
        for variant, remote in s['variant_paths'].items():
            downloaded = hf_hub_download(repo_id=IDIS, repo_type='dataset', filename=remote,
                                         local_dir=str(out/'hf_dataset'))
            variants_local[variant] = str(Path(downloaded).resolve())
        # Optional official ImageNet-9 original directory, matched by stem.
        if args.original_root:
            base = Path(args.original_root).expanduser()
            possibilities = [base/s['class_dir']/(s['stem']+extension) for extension in ('.JPEG','.jpg','.jpeg','.png')]
            original = next((str(p.resolve()) for p in possibilities if p.exists()), None)
            if original:
                variants_local['original'] = original
        rows.append({'id': s['id'], 'q1_label': s['label'], 'q2_id': q['id'],
                     'q2_question': q['question'], 'q2_gold':q['gold'], 'q2_benchmark':args.benchmark,
                     'q2_level':q.get('level'), 'q2_subject':q.get('subject'), 'images':variants_local})
        print(f'[{j+1}/{len(selected)}] {s["id"]} -> {q["id"]}: {list(variants_local)}', flush=True)
    with (out/'pairs.jsonl').open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'metadata.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print(f'PREPARED {len(rows)} matched pairs in {out}/pairs.jsonl')

if __name__ == '__main__': main()
