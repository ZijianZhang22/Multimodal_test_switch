"""Dependency-free dataset, parsing and provenance helpers."""
import hashlib
import json
import random
import re
from pathlib import Path

TEXT_MODEL = 'google/gemma-2-2b'
VLM_MODEL = 'google/paligemma2-3b-mix-448'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def normalize(row):
    """Only WorldCuisines v1.1 task1, English, no-context (prompt_type=1)."""
    if row['lang'] != 'en' or int(row['prompt_type']) != 1:
        return None
    options = row['options']
    answer = row['mcq_answer_index']  # v1.1 is ZERO based; v1.0 is different!
    if not isinstance(options, list) or len(options) != 5 or not all(isinstance(x, str) and x.strip() for x in options):
        raise ValueError('Expected five string options in WorldCuisines v1.1')
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 5:
        raise ValueError('Invalid zero-based answer index')
    if not row['question'].strip() or not row['image_url']:
        raise ValueError('Empty question/image URL')
    return dict(id=str(row['qa_id']), food_id=str(row['food_id']),
                question=row['question'], options=options, answer='ABCDE'[answer],
                image_url=row['image_url'], source_image_path=row.get('image_path', ''))


def partition(rows, n_val, n_test, seed):
    """One question/image, food-disjoint local splits; reject ambiguous image IDs."""
    if n_val < 1 or n_test < 1:
        raise ValueError('Need positive validation and test sizes')
    by_image = {}
    for row in sorted(rows, key=lambda r: (r['food_id'], r['image_url'], r['id'])):
        previous = by_image.get(row['image_url'])
        if previous and previous['food_id'] != row['food_id']:
            raise ValueError('Same image assigned to multiple food IDs; audit source')
        by_image.setdefault(row['image_url'], row)
    groups = {}
    for r in by_image.values():
        groups.setdefault(r['food_id'], []).append(r)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    result, cursor = [], 0
    for split, count in [('val', n_val), ('test', n_test)]:
        selected = []
        while len(selected) < count and cursor < len(ids):
            selected.extend(groups[ids[cursor]])
            cursor += 1
        if len(selected) < count:
            raise ValueError(f'Insufficient unique images/dishes for {split}: {len(selected)} < {count}')
        result.extend(dict(r, split=split) for r in selected[:count])
    return result


def parse_answer(text, options):
    """Full-string matching: never infer a letter from arbitrary prose."""
    s = text.strip()
    m = re.fullmatch(r'(?:(?:answer|option|choice)\s*[:：]?\s*)?\(?([A-E])\)?[.!]?', s, re.I)
    if m:
        return m.group(1).upper()
    matches = [i for i, option in enumerate(options) if s.casefold() == option.strip().casefold()]
    return 'ABCDE'[matches[0]] if len(matches) == 1 else ''


def make_prompt(row, food_prompt=False):
    hint = ('Focus on visible ingredients, preparation, texture, and shape to identify the dish. '
            if food_prompt else '')
    choices = '\n'.join(f'{letter}. {name}' for letter, name in zip('ABCDE', row['options']))
    return 'answer en ' + hint + row['question'] + '\n' + choices + '\nRespond with only the correct option letter (A-E).'
