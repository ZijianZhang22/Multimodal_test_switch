"""Pure NumPy parts of an English semantic-retrieval bridge.

English snippets are from training OTHER images, never the query image.  They
are candidate language cues, not image evidence.  No VLM parameters are tuned.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np

PREFIX = 'answer en Answer the multiple choice question by only responding the letter of the correct answer. '
ARMS = ('baseline_zh', 'baseline_en', 'pairs_only', 'text_only', 'joint', 'joint_caption', 'random')
MAP_ARMS = ('pairs_only', 'text_only', 'joint', 'joint_caption')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def unit(x):
    a = np.asarray(x, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def map_vectors(source_z, target_z, checkpoint):
    """Match the existing Rosetta3 preprocessing: centered rows -> L2 -> W.

    These arguments are *already PCA projected + row-normalized* (like z in
    visual_anchor/run_rosetta3.py).  Do NOT rotate decoder hidden states.
    """
    if source_z.ndim != 2 or target_z.ndim != 2:
        raise ValueError('Rosetta bridge accepts 2D sentence embeddings only')
    w = np.asarray(checkpoint['W'])
    mx, my = np.asarray(checkpoint['mean_x']), np.asarray(checkpoint['mean_y'])
    d = source_z.shape[1]
    if (w.shape != (d, d) or mx.shape != (1, d) or my.shape != (1, d)
            or target_z.shape[1] != d):
        raise ValueError('Checkpoint/PCA dimensions or means do not match')
    if not all(np.isfinite(z).all() for z in (w, mx, my, source_z, target_z)):
        raise ValueError('Non-finite embedding or map')
    if not np.allclose(w.T @ w, np.eye(d), atol=0.025):
        raise ValueError('Rosetta3 W is not approximately orthogonal')
    return unit(unit(source_z - mx) @ w), unit(target_z - my)


def rankings(queries, bank, k):
    if not (1 <= k <= len(bank)):
        raise ValueError('k must be from 1 to bank size')
    sim = unit(queries) @ unit(bank).T
    # Stable deterministic tie-breaking by the original bank row order.
    ids = np.argsort(-sim, axis=1, kind='stable')[:, :k]
    scores = np.take_along_axis(sim, ids, axis=1)
    return ids, scores


def random_rankings(n_queries, n_bank, k, ids, seed):
    """Stable per-question random control, independent of row processing order."""
    if not 1 <= k <= n_bank:
        raise ValueError('Invalid random retrieval size')
    selections = []
    for identifier in ids:
        seed_hash = hashlib.sha256(f'{seed}|{identifier}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_hash[:8], 'little'))
        selections.append(rng.choice(n_bank, size=k, replace=False))
    if len(selections) != n_queries:
        raise ValueError('Question ID count mismatch')
    return np.array(selections, dtype=int)


def extract_question_for_embedding(full_prompt):
    """Extract the bilingual benchmark's question/hint, excluding MC choices.

    In MMBench prompts, the first line is the instruction; optional Hint and
    the question follow, then (A) ... (F) options. The full prompt is still
    passed unchanged to the VLM to preserve exact answer choices.
    """
    lines = str(full_prompt).splitlines()
    lines = lines[1:] if len(lines) > 1 else lines
    stem = []
    for line in lines:
        if re.match(r'^\s*\([A-Fa-f]\)\s', line):
            break
        if line.strip():
            stem.append(line.strip())
    result = ' '.join(stem).strip()
    if not result:
        raise ValueError('Empty question stem after stripping MMBench instructions/options')
    return result


def parse_flickr_captions(path):
    captions = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            token, caption = line.strip().split(maxsplit=1)
            image, index = token.rsplit('#', 1)
            if int(index) == 0:
                stem = Path(image).stem
                if stem in captions:
                    raise ValueError(f'Duplicate bank caption ID: {stem}')
                captions[stem] = caption.strip()
    return captions


def bank_captions(bank_ids, captions):
    unique = [str(x) for x in bank_ids]
    if len(set(unique)) != len(unique):
        raise ValueError('Duplicate image IDs in retrieval bank')
    missing = [x for x in unique if x not in captions]
    if missing:
        raise ValueError(f'Missing raw English caption for bank IDs: {missing[:3]}')
    return [captions[x] for x in unique]


def english_prompt(full_zh, captions):
    """Only linguistic hints; explicitly disallow treating other images as facts."""
    if not captions:
        raise ValueError('At least one retrieved caption is required')
    clean = [re.sub(r'\s+', ' ', c).strip()[:240] for c in captions]
    lines = '\n'.join(f'  {i}. {c}' for i, c in enumerate(clean, 1))
    return (PREFIX + full_zh + '\n\n'
            'Potentially related English descriptions retrieved from OTHER training images:\n'
            + lines + '\n'
            'These are not descriptions of the current image. Use them ONLY as language hints; '
            'base the answer on the actual image and question. Output one option letter.')


def build_prompt(row, arm, snippets):
    if arm == 'baseline_en':
        return PREFIX + row['questions']['en']
    if arm == 'baseline_zh':
        return PREFIX + row['questions']['zh']
    if arm not in ARMS:
        raise ValueError(f'Unknown arm {arm}')
    return english_prompt(row['questions']['zh'], snippets)


def image_cluster_bootstrap(rows, baseline, compared, repeats=1000, seed=42):
    """Paired comparisons resample unique images, not individual questions."""
    b = np.array([int(baseline[r['id']]['correct']) for r in rows])
    c = np.array([int(compared[r['id']]['correct']) for r in rows])
    delta = c - b
    by_image = {}
    for i, row in enumerate(rows):
        by_image.setdefault(row['image_pixel_sha256'], []).append(i)
    chunks = list(by_image.values())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(repeats):
        sample = rng.integers(len(chunks), size=len(chunks))
        indices = [q for idx in sample for q in chunks[int(idx)]]
        boots.append(100 * np.mean(delta[indices]))
    return {'n': len(rows), 'unique_images': len(chunks),
            'baseline_accuracy': float(b.mean()), 'accuracy': float(c.mean()),
            'delta_pp': 100 * float(delta.mean()),
            'ci95_pp': np.quantile(boots, [.025, .975]).tolist(),
            'repaired': int((delta > 0).sum()), 'harmed': int((delta < 0).sum())}