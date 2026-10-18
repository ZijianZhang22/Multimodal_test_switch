#!/usr/bin/env python3
"""CPU-only checks (no model / dataset download). Run before any GPU job.

1. normalize_answer / idx_to_letter_choice are identical to the pinned upstream utils.py
   (functions AST-extracted from upstream and compared on many strings).
2. The split reproduces upstream load_cvbench_dataset's pandas logic.
3. The template translator reproduces all 840 curated ZH/SW questions + choices exactly.
4. Prompt construction: EN equals upstream's prefix + instruction + prompt.
"""
import ast
import random
import string

from common import (UPSTREAM, Translator, build_prompt, load_curated, normalize_answer, render_prompt,
                    score, split_prompt, upstream_split)


def upstream_funcs():
    import warnings
    src = (UPSTREAM / 'utils.py').read_text()
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    from typing import List
    ns = {'List': List}
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name in ('normalize_answer', 'idx_to_letter_choice'):
            exec(compile(ast.get_source_segment(src, n), 'upstream_utils', 'exec'), ns)
    return ns


def test_scoring():
    ns = upstream_funcs()
    up = ns['normalize_answer']
    rng = random.Random(0)
    probes = ['A', 'b', '(C)', 'D.', 'answer: E', 'Answer: (B)', '2', ' left ', 'Right.', 'a cat', 'e.g. A',
              'B because', '', '  ', 'truck', 'The answer is A', '(A) 3', 'f', 'h', 'z']
    probes += [''.join(rng.choice(string.ascii_letters + ' ().:') for _ in range(rng.randint(0, 6))) for _ in range(3000)]
    choices_sets = [['1', '2', '3', '4'], ['left', 'right'], ['pedestrian', 'truck'], ['a', 'B']]
    for ch in choices_sets:
        for s in probes:
            assert up(s, ch) == normalize_answer(s, ch), (s, ch)
    s = score('2', '(B)', ['1', '2'], ['1', '2'], 'en')
    assert s == {'normalized': 'b', 'valid': 1, 'correct': 1}
    assert score('a cat', '(A)', ['x', 'y'], ['x', 'y'], 'en')['valid'] == 0       # legacy bug fixed
    assert score('右边', '(B)', ['left', 'right'], ['左边', '右边'], 'zh')['correct'] == 1
    assert score('right', '(B)', ['left', 'right'], ['左边', '右边'], 'zh')['correct'] == 1
    assert score('答案：B', '(B)', ['left', 'right'], ['左边', '右边'], 'zh')['correct'] == 1


def test_split():
    import pandas as pd
    for n in (788, 650, 600, 37):
        df = pd.DataFrame({'pos': range(n)})                       # upstream code path
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        ts = 150 if 150 < n else n // 2
        tr, te = upstream_split(n)
        assert te == list(shuffled.iloc[-ts:]['pos']) and tr == list(shuffled.iloc[:-ts]['pos'])
        assert not set(tr) & set(te) and len(tr) + len(te) == n


def test_translation():
    tr = Translator(fallback=None)
    n = 0
    for r in load_curated().values():
        for lang in ('zh', 'sw'):
            q, src = tr.question(r['task'], r['question'], lang)
            assert src == 'template' and q == r['translated_questions'][lang], (r['id'], lang, q)
            assert [tr.choice(r['task'], c, lang)[0] for c in r['choices']] == r['translated_choices'][lang]
            n += 1
    assert n == 1680 and not tr.missing_terms


def test_prompt():
    q, ch = 'How many cars are in the image?', ['3', '2', '1']
    prompt = q + ' Select from the following choices.\n(A) 3\n(B) 2\n(C) 1'
    conn = split_prompt(prompt, q, ch)
    assert conn == ' Select from the following choices.\n'
    zh = render_prompt('图中汽车的数量是多少？', ch, conn, 'zh')
    assert zh == '图中汽车的数量是多少？请从以下选项中选择。\n(A) 3\n(B) 2\n(C) 1'
    s = {'prompts': {'en': prompt, 'zh': zh, 'sw': 'x'}}
    up = 'answer en ' + 'Answer the multiple choice question by only responding the letter of the correct answer. ' + prompt
    assert build_prompt(s, 'en', 'native') == up == build_prompt(s, 'en', 'upstream_en')
    assert build_prompt(s, 'zh', 'native').startswith('answer zh 请只回答')
    assert build_prompt(s, 'zh', 'upstream_en').startswith('answer en Answer the multiple')


if __name__ == '__main__':
    for t in (test_scoring, test_split, test_translation, test_prompt):
        t()
        print('PASS', t.__name__)
