#!/usr/bin/env python3
"""Shared protocol for the upstream-aligned ("parity") PaliGemma2 pipeline.

Everything that decides *what is measured* lives here so prepare / evaluate /
analyze cannot drift apart:

* CV-Bench split : same logic as upstream utils.load_cvbench_dataset
                   (pandas .sample(frac=1, random_state=42); test = last 150 per task).
* Prompt         : upstream evaluate_cvbench: prefix + instruction + CV-Bench `prompt`.
* Scoring        : upstream utils.normalize_answer, copied verbatim.
* Translation    : deterministic templates + curated lexicon; sentence-level MT is
                   used only if a question matches no known CV-Bench template.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                                   # paligemma2_cross_lingual_steering/
UPSTREAM = PKG / '.upstream'                        # Woodygan/TextSteersVision checkout
REV = 'ce1c0f0cffed9b859901ce347f9d2b869a512178'
MODEL_ID = 'google/paligemma2-3b-mix-448'
TEXT_ID = 'google/gemma-2-2b'                       # upstream MODEL_TO_BASE_MODEL[...3b...]['meanshift']
N_TEXT_LAYERS = 26                                  # upstream DEFAULT_LAYERS['gemma-2-2b']

TASKS = ('count', 'relation', 'depth', 'distance')
LANGS = ('en', 'zh', 'sw')
# local short name -> upstream taxonomy key in taxonomy_sentences.py
CONCEPTS = {'spatial': 'spatial_relationship', 'counting': 'counting',
            'entity': 'entity', 'attribute': 'attribute'}
# The legacy pipeline's fixed task->concept mapping (kept as an option only).
ALIGNED_CONCEPT = {'count': 'counting', 'relation': 'spatial', 'depth': 'spatial', 'distance': 'spatial'}

FIELDS = ['id', 'task', 'split', 'prompt_lang', 'style', 'concept', 'variant', 'layer', 'alpha',
          'response', 'normalized', 'valid', 'correct']

# ---------------------------------------------------------------- prompt
UPSTREAM_INSTRUCTION = 'Answer the multiple choice question by only responding the letter of the correct answer. '
INSTRUCTION = {
    'en': UPSTREAM_INSTRUCTION,
    'zh': '请只回答正确答案对应的字母来完成这道选择题。',
    'sw': 'Jibu swali hili la kuchagua kwa kutaja herufi ya jibu sahihi pekee. ',
}
# CV-Bench `prompt` = question + connector + "\n(A) ...". Known connector translations:
CONNECTOR = {
    ' Select from the following choices.': {'zh': '请从以下选项中选择。',
                                             'sw': ' Chagua kutoka kwenye chaguo zifuatazo.'},
}
PROMPT_STYLES = ('native', 'upstream_en')


def build_prompt(sample: dict, lang: str, style: str) -> str:
    """native      : 'answer {lang} ' + instruction in {lang} + CV-Bench prompt in {lang}
       upstream_en : 'answer en ' + English instruction + prompt in {lang}
                     (the legacy mixed-language setting, kept for ablation)
    English is byte-identical to upstream in both styles."""
    if style not in PROMPT_STYLES:
        raise ValueError(style)
    body = sample['prompts'][lang]
    if lang == 'en' or style == 'upstream_en':
        return 'answer en ' + UPSTREAM_INSTRUCTION + body
    return f'answer {lang} ' + INSTRUCTION[lang] + body


# ---------------------------------------------------------------- scoring
def idx_to_letter_choice(idx: int) -> str:          # upstream utils.py
    return chr(65 + idx)


def normalize_answer(answer: str, choices) -> str:  # upstream utils.py, verbatim logic
    answer = answer.lower().strip()
    if answer.startswith("answer:"):
        answer = answer[len("answer:"):].strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    if len(answer) >= 3 and answer[0] == '(' and answer[2] == ')':
        return answer[1].lower()
    if len(answer) == 1 and answer.isalpha():
        return answer.lower()
    if answer in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        return answer
    else:
        for idx, choice in enumerate(choices):
            if answer.lower() == choice.lower():
                return idx_to_letter_choice(idx).lower()
    return answer


# Language equivalents of upstream's "answer:" prefix stripping (symmetry across languages).
_PREFIX = {'zh': ('答案：', '答案:'), 'sw': ('jibu:',)}


def score(response: str, answer: str, choices_en, choices_prompt, prompt_lang: str) -> dict:
    """Upstream scoring. For EN this is exactly upstream. For ZH/SW the choice-text
    match uses the prompt-language choices, then the English ones (the model may
    answer in English)."""
    gold = normalize_answer(answer, choices_en)
    resp = (response or '').strip()
    if prompt_lang != 'en':
        low = resp.lower()
        for p in _PREFIX.get(prompt_lang, ()):
            if low.startswith(p):
                resp = resp[len(p):].strip()
                break
    norm = normalize_answer(resp, choices_prompt)
    if prompt_lang != 'en' and not (len(norm) == 1 and norm.isalpha()):
        alt = normalize_answer(resp, choices_en)
        if len(alt) == 1 and alt.isalpha():
            norm = alt
    valid = len(norm) == 1 and norm.isalpha() and (ord(norm) - 97) < len(choices_en)
    return {'normalized': norm, 'valid': int(valid), 'correct': int(norm == gold)}


# ---------------------------------------------------------------- split
def upstream_split(n_rows: int, test_size: int = 150, seed: int = 42):
    """Positions (into the task-filtered, dataset-ordered rows) of the upstream
    train and test subsets, reproducing load_cvbench_dataset()."""
    import pandas as pd
    df = pd.DataFrame({'pos': range(n_rows)})
    if test_size >= n_rows:
        test_size = n_rows // 2
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return [int(x) for x in shuffled['pos'].iloc[:-test_size]], [int(x) for x in shuffled['pos'].iloc[-test_size:]]


# ---------------------------------------------------------------- translation
def load_lexicon():
    lex, origin = {}, {}
    for path in (PKG / 'curated_questions' / 'objects.tsv', HERE / 'glossary_extra.tsv'):
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            en, zh, sw = line.split('\t')
            for form in en.split('|'):
                k = form.strip().lower()
                if k in lex:            # curated file wins; the extra file only fills gaps
                    continue
                lex[k] = {'zh': zh.strip(), 'sw': sw.strip()}
                origin[k] = path.name
    return lex, origin


def lookup(term: str, lex: dict):
    t = term.strip().lower()
    for c in (t, t + 's', t[:-1] if t.endswith('s') else None,
              t[:-2] if t.endswith('es') else None, t + 'es'):
        if c and c in lex:
            return lex[c]
    return None


RE = {
    'count': re.compile(r'How many (?P<x>.+?) are in the image\?'),
    'relation_box': re.compile(r'Considering the relative positions of the (?P<a>.+?) \(annotated by the red box\) and the (?P<b>.+?) in the image provided, where is the (?P=a) \(annotated by the red box\) located with respect to the (?P=b)\?'),
    'relation': re.compile(r'Considering the relative positions of the (?P<a>.+?) and the (?P<b>.+?) in the image provided, where is the (?P=a) located with respect to the (?P=b)\?'),
    'depth': re.compile(r'Which object is closer to the camera taking this photo, the (?P<a>.+?) \(highlighted by a red box\) or the (?P<b>.+?) \(highlighted by a blue box\)\?'),
    'distance': re.compile(r'Estimate the real-world distances between objects in this image\. Which object is closer to the (?P<r>.+?) \(highlighted by a red box\), the (?P<a>.+?) \(highlighted by a blue box\) or the (?P<b>.+?) \(highlighted by a green box\)\?'),
}
TEMPLATE = {
    'count': {'zh': '图中{x}的数量是多少？', 'sw': 'Je, kuna {x} ngapi kwenye picha?'},
    'relation_box': {'zh': '观察图中物体的相对位置：{a}（红框标注）位于{b}的哪一侧？',
                     'sw': 'Kwa kuzingatia nafasi za vitu kwenye picha, {a} (iliyowekwa alama kwa kisanduku chekundu) iko upande gani ukilinganisha na {b}?'},
    'relation': {'zh': '观察图中物体的相对位置：{a}位于{b}的哪一侧？',
                 'sw': 'Kwa kuzingatia nafasi za vitu kwenye picha, {a} iko upande gani ukilinganisha na {b}?'},
    'depth': {'zh': '哪个物体离拍摄这张照片的相机更近：{a}（红框标注）还是{b}（蓝框标注）？',
              'sw': 'Ni kipi kilicho karibu zaidi na kamera iliyopiga picha hii: {a} (kisanduku chekundu) au {b} (kisanduku cha bluu)?'},
    'distance': {'zh': '根据图中物体在真实空间中的距离判断：相对于{r}（红框标注），哪个物体更近——{a}（蓝框标注）还是{b}（绿框标注）？',
                 'sw': 'Kadiria umbali halisi kati ya vitu kwenye picha. Kipi kiko karibu zaidi na {r} (kisanduku chekundu): {a} (kisanduku cha bluu) au {b} (kisanduku cha kijani)?'},
}
DIRECTION = {'left': {'zh': '左边', 'sw': 'kushoto'}, 'right': {'zh': '右边', 'sw': 'kulia'},
             'above': {'zh': '上方', 'sw': 'juu'}, 'below': {'zh': '下方', 'sw': 'chini'}}


def _inverse(template):
    pat = re.escape(template)
    for slot in ('x', 'a', 'b', 'r'):
        pat = pat.replace(r'\{' + slot + r'\}', f'(?P<{slot}>.+?)')
    return re.compile(pat)


def curated_term_map():
    """(kind, lang, english term) -> translation, recovered from the 840 curated rows by
    inverting the templates. kind = 'count' (plural, count questions) or 'obj'."""
    inv = {k: {l: _inverse(v) for l, v in d.items()} for k, d in TEMPLATE.items()}
    out = {}
    for r in load_curated().values():
        t = r['task']
        for key in (['relation_box', 'relation'] if t == 'relation' else [t]):
            me = RE[key].fullmatch(r['question'])
            if me:
                break
        if not me:
            continue
        kind = 'count' if t == 'count' else 'obj'
        for lang in ('zh', 'sw'):
            mt = inv[key][lang].fullmatch(r['translated_questions'][lang])
            if mt:
                for slot, v in me.groupdict().items():
                    out.setdefault((kind, lang, v.lower()), mt.group(slot))
            if t in ('depth', 'distance'):
                for c, ct in zip(r['choices'], r['translated_choices'][lang]):
                    out.setdefault(('obj', lang, c.lower()), ct)
    return out


class Translator:
    """Template + lexicon translation. Unknown terms / templates go to `fallback`
    (callable: list[str], lang -> list[str]) and are recorded for audit."""

    def __init__(self, fallback=None):
        self.lex, self.origin = load_lexicon()
        self.curated = curated_term_map()
        self.fallback = fallback
        self.missing_terms, self.untemplated = set(), set()
        self.cache = {}

    def _mt(self, text, lang):
        if self.fallback is None:
            raise KeyError(text)
        if (text, lang) not in self.cache:
            self.cache[(text, lang)] = self.fallback([text], lang)[0]
        return self.cache[(text, lang)]

    def _term(self, term, lang, kind='obj'):
        exact = self.curated.get((kind, lang, term.strip().lower()))
        if exact:
            return exact, 'lexicon'
        hit = lookup(term, self.lex)
        if hit:
            return hit[lang], 'lexicon'
        self.missing_terms.add(term)
        return self._mt(term, lang), 'mt_term'

    def question(self, task, q, lang):
        for key in (['relation_box', 'relation'] if task == 'relation' else [task]):
            m = RE[key].fullmatch(q.strip())
            if m:
                parts, srcs = {}, set()
                for k, v in m.groupdict().items():
                    parts[k], s = self._term(v, lang, 'count' if task == 'count' else 'obj')
                    srcs.add(s)
                return TEMPLATE[key][lang].format(**parts), ('template' if srcs == {'lexicon'} else 'template+mt_term')
        self.untemplated.add(q)
        return self._mt(q, lang), 'mt_sentence'

    def choice(self, task, c, lang):
        if task == 'count' or c.strip().isdigit():
            return c, 'verbatim'
        if task == 'relation' and c.strip().lower() in DIRECTION:
            return DIRECTION[c.strip().lower()][lang], 'fixed'
        return self._term(c, lang)


def choices_block(choices):
    return '\n'.join(f'({idx_to_letter_choice(i)}) {c}' for i, c in enumerate(choices))


def split_prompt(prompt: str, question: str, choices):
    """CV-Bench prompt -> connector string between question and choice block, or None."""
    block = choices_block(choices)
    if prompt.startswith(question) and prompt.endswith(block):
        return prompt[len(question):len(prompt) - len(block)]
    return None


def render_prompt(question_l, choices_l, connector, lang):
    stripped = connector.rstrip('\n')
    tail = connector[len(stripped):]
    conn = CONNECTOR.get(stripped, {}).get(lang)
    if conn is None:                               # unknown connector: use the standard one
        conn = CONNECTOR[' Select from the following choices.'][lang]
    return question_l + conn + (tail or '\n') + choices_block(choices_l)


def jsonl_read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def jsonl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def load_curated():
    """The 840 GPT-curated rows (ids follow the legacy split)."""
    import base64
    import gzip
    rows = {}
    for t in TASKS:
        raw = (PKG / 'curated_questions' / f'{t}.jsonl.gz.b64').read_text(encoding='ascii')
        for line in gzip.decompress(base64.b64decode(''.join(raw.split()))).decode('utf-8').splitlines():
            if line:
                r = json.loads(line)
                rows[r['id']] = r
    return rows
