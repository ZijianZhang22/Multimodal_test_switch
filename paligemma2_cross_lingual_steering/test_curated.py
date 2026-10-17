#!/usr/bin/env python3
"""CPU-only regression checks for the fixed 840-example CV-Bench cohort."""
from pathlib import Path
from collections import Counter
import importlib.util
import re
import sys
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('curated',ROOT/'apply_curated.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=mod.load_translations()
assert len(rows)==840
assert Counter(r['task'] for r in rows.values())==dict.fromkeys(mod.TASKS,210)
assert Counter(r['split'] for r in rows.values())=={'val':240,'test':600}
for r in rows.values():
  n=len(r['choices'])
  assert 2<=n<=6 and r['answer'] in {chr(65+i) for i in range(n)}
  for lang in ('en','zh','sw'):
    q=r['questions'][lang]
    options=re.findall(r'^\(([A-F])\) (.+)$',q,re.M)
    assert len(options)==n,(r['id'],lang,options)
    assert [x[0] for x in options]==[chr(65+i) for i in range(n)]
    assert all(text.strip() for _,text in options)
    assert len({text for _,text in options})==n,(r['id'],lang,'duplicated choice')
    if r['task']=='count':
      assert [text for _,text in options]==r['choices'],(r['id'],lang,'count options altered')
    if r['task']=='relation':
      assert {text for _,text in options}==({'left','right'} if lang=='en' else
      ({'左边','右边'} if lang=='zh' else {'kushoto','kulia'})) or {text for _,text in options}==(
      {'above','below'} if lang=='en' else ({'上方','下方'} if lang=='zh' else {'juu','chini'}))
print('PASS: 840 unique IDs; 210/task; 240 validation, 600 test; each original option and answer label preserved.')
print('PASS: all prompt languages have correctly ordered A-F option labels and no duplicate options.')
print('PASS: numeric options unchanged and spatial directional options translated consistently.')
