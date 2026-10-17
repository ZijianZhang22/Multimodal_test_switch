#!/usr/bin/env python3
"""Validate parsing and paired scoring without a GPU or web access."""
import json
import tempfile
from pathlib import Path
from score import tag, q2_result, q1_result, main

assert q2_result('Reasoning ... <q2>1,234</q2>', '1234', 'gsm8k')[1] is True
assert q2_result('Reasoning ... <q2>87</q2>', '88', 'gsm8k')[1] is False
assert q2_result('more math <q2>\\frac{1}{2}</q2>', '0.5', 'math500')[1] is not False  # may be unscored offline
assert q1_result('... <answer>Dog</answer>', 'dog', 'q1_only')[1]
assert q1_result('... <q1>Vehicle</q1>', 'vehicle', 'joint_vt')[1]
with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    rows=[]
    def add(variant,cond,pred,gold='42'):
        rows.append({'id':'00_dog/sample','variant':variant,'condition':cond,'rep':0,
                     'q1_gold':'dog','q2_gold':gold,'q2_id':'g-0','q2_benchmark':'gsm8k',
                     'q2_level':None,'pred':pred,'num_new_tokens':15,'hit_max_token':False})
    add('none','q2_only','<q2>42</q2>')
    add('conflicting','image_q2','<q2>42</q2>')
    add('conflicting','q1_only','<answer>Dog</answer>')
    add('conflicting','joint_vt','<q1>dog</q1> <q2>41</q2>')
    add('conflicting','joint_tv','<q2>42</q2> <q1>dog</q1>')
    p=root/'mock.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    import subprocess,sys
    subprocess.run([sys.executable,str(Path(__file__).parent/'score.py'),'--input',str(p),'--out',str(root/'results')],check=True)
    import csv
    with (root/'results/candidate_interference.csv').open() as f:
        c=list(csv.DictReader(f))
        assert len(c)==1 and c[0]['joint']=='joint_vt'
print('SMOKE TEST OK')
