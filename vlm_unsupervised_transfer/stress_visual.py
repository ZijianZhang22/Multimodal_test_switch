#!/usr/bin/env python3
"""Validation-only shuffled-image diagnostic, not a label-based accuracy score.

Compare how often responses change when a DIFFERENT validation image replaces
its intended image. This measures sensitivity, not necessarily *correct* use
of visual evidence. A low or high flip rate alone does not establish grounding.
"""
import argparse,csv,json,random
from pathlib import Path
from types import SimpleNamespace
from experiment import load_eval,load_operator
from vlm import load_model,generate,question,letter


def find_swaps(rows,seed):
    rng=random.Random(seed)
    # Independent shuffled alternatives; each replacement must differ by
    # image file and SHA256 (duplicate scenes may occur across CV-Bench tasks).
    from prepare_data import digest
    hashes={r['image']:digest(r['image']) for r in rows}
    all_idx=list(range(len(rows)))
    result={}
    for i,row in enumerate(rows):
        alternatives=[j for j in all_idx if hashes[rows[j]['image']]!=hashes[row['image']]]
        if not alternatives:raise ValueError('No distinct validation image')
        result[i]=rows[rng.choice(alternatives)]['image']
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--pools',required=True)
    p.add_argument('--val',required=True)
    p.add_argument('--language',default='zh')
    p.add_argument('--model',default='google/paligemma2-3b-mix-448')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-relative-norm',type=float,default=.15)
    p.add_argument('--max-tokens',type=int,default=8)
    a=p.parse_args()
    root=Path(a.out)
    rows=load_eval(a.val,a.language)
    selected=json.loads((root/'selection_val.json').read_text())['selected']
    picks={
        'original':None,
        'A_text_visual_ot':selected['A/text_visual_ot'],
        'B_visual_gate':selected['B/text_visual_ot/confidence_visual'],
    }
    swaps=find_swaps(rows,a.seed)
    proc,model=load_model(a.model)
    result={}
    for name,chosen in picks.items():
        source_tag='val_baseline_'+a.language if chosen is None else (
            f'val_L{chosen["layer"]}_{chosen["method"]}_{chosen["mode"]}_a{chosen["strength"]:g}')
        reference=list(csv.DictReader((root/'predictions'/f'{source_tag}.csv').open(encoding='utf-8')))
        if [r['id'] for r in reference]!=[r['id'] for r in rows]:
            raise ValueError('Validation order differs from saved predictions')
        operator=None if chosen is None else load_operator(a,chosen['layer'],chosen['method'],
                                                           chosen['mode'],chosen['strength'],model)
        changed=0;valid_changed=0;shuffled=[]
        for i,row in enumerate(rows):
            replaced={**row,'image':swaps[i]}
            pred=letter(generate(model,proc,replaced,question(row,a.language),
                                 layer=chosen['layer'] if chosen else None,
                                 operator=operator,max_new_tokens=a.max_tokens))
            shuffled.append({'id':row['id'],'parsed':pred})
            if pred!=reference[i]['parsed']:changed+=1
            if bool(pred)!=bool(reference[i]['parsed']):valid_changed+=1
            if (i+1)%30==0:print('Shuffled-image',name,i+1,'/',len(rows),flush=True)
        result[name]={'n':len(rows),'prediction_flip_rate':changed/len(rows),
                      'validity_change_rate':valid_changed/len(rows),
                      'note':'Shuffled image has no valid original answer; do not interpret as standard accuracy'}
        (root/f'shuffled_image_{name}.json').write_text(json.dumps(shuffled,indent=2))
    (root/'visual_sensitivity_val.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)

if __name__=='__main__':main()