#!/usr/bin/env python3
"""Score paired Q1 and Q2, write CSV summaries and investigate induced errors."""
import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def tag(pred,name):
    matches=re.findall(rf'<{name}\s*>(.*?)</{name}\s*>',pred,flags=re.IGNORECASE|re.DOTALL)
    if matches: return matches[-1].strip()
    # fallback for a run truncated just after the final answer tag
    matches=re.findall(rf'<{name}\s*>([^\n<]{{1,100}})',pred,flags=re.IGNORECASE)
    return matches[-1].strip() if matches else None


def box_extract(s):
    pos=s.rfind('\\boxed{')
    if pos<0: return None
    i=pos+7; depth=1; out=[]
    while i<len(s):
        c=s[i]
        if c=='{':depth+=1
        if c=='}':
            depth-=1
            if depth==0:return ''.join(out)
        out.append(c);i+=1
    return None


def canonical(s):
    if s is None:return ''
    s=str(s).strip().replace('\u2212','-').replace('\u00a0',' ').replace('−','-')
    s=s.replace('\\left','').replace('\\right','').replace('$','').replace(',','')
    s=re.sub(r'\s+','',s)
    return s


def numeric(s):
    s=canonical(s).replace('\\%','%')
    s=re.sub(r'^(?:answer:|=)','',s,flags=re.I)
    if s.endswith('%'):s=s[:-1]  # GSM8K generally not percentages; preserve as raw numeric when gold is percent
    if re.fullmatch(r'[-+]?\d+(\.\d+)?',s):
        try:return Decimal(s)
        except InvalidOperation:pass
    if re.fullmatch(r'[-+]?\d+/\d+',s):
        try:
            a,b=s.split('/');return Decimal(a)/Decimal(b)
        except Exception:pass
    return None


def q2_result(pred, gold, benchmark):
    s=tag(pred,'q2')
    if s is None:
        s=box_extract(pred)
    if s is None:
        # Explicitly fail, never mine intermediate reasoning for a coincidental answer.
        return None,False,'missing_final_tag'
    if canonical(s)==canonical(gold):return s,True,'exact'
    a,b=numeric(s),numeric(gold)
    if a is not None and b is not None:return s,a==b,'numeric'
    if benchmark=='math500':
        try:
            from math_verify import parse,verify
            ok=bool(verify(parse('$'+gold+'$'),parse('$'+s+'$')))
            return s,ok,'math_verify'
        except Exception:
            return s,None,'needs_math_verify_or_manual_review'
    return s,False,'non_numeric_gsm8k'


def q1_result(pred, label, condition):
    s=tag(pred,'answer') if condition=='q1_only' else tag(pred,'q1')
    if s is None:return None,False
    v=re.sub(r'[^a-z ]','',s.strip().lower()).strip()
    if v=='wheeled vehicle':v='vehicle'
    if v=='musical instrument':v='instrument'
    return s,(v==label.lower().strip())


def csv_write(path,rows):
    rows=list(rows)
    if not rows:return
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def avg(values):
    return sum(values)/len(values) if values else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='pilot_results/predictions.jsonl')
    ap.add_argument('--out',default='pilot_results')
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for line in Path(args.input).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        r=json.loads(line)
        q1_text,q1_correct=(None,None)
        if r['condition'] in ('q1_only','joint_vt','joint_tv'):
            q1_text,q1_correct=q1_result(r['pred'],r['q1_gold'],r['condition'])
        q2_text,q2_correct,q2_note=(None,None,'not_applicable')
        if r['condition'] != 'q1_only':
            q2_text,q2_correct,q2_note=q2_result(r['pred'],r['q2_gold'],r['q2_benchmark'])
        records.append({k:r[k] for k in ('id','variant','condition','rep','q2_id','q1_gold','q2_gold')}
                       |{'q1_pred':q1_text,'q1_correct':q1_correct,'q2_pred':q2_text,
                         'q2_correct':q2_correct,'q2_scoring':q2_note,'tokens':r['num_new_tokens'],
                         'hit_max_token':r.get('hit_max_token',False)})
    csv_write(out/'scored.csv',records)
    grouped=defaultdict(list)
    for r in records:grouped[(r['variant'],r['condition'])].append(r)
    summaries=[]
    for (variant,cond),a in sorted(grouped.items()):
        validq1=[int(r['q1_correct']) for r in a if r['q1_correct'] is not None]
        validq2=[int(r['q2_correct']) for r in a if r['q2_correct'] is not None]
        summaries.append({'variant':variant,'condition':cond,'n':len(a),
                          'q1_accuracy':avg(validq1),'q2_accuracy':avg(validq2),
                          'q2_scoreable':len(validq2),'truncated':sum(bool(r['hit_max_token']) for r in a),
                          'mean_new_tokens':round(avg([r['tokens'] for r in a]),1)})
    csv_write(out/'summary.csv',summaries)
    lookup={(r['id'],r['variant'],r['condition'],r['rep']):r for r in records}
    candidates=[]
    print('Paired delta: negative means joint Q2 performs worse than image+Q2')
    for variant in sorted({r['variant'] for r in records if r['variant']!='none'}):
        for joint in ('joint_vt','joint_tv'):
            comp=[];q1comp=[]
            for r in records:
                if r['variant']!=variant or r['condition']!=joint:continue
                base=lookup.get((r['id'],variant,'image_q2',r['rep']))
                text=lookup.get((r['id'],'none','q2_only',r['rep']))
                one=lookup.get((r['id'],variant,'q1_only',r['rep']))
                if base and base['q2_correct'] is not None and r['q2_correct'] is not None:
                    comp.append(int(r['q2_correct'])-int(base['q2_correct']))
                    if base['q2_correct'] and not r['q2_correct']:
                        candidates.append({'id':r['id'],'variant':variant,'joint':joint,
                                           'text_only_correct':text['q2_correct'] if text else None,
                                           'q2_gold':r['q2_gold'],'image_q2_pred':base['q2_pred'],
                                           'joint_q2_pred':r['q2_pred'], 'q1_joint_correct':r['q1_correct'],
                                           'truncated':r['hit_max_token']})
                if one and one['q1_correct'] is not None and r['q1_correct'] is not None:
                    q1comp.append(int(r['q1_correct'])-int(one['q1_correct']))
            if comp:
                print(f'  {variant}/{joint}: Q2 joint-image_only delta={avg(comp):+.3f} n={len(comp)}; Q1 joint-q1_only delta={avg(q1comp) if q1comp else float("nan"):+.3f} n={len(q1comp)}')
    csv_write(out/'candidate_interference.csv',candidates)
    print('\nCondition means:')
    for s in summaries:
        q1='n/a' if s['q1_accuracy'] is None else f'{s["q1_accuracy"]:.3f}'
        q2='n/a' if s['q2_accuracy'] is None else f'{s["q2_accuracy"]:.3f}'
        print(f'  {s["variant"]:12} {s["condition"]:12} Q1={q1} Q2={q2} n={s["n"]} trunc={s["truncated"]}')
    print(f'\nOutput: {out}/summary.csv, scored.csv, candidate_interference.csv')
    print('NOTE: candidate_interference.csv indicates paired performance changes, not causally proven visual leakage.')

if __name__=='__main__': main()
