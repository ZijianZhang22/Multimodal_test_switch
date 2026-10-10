#!/usr/bin/env python3
"""Paired Q2 comparison: same Idis image + text Q1 vs Idis image + second-image Q1."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from score import numeric, q2_result, tag

CONDITIONS = ('joint_text', 'joint_image_q1')


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def judge(record: dict) -> dict:
    q1_answer = tag(record['pred'], 'q1')
    q1_correct = q1_answer is not None and numeric(q1_answer) == numeric(record['q1_gold'])
    q2_answer, q2_correct, note = q2_result(record['pred'], record['q2_gold'], record['q2_benchmark'])
    return {
        'id': record['id'], 'q2_id': record['q2_id'],
        'variant': record['variant'], 'condition': record['condition'],
        'rep': int(record['rep']), 'q1_correct': q1_correct,
        'q1_pred': q1_answer, 'q2_pred': q2_answer,
        'q2_gold': record['q2_gold'], 'q2_correct': q2_correct,
        'q2_score_note': note, 'num_new_tokens': record['num_new_tokens'],
        'hit_max_token': bool(record['hit_max_token'])
    }


def score(pred_path: Path):
    pred_path = Path(pred_path)
    if not pred_path.is_file():
        raise FileNotFoundError(f'Missing predictions at {pred_path}')
    raw = [json.loads(s) for s in pred_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not raw:
        raise ValueError('No records to score')
    rows = [judge(x) for x in raw]
    by_key = {}
    for r in rows:
        k = (r['id'], r['q2_id'], r['variant'], r['rep'], r['condition'])
        if k in by_key:
            raise ValueError(f'Duplicate result: {k}')
        by_key[k] = r
    out_dir = pred_path.parent
    write_csv(out_dir / 'scored_dual_image_q1.csv', rows, list(rows[0]))
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['variant'], r['condition'])].append(r)
    summary=[]
    for (variant, condition),items in sorted(grouped.items()):
        q2 = [int(r['q2_correct']) for r in items if r['q2_correct'] is not None]
        summary.append({'variant':variant,'condition':condition,'n':len(items),
                        'q1_accuracy':sum(int(r['q1_correct']) for r in items)/len(items),
                        'q2_accuracy':sum(q2)/len(q2) if q2 else '',
                        'q2_scored_n':len(q2),
                        'q2_unscored':len(items)-len(q2),
                        'truncated_n':sum(int(r['hit_max_token']) for r in items),
                        'mean_tokens':sum(r['num_new_tokens'] for r in items)/len(items)})
    write_csv(out_dir/'summary_dual_image_q1.csv',summary,list(summary[0]))
    pair_rows=[]
    for k, text in by_key.items():
        id_,qid,variant,rep,cond=k
        if cond!='joint_text':continue
        img=by_key.get((id_,qid,variant,rep,'joint_image_q1'))
        if not img: continue
        if text['q2_correct'] is None or img['q2_correct'] is None:
            delta=''
            changed='needs_review'
        else:
            delta=int(img['q2_correct'])-int(text['q2_correct'])
            changed=('text_correct_image_wrong' if delta == -1 else
                     'text_wrong_image_correct' if delta == 1 else 'no_accuracy_change')
        pair_rows.append({
            'id':id_, 'q2_id':qid,'variant':variant,'rep':rep,
            'q2_gold':text['q2_gold'],'text_q2_pred':text['q2_pred'],
            'image_q2_pred':img['q2_pred'],
            'text_q2_correct':text['q2_correct'],'image_q2_correct':img['q2_correct'],
            'delta_image_minus_text':delta, 'change':changed,
            'text_q1_correct':text['q1_correct'],'image_q1_correct':img['q1_correct'],
            'text_truncated':text['hit_max_token'],'image_truncated':img['hit_max_token'],
            'text_q2_score_note':text['q2_score_note'], 'image_q2_score_note':img['q2_score_note'],
        })
    cols=['id','q2_id','variant','rep','q2_gold','text_q2_pred','image_q2_pred',
          'text_q2_correct','image_q2_correct','delta_image_minus_text','change',
          'text_q1_correct','image_q1_correct','text_truncated','image_truncated',
          'text_q2_score_note','image_q2_score_note']
    write_csv(out_dir/'paired_details_dual_image_q1.csv',pair_rows,cols)
    candidates=[r for r in pair_rows if r['change']=='text_correct_image_wrong']
    write_csv(out_dir/'candidate_interference_dual_image_q1.csv',candidates,cols)
    by_variant=defaultdict(list)
    for r in pair_rows:
        if r['delta_image_minus_text']!='':by_variant[r['variant']].append(r)
    stats=[]
    for v,matched in sorted(by_variant.items()):
        stats.append({'variant':v,'matched_n':len(matched),
                      'text_q2_accuracy':sum(int(r['text_q2_correct']) for r in matched)/len(matched),
                      'image_q2_accuracy':sum(int(r['image_q2_correct']) for r in matched)/len(matched),
                      'delta_image_minus_text':sum(r['delta_image_minus_text'] for r in matched)/len(matched),
                      'text_correct_image_wrong':sum(r['change']=='text_correct_image_wrong' for r in matched),
                      'text_wrong_image_correct':sum(r['change']=='text_wrong_image_correct' for r in matched)})
    columns=['variant','matched_n','text_q2_accuracy','image_q2_accuracy',
             'delta_image_minus_text','text_correct_image_wrong','text_wrong_image_correct']
    write_csv(out_dir/'paired_summary_dual_image_q1.csv',stats,columns)
    print('\nDual-image control results:')
    for r in summary:
        print(f"  {r['variant']} / {r['condition']}: Q1={r['q1_accuracy']:.3f}, Q2={r['q2_accuracy']} "
              f"(n={r['n']}, truncated={r['truncated_n']}, unscored={r['q2_unscored']})")
    for r in stats:
        print(f"  {r['variant']} matched n={r['matched_n']}: image - text Q2 accuracy "
              f"= {r['delta_image_minus_text']:+.3f}; correct->wrong={r['text_correct_image_wrong']}; "
              f"wrong->correct={r['text_wrong_image_correct']}")
    if not stats:
        print('  No fully scored matched text/image pairs yet.')
    print('Saved paired summary, per-sample details, candidate flips in', out_dir)
    return {'summary': summary, 'stats':stats, 'pairs':pair_rows}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',type=Path,required=True)
    args=ap.parse_args()
    score(args.predictions)
