#!/usr/bin/env python3
"""Evaluate same Q2 when Q1=17+23 is text vs image; produces per-Q2 matched differences."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from score import q2_result, tag, numeric

Q1_CONDITIONS = {
    'text_q1_only','image_q1_only',
    'text_q1_q2','image_text_q1_q2','image_q1_q2',
    'text_q2_q1','image_q2_q1',
}
Q2_CONDITIONS = {
    'q2_only','image_q2_only',
    'text_q1_q2','image_text_q1_q2','image_q1_q2',
    'text_q2_q1','image_q2_q1',
}
COMPARISONS = [
    ('text_task_load', 'text_q1_q2', 'q2_only'),
    ('image_presence', 'image_q2_only', 'q2_only'),
    ('active_visual_q1', 'image_q1_q2', 'image_q2_only'),
    ('image_vs_text_q1', 'image_q1_q2', 'text_q1_q2'),
    ('image_vs_text_with_same_image', 'image_q1_q2', 'image_text_q1_q2'),
    ('image_with_text_q1', 'image_text_q1_q2', 'text_q1_q2'),
]


def to_csv(path, records, fields):
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def score_one(record):
    condition = record['condition']
    pred = record['pred']
    q1_pred = tag(pred, 'q1') if condition in Q1_CONDITIONS else None
    q1_correct = None if condition not in Q1_CONDITIONS else (
        numeric(q1_pred) == numeric(record['q1_gold']) if q1_pred is not None else False)
    q2_pred, q2_correct, note = (None, None, 'not_applicable')
    if condition in Q2_CONDITIONS:
        q2_pred, q2_correct, note = q2_result(pred, record['q2_gold'], record['q2_benchmark'])
    return {
        'q2_id':record['q2_id'], 'condition':condition, 'rep':record['rep'],
        'q2_benchmark':record['q2_benchmark'], 'q1_pred':q1_pred, 'q1_correct':q1_correct,
        'q2_gold':record['q2_gold'], 'q2_pred':q2_pred, 'q2_correct':q2_correct,
        'q2_scoring':note, 'hit_max_token':record['hit_max_token'],
        'num_new_tokens':record['num_new_tokens'],
    }


def score(output_dir):
    path = output_dir / 'predictions.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'No predictions yet: {path}')
    records = [score_one(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('No predictions in ' + str(path))
    to_csv(output_dir / 'scored.csv', records, list(records[0]))
    groups = defaultdict(list)
    by_id = {}
    for row in records:
        groups[row['condition']].append(row)
        key = (row['q2_id'], row['rep'], row['condition'])
        if key in by_id:
            raise ValueError(f'Duplicate prediction: {key}')
        by_id[key] = row
    summaries = []
    for condition, items in sorted(groups.items()):
        valid = [int(x['q2_correct']) for x in items if x['q2_correct'] is not None]
        valid_q1 = [int(x['q1_correct']) for x in items if x['q1_correct'] is not None]
        summaries.append({
            'condition':condition, 'n':len(items), 'q2_scored':len(valid),
            'q2_accuracy':sum(valid)/len(valid) if valid else '',
            'q1_scored':len(valid_q1),
            'q1_accuracy':sum(valid_q1)/len(valid_q1) if valid_q1 else '',
            'truncated':sum(x['hit_max_token'] for x in items),
            'mean_tokens':sum(x['num_new_tokens'] for x in items)/len(items),
        })
    to_csv(output_dir / 'summary.csv', summaries, list(summaries[0]))
    diffs, flips = [], []
    for name, treatment, control in COMPARISONS:
        cases = []
        for (qid, rep, cond), row in by_id.items():
            if cond != treatment:
                continue
            baseline = by_id.get((qid, rep, control))
            if baseline is None or baseline['q2_correct'] is None or row['q2_correct'] is None:
                continue
            delta = int(row['q2_correct']) - int(baseline['q2_correct'])
            cases.append(delta)
            if delta == -1:
                flips.append({
                    'comparison':name,'q2_id':qid,'rep':rep,
                    'control':control,'treatment':treatment,
                    'correct_answer':row['q2_gold'],
                    'control_answer':baseline['q2_pred'],
                    'treatment_answer':row['q2_pred'],
                    'treatment_q1_correct':row['q1_correct'],
                    'treatment_truncated':row['hit_max_token'],
                    'treatment_scoring':row['q2_scoring'],
                })
        diffs.append({
            'comparison':name,'treatment':treatment,'control':control,
            'matched_n':len(cases),'mean_accuracy_delta':sum(cases)/len(cases) if cases else '',
            'correct_to_wrong':sum(x==-1 for x in cases),
            'wrong_to_correct':sum(x==1 for x in cases),
        })
    to_csv(output_dir / 'paired_comparisons.csv', diffs, list(diffs[0]))
    flip_columns = ['comparison','q2_id','rep','control','treatment','correct_answer',
                    'control_answer','treatment_answer','treatment_q1_correct',
                    'treatment_truncated','treatment_scoring']
    to_csv(output_dir / 'candidate_interference.csv', flips, flip_columns)
    print('\nCondition summary:')
    for row in summaries:
        print(f"{row['condition']:26} Q2 acc={row['q2_accuracy']} Q1 acc={row['q1_accuracy']} "+
              f"n={row['n']} truncated={row['truncated']}")
    print('\nPaired Q2 accuracy changes (treatment minus control):')
    for row in diffs:
        print(f"{row['comparison']:30} delta={row['mean_accuracy_delta']} matched={row['matched_n']}")
    print(f'\nWrote results to: {output_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    score(args.output_dir)
