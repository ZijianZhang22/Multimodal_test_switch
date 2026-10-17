#!/usr/bin/env python3
"""Q1 modality control: '17 + 23 = ?' as text vs the same equation in a PNG.

Reuses Qwen3-VL and official GSM8K/MATH-500 Q2 questions from prepare.py.
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / 'assets' / 'q1_text_17_plus_23.png'
DEFAULT_CONDITIONS = 'q2_only,text_q1_q2,image_q2_only,image_text_q1_q2,image_q1_q2,image_q1_only,text_q1_only'
ALLOWED = {
    'q2_only', 'text_q1_q2', 'image_q2_only', 'image_text_q1_q2',
    'image_q1_q2', 'image_q1_only', 'text_q1_only',
    'image_q2_q1', 'text_q2_q1',
}
NO_IMAGE = {'q2_only', 'text_q1_q2', 'text_q1_only', 'text_q2_q1'}
Q1_ONLY = {'image_q1_only', 'text_q1_only'}


def prompt_for(condition, q2):
    """Build explicit output-format prompts; identical Q2 and gold answer across conditions."""
    if condition not in ALLOWED:
        raise ValueError(f'Unknown condition: {condition}')
    preface = ('You may receive an image. Questions are independent. '
               'For Question 2, use ONLY the information in its text. '
               'Solve carefully and put final answers in the exact tags requested.\n')
    if condition in Q1_ONLY:
        if condition == 'text_q1_only':
            return preface + 'Question 1: What is 17 + 23?\nEnd with <q1>NUMBER</q1>.'
        return preface + 'Question 1: Solve the arithmetic expression in the image.\nEnd with <q1>NUMBER</q1>.'
    q2_part = f'Question 2: {q2}\n'
    if condition in {'q2_only', 'image_q2_only'}:
        return preface + q2_part + 'End with <q2>FINAL_ANSWER</q2>.'
    if condition in {'text_q1_q2', 'image_text_q1_q2', 'text_q2_q1'}:
        q1_part = 'Question 1: What is 17 + 23?\n'
    else:
        q1_part = 'Question 1: Solve the arithmetic expression in the image.\n'
    if condition in {'image_q2_q1', 'text_q2_q1'}:
        return preface + q2_part + q1_part + 'Answer Question 2 first, then Question 1. End with <q2>FINAL_ANSWER</q2> <q1>NUMBER</q1>.'
    return preface + q1_part + q2_part + 'Answer Question 1 first, then Question 2. End with <q1>NUMBER</q1> <q2>FINAL_ANSWER</q2>.'


def make_manifest(path, benchmark, count, seed, min_level):
    """Persist Q2 items once, preventing accidental re-sampling during resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        if not rows or len(rows) != count or any(r['benchmark'] != benchmark for r in rows):
            raise ValueError('Existing manifest does not match BENCHMARK/PAIRS. Use a new OUTPUT_DIR.')
        print(f'Reusing {len(rows)} saved Q2 samples: {path}', flush=True)
        return rows
    from prepare import load_questions
    rows = load_questions(benchmark, seed, count, min_level)
    with path.open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Saved {len(rows)} matched Q2 questions to {path}', flush=True)
    return rows


def planned_tasks(rows, conditions, samples):
    tasks = []
    for rep in range(samples):
        for cond in conditions:
            if cond in Q1_ONLY:
                tasks.append(({'id':'constant-q1', 'benchmark':'none', 'gold':'', 'question':''}, cond, rep))
            else:
                for row in rows:
                    tasks.append((row, cond, rep))
    return tasks


def generate(args, rows, conditions):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from run import inputs_for

    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU available; run inference on RunPod.')
    image_path = Path(args.image).expanduser().resolve()
    if any(c not in NO_IMAGE for c in conditions) and not image_path.is_file():
        raise FileNotFoundError(f'Q1 image does not exist: {image_path}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto').eval()
    device = next(model.parameters()).device
    prediction_path = args.output_dir / 'predictions.jsonl'
    completed = set()
    if prediction_path.exists():
        for line in prediction_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                completed.add((r['q2_id'], r['condition'], r['rep']))
    tasks = planned_tasks(rows, conditions, args.samples)
    print(f'Planned {len(tasks)} model calls; completed {len(completed)}', flush=True)
    for idx, (row, cond, rep) in enumerate(tasks, start=1):
        task_key = (row['id'], cond, rep)
        if task_key in completed:
            continue
        txt = prompt_for(cond, row['question'])
        image = None if cond in NO_IMAGE else str(image_path)
        inputs = inputs_for(processor, image, txt, device)
        use_sampling = args.temperature > 0
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=use_sampling,
                **({'temperature':args.temperature, 'top_p':args.top_p} if use_sampling else {}),
            )
        output_ids = generated[0][inputs['input_ids'].shape[-1]:]
        prediction = processor.decode(output_ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        record = {
            'q2_id':row['id'], 'q2_question':row['question'],
            'q2_gold':row['gold'], 'q2_benchmark':row['benchmark'],
            'q2_level':row.get('level'), 'q1_gold':'40',
            'condition':cond, 'rep':rep, 'pred':prediction,
            'image_path':image, 'num_new_tokens':int(len(output_ids)),
            'hit_max_token':bool(len(output_ids) >= args.max_new_tokens),
            'model':args.model, 'temperature':args.temperature, 'top_p':args.top_p,
        }
        with prediction_path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'[{idx}/{len(tasks)}] {cond} Q2={row["id"]} repeat={rep} tokens={len(output_ids)}',flush=True)
    print(f'Predictions saved: {prediction_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=Path('pilot_image_text_gsm8k'))
    ap.add_argument('--benchmark', choices=['gsm8k','math500'], default='gsm8k')
    ap.add_argument('--pairs', type=int, default=8)
    ap.add_argument('--math-min-level', type=int, default=3)
    ap.add_argument('--image', default=str(DEFAULT_IMAGE))
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Thinking')
    ap.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--max-new-tokens', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--score-only', action='store_true')
    args = ap.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [x.strip() for x in args.conditions.split(',') if x.strip()]
    if not conditions or any(c not in ALLOWED for c in conditions):
        ap.error(f'Unsupported --conditions: {conditions}')
    if args.pairs < 1 or args.samples < 1:
        ap.error('--pairs and --samples must be positive')
    if not args.score_only:
        rows = make_manifest(args.output_dir / 'questions.jsonl',
                             args.benchmark, args.pairs, args.seed, args.math_min_level)
        if not args.prepare_only:
            generate(args, rows, conditions)
    if not args.prepare_only:
        from score_image_text_q1_control import score
        score(args.output_dir)


if __name__ == '__main__':
    main()
