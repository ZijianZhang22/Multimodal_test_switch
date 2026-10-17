#!/usr/bin/env python3
"""Apply the 840 source-aligned curated EN/ZH/SW CV-Bench prompts.

No LLM or machine-translation call at runtime. Translation JSONL is stored in
four gzip+base64 shards for portable GitHub check-in. Original NLLB generations
and older predictions are kept in an archive when this changes the dataset.

Usage:
  python apply_curated.py --data outputs/paligemma2_cvbench --seed-cache
  python apply_curated.py --data outputs/paligemma2_cvbench --apply
"""
import argparse,base64,csv,gzip,hashlib,json,shutil,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASKS=('count','relation','depth','distance')
LANGS=('zh','sw')

def load_translations():
    by_id={}
    for task in TASKS:
        path=ROOT/'curated_questions'/f'{task}.jsonl.gz.b64'
        compressed=base64.b64decode(''.join(path.read_text(encoding='ascii').split()))
        content=gzip.decompress(compressed).decode('utf-8')
        batch=[json.loads(x) for x in content.splitlines() if x]
        if len(batch)!=210:raise ValueError(f'{task}: expected 210 rows got {len(batch)}')
        for r in batch:
            if r['task']!=task or r['id'] in by_id:raise ValueError('Unexpected task or duplicate ID '+r['id'])
            if any(not r['questions'].get(lang) for lang in ('en','zh','sw')):raise ValueError('missing lang '+r['id'])
            by_id[r['id']]=r
    if len(by_id)!=840:raise ValueError(f'Expected exactly 840 translations, got {len(by_id)}')
    return by_id

def load_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def seed_cache(data,by_id):
    data.mkdir(parents=True,exist_ok=True)
    path=data/'translation_cache.json'
    cache=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    written={}
    for r in by_id.values():
        for lang in LANGS:
            k=f'{lang}\t{r["question"]}'
            cache[k]=r['translated_questions'][lang]
            for original,translated in zip(r['choices'],r['translated_choices'][lang]):
                key=f'{lang}\t{original}'
                if key in written and written[key]!=translated:
                    raise ValueError(f'Inconsistent CURATED translation for cache key: {key}')
                written[key]=translated
                cache[key]=translated
    path.write_text(json.dumps(cache,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'Seeded {len(cache)} translations into {path} (no NLLB download)',flush=True)

def archive_previous(data):
    targets=list(data.glob('predictions_*.csv'))+list(data.glob('predictions_*_settings.json'))
    targets+=list(data.glob('analysis_val'))+list(data.glob('analysis_test'))
    if not targets:return
    archive=data/'legacy_runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive.mkdir(parents=True,exist_ok=True)
    for p in targets:
        if not p.exists():continue
        shutil.move(str(p),str(archive/p.name))
    print(f'Archived old predictions and analyses to {archive}',flush=True)

def apply(data,by_id):
    path=data/'visual_eval.jsonl'
    if not path.is_file():raise FileNotFoundError(path)
    samples=load_jsonl(path)
    if len(samples)!=840:raise ValueError(f'Expected exact curated 840-example cohort; got {len(samples)}. Do not silently mix translation protocols.')
    seen=set(); changed=False
    for row in samples:
        key=row['id'];c=by_id.get(key)
        if c is None:raise ValueError(f'Missing translation for {key}; wrong split/seed?')
        for k in ('task','split','question','choices','answer'):
            if row[k]!=c[k]:raise ValueError(f'{key}: English source or {k} differs from curated source; refusing to overwrite')
        if row.get('questions')!=c['questions']:changed=True
        seen.add(key)
    if len(seen)!=len(by_id):raise ValueError('Dataset IDs not a one-to-one match')
    manifest_path=data/'dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if changed:
        original=data/'visual_eval_before_curated.jsonl'
        if not original.exists():shutil.copy2(path,original)
        archive_previous(data)
        for row in samples:
            row['questions']=by_id[row['id']]['questions']
        write_jsonl(path,samples)
    combined_sha=hashlib.sha256(''.join(json.dumps(by_id[k]['questions'],ensure_ascii=False,sort_keys=True)
                                         for k in sorted(by_id)).encode('utf-8')).hexdigest()
    manifest.update(translation_model='none (GPT-authored curated CV-Bench 840)',
                    translation_verified=False,
                    translation_method='GPT-authored fixed bilingual templates + curated 304-term bilingual vocabulary',
                    translation_choices_preserved=True,
                    translation_set='cvbench840-en-zh-sw-v1',
                    translation_content_sha256=combined_sha,
                    choices_letter_range='A-F')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=data/'translation_audit_curated_all840.csv'
    with audit.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','task','answer','num_choices','en','zh','sw'])
        writer.writeheader()
        for row in samples:
            writer.writerow({'id':row['id'],'task':row['task'],'answer':row['answer'],
                             'num_choices':len(row['choices']),**row['questions']})
    print(f'{"Applied" if changed else "Already applied"} curated translations to {len(samples)} examples at {path}',flush=True)
    print(f'Full bilingual audit: {audit}',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--seed-cache',action='store_true')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if not (args.seed_cache or args.apply):parser.error('Choose --seed-cache and/or --apply')
    by_id=load_translations()
    if args.seed_cache:seed_cache(args.data,by_id)
    if args.apply:apply(args.data,by_id)

if __name__=='__main__':main()