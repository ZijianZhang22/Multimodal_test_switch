"""RunPod orchestrator: real GPU smoke test then held-out validation (no automatic test tuning)."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
UPSTREAM='24fa054932efb5af0e5bd5b3ebb63e4ff97bff0d'

def run(args,cwd=ROOT):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)

def safe_extract(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path=(destination/item.filename).resolve()
            if not path.is_relative_to(destination):raise ValueError('Unsafe archive member')
            if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink refused')
        z.extractall(destination)

def locate_images(root,rows):
    names={x['image'] for x in rows if x.get('image')}
    if not names:raise ValueError('No image rows')
    first=next(iter(names));candidates=[Path(root)]
    for match in Path(root).rglob(Path(first).name):
        candidate=match
        for _ in Path(first).parts:candidate=candidate.parent
        candidates.append(candidate)
    for candidate in candidates:
        if all((candidate/n).is_file() for n in names):return candidate.resolve()
    raise FileNotFoundError(f'Images do not match metadata under {root}; example: {first}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--model',default='Qwen/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--run-dir',default='/workspace/mrre-runs/qwen25vl7b-zh')
    p.add_argument('--image-root',default=os.environ.get('IMAGE_ROOT'))
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--max-pixels',type=int,default=1280*28*28)
    p.add_argument('--reason-layer',type=int,default=22)
    p.add_argument('--anchor-layer',type=int,default=24)
    p.add_argument('--alpha-reason',type=float,default=1)
    p.add_argument('--alpha-anchor',type=float,default=.75)
    a=p.parse_args()
    os.chdir(ROOT)
    run_dir=Path(a.run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(run_dir/'.run.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another pipeline is using this run directory')
    print('Free disk GiB:',round(shutil.disk_usage(run_dir).free/2**30,1),flush=True)
    print('Recommended: >=40 GiB free disk, 48 GB GPU for the default 7B pilot. Capacity depends on image/sequence length.',flush=True)
    from huggingface_hub import snapshot_download,hf_hub_download
    from mrre import read
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this pipeline')
    upstream=ROOT/'upstream'
    if not upstream.exists():run(['git','clone','https://github.com/QimingLi-scir/MRRE.git',upstream])
    run(['git','checkout',UPSTREAM],cwd=upstream)
    run([sys.executable,'mrre.py','prepare','--source',upstream/'evaluation/MathVerse/data/zh.json','--out','data'])
    run([sys.executable,'-m','unittest','-v','test_core.py'])
    rows=read(ROOT/'data/calibration.jsonl')+read(ROOT/'data/validation.jsonl')+read(ROOT/'data/test.jsonl')
    if a.image_root:
        image_root=locate_images(Path(a.image_root),rows)
    else:
        archive=hf_hub_download(repo_id='AI4Math/MathVerse',repo_type='dataset',filename='images.zip')
        dest=run_dir/'images'
        try:image_root=locate_images(dest,rows)
        except FileNotFoundError:
            print('Extracting downloaded image archive...',flush=True);safe_extract(archive,dest)
            image_root=locate_images(dest,rows)
    from PIL import Image
    for row in rows:
        if row.get('image'):
            with Image.open(image_root/row['image']) as img:img.verify()
    if Path(a.model).is_dir():model=str(Path(a.model).resolve())
    else:
        model=snapshot_download(a.model,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.tiktoken'])
    config={k:v for k,v in vars(a).items() if k not in ['smoke_only','image_root']}
    config.update(model_snapshot=model,image_root=str(image_root),upstream=UPSTREAM,
                  code_sha256=hashlib.sha256((ROOT/'mrre.py').read_bytes()).hexdigest(),
                  data_sha256=hashlib.sha256((ROOT/'data/calibration.jsonl').read_bytes()).hexdigest())
    manifest=run_dir/'run_config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Run configuration changed. Use a different --run-dir to avoid mixing results.')
    manifest.write_text(json.dumps(config,indent=2))
    common=['--model',model,'--image-root',str(image_root),'--max-new-tokens',str(a.max_new_tokens),'--max-pixels',str(a.max_pixels)]
    def stage(name,cmd,outputs):
        marker=run_dir/(name+'.done.json')
        if marker.exists() and all(Path(x).is_file() for x in outputs):
            print('Reusing completed stage:',name,flush=True);return
        # Preserve incomplete output and retry this stage from its beginning.
        for x in outputs:
            path=Path(x)
            if path.exists():path.rename(path.with_name(path.name+'.partial.'+str(time.time_ns())))
        log=run_dir/(name+'.log')
        print('Starting',name,'log:',log,flush=True)
        with log.open('a') as f:
            proc=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            if proc.wait():raise RuntimeError(f'Stage {name} failed; see {log}')
        marker.write_text(json.dumps({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
    smoke=run_dir/'smoke_data';smoke.mkdir(exist_ok=True)
    from mrre import write
    write(smoke/'calibration.jsonl',read(ROOT/'data/calibration.jsonl')[:4])
    write(smoke/'validation.jsonl',read(ROOT/'data/validation.jsonl')[:2])
    def experiment(label,data_dir):
        vector=run_dir/(label+'_vectors.pt');result=run_dir/(label+'_validation.jsonl')
        stage(label+'_extract',[sys.executable,'-u','mrre.py','extract',*common,'--data',data_dir/'calibration.jsonl','--out',vector],[vector])
        stage(label+'_evaluate',[sys.executable,'-u','mrre.py','evaluate',*common,'--data',data_dir/'validation.jsonl','--vectors',vector,'--out',result,
             '--reason-layer',a.reason_layer,'--anchor-layer',a.anchor_layer,'--alpha-reason',a.alpha_reason,'--alpha-anchor',a.alpha_anchor],[result,Path(str(result)+'.config.json')])
        run([sys.executable,'mrre.py','summarize','--out',result])
        records=read(result)
        if len(records)%5 or not records:raise ValueError('Incomplete paired evaluation')
        print('Unparsed answers:',sum(x['correct'] is None for x in records),'Truncations:',sum(x['truncated'] for x in records),flush=True)
        return vector
    experiment('smoke',smoke)
    if not a.smoke_only:
        vector=experiment('pilot',ROOT/'data')
        import shlex
        test_cmd=[sys.executable,str(ROOT/'mrre.py'),'evaluate',*common,'--data',str(ROOT/'data/test.jsonl'),'--vectors',str(vector),'--out',str(run_dir/'test.jsonl'),
                  '--reason-layer',str(a.reason_layer),'--anchor-layer',str(a.anchor_layer),'--alpha-reason',str(a.alpha_reason),'--alpha-anchor',str(a.alpha_anchor)]
        (run_dir/'run_test_after_validation.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(test_cmd)+'\n')
    print('DONE. Results and logs:',run_dir,flush=True)
    print('Accuracy here uses conservative local scoring. Review unparsed/truncated outputs before claiming a gain.',flush=True)

if __name__=='__main__':main()
