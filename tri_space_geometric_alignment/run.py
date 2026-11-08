#!/usr/bin/env python3
"""Three-space study. No test data used for fitting.
Default: demo synthetic data; real data: --train train.npz --test test.npz.
NPZ arrays: low/en/image (or zh/en/image for earlier Chinese pilot).
Train arrays low/en may share row indices ONLY for sampled supervised anchors.
All unsupervised objectives shuffle/sample each modality independently.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
from core import unit,procrustes,fit_joint,retrieval_metrics

def load(p):
    with np.load(p) as z:
        lk='low' if 'low' in z.files else 'zh'
        arrays={k:np.asarray(z[v],dtype=np.float64) for k,v in
                {'low':lk,'en':'en','image':'image'}.items()}
    if len(set(map(len,arrays.values())))!=1:
        raise ValueError('Current retrieval evaluation requires one caption per image')
    if len({a.shape[1] for a in arrays.values()})!=1:
        raise ValueError('Vectors must share dimension (otherwise use train-only projections)')
    return {k:unit(v) for k,v in arrays.items()}

def synthetic(n,d,seed):
    rng=np.random.default_rng(seed)
    sem=unit(rng.normal(size=(n,d)))
    rotations=[np.linalg.qr(rng.normal(size=(d,d)))[0] for _ in range(3)]
    return {k:unit(sem@w+.14*rng.normal(size=(n,d))) for k,w in
            zip(('low','en','image'),rotations)}

def ros_preprocess(train,test,a):
    """Original unpaired Rosetta aligns image to English, no paired rows exposed."""
    if a.skip_rosetta:
        return train['image'],test['image'],{'mode':'identity_ablation'}
    try:
        import torch
        from unpaired_rosetta import WassersteinProcrustes
    except ImportError as exc:
        raise RuntimeError('Install Rosetta: pip install -e external/unpaired-rosetta '
                           '(or use --skip-rosetta for ablation)') from exc
    model=WassersteinProcrustes(num_clusters=min(a.clusters,len(train['image'])//2),
        num_restarts=a.restarts,batch_size=min(a.rosetta_batch,len(train['image'])),
        num_iterations=a.rosetta_steps,num_workers=1,num_threads=1,verbose=False)
    rng=np.random.default_rng(a.seed+801)
    x=train['image'][rng.permutation(len(train['image']))].copy()
    y=train['en'][rng.permutation(len(train['en']))].copy()
    model.fit(torch.tensor(x,dtype=torch.float32),
              torch.tensor(y,dtype=torch.float32),randomness=a.seed)
    xtrain=model.transform(torch.tensor(train['image'],dtype=torch.float32)).numpy()
    xtest=model.transform(torch.tensor(test['image'],dtype=torch.float32)).numpy()
    return unit(xtrain),unit(xtest),{'mode':'original_unpaired_rosetta',
         'clusters':a.clusters,'restarts':a.restarts,'iterations':a.rosetta_steps}

def evaluate(a):
    if bool(a.train)!=bool(a.test):
        raise ValueError('Provide both --train and --test, or neither')
    if a.train:
        train,test=load(a.train),load(a.test)
        origin='provided_embeddings'
    else:
        train,test=synthetic(350,a.demo_dim,a.seed),synthetic(90,a.demo_dim,a.seed+10)
        # Synthetic train and test MUST share the same underlying language transforms:
        # use a dedicated fixed generator below instead of independent random rotations.
        train,test=synthetic_splits(350,90,a.demo_dim,a.seed)
        origin='synthetic_smoke_test'
    if len(train['low']) < a.anchors+4: raise ValueError('Too many anchors for training rows')
    # Anchors are randomly selected known bilingual pairs from training data only.
    rng=np.random.default_rng(a.seed)
    anchors=np.sort(rng.choice(len(train['low']),a.anchors,replace=False))
    imtr,imte,rosetta=ros_preprocess(train,test,a)
    tr={**train,'image':imtr}; te={**test,'image':imte}
    wa=procrustes(tr['low'][anchors],tr['en'][anchors])
    pair=fit_joint(tr['low'],tr['en'],tr['image'],anchors,joint=False)
    joint0=fit_joint(tr['low'],tr['en'],tr['image'],anchors,joint=True,
       steps=a.steps,batch=a.ot_batch,eps=a.eps,unsup_weight=a.ot_weight,
       cycle_weight=0.,seed=a.seed)
    joint=fit_joint(tr['low'],tr['en'],tr['image'],anchors,joint=True,
       steps=a.steps,batch=a.ot_batch,eps=a.eps,unsup_weight=a.ot_weight,
       cycle_weight=a.cycle_weight,seed=a.seed)
    # Evaluate all maps in the common English coordinate system.
    condition={
        'raw_L_to_E':(te['low'],te['en']),
        'few_pair_L_to_E':(te['low']@wa,te['en']),
        'joint_L_to_E':(te['low']@joint.low_to_en,te['en']),
        'raw_L_to_Image':(te['low'],te['image']),
        'few_pair_L_to_Image':(te['low']@wa,te['image']),
        'joint_OT_no_cycle_L_to_Image':(te['low']@joint0.low_to_en,
                                        te['image']@joint0.image_to_en),
        'joint_OT_cycle_L_to_Image':(te['low']@joint.low_to_en,
                                     te['image']@joint.image_to_en),
        'direct_L_to_Image_OT':(te['low']@joint.low_to_image,te['image']),
        'english_to_image':(te['en'],te['image']),
    }
    metrics={name:retrieval_metrics(unit(x),unit(y)) for name,(x,y) in condition.items()}
    output={'provenance':origin,'n_train':len(train['low']),'n_test':len(test['low']),
        'anchor_count':len(anchors),'anchor_indices':anchors.tolist(),
        'rosetta':rosetta,'metrics':metrics,
        'cycle_error':joint.cycle_error(),'history':joint.history,
        'limitations':'retrieval only; not VQA; original paired rows used solely for seed anchors and held-out evaluation; all text/image unpaired OT samples are independently permuted'}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','direction','R1','R5','R10','MRR'])
        for c,directions in metrics.items():
            for direction,vals in directions.items():
                w.writerow([c,direction,*[vals[k] for k in ('R1','R5','R10','MRR')]])
    np.savez_compressed(out/'maps.npz',few_pair=wa,
        low_to_en=joint.low_to_en,image_to_en=joint.image_to_en,
        low_to_image=joint.low_to_image)
    for name,v in metrics.items():
        print(f'{name:37s} R@1={v["left_to_right"]["R1"]:.3f} '
              f'R@5={v["left_to_right"]["R5"]:.3f}')
    print('Saved',out.resolve())

def synthetic_splits(ntrain,ntest,d,seed):
    rng=np.random.default_rng(seed)
    rotations=[np.linalg.qr(rng.normal(size=(d,d)))[0] for _ in range(3)]
    def part(n):
        sem=unit(rng.normal(size=(n,d)))
        return {k:unit(sem@w+.08*rng.normal(size=(n,d))) for k,w in
                zip(('low','en','image'),rotations)}
    return part(ntrain),part(ntest)

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path);p.add_argument('--test',type=Path)
    p.add_argument('--out',default='outputs')
    p.add_argument('--anchors',type=int,default=50)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--ot-batch',type=int,default=256)
    p.add_argument('--eps',type=float,default=.09)
    p.add_argument('--ot-weight',type=float,default=.35)
    p.add_argument('--cycle-weight',type=float,default=.12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--demo-dim',type=int,default=24)
    p.add_argument('--skip-rosetta',action='store_true')
    p.add_argument('--clusters',type=int,default=12)
    p.add_argument('--restarts',type=int,default=3)
    p.add_argument('--rosetta-batch',type=int,default=256)
    p.add_argument('--rosetta-steps',type=int,default=20)
    return p.parse_args()

if __name__=='__main__':
    evaluate(parse())
