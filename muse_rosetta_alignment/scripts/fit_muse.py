#!/usr/bin/env python3
"""Directly call original Facebook MUSE supervised Trainer.procrustes."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch import nn

def r1(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)
    b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-9)
    return float(np.mean((a@b.T).argmax(1)==np.arange(len(a))))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--muse-root",type=Path,default=Path("external/MUSE"))
    p.add_argument("--train",type=Path,default=Path("data/train.npz"))
    p.add_argument("--out",type=Path,default=Path("outputs/muse_mapping.npz"))
    a=p.parse_args()
    sys.path.insert(0,str(a.muse_root.resolve()))
    from src.trainer import Trainer
    with np.load(a.train) as f:zh=np.asarray(f["zh"],np.float32);en=np.asarray(f["en"],np.float32)
    assert zh.shape==en.shape and len(zh)>=30
    n,d=zh.shape
    ids=np.random.default_rng(42).permutation(n);v=max(10,round(.1*n));train,dev=ids[v:],ids[:v]
    state=SimpleNamespace(src_emb=nn.Embedding.from_pretrained(torch.from_numpy(zh),freeze=True),
        tgt_emb=nn.Embedding.from_pretrained(torch.from_numpy(en),freeze=True),
        mapping=nn.Linear(d,d,bias=False),
        dico=torch.from_numpy(np.stack([train,train],1).astype("int64")))
    with torch.no_grad():Trainer.procrustes(state)
    w=state.mapping.weight.detach().cpu().numpy().astype(np.float32)
    before,after=r1(zh[dev],en[dev]),r1(zh[dev]@w.T,en[dev])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.out,weight=w,train_indices=train,dev_indices=dev)
    (a.out.with_suffix(".json")).write_text(json.dumps({"train_pairs":len(train),"dev_pairs":len(dev),"R1_before":before,"R1_after":after},indent=2))
    print(f"MUSE original supervised Procrustes: dev R@1 {before:.3f} -> {after:.3f}")
if __name__=="__main__":main()
