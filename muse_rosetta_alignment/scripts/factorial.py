#!/usr/bin/env python3
"""Four-arm retrieval + language-first Rosetta retraining, original Rosetta API."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from unpaired_rosetta import WassersteinProcrustes

def unit(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def load(path):
    with np.load(path) as f:a={k:np.asarray(f[k],np.float32) for k in ("image","en","zh")}
    assert len(set(map(len,a.values())))==1 and len(set(x.shape[1] for x in a.values()))==1
    return a

def query_ranks(scores):
    x=np.asarray(scores,np.float64)
    return (x>=np.diag(x)[:,None]-1e-12).sum(1)

def metrics(s):
    out={}
    for direction,m in (("image_to_text",s),("text_to_image",s.T)):
        r=query_ranks(m)
        out[direction]={"R1":float((r<=1).mean()),"R5":float((r<=5).mean()),"R10":float((r<=10).mean()),"MRR":float((1/r).mean())}
    return out

def bootstrap_delta(s0,s1,seed=42,B=1000):
    rng=np.random.default_rng(seed);out={}
    for name,m0,m1 in (("image_to_text",s0,s1),("text_to_image",s0.T,s1.T)):
        diff=(query_ranks(m1)==1).astype(float)-(query_ranks(m0)==1).astype(float)
        ix=rng.integers(0,len(diff),size=(B,len(diff)))
        out[name]={"delta_R1_pp":float(100*diff.mean()),"ci95_pp":[float(v) for v in np.percentile(100*diff[ix].mean(1),[2.5,97.5])]}
    return out

def fit(images,captions,args,seed):
    align=WassersteinProcrustes(num_clusters=args.clusters,num_restarts=args.restarts,
        batch_size=args.batch_size,num_iterations=args.iterations,num_workers=1,num_threads=1,verbose=True)
    perm=np.random.default_rng(seed).permutation(len(captions))
    align.fit(torch.from_numpy(images),torch.from_numpy(captions[perm].copy()),randomness=seed)
    return align

def similarity(al,img,txt):
    return al.similarity(torch.from_numpy(img),torch.from_numpy(txt)).cpu().numpy()

def save_map(path,al):
    torch.save({"mean_x":al.mean_x,"mean_y":al.mean_y,"weight":al.weight},path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/train.npz"))
    p.add_argument("--test",type=Path,default=Path("data/test.npz"))
    p.add_argument("--muse-mapping",type=Path,default=Path("outputs/muse_mapping.npz"))
    p.add_argument("--out-dir",type=Path,default=Path("outputs"))
    p.add_argument("--clusters",type=int,default=12)
    p.add_argument("--restarts",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--iterations",type=int,default=20)
    p.add_argument("--bootstraps",type=int,default=1000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--skip-sequential",action="store_true")
    p.add_argument("--include-mixture-control",action="store_true")
    p.add_argument("--save-maps",action="store_true")
    a=p.parse_args()
    train,test=load(a.train),load(a.test)
    if len(train["image"])<a.clusters or len(test["image"])<10:raise ValueError("Not enough examples")
    with np.load(a.muse_mapping) as f:w=np.asarray(f["weight"],np.float32)
    assert w.shape==(train["zh"].shape[1],train["zh"].shape[1])
    ztr=unit(train["zh"]@w.T);zte=unit(test["zh"]@w.T)
    scores={}
    scores["Baseline"]=unit(test["image"])@unit(test["zh"]).T
    scores["A_MUSE_only"]=unit(test["image"])@zte.T
    print("[Rosetta] training on English only")
    en=fit(train["image"],train["en"],a,a.seed)
    scores["B_Rosetta_only"]=similarity(en,test["image"],test["zh"])
    scores["C_MUSE_plus_Rosetta_fixed"]=similarity(en,test["image"],zte)
    scores["English_native_reference"]=unit(test["image"])@unit(test["en"]).T
    scores["English_Rosetta_reference"]=similarity(en,test["image"],test["en"])
    if not a.skip_sequential:
        selector=np.random.default_rng(a.seed+400).random(len(train["en"]))<.5
        mix=train["en"].copy();mix[selector]=ztr[selector]
        print("[Rosetta] second fit on English + MUSE-mapped Chinese")
        seq=fit(train["image"],unit(mix),a,a.seed+1)
        scores["D_Language_first_retrain_Rosetta"]=similarity(seq,test["image"],zte)
        if a.save_maps:save_map(a.out_dir/"rosetta_mixed_aligned.pt",seq)
        if a.include_mixture_control:
            mix_raw=train["en"].copy();mix_raw[selector]=train["zh"][selector]
            print("[Rosetta] control: English + raw Chinese")
            control=fit(train["image"],unit(mix_raw),a,a.seed+1)
            scores["E_Rosetta_raw_language_mix_control"]=similarity(control,test["image"],test["zh"])
            if a.save_maps:save_map(a.out_dir/"rosetta_mixed_raw.pt",control)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    if a.save_maps:save_map(a.out_dir/"rosetta_english.pt",en)
    result={name:metrics(s) for name,s in scores.items()}
    comparisons={}
    for label,left,right in [
        ("A_minus_baseline","Baseline","A_MUSE_only"),
        ("B_minus_baseline","Baseline","B_Rosetta_only"),
        ("C_minus_B_incremental_MUSE","B_Rosetta_only","C_MUSE_plus_Rosetta_fixed"),
        ("C_minus_A_incremental_Rosetta","A_MUSE_only","C_MUSE_plus_Rosetta_fixed"),
        ("C_minus_baseline","Baseline","C_MUSE_plus_Rosetta_fixed"),
        ("D_minus_C_sequential_gain","C_MUSE_plus_Rosetta_fixed","D_Language_first_retrain_Rosetta"),
        ("D_minus_E_control","E_Rosetta_raw_language_mix_control","D_Language_first_retrain_Rosetta")]:
        if left in scores and right in scores:comparisons[label]=bootstrap_delta(scores[left],scores[right],a.seed,a.bootstraps)
    output={"metadata":{"n_train":len(train["image"]),"n_test":len(test["image"]),
        "methods":"original supervised MUSE Procrustes and original unpaired Rosetta WassersteinProcrustes",
        "limitation":"Shared pretrained multilingual CLIP baseline; retrieval only, not VQA or cultural understanding"},
        "metrics":result,"paired_bootstrap_R1":comparisons}
    (a.out_dir/"results.json").write_text(json.dumps(output,indent=2))
    with (a.out_dir/"results.csv").open("w",newline="") as f:
        wcsv=csv.writer(f);wcsv.writerow(["condition","direction","R@1_pct","R@5_pct","R@10_pct","MRR"])
        for name,directions in result.items():
            for direction,m in directions.items():
                wcsv.writerow([name,direction,round(m["R1"]*100,4),round(m["R5"]*100,4),round(m["R10"]*100,4),round(m["MRR"],6)])
    print("\n--- Held-out image-to-Chinese retrieval ---")
    for name,dirs in result.items():
        r=dirs["image_to_text"]
        print(f'{name:40s} R@1={r["R1"]*100:6.2f}% R@5={r["R5"]*100:6.2f}% MRR={r["MRR"]:.3f}')
    for name,comp in comparisons.items():print(name,comp["image_to_text"])
    print("Saved",a.out_dir/"results.csv",a.out_dir/"results.json")
if __name__=="__main__":main()
