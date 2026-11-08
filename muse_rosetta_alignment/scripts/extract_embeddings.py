#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,default=Path("data/manifest.csv"))
    p.add_argument("--out",type=Path,default=Path("data"))
    p.add_argument("--batch-size",type=int,default=32)
    a=p.parse_args()
    df=pd.read_csv(a.csv).dropna(subset=["image_path","en","zh"])
    if df.image_path.duplicated().any():raise ValueError("One caption per unique image required")
    paths=[Path(q) if Path(q).is_absolute() else a.csv.parent/q for q in df.image_path]
    if not all(x.exists() for x in paths):raise FileNotFoundError("Missing image in manifest")
    device="cuda" if torch.cuda.is_available() else "cpu"
    im_model=SentenceTransformer("sentence-transformers/clip-ViT-B-32",device=device)
    tx_model=SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1",device=device)
    en=unit(tx_model.encode(df.en.astype(str).tolist(),batch_size=a.batch_size,convert_to_numpy=True,show_progress_bar=True))
    zh=unit(tx_model.encode(df.zh.astype(str).tolist(),batch_size=a.batch_size,convert_to_numpy=True,show_progress_bar=True))
    images=[]
    for k in tqdm(range(0,len(paths),a.batch_size),desc="images"):
        batch=[]
        for path in paths[k:k+a.batch_size]:
            with Image.open(path) as im:batch.append(im.convert("RGB").copy())
        images.append(im_model.encode(batch,batch_size=a.batch_size,convert_to_numpy=True))
    image=unit(np.concatenate(images))
    assert image.shape==en.shape==zh.shape,(image.shape,en.shape,zh.shape)
    a.out.mkdir(parents=True,exist_ok=True)
    for split in ("train","test"):
        ids=np.where(df.split.astype(str).str.lower().to_numpy()==split)[0]
        if not len(ids):raise ValueError(f"empty {split}")
        np.savez_compressed(a.out/f"{split}.npz",image=image[ids],en=en[ids],zh=zh[ids])
        print(split,len(ids),image.shape[1])
if __name__=="__main__":main()
