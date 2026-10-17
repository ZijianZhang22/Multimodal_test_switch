#!/usr/bin/env python3
"""Build a 30-image matched Idis-perception study from OFFICIAL benchmark images.

Exactly four variants of each base ImageNet-9 image:
clean (original ImageNet-9) / aligned / conflicting / irrelevant (published Idis).
No image synthesis or altered labels. Only matched base stems are selected.
"""
import argparse
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "Vail-2000/Idis"
BASE = "Idis-perception/visual_distractor"
OFFICIAL_TEST_ARCHIVE = (
    "https://github.com/MadryLab/backgrounds_challenge/releases/"
    "download/data/backgrounds_challenge_data.tar.gz"
)
VARIANTS = ("clean", "aligned", "conflicting", "irrelevant")
IMAGE_EXTS = (".JPEG", ".jpeg", ".jpg", ".png")
CLASS_NAMES = {
    "00_dog": "Dog", "01_bird": "Bird", "02_wheeled vehicle": "Vehicle",
    "03_reptile": "Reptile", "04_carnivore": "Carnivore",
    "05_insect": "Insect", "06_musical instrument": "Instrument",
    "07_primate": "Primate", "08_fish": "Fish",
}


def remote_files(api, path):
    """Returns stem:path using the Hub's native file tree, without fetching imagery."""
    results = api.list_repo_tree(repo_id=REPO, repo_type="dataset",
                                 path_in_repo=path, recursive=False)
    found = {}
    for item in results:
        loc = item.path
        if loc.lower().endswith((".png", ".jpeg", ".jpg")):
            found[Path(loc).stem] = loc
    return found


def index_clean(root):
    """Locate original/val/<class>/<stem>.JPEG or flat <class>/<stem>.JPEG."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {}
    index = {}
    # Restrict scan to official class directories (avoids unrelated val/train files).
    for class_dir in CLASS_NAMES:
        folders = [root / class_dir, root / "val" / class_dir,
                   root / "original" / "val" / class_dir,
                   root / "original" / class_dir]
        for folder in folders:
            if folder.is_dir():
                for file in folder.iterdir():
                    if file.is_file() and file.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        index[(class_dir, file.stem)] = str(file.resolve())
                break
    return index


def download_official_clean(root):
    """Download official ImageNet-9 test release and extract ONLY original/val.
    The archive is ~280MB; no data from its many other variants is extracted.
    """
    root.mkdir(parents=True, exist_ok=True)
    if sum(1 for c in CLASS_NAMES for _ in (root / c).glob("*.*")) >= 100:
        return
    archive = root.parent / "backgrounds_challenge_data.tar.gz"
    if not archive.exists():
        print("Downloading official ImageNet-9 test archive (~280 MB), once:", OFFICIAL_TEST_ARCHIVE, flush=True)
        req = urllib.request.Request(OFFICIAL_TEST_ARCHIVE, headers={"User-Agent": "Idis-Matched-Pilot/1.0"})
        temp = archive.with_suffix(archive.suffix + ".partial")
        with urllib.request.urlopen(req, timeout=120) as response, temp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        temp.rename(archive)
    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        for entry in tar:
            if not entry.isfile():
                continue
            # Only official original validation, never infer clean from an edited variant.
            pieces = Path(entry.name).parts
            for pos in range(len(pieces)-3):
                if pieces[pos:pos+2] == ("original", "val"):
                    class_dir = pieces[pos+2]
                    filename = pieces[pos+3]
                    if class_dir in CLASS_NAMES and len(pieces) == pos + 4 and Path(filename).suffix.lower() in (".jpeg",".jpg",".png"):
                        target = root / class_dir / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            stream = tar.extractfile(entry)
                            if stream is None:
                                continue
                            with target.open("wb") as out:
                                while True:
                                    block = stream.read(1024 * 1024)
                                    if not block:
                                        break
                                    out.write(block)
                        extracted += 1
                    break
    if extracted < 100:
        raise RuntimeError(
            f"Could not locate ImageNet-9 original/val images in official archive (extracted {extracted}). "
            "Provide --clean-root pointing to the existing original/val folder, "
            "or inspect the release archive layout."
        )
    print(f"Official clean originals ready: {extracted} files at {root}", flush=True)


def select_matched(api, clean_index, count, n, seed):
    """Balance 30 original stems across nine benchmark classes; same stem all variants."""
    by_class = {}
    for class_dir in CLASS_NAMES:
        stem_to_paths = {}
        for variant in VARIANTS[1:]:
            p = BASE + "/" + class_dir + "/" + str(n) + "/" + variant
            paths = remote_files(api, p)
            print(f"Hub catalog {class_dir} {variant}: {len(paths)} files", flush=True)
            stem_to_paths[variant] = paths
        if not stem_to_paths:
            continue
        shared = set.intersection(*(set(m) for m in stem_to_paths.values()))
        shared = sorted(s for s in shared if (class_dir, s) in clean_index)
        random.Random(seed + list(CLASS_NAMES).index(class_dir) * 1009).shuffle(shared)
        by_class[class_dir] = [
            {"id": class_dir + "/" + stem,
             "base_stem": stem, "class_dir": class_dir, "label": CLASS_NAMES[class_dir],
             "images_remote": {"clean": clean_index[(class_dir, stem)]} |
                 {v: stem_to_paths[v][stem] for v in VARIANTS[1:]}}
            for stem in shared
        ]
    chosen = []
    while len(chosen) < count:
        advanced = False
        for class_dir in CLASS_NAMES:
            bucket = by_class.get(class_dir, [])
            if bucket:
                chosen.append(bucket.pop())
                advanced = True
                if len(chosen) == count:
                    break
        if not advanced:
            raise RuntimeError(f"Only {len(chosen)} fully matched original images exist for requested {count}.")
    return chosen


def prepare(args):
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    if manifest.exists():
        entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(entries) != args.count:
            raise ValueError(f"Existing manifest has {len(entries)} pairs, requested {args.count}. Use a new --out.")
        if any(row.get("n_distractors") != args.n_distractors for row in entries):
            raise ValueError("Existing manifest distractor count differs. Use a new --out.")
        for row in entries:
            for variant in VARIANTS:
                if not Path(row["images"][variant]).is_file():
                    raise FileNotFoundError(f"Previously prepared {variant} image missing: {row['images'][variant]}")
        print(f"Reusing EXACT {len(entries)} pairs from {manifest}", flush=True)
        return manifest

    clean_root = Path(args.clean_root).expanduser().resolve() if args.clean_root else out / "official_clean"
    clean = index_clean(clean_root)
    if not clean and args.clean_root:
        raise FileNotFoundError(f"No original ImageNet-9 files found under --clean-root {clean_root}")
    if not clean:
        download_official_clean(clean_root)
        clean = index_clean(clean_root)
    if not clean:
        raise RuntimeError(f"No ImageNet-9 original/val images found under {clean_root}")

    api = HfApi()
    chosen = select_matched(api, clean, args.count, args.n_distractors, args.seed)
    data = []
    for i, item in enumerate(chosen, 1):
        source = item["images_remote"]
        images = {"clean": str(Path(source["clean"]).resolve())}
        for variant in VARIANTS[1:]:
            downloaded = hf_hub_download(repo_id=REPO, repo_type="dataset",
                                         filename=source[variant],
                                         local_dir=str(out / "hf_idis"))
            images[variant] = str(Path(downloaded).resolve())
        data.append({"id":item["id"],"class_dir":item["class_dir"],
                     "base_stem":item["base_stem"],"label":item["label"],
                     "n_distractors":args.n_distractors,"images":images,
                     "text_q2":"What is 17 + 23?", "text_q2_gold":"40"})
        print(f"Selected [{i}/{args.count}]: {item['id']}", flush=True)
    tmp = manifest.with_suffix(".jsonl.partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(manifest)
    (out / "prepare_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Prepared exactly {args.count} matched originals × 4 image variants at {manifest}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",default="idis30/data")
    ap.add_argument("--count",type=int,default=30)
    ap.add_argument("--n-distractors",type=int,choices=[1,2,3,4],default=4)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--clean-root",default="",help="Optional existing official ImageNet-9 original/val; otherwise auto-download official test archive")
    args = ap.parse_args()
    if args.count < 1:
        ap.error("--count must be >= 1")
    prepare(args)


if __name__=="__main__":
    main()
