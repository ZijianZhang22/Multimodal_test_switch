#!/usr/bin/env python3
"""Prepare distinct MathVerse questions with existing official Idis-math images.

Downloads selected official images with optional exclusion of earlier MathVerse
problem IDs (across all problem versions). Never generates/edits distractors.
"""
import argparse
import json
import random
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

IDIS_REPO = "Vail-2000/Idis"
MATHVERSE_REPO = "AI4Math/MathVerse"
REMOTE_ROOT = "Idis-math/visual_distractor"
VARIANTS = ("aligned", "conflicting", "irrelevant")


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def remote_filename(entry, variant, count):
    raw = entry.get("augmented", {}).get("n" + str(count))
    if not raw:
        return None
    raw = str(raw).replace("\\", "/")
    parts = PurePosixPath(raw).parts
    n_key = "n" + str(count)
    if n_key not in parts:
        raise ValueError(f"Unknown Idis image path format: {raw}")
    i = parts.index(n_key)
    rest = parts[i + 1:]
    if not rest:
        raise ValueError(f"Missing image filename in: {raw}")
    return "/".join((REMOTE_ROOT, variant, n_key, *rest))


def download_image(filename, folder, api, path_cache):
    """Use metadata path, with a repo-tree fallback if Hub naming differs."""
    try:
        return hf_hub_download(repo_id=IDIS_REPO, repo_type="dataset",
                               filename=filename, local_dir=str(folder))
    except Exception as first:
        # Lazy one-time listing of the selected variant/count subtree.
        base = filename.split("/n")[0]
        n = filename.split("/n", 1)[1].split("/", 1)[0]
        subtree = base + "/n" + n
        if subtree not in path_cache:
            entries = api.list_repo_tree(repo_id=IDIS_REPO, repo_type="dataset",
                                         path_in_repo=subtree, recursive=True)
            path_cache[subtree] = [x.path for x in entries if
                                   x.path.lower().endswith((".png", ".jpg", ".jpeg"))]
        trailing = filename.split(subtree + "/", 1)[-1]
        hits = [p for p in path_cache[subtree] if p.endswith("/" + trailing)]
        if not hits:
            # Basename-only fallback is safe only when unique.
            hits = [p for p in path_cache[subtree]
                    if PurePosixPath(p).name == PurePosixPath(filename).name]
        if len(hits) != 1:
            raise RuntimeError(
                f"Cannot uniquely resolve official image {filename} "
                f"(matches={len(hits)}); initial error: {first}"
            ) from first
        return hf_hub_download(repo_id=IDIS_REPO, repo_type="dataset",
                               filename=hits[0], local_dir=str(folder))


def priority(item):
    version = str(item.get("problem_version", "")).lower()
    if "vision intensive" in version: return 0
    if "vision dominant" in version: return 1
    if "vision only" in version: return 2
    if "text dominant" in version: return 3
    return 4


def load_exclusions(ids_file=None, manifests=None):
    """Load old sample IDs AND underlying problem IDs (across all versions)."""
    samples, problems = set(), set()
    if ids_file:
        p = Path(ids_file).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Exclusion file not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        samples |= {str(x) for x in payload.get("sample_indices", [])}
        problems |= {str(x) for x in payload.get("problem_indices", [])}
        if not samples and not problems:
            raise ValueError(f"Exclusion file contains no IDs: {p}")
    for fname in manifests or []:
        p = Path(fname).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Old manifest not found: {p}")
        for r in load_jsonl(p):
            if r.get("sample_index") is not None:
                samples.add(str(r["sample_index"]))
            if r.get("problem_index") is not None:
                problems.add(str(r["problem_index"]))
    return samples, problems


def prepare(args):
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    denied_samples, denied_problems = load_exclusions(
        args.exclude_ids_file, args.exclude_manifest
    )
    print(f"Excluding {len(denied_samples)} prior sample IDs and "
          f"{len(denied_problems)} prior problem IDs", flush=True)
    if manifest.exists():
        rows = load_jsonl(manifest)
        if len(rows) != args.count:
            raise ValueError(f"Manifest contains {len(rows)} tasks; requested {args.count}. Choose a new OUT.")
        if any(r["variant"] != args.variant or r["n_distractors"] != args.n_distractors for r in rows):
            raise ValueError("Manifest variant/count mismatch; use a new OUT.")
        for row in rows:
            if (str(row["sample_index"]) in denied_samples or
                    str(row["problem_index"]) in denied_problems):
                raise ValueError(
                    f"Existing manifest overlaps excluded problems: {row['sample_index']}"
                )
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(f"Previously downloaded image missing: {row['image']}")
        print(f"Reusing manifest with {len(rows)} identical image-question pairs: {manifest}", flush=True)
        return manifest

    testmini_path = hf_hub_download(repo_id=MATHVERSE_REPO, repo_type="dataset",
                                    filename="testmini.json")
    testmini = json.loads(Path(testmini_path).read_text(encoding="utf-8"))
    if isinstance(testmini, dict):
        testmini = list(testmini.values())
    indices = {str(r["sample_index"]): r for r in testmini if r.get("sample_index") is not None}
    print(f"Official MathVerse testmini indexed: {len(indices)} rows", flush=True)

    meta_file = REMOTE_ROOT + "/" + args.variant + "/meta.jsonl"
    meta_path = hf_hub_download(repo_id=IDIS_REPO, repo_type="dataset",
                                filename=meta_file)
    meta = load_jsonl(meta_path)
    rng = random.Random(args.seed)
    rng.shuffle(meta)
    # Prioritize visually demanding MathVerse versions, but DO NOT cherry-pick by model result.
    meta.sort(key=lambda m: priority(indices.get(str(m.get("sample_index")), {})))
    seen_samples, seen_problems, selected = set(), set(), []
    for m in meta:
        sid = str(m.get("sample_index", ""))
        original = indices.get(sid)
        if not original:
            continue
        question = str(original.get("question", "")).strip()
        answer = str(original.get("answer", "")).strip()
        if not question or not answer:
            continue
        problem_key = str(original.get("problem_index") or sid)
        if (sid in seen_samples or problem_key in seen_problems
                or sid in denied_samples or problem_key in denied_problems):
            continue
        remote = remote_filename(m, args.variant, args.n_distractors)
        if not remote:
            continue
        seen_samples.add(sid)
        seen_problems.add(problem_key)
        selected.append({
            "sample_index":sid, "problem_index":problem_key,
            "problem_version":str(original.get("problem_version", "")),
            "question":question, "answer":answer,
            "question_for_eval":str(original.get("question_for_eval") or question),
            "image_remote":remote, "variant":args.variant,
            "n_distractors":args.n_distractors,
        })
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        raise RuntimeError(f"Found only {len(selected)} eligible unique problems; need {args.count}")

    api=HfApi()
    cache={}
    data=[]
    for i, row in enumerate(selected, 1):
        image = download_image(row["image_remote"], out / "hf_idis", api, cache)
        row["image"] = str(Path(image).resolve())
        data.append(row)
        print(f"[{i}/{args.count}] MathVerse {row['sample_index']}, "
              f"{row['problem_version']}, {args.variant}/n{args.n_distractors}", flush=True)
    # Atomic write so an interrupted download doesn't produce a partial, resumable manifest.
    tmp=manifest.with_suffix(".jsonl.tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for row in data: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    tmp.replace(manifest)
    print("READY:",manifest, flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="idis_math10/data")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--variant", choices=VARIANTS, default="irrelevant")
    p.add_argument("--n-distractors", type=int,choices=[1,2,3,4],default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--exclude-ids-file", default=None,
                   help="JSON of prior sample_indices and problem_indices")
    p.add_argument("--exclude-manifest", action="append", default=[],
                   help="Previous pilot JSONL manifest; may be specified repeatedly")
    a=p.parse_args()
    if a.count<1:p.error("count must be >= 1")
    prepare(a)


if __name__=="__main__":
    main()
