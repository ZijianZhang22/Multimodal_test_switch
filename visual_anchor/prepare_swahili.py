"""Prepare a Swahili L-space for the existing Flickr8K-CN Rosetta3 protocol.

Reuses E/I features, image IDs and all fixed group splits. Only Swahili L
embeddings change. Swahili caption rows are aligned to EN by both image ID and
normalized English text; image-only fallback is opt-in and explicitly audited.

SwaFlickr8k: https://huggingface.co/datasets/Alfaxad/SwaFlickr8k
"""
import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from prepare_real import atomic_npz, parse_captions
from run import validate

DATASET_ID = "Alfaxad/SwaFlickr8k"
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
L_KEYS = ("l_pair", "l_pool", "l_test", "l_img_test")


def norm_english(value):
    """Normalize Flickr8k English captions to avoid whitespace/punctuation mismatches."""
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", value) if c.isalnum())


def load_swahili_csv(path):
    """Map each image stem to ordered triples (normalized_en, en, sw)."""
    by_image = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"image", "caption", "caption_SW"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError("Expected CSV columns image, caption, caption_SW; got {}".format(reader.fieldnames))
        for row in reader:
            stem = Path(row["image"].strip()).stem
            en, sw = row["caption"].strip(), row["caption_SW"].strip()
            if not stem or not en or not sw:
                continue
            by_image[stem].append((norm_english(en), en, sw))
    if not by_image:
        raise ValueError("No valid Swahili caption rows")
    return dict(by_image)


def aligned_caption(by_image, image_id, english, allow_image_fallback=False):
    """Return caption + exact_match flag; never silently mix English caption identities."""
    rows = by_image.get(str(image_id), [])
    if not rows:
        raise ValueError("SwaFlickr8k has no captions for image {}".format(image_id))
    english_key = norm_english(english)
    for key, en, sw in rows:
        if key == english_key:
            return sw, True
    if not allow_image_fallback:
        raise ValueError("No exact English-caption match for image {}: {!r}. "
                         "Use --allow-image-fallback ONLY after auditing.".format(image_id, english))
    return rows[0][2], False


def independent_image_caption(by_image, image_id, target_english):
    """Find a distinct caption of the same image for L-I evaluation, when possible."""
    rows = by_image.get(str(image_id), [])
    if not rows:
        raise ValueError("No Swahili image caption for {}".format(image_id))
    key = norm_english(target_english)
    alternatives = [row[2] for row in rows if row[0] != key]
    return (alternatives[0] if alternatives else rows[0][2],
            bool(alternatives))


def build_text_fields(base, en_by_image, by_image, allow_image_fallback=False):
    """Maintain all existing ID arrays and fixed train/test splits."""
    result = {}
    audits = {"pair_exact": 0, "pair_image_fallback": 0,
              "test_exact": 0, "test_image_fallback": 0,
              "test_distinct_img_caption": 0}
    pair_texts = []
    for im in base["l_pair_ids"]:
        en = en_by_image[(str(im), 0)]
        caption, exact = aligned_caption(by_image, im, en, allow_image_fallback)
        pair_texts.append(caption)
        audits["pair_exact" if exact else "pair_image_fallback"] += 1
    result["l_pair"] = pair_texts

    # These captions belong to different images from the English Y pool.
    result["l_pool"] = [by_image[str(im)][0][2] for im in base["l_pool_ids"]]

    test_texts, image_test_texts = [], []
    for im in base["l_test_ids"]:
        en = en_by_image[(str(im), 0)]
        caption, exact = aligned_caption(by_image, im, en, allow_image_fallback)
        test_texts.append(caption)
        audits["test_exact" if exact else "test_image_fallback"] += 1
    for im in base["l_img_test_ids"]:
        en = en_by_image[(str(im), 0)]
        caption, distinct = independent_image_caption(by_image, im, en)
        image_test_texts.append(caption)
        audits["test_distinct_img_caption"] += int(distinct)
    result["l_test"], result["l_img_test"] = test_texts, image_test_texts

    # PCA is fitted on train-only L_fit, as in prepare_real.py.
    result["l_fit"] = pair_texts + result["l_pool"]
    for key, texts in result.items():
        if len(texts) != len(base[key + "_ids"]):
            raise ValueError("ID/Swahili text count mismatch in {}".format(key))
    if not np.array_equal(
        base["l_fit_ids"],
        np.concatenate((base["l_pair_ids"], base["l_pool_ids"]))
    ):
        raise ValueError("Unexpected L_fit ID order in source dataset")
    return result, audits


def swahili_vectors(texts, model_id, revision, device, batch_size, max_length):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                              trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, revision=revision,
                                      trust_remote_code=False).to(device).eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        part = texts[start:start + batch_size]
        # Multilingual E5 requires the query prefix for symmetric text matching.
        tokens = tokenizer(["query: " + text for text in part],
                           padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.float().cpu().numpy())
        print("Swahili encoded: {}/{}".format(min(start + batch_size, len(texts)), len(texts)),
              flush=True)
    return np.concatenate(chunks).astype(np.float32)


def cached_vectors(cache_path, texts, model_id, revision, device, batch_size, max_length):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        array = np.load(cache_path, allow_pickle=False)
        if array.shape[0] != len(texts) or not np.isfinite(array).all():
            raise ValueError("Invalid cached Swahili features: {}".format(cache_path))
        return array
    vectors = swahili_vectors(texts, model_id, revision, device, batch_size, max_length)
    temporary = cache_path.with_suffix(".tmp.npy")
    np.save(temporary, vectors)
    temporary.replace(cache_path)
    return vectors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-data", type=Path, default=Path("real_work/features.npz"))
    p.add_argument("--out", type=Path, default=Path("real_work/features_sw.npz"))
    p.add_argument("--english-captions", type=Path,
                   default=Path("real_work/raw/flickr8kenc.caption.txt"))
    p.add_argument("--csv", type=Path, help="Local SwaFlickr8k CSV; otherwise download pinned HF dataset")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--allow-image-fallback", action="store_true",
                   help="Allow unequal English captions for the same image; logged in manifest")
    a = p.parse_args()
    if a.batch_size < 1 or a.max_length < 1:
        p.error("batch-size/max-length must be positive")
    if not a.base_data.exists() or not a.english_captions.exists():
        p.error("First run prepare_real.py and retain its raw English caption file")
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; set --device cpu")
    from huggingface_hub import hf_hub_download, list_repo_files, model_info, repo_info

    dataset_sha = None
    if a.csv is None:
        dataset_sha = repo_info(DATASET_ID, repo_type="dataset").sha
        files = list_repo_files(DATASET_ID, repo_type="dataset", revision=dataset_sha)
        matches = [name for name in files if name.lower().endswith(".csv")]
        if len(matches) > 1:
            preferred = [name for name in matches if "caption" in name.lower() or "swaflickr" in name.lower()]
            if len(preferred) == 1:
                matches = preferred
        if len(matches) != 1:
            p.error("Could not uniquely identify Swahili caption CSV: {}".format(matches))
        csv_path = Path(hf_hub_download(DATASET_ID, matches[0],
                                        repo_type="dataset", revision=dataset_sha))
    else:
        csv_path = a.csv
    model_sha = model_info(a.model).sha
    base = dict(np.load(a.base_data, allow_pickle=False))
    validate(base)
    by_image = load_swahili_csv(csv_path)
    en_by_image = parse_captions(a.english_captions.read_text(encoding="utf-8"))
    fields, audits = build_text_fields(base, en_by_image, by_image,
                                      allow_image_fallback=a.allow_image_fallback)
    print("Swahili alignment audit:", audits, flush=True)

    output = dict(base)
    for key in ("l_fit",) + L_KEYS:
        texts = fields[key]
        digest = hashlib.sha256(("\n".join(texts) + model_sha +
                                 str(a.max_length)).encode("utf-8")).hexdigest()[:20]
        cache_path = a.out.parent / "sw_cache" / (key + "_" + digest + ".npy")
        output[key] = cached_vectors(cache_path, texts, a.model, model_sha,
                                     a.device, a.batch_size, a.max_length)
    validate(output)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(a.out, **output)
    manifest = {
        "language": "sw", "dataset_id": DATASET_ID, "dataset_revision": dataset_sha,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_features": str(a.base_data), "output": str(a.out),
        "model": a.model, "model_revision": model_sha,
        "model_input_prefix": "query: ", "max_length": a.max_length,
        "audit": audits,
        "notes": [
            "English/image spaces and all ID splits copied from the original feature file.",
            "Swahili captions are caption-aligned to English by image and normalized English sentence.",
            "Swahili caption provenance and translation quality require independent audit.",
            "L-I uses an alternate Swahili caption of the same image when available.",
            "Do not fit PCA on held-out test embeddings.",
        ],
    }
    path = a.out.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Swahili features ready:", a.out, "manifest:", path, flush=True)


if __name__ == "__main__":
    main()
