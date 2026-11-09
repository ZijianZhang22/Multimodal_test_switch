"""Make Chinese/E5 features with EXACT same image IDs/splits/E/I/anchors as Swahili/E5.

Requires the finished real_work_clip_native/features_sw_clip.npz and its
manifest. English/image features are byte-for-byte carried forward; only
Chinese L vectors are re-encoded. Chinese text is taken from Flickr8K-CN:
Google translations of EN caption #0 for anchors/unpaired pool; human
translation of EN caption #0 for L-E tests; translated English caption #1
for L-I tests. This avoids independently described Chinese image captions
in the controlled comparison, while remaining explicit about translation
quality differences between datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_real import atomic_npz, parse_captions
from run import validate

L_KEYS = ("l_pair", "l_pool", "l_test", "l_img_test")
E5_MODEL = "intfloat/multilingual-e5-base"
PREFIX = "query: "


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chinese_fields(sw_data, google, human, english):
    """Build corresponding Chinese text rows in Swahili's *exact* ID order."""
    texts = {}
    counts = {"img_caption_index_1": 0, "img_caption_index_0_fallback": 0,
              "translated_english_index_0_pairs": 0, "human_english_index_0_test": 0}
    sources = {"l_pair": google, "l_pool": google, "l_test": human}
    for key, corpus in sources.items():
        out = []
        for image_id in sw_data[key + "_ids"]:
            im = str(image_id)
            if (im, 0) not in english:
                raise ValueError(f"English caption #0 absent for image {im} ({key})")
            if (im, 0) not in corpus:
                raise ValueError(f"Chinese caption #0 absent for image {im} ({key})")
            out.append(corpus[im, 0])
        texts[key] = out
    counts["translated_english_index_0_pairs"] = len(texts["l_pair"])
    counts["human_english_index_0_test"] = len(texts["l_test"])
    img = []
    for image_id in sw_data["l_img_test_ids"]:
        im = str(image_id)
        index = 1 if (im, 1) in google and (im, 1) in english else 0
        if (im, index) not in google or (im, index) not in english:
            raise ValueError(f"Missing aligned Chinese image test caption: {im}#{index}")
        counts[f"img_caption_index_{index}" + ("_fallback" if index == 0 else "")] += 1
        img.append(google[im, index])
    texts["l_img_test"] = img
    texts["l_fit"] = texts["l_pair"] + texts["l_pool"]
    for key, strings in texts.items():
        if len(strings) != len(sw_data[key + "_ids"]):
            raise ValueError(f"Chinese text / shared ID row count differs for {key}")
        if not all(s.strip() for s in strings):
            raise ValueError(f"Empty Chinese text in {key}")
    if not np.array_equal(sw_data["l_fit_ids"],
                          np.concatenate([sw_data["l_pair_ids"], sw_data["l_pool_ids"]])):
        raise ValueError("Unexpected Swahili L-fit ordering; cannot match train PCA")
    return texts, counts


def check_shared_protocol(chinese, swahili):
    """Assert every E/I representation and every row ID is exactly preserved."""
    if set(chinese) != set(swahili):
        raise ValueError("Keys in Chinese and Swahili features differ")
    for key in swahili:
        if key.startswith("l_") and not key.endswith("_ids"):
            if len(chinese[key]) != len(swahili[key]):
                raise ValueError(f"Low-resource row count mismatch: {key}")
            continue
        if not np.array_equal(chinese[key], swahili[key]):
            raise ValueError(f"Non-Chinese feature/ID array changed: {key}")
    validate(chinese)


def encode_fields(fields, cache_dir, model, revision, device, batch_size, max_length):
    """Match prepare_swahili's query-prefix, mean-pooling and truncation exactly."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    to_compute = {}
    for key in L_KEYS:
        contents = "\n".join(fields[key]) + model + revision + str(max_length) + PREFIX
        digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()[:20]
        path = cache_dir / f"{key}_{digest}.npy"
        if path.exists():
            vec = np.load(path, allow_pickle=False)
            if vec.ndim != 2 or vec.shape[0] != len(fields[key]) or not np.isfinite(vec).all():
                raise ValueError(f"Corrupted E5 feature cache {path}")
            existing[key] = vec
        else:
            to_compute[key] = path

    if to_compute:
        import torch
        from transformers import AutoTokenizer, AutoModel
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; pass --device cpu")
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        net = AutoModel.from_pretrained(model, revision=revision, trust_remote_code=False).to(device).eval()
        for key, path in to_compute.items():
            chunks = []
            strings = fields[key]
            for start in range(0, len(strings), batch_size):
                batch = strings[start:start+batch_size]
                inp = tokenizer([PREFIX + t for t in batch], return_tensors="pt",
                                padding=True, truncation=True, max_length=max_length).to(device)
                with torch.inference_mode():
                    hidden = net(**inp).last_hidden_state
                    mask = inp["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    chunks.append(pooled.float().cpu().numpy())
                print(f"Chinese E5: {key} {min(start+batch_size,len(strings))}/{len(strings)}", flush=True)
            vec = np.concatenate(chunks).astype(np.float32)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, vec)
            tmp.replace(path)
            existing[key] = vec
    existing["l_fit"] = np.concatenate([existing["l_pair"], existing["l_pool"]], axis=0)
    return existing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sw-data", type=Path, default=Path("real_work_clip_native/features_sw_clip.npz"))
    p.add_argument("--sw-manifest", type=Path, default=None)
    p.add_argument("--raw-dir", type=Path, default=Path("real_work_clip_native/raw"))
    p.add_argument("--out", type=Path, default=Path("real_work_clip_native/features_zh_e5_swmatched.npz"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    if a.sw_manifest is None:
        a.sw_manifest = a.sw_data.with_suffix(".manifest.json")
    for path in (a.sw_data, a.sw_manifest):
        if not path.is_file():
            p.error(f"Missing {path}. Run bash run_swahili_clip.sh first.")
    sw_meta = json.loads(a.sw_manifest.read_text(encoding="utf-8"))
    model, revision = sw_meta.get("model"), sw_meta.get("model_revision")
    max_length = sw_meta.get("max_length")
    if model != E5_MODEL or not isinstance(revision, str) or not revision or not isinstance(max_length, int):
        p.error("Swahili features were not encoded with pinned multilingual-e5-base metadata")
    if sw_meta.get("model_input_prefix") != PREFIX:
        p.error("Swahili E5 prefix differs; cannot claim a shared encoder protocol")
    if a.verify_only:
        if not a.out.is_file() or not a.out.with_suffix(".manifest.json").is_file():
            p.error(f"Missing Chinese features or manifest: {a.out}")
        meta = json.loads(a.out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (meta.get("sw_features_sha256") != sha256(a.sw_data)
                or meta.get("model_revision") != revision
                or meta.get("max_length") != max_length):
            p.error("Chinese output does not match current Swahili source/model; rebuild it")
        with np.load(a.sw_data, allow_pickle=False) as f:
            sw = dict(f)
        with np.load(a.out, allow_pickle=False) as f:
            zh = dict(f)
        check_shared_protocol(zh, sw)
        print("Verified identical IDs and E/I features:", a.out, flush=True)
        return
    raw = {"google": a.raw_dir / "flickr8kzhg.caption.txt",
           "human": a.raw_dir / "flickr8kzhmtest.captions.txt",
           "english": a.raw_dir / "flickr8kenc.caption.txt"}
    for name, path in raw.items():
        if not path.is_file():
            p.error(f"Missing {name} caption file {path}; first run BRIDGE_ONLY=1 bash run_clip_native.sh")
    corpora = {name: parse_captions(path.read_text(encoding="utf-8")) for name, path in raw.items()}
    with np.load(a.sw_data, allow_pickle=False) as f:
        sw = dict(f)
    validate(sw)
    texts, counts = chinese_fields(sw, corpora["google"], corpora["human"], corpora["english"])
    print("Chinese/Swahili matched caption audit:", counts, flush=True)
    vectors = encode_fields(texts, a.out.parent / "zh_e5_cache", model, revision,
                            a.device, a.batch_size, max_length)
    zh = dict(sw)
    for key in ("l_fit",) + L_KEYS:
        zh[key] = vectors[key]
    check_shared_protocol(zh, sw)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(a.out, **zh)
    meta = {
        "language": "zh", "source_language_encoder": model, "model_revision": revision,
        "model_input_prefix": PREFIX, "max_length": max_length,
        "sw_features": str(a.sw_data), "sw_features_sha256": sha256(a.sw_data),
        "sw_manifest": str(a.sw_manifest),
        "caption_sha256": {name: sha256(path) for name, path in raw.items()},
        "caption_protocol": {
            "l_pair": "Flickr8K-CN Google translation of English caption #0",
            "l_pool": "Flickr8K-CN Google translation of English caption #0",
            "l_test": "Flickr8K-CN human translation of English caption #0",
            "l_img_test": "Flickr8K-CN Google translation of English caption #1 (fallback #0, audited)",
        },
        "audit": counts,
        "training_anchor_ids_identical": True,
        "test_image_ids_identical": True,
        "english_image_embeddings_identical": True,
        "limits": ["Source texts are translated differently across datasets; same semantics not guaranteed.",
                   "Controlled source encoder and image IDs, not controlled pretraining-language coverage."]
    }
    manifest_path = a.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Chinese E5 matched features:", a.out, flush=True)
    print("Manifest:", manifest_path, flush=True)


if __name__ == "__main__":
    main()