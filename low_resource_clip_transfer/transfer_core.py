"""Dataset, alignment and paired metrics for zero-shot CLIP language transfer.

All examples come from the *unseen CIFAR-10 test split*. Rosetta maps are fitted
separately on Flickr8K-CN training data; this module never fits any parameters.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np

CIFAR10_EN = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_ZH = (
    "飞机", "轿车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车",
)

# Keep these fixed for all arms; zero-shot prototypes, not learned labels.
ZH_TEMPLATES = (
    "这是一张{label}的照片。",
    "图片中的主要物体是{label}。",
)
EN_TEMPLATES = (
    "a photo of {label}.",
    "a picture featuring {label}.",
)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
RECORD_BYTES = 1 + 32 * 32 * 3


def sha256_file(filename):
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def md5_file(filename):
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def decode_test_records(payload: bytes):
    """Official CIFAR-10 binary: label byte then CHW uint8 (R,G,B) pixels."""
    if len(payload) != 10_000 * RECORD_BYTES:
        raise ValueError(f"Expected exactly 10k CIFAR-10 test records; got {len(payload)} bytes")
    rec = np.frombuffer(payload, dtype=np.uint8).reshape(10_000, RECORD_BYTES)
    labels = rec[:, 0].astype(np.int64).copy()
    if (labels > 9).any():
        raise ValueError("CIFAR-10 label outside 0..9")
    images = rec[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return images, labels


def read_cifar_test(archive: Path):
    """Read the expected member only; never extract untrusted archive paths."""
    archive = Path(archive)
    if md5_file(archive) != CIFAR10_MD5:
        raise ValueError(f"CIFAR archive checksum mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("cifar-10-batches-bin/test_batch.bin")
        if not member.isfile() or member.size != 10_000 * RECORD_BYTES:
            raise ValueError("Unexpected CIFAR test member size/type")
        f = tar.extractfile(member)
        if f is None:
            raise ValueError("Missing CIFAR test data")
        return decode_test_records(f.read())


def select_balanced_indices(labels, num_images: int, seed: int):
    """Fixed-stratified test subset. num_images=0 means full official test split."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Invalid CIFAR-10 labels")
    if num_images == 0:
        return np.arange(len(labels), dtype=np.int64)
    if num_images < 10 or num_images > len(labels):
        raise ValueError("Choose 10..10000 images, or 0 for all")
    rng = np.random.default_rng(seed)
    base, extra = divmod(num_images, 10)
    chosen = []
    for c in range(10):
        pool = np.flatnonzero(labels == c)
        count = base + int(c < extra)
        if len(pool) < count:
            raise ValueError(f"Not enough test images for class {c}")
        chosen.extend(rng.choice(pool, count, replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite two-dimensional embedding matrix")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def make_prompts(names, templates):
    if len(names) != 10 or not templates:
        raise ValueError("Expected 10 labels and nonempty templates")
    return [t.format(label=name) for name in names for t in templates]


def class_prototypes(x, n_templates):
    """Normalize each template, average across identical class prompts, normalize."""
    x = unit(x)
    if len(x) != 10 * n_templates:
        raise ValueError("Incorrect template count")
    return unit(x.reshape(10, n_templates, -1).mean(axis=1))


def project_map(source_pca, checkpoint):
    """Replicate WassersteinProcrustes transform on existing PCA-normalized L."""
    source_pca = unit(source_pca)
    w = np.asarray(checkpoint["W"], dtype=np.float64)
    mx = np.asarray(checkpoint["mean_x"], dtype=np.float64)
    my = np.asarray(checkpoint["mean_y"], dtype=np.float64)
    d = source_pca.shape[1]
    if w.shape != (d, d) or mx.shape != (1, d) or my.shape != (1, d):
        raise ValueError("Rosetta checkpoint/PCA shape mismatch")
    if not np.isfinite(w).all() or not np.isfinite(mx).all() or not np.isfinite(my).all():
        raise ValueError("Nonfinite checkpoint")
    if not np.allclose(w.T @ w, np.eye(d), atol=0.03):
        raise ValueError("Rosetta W is not orthogonal")
    return unit(unit(source_pca - mx) @ w)


def project_images(raw_img, image_pca, bank_mean):
    """Images use the FIXED training-image-bank mean, never an English map mean."""
    z = image_pca.transform(raw_img)
    mean = np.asarray(bank_mean)
    if mean.shape != (1, z.shape[1]):
        raise ValueError("Incorrect image-bank mean dimensions")
    return unit(z - mean)


def classify(image_embs, class_embs):
    if class_embs.shape[0] != 10 or image_embs.shape[1] != class_embs.shape[1]:
        raise ValueError("Expected 10 class vectors in common CLIP space")
    similarity = unit(image_embs) @ unit(class_embs).T
    return np.argmax(similarity, axis=1), similarity


def paired_bootstrap(labels, base_pred, alternative_pred, seed=42, n_boot=2000):
    """Per-image paired bootstrap, stratified by class (fixed class weighting)."""
    labels = np.asarray(labels)
    b = np.asarray(base_pred)
    a = np.asarray(alternative_pred)
    if labels.ndim != 1 or a.shape != labels.shape or b.shape != labels.shape:
        raise ValueError("Mismatched label/prediction lengths")
    if len(labels) == 0 or n_boot < 1:
        raise ValueError("Empty test set or invalid bootstrap")
    delta = (a == labels).astype(int) - (b == labels).astype(int)
    by_class = [np.flatnonzero(labels == c) for c in range(10)]
    if any(len(x) == 0 for x in by_class):
        raise ValueError("Require all 10 classes for stratified bootstrap")
    rng = np.random.default_rng(seed)
    samples = np.zeros(n_boot)
    for j in range(n_boot):
        pick = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in by_class])
        samples[j] = 100 * float(delta[pick].mean())
    return {
        "delta_pp": float(100 * delta.mean()),
        "ci95_pp": [float(x) for x in np.quantile(samples, [.025, .975])],
        "repairs": int((delta > 0).sum()),
        "harms": int((delta < 0).sum()),
    }