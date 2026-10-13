#!/usr/bin/env python3
"""Download a stratified, reasoning-only subset of the official MME-CoT test set."""
import argparse
import base64
import json
import random
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def choices_from_row(row):
    choices = []
    for letter in "ABCDEFGHIJKL":
        value = row.get(letter)
        if value is not None and str(value).strip() not in ("", "nan", "None"):
            choices.append(f"{letter}. {str(value).strip()}")
    return choices


def to_pil(value):
    """Decode official MME-CoT images (base64 strings) or regular PIL/HF images.

    MME-CoT uses a sequence of base64-encoded JPEG/PNG strings, NOT file paths.
    Only short, existing local strings are interpreted as filesystem paths.
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return to_pil(value["bytes"])
        if value.get("path"):
            return to_pil(value["path"])
        raise ValueError("Image dictionary contains neither bytes nor path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, Path):
        return Image.open(value).convert("RGB")
    if isinstance(value, str):
        data = value.strip()
        if data.startswith("data:"):
            if ";base64," not in data:
                raise ValueError("Expected a base64 image data URI")
            data = data.split(",", 1)[1]
        # Avoid handing ~100 KB base64 strings to os.stat/open:
        # this was the original source of OSError: File name too long.
        if len(data) < 240:
            path = Path(data)
            if path.is_file():
                return Image.open(path).convert("RGB")
        try:
            decoded = base64.b64decode(re.sub(r"\\s+", "", data), validate=True)
            return Image.open(BytesIO(decoded)).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Cannot decode image as a Base64 image; "
                f"input length={len(data)}, prefix={data[:24]!r}"
            ) from exc
    raise TypeError(f"Unsupported image payload type: {type(value).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot_data")
    ap.add_argument("--max-examples", type=int, default=18)
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--categories", default="all",
                    help="Comma-separated official category names, or all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    print("Downloading the official CaraJ/MME-CoT test parquet (~279 MB)...", flush=True)
    ds = load_dataset("CaraJ/MME-CoT", split="test")
    allowed = None if args.categories.lower() == "all" else {
        s.strip().lower() for s in args.categories.split(",") if s.strip()
    }
    pools = defaultdict(list)
    skipped = defaultdict(int)
    for row in ds:
        typ = str(row.get("question_type") or "").strip().lower()
        if typ != "reasoning":
            skipped["non_reasoning"] += 1
            continue
        category = str(row.get("category") or "unknown").strip()
        if allowed is not None and category.lower() not in allowed:
            continue
        images = row.get("image")
        if not images:
            skipped["no_image"] += 1
            continue
        if not str(row.get("question") or "").strip():
            skipped["no_question"] += 1
            continue
        pools[category].append(row)
    if not pools:
        raise SystemExit("No eligible MME-CoT Reasoning examples. Check --categories.")
    for arr in pools.values():
        rng.shuffle(arr)

    categories = sorted(pools)
    print("Eligible reasoning examples by category:",
          {k: len(pools[k]) for k in categories}, flush=True)
    selected = []
    # Round-robin across official categories, rather than taking first N parquet rows.
    for round_id in range(args.per_category):
        for category in categories:
            if len(selected) >= args.max_examples:
                break
            arr = pools[category]
            if round_id < len(arr):
                selected.append(arr[round_id])
        if len(selected) >= args.max_examples:
            break
    if not selected:
        raise SystemExit("No examples selected; check count arguments.")

    records = []
    for number, row in enumerate(selected):
        index = str(row.get("index", number))
        safe_index = re.sub(r"[^a-zA-Z0-9_-]", "_", index)
        image_paths = []
        for image_no, image in enumerate(row["image"]):
            if image is None:
                continue
            obj = to_pil(image)
            file = out / "images" / f"{number:04d}_{safe_index}_{image_no}.png"
            file.parent.mkdir(parents=True, exist_ok=True)
            obj.save(file)
            image_paths.append(str(file))
        if not image_paths:
            continue
        record = {
            "id": index,
            "question": str(row["question"]).strip(),
            "options": choices_from_row(row),
            "answer": str(row.get("answer") or "").strip(),
            "category": str(row.get("category") or "unknown"),
            "subcategory": str(row.get("subcategory") or ""),
            "question_type": str(row.get("question_type") or ""),
            "image_paths": image_paths,
            "key_annotation_steps": row.get("key_annotation_steps"),
        }
        records.append(record)
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    (out / "settings.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Prepared:", len(records), "Reasoning examples (two prompt conditions each)", flush=True)
    print("Output:", out / "pairs.jsonl", flush=True)


if __name__ == "__main__":
    main()
