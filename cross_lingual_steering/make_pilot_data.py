#!/usr/bin/env python3
"""Generate paired EN/ZH concept corpora and a small synthetic visual pilot.

This is intentionally a sanity-check dataset, NOT a real-world benchmark.
"""
import argparse
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

NUM_EN = ["one", "two", "three", "four", "five"]
NUM_ZH = ["一", "二", "三", "四", "五"]
PLACES = [("garden", "花园"), ("room", "房间"), ("yard", "院子"), ("park", "公园")]
OBJECTS = [("stars", "星星"), ("circles", "圆形"), ("balls", "球"), ("cups", "杯子"),
           ("flowers", "花"), ("books", "书"), ("chairs", "椅子"), ("flags", "旗子")]
SPATIAL = [("left", "左边"), ("right", "右边"), ("above", "上方"), ("below", "下方")]
COLORS = [("red", "红色"), ("blue", "蓝色"), ("green", "绿色"), ("yellow", "黄色")]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def corpora(out, n, seed):
    rng = random.Random(seed)
    rows = {"counting": {"en": [], "zh": []}, "spatial": {"en": [], "zh": []}}
    for i in range(n):
        number = i % 5
        place = PLACES[(i // 5) % len(PLACES)]
        obj = OBJECTS[(i // 20) % len(OBJECTS)]
        color = COLORS[(i // 160) % len(COLORS)]
        en = f"There are {NUM_EN[number]} {color[0]} {obj[0]} in the {place[0]}."
        zh = f"{place[1]}里有{NUM_ZH[number]}个{color[1]}{obj[1]}。"
        rows["counting"]["en"].append({"id": i, "sentence": en, "target": NUM_EN[number]})
        rows["counting"]["zh"].append({"id": i, "sentence": zh, "target": NUM_ZH[number]})
        direction = SPATIAL[i % 4]
        # Vary the context independently of the target direction.
        first = COLORS[(i // 4) % 4]
        second = COLORS[((i // 4) + 1) % 4]
        en = f"The {first[0]} square is {direction[0]} of the {second[0]} circle."
        zh = f"{first[1]}方块在{second[1]}圆形的{direction[1]}。"
        rows["spatial"]["en"].append({"id": i, "sentence": en, "target": direction[0]})
        rows["spatial"]["zh"].append({"id": i, "sentence": zh, "target": direction[1]})

    # Shuffle both languages with a common permutation; ids stay paired.
    order = list(range(n))
    rng.shuffle(order)
    for concept in rows:
        for lang in rows[concept]:
            data = rows[concept][lang]
            write_jsonl(out / "corpus" / f"{concept}_{lang}.jsonl", [data[j] for j in order])


def draw_count(image, n, rng):
    d = ImageDraw.Draw(image)
    spots = [(x, y) for x in (60, 140, 220, 300) for y in (60, 145, 225)]
    rng.shuffle(spots)
    for x, y in spots[:n]:
        d.ellipse((x-18, y-18, x+18, y+18), fill=(215, 44, 44), outline="black", width=2)
    for x, y in spots[n:n+3]:
        d.ellipse((x-18, y-18, x+18, y+18), fill=(40, 92, 220), outline="black", width=2)


def draw_spatial(image, direction):
    d = ImageDraw.Draw(image)
    anchor = (200, 150)
    offsets = {"left": (-105, 0), "right": (105, 0), "above": (0, -95), "below": (0, 95)}
    rx, ry = (anchor[0] + offsets[direction][0], anchor[1] + offsets[direction][1])
    d.ellipse((anchor[0]-24, anchor[1]-24, anchor[0]+24, anchor[1]+24),
              fill=(40, 92, 220), outline="black", width=2)
    d.rectangle((rx-23, ry-23, rx+23, ry+23), fill=(215, 44, 44), outline="black", width=2)


def visual_set(out, n_val, n_test, seed):
    rng = random.Random(seed + 100_003)
    rows = []
    imgdir = out / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    for split, n in (("val", n_val), ("test", n_test)):
        for task in ("counting", "spatial"):
            for i in range(n):
                img = Image.new("RGB", (400, 300), "white")
                if task == "counting":
                    answer = str(rng.randint(1, 5))
                    draw_count(img, int(answer), rng)
                    questions = {
                        "en": "How many red circles are in this image? Answer with one digit only.",
                        "zh": "图片里有几个红色圆形？只回答一个阿拉伯数字。",
                    }
                else:
                    answer = SPATIAL[rng.randrange(4)][0]
                    draw_spatial(img, answer)
                    questions = {
                        "en": "Where is the red square relative to the blue circle? Answer left, right, above, or below.",
                        "zh": "红色方块相对蓝色圆形在什么方向？只回答左、右、上或下。",
                    }
                path = imgdir / f"{split}_{task}_{i:04d}.png"
                img.save(path)
                rows.append({"id": f"{split}_{task}_{i:04d}", "split": split,
                             "task": task, "image": str(path.relative_to(out)),
                             "answer": answer, "questions": questions})
    write_jsonl(out / "visual_eval.jsonl", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/pilot")
    p.add_argument("--n-corpus", type=int, default=160)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if min(a.n_corpus, a.n_val, a.n_test) < 1:
        p.error("All sample counts must be positive")
    out = Path(a.out)
    corpora(out, a.n_corpus, a.seed)
    visual_set(out, a.n_val, a.n_test, a.seed)
    print(f"Generated paired corpora and visual evaluation samples in {out.resolve()}")


if __name__ == "__main__":
    main()
