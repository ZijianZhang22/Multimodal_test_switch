#!/usr/bin/env python3
"""Bilingual translations of the upstream taxonomy examples + simple image smoke test."""
import argparse
import json
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / ".upstream"
sys.path.insert(0, str(UPSTREAM / "get_features"))

# Chinese sentences are paired in ORDER with the author's exact English examples.
ZH = {
    "counting": [
        ("篮子里有三个苹果", "三"),
        ("老师数了五个孩子", "五"),
        ("他们买了一打鸡蛋", "一打"),
        ("我需要两杯糖", "两"),
        ("桌子上有一杯水", "一"),
        ("食谱需要四个鸡蛋", "四"),
        ("他在考试中犯了零个错误", "零"),
        ("她在比赛中赢得七枚奖牌", "七"),
        ("我看到电线上有六只鸟", "六"),
        ("盒子里有八支铅笔", "八"),
        ("她早餐吃了九颗葡萄", "九"),
        ("我们需要十名志愿者", "十"),
        ("包裹里有十一件物品", "十一"),
        ("一年有十二个月", "十二"),
        ("我在市场买了十五个苹果", "十五"),
        ("钟表显示已经过了二十分钟", "二十"),
        ("球队得到了三十分", "三十"),
        ("这个瓶子能装五十毫升水", "五十"),
        ("这本书有四十章", "四十"),
        ("大约六十个人参加了会议", "六十"),
    ],
    "spatial": [
        ("猫在桌子上面", "上面"),
        ("她把书放在椅子下面", "下面"),
        ("植物长在窗户旁边", "旁边"),
        ("灯挂在书桌上方", "上方"),
        ("钥匙在抽屉里面", "里面"),
        ("狗站在两棵树之间", "之间"),
        ("汽车停在房子后面", "后面"),
        ("孩子站在喷泉附近", "附近"),
        ("他把鞋子放在床底下", "底下"),
        ("画挂在壁炉上方", "上方"),
        ("玩具从架子上掉下来", "掉下来"),
        ("小船沿着河流漂浮", "沿着"),
        ("篱笆围绕着院子", "围绕"),
        ("小路通向山顶", "通向"),
        ("餐厅在银行对面", "对面"),
        ("雕像矗立在花园中央", "中央"),
        ("她站在商店左边", "左边"),
        ("钢笔沿着桌面向下滚动", "向下"),
        ("猫在椅子的右边", "右边"),
        ("汽车开过隧道", "开过"),
    ],
}
TAXONOMY = {"counting": "counting", "spatial": "spatial_relationship"}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_corpus(out):
    if not UPSTREAM.is_dir():
        raise RuntimeError("Clone upstream first: run bash run.sh")
    from taxonomy_sentences import generate_taxonomy_sentences
    for concept, taxonomy in TAXONOMY.items():
        english = generate_taxonomy_sentences(taxonomy)
        chinese = ZH[concept]
        if len(english) != len(chinese):
            raise RuntimeError(f"Upstream taxonomy changed: {concept}, expected {len(chinese)} rows")
        for lang, pairs in (("en", english), ("zh", chinese)):
            records = []
            for i, (sentence, target) in enumerate(pairs):
                if sentence.count(target) != 1:
                    raise ValueError(f"{lang} {concept} target not unique: {sentence!r}, {target!r}")
                records.append({"id": i, "sentence": sentence, "target": target})
            write_jsonl(out / "corpus" / f"{concept}_{lang}.jsonl", records)


def build_images(out, n_val=8, n_test=20, seed=42):
    rng = random.Random(seed)
    records = []
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    offsets = {
        "left": (-105, 0), "right": (105, 0),
        "above": (0, -92), "below": (0, 92),
    }
    for split, n in (("val", n_val), ("test", n_test)):
        for task in ("counting", "spatial"):
            for i in range(n):
                im = Image.new("RGB", (400, 300), "white")
                draw = ImageDraw.Draw(im)
                if task == "counting":
                    answer = str(rng.randint(1, 5))
                    positions = [(x, y) for x in (60, 140, 220, 300)
                                 for y in (60, 145, 230)]
                    rng.shuffle(positions)
                    for x, y in positions[:int(answer)]:
                        draw.ellipse((x-17, y-17, x+17, y+17), fill="red", outline="black", width=2)
                    for x, y in positions[int(answer):int(answer)+3]:
                        draw.ellipse((x-17, y-17, x+17, y+17), fill="blue", outline="black", width=2)
                    questions = {
                        "en": "How many red circles are in the image? Answer with a single digit.",
                        "zh": "图片中有几个红色的圆形？请只回答一个阿拉伯数字。",
                    }
                else:
                    answer = rng.choice(list(offsets))
                    x, y = offsets[answer]
                    draw.ellipse((176, 126, 224, 174), fill="blue", outline="black", width=2)
                    draw.rectangle((177+x, 127+y, 223+x, 173+y), fill="red",
                                   outline="black", width=2)
                    questions = {
                        "en": "Where is the red square relative to the blue circle? Answer left, right, above, or below.",
                        "zh": "红色方块相对于蓝色圆形在哪个方向？请只回答左、右、上、下。",
                    }
                image_path = image_dir / f"{split}_{task}_{i:04d}.png"
                im.save(image_path)
                records.append({"id": image_path.stem, "split": split, "task": task,
                                "image": str(image_path.relative_to(out)),
                                "answer": answer, "questions": questions})
    write_jsonl(out / "visual_eval.jsonl", records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/gemma3")
    p.add_argument("--n-val", type=int, default=8)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.out)
    build_corpus(out)
    build_images(out, a.n_val, a.n_test, a.seed)
    print(f"Created {out} (20 original EN examples/concept and matched ZH translations)")


if __name__ == "__main__":
    main()
