#!/usr/bin/env python3
"""
Generate variable-hop depth-scaling datasets.

For each hop count H, create matched variants of the same latent spatial chain:
  - all-Text baseline
  - exactly one visual fact at every logical role k=1..H

This gives O(H) carrier interventions per presentation order instead of 2^H.

Presentation-order modes:
  identity         : logical order only; cheapest depth-scaling experiment
  reverse_pair     : identity + reversed physical order
  balanced_cyclic  : H cyclic shifts, so every logical role appears once at
                     every physical position (strongest but O(H^2) examples)

The same generator works for 4/6/8/12-hop experiments.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DIRECTIONS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}
DIR_WORD = {
    "N": "north of",
    "S": "south of",
    "E": "east of",
    "W": "west of",
}
ANSWER_LABELS = [
    "NORTH", "SOUTH", "EAST", "WEST",
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST", "SAME",
]


def vector_to_answer(x, y):
    if x == 0 and y == 0:
        return "SAME"
    vertical = "NORTH" if y < 0 else "SOUTH" if y > 0 else ""
    horizontal = "EAST" if x > 0 else "WEST" if x < 0 else ""
    return vertical + horizontal if vertical and horizontal else vertical or horizontal


def load_font(size):
    candidates = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_relation_image(reference, subject, direction, out_path, size=384):
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    center = (size // 2, size // 2)
    offset = size // 4
    dx, dy = DIRECTIONS[direction]

    ref_xy = center
    subj_xy = (center[0] + dx * offset, center[1] + dy * offset)
    radius = 34

    draw.line([ref_xy, subj_xy], fill="black", width=5)
    for xy in [ref_xy, subj_xy]:
        draw.ellipse(
            [xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius],
            fill="white",
            outline="black",
            width=4,
        )

    font = load_font(30)

    def centered_text(xy, txt):
        box = draw.textbbox((0, 0), txt, font=font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        draw.text(
            (xy[0] - tw / 2, xy[1] - th / 2 - 2),
            txt,
            fill="black",
            font=font,
        )

    centered_text(ref_xy, reference)
    centered_text(subj_xy, subject)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def make_latent_problem(rng, pid, hop_count):
    entities = [f"N{i}" for i in range(hop_count + 1)]
    directions = [rng.choice(list(DIRECTIONS)) for _ in range(hop_count)]

    facts = []
    x = 0
    y = 0
    for i, direction in enumerate(directions, start=1):
        reference = entities[i - 1]
        subject = entities[i]
        dx, dy = DIRECTIONS[direction]
        x += dx
        y += dy

        facts.append({
            "step": i,
            "logical_step": i,
            "reference": reference,
            "subject": subject,
            "direction": direction,
            "text": f"{subject} is {DIR_WORD[direction]} {reference}.",
            "state_x": x,
            "state_y": y,
        })

    return {
        "problem_id": f"h{hop_count}_p{pid:05d}",
        "hop_count": hop_count,
        "entities": entities,
        "directions": directions,
        "facts": facts,
        "question": f"Where is {entities[-1]} relative to {entities[0]}?",
        "answer": vector_to_answer(x, y),
    }


def generate_balanced_latents(n, seed, hop_count):
    rng = random.Random(seed + 1009 * hop_count)
    quota = {label: math.ceil(n / len(ANSWER_LABELS)) for label in ANSWER_LABELS}
    counts = Counter()
    latents = []

    attempts = 0
    while len(latents) < n:
        attempts += 1
        if attempts > n * 5000:
            raise RuntimeError(
                f"Could not balance answer classes for H={hop_count}; "
                "try increasing n or using another seed."
            )

        problem = make_latent_problem(rng, len(latents), hop_count)
        answer = problem["answer"]
        if counts[answer] >= quota[answer]:
            continue

        counts[answer] += 1
        problem["problem_id"] = f"h{hop_count}_p{len(latents):05d}"
        latents.append(problem)

    return latents, counts


def choose_orders(hop_count, mode):
    identity = list(range(1, hop_count + 1))
    if mode == "identity":
        return [identity]
    if mode == "reverse_pair":
        reverse = list(reversed(identity))
        return [identity, reverse]
    if mode == "balanced_cyclic":
        return [
            identity[shift:] + identity[:shift]
            for shift in range(hop_count)
        ]
    raise ValueError(mode)


def make_example(problem, order, order_id, visual_role):
    hop_count = int(problem["hop_count"])
    facts_by_step = {int(f["step"]): f for f in problem["facts"]}

    facts = []
    for physical_position, logical_step in enumerate(order, start=1):
        fact = facts_by_step[logical_step]
        source = "V" if visual_role == logical_step else "T"
        facts.append({
            **fact,
            "logical_step": logical_step,
            "presentation_position": physical_position,
            "source": source,
        })

    if visual_role is None:
        condition = "text_baseline"
        visual_position = None
        normalized_depth = None
        suffix = "baseline"
    else:
        condition = "single_visual"
        visual_position = order.index(visual_role) + 1
        normalized_depth = (
            0.0 if hop_count == 1
            else (visual_role - 1) / (hop_count - 1)
        )
        suffix = f"Vrole{visual_role}_P{visual_position}"

    logical_schedule = "".join(
        "V" if visual_role == k else "T"
        for k in range(1, hop_count + 1)
    )

    return {
        "task_type": "depth_scaling",
        "condition": condition,
        "example_id": f"{problem['problem_id']}_{order_id}_{suffix}",
        "problem_id": problem["problem_id"],
        "hop_count": hop_count,
        "order_id": order_id,
        "presentation_order": order,
        "logical_schedule": logical_schedule,
        "presentation_schedule": "".join(f["source"] for f in facts),
        "visual_logical_step": visual_role,
        "visual_presentation_position": visual_position,
        "normalized_logical_depth": normalized_depth,
        "vision_count": 0 if visual_role is None else 1,
        "text_count": hop_count if visual_role is None else hop_count - 1,
        "facts": facts,
        "question": problem["question"],
        "answer": problem["answer"],
    }


def parse_hops(text):
    hops = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not hops or min(hops) < 2:
        raise ValueError("Use hop counts >= 2, e.g. --hops 4,6,8,12")
    return hops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data_depth_scaling"))
    parser.add_argument("--hops", default="4,6,8,12")
    parser.add_argument("--n_problems_per_hop", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--order_mode",
        choices=["identity", "reverse_pair", "balanced_cyclic"],
        default="identity",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Do not render images; useful only for dataset-structure debugging.",
    )
    args = parser.parse_args()

    hops = parse_hops(args.hops)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    latent_path = args.out_dir / "latent_problems.jsonl"
    example_path = args.out_dir / "depth_scaling_examples.jsonl"

    total_latents = 0
    total_examples = 0

    with latent_path.open("w", encoding="utf-8") as latent_file, \
         example_path.open("w", encoding="utf-8") as example_file:

        for hop_count in hops:
            latents, answer_counts = generate_balanced_latents(
                args.n_problems_per_hop,
                args.seed,
                hop_count,
            )
            orders = choose_orders(hop_count, args.order_mode)

            print(
                f"H={hop_count}: {len(latents)} problems; "
                f"{len(orders)} presentation order(s); answers={dict(answer_counts)}"
            )

            for problem in latents:
                for fact in problem["facts"]:
                    image_rel = (
                        Path("images")
                        / f"{problem['problem_id']}_step{fact['step']}.png"
                    )
                    fact["image"] = str(image_rel)

                    if not args.no_images:
                        draw_relation_image(
                            reference=fact["reference"],
                            subject=fact["subject"],
                            direction=fact["direction"],
                            out_path=args.out_dir / image_rel,
                        )

                latent_file.write(json.dumps(problem, ensure_ascii=False) + "\n")
                total_latents += 1

                for order_idx, order in enumerate(orders):
                    order_id = f"o{order_idx:02d}"

                    baseline = make_example(
                        problem=problem,
                        order=order,
                        order_id=order_id,
                        visual_role=None,
                    )
                    example_file.write(
                        json.dumps(baseline, ensure_ascii=False) + "\n"
                    )
                    total_examples += 1

                    for visual_role in range(1, hop_count + 1):
                        example = make_example(
                            problem=problem,
                            order=order,
                            order_id=order_id,
                            visual_role=visual_role,
                        )
                        example_file.write(
                            json.dumps(example, ensure_ascii=False) + "\n"
                        )
                        total_examples += 1

    print(f"Generated {total_latents} latent problems.")
    print(f"Generated {total_examples} examples.")
    print("Latents:", latent_path)
    print("Examples:", example_path)
    print("Images:", image_dir)


if __name__ == "__main__":
    main()
