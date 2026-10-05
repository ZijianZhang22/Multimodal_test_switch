#!/usr/bin/env python3
"""
Generate a controlled 4-hop multimodal reasoning dataset.

Round 2 uses the full 2^4 factorial design:
every reasoning step can carry its fact as Text (T) or Vision (V), giving
16 schedules from TTTT through VVVV.

The latent problem, fact order, answer, and difficulty stay fixed. Only the
modality carrier of each fact changes.
"""

import argparse
import itertools
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
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST", "SAME"
]

CORE6 = ["TTVV", "VVTT", "TVVT", "VTTV", "TVTV", "VTVT"]
CORE8 = CORE6 + ["TTTT", "VVVV"]
ALL16 = ["".join(bits) for bits in itertools.product("TV", repeat=4)]


def schedule_metadata(schedule):
    switch_count = sum(
        schedule[i] != schedule[i + 1]
        for i in range(len(schedule) - 1)
    )
    tv_transitions = sum(
        schedule[i] == "T" and schedule[i + 1] == "V"
        for i in range(len(schedule) - 1)
    )
    vt_transitions = sum(
        schedule[i] == "V" and schedule[i + 1] == "T"
        for i in range(len(schedule) - 1)
    )
    vision_count = schedule.count("V")

    return {
        "switch_count": switch_count,
        "tv_transitions": tv_transitions,
        "vt_transitions": vt_transitions,
        "vision_count": vision_count,
        "text_count": len(schedule) - vision_count,
        "vision_fraction": vision_count / len(schedule),
        "starts_with": schedule[0],
        "ends_with": schedule[-1],
        "V1": int(schedule[0] == "V"),
        "V2": int(schedule[1] == "V"),
        "V3": int(schedule[2] == "V"),
        "V4": int(schedule[3] == "V"),
    }


def vector_to_answer(x, y):
    if x == 0 and y == 0:
        return "SAME"

    vertical = ""
    horizontal = ""

    if y < 0:
        vertical = "NORTH"
    elif y > 0:
        vertical = "SOUTH"

    if x > 0:
        horizontal = "EAST"
    elif x < 0:
        horizontal = "WEST"

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
    """
    Draw the spatial relation without writing the relation as a sentence.
    The node labels remain visible so the model can bind the two entities.
    """
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    center = (size // 2, size // 2)
    offset = size // 4
    dx, dy = DIRECTIONS[direction]

    ref_xy = center
    subj_xy = (center[0] + dx * offset, center[1] + dy * offset)

    radius = 34
    draw.line([ref_xy, subj_xy], fill="black", width=5)

    draw.ellipse(
        [ref_xy[0] - radius, ref_xy[1] - radius,
         ref_xy[0] + radius, ref_xy[1] + radius],
        fill="white", outline="black", width=4,
    )
    draw.ellipse(
        [subj_xy[0] - radius, subj_xy[1] - radius,
         subj_xy[0] + radius, subj_xy[1] + radius],
        fill="white", outline="black", width=4,
    )

    font = load_font(34)

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


def make_latent_problem(rng, pid):
    entities = [f"N{i}" for i in range(5)]
    directions = [rng.choice(list(DIRECTIONS)) for _ in range(4)]

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
            "reference": reference,
            "subject": subject,
            "direction": direction,
            "text": f"{subject} is {DIR_WORD[direction]} {reference}.",
        })

    return {
        "problem_id": f"p{pid:05d}",
        "entities": entities,
        "directions": directions,
        "facts": facts,
        "question": f"Where is {entities[-1]} relative to {entities[0]}?",
        "answer": vector_to_answer(x, y),
    }


def generate_balanced_latents(n, seed):
    rng = random.Random(seed)
    quota = {label: math.ceil(n / len(ANSWER_LABELS)) for label in ANSWER_LABELS}
    counts = Counter()
    latents = []

    attempts = 0
    while len(latents) < n:
        attempts += 1
        if attempts > n * 1000:
            raise RuntimeError("Could not balance answer classes; try another seed.")

        problem = make_latent_problem(rng, len(latents))
        answer = problem["answer"]

        if counts[answer] >= quota[answer]:
            continue

        counts[answer] += 1
        problem["problem_id"] = f"p{len(latents):05d}"
        latents.append(problem)

    return latents, counts


def choose_schedules(schedule_set):
    if schedule_set == "all16":
        return ALL16
    if schedule_set == "core8":
        return CORE8
    if schedule_set == "core6":
        return CORE6
    raise ValueError(schedule_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--n_problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schedule_set",
        choices=["all16", "core8", "core6"],
        default="all16",
        help="Round 2 default is all16. core6/core8 reproduce the earlier pilot.",
    )
    parser.add_argument(
        "--include_unimodal",
        action="store_true",
        help=(
            "Deprecated compatibility flag. If used with --schedule_set core6, "
            "it upgrades the set to core8. all16 already contains TTTT/VVVV."
        ),
    )
    args = parser.parse_args()

    schedule_set = args.schedule_set
    if args.include_unimodal and schedule_set == "core6":
        schedule_set = "core8"

    schedules = choose_schedules(schedule_set)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    latents, answer_counts = generate_balanced_latents(
        args.n_problems,
        args.seed,
    )

    latent_path = args.out_dir / "latent_problems.jsonl"
    example_path = args.out_dir / "examples.jsonl"

    with latent_path.open("w", encoding="utf-8") as latent_file, \
         example_path.open("w", encoding="utf-8") as example_file:

        for problem in latents:
            # Render each fact once and reuse it across all 16 schedules.
            for fact in problem["facts"]:
                image_rel = (
                    Path("images")
                    / f"{problem['problem_id']}_step{fact['step']}.png"
                )
                image_abs = args.out_dir / image_rel

                draw_relation_image(
                    reference=fact["reference"],
                    subject=fact["subject"],
                    direction=fact["direction"],
                    out_path=image_abs,
                )
                fact["image"] = str(image_rel)

            latent_file.write(
                json.dumps(problem, ensure_ascii=False) + "\n"
            )

            for schedule in schedules:
                meta = schedule_metadata(schedule)

                facts = []
                for i, fact in enumerate(problem["facts"]):
                    facts.append({
                        **fact,
                        "source": schedule[i],
                    })

                example = {
                    "example_id": f"{problem['problem_id']}_{schedule}",
                    "problem_id": problem["problem_id"],
                    "schedule": schedule,
                    **meta,
                    "modality_budget": {
                        "text_facts": meta["text_count"],
                        "vision_facts": meta["vision_count"],
                    },
                    "facts": facts,
                    "question": problem["question"],
                    "answer": problem["answer"],
                }

                example_file.write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )

    print(f"Schedule set: {schedule_set}")
    print(f"Schedules ({len(schedules)}): {', '.join(schedules)}")
    print(f"Generated {len(latents)} latent problems.")
    print(f"Generated {len(latents) * len(schedules)} examples.")
    print("Answer counts:", dict(answer_counts))
    print("Latents:", latent_path)
    print("Examples:", example_path)
    print("Images:", image_dir)


if __name__ == "__main__":
    main()
