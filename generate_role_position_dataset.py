#!/usr/bin/env python3
"""
Round 3 dataset generator.

Goal:
Disentangle LOGICAL REASONING ROLE from PHYSICAL PRESENTATION POSITION.

For each 4-hop latent problem:
- keep the graph, facts, question, and answer fixed
- reorder the four facts at presentation time
- replace exactly one logical fact with its visual realization (3T + 1V)
- include a matched all-text baseline for the exact same presentation order

This creates a clean matched effect:
    delta(role, position)
      = accuracy(single-visual) - accuracy(all-text)
under the same problem and the same presentation order.

The script also creates a single-fact perception-control dataset to test whether
a visual relation can be read correctly outside the 4-hop reasoning context.
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path


BALANCED4_ORDERS = [
    [1, 2, 3, 4],
    [2, 1, 4, 3],
    [3, 4, 1, 2],
    [4, 3, 2, 1],
]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def choose_orders(order_set):
    if order_set == "balanced4":
        return BALANCED4_ORDERS
    if order_set == "all24":
        return [list(p) for p in itertools.permutations([1, 2, 3, 4])]
    raise ValueError(order_set)


def copy_images(latents, source_dir, out_dir):
    src_image_dir = source_dir / "images"
    dst_image_dir = out_dir / "images"
    dst_image_dir.mkdir(parents=True, exist_ok=True)

    copied = set()
    for problem in latents:
        for fact in problem["facts"]:
            rel = Path(fact["image"])
            src = source_dir / rel
            dst = out_dir / rel
            if str(rel) in copied:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")
            shutil.copy2(src, dst)
            copied.add(str(rel))


def make_chain_example(problem, order, order_id, visual_step):
    """
    visual_step=None gives the matched all-text baseline.
    Otherwise exactly one logical step is visual.
    """
    facts_by_step = {f["step"]: f for f in problem["facts"]}
    facts = []

    for presentation_position, logical_step in enumerate(order, start=1):
        fact = facts_by_step[logical_step]
        source = "V" if visual_step == logical_step else "T"
        facts.append({
            **fact,
            "logical_step": logical_step,
            "presentation_position": presentation_position,
            "source": source,
        })

    logical_schedule = "".join(
        "V" if visual_step == step else "T"
        for step in [1, 2, 3, 4]
    )
    presentation_schedule = "".join(f["source"] for f in facts)

    if visual_step is None:
        condition = "text_baseline"
        visual_presentation_position = None
        visual_logical_step = None
    else:
        condition = "single_visual"
        visual_logical_step = visual_step
        visual_presentation_position = order.index(visual_step) + 1

    suffix = (
        "baseline"
        if visual_step is None
        else f"Vrole{visual_step}_P{visual_presentation_position}"
    )

    return {
        "task_type": "role_position",
        "condition": condition,
        "example_id": f"{problem['problem_id']}_{order_id}_{suffix}",
        "problem_id": problem["problem_id"],
        "order_id": order_id,
        "presentation_order": order,
        "logical_schedule": logical_schedule,
        "presentation_schedule": presentation_schedule,
        "visual_logical_step": visual_logical_step,
        "visual_presentation_position": visual_presentation_position,
        "vision_count": 0 if visual_step is None else 1,
        "text_count": 4 if visual_step is None else 3,
        "facts": facts,
        "question": problem["question"],
        "answer": problem["answer"],
    }


def make_perception_examples(problem):
    examples = []

    for fact in problem["facts"]:
        for source in ["T", "V"]:
            logical_step = fact["step"]
            examples.append({
                "task_type": "perception",
                "condition": "single_fact",
                "example_id": (
                    f"{problem['problem_id']}_perception_"
                    f"step{logical_step}_{source}"
                ),
                "problem_id": problem["problem_id"],
                "logical_step": logical_step,
                "source": source,
                "facts": [{
                    **fact,
                    "logical_step": logical_step,
                    "presentation_position": 1,
                    "source": source,
                }],
                "question": (
                    f"Where is {fact['subject']} relative to "
                    f"{fact['reference']}?"
                ),
                "answer": {
                    "N": "NORTH",
                    "S": "SOUTH",
                    "E": "EAST",
                    "W": "WEST",
                }[fact["direction"]],
            })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("data16"),
        help="Directory containing latent_problems.jsonl and images/ from Round 2.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data_role_position"),
    )
    parser.add_argument(
        "--order_set",
        choices=["balanced4", "all24"],
        default="balanced4",
        help=(
            "balanced4 is the cheap pilot: every logical role appears once at "
            "each physical position. all24 is the stronger follow-up."
        ),
    )
    parser.add_argument(
        "--limit_problems",
        type=int,
        default=None,
        help="Optional small pilot on the first N latent problems.",
    )
    parser.add_argument(
        "--no_text_baseline",
        action="store_true",
        help="Do not add matched all-text baselines.",
    )
    parser.add_argument(
        "--skip_perception",
        action="store_true",
        help="Do not generate the single-fact perception-control dataset.",
    )
    args = parser.parse_args()

    latent_path = args.source_dir / "latent_problems.jsonl"
    latents = list(load_jsonl(latent_path))
    if args.limit_problems is not None:
        latents = latents[:args.limit_problems]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    copy_images(latents, args.source_dir, args.out_dir)

    # Keep a portable copy of the latent problems next to the new datasets.
    with (args.out_dir / "latent_problems.jsonl").open("w", encoding="utf-8") as f:
        for problem in latents:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    orders = choose_orders(args.order_set)
    role_path = args.out_dir / "role_position_examples.jsonl"

    n_role = 0
    with role_path.open("w", encoding="utf-8") as f:
        for problem in latents:
            for order_idx, order in enumerate(orders):
                order_id = f"o{order_idx:02d}"

                if not args.no_text_baseline:
                    example = make_chain_example(
                        problem=problem,
                        order=order,
                        order_id=order_id,
                        visual_step=None,
                    )
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    n_role += 1

                for visual_step in [1, 2, 3, 4]:
                    example = make_chain_example(
                        problem=problem,
                        order=order,
                        order_id=order_id,
                        visual_step=visual_step,
                    )
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    n_role += 1

    print(f"Loaded {len(latents)} latent problems.")
    print(f"Order set: {args.order_set} ({len(orders)} presentation orders)")
    print(f"Generated {n_role} role-position examples.")
    print("Role-position dataset:", role_path)

    if not args.skip_perception:
        perception_path = args.out_dir / "perception_examples.jsonl"
        n_perception = 0
        with perception_path.open("w", encoding="utf-8") as f:
            for problem in latents:
                for example in make_perception_examples(problem):
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    n_perception += 1

        print(f"Generated {n_perception} perception-control examples.")
        print("Perception dataset:", perception_path)


if __name__ == "__main__":
    main()
