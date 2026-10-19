"""Streaming, strict, restartable CSV evaluation for long VLM sweeps.

A .partial.csv is flushed after every generation and fsynced after each item.
The final CSV appears only after ALL (item, config) cells are complete.
A rerun automatically skips completed cells. This changes no model behavior.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path


def key_of(row):
    return (str(row["id"]), str(row["source"]), str(row["mask"]),
            int(row["layer"]), float(row["alpha"]))


def load_rows(path, headers):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(headers):
            raise ValueError(f"Unexpected CSV header in {path}: {reader.fieldnames}")
        rows = list(reader)
    for i, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Truncated CSV record #{i} in {path}")
        try:
            key_of(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CSV record #{i} in {path}") from exc
    return rows


def evaluate_resumable(samples, configurations, destination, headers,
                       sample_rows, label):
    """Evaluate cells once; support interruption even part-way through a sample.

    sample_rows(sample, missing_configurations) yields one row PER missing
    configuration, in exactly the same order. Model generation is thus
    streamed and safely separated from checkpoint I/O.

    Sample IDs and configurations must be unique. An existing final file
    is verified, never overwritten. Partial data has a distinct suffix.
    """
    destination = Path(destination)
    partial = destination.with_name(destination.stem + ".partial.csv")
    configurations = list(configurations)
    samples = list(samples)
    if not configurations or not samples:
        raise ValueError("No evaluation samples or configurations")
    expected = {}
    for sample in samples:
        for source, mask, layer, alpha in configurations:
            item = {"id": sample["id"], "source": source,
                    "mask": mask, "layer": layer, "alpha": alpha}
            k = key_of(item)
            if k in expected:
                raise ValueError(f"Duplicate expected evaluation cell: {k}")
            expected[k] = sample["id"]
    if destination.exists() and partial.exists():
        raise RuntimeError(f"Both complete and partial CSV exist: {destination}, {partial}")
    active = destination if destination.exists() else partial
    previous = load_rows(active, headers) if active.exists() else []
    completed = {}
    for r in previous:
        k = key_of(r)
        if k not in expected:
            raise RuntimeError(f"Unknown/stale evaluation cell in {active}: {k}")
        if k in completed:
            raise RuntimeError(f"Duplicate evaluation cell in {active}: {k}")
        completed[k] = r
    if destination.exists():
        if len(completed) != len(expected):
            raise RuntimeError(f"Existing final CSV is incomplete: {destination}")
        print(f"{label}: final CSV already complete ({len(completed)} cells).",
              flush=True)
        return previous

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label}: resume {len(completed)}/{len(expected)} cells "
          f"from {partial}", flush=True)
    mode = "a" if partial.exists() else "w"
    with partial.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        if mode == "w":
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
        for index, sample in enumerate(samples, 1):
            missing = [config for config in configurations
                       if key_of({"id": sample["id"], "source": config[0],
                                  "mask": config[1], "layer": config[2],
                                  "alpha": config[3]}) not in completed]
            if missing:
                n_seen = 0
                for config, row in zip(missing, sample_rows(sample, missing)):
                    k = key_of(row)
                    expected_key = key_of({"id": sample["id"],
                                           "source": config[0], "mask": config[1],
                                           "layer": config[2], "alpha": config[3]})
                    if k != expected_key:
                        raise RuntimeError(f"Out-of-order/mismatched row {k} vs {expected_key}")
                    if list(row.keys()) and set(row) != set(headers):
                        raise RuntimeError("Generated row columns do not match CSV headers")
                    writer.writerow(row)
                    f.flush()
                    completed[k] = row
                    n_seen += 1
                if n_seen != len(missing):
                    raise RuntimeError("Evaluation produced fewer rows than configurations")
                os.fsync(f.fileno())
            print(f"{label}: {index}/{len(samples)} samples, "
                  f"{len(completed)}/{len(expected)} cells saved", flush=True)
    if len(completed) != len(expected):
        raise RuntimeError("Evaluation did not complete all cells")
    # Publication is atomic: no incomplete file is ever named validation.csv/test.csv.
    os.replace(partial, destination)
    return load_rows(destination, headers)


def select_validation_parameters(rows, sample_ids, sources, masks, layers, alphas):
    """Select (layer, alpha) from CSV data, normalizing its string fields.

    All cells must cover every validation ID exactly once. This supports
    reload from validation.csv as well as the original in-memory results.
    Tie-breaking: maximize accuracy, then smallest alpha, then lowest layer.
    """
    wanted_ids = {str(i) for i in sample_ids}
    if not wanted_ids or len(wanted_ids) != len(sample_ids):
        raise ValueError("Validation sample IDs are empty or duplicated")
    by_cell = {}
    for row in rows:
        if row["source"] == "none":
            continue
        key = (str(row["source"]), str(row["mask"]),
               int(row["layer"]), float(row["alpha"]))
        if key not in by_cell:
            by_cell[key] = {}
        rid = str(row["id"])
        if rid in by_cell[key]:
            raise RuntimeError(f"Duplicate validation sample {rid} for {key}")
        by_cell[key][rid] = int(row["correct"])
    chosen = {}
    for source in sources:
        chosen[source] = {}
        for mask in masks:
            candidates = []
            for layer in layers:
                for alpha in alphas:
                    key = (source, mask, int(layer), float(alpha))
                    cells = by_cell.get(key, {})
                    if set(cells) != wanted_ids:
                        raise RuntimeError(
                            f"Incomplete validation cells for {key}: "
                            f"found {len(cells)}/{len(wanted_ids)} samples")
                    accuracy = sum(cells.values()) / len(wanted_ids)
                    candidates.append((accuracy, -float(alpha), -int(layer),
                                       int(layer), float(alpha)))
            best = max(candidates)
            chosen[source][mask] = {"layer": best[3], "alpha": best[4]}
    return chosen
