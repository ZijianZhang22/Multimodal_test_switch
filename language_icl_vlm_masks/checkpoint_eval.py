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
