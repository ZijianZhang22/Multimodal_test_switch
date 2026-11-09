"""Frozen PaliGemma2 activation alignment: supervised vs. Rosetta-like OT.

Usage: python frozen.py --data outputs/data --out outputs/frozen --language zh
Requires one GPU, but does not train or change any VLM weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from protocol import (MODEL_ID, load_model, load_splits, file_hash, extract_hidden,
                      evaluate)
from rosetta_math import fit_mapping, save_mapping, load_mapping, correction


def collect(model, processor, rows, layers, language, cache):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    result = {l: {"en": [], language: []} for l in layers}
    for index, row in enumerate(rows):
        path = cache / f"{index:05d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["id"].item()) != row["id"] or set(z.files) != {"id"} | {
                        f"{lang}_{layer}" for lang in ("en", language) for layer in layers}:
                    raise ValueError(f"Mismatched activation cache: {path}")
                states = {f"{lang}_{layer}": z[f"{lang}_{layer}"]
                          for lang in ("en", language) for layer in layers}
        else:
            states = {}
            for lang in ("en", language):
                hidden = extract_hidden(model, processor, row, lang, layers)
                states.update({f"{lang}_{layer}": hidden[layer] for layer in layers})
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, id=np.array(row["id"]), **states)
            tmp.replace(path)
        for layer in layers:
            for lang in ("en", language):
                result[layer][lang].append(states[f"{lang}_{layer}"])
        if (index + 1) % 20 == 0:
            print(f"Extracted {index+1}/{len(rows)} image-conditioned activation pairs", flush=True)
    return {l: {lang: np.stack(v) for lang, v in group.items()} for l, group in result.items()}


def split_anchor_unpaired(rows, budget, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(len(rows))
    rng.shuffle(ids)
    if not 2 <= budget <= len(rows) - 4:
        raise ValueError(f"Need 2 <= pair-budget <= n_train-4; got {budget}/{len(rows)}")
    a, remaining = ids[:budget], ids[budget:]
    x = remaining[::2]
    y = remaining[1::2]
    assert not (set(a) & (set(x) | set(y)) or set(x) & set(y))
    return a, x, y


def fit_all(features, rows, layers, budget, dim, pool_weight, iterations, seed,
            output_dir, model_id, data_sha, language):
    anchor, ix, iy = split_anchor_unpaired(rows, budget, seed)
    configs = []
    for layer in layers:
        src, tgt = features[layer][language], features[layer]["en"]
        for method in ("supervised", "rosetta"):
            mapping = fit_mapping(src[anchor], tgt[anchor], src[ix], tgt[iy],
                                  dim=dim, method=method, iterations=iterations,
                                  pool_weight=pool_weight, seed=seed)
            name = f"layer{layer}_{method}"
            dest = Path(output_dir) / f"{name}.npz"
            save_mapping(dest, mapping, dict(model_id=model_id, data_sha=data_sha,
                                             language=language, layer=layer,
                                             seed=seed, pool_weight=pool_weight, iterations=iterations,
                                             anchor_ids=[rows[i]["id"] for i in anchor]))
            print("Saved mapping", dest, flush=True)
            configs.append((layer, method, dest))
    return configs


def run(args):
    if args.language not in ("zh", "sw"):
        raise ValueError("Choose zh or sw")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.data)
    data_sha = file_hash(Path(args.data) / "train.jsonl")
    layers = [int(k) for k in args.layers.split(",")]
    if len(set(layers)) != len(layers):
        raise ValueError("Duplicate layers")
    alphas = sorted({float(s) for s in args.strengths.split(",")})
    if any(not 0 <= s <= 2 for s in alphas):
        raise ValueError("Strength outside [0, 2]")
    (out / "experiment_config.json").write_text(json.dumps(vars(args), indent=2))
    processor, model = load_model(args.model_id)
    if max(layers) >= len(__import__('protocol').decoder_layers(model)):
        raise ValueError("Out-of-range decoder layer")
    cache_meta = {"model_id": args.model_id, "data_sha": data_sha,
                  "layers": layers, "language": args.language}
    cache_manifest = out / "activation_cache" / "cache_manifest.json"
    cache_manifest.parent.mkdir(exist_ok=True)
    if cache_manifest.is_file():
        if json.loads(cache_manifest.read_text()) != cache_meta:
            raise ValueError("Activation cache was created with another model/data/language/layer grid; use a NEW --out")
    else:
        cache_manifest.write_text(json.dumps(cache_meta, indent=2))
    if args.stage in ("fit", "all"):
        features = collect(model, processor, splits["train"], layers, args.language,
                           out / "activation_cache")
        fit_all(features, splits["train"], layers, args.pair_budget,
                args.dim, args.pool_weight, args.iterations, args.seed,
                out / "mappings", args.model_id, data_sha, args.language)
    if args.stage == "fit":
        return
    configs = [(l, m, out / "mappings" / f"layer{l}_{m}.npz")
               for l in layers for m in ("supervised", "rosetta")]
    for l, m, path in configs:
        if not path.exists():
            raise FileNotFoundError(f"Missing mapping: {path}")
    # Include original English as reference and the unmodified target-language model.
    for lang in ("en", args.language):
        summary = evaluate(model, processor, splits["val"], lang, "baseline",
                           out / f"val_baseline_{lang}.csv", max_new_tokens=args.max_new_tokens)
        print(f"VALIDATION baseline {lang}: {summary}", flush=True)
    scored = []
    for layer, method, path in configs:
        mapping, _ = load_mapping(path, model_id=args.model_id, data_sha=data_sha,
                                  language=args.language, layer=layer)
        for strength in alphas:
            if strength == 0:
                continue  # baseline already evaluated
            def intervene(h, mp=mapping, alpha=strength):
                return correction(h, mp, alpha, args.max_relative_norm)
            name = f"layer{layer}_{method}_alpha{strength:g}"
            summary = evaluate(model, processor, splits["val"], args.language, name,
                               out / f"val_{name}.csv", layer=layer,
                               intervention=intervene, max_new_tokens=args.max_new_tokens)
            scored.append({"name": name, "layer": layer, "method": method,
                           "strength": strength, "accuracy": summary["accuracy"],
                           "valid_rate": summary["valid_rate"]})
            print("VALIDATION", name, summary, flush=True)
    if not scored:
        raise ValueError("Require at least one positive strength")
    # Selection is done ONLY on local validation. Tie-break with valid rate and lower alpha.
    best = sorted(scored, key=lambda r: (-r["accuracy"], -r["valid_rate"],
                                         r["strength"], r["layer"], r["method"]))[0]
    (out / "val_grid.json").write_text(json.dumps(scored, indent=2))
    (out / "selection.json").write_text(json.dumps(best, indent=2))
    # Also select EACH method independently. Do not drop the weaker arm from
    # held-out testing: otherwise no claim about adding unpaired OT is possible.
    chosen = {}
    for method in ("supervised", "rosetta"):
        candidates = [r for r in scored if r["method"] == method]
        chosen[method] = sorted(candidates, key=lambda r: (-r["accuracy"],
                            -r["valid_rate"], r["strength"], r["layer"]))[0]
    (out / "selection_by_method.json").write_text(json.dumps(chosen, indent=2))
    print("VAL-SELECTED by method", chosen, flush=True)
    if args.stage == "eval":
        return
    for lang in ("en", args.language):
        summary = evaluate(model, processor, splits["test"], lang, "baseline",
                           out / f"test_baseline_{lang}.csv", max_new_tokens=args.max_new_tokens)
        print(f"TEST baseline {lang}", summary, flush=True)
    for method, best_method in chosen.items():
        selected, _ = load_mapping(out / "mappings" / f"layer{best_method['layer']}_{method}.npz",
                                   model_id=args.model_id, data_sha=data_sha,
                                   language=args.language, layer=best_method["layer"])
        def intervene(h, mapping=selected, alpha=best_method["strength"]):
            return correction(h, mapping, alpha, args.max_relative_norm)
        test = evaluate(model, processor, splits["test"], args.language, best_method["name"],
                        out / f"test_{best_method['name']}.csv", layer=best_method["layer"],
                        intervention=intervene, max_new_tokens=args.max_new_tokens)
        print("TEST selected", method, best_method, test, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/data")
    p.add_argument("--out", default="outputs/frozen_zh")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--language", default="zh", choices=("zh", "sw"))
    p.add_argument("--layers", default="8,13,18")
    p.add_argument("--strengths", default="0,0.15,0.3,0.6")
    p.add_argument("--pair-budget", type=int, default=25)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--pool-weight", type=float, default=0.3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-relative-norm", type=float, default=0.15)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--stage", choices=("fit", "eval", "all"), default="all")
    run(p.parse_args())


if __name__ == "__main__":
    main()