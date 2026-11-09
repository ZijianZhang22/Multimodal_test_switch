"""PaliGemma2 LoRA: answer-only CE, or answer CE + cross-language hidden alignment.

The EN teacher is the same frozen base VLM with LoRA DISABLED.
LoRA updates only selected decoder layers' q_proj/v_proj. The vision backbone,
base decoder, and multimodal projector are frozen. Same train/val/test as frozen.py.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from protocol import (MODEL_ID, load_model, load_splits, file_hash, decoder_layers,
                      get_prompt, multimodal_inputs, prompt_last_index, evaluate)


def capture_one(model, layer, batch, with_grad=False):
    """Capture one decoder-layer state from the last PROMPT token.

    When suffix is present, only prompt tokens are used; suffix contributes to CE.
    """
    pos = prompt_last_index(batch)
    captured = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[:, pos, :]
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        if with_grad:
            result = model(**batch, use_cache=False)
        else:
            with torch.no_grad():
                result = model(**batch, use_cache=False)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError("Missing hidden state hook")
    return result, captured["h"]


def alignment_loss(student, teacher):
    """Cosine matching, teacher detached, normalized to avoid hidden scale drift."""
    return (1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)).mean()


def target_module_names(layers):
    return [f"language_model.model.layers.{layer}.self_attn.{proj}"
            for layer in layers for proj in ("q_proj", "v_proj")]


def verify_lora_frozen(model, layers):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or not all("lora_" in n for n in names):
        raise RuntimeError("Expected LoRA-only trainable parameters")
    if not all(any(f"layers.{layer}." in n for n in names) for layer in layers):
        raise RuntimeError("Missing LoRA on requested layers")
    if any("vision_tower" in n for n in names):
        raise RuntimeError("Vision tower should be frozen")
    return names


def run(args):
    if args.align_weight < 0 or args.ce_weight <= 0:
        raise ValueError("CE weight >0 and align weight >=0 required")
    if args.language not in ("zh", "sw"):
        raise ValueError("Language must be zh or sw")
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("Bad training settings")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    splits = load_splits(args.data)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "train_sha256": file_hash(Path(args.data) / "train.jsonl")}
    fingerprint = output / "config.json"
    if fingerprint.exists() and json.loads(fingerprint.read_text()) != config:
        raise ValueError("Existing output is from a different configuration; choose another --out")
    fingerprint.write_text(json.dumps(config, indent=2))
    processor, model = load_model(args.model_id)
    from peft import LoraConfig, TaskType, get_peft_model
    modules = decoder_layers(model)
    layers = sorted(set(int(x) for x in args.layers.split(",")))
    if min(layers) < 0 or max(layers) >= len(modules):
        raise ValueError(f"Layer out of range; PaliGemma2 has {len(modules)}")
    if args.align_layer not in layers:
        raise ValueError("align-layer must be one of --layers")
    # Full module names; avoid adapting the whole model to study middle layers.
    names = target_module_names(layers)
    actual = {n for n, _ in model.named_modules()}
    if not set(names) <= actual:
        raise RuntimeError(f"PaliGemma decoder module names changed: {set(names) - actual}")
    conf = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0,
                      bias="none", target_modules=names,
                      task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, conf)
    verify_lora_frozen(model, layers)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # Frozen English teacher states cached; teacher is unaffected by future LoRA updates.
    teacher_dir = output / "teacher_cache"; teacher_dir.mkdir(exist_ok=True)
    # Match frozen Rosetta: identical deterministic 25-pair image IDs.
    # The non-paired rows are NOT used for LoRA answer supervision.
    from frozen import split_anchor_unpaired
    anchor_indices, _, _ = split_anchor_unpaired(splits["train"], args.pair_budget, args.seed)
    train = [splits["train"][int(i)] for i in anchor_indices]
    (output / "train_anchor_ids.json").write_text(json.dumps([r["id"] for r in train], indent=2))
    print(f"Training LoRA on same {len(train)} paired image-question IDs as frozen mapping", flush=True)
    rng = random.Random(args.seed)
    best_score = (-1.0, -1.0)
    best_state = None
    for epoch in range(args.epochs):
        order = list(range(len(train))); rng.shuffle(order)
        losses = []
        for step, i in enumerate(order):
            row = train[i]
            model.train()
            src = multimodal_inputs(processor, row["image"],
                                    get_prompt(row, args.language), suffix=str(row["answer"]))
            if "labels" not in src or "token_type_ids" not in src or not src["labels"].ne(-100).any():
                raise ValueError("PaliGemmaProcessor must provide suffix labels/token_type_ids")
            reference = None
            if args.align_weight > 0:
                tfile = teacher_dir / f"{i:05d}.npy"
                if tfile.exists():
                    reference = torch.from_numpy(np.load(tfile)).to(device="cuda", dtype=torch.float32)
                else:
                    tgt = multimodal_inputs(processor, row["image"], get_prompt(row, "en"))
                    model.eval()
                    # PEFT disable_adapter() is essential: teacher is frozen original VLM.
                    with model.disable_adapter():
                        _, reference = capture_one(model, args.align_layer, tgt)
                    reference = reference.detach().float().cpu().numpy()
                    np.save(tfile, reference)
                    reference = torch.from_numpy(reference).to(device="cuda", dtype=torch.float32)
            model.train()
            output_obj, student = capture_one(model, args.align_layer, src, with_grad=True)
            ce = output_obj.loss
            if ce is None or not torch.isfinite(ce):
                raise RuntimeError("Invalid suffix CE loss")
            align = (alignment_loss(student, reference) if reference is not None
                     else torch.zeros((), device=ce.device))
            loss = args.ce_weight * ce + args.align_weight * align
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite optimization loss")
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append([float(ce.detach()), float(align.detach()), float(loss.detach())])
            if (step + 1) % 25 == 0:
                print(f"epoch={epoch+1} row={step+1}/{len(order)} "
                      f"CE={np.mean(losses[-25:],axis=0)[0]:.4f} "
                      f"align={np.mean(losses[-25:],axis=0)[1]:.4f}", flush=True)
        model.eval()
        val = evaluate(model, processor, splits["val"], args.language,
                       f"lora_epoch{epoch+1}", output / f"val_epoch{epoch+1}.csv",
                       max_new_tokens=args.max_new_tokens)
        print("LoRA VAL", epoch+1, val, flush=True)
        current = (val["accuracy"], val["valid_rate"])
        if current > best_score:
            best_score = current
            model.save_pretrained(output / "best_adapter")
            best_state = {n: p.detach().cpu().clone()
                          for n, p in model.named_parameters() if p.requires_grad}
            (output / "best_val.json").write_text(json.dumps({"epoch": epoch+1,
                "accuracy": val["accuracy"], "valid_rate": val["valid_rate"]}, indent=2))
    if best_state is None:
        raise RuntimeError("No trained adapter")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(p.device))
    model.eval()
    for lang in ("en", args.language):
        # For reference: both original English and English after LoRA, as the
        # latter may degrade. The frozen baseline is evaluated separately below.
        summary = evaluate(model, processor, splits["test"], lang,
                           "lora", output / f"test_lora_{lang}.csv",
                           max_new_tokens=args.max_new_tokens)
        print("LoRA TEST", lang, summary, flush=True)
    # Frozen baseline without adapter; needed to measure actual improvement.
    with model.disable_adapter():
        for lang in ("en", args.language):
            summary = evaluate(model, processor, splits["test"], lang,
                               "base", output / f"test_baseline_{lang}.csv",
                               max_new_tokens=args.max_new_tokens)
            print("BASE TEST", lang, summary, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/data")
    p.add_argument("--out", default="outputs/lora_zh_alignment")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--language", choices=("zh", "sw"), default="zh")
    p.add_argument("--layers", default="8,13,18")
    p.add_argument("--align-layer", type=int, default=13)
    p.add_argument("--pair-budget", type=int, default=25)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--align-weight", type=float, default=0.3,
                   help="0 trains standard answer-only LoRA control")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=8)
    run(p.parse_args())


if __name__ == "__main__":
    main()