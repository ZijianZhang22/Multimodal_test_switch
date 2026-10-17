"""CPU-only sanity checks for the Cross-Lingual Steering experiment."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from make_pilot_data import corpora, visual_set
from model_utils import canonical_answer, read_jsonl, token_mask, steer_layer
from analyze_results import cosine, paired_interval


def test_generated_paired_data():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        corpora(root, 40, 42)
        visual_set(root, 2, 3, 42)
        for concept in ("counting", "spatial"):
            en = read_jsonl(root / "corpus" / f"{concept}_en.jsonl")
            zh = read_jsonl(root / "corpus" / f"{concept}_zh.jsonl")
            assert [x["id"] for x in en] == [x["id"] for x in zh]
            assert all(row["sentence"].count(row["target"]) == 1 for row in en + zh)
        samples = read_jsonl(root / "visual_eval.jsonl")
        assert len(samples) == 10
        assert all((root / x["image"]).exists() for x in samples)


def test_answer_parsing():
    assert canonical_answer("三", "counting") == "3"
    assert canonical_answer("five", "counting") == "5"
    assert canonical_answer("在左边", "spatial") == "left"
    assert canonical_answer("below", "spatial") == "below"


def test_image_masks_and_hook_cleanup():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Identity()])
    model = Toy()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[0, 99]))
    mask = token_mask(torch.tensor([[0, 10, 99, 5, 5, 20]]), processor, "image", 5)
    assert mask.tolist() == [False, False, False, True, True, False]
    x = torch.zeros(1, 6, 2)
    with steer_layer(model, 0, torch.tensor([3.0, 4.0]), 5.0, mask):
        y = model.model.language_model.layers[0](x)
        assert torch.allclose(y[0, 3, :], torch.tensor([3.0, 4.0]))
        assert torch.allclose(y[0, 0, :], torch.zeros(2))
        # Prefill-only: autoregressive generation has a shorter sequence.
        assert torch.equal(model.model.language_model.layers[0](torch.zeros(1, 1, 2)),
                           torch.zeros(1, 1, 2))
    assert torch.equal(model.model.language_model.layers[0](x), x)


def test_geometry_and_paired_bootstrap():
    assert abs(cosine(torch.tensor([1., 0.]), torch.tensor([0., 1.]))) < 1e-8
    mean, lo, hi = paired_interval([1, 0, -1, 0], seed=4, n_boot=500)
    assert mean == 0.0
    assert lo <= mean <= hi
