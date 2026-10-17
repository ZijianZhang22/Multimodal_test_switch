"""CPU-only tests of bilingual data, token pooling and scoring helpers."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from make_data import ZH, build_images
from extract import locate_tokens, mean_shift_from_hidden
from evaluate import canonical
from analysis import cos


class TokenBatch(dict):
    def to(self, device):
        return self


class CharTokenizer:
    bos_token_id = 99

    def __call__(self, sentence, **kwargs):
        # Fake fast tokenizer whose offsets align to characters.
        offsets = [(0, 0)] + [(i, i+1) for i in range(len(sentence))]
        ids = [99] + list(range(10, 10+len(sentence)))
        return TokenBatch(
            input_ids=torch.tensor([ids]),
            offset_mapping=torch.tensor([offsets]),
        )


class TinyModel:
    device = torch.device("cpu")

    def language_model(self, input_ids=None, output_hidden_states=None, use_cache=None):
        # BOS, first control, target, second control.
        h = torch.tensor([[[0.0], [2.0], [10.0], [4.0]]])
        return SimpleNamespace(hidden_states=(h*0, h))


def test_chinese_translations_unique():
    assert all(len(v)==20 for v in ZH.values())
    for pairs in ZH.values():
        for sentence, target in pairs:
            assert sentence.count(target) == 1, (sentence, target)


def test_mean_shift_non_bos_token_pools():
    tokenizer=CharTokenizer()
    batch, t, c=locate_tokens(tokenizer, "a中b", "中")
    assert t==[2] and c==[1,3]
    rows=[{"sentence":"a中b","target":"中"}]
    result=mean_shift_from_hidden(rows,tokenizer,TinyModel(),[0])
    assert torch.allclose(result[0],torch.tensor([7.0]))


def test_pilot_images():
    with TemporaryDirectory() as td:
        base=Path(td)
        build_images(base,2,3,42)
        images=list((base/"images").glob("*.png"))
        assert len(images)==10
        assert (base/"visual_eval.jsonl").exists()


def test_answer_normalization():
    assert canonical("Three","counting")=="3"
    assert canonical("三","counting")=="3"
    assert canonical("左边","spatial")=="left"
    assert canonical("below","spatial")=="below"


def test_cos_geometry():
    assert abs(cos(torch.tensor([1.,0.]),torch.tensor([0.,1.])))<1e-7
