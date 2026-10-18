"""CPU-only tests for A/B experiment; run from any cwd."""
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from core import partition, sample_pairs, token_masks, steer_decoder
from run_ab import compare_vectors, select_samples, summarize, parse_ints
from torch import nn

rows=[]
for task in ("count","relation"):
    for i in range(12):
        rows.append({"id":task+str(i),"task":task,"split":"test" if i>=8 else "tune",
            "answer":"A","prompts":{"en":"Select A.","zh":"选择 A。"},
            "choices_i18n":{"en":["one","two"],"zh":["一","二"]}})
groups=partition(rows)
assert len(groups["compute"])==8
assert len(groups["val"])==8
assert len(groups["test"])==8
assert len(set(x["id"] for v in groups.values() for x in v))==len(rows)
pairs=sample_pairs(groups["compute"],"zh",6,2)
assert len(pairs)==2 and all("问题：" in target for _,target in pairs)
assert parse_ints("8,12-14,12")==[8,12,13,14]
assert len(select_samples(groups["val"],2))==4

a={8:torch.tensor([1.,0.]),12:torch.tensor([0.,2.])}
b={8:torch.tensor([2.,0.]),12:torch.tensor([0.,-2.])}
with tempfile.TemporaryDirectory() as temp:
    report=compare_vectors(a,b,[8,12],Path(temp)/"geometry.csv")
    assert abs(report[0]["cosine_similarity"]-1.)<1e-6
    assert abs(report[1]["cosine_similarity"]+1.)<1e-6
    assert (Path(temp)/"geometry.csv").exists()
out=[
    dict(id="a",lang="zh",source="none",mask="baseline",correct=0),
    dict(id="a",lang="zh",source="base",mask="image",correct=1),
    dict(id="a",lang="zh",source="vlm",mask="image",correct=0),
    dict(id="b",lang="zh",source="none",mask="baseline",correct=1),
    dict(id="b",lang="zh",source="base",mask="image",correct=0),
    dict(id="b",lang="zh",source="vlm",mask="image",correct=1),
]
sc=summarize(out,["base","vlm"],["image"])
assert len(sc)==2 and sc[0]["n"]==2 and sc[0]["fixed"]==1 and sc[0]["broken"]==1
assert sc[0]["delta_pp"]==0.

class Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model=nn.Module()
        self.language_model.model=nn.Module()
        self.language_model.model.layers=nn.ModuleList([nn.Identity()])
fake=Fake()
ids=torch.tensor([[2,9,9,17,18,1]])
masks=token_masks(ids,9,[1,2])
x=torch.ones(1,6,3)
with steer_decoder(fake,0,torch.ones(3),2.,masks["image"]) as edits:
    y=fake.language_model.model.layers[0](x)
assert torch.equal(y[0,1],torch.full((3,),3.))
assert torch.equal(y[0,3],torch.ones(3))
assert edits["tokens"]==2
assert torch.equal(fake.language_model.model.layers[0](x),x)
print("A/B CPU unit tests passed")

# Different token IDs are an audit result, not an incompatibility error.
from run_ab import compare_tokenizers
class TinyTokenizer:
    def __init__(self, offset):
        self.offset=offset
    def __call__(self, text, add_special_tokens=True):
        ids=[ord(ch)+self.offset for ch in text]
        return {"input_ids":([2+self.offset]+ids) if add_special_tokens else ids}
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(n-self.offset) for n in ids)
audit=compare_tokenizers(TinyTokenizer(0),TinyTokenizer(100),[("hello","你好")])
assert audit["checked"]==2 and audit["full_token_ids_equal"]==0
assert audit["source_roundtrip_changed"]==0 and audit["vlm_roundtrip_changed"]==0
print("Tokenizer mismatch audit checks passed")

# Task prefix must lead the ICL prompt, not follow demonstrations.
from run_ab import compose_vlm_prompt
sample={"prompts":{"zh":"问题：图中有几只狗？\n(A) 一\n(B) 二"}}
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parent.parent/"paligemma2_cross_lingual_steering"/"parity"))
from common import build_prompt as _build_prompt
expected=_build_prompt(sample,"zh","native")
assert compose_vlm_prompt(sample,"zh","")==expected
t=compose_vlm_prompt(sample,"zh","English examples:\nQuestion: x\nAnswer: A\n\n")
assert t.startswith("answer zh ")
assert t.endswith(sample["prompts"]["zh"])
assert t.find("English examples:")<t.find("问题：")
print("PaliGemma prompt prefix-order tests passed")
