import torch
from torch import nn
from core import partition, sample_pairs, token_masks, steer_decoder, paired_metrics
rows=[]
for task in ("count","depth"):
    for i in range(8):
        rows.append(dict(id=task+str(i),task=task,split="test" if i>5 else "tune",answer="A",
                         prompts={"en":"A or B?","zh":"选 A 还是 B？"},
                         choices_i18n={"en":["x","y"],"zh":["甲","乙"]}))
sp=partition(rows)
assert len(sp["compute"])==6 and len(sp["val"])==6 and len(sp["test"])==4
assert len(sample_pairs(sp["compute"],"zh",k=6,max_samples=2))==2
ids=torch.tensor([[2,9,9,17,18,1]])
mask=token_masks(ids,9,[1,2])
assert int(mask["image"].sum())==2 and int(mask["text"].sum())==1 and int(mask["both"].sum())==3
class Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model=nn.Module()
        self.language_model.model=nn.Module()
        self.language_model.model.layers=nn.ModuleList([nn.Identity()])
fake=Fake()
a=torch.ones(1,6,3)
with steer_decoder(fake,0,torch.ones(3),2.,mask["image"]) as edits:
    b=fake.language_model.model.layers[0](a)
assert torch.equal(b[0,1],torch.full((3,),3.))
assert torch.equal(b[0,3],torch.ones(3))
assert edits["tokens"]==2
assert torch.equal(fake.language_model.model.layers[0](a),a)
assert paired_metrics([dict(id="x",lang="zh",mask="baseline",correct=0),
                       dict(id="x",lang="zh",mask="image",correct=1)])[("zh","image")]["delta_pp"]==100
print("CPU smoke checks passed")
