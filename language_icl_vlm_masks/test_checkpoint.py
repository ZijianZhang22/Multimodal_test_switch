"""CPU-only tests for checkpointing and resuming long evaluations."""
import csv
import tempfile
from pathlib import Path
from checkpoint_eval import evaluate_resumable

HEADERS = ["id","source","mask","layer","alpha","response"]
samples = [{"id":"a"},{"id":"b"}]
configs = [("none","baseline",-1,0.0),("base","image",12,1.0)]
calls = []

def generate(sample, missing):
    for source,mask,layer,alpha in missing:
        calls.append((sample["id"],source))
        yield {
            "id":sample["id"],"source":source,"mask":mask,"layer":layer,
            "alpha":alpha,"response":"A"}

with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)/"validation.csv"
    partial = Path(temp)/"validation.partial.csv"
    with partial.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow({"id":"a","source":"none","mask":"baseline",
                         "layer":-1,"alpha":0.0,"response":"A"})
    rows = evaluate_resumable(samples,configs,path,HEADERS,generate,"validation")
    assert len(rows)==4 and path.exists() and not partial.exists()
    assert calls == [("a","base"),("b","none"),("b","base")]
    calls.clear()
    rows2=evaluate_resumable(samples,configs,path,HEADERS,generate,"validation")
    assert len(rows2)==4 and not calls
print("Checkpoint resume unit tests passed")

# Regression: validation selection must work with CSV string layer/alpha fields.
from checkpoint_eval import select_validation_parameters
validation = [
    {"id": "a", "source": "base", "mask": "image",
     "layer": "8", "alpha": "0.5", "correct": "0"},
    {"id": "b", "source": "base", "mask": "image",
     "layer": "8", "alpha": "0.5", "correct": "0"},
    {"id": "a", "source": "base", "mask": "image",
     "layer": "12", "alpha": "1.0", "correct": "1"},
    {"id": "b", "source": "base", "mask": "image",
     "layer": "12", "alpha": "1.0", "correct": "1"},
]
# Every cross-product cell must exist; give four combinations below.
validation += [
    {"id": id_, "source":"base", "mask":"image",
     "layer":str(layer), "alpha":str(alpha), "correct":"0"}
    for layer, alpha in ((8, 1.0), (12, 0.5))
    for id_ in ("a","b")
]
chosen = select_validation_parameters(validation, ["a","b"],
                                      ["base"], ["image"], [8,12], [0.5,1.0])
assert chosen == {"base":{"image":{"layer":12,"alpha":1.0}}}
try:
    select_validation_parameters(validation[:-1], ["a","b"],
                                 ["base"], ["image"], [8,12], [0.5,1.0])
except RuntimeError as e:
    assert "Incomplete validation cells" in str(e)
else:
    raise AssertionError("Incomplete validation must be rejected")
print("CSV validation selection regression passed")
