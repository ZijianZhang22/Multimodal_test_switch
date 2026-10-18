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
        yield (source,mask,layer,alpha), {
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
