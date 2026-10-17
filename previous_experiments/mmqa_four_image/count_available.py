#!/usr/bin/env python3
"""Count MultiModalQA dev examples by candidate-image count under the experiment's exact filters.

Run from mmqa_four_image/: python count_available.py
No GPU or image ZIP required.
"""
from collections import Counter, defaultdict
from pathlib import Path
from experiment import read_jsonl, table_to_text

data=Path("data/multimodalqa/dataset")
images={x["id"] for x in read_jsonl(data/"MMQA_images.jsonl.gz")}
texts={x["id"]:x for x in read_jsonl(data/"MMQA_texts.jsonl.gz")}
tables={x["id"]:x for x in read_jsonl(data/"MMQA_tables.jsonl.gz")}

raw=Counter()
eligible=Counter()
support_per_n=defaultdict(Counter)
multi_support=Counter()
example_qids=defaultdict(list)

for q in read_jsonl(data/"MMQA_dev.jsonl.gz"):
    m=q.get("metadata") or {}
    ids=m.get("image_doc_ids") or []
    n=len(ids)
    raw[n]+=1
    support=q.get("supporting_context") or []
    supimgs=[x["doc_id"] for x in support
             if x.get("doc_part")=="image" and x.get("doc_id") in ids]
    modes={x.get("doc_part") for x in support}
    if len(ids)!=len(set(ids)) or any(i not in images for i in ids): continue
    if len(support)<2 or len(modes)<2 or not supimgs: continue
    if "image" not in (m.get("modalities") or []): continue
    tdocs=[texts[i] for i in m.get("text_doc_ids",[]) if i in texts]
    tab=tables.get(m.get("table_id"))
    tabtext=table_to_text(tab) if tab else ""
    if len(tabtext)+sum(len(x.get("text","")) for x in tdocs)>14000: continue
    eligible[n]+=1
    support_per_n[n][len(set(supimgs))]+=1
    if len(set(supimgs))>=2: multi_support[n]+=1
    if len(example_qids[n])<3:example_qids[n].append(q.get("qid"))

print("Images | Dev raw | Eligible under current filters | >=2 supporting images")
for n in range(1,16):
    print(f"{n:>6} | {raw[n]:>7} | {eligible[n]:>30} | {multi_support[n]:>20}")
print("\nEligible support-image counts for 2,3,4 candidates:")
for n in (2,3,4):
    print(f"{n} images: {dict(sorted(support_per_n[n].items()))}; example QIDs: {example_qids[n]}")
print("Eligible with 2/3/4 candidate images combined:",sum(eligible[n] for n in (2,3,4)))
