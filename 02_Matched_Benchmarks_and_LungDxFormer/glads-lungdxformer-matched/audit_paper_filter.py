
import json
import collections
import numpy as np

with open("/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001/VIEW-MANIFEST.MATERIALIZED.json") as f:
    manifest = json.load(f)

records = manifest["records"]
print(f"Total nodule records: {len(records)}")

# Let's inspect where reader ratings or annotation IDs come from
# In the first record earlier:
# "selected_annotation_ids": [88, 89]
# Can we query pylidc Annotation objects with these IDs?
import pylidc as pl
np.int = int

nodules_under_paper_rule = collections.Counter()
slices_under_paper_rule = collections.Counter()
discarded_nodules = 0
discarded_slices = 0

all_slice_records = []

for i, rec in enumerate(records):
    ann_ids = rec.get("selected_annotation_ids", [])
    if not ann_ids:
        continue
    # Fetch annotations
    anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_(ann_ids)).all()
    if not anns:
        continue
    mals = [a.malignancy for a in anns]
    mean_mal = float(np.mean(mals))
    
    # Paper's exact rule:
    # benign: mean < 2.5
    # indeterminate: mean == 3.0
    # malignant: mean > 3.5
    # otherwise: discard
    if mean_mal < 2.5:
        cls_name = "benign"
        cls_id = 0
    elif mean_mal == 3.0:
        cls_name = "indeterminate"
        cls_id = 1
    elif mean_mal > 3.5:
        cls_name = "malignant"
        cls_id = 2
    else:
        cls_name = "discard"
        cls_id = -1
        
    views = rec.get("views", [])
    num_views = len(views)
    
    if cls_id != -1:
        nodules_under_paper_rule[cls_name] += 1
        slices_under_paper_rule[cls_name] += num_views
        for v in views:
            all_slice_records.append({
                "nodule_id": rec["nodule_id"],
                "patient_id": rec["patient_id"],
                "view_id": v["view_id"],
                "slice_index": v.get("slice_index"),
                "image_path": v["image_path"],
                "image_sha256": v["image_sha256"],
                "mean_malignancy": mean_mal,
                "label": cls_id,
                "class_name": cls_name,
                "support_readers": v.get("support_readers", len(ann_ids)),
                "disagreement": rec.get("disagreement", 0.0)
            })
    else:
        discarded_nodules += 1
        discarded_slices += num_views

print("=== NODULES UNDER PAPER RULE ===")
print("Kept nodules:", dict(nodules_under_paper_rule), "Total kept:", sum(nodules_under_paper_rule.values()))
print("Discarded borderline nodules:", discarded_nodules)

print("=== SLICES UNDER PAPER RULE ===")
print("Kept slices:", dict(slices_under_paper_rule), "Total kept:", sum(slices_under_paper_rule.values()))
print("Discarded borderline slices:", discarded_slices)
print("Total slice records generated:", len(all_slice_records))
