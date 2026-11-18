
import json
import os
import numpy as np

with open("/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001/VIEW-MANIFEST.MATERIALIZED.json") as f:
    manifest = json.load(f)

records = manifest["records"]
missing_files = 0
found_files = 0

rep_root = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/03_ConsenDxFormer_and_Datasets/ConsenDxFormer/data/processed/consendxformer_phase2/rendered_64"

for rec in records:
    views = rec.get("views", [])
    for v in views:
        p = v.get("image_path")
        if not os.path.isabs(p):
            p = os.path.join(rep_root, p)
        if os.path.exists(p):
            found_files += 1
        else:
            missing_files += 1

print(f"Verified files: {found_files} present, {missing_files} missing (Total expected: {found_files + missing_files})")
