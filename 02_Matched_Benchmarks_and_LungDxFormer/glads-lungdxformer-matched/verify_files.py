
import json
import os
import numpy as np

with open("/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001/VIEW-MANIFEST.MATERIALIZED.json") as f:
    manifest = json.load(f)

records = manifest["records"]
missing_files = 0
found_files = 0
shapes = set()
dtypes = set()

for rec in records:
    views = rec.get("views", [])
    for v in views:
        p = v.get("image_path")
        if not os.path.isabs(p):
            p = os.path.join("/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001", p)
        if os.path.exists(p):
            found_files += 1
            if found_files <= 5:
                arr = np.load(p)
                shapes.add(arr.shape)
                dtypes.add(str(arr.dtype))
        else:
            missing_files += 1

print(f"Verified files: {found_files} present, {missing_files} missing")
print(f"Sample shapes: {shapes}, dtypes: {dtypes}")
