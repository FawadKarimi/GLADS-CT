
import json
import os
import pandas as pd
import hashlib

manifest_path = "/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001/VIEW-MANIFEST.MATERIALIZED.json"
rep_root = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/03_ConsenDxFormer_and_Datasets/ConsenDxFormer/data/processed/consendxformer_phase2/rendered_64"

with open(manifest_path) as f:
    manifest = json.load(f)

records = manifest["records"]
slice_rows = []

for rec in records:
    label = rec["label"]
    if label == 0:
        cname = "benign"
    elif label == 1:
        cname = "indeterminate"
    elif label == 2:
        cname = "malignant"
    else:
        continue

    rt = rec.get("reader_target", [0.0, 0.0, 0.0])
    disag = rec.get("disagreement", 0.0)

    for v in rec.get("views", []):
        p = v.get("image_path")
        if not os.path.isabs(p):
            p = os.path.join(rep_root, p)
        
        slice_rows.append({
            "slice_id": v["view_id"],
            "patient_id": rec["patient_id"],
            "nodule_id": rec["nodule_id"],
            "slice_index": v.get("slice_index"),
            "z": v.get("z", 0.0),
            "image_path": p,
            "label": label,
            "class_name": cname,
            "disagreement": disag,
            "reader_target_0": rt[0],
            "reader_target_1": rt[1],
            "reader_target_2": rt[2],
            "is_representative": v.get("existing_representative", False),
            "consensus_area_pixels": v.get("consensus_area_pixels", 0)
        })

df = pd.DataFrame(slice_rows)
out_csv = "/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/02_Matched_Benchmarks_and_LungDxFormer/glads-lungdxformer-matched/lidc_slice_manifest.csv"
df.to_csv(out_csv, index=False)

# Compute sha256
with open(out_csv, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()

print(f"Exported {len(df)} slices to {out_csv}")
print(f"SHA-256: {h}")
print("\n--- Class Distribution (Slices) ---")
print(df["class_name"].value_counts())
print("\n--- Class Distribution (Nodules) ---")
print(df.drop_duplicates("nodule_id")["class_name"].value_counts())
print("\n--- Unique Patients ---")
print("Total unique patients:", df["patient_id"].nunique())
