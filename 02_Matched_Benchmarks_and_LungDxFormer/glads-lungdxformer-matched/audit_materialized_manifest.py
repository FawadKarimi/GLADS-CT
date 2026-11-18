
import json
import collections

with open("/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001/VIEW-MANIFEST.MATERIALIZED.json") as f:
    manifest = json.load(f)

records = manifest["records"]
print(f"Total nodule records in materialized manifest: {len(records)}")

nodule_classes = collections.Counter()
slice_classes = collections.Counter()
discarded_nodules = 0
discarded_slices = 0

total_views = 0

for rec in records:
    # Check reader targets or labels
    # reader_target is [p_benign, p_indeterminate, p_malignant]
    label = rec.get("label")
    views = rec.get("views", [])
    num_views = len(views)
    total_views += num_views
    
    # Check if there is an exact mean malignancy or label
    # In GLADS-CT: 0: benign, 1: indeterminate, 2: malignant
    if label == 0:
        nodule_classes["benign"] += 1
        slice_classes["benign"] += num_views
    elif label == 1:
        nodule_classes["indeterminate"] += 1
        slice_classes["indeterminate"] += num_views
    elif label == 2:
        nodule_classes["malignant"] += 1
        slice_classes["malignant"] += num_views
    else:
        discarded_nodules += 1
        discarded_slices += num_views

print("Materialized Nodule counts by class:", dict(nodule_classes))
print("Materialized Slice counts by class:", dict(slice_classes))
print(f"Total materialized slice views: {total_views}")
