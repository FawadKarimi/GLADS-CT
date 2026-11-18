
import numpy as np
# Monkeypatch deprecated numpy aliases for pylidc compatibility
np.int = int
np.float = float
np.bool = bool

import pylidc as pl
import collections

scans = pl.query(pl.Scan).all()
print(f"Total scans queried: {len(scans)}")

nodule_counts = collections.Counter()
slice_counts = collections.Counter()
discard_nodules = 0
discard_slices = 0

total_nodules = 0
total_slices = 0

# Sample first 100 scans to get quick distribution and timing
for i, scan in enumerate(scans[:100]):
    nods = scan.cluster_annotations()
    for nod in nods:
        mals = [a.malignancy for a in nod]
        m = np.mean(mals)
        total_nodules += 1
        
        # Calculate number of slices spanned by this nodule
        zs = set()
        for a in nod:
            c = a.contours
            for contour in c:
                zs.add(contour.image_z_position)
        num_slices = len(zs) if len(zs) > 0 else 1
        total_slices += num_slices
        
        if m < 2.5:
            nodule_counts['benign'] += 1
            slice_counts['benign'] += num_slices
        elif m == 3.0:
            nodule_counts['indeterminate'] += 1
            slice_counts['indeterminate'] += num_slices
        elif m > 3.5:
            nodule_counts['malignant'] += 1
            slice_counts['malignant'] += num_slices
        else:
            discard_nodules += 1
            discard_slices += num_slices

print("--- 100 Scans Pilot ---")
print("Total nodules in 100 scans:", total_nodules)
print("Nodules by class (<2.5, ==3.0, >3.5):", dict(nodule_counts), "Discarded:", discard_nodules)
print("Total slices in 100 scans:", total_slices)
print("Slices by class:", dict(slice_counts), "Discarded slices:", discard_slices)

factor = len(scans) / 100
print("Extrapolated slices to 1018 scans:", {k: round(v * factor) for k, v in slice_counts.items()})
print("Total extrapolated slices:", round(sum(slice_counts.values()) * factor))
