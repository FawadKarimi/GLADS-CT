
import numpy as np
np.int = int

import pylidc as pl
print("pylidc imported successfully!")
scans = pl.query(pl.Scan).all()
print(f"Total scans queried: {len(scans)}")
scan0 = scans[0]
nods = scan0.cluster_annotations()
print(f"Scan 0 clustered nodules: {len(nods)}")
