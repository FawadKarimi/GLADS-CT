from __future__ import annotations
from pathlib import Path
import numpy as np
import pydicom

def load_dicom_series(series_dir: str | Path) -> tuple[np.ndarray, list]:
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    datasets = [pydicom.dcmread(str(f)) for f in files]
    datasets.sort(key=lambda d: float(getattr(d, "ImagePositionPatient", [0,0,getattr(d, "InstanceNumber", 0)])[2]
                                  if hasattr(d, "ImagePositionPatient") else getattr(d, "InstanceNumber", 0)))
    volume = np.stack([ds.pixel_array.astype(np.int16) for ds in datasets], axis=0)
    if hasattr(datasets[0], "RescaleSlope") and hasattr(datasets[0], "RescaleIntercept"):
        slope = float(datasets[0].RescaleSlope)
        intercept = float(datasets[0].RescaleIntercept)
        volume = volume.astype(np.float32) * slope + intercept
    return volume.astype(np.float32), datasets
