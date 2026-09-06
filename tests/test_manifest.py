from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.manifest import allocate_split_counts, build_patient_split, build_manifests
from consendxformer.config import load_config


def test_1010_allocation_is_deterministic():
    counts = allocate_split_counts(
        1010,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        ["train", "val", "test"],
    )
    assert counts == {"train": 707, "val": 152, "test": 151}


def test_patient_hash_split_is_order_independent():
    ids = [f"LIDC-IDRI-{i:04d}" for i in range(1, 51)]
    a = build_patient_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, 2026, ["train", "val", "test"])
    b = build_patient_split(reversed(ids), {"train": 0.7, "val": 0.15, "test": 0.15}, 2026, ["train", "val", "test"])
    pd.testing.assert_frame_equal(a, b)
    assert a["patient_id"].nunique() == len(ids)
    assert a.groupby("patient_id")["split"].nunique().max() == 1


def test_multiple_series_stay_with_patient(tmp_path):
    rows = []
    for i in range(1, 13):
        pid = f"LIDC-IDRI-{i:04d}"
        rows.append({"Subject ID": pid, "Series UID": f"1.2.3.{i}.1", "Modality": "CT"})
        if i in (1, 2):
            rows.append({"Subject ID": pid, "Series UID": f"1.2.3.{i}.2", "Modality": "CT"})
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    cfg = load_config(ROOT / "configs" / "phase0.yaml")
    result = build_manifests(csv_path, cfg, allow_incomplete_inventory=True)
    series = result["series_split"]
    assert series.groupby("patient_id")["split"].nunique().max() == 1
    assert result["summary"]["observed_unique_patients"] == 12
    assert result["summary"]["observed_unique_ct_series"] == 14
