#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.config import load_config
from consendxformer.hashing import sha256_json
from consendxformer.manifest import build_manifests, write_manifests


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create an immutable patient-disjoint LIDC-IDRI split from a CT series inventory."
    )
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--inventory-csv", required=True, help="TCIA DICOM metadata digest or equivalent CT series inventory CSV")
    ap.add_argument(
        "--allow-incomplete-inventory",
        action="store_true",
        help="DEBUG ONLY: allow counts other than TCIA's locked 1,010 patients / 1,018 CT series.",
    )
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    config_hash = sha256_json(cfg)
    result = build_manifests(
        inventory_csv=args.inventory_csv,
        cfg=cfg,
        allow_incomplete_inventory=args.allow_incomplete_inventory,
    )
    out_dir = ROOT / cfg["paths"]["manifest_dir"]
    written = write_manifests(result, out_dir, config_hash=config_hash)
    s = result["summary"]

    print("PHASE0_MANIFEST=PASS")
    print(f"observed_unique_patients={s['observed_unique_patients']}")
    print(f"observed_unique_ct_series={s['observed_unique_ct_series']}")
    print(f"expected_unique_patients_tcia={s['expected_unique_patients_tcia']}")
    print(f"expected_unique_ct_series_tcia={s['expected_unique_ct_series_tcia']}")
    print(f"patients_with_multiple_ct_series={s['patients_with_multiple_ct_series']}")
    print(f"extra_ct_series_beyond_one_per_patient={s['extra_ct_series_beyond_one_per_patient']}")
    print(f"split_counts={s['split_counts']}")
    print(f"leakage_check_passed={s['leakage_check_passed']}")
    print(f"patient_split_csv={written['patient_split_csv']}")
    print(f"series_split_csv={written['series_split_csv']}")
    print(f"bundle_sha256={written['bundle_sha256']}")
    print("NOTE: Phase 0 does not know the 4,095-nodule cohort yet. Phase 2 will filter nodules while preserving these patient assignments.")


if __name__ == "__main__":
    main()
