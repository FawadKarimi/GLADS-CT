from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math
import pandas as pd

from .hashing import patient_rank_hash, sha256_file, sha256_json


@dataclass(frozen=True)
class DetectedColumns:
    patient_id: str
    series_uid: str
    study_uid: str | None
    modality: str | None


def _detect_column(columns: Iterable[str], candidates: list[str], required: bool) -> str | None:
    existing = {str(c).strip(): str(c) for c in columns}
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        if cand in existing:
            return existing[cand]
        if cand.strip().lower() in lower:
            return lower[cand.strip().lower()]
    if required:
        raise ValueError(
            f"Could not detect required column. Tried {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None


def detect_columns(df: pd.DataFrame, candidate_cfg: dict) -> DetectedColumns:
    return DetectedColumns(
        patient_id=_detect_column(df.columns, candidate_cfg["patient_id"], required=True),
        series_uid=_detect_column(df.columns, candidate_cfg["series_uid"], required=True),
        study_uid=_detect_column(df.columns, candidate_cfg.get("study_uid", []), required=False),
        modality=_detect_column(df.columns, candidate_cfg.get("modality", []), required=False),
    )


def _clean_id(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    cleaned = series.astype(str).str.strip()
    if (cleaned == "").any():
        raise ValueError(f"{name} contains blank values")
    return cleaned


def allocate_split_counts(n: int, ratios: dict[str, float], tie_priority: list[str]) -> dict[str, int]:
    if n <= 0:
        raise ValueError("Cannot split zero patients")
    if not ratios:
        raise ValueError("No split ratios configured")
    total = sum(float(v) for v in ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0; got {total}")
    if set(tie_priority) != set(ratios):
        raise ValueError("tie_priority must contain each split name exactly once")

    raw = {k: n * float(v) for k, v in ratios.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    remaining = n - sum(counts.values())
    priority_index = {name: i for i, name in enumerate(tie_priority)}
    order = sorted(
        ratios,
        key=lambda k: (-(raw[k] - counts[k]), priority_index[k]),
    )
    for k in order[:remaining]:
        counts[k] += 1
    assert sum(counts.values()) == n
    return counts


def build_patient_split(
    patient_ids: Iterable[str],
    ratios: dict[str, float],
    seed: int,
    tie_priority: list[str],
    algorithm_version: str = "sha256_rank_v1",
) -> pd.DataFrame:
    patients = sorted({str(x).strip() for x in patient_ids})
    if any(not x for x in patients):
        raise ValueError("Blank patient ID encountered")
    counts = allocate_split_counts(len(patients), ratios, tie_priority)

    ranked = sorted(
        ((patient_rank_hash(pid, seed, algorithm_version), pid) for pid in patients),
        key=lambda x: (x[0], x[1]),
    )

    split_order = list(ratios.keys())
    rows = []
    start = 0
    for split_name in split_order:
        end = start + counts[split_name]
        for rank_hash, patient_id in ranked[start:end]:
            rows.append({
                "patient_id": patient_id,
                "split": split_name,
                "split_rank_sha256": rank_hash,
            })
        start = end

    out = pd.DataFrame(rows).sort_values(["split", "patient_id"]).reset_index(drop=True)
    if out["patient_id"].duplicated().any():
        raise AssertionError("Patient appears more than once in patient split")
    return out


def build_manifests(
    inventory_csv: str | Path,
    cfg: dict,
    allow_incomplete_inventory: bool = False,
) -> dict:
    inventory_csv = Path(inventory_csv).resolve()
    if not inventory_csv.exists():
        raise FileNotFoundError(inventory_csv)

    df = pd.read_csv(inventory_csv, low_memory=False)
    detected = detect_columns(df, cfg["dataset_inventory"]["column_candidates"])

    working = pd.DataFrame({
        "patient_id": _clean_id(df[detected.patient_id], "patient_id"),
        "series_instance_uid": _clean_id(df[detected.series_uid], "series_instance_uid"),
    })
    if detected.study_uid:
        working["study_instance_uid"] = _clean_id(df[detected.study_uid], "study_instance_uid")
    else:
        working["study_instance_uid"] = ""

    modality_filtered = False
    expected_modality = str(cfg["dataset_inventory"].get("expected_modality", "CT")).strip().upper()
    if detected.modality:
        modality = _clean_id(df[detected.modality], "modality").str.upper()
        working["modality"] = modality
        working = working[working["modality"] == expected_modality].copy()
        modality_filtered = True
    else:
        working["modality"] = expected_modality

    working = working.drop_duplicates(subset=["patient_id", "series_instance_uid"]).reset_index(drop=True)
    if working.empty:
        raise ValueError("No CT series remain after modality filtering")

    uid_patient_counts = working.groupby("series_instance_uid")["patient_id"].nunique()
    bad_uids = uid_patient_counts[uid_patient_counts > 1]
    if not bad_uids.empty:
        raise ValueError(
            "A SeriesInstanceUID is linked to more than one patient ID; inventory is inconsistent. "
            f"Examples: {bad_uids.index[:5].tolist()}"
        )

    unique_patients = int(working["patient_id"].nunique())
    unique_series = int(working["series_instance_uid"].nunique())
    per_patient_series = working.groupby("patient_id")["series_instance_uid"].nunique()
    multi_series_patients = per_patient_series[per_patient_series > 1]

    ref = cfg["dataset_inventory"]["tcia_reference"]
    expected_patients = int(ref["unique_patients"])
    expected_series = int(ref["unique_ct_series"])
    patients_match = unique_patients == expected_patients
    series_match = unique_series == expected_series

    strict = bool(cfg["dataset_inventory"].get("enforce_full_inventory", True)) and not allow_incomplete_inventory
    if strict and (not patients_match or not series_match):
        raise ValueError(
            "Inventory does not match the locked TCIA full-collection reference. "
            f"Observed patients={unique_patients}, CT series={unique_series}; "
            f"expected patients={expected_patients}, CT series={expected_series}. "
            "Do not coerce the counts. Fix or replace the inventory. For debugging only, "
            "rerun with --allow-incomplete-inventory; that output must not be used for final experiments."
        )

    split_cfg = cfg["split"]
    seed = int(cfg["reproducibility"]["seed"])
    algorithm_version = str(cfg["reproducibility"]["split_algorithm"])
    patient_split = build_patient_split(
        working["patient_id"].unique(),
        ratios=split_cfg["ratios"],
        seed=seed,
        tie_priority=split_cfg["tie_priority"],
        algorithm_version=algorithm_version,
    )

    series_split = working.merge(patient_split[["patient_id", "split"]], on="patient_id", how="left", validate="many_to_one")
    if series_split["split"].isna().any():
        raise AssertionError("Some CT series did not receive a patient split")

    # Strong leakage check: each patient belongs to one split; all its series inherit that split.
    leakage = series_split.groupby("patient_id")["split"].nunique()
    if int(leakage.max()) != 1:
        raise AssertionError("Patient leakage detected across splits")

    return {
        "inventory_csv": str(inventory_csv),
        "inventory_sha256": sha256_file(inventory_csv),
        "detected_columns": detected.__dict__,
        "modality_filtered_using_inventory_column": modality_filtered,
        "patient_split": patient_split,
        "series_split": series_split.sort_values(["split", "patient_id", "series_instance_uid"]).reset_index(drop=True),
        "summary": {
            "collection": cfg["dataset_inventory"].get("expected_collection", "LIDC-IDRI"),
            "expected_modality": expected_modality,
            "observed_unique_patients": unique_patients,
            "observed_unique_ct_series": unique_series,
            "expected_unique_patients_tcia": expected_patients,
            "expected_unique_ct_series_tcia": expected_series,
            "matches_tcia_patient_count": patients_match,
            "matches_tcia_ct_series_count": series_match,
            "patients_with_multiple_ct_series": int(len(multi_series_patients)),
            "extra_ct_series_beyond_one_per_patient": int(unique_series - unique_patients),
            "strict_full_inventory": strict,
            "seed": seed,
            "split_algorithm": algorithm_version,
            "split_ratios": split_cfg["ratios"],
            "split_counts": patient_split["split"].value_counts().sort_index().to_dict(),
            "leakage_check_passed": True,
        },
    }


def write_manifests(result: dict, output_dir: str | Path, config_hash: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_path = output_dir / "patient_split.csv"
    series_path = output_dir / "series_split.csv"
    summary_path = output_dir / "split_summary.json"

    result["patient_split"].to_csv(patient_path, index=False)
    result["series_split"].to_csv(series_path, index=False)

    for split_name in ["train", "val", "test"]:
        ids = result["patient_split"].loc[result["patient_split"]["split"] == split_name, "patient_id"].sort_values()
        (output_dir / f"{split_name}_patient_ids.txt").write_text("\n".join(ids.tolist()) + "\n", encoding="utf-8")

    summary = dict(result["summary"])
    summary["inventory_csv"] = result["inventory_csv"]
    summary["inventory_sha256"] = result["inventory_sha256"]
    summary["detected_columns"] = result["detected_columns"]
    summary["config_sha256"] = config_hash
    summary["manifest_schema_version"] = 1
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    files = [
        patient_path,
        series_path,
        summary_path,
        output_dir / "train_patient_ids.txt",
        output_dir / "val_patient_ids.txt",
        output_dir / "test_patient_ids.txt",
    ]
    checksums = {p.name: sha256_file(p) for p in files}
    checksums["bundle_sha256"] = sha256_json(checksums)
    checksum_path = output_dir / "manifest_checksums.json"
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "patient_split_csv": str(patient_path),
        "series_split_csv": str(series_path),
        "summary_json": str(summary_path),
        "checksum_json": str(checksum_path),
        "bundle_sha256": checksums["bundle_sha256"],
    }
