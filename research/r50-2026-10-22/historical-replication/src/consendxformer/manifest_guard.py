"""Read-only verification of the frozen Phase 0 manifest bundle.

The checksum ledger is data inside the bundle, not a trust anchor.  The trust
anchor below is deliberately kept in source code so replacing both a manifest
and its ledger cannot silently bless a new split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from .hashing import sha256_json


FROZEN_BUNDLE_SHA256 = "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
# The ledger is not included in its own bundle hash, so pin its canonical text
# bytes separately to detect even a ledger-only edit.
FROZEN_LEDGER_SHA256 = "b139beb0b329dd1b30baf138e161360c23821c2c4a54fc80cda3f95c328df6b9"
FROZEN_PATIENT_COUNT = 1010
FROZEN_SERIES_COUNT = 1018
FROZEN_SPLIT_COUNTS = {"train": 707, "val": 152, "test": 151}

# This is intentionally an allowlist rather than a list derived from the
# checksum ledger.  In particular, manifest_checksums.json never gets to tell
# the verifier which files constitute the trusted payload.
FROZEN_MANIFEST_FILES = (
    "patient_split.csv",
    "series_split.csv",
    "train_patient_ids.txt",
    "val_patient_ids.txt",
    "test_patient_ids.txt",
    "split_summary.json",
)
CHECKSUM_LEDGER = "manifest_checksums.json"

_PATIENT_COLUMNS = ("patient_id", "split", "split_rank_sha256")
_SERIES_COLUMNS = (
    "patient_id",
    "series_instance_uid",
    "study_instance_uid",
    "modality",
    "split",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ManifestIntegrityError(RuntimeError):
    """Raised when the frozen Phase 0 manifest cannot be trusted."""


@dataclass(frozen=True)
class ManifestGuardReport:
    """Verified frozen mappings plus compact provenance for downstream code."""

    manifest_dir: Path
    bundle_sha256: str
    file_sha256: Mapping[str, str]
    patient_split: pd.DataFrame
    series_split: pd.DataFrame
    summary: Mapping[str, Any]

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            split: int((self.patient_split["split"] == split).sum())
            for split in FROZEN_SPLIT_COUNTS
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable provenance without embedding full tables."""
        return {
            "manifest_dir": str(self.manifest_dir),
            "bundle_sha256": self.bundle_sha256,
            "checksum_ledger_sha256": FROZEN_LEDGER_SHA256,
            "file_sha256": dict(self.file_sha256),
            "unique_patients": int(self.patient_split["patient_id"].nunique()),
            "unique_ct_series": int(self.series_split["series_instance_uid"].nunique()),
            "split_counts": self.split_counts,
        }


def _canonical_payload_sha256(path: Path) -> str:
    """Hash a tracked text payload after Git's CRLF checkout translation.

    Phase 0 recorded hashes for the repository's LF bytes.  Git may check the
    same text out with CRLF on Windows, so only CRLF pairs are canonicalized.
    Other byte changes remain visible and cause verification to fail.
    """
    digest = hashlib.sha256()
    pending_cr = False
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if pending_cr:
                chunk = b"\r" + chunk
                pending_cr = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_cr = True
            digest.update(chunk.replace(b"\r\n", b"\n"))
    if pending_cr:
        digest.update(b"\r")
    return digest.hexdigest()


def _read_ledger(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestIntegrityError(f"Cannot read frozen manifest checksum ledger: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestIntegrityError("Frozen manifest checksum ledger must be a JSON object")

    expected_keys = set(FROZEN_MANIFEST_FILES) | {"bundle_sha256"}
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ManifestIntegrityError(
            "Frozen manifest checksum ledger must contain exactly the six allowed payload files "
            f"and bundle_sha256; missing={missing}, unexpected={unexpected}"
        )
    for name in expected_keys:
        checksum = value[name]
        if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
            raise ManifestIntegrityError(f"Invalid SHA-256 value in checksum ledger for {name!r}")
    return value


def _verify_checksums(manifest_dir: Path) -> tuple[dict[str, str], str]:
    ledger_path = manifest_dir / CHECKSUM_LEDGER
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Frozen manifest checksum ledger is missing: {ledger_path}")
    ledger = _read_ledger(ledger_path)
    actual_ledger_sha = _canonical_payload_sha256(ledger_path)
    if actual_ledger_sha != FROZEN_LEDGER_SHA256:
        raise ManifestIntegrityError(
            "Frozen manifest checksum ledger was modified: "
            f"expected {FROZEN_LEDGER_SHA256}, got {actual_ledger_sha}"
        )

    recorded_bundle = ledger["bundle_sha256"]
    if recorded_bundle != FROZEN_BUNDLE_SHA256:
        raise ManifestIntegrityError(
            "Frozen manifest ledger bundle SHA-256 disagrees with the Phase 0 source pin: "
            f"expected {FROZEN_BUNDLE_SHA256}, got {recorded_bundle}"
        )

    recorded_files = {name: ledger[name] for name in FROZEN_MANIFEST_FILES}
    computed_bundle = sha256_json(recorded_files)
    if computed_bundle != recorded_bundle:
        raise ManifestIntegrityError(
            "Frozen manifest checksum ledger is internally inconsistent: "
            f"computed bundle {computed_bundle}, recorded {recorded_bundle}"
        )

    for name in FROZEN_MANIFEST_FILES:
        path = manifest_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Frozen manifest payload is missing: {path}")
        actual = _canonical_payload_sha256(path)
        expected = recorded_files[name]
        if actual != expected:
            raise ManifestIntegrityError(
                f"Frozen manifest payload checksum mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
    return recorded_files, computed_bundle


def _read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ManifestIntegrityError(f"Cannot parse frozen manifest CSV: {path}") from exc
    if tuple(frame.columns) != columns:
        raise ManifestIntegrityError(
            f"Unexpected columns in {path.name}: expected {list(columns)}, got {list(frame.columns)}"
        )
    if not frame.columns.is_unique:
        raise ManifestIntegrityError(f"Duplicate columns in frozen manifest CSV: {path.name}")
    return frame


def _require_clean_text_column(frame: pd.DataFrame, column: str, source: str) -> None:
    values = frame[column]
    if values.isna().any() or (values == "").any():
        raise ManifestIntegrityError(f"{source} contains missing or blank {column} values")
    if (values != values.str.strip()).any():
        raise ManifestIntegrityError(f"{source} contains whitespace-padded {column} values")


def _validate_patient_split(patient_split: pd.DataFrame) -> dict[str, set[str]]:
    for column in _PATIENT_COLUMNS:
        _require_clean_text_column(patient_split, column, "patient_split.csv")
    if len(patient_split) != FROZEN_PATIENT_COUNT:
        raise ManifestIntegrityError(
            f"Frozen patient count mismatch: expected {FROZEN_PATIENT_COUNT}, got {len(patient_split)}"
        )
    if patient_split["patient_id"].duplicated().any():
        duplicates = sorted(patient_split.loc[patient_split["patient_id"].duplicated(), "patient_id"].unique())
        raise ManifestIntegrityError(f"Patients occur more than once in patient_split.csv: {duplicates[:10]}")
    if not patient_split["split_rank_sha256"].map(lambda value: _SHA256_RE.fullmatch(value) is not None).all():
        raise ManifestIntegrityError("patient_split.csv contains an invalid split_rank_sha256")

    observed_splits = set(patient_split["split"])
    expected_splits = set(FROZEN_SPLIT_COUNTS)
    if observed_splits != expected_splits:
        raise ManifestIntegrityError(
            f"Frozen split names mismatch: expected {sorted(expected_splits)}, got {sorted(observed_splits)}"
        )
    split_sets = {
        split: set(patient_split.loc[patient_split["split"] == split, "patient_id"])
        for split in FROZEN_SPLIT_COUNTS
    }
    actual_counts = {split: len(ids) for split, ids in split_sets.items()}
    if actual_counts != FROZEN_SPLIT_COUNTS:
        raise ManifestIntegrityError(
            f"Frozen split counts mismatch: expected {FROZEN_SPLIT_COUNTS}, got {actual_counts}"
        )
    for left_index, left in enumerate(FROZEN_SPLIT_COUNTS):
        for right in tuple(FROZEN_SPLIT_COUNTS)[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ManifestIntegrityError(
                    f"Patient leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )
    return split_sets


def _validate_series_split(series_split: pd.DataFrame, patient_split: pd.DataFrame) -> None:
    for column in _SERIES_COLUMNS:
        _require_clean_text_column(series_split, column, "series_split.csv")
    if len(series_split) != FROZEN_SERIES_COUNT:
        raise ManifestIntegrityError(
            f"Frozen CT series count mismatch: expected {FROZEN_SERIES_COUNT}, got {len(series_split)}"
        )
    if series_split["series_instance_uid"].duplicated().any():
        duplicates = sorted(
            series_split.loc[series_split["series_instance_uid"].duplicated(), "series_instance_uid"].unique()
        )
        raise ManifestIntegrityError(f"Series occur more than once in series_split.csv: {duplicates[:10]}")
    if set(series_split["split"]) != set(FROZEN_SPLIT_COUNTS):
        raise ManifestIntegrityError("series_split.csv contains an unknown or missing split")

    patient_mapping = patient_split[["patient_id", "split"]].rename(columns={"split": "patient_split"})
    try:
        joined = series_split.merge(
            patient_mapping,
            on="patient_id",
            how="left",
            sort=False,
            validate="many_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ManifestIntegrityError("Patient-to-split mapping is not one-to-one") from exc
    missing = joined["patient_split"].isna()
    if missing.any():
        unknown = sorted(joined.loc[missing, "patient_id"].unique())
        raise ManifestIntegrityError(f"Series reference patients absent from patient_split.csv: {unknown[:10]}")
    mismatched = joined["split"] != joined["patient_split"]
    if mismatched.any():
        examples = joined.loc[mismatched, ["patient_id", "split", "patient_split"]].head(10).to_dict("records")
        raise ManifestIntegrityError(f"Series do not inherit their patient split: {examples}")
    if joined["patient_id"].nunique() != FROZEN_PATIENT_COUNT:
        raise ManifestIntegrityError("Not every frozen patient has a CT series assignment")


def _read_text_projection(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestIntegrityError(f"Cannot read frozen patient-ID projection: {path}") from exc
    values = text.splitlines()
    if any(not value or value != value.strip() for value in values):
        raise ManifestIntegrityError(f"Blank or whitespace-padded patient ID in {path.name}")
    if len(values) != len(set(values)):
        raise ManifestIntegrityError(f"Duplicate patient ID in {path.name}")
    return values


def _validate_text_projections(
    manifest_dir: Path,
    patient_split: pd.DataFrame,
    split_sets: Mapping[str, set[str]],
) -> None:
    for split, expected_set in split_sets.items():
        values = _read_text_projection(manifest_dir / f"{split}_patient_ids.txt")
        expected_order = sorted(expected_set)
        if values != expected_order:
            raise ManifestIntegrityError(
                f"{split}_patient_ids.txt is not the exact sorted projection of patient_split.csv"
            )
        csv_order = patient_split.loc[patient_split["split"] == split, "patient_id"].tolist()
        if csv_order != expected_order:
            raise ManifestIntegrityError(f"patient_split.csv {split} rows are not in frozen patient-ID order")


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestIntegrityError(f"Cannot parse frozen split summary: {path}") from exc
    if not isinstance(summary, dict):
        raise ManifestIntegrityError("split_summary.json must contain a JSON object")
    return summary


def _validate_summary(summary: Mapping[str, Any], series_split: pd.DataFrame) -> None:
    # inventory_csv is provenance only.  It can point to the Phase 0 machine and
    # is intentionally never resolved, opened, or checked by this Phase 1 guard.
    expected_scalars = {
        "collection": "LIDC-IDRI",
        "expected_modality": "CT",
        "observed_unique_patients": FROZEN_PATIENT_COUNT,
        "observed_unique_ct_series": FROZEN_SERIES_COUNT,
        "expected_unique_patients_tcia": FROZEN_PATIENT_COUNT,
        "expected_unique_ct_series_tcia": FROZEN_SERIES_COUNT,
        "matches_tcia_patient_count": True,
        "matches_tcia_ct_series_count": True,
        "patients_with_multiple_ct_series": 8,
        "extra_ct_series_beyond_one_per_patient": 8,
        "strict_full_inventory": True,
        "leakage_check_passed": True,
        "manifest_schema_version": 1,
        "seed": 2026,
        "split_algorithm": "sha256_rank_v1",
    }
    for key, expected in expected_scalars.items():
        if summary.get(key) != expected:
            raise ManifestIntegrityError(
                f"split_summary.json {key!r} mismatch: expected {expected!r}, got {summary.get(key)!r}"
            )
    if summary.get("split_counts") != FROZEN_SPLIT_COUNTS:
        raise ManifestIntegrityError(
            f"split_summary.json split_counts mismatch: {summary.get('split_counts')!r}"
        )
    if summary.get("split_ratios") != {"train": 0.70, "val": 0.15, "test": 0.15}:
        raise ManifestIntegrityError(
            f"split_summary.json split_ratios mismatch: {summary.get('split_ratios')!r}"
        )

    series_per_patient = series_split.groupby("patient_id", sort=False)["series_instance_uid"].nunique()
    actual_multi = int((series_per_patient > 1).sum())
    actual_extra = int(series_per_patient.sum() - len(series_per_patient))
    if actual_multi != 8 or actual_extra != 8:
        raise ManifestIntegrityError(
            "Frozen series multiplicity mismatch: "
            f"patients_with_multiple_ct_series={actual_multi}, extra_series={actual_extra}"
        )


def validate_frozen_manifest_bundle(manifest_dir: str | Path) -> ManifestGuardReport:
    """Verify checksums and semantics of the immutable Phase 0 split.

    This function is strictly read-only.  It neither follows the stale
    ``inventory_csv`` provenance path nor offers a checksum-regeneration path.
    """
    directory = Path(manifest_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Frozen manifest directory does not exist: {directory}")

    file_sha256, bundle_sha256 = _verify_checksums(directory)
    patient_split = _read_csv(directory / "patient_split.csv", _PATIENT_COLUMNS)
    series_split = _read_csv(directory / "series_split.csv", _SERIES_COLUMNS)
    split_sets = _validate_patient_split(patient_split)
    _validate_series_split(series_split, patient_split)
    _validate_text_projections(directory, patient_split, split_sets)
    summary = _read_summary(directory / "split_summary.json")
    _validate_summary(summary, series_split)

    return ManifestGuardReport(
        manifest_dir=directory,
        bundle_sha256=bundle_sha256,
        file_sha256=file_sha256,
        patient_split=patient_split,
        series_split=series_split,
        summary=summary,
    )
