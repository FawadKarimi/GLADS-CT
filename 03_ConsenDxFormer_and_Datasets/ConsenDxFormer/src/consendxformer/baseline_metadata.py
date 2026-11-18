"""External metadata translation for the locked LungDxFormer baseline.

The adapter performs the only permitted split operation in Phase 1: a
many-to-one join from sample rows to the already frozen patient mapping.  It
does not inspect image files and never creates a row-level random split.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .manifest_guard import (
    FROZEN_BUNDLE_SHA256,
    FROZEN_PATIENT_COUNT,
    FROZEN_SPLIT_COUNTS,
    ManifestGuardReport,
    validate_frozen_manifest_bundle,
)


CLASS_NAMES = ("benign", "indeterminate", "malignant")
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {index: name for name, index in LABEL_TO_ID.items()}
CANONICAL_COLUMNS = ("patient_id", "image_path", "label", "label_id", "split")
_INPUT_COLUMNS = ("patient_id", "image_path", "label")
_SPLIT_ORDER = tuple(FROZEN_SPLIT_COUNTS)


class MetadataAdapterError(ValueError):
    """Raised for metadata that cannot safely enter the three-class baseline."""


@dataclass(frozen=True)
class AdaptedMetadataFiles:
    """Paths and row counts produced by :func:`write_adapted_metadata`."""

    output_dir: Path
    combined_csv: Path
    split_csvs: Mapping[str, Path]
    row_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "combined_csv": str(self.combined_csv),
            "split_csvs": {split: str(path) for split, path in self.split_csvs.items()},
            "row_counts": dict(self.row_counts),
        }


# A descriptive compatibility alias for callers that prefer a report name.
MetadataWriteReport = AdaptedMetadataFiles


def _resolve_columns(columns: Mapping[str, str] | None) -> dict[str, str]:
    resolved = dict(zip(_INPUT_COLUMNS, _INPUT_COLUMNS)) if columns is None else dict(columns)
    expected = set(_INPUT_COLUMNS)
    actual = set(resolved)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise MetadataAdapterError(
            f"Column mapping must define exactly {list(_INPUT_COLUMNS)}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(not isinstance(value, str) or not value for value in resolved.values()):
        raise MetadataAdapterError("Configured metadata column names must be non-empty strings")
    if len(set(resolved.values())) != len(resolved):
        raise MetadataAdapterError("Configured patient_id, image_path, and label columns must be distinct")
    return resolved


def _load_input(input_csv_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_csv_or_df, pd.DataFrame):
        frame = input_csv_or_df.copy(deep=True)
    elif isinstance(input_csv_or_df, (str, Path)):
        path = Path(input_csv_or_df).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input metadata CSV does not exist: {path}")
        try:
            # Preserve identifiers byte-for-byte (notably leading zeroes). All
            # three conceptual fields are normalized explicitly below.
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise MetadataAdapterError(f"Cannot parse input metadata CSV: {path}") from exc
    else:
        raise TypeError("input_csv_or_df must be a CSV path or pandas DataFrame")
    if not frame.columns.is_unique:
        raise MetadataAdapterError("Input metadata contains duplicate column names")
    return frame


def _clean_text(values: pd.Series, concept: str) -> pd.Series:
    if values.isna().any():
        rows = values.index[values.isna()].tolist()[:10]
        raise MetadataAdapterError(f"{concept} contains missing values at input rows {rows}")
    cleaned = values.map(str).str.strip()
    blank = cleaned == ""
    if blank.any():
        rows = cleaned.index[blank].tolist()[:10]
        raise MetadataAdapterError(f"{concept} contains blank values at input rows {rows}")
    return cleaned


def _canonical_label(value: Any) -> tuple[str, int]:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in LABEL_TO_ID:
            return normalized, LABEL_TO_ID[normalized]
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = math.nan
        if math.isfinite(numeric) and numeric.is_integer():
            label_id = int(numeric)
            if label_id in ID_TO_LABEL:
                return ID_TO_LABEL[label_id], label_id
        raise MetadataAdapterError(
            f"Invalid label {value!r}; expected {list(CLASS_NAMES)} or numeric 0, 1, 2"
        )
    if isinstance(value, bool):
        raise MetadataAdapterError("Boolean labels are invalid; expected a canonical class or numeric 0, 1, 2")
    if isinstance(value, numbers.Integral):
        label_id = int(value)
    elif isinstance(value, numbers.Real) and float(value).is_integer():
        label_id = int(value)
    else:
        raise MetadataAdapterError(
            f"Invalid label {value!r}; expected exactly {list(CLASS_NAMES)} or numeric 0, 1, 2"
        )
    if label_id not in ID_TO_LABEL:
        raise MetadataAdapterError(f"Invalid numeric label {value!r}; expected 0, 1, or 2")
    return ID_TO_LABEL[label_id], label_id


def _canonicalize_labels(values: pd.Series) -> tuple[list[str], list[int]]:
    if values.isna().any():
        rows = values.index[values.isna()].tolist()[:10]
        raise MetadataAdapterError(f"label contains missing values at input rows {rows}")
    labels: list[str] = []
    label_ids: list[int] = []
    for row, value in values.items():
        try:
            label, label_id = _canonical_label(value)
        except MetadataAdapterError as exc:
            raise MetadataAdapterError(f"Invalid label at input row {row}: {exc}") from exc
        labels.append(label)
        label_ids.append(label_id)
    return labels, label_ids


def _patient_mapping(report: ManifestGuardReport) -> pd.DataFrame:
    if not isinstance(report, ManifestGuardReport):
        raise TypeError("manifest_report must be returned by validate_frozen_manifest_bundle")
    if report.bundle_sha256 != FROZEN_BUNDLE_SHA256:
        raise MetadataAdapterError(
            f"Manifest report is not the frozen Phase 0 bundle: {report.bundle_sha256}"
        )

    # DataFrames remain mutable even when their containing dataclass is frozen.
    # Re-establish trust from the pinned on-disk bundle instead of consuming a
    # caller-held frame that may have changed after validation.
    verified_report = validate_frozen_manifest_bundle(report.manifest_dir)
    patient_split = verified_report.patient_split
    required = {"patient_id", "split"}
    if not required.issubset(patient_split.columns):
        raise MetadataAdapterError("Manifest report lacks patient_id/split mapping columns")
    mapping = patient_split[["patient_id", "split"]].copy()
    if len(mapping) != FROZEN_PATIENT_COUNT or mapping["patient_id"].nunique() != FROZEN_PATIENT_COUNT:
        raise MetadataAdapterError("Manifest report patient mapping is not the frozen 1010-patient mapping")
    if mapping["patient_id"].isna().any() or mapping["split"].isna().any():
        raise MetadataAdapterError("Manifest report patient mapping contains missing values")
    mapping["patient_id"] = mapping["patient_id"].map(str)
    actual_counts = {
        split: int((mapping["split"] == split).sum())
        for split in FROZEN_SPLIT_COUNTS
    }
    if actual_counts != FROZEN_SPLIT_COUNTS or not mapping["split"].isin(_SPLIT_ORDER).all():
        raise MetadataAdapterError("Manifest report split assignments do not match the frozen split")
    return mapping


def adapt_metadata(
    input_csv_or_df: str | Path | pd.DataFrame,
    manifest_report: ManifestGuardReport,
    *,
    columns: Mapping[str, str] | None = None,
    unknown_patient_policy: str = "error",
) -> pd.DataFrame:
    """Translate sample metadata and attach the frozen patient split.

    Output rows retain input order.  ``image_path`` is treated as metadata: its
    target is never opened or required to exist during adaptation.
    """
    if unknown_patient_policy not in {"error", "drop"}:
        raise MetadataAdapterError("unknown_patient_policy must be exactly 'error' or 'drop'")
    source = _load_input(input_csv_or_df)
    column_map = _resolve_columns(columns)
    missing = [actual for actual in column_map.values() if actual not in source.columns]
    if missing:
        raise MetadataAdapterError(
            f"Input metadata is missing configured columns {missing}; available={list(source.columns)}"
        )
    if source.empty:
        raise MetadataAdapterError("Input metadata contains no sample rows")

    patient_ids = _clean_text(source[column_map["patient_id"]], "patient_id")
    image_paths = _clean_text(source[column_map["image_path"]], "image_path")
    labels, label_ids = _canonicalize_labels(source[column_map["label"]])
    canonical = pd.DataFrame(
        {
            "patient_id": patient_ids.to_numpy(),
            "image_path": image_paths.to_numpy(),
            "label": labels,
            "label_id": pd.array(label_ids, dtype="int64"),
            "_input_order": range(len(source)),
        }
    )

    mapping = _patient_mapping(manifest_report)
    try:
        joined = canonical.merge(
            mapping,
            on="patient_id",
            how="left",
            sort=False,
            validate="many_to_one",
        )
    except pd.errors.MergeError as exc:
        raise MetadataAdapterError("Frozen patient mapping is not many-to-one joinable") from exc

    unknown = joined["split"].isna()
    if unknown.any():
        unknown_ids = sorted(joined.loc[unknown, "patient_id"].unique())
        if unknown_patient_policy == "error":
            raise MetadataAdapterError(
                f"Input metadata contains {len(unknown_ids)} patient ID(s) absent from the frozen split: "
                f"{unknown_ids[:10]}"
            )
        joined = joined.loc[~unknown].copy()
    if joined.empty:
        raise MetadataAdapterError("No metadata rows remain after frozen-patient validation")

    joined = joined.sort_values("_input_order", kind="stable").reset_index(drop=True)
    return joined.loc[:, CANONICAL_COLUMNS].copy()


def _validate_adapted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise MetadataAdapterError("Adapted metadata contains duplicate column names")
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise MetadataAdapterError(f"Adapted metadata is missing canonical columns: {missing}")
    canonical = frame.loc[:, CANONICAL_COLUMNS].copy()
    for column in ("patient_id", "image_path"):
        canonical[column] = _clean_text(canonical[column], column).to_numpy()
    if canonical["split"].isna().any() or not canonical["split"].isin(_SPLIT_ORDER).all():
        raise MetadataAdapterError(f"Adapted metadata split must be one of {list(_SPLIT_ORDER)}")

    expected_labels, expected_ids = _canonicalize_labels(canonical["label"])
    if canonical["label"].tolist() != expected_labels:
        raise MetadataAdapterError("Adapted metadata labels must already use canonical class names")
    actual_ids: list[int] = []
    for row, value in canonical["label_id"].items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise MetadataAdapterError(f"label_id at row {row} must be an integer in 0..2")
        actual_ids.append(int(value))
    if actual_ids != expected_ids:
        raise MetadataAdapterError("Adapted metadata label and label_id columns disagree")
    canonical["label_id"] = pd.array(actual_ids, dtype="int64")
    return canonical.reset_index(drop=True)


def write_adapted_metadata(frame: pd.DataFrame, output_dir: str | Path) -> AdaptedMetadataFiles:
    """Write deterministic combined and per-split CSVs outside the baseline tree."""
    canonical = _validate_adapted_frame(frame)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"Metadata output path is not a directory: {directory}")

    combined_csv = directory / "metadata.csv"
    canonical.to_csv(combined_csv, index=False, encoding="utf-8", lineterminator="\n")

    split_csvs: dict[str, Path] = {}
    row_counts: dict[str, int] = {"all": len(canonical)}
    for split in _SPLIT_ORDER:
        split_frame = canonical.loc[canonical["split"] == split].reset_index(drop=True)
        split_path = directory / f"{split}_metadata.csv"
        split_frame.to_csv(split_path, index=False, encoding="utf-8", lineterminator="\n")
        split_csvs[split] = split_path
        row_counts[split] = len(split_frame)

    return AdaptedMetadataFiles(
        output_dir=directory,
        combined_csv=combined_csv,
        split_csvs=split_csvs,
        row_counts=row_counts,
    )
