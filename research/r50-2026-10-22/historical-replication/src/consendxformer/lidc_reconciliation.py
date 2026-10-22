"""Frozen patient/series reconciliation and explicit source-completeness checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

from .lidc_records import ScanRecord, ensure_existing_directory
from .manifest_guard import ManifestGuardReport, validate_frozen_manifest_bundle
from .phase2_config import SOURCE_MODES


PATIENT_ID_RE = re.compile(r"LIDC-IDRI-[0-9]{4}\Z")


class SourceReadinessError(RuntimeError):
    pass


class ReconciliationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenSeriesIdentity:
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    split: str


@dataclass(frozen=True)
class SourceReadinessReport:
    source_mode: str
    final_dataset_eligible: bool
    raw_root: str
    raw_root_nonempty: bool
    source_patient_count: int
    source_series_count: int
    annotated_series_count: int
    readable_series_count: int
    annotation_count: int
    duplicate_series: tuple[str, ...]
    malformed_patient_ids: tuple[str, ...]
    unknown_patients: tuple[str, ...]
    unknown_series: tuple[str, ...]
    mismatched_patient_series: tuple[str, ...]
    mismatched_study_series: tuple[str, ...]
    missing_frozen_patients: tuple[str, ...]
    missing_frozen_series: tuple[str, ...]
    annotation_source_available: bool
    completeness_errors: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_source_report_for_run(
    source_report: SourceReadinessReport, source_config: Mapping[str, object]
) -> None:
    """Prevent callers from bypassing strict/partial source semantics."""
    configured_mode = source_config.get("mode")
    if configured_mode != source_report.source_mode:
        raise SourceReadinessError(
            "Source report mode does not match the effective Phase 2 configuration"
        )
    configured_root = source_config.get("raw_root")
    if configured_root is None or Path(str(configured_root)).resolve() != Path(
        source_report.raw_root
    ).resolve():
        raise SourceReadinessError(
            "Source report root does not match the effective Phase 2 configuration"
        )
    if source_report.source_mode == "strict_full" and not source_report.final_dataset_eligible:
        raise SourceReadinessError(
            "A strict-full run requires a complete, final-dataset-eligible source report"
        )
    if source_report.source_mode == "partial_development" and source_report.final_dataset_eligible:
        raise SourceReadinessError(
            "A partial/development source report can never be final-dataset eligible"
        )


def frozen_series_index(report: ManifestGuardReport) -> dict[str, FrozenSeriesIdentity]:
    if not isinstance(report, ManifestGuardReport):
        raise TypeError("report must come from validate_frozen_manifest_bundle")
    # Re-read the immutable files rather than trusting mutable DataFrames held by callers.
    verified = validate_frozen_manifest_bundle(report.manifest_dir)
    return {
        row.series_instance_uid: FrozenSeriesIdentity(
            patient_id=row.patient_id,
            study_instance_uid=row.study_instance_uid,
            series_instance_uid=row.series_instance_uid,
            split=row.split,
        )
        for row in verified.series_split.itertuples(index=False)
    }


def reconcile_scan(
    scan: ScanRecord, report: ManifestGuardReport
) -> FrozenSeriesIdentity:
    return reconcile_scan_against_index(scan, frozen_series_index(report))


def reconcile_scan_against_index(
    scan: ScanRecord, index: dict[str, FrozenSeriesIdentity]
) -> FrozenSeriesIdentity:
    identity = index.get(scan.series_instance_uid)
    if identity is None:
        raise ReconciliationError(
            f"Unknown SeriesInstanceUID absent from frozen Phase 0 inventory: {scan.series_instance_uid}"
        )
    if scan.patient_id != identity.patient_id:
        raise ReconciliationError(
            f"Series/patient mismatch for {scan.series_instance_uid}: "
            f"source={scan.patient_id}, frozen={identity.patient_id}"
        )
    if scan.study_instance_uid != identity.study_instance_uid:
        raise ReconciliationError(
            f"Series/study mismatch for {scan.series_instance_uid}: "
            f"source={scan.study_instance_uid}, frozen={identity.study_instance_uid}"
        )
    return identity


def assess_source_readiness(
    raw_root: str | Path,
    scans: Iterable[ScanRecord],
    report: ManifestGuardReport,
    *,
    mode: str,
    raise_on_incomplete: bool = True,
) -> SourceReadinessReport:
    if mode not in SOURCE_MODES:
        raise ValueError(f"mode must be one of {SOURCE_MODES}")
    root = ensure_existing_directory(raw_root, "Raw LIDC source")
    try:
        raw_nonempty = next(root.iterdir(), None) is not None
    except OSError as exc:
        raise SourceReadinessError(f"Raw LIDC root is not readable: {root}") from exc
    scan_rows = tuple(scans)
    frozen = frozen_series_index(report)
    frozen_patients = set(report.patient_split["patient_id"].astype(str))
    seen_series: set[str] = set()
    duplicates: set[str] = set()
    malformed_patients: set[str] = set()
    unknown_patients: set[str] = set()
    unknown_series: set[str] = set()
    patient_mismatches: set[str] = set()
    study_mismatches: set[str] = set()
    source_patients: set[str] = set()
    annotation_total = 0
    annotated_series: set[str] = set()
    readable_series: set[str] = set()
    annotation_source_available = True

    for scan in scan_rows:
        patient = str(scan.patient_id).strip()
        series = str(scan.series_instance_uid).strip()
        study = str(scan.study_instance_uid).strip()
        source_patients.add(patient)
        if PATIENT_ID_RE.fullmatch(patient) is None:
            malformed_patients.add(patient)
        if patient not in frozen_patients:
            unknown_patients.add(patient)
        if series in seen_series:
            duplicates.add(series)
        seen_series.add(series)
        expected = frozen.get(series)
        if expected is None:
            unknown_series.add(series)
        else:
            if expected.patient_id != patient:
                patient_mismatches.add(series)
            if expected.study_instance_uid != study:
                study_mismatches.add(series)
        if scan.dicom_readable:
            readable_series.add(series)
        if scan.annotation_count > 0:
            annotated_series.add(series)
        annotation_total += int(scan.annotation_count)
        annotation_source_available = (
            annotation_source_available and bool(scan.annotation_source_available)
        )

    missing_series = set(frozen) - seen_series
    missing_patients = frozen_patients - source_patients
    errors: list[str] = []
    checks = (
        (not raw_nonempty, "raw source root is empty"),
        (bool(duplicates), f"duplicate series: {len(duplicates)}"),
        (bool(malformed_patients), f"malformed patient IDs: {len(malformed_patients)}"),
        (bool(unknown_patients), f"unknown patients: {len(unknown_patients)}"),
        (bool(unknown_series), f"unknown series: {len(unknown_series)}"),
        (bool(patient_mismatches), f"patient/series mismatches: {len(patient_mismatches)}"),
        (bool(study_mismatches), f"study/series mismatches: {len(study_mismatches)}"),
        (bool(missing_patients), f"missing frozen patients: {len(missing_patients)}"),
        (bool(missing_series), f"missing frozen series: {len(missing_series)}"),
        (len(readable_series) != len(seen_series), "one or more DICOM series are unreadable"),
        (not annotation_source_available, "annotation source is unavailable"),
    )
    errors.extend(message for failed, message in checks if failed)
    result = SourceReadinessReport(
        source_mode=mode,
        final_dataset_eligible=mode == "strict_full" and not errors,
        raw_root=str(root),
        raw_root_nonempty=raw_nonempty,
        source_patient_count=len(source_patients),
        source_series_count=len(seen_series),
        annotated_series_count=len(annotated_series),
        readable_series_count=len(readable_series),
        annotation_count=annotation_total,
        duplicate_series=tuple(sorted(duplicates)),
        malformed_patient_ids=tuple(sorted(malformed_patients)),
        unknown_patients=tuple(sorted(unknown_patients)),
        unknown_series=tuple(sorted(unknown_series)),
        mismatched_patient_series=tuple(sorted(patient_mismatches)),
        mismatched_study_series=tuple(sorted(study_mismatches)),
        missing_frozen_patients=tuple(sorted(missing_patients)),
        missing_frozen_series=tuple(sorted(missing_series)),
        annotation_source_available=annotation_source_available,
        completeness_errors=tuple(errors),
    )
    if mode == "strict_full" and errors and raise_on_incomplete:
        raise SourceReadinessError(
            "STRICT FULL SOURCE validation failed: " + "; ".join(errors)
        )
    return result


def leakage_check(rows: Iterable[dict], report: ManifestGuardReport) -> dict:
    verified = validate_frozen_manifest_bundle(report.manifest_dir)
    patient_splits = dict(
        zip(verified.patient_split["patient_id"], verified.patient_split["split"])
    )
    series_splits = dict(
        zip(verified.series_split["series_instance_uid"], verified.series_split["split"])
    )
    seen: dict[str, set[str]] = {}
    violations: list[str] = []
    count = 0
    for row in rows:
        count += 1
        patient = str(row.get("patient_id", ""))
        series = str(row.get("series_instance_uid", ""))
        split = str(row.get("split", ""))
        expected_patient = patient_splits.get(patient)
        expected_series = series_splits.get(series)
        if expected_patient is None or expected_series is None:
            violations.append(f"unknown identity: patient={patient}, series={series}")
            continue
        if split != expected_patient or split != expected_series:
            violations.append(
                f"split mismatch: patient={patient}, series={series}, row={split}, "
                f"expected={expected_patient}"
            )
        seen.setdefault(patient, set()).add(split)
    cross_split = sorted(patient for patient, splits in seen.items() if len(splits) != 1)
    if cross_split:
        violations.append(f"patients mapped to multiple splits: {cross_split[:10]}")
    return {
        "schema_version": 1,
        "nodule_rows_checked": count,
        "passed": not violations,
        "violations": violations,
        "split_source": "frozen_phase0_patient_assignment_only",
    }
