"""Fail-closed validation of Phase 3 dataset inputs.

Real experiments may consume only a completed, checksummed Phase 2 canonical
extraction and its linked rendered view.  Synthetic data is deliberately a
separate dry-run mode and can never satisfy the real-data readiness claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .baseline_metadata import CANONICAL_COLUMNS, LABEL_TO_ID, adapt_metadata
from .baseline_metadata_export import COHORT_VIEWS, export_baseline_metadata
from .hashing import sha256_file, sha256_json
from .lidc_ledger import LEDGER_COLUMNS
from .lidc_masks import load_mask_artifact
from .lidc_reconciliation import frozen_series_index
from .lidc_render_pipeline import RENDERED_COLUMNS
from .manifest_guard import (
    FROZEN_BUNDLE_SHA256,
    FROZEN_PATIENT_COUNT,
    FROZEN_SERIES_COUNT,
    FROZEN_SPLIT_COUNTS,
    ManifestGuardReport,
    validate_frozen_manifest_bundle,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DATA_KEYS = {
    "mode",
    "metadata_path",
    "frozen_split_dir",
    "source_readiness_report",
    "canonical_dir",
    "rendered_dir",
    "cohort_view",
    "synthetic_marker",
}
_BASELINE_EXPORT_COLUMNS = (
    "patient_id",
    "image_path",
    "label",
    "nodule_id",
    "series_instance_uid",
    "split",
    "resolution",
    "n_valid_malignancy_ratings",
    "cohort_view",
)
_SYNTHETIC_COLUMNS = (*_BASELINE_EXPORT_COLUMNS, "data_origin")
_SYNTHETIC_MARKER = "SYNTHETIC_NON_RESEARCH_ONLY"
_SYNTHETIC_COHORT_VIEW = "synthetic_tiny_fixture"
PHASE3_CANONICAL_COLUMNS = (
    *CANONICAL_COLUMNS,
    "nodule_id",
    "series_instance_uid",
)
_LOGICAL_SAMPLE_FIELDS = (
    "patient_id",
    "series_instance_uid",
    "nodule_id",
    "split",
    "label",
)
_SOURCE_CORE_KEYS = (
    "source_mode",
    "final_dataset_eligible",
    "raw_root",
    "raw_root_nonempty",
    "source_patient_count",
    "source_series_count",
    "annotated_series_count",
    "readable_series_count",
    "annotation_count",
    "duplicate_series",
    "malformed_patient_ids",
    "unknown_patients",
    "unknown_series",
    "mismatched_patient_series",
    "mismatched_study_series",
    "missing_frozen_patients",
    "missing_frozen_series",
    "annotation_source_available",
    "completeness_errors",
)
_EMPTY_SOURCE_LISTS = (
    "duplicate_series",
    "malformed_patient_ids",
    "unknown_patients",
    "unknown_series",
    "mismatched_patient_series",
    "mismatched_study_series",
    "missing_frozen_patients",
    "missing_frozen_series",
    "completeness_errors",
)
_VALID_SOURCE_STATUSES = {
    "valid",
    "valid_with_missing_ratings",
    "excluded_identity_reconciliation",
    "excluded_malformed_geometry",
    "excluded_invalid_rating",
    "excluded_no_valid_rating",
    "excluded_mask_unavailable",
}
_RENDERABLE_SOURCE_STATUSES = {"valid", "valid_with_missing_ratings"}


class Phase3ReadinessError(RuntimeError):
    """Raised whenever a Phase 3 input cannot be proven safe to consume."""


@dataclass(frozen=True)
class Phase3InputFingerprint:
    """Stable hashes for exact bytes, logical samples, and Phase 2 parents."""

    metadata_sha256: str
    logical_cohort_sha256: str
    ordered_samples_sha256: str
    source_readiness_sha256: str | None
    canonical_bundle_sha256: str | None
    canonical_checksum_ledger_sha256: str | None
    rendered_bundle_sha256: str | None
    rendered_checksum_ledger_sha256: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "metadata_sha256": self.metadata_sha256,
            "logical_cohort_sha256": self.logical_cohort_sha256,
            "ordered_samples_sha256": self.ordered_samples_sha256,
            "source_readiness_sha256": self.source_readiness_sha256,
            "canonical_bundle_sha256": self.canonical_bundle_sha256,
            "canonical_checksum_ledger_sha256": self.canonical_checksum_ledger_sha256,
            "rendered_bundle_sha256": self.rendered_bundle_sha256,
            "rendered_checksum_ledger_sha256": self.rendered_checksum_ledger_sha256,
        }


@dataclass(frozen=True)
class Phase3ReadinessReport:
    """Immutable evidence describing what the readiness gate actually proved."""

    schema_version: int
    mode: str
    ready_for_requested_mode: bool
    real_research_ready: bool
    software_annotation_runtime_ready: bool
    dicom_source_ready: bool
    extracted_cohort_ready: bool
    source_readiness_validated: bool
    phase2_artifacts_validated: bool
    cohort_projection_validated: bool
    frozen_split_validated: bool
    synthetic_non_research: bool
    cohort_view: str
    expected_resolution: int
    metadata_path: str
    sample_count: int
    split_counts: tuple[tuple[str, int], ...]
    manifest_bundle_sha256: str
    fingerprint: Phase3InputFingerprint
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "ready_for_requested_mode": self.ready_for_requested_mode,
            "real_research_ready": self.real_research_ready,
            "software_annotation_runtime_ready": self.software_annotation_runtime_ready,
            "dicom_source_ready": self.dicom_source_ready,
            "extracted_cohort_ready": self.extracted_cohort_ready,
            "source_readiness_validated": self.source_readiness_validated,
            "phase2_artifacts_validated": self.phase2_artifacts_validated,
            "cohort_projection_validated": self.cohort_projection_validated,
            "frozen_split_validated": self.frozen_split_validated,
            "synthetic_non_research": self.synthetic_non_research,
            "cohort_view": self.cohort_view,
            "expected_resolution": self.expected_resolution,
            "metadata_path": self.metadata_path,
            "sample_count": self.sample_count,
            "split_counts": dict(self.split_counts),
            "manifest_bundle_sha256": self.manifest_bundle_sha256,
            "fingerprint": self.fingerprint.to_dict(),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class _CompletedPhase2Artifact:
    directory: Path
    checksums: Mapping[str, str]
    bundle_sha256: str
    checksum_ledger_sha256: str
    configuration_sha256: str
    provenance: Mapping[str, Any]


def _fail(message: str) -> None:
    raise Phase3ReadinessError(message)


def _project_path(value: Any, project_root: Path, role: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        _fail(f"{role} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{role} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase3ReadinessError(f"Cannot parse {role}: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"{role} must contain a JSON object")
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{role} must be a lowercase SHA-256 digest")
    return value


def _safe_member(root: Path, value: Any) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        _fail("Phase 2 checksum member names must be non-empty strings")
    member = PurePosixPath(value)
    if (
        member.is_absolute()
        or value != member.as_posix()
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        _fail(f"Unsafe or non-canonical Phase 2 checksum member: {value!r}")
    target = root.joinpath(*member.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise Phase3ReadinessError(
            f"Phase 2 checksum member escapes its artifact root: {value!r}"
        ) from exc
    return value, target


def _validate_phase0_provenance(
    provenance: Mapping[str, Any], manifest: ManifestGuardReport, role: str
) -> None:
    phase0 = provenance.get("phase0_manifest")
    if not isinstance(phase0, dict):
        _fail(f"{role} provenance lacks the Phase 0 manifest record")
    expected = manifest.to_dict()
    for key in (
        "bundle_sha256",
        "checksum_ledger_sha256",
        "file_sha256",
        "unique_patients",
        "unique_ct_series",
        "split_counts",
    ):
        if phase0.get(key) != expected.get(key):
            _fail(f"{role} provenance Phase 0 field {key!r} is not frozen")


def _validate_completed_phase2(
    directory: Path,
    *,
    role: str,
    expected_run_type: str,
    manifest: ManifestGuardReport,
) -> _CompletedPhase2Artifact:
    if not directory.is_dir():
        _fail(f"{role} Phase 2 directory does not exist: {directory}")
    completed = _read_json(directory / "COMPLETED.json", f"{role} COMPLETED marker")
    if set(completed) != {
        "schema_version",
        "checksum_ledger",
        "checksum_ledger_sha256",
    } or completed.get("schema_version") != 1:
        _fail(f"{role} COMPLETED marker has an unexpected schema")
    if completed.get("checksum_ledger") != "checksums.json":
        _fail(f"{role} COMPLETED marker does not select checksums.json")
    checksums_path = directory / "checksums.json"
    ledger_sha = _require_sha256(
        completed.get("checksum_ledger_sha256"),
        f"{role} completed checksum-ledger digest",
    )
    if not checksums_path.is_file() or sha256_file(checksums_path) != ledger_sha:
        _fail(f"{role} checksum ledger does not match its COMPLETED marker")
    ledger = _read_json(checksums_path, f"{role} checksum ledger")
    if set(ledger) != {"schema_version", "files", "bundle_sha256", "hash_scope"}:
        _fail(f"{role} checksum ledger has an unexpected schema")
    if ledger.get("schema_version") != 1:
        _fail(f"{role} checksum ledger schema version is unsupported")
    if ledger.get("hash_scope") != "listed generated files; raw DICOM bytes are not hashed":
        _fail(f"{role} checksum ledger has an unexpected hash scope")
    files = ledger.get("files")
    if not isinstance(files, dict) or not files:
        _fail(f"{role} checksum ledger has no file inventory")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in files.items():
        name, target = _safe_member(directory, raw_name)
        digest = _require_sha256(raw_digest, f"{role} checksum for {name}")
        if not target.is_file():
            _fail(f"{role} checksummed member is missing: {name}")
        if sha256_file(target) != digest:
            _fail(f"{role} checksummed member was modified: {name}")
        normalized[name] = digest
    bundle = _require_sha256(ledger.get("bundle_sha256"), f"{role} bundle digest")
    if sha256_json(normalized) != bundle:
        _fail(f"{role} checksum bundle digest is internally inconsistent")
    for required in ("run_configuration.json", "provenance.json"):
        if required not in normalized:
            _fail(f"{role} checksum ledger omits required member {required}")
    run_configuration = _read_json(
        directory / "run_configuration.json", f"{role} run configuration"
    )
    if set(run_configuration) != {"configuration_sha256"}:
        _fail(f"{role} run configuration has an unexpected schema")
    configuration_sha = _require_sha256(
        run_configuration.get("configuration_sha256"), f"{role} configuration digest"
    )
    provenance = _read_json(directory / "provenance.json", f"{role} provenance")
    if provenance.get("schema_version") != 1:
        _fail(f"{role} provenance schema version is unsupported")
    if provenance.get("run_type") != expected_run_type:
        _fail(f"{role} provenance run type is not {expected_run_type!r}")
    if provenance.get("configuration_sha256") != configuration_sha:
        _fail(f"{role} provenance/configuration digest linkage failed")
    if provenance.get("performance_claimed") is not False:
        _fail(f"{role} Phase 2 provenance makes an unexpected performance claim")
    try:
        recorded_output = Path(str(provenance.get("output_path", ""))).resolve()
    except (OSError, ValueError) as exc:
        raise Phase3ReadinessError(f"{role} provenance output path is malformed") from exc
    if recorded_output != directory:
        _fail(f"{role} provenance output path does not identify the supplied directory")
    _validate_phase0_provenance(provenance, manifest, role)
    return _CompletedPhase2Artifact(
        directory=directory,
        checksums=normalized,
        bundle_sha256=bundle,
        checksum_ledger_sha256=ledger_sha,
        configuration_sha256=configuration_sha,
        provenance=provenance,
    )


def _validate_official_provider(value: Any, role: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{role} lacks provider version metadata")
    if value.get("provider") != "official_pylidc" or value.get("pylidc") != "0.2.3":
        _fail(f"{role} did not use exact official pylidc==0.2.3")
    numpy_version = value.get("numpy")
    if not isinstance(numpy_version, str) or not numpy_version:
        _fail(f"{role} lacks the NumPy version")
    compatibility = value.get("pylidc_numpy_compatibility")
    if not isinstance(compatibility, dict):
        _fail(f"{role} lacks pylidc NumPy-compatibility metadata")
    aliases = compatibility.get("aliases")
    required = compatibility.get("required")
    if (
        type(required) is not bool
        or not isinstance(aliases, list)
        or any(alias not in {"int", "bool"} for alias in aliases)
        or len(aliases) != len(set(aliases))
        or required != bool(aliases)
    ):
        _fail(f"{role} has invalid pylidc NumPy-compatibility alias metadata")
    if compatibility.get("algorithm_modified") is not False:
        _fail(f"{role} reports a modified pylidc algorithm")
    if compatibility.get("pylidc_source_modified") is not False:
        _fail(f"{role} reports modified pylidc source")
    if compatibility.get("numpy_module_modified") is not False:
        _fail(f"{role} reports global NumPy-module modification")


def _require_int(value: Any, expected: int, role: str) -> None:
    if type(value) is not int or value != expected:
        _fail(f"{role} must be exactly {expected}")


def _validate_source_readiness(
    report: Mapping[str, Any], manifest: ManifestGuardReport
) -> None:
    missing = [key for key in _SOURCE_CORE_KEYS if key not in report]
    if missing:
        _fail(f"Source readiness report lacks required fields: {missing}")
    if report.get("source_mode") != "strict_full":
        _fail("Real Phase 3 requires a strict_full source readiness report")
    if report.get("configuration_mode") != "strict_full":
        _fail("Source readiness configuration was not strict_full")
    if report.get("final_dataset_eligible") is not True:
        _fail("Source readiness report is not final-dataset eligible")
    if report.get("raw_root_nonempty") is not True:
        _fail("Source readiness report describes an empty raw source")
    if report.get("annotation_source_available") is not True:
        _fail("Source readiness report lacks annotation-source availability")
    _require_int(report.get("source_patient_count"), FROZEN_PATIENT_COUNT, "Source patient count")
    _require_int(report.get("source_series_count"), FROZEN_SERIES_COUNT, "Source series count")
    _require_int(report.get("readable_series_count"), FROZEN_SERIES_COUNT, "Readable series count")
    for key in ("annotated_series_count", "annotation_count"):
        value = report.get(key)
        if type(value) is not int or value <= 0:
            _fail(f"Source readiness field {key} must be a positive integer")
    for key in _EMPTY_SOURCE_LISTS:
        if report.get(key) != []:
            _fail(f"Source readiness field {key} must be empty for a real run")
    if report.get("manifest_bundle_sha256") != manifest.bundle_sha256:
        _fail("Source readiness report is not linked to the frozen Phase 0 bundle")
    _validate_official_provider(report.get("provider_versions"), "Source readiness report")


def _validate_provenance_source(
    provenance: Mapping[str, Any], source_report: Mapping[str, Any], role: str
) -> None:
    nested = provenance.get("source_validation")
    if not isinstance(nested, dict):
        _fail(f"{role} provenance lacks source validation")
    for key in _SOURCE_CORE_KEYS:
        if nested.get(key) != source_report.get(key):
            _fail(f"{role} provenance source field {key!r} differs from readiness")
    if provenance.get("source_root") != source_report.get("raw_root"):
        _fail(f"{role} provenance source root differs from readiness")
    versions = provenance.get("versions")
    if not isinstance(versions, dict):
        _fail(f"{role} provenance lacks runtime versions")
    _validate_official_provider(versions.get("provider"), f"{role} provenance")


def _read_csv_exact(path: Path, columns: tuple[str, ...], role: str) -> pd.DataFrame:
    if not path.is_file():
        _fail(f"{role} does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise Phase3ReadinessError(f"Cannot parse {role}: {path}") from exc
    if tuple(frame.columns) != columns or not frame.columns.is_unique:
        _fail(
            f"{role} columns differ from the required schema: "
            f"expected={list(columns)}, actual={list(frame.columns)}"
        )
    if frame.empty:
        _fail(f"{role} contains no samples")
    return frame


def _as_bool(value: Any, role: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    _fail(f"{role} is not a canonical boolean")
    raise AssertionError("unreachable")


def _validate_identity_rows(
    frame: pd.DataFrame, manifest: ManifestGuardReport, role: str
) -> None:
    frozen = frozen_series_index(manifest)
    for row in frame.itertuples(index=False):
        series = str(row.series_instance_uid)
        identity = frozen.get(series)
        if identity is None:
            _fail(f"{role} contains unknown frozen series {series!r}")
        if (
            str(row.patient_id) != identity.patient_id
            or str(row.split) != identity.split
            or (
                hasattr(row, "study_instance_uid")
                and str(row.study_instance_uid) != identity.study_instance_uid
            )
        ):
            _fail(f"{role} contains a frozen identity/split mismatch for {series}")


def _validate_canonical_ledger(
    artifact: _CompletedPhase2Artifact, manifest: ManifestGuardReport
) -> pd.DataFrame:
    name = "canonical_nodule_ledger.csv"
    if name not in artifact.checksums:
        _fail("Canonical checksum ledger omits canonical_nodule_ledger.csv")
    frame = _read_csv_exact(artifact.directory / name, LEDGER_COLUMNS, "Canonical nodule ledger")
    if frame["nodule_id"].duplicated().any() or frame["nodule_sha256"].duplicated().any():
        _fail("Canonical nodule ledger contains duplicate stable identities")
    _validate_identity_rows(frame, manifest, "Canonical nodule ledger")
    for row in frame.itertuples(index=False):
        digest = _require_sha256(row.nodule_sha256, "Canonical nodule SHA-256")
        expected_id = f"{row.patient_id}__nodule-{digest[:16]}"
        if row.nodule_id != expected_id:
            _fail(f"Canonical nodule ID is inconsistent with its digest: {row.nodule_id}")
        status = str(row.source_validity_status)
        if status not in _VALID_SOURCE_STATUSES:
            _fail(f"Canonical nodule has an unknown source-validity status: {status!r}")
        eligible = _as_bool(row.paper_primary_eligible, "paper_primary_eligible")
        boundary = _as_bool(row.boundary_zone, "boundary_zone")
        if eligible and (
            status not in _RENDERABLE_SOURCE_STATUSES
            or row.paper_label not in LABEL_TO_ID
            or boundary
        ):
            _fail(f"Canonical paper eligibility is internally inconsistent: {row.nodule_id}")
        if row.paper_label == "boundary_zone" and (eligible or not boundary):
            _fail(f"Canonical boundary label is internally inconsistent: {row.nodule_id}")
        if row.mask_path:
            member, path = _safe_member(artifact.directory, row.mask_path)
            if artifact.checksums.get(member) != row.mask_sha256:
                _fail(f"Canonical mask checksum linkage failed: {row.nodule_id}")
            mask = load_mask_artifact(path)
            try:
                signatures = json.loads(row.annotation_sha256s_json)
            except json.JSONDecodeError as exc:
                raise Phase3ReadinessError(
                    f"Canonical annotation signatures are malformed: {row.nodule_id}"
                ) from exc
            if signatures != list(mask.annotation_sha256s):
                _fail(f"Canonical annotation/mask signature linkage failed: {row.nodule_id}")
        elif status in _RENDERABLE_SOURCE_STATUSES:
            _fail(f"Renderable canonical nodule has no mask: {row.nodule_id}")
    leakage_name = "leakage_check.json"
    if leakage_name not in artifact.checksums:
        _fail("Canonical checksum ledger omits leakage_check.json")
    leakage = _read_json(artifact.directory / leakage_name, "Canonical leakage report")
    if (
        leakage.get("schema_version") != 1
        or leakage.get("passed") is not True
        or leakage.get("violations") != []
        or leakage.get("nodule_rows_checked") != len(frame)
        or leakage.get("split_source") != "frozen_phase0_patient_assignment_only"
    ):
        _fail("Canonical Phase 2 leakage report did not pass exactly")
    return frame


def _validate_rendered_arrays(
    artifact: _CompletedPhase2Artifact, row: Any, resolution: int
) -> None:
    image_member, image_path = _safe_member(artifact.directory, row.image_path)
    mask_member, mask_path = _safe_member(artifact.directory, row.rendered_mask_path)
    if image_member not in artifact.checksums or mask_member not in artifact.checksums:
        _fail(f"Rendered ROI paths are absent from checksums: {row.nodule_id}")
    try:
        image = np.load(image_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise Phase3ReadinessError(f"Cannot load rendered image {row.nodule_id}") from exc
    if (
        image.shape != (resolution, resolution)
        or image.dtype != np.float32
        or not bool(np.isfinite(image).all())
        or bool((image < 0).any())
        or bool((image > 1).any())
    ):
        _fail(f"Rendered image array is invalid: {row.nodule_id}")
    try:
        with np.load(mask_path, allow_pickle=False) as archive:
            expected = {
                "binary_consensus_50",
                "crop_box_ascii",
                "individual_masks",
                "soft_consensus_mask",
            }
            if set(archive.files) != expected:
                _fail(f"Rendered mask members are invalid: {row.nodule_id}")
            individual = archive["individual_masks"].copy()
            soft = archive["soft_consensus_mask"].copy()
            binary = archive["binary_consensus_50"].copy()
    except (OSError, ValueError, KeyError) as exc:
        raise Phase3ReadinessError(f"Cannot load rendered masks {row.nodule_id}") from exc
    if (
        individual.ndim != 3
        or individual.shape[1:] != (resolution, resolution)
        or individual.dtype != np.uint8
        or soft.shape != (resolution, resolution)
        or soft.dtype != np.float32
        or binary.shape != soft.shape
        or binary.dtype != np.uint8
        or bool(((individual != 0) & (individual != 1)).any())
        or not bool(np.isfinite(soft).all())
        or bool((soft < 0).any())
        or bool((soft > 1).any())
        or not np.allclose(soft, individual.mean(axis=0), rtol=0.0, atol=1e-6)
        or not np.array_equal(binary, (soft >= 0.5).astype(np.uint8))
    ):
        _fail(f"Rendered mask arrays are invalid or misaligned: {row.nodule_id}")


def _validate_rendered_ledger(
    artifact: _CompletedPhase2Artifact,
    canonical: pd.DataFrame,
    manifest: ManifestGuardReport,
    resolution: int,
) -> pd.DataFrame:
    name = "rendered_rois.csv"
    if name not in artifact.checksums:
        _fail("Rendered checksum ledger omits rendered_rois.csv")
    frame = _read_csv_exact(artifact.directory / name, RENDERED_COLUMNS, "Rendered ROI ledger")
    if frame["nodule_id"].duplicated().any() or frame["nodule_sha256"].duplicated().any():
        _fail("Rendered ROI ledger contains duplicate stable identities")
    if set(frame["resolution"]) != {str(resolution)}:
        _fail(f"Rendered ROI ledger is not uniformly resolution {resolution}")
    _validate_identity_rows(frame, manifest, "Rendered ROI ledger")
    renderable = canonical.loc[
        canonical["source_validity_status"].isin(_RENDERABLE_SOURCE_STATUSES)
    ].copy()
    if frame["nodule_id"].tolist() != renderable["nodule_id"].tolist():
        _fail(
            "Rendered ROI ledger is not the ordered complete projection of valid "
            "canonical nodules"
        )
    canonical_by_id = renderable.set_index("nodule_id", drop=False)
    for row in frame.itertuples(index=False):
        source = canonical_by_id.loc[row.nodule_id]
        comparisons = {
            "nodule_sha256": row.nodule_sha256,
            "patient_id": row.patient_id,
            "study_instance_uid": row.study_instance_uid,
            "series_instance_uid": row.series_instance_uid,
            "split": row.split,
            "paper_label": row.paper_label,
            "n_valid_malignancy_ratings": row.n_valid_malignancy_ratings,
        }
        for field, rendered_value in comparisons.items():
            if str(source[field]) != str(rendered_value):
                _fail(f"Canonical/rendered field {field!r} differs for {row.nodule_id}")
        for field in ("paper_primary_eligible", "boundary_zone", "has_four_valid_readers"):
            if _as_bool(source[field], field) != _as_bool(getattr(row, field), field):
                _fail(f"Canonical/rendered boolean {field!r} differs for {row.nodule_id}")
        _validate_rendered_arrays(artifact, row, resolution)
    return frame


def _string_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.map(lambda value: str(value))


def _validate_exact_baseline_projection(
    metadata_path: Path,
    rendered: pd.DataFrame,
    rendered_root: Path,
    manifest: ManifestGuardReport,
    cohort_view: str,
    resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supplied = _read_csv_exact(metadata_path, _BASELINE_EXPORT_COLUMNS, "Phase 3 metadata")
    if "data_origin" in supplied.columns:
        _fail("Synthetic metadata is forbidden in real mode")
    expected = export_baseline_metadata(
        rendered,
        manifest,
        cohort_view=cohort_view,
        rendered_root=rendered_root,
    )
    if tuple(expected.columns) != _BASELINE_EXPORT_COLUMNS:
        _fail("Phase 2 baseline exporter returned an unexpected schema")
    if not _string_frame(supplied).equals(_string_frame(expected)):
        _fail("Metadata is not the exact selected Phase 2 baseline-export projection")
    if set(supplied["resolution"]) != {str(resolution)}:
        _fail("Metadata resolution differs from the configured Phase 3 resolution")
    if set(supplied["cohort_view"]) != {cohort_view}:
        _fail("Metadata cohort view differs from the configured Phase 3 cohort")
    if supplied["nodule_id"].duplicated().any():
        _fail("Metadata contains duplicate nodules")
    canonical = adapt_metadata(supplied, manifest)
    canonical["nodule_id"] = supplied["nodule_id"].astype(str).to_numpy()
    canonical["series_instance_uid"] = (
        supplied["series_instance_uid"].astype(str).to_numpy()
    )
    canonical = canonical.loc[:, PHASE3_CANONICAL_COLUMNS]
    return supplied, canonical


def _validate_synthetic_metadata(
    metadata_path: Path,
    manifest: ManifestGuardReport,
    cohort_view: str,
    resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _read_csv_exact(metadata_path, _SYNTHETIC_COLUMNS, "Synthetic Phase 3 metadata")
    if set(frame["data_origin"]) != {"synthetic_non_research"}:
        _fail("Every synthetic metadata row must explicitly be synthetic_non_research")
    if set(frame["cohort_view"]) != {cohort_view}:
        _fail("Synthetic metadata cohort view differs from configuration")
    if set(frame["resolution"]) != {str(resolution)}:
        _fail("Synthetic metadata resolution differs from configuration")
    if frame["nodule_id"].duplicated().any():
        _fail("Synthetic metadata contains duplicate nodules")
    if not frame["label"].isin(LABEL_TO_ID).all():
        _fail("Synthetic metadata labels must use the three canonical classes")
    if not frame["split"].isin(FROZEN_SPLIT_COUNTS).all():
        _fail("Synthetic metadata split names are invalid")
    verified = validate_frozen_manifest_bundle(manifest.manifest_dir)
    patient_split = dict(
        zip(verified.patient_split["patient_id"], verified.patient_split["split"])
    )
    series = frozen_series_index(verified)
    for row in frame.itertuples(index=False):
        known_series = series.get(row.series_instance_uid)
        known_patient_split = patient_split.get(row.patient_id)
        if known_series is not None and (
            known_series.patient_id != row.patient_id or known_series.split != row.split
        ):
            _fail(f"Synthetic row misuses a frozen series identity: {row.nodule_id}")
        if known_patient_split is not None and known_patient_split != row.split:
            _fail(f"Synthetic row misuses a frozen patient split: {row.nodule_id}")
    canonical = pd.DataFrame(
        {
            "patient_id": frame["patient_id"].astype(str),
            "image_path": frame["image_path"].astype(str),
            "label": frame["label"].astype(str),
            "label_id": pd.array([LABEL_TO_ID[value] for value in frame["label"]], dtype="int64"),
            "split": frame["split"].astype(str),
            "nodule_id": frame["nodule_id"].astype(str),
            "series_instance_uid": frame["series_instance_uid"].astype(str),
        },
        columns=PHASE3_CANONICAL_COLUMNS,
    )
    return frame, canonical


def _sample_hashes(frame: pd.DataFrame) -> tuple[str, str]:
    missing = [field for field in _LOGICAL_SAMPLE_FIELDS if field not in frame.columns]
    if missing:
        _fail(f"Metadata lacks logical fingerprint fields: {missing}")
    ordered: list[dict[str, str]] = []
    for row in frame.loc[:, _LOGICAL_SAMPLE_FIELDS].itertuples(index=False, name=None):
        values = tuple(str(value).strip() for value in row)
        if any(not value for value in values):
            _fail("Metadata logical fingerprint fields must be non-empty")
        ordered.append(dict(zip(_LOGICAL_SAMPLE_FIELDS, values)))
    logical = sorted(
        ordered,
        key=lambda record: tuple(record[field] for field in _LOGICAL_SAMPLE_FIELDS),
    )
    return sha256_json(logical), sha256_json(ordered)


def _split_counts(frame: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    return tuple(
        (split, int((frame["split"] == split).sum()))
        for split in ("train", "val", "test")
    )


def _verified_manifest(manifest: ManifestGuardReport) -> ManifestGuardReport:
    if not isinstance(manifest, ManifestGuardReport):
        _fail("manifest must be returned by validate_frozen_manifest_bundle")
    verified = validate_frozen_manifest_bundle(manifest.manifest_dir)
    if (
        verified.bundle_sha256 != FROZEN_BUNDLE_SHA256
        or manifest.bundle_sha256 != verified.bundle_sha256
    ):
        _fail("Phase 3 manifest is not the frozen Phase 0 bundle")
    return verified


def validate_phase3_readiness(
    data_config: Mapping[str, Any],
    *,
    manifest: ManifestGuardReport,
    expected_resolution: int,
    project_root: str | Path,
    synthetic_dry_run: bool,
) -> tuple[Phase3ReadinessReport, pd.DataFrame]:
    """Validate and fingerprint Phase 3 metadata and all real-data parents.

    The returned metadata extends the Phase 1 canonical contract with the
    stable nodule and series identifiers required by prediction exports. In
    synthetic mode it is suitable only for a non-research infrastructure dry
    run; every real-readiness claim remains false.
    """

    if not isinstance(data_config, Mapping):
        _fail("data_config must be a mapping")
    missing_keys = sorted(_DATA_KEYS - set(data_config))
    if missing_keys:
        _fail(f"data_config lacks required keys: {missing_keys}")
    if type(expected_resolution) is not int or expected_resolution not in {64, 96}:
        _fail("expected_resolution must be exactly 64 or 96")
    if type(synthetic_dry_run) is not bool:
        _fail("synthetic_dry_run must be boolean")
    root = Path(project_root).resolve()
    if not root.is_dir():
        _fail(f"project_root does not exist: {root}")
    verified = _verified_manifest(manifest)
    configured_manifest = _project_path(
        data_config.get("frozen_split_dir"), root, "frozen_split_dir"
    )
    if configured_manifest != verified.manifest_dir:
        _fail(
            "data.frozen_split_dir does not identify the verified immutable "
            "Phase 0 manifest bundle"
        )
    mode = data_config.get("mode")
    if mode not in {"real", "synthetic"}:
        _fail("data.mode must be exactly 'real' or 'synthetic'")
    cohort_view = data_config.get("cohort_view")
    if mode == "real" and cohort_view not in COHORT_VIEWS:
        _fail(f"Real data.cohort_view must be one of {COHORT_VIEWS}")
    if mode == "synthetic" and cohort_view != _SYNTHETIC_COHORT_VIEW:
        _fail(
            "Synthetic data.cohort_view must be exactly "
            f"{_SYNTHETIC_COHORT_VIEW!r}"
        )
    metadata_path = _project_path(data_config.get("metadata_path"), root, "metadata_path")

    if mode == "synthetic":
        if not synthetic_dry_run:
            _fail("Synthetic data is permitted only with explicit synthetic_dry_run=True")
        if data_config.get("synthetic_marker") != _SYNTHETIC_MARKER:
            _fail(f"Synthetic mode requires synthetic_marker={_SYNTHETIC_MARKER!r}")
        for key in ("source_readiness_report", "canonical_dir", "rendered_dir"):
            if data_config.get(key) not in {None, ""}:
                _fail(f"Synthetic mode must not claim a real Phase 2 {key}")
        fingerprint_frame, canonical = _validate_synthetic_metadata(
            metadata_path,
            verified,
            str(cohort_view),
            expected_resolution,
        )
        logical_sha, ordered_sha = _sample_hashes(fingerprint_frame)
        fingerprint = Phase3InputFingerprint(
            metadata_sha256=sha256_file(metadata_path),
            logical_cohort_sha256=logical_sha,
            ordered_samples_sha256=ordered_sha,
            source_readiness_sha256=None,
            canonical_bundle_sha256=None,
            canonical_checksum_ledger_sha256=None,
            rendered_bundle_sha256=None,
            rendered_checksum_ledger_sha256=None,
        )
        report = Phase3ReadinessReport(
            schema_version=1,
            mode="synthetic",
            ready_for_requested_mode=True,
            real_research_ready=False,
            software_annotation_runtime_ready=False,
            dicom_source_ready=False,
            extracted_cohort_ready=False,
            source_readiness_validated=False,
            phase2_artifacts_validated=False,
            cohort_projection_validated=False,
            frozen_split_validated=True,
            synthetic_non_research=True,
            cohort_view=str(cohort_view),
            expected_resolution=expected_resolution,
            metadata_path=str(metadata_path),
            sample_count=len(canonical),
            split_counts=_split_counts(canonical),
            manifest_bundle_sha256=verified.bundle_sha256,
            fingerprint=fingerprint,
            limitations=(
                "Synthetic dry-run data provides no real source, software, cohort, "
                "or performance evidence.",
            ),
        )
        return report, canonical

    if synthetic_dry_run:
        _fail("Real mode cannot be combined with synthetic_dry_run=True")
    if data_config.get("synthetic_marker") not in {None, ""}:
        _fail("Real mode forbids a synthetic marker")
    source_path = _project_path(
        data_config.get("source_readiness_report"), root, "source_readiness_report"
    )
    canonical_dir = _project_path(data_config.get("canonical_dir"), root, "canonical_dir")
    rendered_dir = _project_path(data_config.get("rendered_dir"), root, "rendered_dir")
    source_report = _read_json(source_path, "Phase 2 source readiness report")
    _validate_source_readiness(source_report, verified)
    canonical_artifact = _validate_completed_phase2(
        canonical_dir,
        role="Canonical",
        expected_run_type="canonical_lidc_extraction",
        manifest=verified,
    )
    rendered_artifact = _validate_completed_phase2(
        rendered_dir,
        role="Rendered",
        expected_run_type="model_ready_2d_rendering",
        manifest=verified,
    )
    _validate_provenance_source(canonical_artifact.provenance, source_report, "Canonical")
    _validate_provenance_source(rendered_artifact.provenance, source_report, "Rendered")
    if rendered_artifact.provenance.get("output_resolution") != expected_resolution:
        _fail("Rendered provenance resolution differs from Phase 3 configuration")
    for field in ("clustering", "consensus", "representative_slice", "crop"):
        if canonical_artifact.provenance.get(field) != rendered_artifact.provenance.get(field):
            _fail(f"Canonical/rendered provenance differs for {field}")
    canonical_ledger = _validate_canonical_ledger(canonical_artifact, verified)
    rendered_ledger = _validate_rendered_ledger(
        rendered_artifact,
        canonical_ledger,
        verified,
        expected_resolution,
    )
    fingerprint_frame, canonical_metadata = _validate_exact_baseline_projection(
        metadata_path,
        rendered_ledger,
        rendered_dir,
        verified,
        str(cohort_view),
        expected_resolution,
    )
    logical_sha, ordered_sha = _sample_hashes(fingerprint_frame)
    fingerprint = Phase3InputFingerprint(
        metadata_sha256=sha256_file(metadata_path),
        logical_cohort_sha256=logical_sha,
        ordered_samples_sha256=ordered_sha,
        source_readiness_sha256=sha256_file(source_path),
        canonical_bundle_sha256=canonical_artifact.bundle_sha256,
        canonical_checksum_ledger_sha256=(
            canonical_artifact.checksum_ledger_sha256
        ),
        rendered_bundle_sha256=rendered_artifact.bundle_sha256,
        rendered_checksum_ledger_sha256=rendered_artifact.checksum_ledger_sha256,
    )
    report = Phase3ReadinessReport(
        schema_version=1,
        mode="real",
        ready_for_requested_mode=True,
        real_research_ready=True,
        software_annotation_runtime_ready=True,
        dicom_source_ready=True,
        extracted_cohort_ready=True,
        source_readiness_validated=True,
        phase2_artifacts_validated=True,
        cohort_projection_validated=True,
        frozen_split_validated=True,
        synthetic_non_research=False,
        cohort_view=str(cohort_view),
        expected_resolution=expected_resolution,
        metadata_path=str(metadata_path),
        sample_count=len(canonical_metadata),
        split_counts=_split_counts(canonical_metadata),
        manifest_bundle_sha256=verified.bundle_sha256,
        fingerprint=fingerprint,
        limitations=(
            "Phase 2 verified raw-source identities and headers, not full raw DICOM byte hashes.",
        ),
    )
    return report, canonical_metadata
