"""Fail-closed readiness validation for Phase 4 multi-rater supervision.

Phase 3 remains authoritative for the real cohort, its ordering, and all
Phase 0/2 provenance.  This module adds only the Phase 4 supervision join,
reader-count contract, and stable supervision fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import pandas as pd

from .hashing import sha256_json
from .manifest_guard import ManifestGuardReport
from .phase3_readiness import (
    Phase3InputFingerprint,
    Phase3ReadinessError,
    Phase3ReadinessReport,
    validate_phase3_readiness,
)
from .phase4_config import PRIMARY_COHORT_VIEW
from .phase4_supervision import build_supervision_frame


_PHASE4_RESOLUTION = 64
_IDENTITY_FIELDS = ("nodule_id", "series_instance_uid", "split")
_READER_FIELD = "n_valid_malignancy_ratings"
_TEXT_FINGERPRINT_FIELDS = (
    "patient_id",
    "nodule_id",
    "nodule_sha256",
    "series_instance_uid",
    "split",
    "label",
    "cohort_view",
    "source_validity_status",
)
_PATH_FINGERPRINT_FIELDS = ("rendered_mask_path", "mask_path")
_INTEGER_FINGERPRINT_FIELDS = ("resolution", _READER_FIELD)
_FLOAT_FINGERPRINT_FIELDS = (
    "q_benign",
    "q_indeterminate",
    "q_malignant",
    "disagreement_entropy",
    "ordinal_t1",
    "ordinal_t2",
    "ordinal_t3",
    "ordinal_t4",
)


class Phase4ReadinessError(RuntimeError):
    """Raised when Phase 4 inputs cannot be proven ready for research use."""


@dataclass(frozen=True)
class Phase4ReadinessReport:
    """Immutable evidence for the validated Phase 4 supervision inputs."""

    schema_version: int
    ready_for_requested_mode: bool
    real_research_ready: bool
    phase3_readiness_validated: bool
    multirater_supervision_validated: bool
    cohort_view: str
    expected_resolution: int
    sample_count: int
    split_counts: tuple[tuple[str, int], ...]
    reader_count_counts: tuple[tuple[int, int], ...]
    manifest_bundle_sha256: str
    phase3_input_fingerprint: Phase3InputFingerprint
    phase3_readiness_report: Phase3ReadinessReport
    supervision_sha256: str
    ordered_supervision_sha256: str
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ready_for_requested_mode": self.ready_for_requested_mode,
            "real_research_ready": self.real_research_ready,
            "phase3_readiness_validated": self.phase3_readiness_validated,
            "multirater_supervision_validated": (
                self.multirater_supervision_validated
            ),
            "cohort_view": self.cohort_view,
            "expected_resolution": self.expected_resolution,
            "sample_count": self.sample_count,
            "split_counts": dict(self.split_counts),
            "reader_count_counts": {
                str(readers): count
                for readers, count in self.reader_count_counts
            },
            "manifest_bundle_sha256": self.manifest_bundle_sha256,
            "phase3_input_fingerprint": self.phase3_input_fingerprint.to_dict(),
            "phase3_readiness_evidence": self.phase3_readiness_report.to_dict(),
            "supervision_sha256": self.supervision_sha256,
            "ordered_supervision_sha256": self.ordered_supervision_sha256,
            "limitations": list(self.limitations),
        }


def _fail(message: str) -> None:
    raise Phase4ReadinessError(message)


def _resolved_data_directory(
    value: Any,
    *,
    project_root: Path,
    role: str,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        _fail(f"{role} must be a non-empty path")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_dir():
        _fail(f"Phase 3-validated {role} does not exist: {resolved}")
    return resolved


def _read_validated_ledger(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        _fail(f"Phase 3-validated {role} does not exist: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise Phase4ReadinessError(
            f"Cannot read Phase 3-validated {role}: {path}"
        ) from exc


def _identity_sequence(frame: pd.DataFrame, role: str) -> list[tuple[str, ...]]:
    missing = [field for field in _IDENTITY_FIELDS if field not in frame.columns]
    if missing:
        _fail(f"{role} lacks stable identity fields: {missing}")
    result: list[tuple[str, ...]] = []
    for row in frame.loc[:, _IDENTITY_FIELDS].itertuples(index=False, name=None):
        identity = tuple(str(value).strip() for value in row)
        if any(not value for value in identity):
            _fail(f"{role} contains an empty stable identity field")
        result.append(identity)
    return result


def _exact_reader_counts(values: pd.Series, role: str) -> list[int]:
    counts: list[int] = []
    for value in values.tolist():
        if isinstance(value, bool):
            _fail(f"{role} must contain exact integer reader counts")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise Phase4ReadinessError(
                f"{role} must contain exact integer reader counts"
            ) from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            _fail(f"{role} must contain exact integer reader counts")
        counts.append(int(numeric))
    return counts


def _phase4_primary_frame(
    authoritative: pd.DataFrame,
    rendered: pd.DataFrame,
    *,
    phase3_report: Phase3ReadinessReport,
    expected_resolution: int,
) -> pd.DataFrame:
    if not isinstance(authoritative, pd.DataFrame):
        _fail("Phase 3 did not return an authoritative metadata DataFrame")
    if not isinstance(rendered, pd.DataFrame):
        _fail("rendered ledger must be a pandas DataFrame")

    primary = authoritative.copy(deep=True).reset_index(drop=True)
    authoritative_order = _identity_sequence(primary, "Phase 3 metadata")
    if len(primary) != phase3_report.sample_count:
        _fail("Phase 3 metadata row count differs from its readiness report")

    prior_resolution = primary.get("resolution")
    if prior_resolution is not None:
        values = _exact_reader_counts(prior_resolution, "Phase 3 resolution")
        if any(value != expected_resolution for value in values):
            _fail("Phase 3 metadata resolution differs from Phase 4 configuration")

    prior_cohort = primary.get("cohort_view")
    if prior_cohort is not None and not all(
        str(value).strip() == phase3_report.cohort_view
        for value in prior_cohort.tolist()
    ):
        _fail("Phase 3 metadata cohort differs from its readiness report")

    prior_readers = primary.get(_READER_FIELD)
    prior_reader_counts = (
        _exact_reader_counts(prior_readers, "Phase 3 metadata reader counts")
        if prior_readers is not None
        else None
    )

    primary = primary.drop(
        columns=[
            field
            for field in ("resolution", "cohort_view", _READER_FIELD)
            if field in primary.columns
        ]
    )
    # Phase 2 ledgers are read as strings, matching the proven Phase 3 gate.
    # Keep the join key in that representation; fingerprints normalize it to
    # a native integer later.
    primary["resolution"] = str(expected_resolution)
    primary["cohort_view"] = phase3_report.cohort_view

    reader_columns = (*_IDENTITY_FIELDS, _READER_FIELD)
    missing = [field for field in reader_columns if field not in rendered.columns]
    if missing:
        _fail(f"rendered ledger lacks Phase 4 reader linkage fields: {missing}")
    if rendered["nodule_id"].duplicated().any():
        _fail("Phase 4 primary-to-rendered reader linkage is not one-to-one")

    try:
        primary = primary.merge(
            rendered.loc[:, reader_columns],
            on=list(_IDENTITY_FIELDS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    except (KeyError, pd.errors.MergeError, ValueError) as exc:
        raise Phase4ReadinessError(
            "Phase 4 primary-to-rendered reader linkage is not one-to-one"
        ) from exc

    if primary[_READER_FIELD].isna().any() or (
        primary[_READER_FIELD].astype(str).str.strip() == ""
    ).any():
        _fail("rendered reader counts are missing for one or more primary samples")

    reader_counts = _exact_reader_counts(
        primary[_READER_FIELD],
        "Phase 4 primary reader counts",
    )
    primary[_READER_FIELD] = pd.array(reader_counts, dtype="int64")

    if prior_reader_counts is not None and prior_reader_counts != reader_counts:
        _fail("Phase 3 metadata reader counts disagree with the rendered ledger")

    invalid = [
        f"{nodule_id}={count}"
        for nodule_id, count in zip(primary["nodule_id"], reader_counts)
        if count not in {2, 3, 4}
    ]
    if invalid:
        _fail(
            "Every Phase 4 primary sample must have exactly 2, 3, or 4 valid "
            f"malignancy readers; invalid samples: {invalid}"
        )

    if _identity_sequence(primary, "enriched Phase 4 primary cohort") != authoritative_order:
        _fail("Enriching Phase 3 metadata changed primary cohort membership or order")
    return primary


def _primary_canonical_projection(
    primary: pd.DataFrame,
    rendered: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Keep canonical rows relevant to the authoritative Phase 4 cohort.

    The completed canonical ledger also contains excluded nodules, which can
    legitimately have blank mask paths and no supervision values. Those rows
    must not create duplicate empty mask identities inside the authoritative
    supervision builder. Rows sharing a relevant SHA are deliberately all
    retained so that the builder still rejects relevant duplicates.
    """

    for frame, required, role in (
        (primary, {"nodule_id"}, "Phase 4 primary cohort"),
        (rendered, {"nodule_id", "nodule_sha256"}, "rendered ledger"),
        (canonical, {"nodule_sha256"}, "canonical supervision"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            _fail(f"{role} lacks canonical projection fields: {missing}")

    primary_ids = set(primary["nodule_id"].astype(str))
    relevant_rendered = rendered.loc[
        rendered["nodule_id"].astype(str).isin(primary_ids),
        ["nodule_id", "nodule_sha256"],
    ].copy()
    if len(relevant_rendered) != len(primary):
        _fail("Rendered canonical identities do not match the Phase 4 primary cohort")

    sha_values = [str(value).strip() for value in relevant_rendered["nodule_sha256"]]
    if any(not value for value in sha_values) or len(set(sha_values)) != len(primary):
        _fail("Rendered canonical identities are empty or not one-to-one")

    return canonical.loc[
        canonical["nodule_sha256"].astype(str).str.strip().isin(set(sha_values))
    ].copy()


def _native_text(value: Any, role: str) -> str:
    if pd.isna(value):
        _fail(f"Supervision fingerprint field {role!r} is missing")
    text = str(value).strip()
    if not text:
        _fail(f"Supervision fingerprint field {role!r} is empty")
    return text


def _native_relative_path(value: Any, role: str) -> str:
    text = _native_text(value, role).replace("\\", "/")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"Supervision fingerprint path {role!r} must be canonical and relative")
    return path.as_posix()


def _native_integer(value: Any, role: str) -> int:
    return _exact_reader_counts(pd.Series([value]), role)[0]


def _native_float(value: Any, role: str) -> float:
    if isinstance(value, bool):
        _fail(f"Supervision fingerprint field {role!r} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase4ReadinessError(
            f"Supervision fingerprint field {role!r} must be numeric"
        ) from exc
    if not math.isfinite(numeric):
        _fail(f"Supervision fingerprint field {role!r} must be finite")
    return 0.0 if numeric == 0.0 else numeric


def _supervision_hashes(frame: pd.DataFrame) -> tuple[str, str]:
    fields = (
        *_TEXT_FINGERPRINT_FIELDS,
        *_PATH_FINGERPRINT_FIELDS,
        *_INTEGER_FINGERPRINT_FIELDS,
        *_FLOAT_FINGERPRINT_FIELDS,
    )
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        _fail(f"Validated supervision lacks fingerprint fields: {missing}")

    ordered: list[dict[str, str | int | float]] = []
    for _, row in frame.loc[:, fields].iterrows():
        record: dict[str, str | int | float] = {}
        for field in _TEXT_FINGERPRINT_FIELDS:
            record[field] = _native_text(row[field], field)
        for field in _PATH_FINGERPRINT_FIELDS:
            record[field] = _native_relative_path(row[field], field)
        for field in _INTEGER_FINGERPRINT_FIELDS:
            record[field] = _native_integer(row[field], field)
        for field in _FLOAT_FINGERPRINT_FIELDS:
            record[field] = _native_float(row[field], field)
        ordered.append(record)

    logical = sorted(
        ordered,
        key=lambda record: (
            record["nodule_sha256"],
            record["nodule_id"],
            record["series_instance_uid"],
            record["split"],
        ),
    )
    return sha256_json(logical), sha256_json(ordered)


def _split_counts(frame: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    return tuple(
        (split, int((frame["split"].astype(str) == split).sum()))
        for split in ("train", "val", "test")
    )


def _reader_count_counts(frame: pd.DataFrame) -> tuple[tuple[int, int], ...]:
    return tuple(
        (readers, int((frame[_READER_FIELD] == readers).sum()))
        for readers in (2, 3, 4)
    )


def validate_phase4_readiness(
    data_config: Mapping[str, Any],
    *,
    manifest: ManifestGuardReport,
    expected_resolution: int,
    project_root: str | Path,
    synthetic_dry_run: bool = False,
) -> tuple[Phase4ReadinessReport, pd.DataFrame]:
    """Validate Phase 3 first, then add Phase 4 multi-rater evidence.

    The returned supervision frame retains the exact Phase 3-authoritative
    membership and ordering. Persisted Phase 2 ledgers are read only after the
    Phase 3 gate succeeds.
    """

    try:
        phase3_report, authoritative = validate_phase3_readiness(
            data_config,
            manifest=manifest,
            expected_resolution=expected_resolution,
            project_root=project_root,
            synthetic_dry_run=synthetic_dry_run,
        )
    except Phase3ReadinessError as exc:
        raise Phase4ReadinessError(
            f"Phase 3 readiness failed before Phase 4 validation: {exc}"
        ) from exc

    if not isinstance(phase3_report, Phase3ReadinessReport):
        _fail("Phase 3 readiness did not return a Phase3ReadinessReport")
    if not isinstance(authoritative, pd.DataFrame):
        _fail("Phase 3 readiness did not return authoritative metadata")

    if data_config.get("mode") != "real":
        _fail("Phase 4 readiness accepts only the locked real-data configuration")
    if (
        phase3_report.mode != "real"
        or not phase3_report.ready_for_requested_mode
        or not phase3_report.real_research_ready
        or phase3_report.synthetic_non_research
    ):
        _fail("Phase 4 requires Phase 3 to establish real research readiness")
    if synthetic_dry_run:
        _fail("Phase 4 real research readiness cannot use a synthetic dry run")

    if (
        data_config.get("cohort_view") != PRIMARY_COHORT_VIEW
        or phase3_report.cohort_view != PRIMARY_COHORT_VIEW
    ):
        _fail(f"Phase 4 cohort must be exactly {PRIMARY_COHORT_VIEW!r}")
    if (
        expected_resolution != _PHASE4_RESOLUTION
        or phase3_report.expected_resolution != _PHASE4_RESOLUTION
    ):
        _fail("Phase 4 expected resolution must be exactly 64")

    root = Path(project_root).resolve()
    if not root.is_dir():
        _fail(f"project_root does not exist: {root}")
    canonical_dir = _resolved_data_directory(
        data_config.get("canonical_dir"),
        project_root=root,
        role="canonical directory",
    )
    rendered_dir = _resolved_data_directory(
        data_config.get("rendered_dir"),
        project_root=root,
        role="rendered directory",
    )
    canonical = _read_validated_ledger(
        canonical_dir / "canonical_nodule_ledger.csv",
        "canonical nodule ledger",
    )
    rendered = _read_validated_ledger(
        rendered_dir / "rendered_rois.csv",
        "rendered ROI ledger",
    )

    primary = _phase4_primary_frame(
        authoritative,
        rendered,
        phase3_report=phase3_report,
        expected_resolution=expected_resolution,
    )
    canonical_primary = _primary_canonical_projection(primary, rendered, canonical)
    authoritative_order = _identity_sequence(authoritative, "Phase 3 metadata")
    try:
        supervision = build_supervision_frame(primary, rendered, canonical_primary)
    except Exception as exc:
        raise Phase4ReadinessError(
            f"Phase 4 multi-rater supervision validation failed: {exc}"
        ) from exc

    if len(supervision) != len(authoritative):
        _fail("Phase 4 supervision row count differs from Phase 3 metadata")
    if _identity_sequence(supervision, "Phase 4 supervision") != authoritative_order:
        _fail("Phase 4 supervision changed Phase 3 cohort membership or order")

    split_counts = _split_counts(supervision)
    if dict(split_counts) != dict(phase3_report.split_counts):
        _fail("Phase 4 supervision split counts differ from Phase 3 readiness")
    supervision_sha256, ordered_supervision_sha256 = _supervision_hashes(supervision)
    limitations = tuple(
        dict.fromkeys(
            (
                *phase3_report.limitations,
                "Phase 4 readiness validates supervision inputs, not trained-model performance.",
            )
        )
    )
    report = Phase4ReadinessReport(
        schema_version=1,
        ready_for_requested_mode=True,
        real_research_ready=True,
        phase3_readiness_validated=True,
        multirater_supervision_validated=True,
        cohort_view=PRIMARY_COHORT_VIEW,
        expected_resolution=_PHASE4_RESOLUTION,
        sample_count=len(supervision),
        split_counts=split_counts,
        reader_count_counts=_reader_count_counts(supervision),
        manifest_bundle_sha256=phase3_report.manifest_bundle_sha256,
        phase3_input_fingerprint=phase3_report.fingerprint,
        phase3_readiness_report=phase3_report,
        supervision_sha256=supervision_sha256,
        ordered_supervision_sha256=ordered_supervision_sha256,
        limitations=limitations,
    )
    return report, supervision
