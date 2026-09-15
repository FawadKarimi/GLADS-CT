"""Fail-closed construction contract for the Phase 2 boundary cohort.

The prompt's 2.6--2.9 and 3.1--3.4 ranges are approximate descriptions.  The
authoritative Phase 2 integer rule includes the attainable endpoints 2.5 and
3.5, excludes 3.0, and is reused here rather than redefined for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .lidc_ledger import LEDGER_COLUMNS
from .lidc_targets import paper_mean_label
from .lidc_render_pipeline import RENDERED_COLUMNS
from .manifest_guard import (
    CHECKSUM_LEDGER,
    FROZEN_MANIFEST_FILES,
    ManifestGuardReport,
    validate_frozen_manifest_bundle,
)
from .phase4_config import PRIMARY_COHORT_VIEW, load_phase4_config
from .phase4_input_lock import (
    Phase4InputLock,
    load_phase4_input_lock,
    validate_phase4_input_lock,
)
from .phase4_readiness import Phase4ReadinessReport, validate_phase4_readiness


# These pairs describe the exact Phase 2 intervals. Inclusivity is recorded in
# BOUNDARY_RANGE_RECORDS because it differs on the shared 3.0 endpoint.
BOUNDARY_RANGES: tuple[tuple[float, float], tuple[float, float]] = (
    (2.5, 3.0),
    (3.0, 3.5),
)
BOUNDARY_RANGE_RECORDS: tuple[dict[str, float | bool], ...] = (
    {
        "lower": 2.5,
        "upper": 3.0,
        "lower_inclusive": True,
        "upper_inclusive": False,
    },
    {
        "lower": 3.0,
        "upper": 3.5,
        "lower_inclusive": False,
        "upper_inclusive": True,
    },
)
BOUNDARY_TARGET_INTERPRETATION = (
    "Boundary nodules retain Phase 2 paper_label='boundary_zone' and are not "
    "assigned a primary three-class target. Lower and upper strata describe "
    "which side of mean malignancy 3.0 they occupy; they do not relabel a "
    "boundary case as benign, indeterminate, or malignant. Model outputs are "
    "therefore reported as a separate descriptive stress analysis."
)
BOUNDARY_LIMITATIONS: tuple[str, ...] = (
    "Boundary nodules were excluded from the primary model-development cohort.",
    "The boundary analysis is descriptive and must not be mixed into the primary test headline.",
    "No primary three-class ground truth is invented for excluded boundary-zone cases.",
    "Boundary results must not be used for threshold, loss-weight, or architecture tuning.",
    "Attention or prediction behavior on this selected stress cohort does not establish causal explanation or clinical utility.",
    "Excluded nodules may share patients with primary training nodules; this is not an independent held-out clinical validation cohort.",
)

_IDENTITY_FIELDS = (
    "patient_id",
    "nodule_id",
    "nodule_sha256",
    "study_instance_uid",
    "series_instance_uid",
    "split",
)
_EXACT_RULE_FIELDS = (
    "mean_malignancy",
    "malignancy_score_sum",
    "n_valid_malignancy_ratings",
)
_SPLITS = ("train", "val", "test")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PRIMARY_HASH_FIELDS = (
    "patient_id",
    "series_instance_uid",
    "nodule_id",
    "split",
    "label",
)
_FULL_BOUNDARY_IDENTITY_FIELDS = (
    *_IDENTITY_FIELDS,
    "image_path",
    "rendered_mask_path",
    "mean_malignancy",
    "malignancy_score_sum",
    "n_valid_malignancy_ratings",
    "boundary_stratum",
)

BOUNDARY_SOURCE_VALIDITY_STATUSES = frozenset(
    {"valid", "valid_with_missing_ratings"}
)
_SOURCE_INVALID_STATUS = "excluded_malformed_geometry"
_SOURCE_INVALID_REASON = "pre_existing_phase2_malformed_geometry"
BOUNDARY_EVALUATED_POPULATION = "source_valid_boundary_subset"
BOUNDARY_SOURCE_VALIDITY_NOTE = (
    "Only the source-valid boundary subset is evaluated, after all existing "
    "rendering and evaluation checks. Pre-existing Phase 2 malformed-geometry "
    "exclusions remain in full boundary accounting; missing renders never "
    "determine exclusions. Results remain descriptive stress testing only."
)
_MEMBERSHIP_ACCOUNTING_FIELDS = (
    *_IDENTITY_FIELDS, *_EXACT_RULE_FIELDS, "paper_label",
    "paper_primary_eligible", "boundary_zone", "source_validity_status",
)
BOUNDARY_EXCLUSION_FIELDS = (
    "patient_id", "nodule_id", "split", "mean_malignancy",
    "source_validity_status", "exclusion_reason",
)
FROZEN_BOUNDARY_RESOLUTION = 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE4_CONFIG = PROJECT_ROOT / "configs" / "phase4_consensdxformer.yaml"
DEFAULT_PHASE4_INPUT_LOCK = PROJECT_ROOT / "configs" / "phase4_input_lock.json"


class Phase4BoundaryError(ValueError):
    """Raised when boundary membership or provenance is not authoritative."""


def _boolean(value: Any, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in {"true", "1"}:
            return True
        if token in {"false", "0"}:
            return False
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in {0, 1}:
            return bool(value)
    raise Phase4BoundaryError(f"{role} must contain explicit boolean values")


def _exact_integer(value: Any, role: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise Phase4BoundaryError(f"{role} must be an exact integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    raise Phase4BoundaryError(f"{role} must be an exact canonical integer")


def _finite_mean(value: Any, role: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise Phase4BoundaryError(f"{role} must be finite numeric data")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase4BoundaryError(f"{role} must be finite numeric data") from exc
    if not math.isfinite(numeric) or numeric < 1.0 or numeric > 5.0:
        raise Phase4BoundaryError(f"{role} must be finite and lie in [1, 5]")
    return numeric


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value)


def _canonical_text(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Phase4BoundaryError(
            f"{role} must be a non-empty canonical string without surrounding whitespace"
        )
    return value


def _sha256_text(value: Any, role: str) -> str:
    text = _canonical_text(value, role)
    if _SHA256_RE.fullmatch(text) is None:
        raise Phase4BoundaryError(f"{role} must be a lowercase SHA-256 digest")
    return text


def _canonical_relative_path(value: Any, role: str) -> str:
    text = _canonical_text(value, role)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise Phase4BoundaryError(f"{role} must be a canonical relative path")
    return text


def _phase2_mean_boundary(mean: float) -> tuple[bool, str | None]:
    if 2.5 <= mean < 3.0:
        return True, "lower_boundary"
    if 3.0 < mean <= 3.5:
        return True, "upper_boundary"
    return False, None


def _validate_ranges(ranges: Sequence[Sequence[float]]) -> None:
    try:
        normalized = tuple(
            tuple(float(endpoint) for endpoint in interval) for interval in ranges
        )
    except (TypeError, ValueError) as exc:
        raise Phase4BoundaryError("Phase 2 boundary ranges must not be redefined") from exc
    if normalized != BOUNDARY_RANGES:
        raise Phase4BoundaryError("Phase 2 boundary ranges must not be redefined")


def _primary_ids(values: Iterable[str] | pd.DataFrame) -> set[str]:
    if isinstance(values, pd.DataFrame):
        if "nodule_id" not in values.columns:
            raise Phase4BoundaryError("primary cohort lacks nodule_id")
        raw = values["nodule_id"].tolist()
    elif isinstance(values, (str, bytes)):
        raise TypeError("primary_nodule_ids must be an iterable of identifiers")
    else:
        try:
            raw = list(values)
        except TypeError as exc:
            raise TypeError(
                "primary_nodule_ids must be an iterable of identifiers"
            ) from exc
    normalized = [
        _canonical_text(value, "primary cohort nodule_id") for value in raw
    ]
    if len(set(normalized)) != len(normalized):
        raise Phase4BoundaryError("primary cohort contains duplicate nodule_id")
    return set(normalized)


def _json_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise Phase4BoundaryError("source_provenance must be a non-empty mapping")
    try:
        payload = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        copied = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise Phase4BoundaryError("source_provenance must be JSON-serializable") from exc
    if not isinstance(copied, dict):
        raise Phase4BoundaryError("source_provenance must serialize to an object")
    return copied


def _phase2_membership(
    frame: pd.DataFrame,
) -> tuple[pd.Series, list[float | None], list[str | None]]:
    """Recompute the Phase 2 rule and reconcile any persisted marker fields."""

    means: list[float | None] = []
    derived: list[bool] = []
    strata: list[str | None] = []
    derived_labels: list[str] = []
    for index, row in frame.iterrows():
        role = f"Phase 2 row {index}"
        mean_blank = _is_blank(row["mean_malignancy"])
        score_blank = _is_blank(row["malignancy_score_sum"])
        readers_blank = _is_blank(row["n_valid_malignancy_ratings"])
        if mean_blank:
            if (
                not score_blank
                or readers_blank
                or _exact_integer(
                    row["n_valid_malignancy_ratings"],
                    f"{role} n_valid_malignancy_ratings",
                )
                != 0
            ):
                raise Phase4BoundaryError(
                    f"{role} has partially missing Phase 2 rating targets"
                )
            means.append(None)
            derived.append(False)
            strata.append(None)
            derived_labels.append("unrated")
            continue
        if score_blank or readers_blank:
            raise Phase4BoundaryError(
                f"{role} has mean_malignancy without Phase 2 rating totals"
            )
        mean = _finite_mean(row["mean_malignancy"], f"{role} mean_malignancy")
        score_sum = _exact_integer(
            row["malignancy_score_sum"], f"{role} malignancy_score_sum"
        )
        readers = _exact_integer(
            row["n_valid_malignancy_ratings"],
            f"{role} n_valid_malignancy_ratings",
        )
        if readers <= 0 or score_sum < readers or score_sum > 5 * readers:
            raise Phase4BoundaryError(f"{role} has impossible Phase 2 rating totals")
        expected_mean = score_sum / readers
        if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1e-12):
            raise Phase4BoundaryError(
                f"{role} mean_malignancy disagrees with Phase 2 rating totals"
            )
        label, _eligible, boundary = paper_mean_label(score_sum, readers)
        mean_boundary, stratum = _phase2_mean_boundary(mean)
        if boundary != mean_boundary:
            raise Phase4BoundaryError(
                f"{role} cannot be reconciled to the exact Phase 2 boundary rule"
            )
        means.append(mean)
        derived.append(boundary)
        strata.append(stratum)
        derived_labels.append(label)

    result = pd.Series(derived, index=frame.index, dtype=bool)
    if "boundary_zone" in frame.columns:
        persisted_boundary = frame["boundary_zone"].map(
            lambda value: _boolean(value, "boundary_zone")
        )
        if not persisted_boundary.equals(result):
            raise Phase4BoundaryError(
                "Persisted boundary_zone values disagree with the Phase 2 boundary rule"
            )
    if "paper_label" in frame.columns:
        persisted_labels = frame["paper_label"].map(
            lambda value: _canonical_text(value, "paper_label")
        )
        expected_labels = pd.Series(derived_labels, index=frame.index)
        if not persisted_labels.equals(expected_labels):
            raise Phase4BoundaryError(
                "Persisted paper_label values disagree with the Phase 2 boundary rule"
            )
    if "paper_primary_eligible" in frame.columns:
        eligible = frame["paper_primary_eligible"].map(
            lambda value: _boolean(value, "paper_primary_eligible")
        )
        if bool((eligible & result).any()):
            raise Phase4BoundaryError(
                "A Phase 2 boundary row is incorrectly primary-cohort eligible"
            )
    return result, means, strata


def isolate_phase2_boundary_cohort(
    phase2_rows: pd.DataFrame,
    *,
    primary_nodule_ids: Iterable[str] | pd.DataFrame,
    source_provenance: Mapping[str, Any],
    boundary_ranges: Sequence[Sequence[float]] = BOUNDARY_RANGES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Account for all boundary members, then return the source-valid subset.

    The input is never mutated. The returned frame preserves source order and
    every source column of eligible rows, adding only ``boundary_stratum``.
    Source-invalid members remain in the audit record. No model inference,
    target assignment, threshold selection, or artifact write occurs here.
    """

    _validate_ranges(boundary_ranges)
    if not isinstance(phase2_rows, pd.DataFrame) or phase2_rows.empty:
        raise Phase4BoundaryError("phase2_rows must be a non-empty DataFrame")
    required = {
        *_IDENTITY_FIELDS,
        *_EXACT_RULE_FIELDS,
        "paper_label",
        "paper_primary_eligible",
        "boundary_zone",
        "source_validity_status",
    }
    missing = sorted(required - set(phase2_rows.columns))
    if missing:
        raise Phase4BoundaryError(f"Phase 2 boundary source lacks fields: {missing}")

    source = phase2_rows.copy(deep=True)
    for field in _IDENTITY_FIELDS:
        source[field].map(
            lambda value: _canonical_text(value, f"Phase 2 {field}")
        )
    source["nodule_sha256"].map(
        lambda value: _sha256_text(value, "Phase 2 nodule_sha256")
    )
    if source["nodule_id"].duplicated().any():
        raise Phase4BoundaryError("Phase 2 boundary source has duplicate nodule_id")
    if source["nodule_sha256"].duplicated().any():
        raise Phase4BoundaryError("Phase 2 boundary source has duplicate nodule_sha256")

    membership, means, strata = _phase2_membership(source)
    source["boundary_stratum"] = strata
    source["_validated_mean"] = means
    boundary = source.loc[membership].copy(deep=True)
    if boundary.empty:
        raise Phase4BoundaryError("Phase 2 source contains no boundary-zone rows")
    if not boundary["split"].isin(_SPLITS).all():
        raise Phase4BoundaryError("Boundary cohort contains a noncanonical split")
    statuses = boundary["source_validity_status"].map(
        lambda value: _canonical_text(value, "source_validity_status")
    )
    if not statuses.isin(BOUNDARY_SOURCE_VALIDITY_STATUSES | {_SOURCE_INVALID_STATUS}).all():
        unknown = sorted(set(statuses) - BOUNDARY_SOURCE_VALIDITY_STATUSES - {_SOURCE_INVALID_STATUS})
        raise Phase4BoundaryError(
            "Boundary cohort source validity is outside the exact renderable "
            f"allowlist and known Phase 2 source-invalid exclusion: {unknown}"
        )

    primary = _primary_ids(primary_nodule_ids)
    overlap = sorted(set(boundary["nodule_id"].astype(str)) & primary)
    if overlap:
        raise Phase4BoundaryError(
            f"Boundary cohort overlaps the primary cohort: {overlap}"
        )

    # Partition only AFTER full-membership identity, rule, split and primary
    # overlap checks. Preserve raw canonical values (not rounded float means)
    # in the accounting/exclusion ledger, including all source-invalid rows.
    accounting = _source_validity_accounting(boundary)
    boundary = boundary.loc[statuses.isin(BOUNDARY_SOURCE_VALIDITY_STATUSES)].copy()
    if boundary.empty:
        raise Phase4BoundaryError("No source-valid boundary rows remain for evaluation")

    ordered_identities: list[dict[str, Any]] = []
    for _, row in boundary.iterrows():
        identity: dict[str, Any] = {
            field: str(row[field]) for field in _IDENTITY_FIELDS
        }
        identity.update(
            {
                "mean_malignancy": float(row["_validated_mean"]),
                "boundary_stratum": str(row["boundary_stratum"]),
            }
        )
        identity["malignancy_score_sum"] = _exact_integer(
            row["malignancy_score_sum"], "malignancy_score_sum"
        )
        identity["n_valid_malignancy_ratings"] = _exact_integer(
            row["n_valid_malignancy_ratings"],
            "n_valid_malignancy_ratings",
        )
        for field in ("image_path", "rendered_mask_path"):
            if field in boundary.columns:
                identity[field] = _canonical_relative_path(row[field], field)
        ordered_identities.append(identity)

    logical_identities = sorted(
        ordered_identities,
        key=lambda row: (
            row["nodule_sha256"],
            row["nodule_id"],
            row["series_instance_uid"],
            row["split"],
        ),
    )
    provenance = _json_provenance(source_provenance)
    record: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "phase4_boundary_stress",
        "source_definition": "authoritative_phase2_paper_mean_label",
        "phase2_exact_rule": (
            "boundary_zone iff 2*score_sum is in [5*n_ratings, 6*n_ratings) "
            "or (6*n_ratings, 7*n_ratings]"
        ),
        "prompt_approximate_ranges": ["2.6-2.9", "3.1-3.4"],
        "boundary_ranges": [dict(item) for item in BOUNDARY_RANGE_RECORDS],
        "ranges_redefined": False,
        "sample_count": len(boundary),
        "split_counts": {
            split: int((boundary["split"].astype(str) == split).sum())
            for split in _SPLITS
        },
        "stratum_counts": {
            stratum: int((boundary["boundary_stratum"] == stratum).sum())
            for stratum in ("lower_boundary", "upper_boundary")
        },
        "ordered_identities": ordered_identities,
        "boundary_cohort_sha256": sha256_json(logical_identities),
        "ordered_boundary_cohort_sha256": sha256_json(ordered_identities),
        "primary_cohort_excluded": True,
        "primary_overlap_count": 0,
        "primary_headline_inclusion_permitted": False,
        "frozen_final_model_required": True,
        "boundary_driven_tuning_forbidden": True,
        "threshold_or_architecture_selection_permitted": False,
        "target_interpretation": BOUNDARY_TARGET_INTERPRETATION,
        "limitations": list(BOUNDARY_LIMITATIONS),
        "source_provenance": provenance,
        "evaluated_population": BOUNDARY_EVALUATED_POPULATION,
        "source_validity_note": BOUNDARY_SOURCE_VALIDITY_NOTE,
        "source_validity_accounting": accounting,
    }
    # Keep the public frame source-faithful apart from its declared stratum.
    boundary.drop(columns=["_validated_mean"], inplace=True)
    boundary.reset_index(drop=True, inplace=True)
    return boundary, record


def _source_validity_accounting(boundary: pd.DataFrame) -> dict[str, Any]:
    """Exact canonical values and reproducible counts; no file-based selection."""
    members = boundary.loc[:, list(_MEMBERSHIP_ACCOUNTING_FIELDS)].to_dict("records")
    exclusions = [
        {**{field: row[field] for field in BOUNDARY_EXCLUSION_FIELDS if field != "exclusion_reason"},
         "exclusion_reason": _SOURCE_INVALID_REASON}
        for row in members if row["source_validity_status"] == _SOURCE_INVALID_STATUS
    ]
    eligible = [row for row in members if row["source_validity_status"] in BOUNDARY_SOURCE_VALIDITY_STATUSES]
    def split_counts(rows):
        return {split: sum(row["split"] == split for row in rows) for split in _SPLITS}
    return {
        "full_boundary_membership_count": len(members),
        "full_boundary_split_counts": split_counts(members),
        "source_invalid_exclusion_count": len(exclusions),
        "source_invalid_exclusion_counts_by_reason": {_SOURCE_INVALID_REASON: len(exclusions)},
        "source_invalid_exclusion_split_counts": split_counts(exclusions),
        "source_invalid_exclusion_counts_by_reason_and_split": {_SOURCE_INVALID_REASON: split_counts(exclusions)},
        "eligible_subset_count": len(eligible),
        "eligible_subset_split_counts": split_counts(eligible),
        "exclusion_records": exclusions,
        "full_boundary_membership": members,
        "ordered_full_boundary_membership_sha256": sha256_json(members),
        "full_boundary_membership_sha256": sha256_json(sorted(members, key=lambda row: row["nodule_id"])),
        "exclusion_records_sha256": sha256_json(exclusions),
    }


@dataclass(frozen=True)
class Phase4BoundarySource:
    """A readiness-gated boundary cohort and immutable source evidence."""

    frame: pd.DataFrame
    provenance: Mapping[str, Any]
    source_snapshot: Mapping[str, str]
    rendered_root: Path
    readiness_report: Phase4ReadinessReport


def _report_payload(value: Any, role: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif callable(getattr(value, "to_dict", None)):
        payload = dict(value.to_dict())
    else:
        raise Phase4BoundaryError(f"{role} must expose a mapping payload")
    return _json_provenance(payload)


def _primary_identity_payloads(
    primary_supervision: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(primary_supervision, pd.DataFrame) or primary_supervision.empty:
        raise Phase4BoundaryError(
            "primary_supervision must be a non-empty validated DataFrame"
        )
    required = {
        *_PRIMARY_HASH_FIELDS,
        "nodule_sha256",
        "n_valid_malignancy_ratings",
    }
    missing = sorted(required - set(primary_supervision.columns))
    if missing:
        raise Phase4BoundaryError(
            f"primary supervision lacks frozen identity fields: {missing}"
        )

    extended: list[dict[str, Any]] = []
    phase3: list[dict[str, str]] = []
    for position, row in primary_supervision.iterrows():
        record: dict[str, Any] = {}
        for field in (
            "patient_id",
            "nodule_id",
            "series_instance_uid",
            "split",
            "label",
        ):
            record[field] = _canonical_text(
                row[field], f"primary row {position} {field}"
            )
        record["nodule_sha256"] = _sha256_text(
            row["nodule_sha256"], f"primary row {position} nodule_sha256"
        )
        if record["split"] not in _SPLITS:
            raise Phase4BoundaryError(
                f"primary row {position} contains a noncanonical split"
            )
        if record["label"] not in {"benign", "indeterminate", "malignant"}:
            raise Phase4BoundaryError(
                f"primary row {position} contains a noncanonical primary label"
            )
        readers = _exact_integer(
            row["n_valid_malignancy_ratings"],
            f"primary row {position} n_valid_malignancy_ratings",
        )
        if readers not in {2, 3, 4}:
            raise Phase4BoundaryError(
                f"primary row {position} violates the frozen 2-to-4 reader cohort"
            )
        record["n_valid_malignancy_ratings"] = readers
        extended.append(record)
        phase3.append({field: str(record[field]) for field in _PRIMARY_HASH_FIELDS})

    keys = [(row["nodule_id"], row["nodule_sha256"]) for row in extended]
    if len(set(keys)) != len(keys):
        raise Phase4BoundaryError("primary supervision contains duplicate identities")
    return extended, phase3


def _primary_binding(
    primary_supervision: pd.DataFrame,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    extended, phase3 = _primary_identity_payloads(primary_supervision)
    logical_phase3 = sorted(
        phase3,
        key=lambda row: tuple(row[field] for field in _PRIMARY_HASH_FIELDS),
    )
    phase3_logical_sha = sha256_json(logical_phase3)
    phase3_ordered_sha = sha256_json(phase3)
    fingerprint = readiness.get("phase3_input_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise Phase4BoundaryError(
            "Phase 4 readiness lacks the Phase 3 input fingerprint"
        )
    if fingerprint.get("logical_cohort_sha256") != phase3_logical_sha:
        raise Phase4BoundaryError(
            "primary supervision differs from the Phase 3 logical cohort fingerprint"
        )
    if fingerprint.get("ordered_samples_sha256") != phase3_ordered_sha:
        raise Phase4BoundaryError(
            "primary supervision differs from the Phase 3 ordered cohort fingerprint"
        )

    split_counts = {
        split: sum(row["split"] == split for row in extended) for split in _SPLITS
    }
    reader_counts = {
        str(readers): sum(
            row["n_valid_malignancy_ratings"] == readers for row in extended
        )
        for readers in (2, 3, 4)
    }
    expected_split_counts = {
        str(key): int(value)
        for key, value in dict(readiness.get("split_counts", {})).items()
    }
    expected_reader_counts = {
        str(key): int(value)
        for key, value in dict(readiness.get("reader_count_counts", {})).items()
    }
    if len(extended) != readiness.get("sample_count"):
        raise Phase4BoundaryError(
            "primary supervision count differs from Phase 4 readiness"
        )
    if split_counts != expected_split_counts:
        raise Phase4BoundaryError(
            "primary supervision split counts differ from Phase 4 readiness"
        )
    if reader_counts != expected_reader_counts:
        raise Phase4BoundaryError(
            "primary supervision reader counts differ from Phase 4 readiness"
        )

    logical_extended = sorted(
        extended,
        key=lambda row: (
            row["nodule_sha256"],
            row["nodule_id"],
            row["series_instance_uid"],
            row["split"],
        ),
    )
    return {
        "cohort_view": readiness.get("cohort_view"),
        "expected_resolution": readiness.get("expected_resolution"),
        "sample_count": len(extended),
        "split_counts": split_counts,
        "reader_count_counts": reader_counts,
        "manifest_bundle_sha256": readiness.get("manifest_bundle_sha256"),
        "supervision_sha256": readiness.get("supervision_sha256"),
        "ordered_supervision_sha256": readiness.get(
            "ordered_supervision_sha256"
        ),
        "phase3_input_fingerprint": dict(fingerprint),
        "identity_fields": [
            "patient_id",
            "nodule_id",
            "nodule_sha256",
            "series_instance_uid",
            "split",
            "label",
            "n_valid_malignancy_ratings",
        ],
        "logical_primary_identity_sha256": sha256_json(logical_extended),
        "ordered_primary_identity_sha256": sha256_json(extended),
    }


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase4BoundaryError(f"{role} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4BoundaryError(f"Cannot parse {role}: {path}") from exc
    if not isinstance(value, dict):
        raise Phase4BoundaryError(f"{role} must contain a JSON object")
    return value


def _safe_artifact_member(root: Path, value: Any, role: str) -> tuple[str, Path]:
    relative = _canonical_relative_path(value, role)
    target = root.joinpath(*PurePosixPath(relative).parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise Phase4BoundaryError(f"{role} escapes its Phase 2 root") from exc
    return relative, target


def _completed_phase2_snapshot(
    directory: Path,
    *,
    role: str,
    expected_bundle_sha256: Any,
    expected_checksum_ledger_sha256: Any,
    required_member: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    completed_path = directory / "COMPLETED.json"
    completed = _read_json_object(completed_path, f"{role} COMPLETED marker")
    if set(completed) != {
        "schema_version",
        "checksum_ledger",
        "checksum_ledger_sha256",
    } or completed.get("schema_version") != 1:
        raise Phase4BoundaryError(f"{role} COMPLETED marker schema is invalid")
    if completed.get("checksum_ledger") != "checksums.json":
        raise Phase4BoundaryError(f"{role} COMPLETED marker selects another ledger")
    ledger_sha = _sha256_text(
        completed.get("checksum_ledger_sha256"),
        f"{role} checksum-ledger digest",
    )
    if ledger_sha != _sha256_text(
        expected_checksum_ledger_sha256,
        f"expected {role} checksum-ledger digest",
    ):
        raise Phase4BoundaryError(
            f"{role} checksum ledger differs from Phase 4 readiness"
        )
    checksums_path = directory / "checksums.json"
    if sha256_file(checksums_path) != ledger_sha:
        raise Phase4BoundaryError(f"{role} checksum ledger was modified")
    ledger = _read_json_object(checksums_path, f"{role} checksum ledger")
    if set(ledger) != {"schema_version", "files", "bundle_sha256", "hash_scope"}:
        raise Phase4BoundaryError(f"{role} checksum ledger schema is invalid")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("hash_scope")
        != "listed generated files; raw DICOM bytes are not hashed"
    ):
        raise Phase4BoundaryError(f"{role} checksum ledger contract is invalid")
    files = ledger.get("files")
    if not isinstance(files, dict) or not files or required_member not in files:
        raise Phase4BoundaryError(
            f"{role} checksum ledger omits {required_member}"
        )
    if not {"run_configuration.json", "provenance.json"}.issubset(files):
        raise Phase4BoundaryError(f"{role} checksum ledger omits provenance members")

    snapshot: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in files.items():
        name, target = _safe_artifact_member(
            directory, raw_name, f"{role} checksum member"
        )
        digest = _sha256_text(raw_digest, f"{role} checksum for {name}")
        if not target.is_file() or sha256_file(target) != digest:
            raise Phase4BoundaryError(f"{role} checksummed member was modified: {name}")
        normalized[name] = digest
        snapshot[str(target)] = digest
    bundle = _sha256_text(ledger.get("bundle_sha256"), f"{role} bundle digest")
    if bundle != sha256_json(normalized) or bundle != _sha256_text(
        expected_bundle_sha256, f"expected {role} bundle digest"
    ):
        raise Phase4BoundaryError(f"{role} bundle differs from Phase 4 readiness")
    snapshot[str(checksums_path.resolve())] = ledger_sha
    snapshot[str(completed_path.resolve())] = sha256_file(completed_path)
    return snapshot, {
        "directory": str(directory),
        "bundle_sha256": bundle,
        "checksum_ledger_sha256": ledger_sha,
        "primary_ledger_member": required_member,
        "primary_ledger_sha256": normalized[required_member],
        "checksummed_member_count": len(normalized),
    }


def _configured_path(
    data_config: Mapping[str, Any],
    field: str,
    root: Path,
    *,
    directory: bool,
) -> Path:
    value = data_config.get(field)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise Phase4BoundaryError(f"Phase 4 data.{field} must be a non-empty path")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise Phase4BoundaryError(
            f"Phase 4 data.{field} {kind} does not exist: {resolved}"
        )
    return resolved


def _read_exact_csv(path: Path, columns: Sequence[str], role: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise Phase4BoundaryError(f"Cannot parse {role}: {path}") from exc
    if frame.empty or tuple(frame.columns) != tuple(columns) or not frame.columns.is_unique:
        raise Phase4BoundaryError(f"{role} does not have the exact Phase 2 schema")
    return frame


def _join_boundary_rendered_rows(
    boundary: pd.DataFrame,
    rendered: pd.DataFrame,
    *,
    rendered_root: Path,
    expected_resolution: int,
) -> pd.DataFrame:
    selected = rendered.loc[rendered["nodule_id"].isin(boundary["nodule_id"])].copy()
    if len(selected) != len(boundary):
        raise Phase4BoundaryError(
            "Rendered Phase 2 ledger does not contain every boundary nodule exactly once"
        )
    canonical_by_id = boundary.set_index("nodule_id", drop=False)
    for position, row in selected.iterrows():
        nodule_id = _canonical_text(
            row["nodule_id"], f"rendered boundary row {position} nodule_id"
        )
        if nodule_id not in canonical_by_id.index:
            raise Phase4BoundaryError("Rendered boundary membership is inconsistent")
        source = canonical_by_id.loc[nodule_id]
        for field in (
            "nodule_sha256",
            "patient_id",
            "study_instance_uid",
            "series_instance_uid",
            "split",
            "paper_label",
        ):
            actual = _canonical_text(
                row[field], f"rendered boundary row {position} {field}"
            )
            if actual != source[field]:
                raise Phase4BoundaryError(
                    f"Canonical/rendered boundary field {field!r} differs for {nodule_id}"
                )
        if _exact_integer(
            row["n_valid_malignancy_ratings"],
            f"rendered boundary row {position} reader count",
        ) != _exact_integer(
            source["n_valid_malignancy_ratings"],
            f"canonical boundary row {position} reader count",
        ):
            raise Phase4BoundaryError(
                f"Canonical/rendered boundary reader count differs for {nodule_id}"
            )
        if _boolean(row["paper_primary_eligible"], "paper_primary_eligible"):
            raise Phase4BoundaryError(
                f"Rendered boundary row is primary eligible: {nodule_id}"
            )
        if not _boolean(row["boundary_zone"], "boundary_zone"):
            raise Phase4BoundaryError(
                f"Rendered boundary marker is false: {nodule_id}"
            )
        if _exact_integer(
            row["resolution"], f"rendered boundary row {position} resolution"
        ) != expected_resolution:
            raise Phase4BoundaryError(
                f"Rendered boundary resolution differs for {nodule_id}"
            )
        for field in ("image_path", "rendered_mask_path"):
            relative = _canonical_relative_path(
                row[field], f"rendered boundary row {position} {field}"
            )
            target = rendered_root.joinpath(*PurePosixPath(relative).parts).resolve()
            try:
                target.relative_to(rendered_root)
            except ValueError as exc:
                raise Phase4BoundaryError(
                    f"Rendered boundary {field} escapes the rendered root"
                ) from exc
            if not target.is_file():
                raise Phase4BoundaryError(
                    f"Rendered boundary {field} is missing: {relative}"
                )

    projection = selected.loc[
        :, ["nodule_id", "resolution", "image_path", "rendered_mask_path"]
    ]
    try:
        joined = boundary.merge(
            projection,
            on="nodule_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    except (KeyError, pd.errors.MergeError, ValueError) as exc:
        raise Phase4BoundaryError(
            "Boundary-to-rendered linkage is not one-to-one"
        ) from exc
    if joined["nodule_id"].tolist() != boundary["nodule_id"].tolist():
        raise Phase4BoundaryError("Boundary-to-rendered linkage changed source order")
    return joined


def _full_boundary_identities(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = sorted(set(_FULL_BOUNDARY_IDENTITY_FIELDS) - set(frame.columns))
    if missing:
        raise Phase4BoundaryError(
            f"Validated boundary cohort lacks inference identity fields: {missing}"
        )
    identities: list[dict[str, Any]] = []
    for position, row in frame.iterrows():
        identity: dict[str, Any] = {}
        for field in _IDENTITY_FIELDS:
            identity[field] = _canonical_text(
                row[field], f"boundary identity {position} {field}"
            )
        identity["nodule_sha256"] = _sha256_text(
            identity["nodule_sha256"],
            f"boundary identity {position} nodule_sha256",
        )
        if identity["split"] not in _SPLITS:
            raise Phase4BoundaryError(
                f"boundary identity {position} has a noncanonical split"
            )
        for field in ("image_path", "rendered_mask_path"):
            identity[field] = _canonical_relative_path(
                row[field], f"boundary identity {position} {field}"
            )
        score_sum = _exact_integer(
            row["malignancy_score_sum"],
            f"boundary identity {position} malignancy_score_sum",
        )
        readers = _exact_integer(
            row["n_valid_malignancy_ratings"],
            f"boundary identity {position} n_valid_malignancy_ratings",
        )
        mean = _finite_mean(
            row["mean_malignancy"],
            f"boundary identity {position} mean_malignancy",
        )
        label, _eligible, boundary = paper_mean_label(score_sum, readers)
        if (
            label != "boundary_zone"
            or not boundary
            or not math.isclose(mean, score_sum / readers, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise Phase4BoundaryError(
                f"boundary identity {position} violates the exact Phase 2 rule"
            )
        stratum = _canonical_text(
            row["boundary_stratum"],
            f"boundary identity {position} boundary_stratum",
        )
        expected_stratum = (
            "lower_boundary" if score_sum < 3 * readers else "upper_boundary"
        )
        if stratum != expected_stratum:
            raise Phase4BoundaryError(
                f"boundary identity {position} has an invalid boundary stratum"
            )
        identity.update(
            {
                "image_path": _canonical_relative_path(row["image_path"], "image_path"),
                "rendered_mask_path": _canonical_relative_path(
                    row["rendered_mask_path"], "rendered_mask_path"
                ),
                "mean_malignancy": mean,
                "malignancy_score_sum": score_sum,
                "n_valid_malignancy_ratings": readers,
                "boundary_stratum": stratum,
            }
        )
        identities.append(identity)
    if any(len({row[field] for row in identities}) != len(identities)
           for field in ("nodule_id", "nodule_sha256")):
        raise Phase4BoundaryError("Boundary provenance contains duplicate identities")
    return identities


def validate_boundary_provenance_record(
    record: Mapping[str, Any],
    *,
    primary_supervision: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the complete boundary record against the frozen primary cohort."""

    payload = _json_provenance(record)
    required_values = {
        "schema_version": 3,
        "analysis": "phase4_boundary_stress",
        "source_definition": "authoritative_phase2_paper_mean_label",
        "ranges_redefined": False,
        "primary_cohort_excluded": True,
        "primary_overlap_count": 0,
        "primary_headline_inclusion_permitted": False,
        "frozen_final_model_required": True,
        "boundary_driven_tuning_forbidden": True,
        "threshold_or_architecture_selection_permitted": False,
        "evaluated_population": BOUNDARY_EVALUATED_POPULATION,
        "source_validity_note": BOUNDARY_SOURCE_VALIDITY_NOTE,
    }
    for field, expected in required_values.items():
        if type(payload.get(field)) is not type(expected) or payload.get(field) != expected:
            raise Phase4BoundaryError(
                f"Boundary provenance field {field!r} is not frozen"
            )
    if payload.get("boundary_ranges") != [
        dict(value) for value in BOUNDARY_RANGE_RECORDS
    ]:
        raise Phase4BoundaryError("Boundary provenance redefines the Phase 2 ranges")
    if payload.get("target_interpretation") != BOUNDARY_TARGET_INTERPRETATION:
        raise Phase4BoundaryError("Boundary target interpretation is not prespecified")
    if payload.get("limitations") != list(BOUNDARY_LIMITATIONS):
        raise Phase4BoundaryError("Boundary limitations are not prespecified")
    if not isinstance(payload.get("source_provenance"), dict):
        raise Phase4BoundaryError("Boundary provenance lacks Phase 2 source evidence")

    raw_identities = payload.get("ordered_identities")
    if not isinstance(raw_identities, list) or not raw_identities:
        raise Phase4BoundaryError("Boundary provenance lacks ordered identities")
    identity_frame = pd.DataFrame(raw_identities)
    identities = _full_boundary_identities(identity_frame)
    if identities != raw_identities:
        raise Phase4BoundaryError("Boundary ordered identities are not canonical")
    if payload.get("sample_count") != len(identities):
        raise Phase4BoundaryError("Boundary sample count differs from its identities")
    split_counts = {
        split: sum(row["split"] == split for row in identities) for split in _SPLITS
    }
    stratum_counts = {
        stratum: sum(row["boundary_stratum"] == stratum for row in identities)
        for stratum in ("lower_boundary", "upper_boundary")
    }
    if payload.get("split_counts") != split_counts:
        raise Phase4BoundaryError("Boundary split counts are inconsistent")
    if payload.get("stratum_counts") != stratum_counts:
        raise Phase4BoundaryError("Boundary stratum counts are inconsistent")
    logical = sorted(
        identities,
        key=lambda row: (
            row["nodule_sha256"],
            row["nodule_id"],
            row["series_instance_uid"],
            row["split"],
        ),
    )
    if payload.get("boundary_cohort_sha256") != sha256_json(logical):
        raise Phase4BoundaryError("Boundary logical cohort hash is inconsistent")
    if payload.get("ordered_boundary_cohort_sha256") != sha256_json(identities):
        raise Phase4BoundaryError("Boundary ordered cohort hash is inconsistent")

    # Reconcile the NEW population ledger to the same frozen, checksummed
    # canonical bytes. Self-consistent counts/hashes alone cannot authorize a
    # fabricated exclusion or hide a full-membership contract violation.
    source = payload["source_provenance"].get("canonical_artifact")
    if not isinstance(source, dict) or source.get("primary_ledger_member") != "canonical_nodule_ledger.csv":
        raise Phase4BoundaryError("Boundary accounting lacks the canonical ledger binding")
    canonical_path = Path(_canonical_text(source.get("directory"), "canonical directory")) / "canonical_nodule_ledger.csv"
    digest = _sha256_text(source.get("primary_ledger_sha256"), "canonical ledger digest")
    if not canonical_path.is_absolute() or not canonical_path.is_file() or sha256_file(canonical_path) != digest:
        raise Phase4BoundaryError("Boundary accounting canonical ledger changed or is missing")
    binding = payload.get("primary_cohort_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("phase3_input_fingerprint"), dict):
        raise Phase4BoundaryError("Boundary accounting lacks the frozen primary fingerprint")
    checksum_path = canonical_path.parent / "checksums.json"
    expected_checksum = _sha256_text(
        binding["phase3_input_fingerprint"].get("canonical_checksum_ledger_sha256"),
        "frozen canonical checksum ledger digest",
    )
    if not checksum_path.is_file() or sha256_file(checksum_path) != expected_checksum:
        raise Phase4BoundaryError("Boundary accounting checksum ledger differs from the frozen fingerprint")
    checksum_record = _read_json_object(checksum_path, "canonical checksum ledger")
    if checksum_record.get("files", {}).get(canonical_path.name) != digest:
        raise Phase4BoundaryError("Boundary accounting canonical digest differs from the frozen checksum ledger")
    canonical = _read_exact_csv(canonical_path, LEDGER_COLUMNS, "canonical boundary accounting")
    eligible, reconstructed = isolate_phase2_boundary_cohort(
        canonical, primary_nodule_ids=primary_supervision,
        source_provenance=payload["source_provenance"],
    )
    if sha256_file(canonical_path) != digest:
        raise Phase4BoundaryError("Boundary accounting canonical ledger changed during validation")
    if sha256_json(payload.get("source_validity_accounting")) != sha256_json(reconstructed["source_validity_accounting"]):
        raise Phase4BoundaryError("Boundary source-validity accounting differs from the canonical ledger")
    eligible_ids = reconstructed["ordered_identities"]
    projected = [{key: row[key] for key in expected} for row, expected in zip(identities, eligible_ids)]
    if len(identities) != len(eligible) or projected != eligible_ids:
        raise Phase4BoundaryError("Evaluated boundary identities differ from the source-valid subset")

    calculated_binding = _primary_binding(primary_supervision, binding)
    for field in (
        "cohort_view",
        "expected_resolution",
        "sample_count",
        "split_counts",
        "reader_count_counts",
        "manifest_bundle_sha256",
        "supervision_sha256",
        "ordered_supervision_sha256",
        "phase3_input_fingerprint",
        "identity_fields",
        "logical_primary_identity_sha256",
        "ordered_primary_identity_sha256",
    ):
        if calculated_binding.get(field) != binding.get(field):
            raise Phase4BoundaryError(
                f"Boundary primary cohort binding {field!r} is inconsistent"
            )
    primary_ids = set(primary_supervision["nodule_id"].tolist())
    overlap = sorted({row["nodule_id"] for row in identities} & primary_ids)
    if overlap:
        raise Phase4BoundaryError(
            f"Boundary provenance overlaps the frozen primary cohort: {overlap}"
        )
    snapshot_sha = payload.get("source_snapshot_sha256")
    if snapshot_sha is not None:
        _sha256_text(snapshot_sha, "boundary source snapshot digest")
    return payload


def assert_boundary_source_snapshot_unchanged(
    snapshot: Mapping[str, str],
) -> None:
    """Fail when any readiness-gated source file changed after construction."""

    if not isinstance(snapshot, Mapping) or not snapshot:
        raise Phase4BoundaryError("Boundary source snapshot must be non-empty")
    for raw_path, raw_digest in snapshot.items():
        path_text = _canonical_text(raw_path, "boundary source snapshot path")
        expected = _sha256_text(
            raw_digest, f"boundary source snapshot digest for {path_text}"
        )
        path = Path(path_text)
        if not path.is_absolute() or not path.is_file():
            raise Phase4BoundaryError(
                f"Boundary source snapshot member is missing: {path_text}"
            )
        if sha256_file(path) != expected:
            raise Phase4BoundaryError(
                f"Boundary source changed after validation: {path_text}"
            )


def _add_snapshot_path(snapshot: dict[str, str], path: Path, role: str) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise Phase4BoundaryError(f"Boundary {role} does not exist: {resolved}")
    key = str(resolved)
    digest = sha256_file(resolved)
    prior = snapshot.get(key)
    if prior is not None and prior != digest:
        raise Phase4BoundaryError(f"Boundary {role} snapshot digest is inconsistent")
    snapshot[key] = digest


def load_validated_phase2_boundary_source(
    data_config: Mapping[str, Any],
    *,
    manifest: ManifestGuardReport,
    input_lock: Phase4InputLock,
    expected_resolution: int,
    project_root: str | Path,
    additional_snapshot_paths: Iterable[str | Path] = (),
) -> Phase4BoundarySource:
    """Load boundary images only after the full real Phase 2/3/4 trust chain."""

    if not isinstance(data_config, Mapping):
        raise TypeError("data_config must be a mapping")
    if data_config.get("mode") != "real":
        raise Phase4BoundaryError("Boundary evaluation requires real Phase 2 data")
    if data_config.get("cohort_view") != PRIMARY_COHORT_VIEW:
        raise Phase4BoundaryError("Boundary evaluation requires the frozen primary view")
    if type(expected_resolution) is not int or expected_resolution != FROZEN_BOUNDARY_RESOLUTION:
        raise Phase4BoundaryError("Boundary evaluation resolution must be exactly 64")
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise Phase4BoundaryError(f"project_root does not exist: {root}")

    try:
        readiness, primary = validate_phase4_readiness(
            data_config,
            manifest=manifest,
            expected_resolution=expected_resolution,
            project_root=root,
            synthetic_dry_run=False,
        )
        validate_phase4_input_lock(input_lock, readiness)
    except Exception as exc:
        raise Phase4BoundaryError(
            f"Phase 4 readiness/input-lock validation failed: {exc}"
        ) from exc
    readiness_payload = _report_payload(readiness, "Phase 4 readiness report")
    for field in (
        "ready_for_requested_mode",
        "real_research_ready",
        "phase3_readiness_validated",
        "multirater_supervision_validated",
    ):
        if readiness_payload.get(field) is not True:
            raise Phase4BoundaryError(
                f"Phase 4 readiness field {field!r} was not established"
            )
    if (
        readiness_payload.get("cohort_view") != PRIMARY_COHORT_VIEW
        or readiness_payload.get("expected_resolution") != expected_resolution
    ):
        raise Phase4BoundaryError("Phase 4 readiness differs from the boundary contract")
    primary_binding = _primary_binding(primary, readiness_payload)

    canonical_dir = _configured_path(
        data_config, "canonical_dir", root, directory=True
    )
    rendered_root = _configured_path(
        data_config, "rendered_dir", root, directory=True
    )
    metadata_path = _configured_path(
        data_config, "metadata_path", root, directory=False
    )
    source_readiness_path = _configured_path(
        data_config, "source_readiness_report", root, directory=False
    )
    canonical_path = canonical_dir / "canonical_nodule_ledger.csv"
    rendered_path = rendered_root / "rendered_rois.csv"
    canonical = _read_exact_csv(
        canonical_path, LEDGER_COLUMNS, "canonical Phase 2 ledger"
    )
    rendered = _read_exact_csv(
        rendered_path, RENDERED_COLUMNS, "rendered Phase 2 ledger"
    )

    fingerprint = readiness_payload.get("phase3_input_fingerprint")
    if not isinstance(fingerprint, dict):
        raise Phase4BoundaryError("Phase 4 readiness lacks Phase 3 fingerprints")
    canonical_snapshot, canonical_source = _completed_phase2_snapshot(
        canonical_dir,
        role="canonical Phase 2 artifact",
        expected_bundle_sha256=fingerprint.get("canonical_bundle_sha256"),
        expected_checksum_ledger_sha256=fingerprint.get(
            "canonical_checksum_ledger_sha256"
        ),
        required_member="canonical_nodule_ledger.csv",
    )
    rendered_snapshot, rendered_source = _completed_phase2_snapshot(
        rendered_root,
        role="rendered Phase 2 artifact",
        expected_bundle_sha256=fingerprint.get("rendered_bundle_sha256"),
        expected_checksum_ledger_sha256=fingerprint.get(
            "rendered_checksum_ledger_sha256"
        ),
        required_member="rendered_rois.csv",
    )
    if sha256_file(metadata_path) != _sha256_text(
        fingerprint.get("metadata_sha256"), "Phase 3 metadata digest"
    ):
        raise Phase4BoundaryError("Phase 3 metadata changed after readiness")
    if sha256_file(source_readiness_path) != _sha256_text(
        fingerprint.get("source_readiness_sha256"),
        "Phase 2 source-readiness digest",
    ):
        raise Phase4BoundaryError("Phase 2 source-readiness report changed")

    source_provenance = {
        "phase": 2,
        "validation_chain": "phase3_readiness_then_phase4_readiness_then_input_lock",
        "canonical_artifact": canonical_source,
        "rendered_artifact": rendered_source,
        "source_readiness_path": str(source_readiness_path),
        "source_readiness_sha256": fingerprint["source_readiness_sha256"],
        "phase3_metadata_path": str(metadata_path),
        "phase3_metadata_sha256": fingerprint["metadata_sha256"],
        "phase4_readiness": readiness_payload,
        "phase4_input_lock": _report_payload(input_lock, "Phase 4 input lock"),
    }
    boundary, record = isolate_phase2_boundary_cohort(
        canonical,
        primary_nodule_ids=primary,
        source_provenance=source_provenance,
    )
    boundary = _join_boundary_rendered_rows(
        boundary,
        rendered,
        rendered_root=rendered_root,
        expected_resolution=expected_resolution,
    )
    identities = _full_boundary_identities(boundary)
    logical = sorted(
        identities,
        key=lambda row: (
            row["nodule_sha256"],
            row["nodule_id"],
            row["series_instance_uid"],
            row["split"],
        ),
    )
    record.update(
        {
            "schema_version": 3,
            "ordered_identities": identities,
            "boundary_cohort_sha256": sha256_json(logical),
            "ordered_boundary_cohort_sha256": sha256_json(identities),
            "primary_cohort_binding": primary_binding,
        }
    )

    snapshot = {**canonical_snapshot, **rendered_snapshot}
    _add_snapshot_path(snapshot, metadata_path, "Phase 3 metadata")
    _add_snapshot_path(snapshot, source_readiness_path, "source-readiness report")
    manifest_dir = Path(getattr(manifest, "manifest_dir", "")).resolve()
    for name in (*FROZEN_MANIFEST_FILES, CHECKSUM_LEDGER):
        _add_snapshot_path(snapshot, manifest_dir / name, f"Phase 0 manifest {name}")
    for raw_path in additional_snapshot_paths:
        _add_snapshot_path(snapshot, Path(raw_path), "configuration source")
    assert_boundary_source_snapshot_unchanged(snapshot)
    record["source_snapshot_sha256"] = sha256_json(dict(sorted(snapshot.items())))
    validated_record = validate_boundary_provenance_record(
        record, primary_supervision=primary
    )
    assert_boundary_source_snapshot_unchanged(snapshot)
    return Phase4BoundarySource(
        frame=boundary.reset_index(drop=True),
        provenance=validated_record,
        source_snapshot=dict(sorted(snapshot.items())),
        rendered_root=rendered_root,
        readiness_report=readiness,
    )


def load_repository_phase4_boundary_source(
    *,
    config_path: str | Path = DEFAULT_PHASE4_CONFIG,
    input_lock_path: str | Path = DEFAULT_PHASE4_INPUT_LOCK,
    project_root: str | Path = PROJECT_ROOT,
) -> Phase4BoundarySource:
    """Load the repository's canonical, locked real-data boundary cohort."""

    root = Path(project_root).resolve()
    canonical_config = (root / "configs" / "phase4_consensdxformer.yaml").resolve()
    canonical_lock = (root / "configs" / "phase4_input_lock.json").resolve()
    requested_config = Path(config_path).resolve()
    requested_lock = Path(input_lock_path).resolve()
    if requested_config != canonical_config or requested_lock != canonical_lock:
        raise Phase4BoundaryError(
            "Boundary source requires the repository's canonical Phase 4 config and lock"
        )
    try:
        bundle = load_phase4_config(requested_config)
        lock = load_phase4_input_lock(requested_lock)
        data_config = bundle.phase4_config["data"]
        manifest_value = Path(data_config["frozen_split_dir"])
        manifest_dir = (
            manifest_value.resolve()
            if manifest_value.is_absolute()
            else (root / manifest_value).resolve()
        )
        manifest = validate_frozen_manifest_bundle(manifest_dir)
    except Exception as exc:
        raise Phase4BoundaryError(
            f"Cannot load canonical Phase 4 boundary inputs: {exc}"
        ) from exc
    return load_validated_phase2_boundary_source(
        data_config,
        manifest=manifest,
        input_lock=lock,
        expected_resolution=int(bundle.phase4_config["dataset"]["image_size"]),
        project_root=root,
        additional_snapshot_paths=(
            requested_config,
            requested_lock,
            bundle.b1_anchor_path,
        ),
    )


# Explicit alias for callers that use construction terminology.
construct_boundary_cohort = isolate_phase2_boundary_cohort


__all__ = [
    "BOUNDARY_EVALUATED_POPULATION",
    "BOUNDARY_EXCLUSION_FIELDS",
    "BOUNDARY_SOURCE_VALIDITY_NOTE",
    "BOUNDARY_LIMITATIONS",
    "BOUNDARY_RANGES",
    "BOUNDARY_RANGE_RECORDS",
    "BOUNDARY_SOURCE_VALIDITY_STATUSES",
    "BOUNDARY_TARGET_INTERPRETATION",
    "DEFAULT_PHASE4_CONFIG",
    "DEFAULT_PHASE4_INPUT_LOCK",
    "FROZEN_BOUNDARY_RESOLUTION",
    "PROJECT_ROOT",
    "Phase4BoundarySource",
    "Phase4BoundaryError",
    "assert_boundary_source_snapshot_unchanged",
    "construct_boundary_cohort",
    "isolate_phase2_boundary_cohort",
    "load_repository_phase4_boundary_source",
    "load_validated_phase2_boundary_source",
    "validate_boundary_provenance_record",
]
