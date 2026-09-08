"""Stable physical-nodule identity and multi-rater diagnostic targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import numbers
from typing import Any, Iterable, Sequence

from .hashing import sha256_json
from .lidc_records import AnnotationRecord, ContourRecord, NoduleClusterRecord


CLASS_NAMES = ("benign", "indeterminate", "malignant")
IDENTITY_SCHEMA = "lidc_annotation_geometry_signature_v1"


class AnnotationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StableNoduleIdentity:
    nodule_id: str
    nodule_sha256: str
    annotation_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class RatingTargets:
    malignancy_ratings: tuple[int, ...]
    score_sum: int
    mean_malignancy: float
    individual_classes: tuple[str, ...]
    q_benign: float
    q_indeterminate: float
    q_malignant: float
    disagreement_entropy: float
    ordinal_t1: float
    ordinal_t2: float
    ordinal_t3: float
    ordinal_t4: float
    paper_label: str
    paper_primary_eligible: bool
    boundary_zone: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["malignancy_ratings"] = list(self.malignancy_ratings)
        value["individual_classes"] = list(self.individual_classes)
        return value


def _safe_json_value(value: Any) -> Any:
    """Create a deterministic diagnostic token for malformed source values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return format(value, ".17g")
    if isinstance(value, (tuple, list)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _safe_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return {"type": type(value).__name__, "text": str(value)}


def diagnostic_json_value(value: Any) -> Any:
    """Public stable JSON conversion for preserving malformed source values."""
    return _safe_json_value(value)


def validate_malignancy_rating(value: Any) -> int | None:
    """Return an integer score, preserve missing as None, reject malformed scores."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise AnnotationValidationError(
            f"Malignancy rating must be an integer in 1..5 or missing; got {value!r}"
        )
    score = int(value)
    if score not in {1, 2, 3, 4, 5}:
        raise AnnotationValidationError(
            f"Malignancy rating must be in 1..5; got {score}"
        )
    return score


def score_to_class(score: int) -> str:
    value = validate_malignancy_rating(score)
    if value is None:
        raise AnnotationValidationError("A missing rating has no diagnostic class")
    if value <= 2:
        return "benign"
    if value == 3:
        return "indeterminate"
    return "malignant"


def _float_token(value: Any, role: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{role} must be numeric; got {value!r}") from exc
    if not math.isfinite(numeric):
        raise AnnotationValidationError(f"{role} must be finite; got {value!r}")
    normalized = 0.0 if numeric == 0 else numeric
    return format(normalized, ".6f")


def _canonical_polygon(points: Sequence[Sequence[Any]]) -> list[list[str]]:
    if len(points) < 3:
        raise AnnotationValidationError("Contour must contain at least three points")
    normalized = [(_float_token(point[0], "contour row"), _float_token(point[1], "contour column")) for point in points]
    if len(normalized) >= 4 and normalized[0] == normalized[-1]:
        normalized.pop()
    if len(normalized) < 3 or len(set(normalized)) < 3:
        raise AnnotationValidationError("Contour must contain three distinct positions")

    def rotations(values: list[tuple[str, str]]) -> Iterable[tuple[tuple[str, str], ...]]:
        for index in range(len(values)):
            yield tuple(values[index:] + values[:index])

    forward = min(rotations(normalized))
    reverse = min(rotations(list(reversed(normalized))))
    canonical = min(forward, reverse)
    return [[row, column] for row, column in canonical]


def canonical_contour_payload(contour: ContourRecord) -> dict[str, Any]:
    if contour.slice_index is None or isinstance(contour.slice_index, bool):
        raise AnnotationValidationError("Contour slice_index must be an integer")
    if not isinstance(contour.slice_index, numbers.Integral) or int(contour.slice_index) < 0:
        raise AnnotationValidationError("Contour slice_index must be a non-negative integer")
    sop = None if contour.sop_instance_uid is None else str(contour.sop_instance_uid).strip()
    if sop == "":
        raise AnnotationValidationError("Contour SOPInstanceUID cannot be blank")
    z_token = None if contour.z_position is None else _float_token(contour.z_position, "z position")
    return {
        "slice_index": int(contour.slice_index),
        "sop_instance_uid": sop,
        "z_position": z_token,
        "inclusion": bool(contour.inclusion),
        "points": _canonical_polygon(contour.points),
    }


def canonical_annotation_payload(
    annotation: AnnotationRecord,
    *,
    patient_id: str,
    study_instance_uid: str,
    series_instance_uid: str,
) -> dict[str, Any]:
    if annotation.source_status != "available" or annotation.source_error is not None:
        raise AnnotationValidationError(
            "Annotation source is marked "
            f"{annotation.source_status!r}: {annotation.source_error or 'unspecified source error'}"
        )
    identity = {
        "patient_id": str(patient_id).strip(),
        "study_instance_uid": str(study_instance_uid).strip(),
        "series_instance_uid": str(series_instance_uid).strip(),
    }
    if any(not value for value in identity.values()):
        raise AnnotationValidationError("Patient/study/series identifiers must be non-blank")
    rating = validate_malignancy_rating(annotation.malignancy)
    contours = [canonical_contour_payload(contour) for contour in annotation.contours]
    if not contours:
        raise AnnotationValidationError("Annotation contains no contours")
    contours.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    return {
        "schema": IDENTITY_SCHEMA,
        **identity,
        "malignancy": rating,
        "contours": contours,
    }


def annotation_sha256(annotation: AnnotationRecord, cluster: NoduleClusterRecord) -> str:
    scan = cluster.scan
    return sha256_json(
        canonical_annotation_payload(
            annotation,
            patient_id=scan.patient_id,
            study_instance_uid=scan.study_instance_uid,
            series_instance_uid=scan.series_instance_uid,
        )
    )


def annotation_geometry_issue(
    annotation: AnnotationRecord, cluster: NoduleClusterRecord
) -> str | None:
    """Validate source/identity/geometry separately from the diagnostic rating."""
    geometry_only = AnnotationRecord(
        malignancy=None,
        contours=annotation.contours,
        source_status=annotation.source_status,
        source_error=annotation.source_error,
        provider_payload=annotation.provider_payload,
    )
    try:
        canonical_annotation_payload(
            geometry_only,
            patient_id=cluster.scan.patient_id,
            study_instance_uid=cluster.scan.study_instance_uid,
            series_instance_uid=cluster.scan.series_instance_uid,
        )
    except AnnotationValidationError as exc:
        return str(exc)
    return None


def annotation_signature_preserving_malformed(
    annotation: AnnotationRecord, cluster: NoduleClusterRecord
) -> tuple[str, str | None]:
    """Hash valid canonical geometry, or a stable diagnostic fallback for bad input."""
    try:
        return annotation_sha256(annotation, cluster), None
    except AnnotationValidationError as exc:
        payload = {
            "schema": "lidc_malformed_annotation_signature_v1",
            "patient_id": cluster.scan.patient_id,
            "study_instance_uid": cluster.scan.study_instance_uid,
            "series_instance_uid": cluster.scan.series_instance_uid,
            "malignancy": _safe_json_value(annotation.malignancy),
            "source_status": annotation.source_status,
            "source_error": annotation.source_error,
            "contours": [
                {
                    "slice_index": _safe_json_value(contour.slice_index),
                    "sop_instance_uid": _safe_json_value(contour.sop_instance_uid),
                    "z_position": _safe_json_value(contour.z_position),
                    "inclusion": _safe_json_value(contour.inclusion),
                    "points": _safe_json_value(contour.points),
                }
                for contour in annotation.contours
            ],
            "validation_error": str(exc),
        }
        return sha256_json(payload), str(exc)


def stable_nodule_identity(cluster: NoduleClusterRecord) -> StableNoduleIdentity:
    if not cluster.annotations:
        raise AnnotationValidationError("Physical nodule cluster contains no annotations")
    signatures = tuple(sorted(annotation_sha256(annotation, cluster) for annotation in cluster.annotations))
    payload = {
        "schema": "lidc_physical_nodule_signature_v1",
        "patient_id": cluster.scan.patient_id,
        "study_instance_uid": cluster.scan.study_instance_uid,
        "series_instance_uid": cluster.scan.series_instance_uid,
        "annotation_sha256s": list(signatures),
    }
    digest = sha256_json(payload)
    human = f"{cluster.scan.patient_id}__nodule-{digest[:16]}"
    return StableNoduleIdentity(human, digest, signatures)


def stable_nodule_identity_preserving_malformed(
    cluster: NoduleClusterRecord,
) -> tuple[StableNoduleIdentity, tuple[str, ...]]:
    if not cluster.annotations:
        raise AnnotationValidationError("Physical nodule cluster contains no annotations")
    pairs = [
        annotation_signature_preserving_malformed(annotation, cluster)
        for annotation in cluster.annotations
    ]
    signatures = tuple(sorted(signature for signature, _ in pairs))
    issues = tuple(sorted(issue for _, issue in pairs if issue is not None))
    payload = {
        "schema": "lidc_physical_nodule_signature_v1",
        "patient_id": cluster.scan.patient_id,
        "study_instance_uid": cluster.scan.study_instance_uid,
        "series_instance_uid": cluster.scan.series_instance_uid,
        "annotation_sha256s": list(signatures),
    }
    digest = sha256_json(payload)
    return (
        StableNoduleIdentity(
            f"{cluster.scan.patient_id}__nodule-{digest[:16]}", digest, signatures
        ),
        issues,
    )


def canonical_annotation_order(
    cluster: NoduleClusterRecord,
) -> tuple[tuple[str, AnnotationRecord], ...]:
    pairs = [(annotation_sha256(annotation, cluster), annotation) for annotation in cluster.annotations]
    return tuple(sorted(pairs, key=lambda item: item[0]))


def canonical_annotation_order_preserving_malformed(
    cluster: NoduleClusterRecord,
) -> tuple[tuple[str, AnnotationRecord, str | None], ...]:
    pairs = [
        (*annotation_signature_preserving_malformed(annotation, cluster), annotation)
        for annotation in cluster.annotations
    ]
    # Reorder tuple fields to (signature, annotation, issue) for callers.
    return tuple(
        (signature, annotation, issue)
        for signature, issue, annotation in sorted(pairs, key=lambda item: item[0])
    )


def paper_mean_label(score_sum: int, n_ratings: int) -> tuple[str, bool, bool]:
    """Apply the paper wording with integer arithmetic at 2.5/3.0/3.5."""
    if n_ratings <= 0:
        raise ValueError("At least one valid rating is required")
    if 2 * score_sum < 5 * n_ratings:
        return "benign", True, False
    if score_sum == 3 * n_ratings:
        return "indeterminate", True, False
    if 2 * score_sum > 7 * n_ratings:
        return "malignant", True, False
    return "boundary_zone", False, True


def compute_rating_targets(ratings: Sequence[Any]) -> RatingTargets:
    validated = [validate_malignancy_rating(value) for value in ratings]
    if any(value is None for value in validated):
        raise AnnotationValidationError(
            "compute_rating_targets accepts valid ratings only; filter missing values explicitly"
        )
    scores = tuple(int(value) for value in validated if value is not None)
    if not scores:
        raise AnnotationValidationError("At least one valid malignancy rating is required")
    classes = tuple(score_to_class(score) for score in scores)
    counts = tuple(classes.count(name) for name in CLASS_NAMES)
    total = len(scores)
    q = tuple(count / total for count in counts)
    if not math.isclose(sum(q), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Diagnostic class distribution does not sum to one")
    entropy = -sum(probability * math.log(probability) for probability in q if probability > 0)
    normalized_entropy = entropy / math.log(3.0)
    normalized_entropy = min(1.0, max(0.0, normalized_entropy))
    ordinal = tuple(sum(score > threshold for score in scores) / total for threshold in range(1, 5))
    if not ordinal[0] >= ordinal[1] >= ordinal[2] >= ordinal[3]:
        raise AssertionError("Ordinal targets are not monotone")
    score_sum = sum(scores)
    paper_label, eligible, boundary = paper_mean_label(score_sum, total)
    return RatingTargets(
        malignancy_ratings=scores,
        score_sum=score_sum,
        mean_malignancy=score_sum / total,
        individual_classes=classes,
        q_benign=q[0],
        q_indeterminate=q[1],
        q_malignant=q[2],
        disagreement_entropy=normalized_entropy,
        ordinal_t1=ordinal[0],
        ordinal_t2=ordinal[1],
        ordinal_t3=ordinal[2],
        ordinal_t4=ordinal[3],
        paper_label=paper_label,
        paper_primary_eligible=eligible,
        boundary_zone=boundary,
    )
