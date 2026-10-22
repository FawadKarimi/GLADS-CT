"""Canonical one-row-per-physical-nodule ledger construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .lidc_masks import RepresentativeMaskRecord
from .lidc_records import AnnotationRecord, NoduleClusterRecord
from .lidc_reconciliation import FrozenSeriesIdentity
from .lidc_targets import (
    AnnotationValidationError,
    RatingTargets,
    StableNoduleIdentity,
    annotation_geometry_issue,
    canonical_annotation_order_preserving_malformed,
    compute_rating_targets,
    diagnostic_json_value,
    stable_nodule_identity_preserving_malformed,
    validate_malignancy_rating,
)


LEDGER_COLUMNS = (
    "nodule_id",
    "nodule_sha256",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "split",
    "representative_slice_index",
    "representative_sop_instance_uid",
    "representative_z",
    "representative_slice_rule",
    "n_annotations",
    "n_valid_malignancy_ratings",
    "n_missing_malignancy_ratings",
    "mean_malignancy",
    "malignancy_score_sum",
    "raw_malignancy_ratings_json",
    "malignancy_ratings_json",
    "individual_classes_json",
    "q_benign",
    "q_indeterminate",
    "q_malignant",
    "disagreement_entropy",
    "ordinal_t1",
    "ordinal_t2",
    "ordinal_t3",
    "ordinal_t4",
    "paper_label",
    "paper_primary_eligible",
    "boundary_zone",
    "has_multiple_readers",
    "has_four_valid_readers",
    "paper_strict_four_reader_candidate",
    "local_annotation_ids_json",
    "annotation_sha256s_json",
    "annotation_source_status_json",
    "annotation_source_errors_json",
    "mask_path",
    "mask_sha256",
    "source_validity_status",
    "exclusion_reason_json",
)


@dataclass(frozen=True)
class ClusterAnalysis:
    identity: StableNoduleIdentity
    ordered_annotations: tuple[tuple[str, AnnotationRecord, str | None], ...]
    valid_ratings: tuple[int, ...]
    missing_rating_count: int
    invalid_rating_reasons: tuple[str, ...]
    geometry_reasons: tuple[str, ...]
    targets: RatingTargets | None

    @property
    def local_annotation_ids(self) -> tuple[str, ...]:
        return tuple(
            f"local-{index + 1:02d}-{signature[:10]}"
            for index, (signature, _annotation, _issue) in enumerate(
                self.ordered_annotations
            )
        )


def analyze_cluster(cluster: NoduleClusterRecord) -> ClusterAnalysis:
    identity, _identity_issues = stable_nodule_identity_preserving_malformed(cluster)
    ordered = canonical_annotation_order_preserving_malformed(cluster)
    valid: list[int] = []
    missing = 0
    invalid: list[str] = []
    geometry: list[str] = []
    for signature, annotation, _issue in ordered:
        geometry_issue = annotation_geometry_issue(annotation, cluster)
        if geometry_issue is not None and geometry_issue not in geometry:
            geometry.append(geometry_issue)
        try:
            rating = validate_malignancy_rating(annotation.malignancy)
        except AnnotationValidationError as exc:
            invalid.append(f"{signature}:{exc}")
            continue
        if rating is None:
            missing += 1
        else:
            valid.append(rating)
    targets = compute_rating_targets(valid) if valid else None
    return ClusterAnalysis(
        identity=identity,
        ordered_annotations=ordered,
        valid_ratings=tuple(valid),
        missing_rating_count=missing,
        invalid_rating_reasons=tuple(sorted(invalid)),
        geometry_reasons=tuple(sorted(set(geometry))),
        targets=targets,
    )


def reordered_cluster(cluster: NoduleClusterRecord, analysis: ClusterAnalysis) -> NoduleClusterRecord:
    return NoduleClusterRecord(
        scan=cluster.scan,
        annotations=tuple(annotation for _signature, annotation, _issue in analysis.ordered_annotations),
    )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_ledger_row(
    cluster: NoduleClusterRecord,
    analysis: ClusterAnalysis,
    *,
    frozen_identity: FrozenSeriesIdentity | None,
    reconciliation_error: str | None,
    representative_masks: RepresentativeMaskRecord | None,
    mask_path: str | None,
    mask_sha256: str | None,
    mask_error: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if reconciliation_error:
        reasons.append(reconciliation_error)
    reasons.extend(f"geometry:{reason}" for reason in analysis.geometry_reasons)
    reasons.extend(f"rating:{reason}" for reason in analysis.invalid_rating_reasons)
    if analysis.targets is None:
        reasons.append("no_valid_malignancy_rating")
    if mask_error:
        reasons.append(f"mask:{mask_error}")

    if reconciliation_error:
        status = "excluded_identity_reconciliation"
    elif analysis.geometry_reasons:
        status = "excluded_malformed_geometry"
    elif analysis.invalid_rating_reasons:
        status = "excluded_invalid_rating"
    elif analysis.targets is None:
        status = "excluded_no_valid_rating"
    elif representative_masks is None:
        status = "excluded_mask_unavailable"
    elif analysis.missing_rating_count:
        status = "valid_with_missing_ratings"
    else:
        status = "valid"

    source_valid = status in {"valid", "valid_with_missing_ratings"}
    targets = analysis.targets
    paper_eligible = bool(
        source_valid and targets is not None and targets.paper_primary_eligible
    )
    n_valid = len(analysis.valid_ratings)
    row: dict[str, Any] = {
        "nodule_id": analysis.identity.nodule_id,
        "nodule_sha256": analysis.identity.nodule_sha256,
        "patient_id": cluster.scan.patient_id,
        "study_instance_uid": cluster.scan.study_instance_uid,
        "series_instance_uid": cluster.scan.series_instance_uid,
        "split": frozen_identity.split if frozen_identity is not None else "",
        "representative_slice_index": (
            representative_masks.representative_slice_index
            if representative_masks is not None
            else ""
        ),
        "representative_sop_instance_uid": (
            representative_masks.representative_sop_instance_uid or ""
            if representative_masks is not None
            else ""
        ),
        "representative_z": (
            representative_masks.representative_z
            if representative_masks is not None
            and representative_masks.representative_z is not None
            else ""
        ),
        "representative_slice_rule": (
            representative_masks.selection_rule if representative_masks is not None else ""
        ),
        "n_annotations": len(cluster.annotations),
        "n_valid_malignancy_ratings": n_valid,
        "n_missing_malignancy_ratings": analysis.missing_rating_count,
        "mean_malignancy": targets.mean_malignancy if targets is not None else "",
        "malignancy_score_sum": targets.score_sum if targets is not None else "",
        "raw_malignancy_ratings_json": _json(
            [
                diagnostic_json_value(annotation.malignancy)
                for _signature, annotation, _issue in analysis.ordered_annotations
            ]
        ),
        "malignancy_ratings_json": _json(list(analysis.valid_ratings)),
        "individual_classes_json": _json(
            list(targets.individual_classes) if targets is not None else []
        ),
        "q_benign": targets.q_benign if targets is not None else "",
        "q_indeterminate": targets.q_indeterminate if targets is not None else "",
        "q_malignant": targets.q_malignant if targets is not None else "",
        "disagreement_entropy": (
            targets.disagreement_entropy if targets is not None else ""
        ),
        "ordinal_t1": targets.ordinal_t1 if targets is not None else "",
        "ordinal_t2": targets.ordinal_t2 if targets is not None else "",
        "ordinal_t3": targets.ordinal_t3 if targets is not None else "",
        "ordinal_t4": targets.ordinal_t4 if targets is not None else "",
        "paper_label": targets.paper_label if targets is not None else "unrated",
        "paper_primary_eligible": paper_eligible,
        "boundary_zone": bool(targets.boundary_zone) if targets is not None else False,
        "has_multiple_readers": n_valid >= 2,
        "has_four_valid_readers": n_valid == 4,
        "paper_strict_four_reader_candidate": paper_eligible and n_valid == 4,
        "local_annotation_ids_json": _json(list(analysis.local_annotation_ids)),
        "annotation_sha256s_json": _json(list(analysis.identity.annotation_sha256s)),
        "annotation_source_status_json": _json(
            [
                annotation.source_status
                for _signature, annotation, _issue in analysis.ordered_annotations
            ]
        ),
        "annotation_source_errors_json": _json(
            [
                annotation.source_error
                for _signature, annotation, _issue in analysis.ordered_annotations
            ]
        ),
        "mask_path": mask_path or "",
        "mask_sha256": mask_sha256 or "",
        "source_validity_status": status,
        "exclusion_reason_json": _json(sorted(set(reasons))),
    }
    return {column: row[column] for column in LEDGER_COLUMNS}
