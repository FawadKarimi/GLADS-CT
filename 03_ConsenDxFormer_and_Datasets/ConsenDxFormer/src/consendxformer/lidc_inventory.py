"""Inventory-only aggregation over provider-neutral physical nodule clusters."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Iterable

import pandas as pd

from .lidc_ledger import analyze_cluster
from .lidc_records import LIDCProvider, ScanRecord
from .lidc_reconciliation import (
    FrozenSeriesIdentity,
    ReconciliationError,
    frozen_series_index,
    reconcile_scan_against_index,
)
from .manifest_guard import ManifestGuardReport
from .phase2_config import ClusteringConfig


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def inventory_lidc(
    provider: LIDCProvider,
    scans: Iterable[ScanRecord],
    manifest: ManifestGuardReport,
    clustering: ClusteringConfig,
) -> tuple[dict, pd.DataFrame]:
    frozen = frozen_series_index(manifest)
    rows: list[dict] = []
    annotations_total = 0
    missing_ratings = 0
    malformed_annotations = 0
    clusters_by_annotations: Counter[int] = Counter()
    clusters_by_valid_ratings: Counter[int] = Counter()
    mean_distribution: Counter[str] = Counter()
    label_distribution: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()

    for scan in scans:
        try:
            identity: FrozenSeriesIdentity | None = reconcile_scan_against_index(scan, frozen)
            reconciliation_error = None
        except ReconciliationError as exc:
            identity = None
            reconciliation_error = str(exc)
        for cluster in provider.cluster_annotations(scan, clustering):
            analysis = analyze_cluster(cluster)
            n_annotations = len(cluster.annotations)
            n_valid = len(analysis.valid_ratings)
            annotations_total += n_annotations
            missing_ratings += analysis.missing_rating_count
            malformed_annotations += len(analysis.geometry_reasons) + len(
                analysis.invalid_rating_reasons
            )
            clusters_by_annotations[n_annotations] += 1
            clusters_by_valid_ratings[n_valid] += 1
            if analysis.targets is not None:
                fraction = Fraction(
                    analysis.targets.score_sum, len(analysis.valid_ratings)
                )
                mean_distribution[f"{fraction.numerator}/{fraction.denominator}"] += 1
                label_distribution[analysis.targets.paper_label] += 1
            else:
                label_distribution["unrated"] += 1
            reasons: list[str] = []
            if reconciliation_error:
                reasons.append("identity_reconciliation")
            if analysis.geometry_reasons:
                reasons.append("malformed_geometry")
            if analysis.invalid_rating_reasons:
                reasons.append("invalid_rating")
            if analysis.targets is None:
                reasons.append("no_valid_rating")
            for reason in reasons:
                exclusion_reasons[reason] += 1
            rows.append(
                {
                    "nodule_id": analysis.identity.nodule_id,
                    "nodule_sha256": analysis.identity.nodule_sha256,
                    "patient_id": scan.patient_id,
                    "study_instance_uid": scan.study_instance_uid,
                    "series_instance_uid": scan.series_instance_uid,
                    "split": identity.split if identity is not None else "",
                    "n_annotations": n_annotations,
                    "n_valid_malignancy_ratings": n_valid,
                    "mean_malignancy": (
                        analysis.targets.mean_malignancy if analysis.targets else ""
                    ),
                    "paper_label": analysis.targets.paper_label if analysis.targets else "unrated",
                    "boundary_zone": bool(
                        analysis.targets and analysis.targets.boundary_zone
                    ),
                    "has_multiple_readers": n_valid >= 2,
                    "has_four_valid_readers": n_valid == 4,
                    "paper_strict_four_reader_candidate": bool(
                        analysis.targets
                        and analysis.targets.paper_primary_eligible
                        and n_valid == 4
                        and not reasons
                    ),
                    "source_issues": ";".join(reasons),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("nodule_sha256", kind="stable").reset_index(drop=True)
    summary = {
        "schema_version": 1,
        "provider": provider.name,
        "physical_nodule_cluster_count": len(rows),
        "annotation_count": annotations_total,
        "clusters_by_number_of_annotations": _counter_dict(clusters_by_annotations),
        "clusters_by_number_of_valid_malignancy_ratings": _counter_dict(
            clusters_by_valid_ratings
        ),
        "mean_score_fraction_distribution": _counter_dict(mean_distribution),
        "paper_label_distribution": _counter_dict(label_distribution),
        "boundary_zone_count": int(label_distribution["boundary_zone"]),
        "four_reader_candidate_count": int(
            frame["paper_strict_four_reader_candidate"].sum()
        )
        if not frame.empty
        else 0,
        "multi_reader_candidate_count": int(frame["has_multiple_readers"].sum())
        if not frame.empty
        else 0,
        "missing_malignancy_rating_count": missing_ratings,
        "malformed_annotation_value_count": malformed_annotations,
        "exclusion_rejection_reason_counts": _counter_dict(exclusion_reasons),
        "paper_cohort_size_4095_enforced": False,
    }
    return summary, frame
