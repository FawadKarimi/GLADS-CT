from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from consendxformer.lidc_ledger import analyze_cluster, build_ledger_row
from consendxformer.lidc_masks import RepresentativeMaskRecord
from consendxformer.lidc_reconciliation import FrozenSeriesIdentity
from consendxformer.lidc_records import AnnotationRecord, ContourRecord, NoduleClusterRecord, ScanRecord
from consendxformer.lidc_targets import (
    AnnotationValidationError,
    compute_rating_targets,
    paper_mean_label,
    score_to_class,
    stable_nodule_identity,
    validate_malignancy_rating,
)
from phase2_fakes import cluster_with_ratings, square_annotation


SCAN = ScanRecord(
    patient_id="LIDC-IDRI-0001",
    study_instance_uid="study-1",
    series_instance_uid="series-1",
    annotation_count=4,
    dicom_readable=True,
    annotation_source_available=True,
)


def _masks(n_annotations: int) -> RepresentativeMaskRecord:
    individual = np.ones((n_annotations, 2, 2), dtype=np.uint8)
    return RepresentativeMaskRecord(
        individual_masks=individual,
        soft_consensus_mask=individual.mean(axis=0).astype(np.float32),
        binary_consensus_50=np.ones((2, 2), dtype=np.uint8),
        full_image_bbox=((0, 2), (0, 2), (0, 1)),
        representative_slice_index_common=0,
        representative_slice_index=0,
        representative_sop_instance_uid="sop-0",
        representative_z=0.0,
        annotation_sha256s=tuple(f"{index:064x}" for index in range(n_annotations)),
    )


def _ledger(ratings):
    cluster = cluster_with_ratings(SCAN, ratings)
    analysis = analyze_cluster(cluster)
    identity = FrozenSeriesIdentity(SCAN.patient_id, SCAN.study_instance_uid, SCAN.series_instance_uid, "test")
    return build_ledger_row(
        cluster,
        analysis,
        frozen_identity=identity,
        reconciliation_error=None,
        representative_masks=_masks(len(ratings)),
        mask_path="masks/x.npz",
        mask_sha256="0" * 64,
        mask_error=None,
    )


def test_stable_nodule_id_is_invariant_to_annotation_order():
    annotations = (square_annotation(1, offset=0.0), square_annotation(5, offset=0.2))
    left = stable_nodule_identity(NoduleClusterRecord(SCAN, annotations))
    right = stable_nodule_identity(NoduleClusterRecord(SCAN, tuple(reversed(annotations))))
    assert left == right
    assert left.nodule_id.startswith("LIDC-IDRI-0001__nodule-")
    assert len(left.nodule_sha256) == 64


def test_stable_id_canonicalizes_polygon_start_and_orientation():
    points = ((1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0))
    rotated = points[2:] + points[:2]
    reversed_points = tuple(reversed(points))
    annotations = [
        AnnotationRecord(3, (ContourRecord(1, value, "sop", 1.0),))
        for value in (points, rotated, reversed_points)
    ]
    hashes = [
        stable_nodule_identity(NoduleClusterRecord(SCAN, (annotation,))).nodule_sha256
        for annotation in annotations
    ]
    assert len(set(hashes)) == 1


@pytest.mark.parametrize("invalid", [0, 6, -1, 2.5, "3", True])
def test_malignancy_ratings_are_restricted_to_integer_one_through_five(invalid):
    with pytest.raises(AnnotationValidationError, match="1..5"):
        validate_malignancy_rating(invalid)


def test_individual_score_to_three_class_mapping():
    assert [score_to_class(score) for score in range(1, 6)] == [
        "benign",
        "benign",
        "indeterminate",
        "malignant",
        "malignant",
    ]


def test_distribution_entropy_and_ordinal_targets():
    targets = compute_rating_targets([1, 3, 4, 5])
    assert targets.q_benign == pytest.approx(0.25)
    assert targets.q_indeterminate == pytest.approx(0.25)
    assert targets.q_malignant == pytest.approx(0.5)
    assert targets.q_benign + targets.q_indeterminate + targets.q_malignant == pytest.approx(1.0)
    assert 0 <= targets.disagreement_entropy <= 1
    assert (targets.ordinal_t1, targets.ordinal_t2, targets.ordinal_t3, targets.ordinal_t4) == pytest.approx(
        (0.75, 0.75, 0.5, 0.25)
    )
    assert targets.ordinal_t1 >= targets.ordinal_t2 >= targets.ordinal_t3 >= targets.ordinal_t4


def test_unanimous_mapped_class_entropy_is_zero():
    targets = compute_rating_targets([1, 2, 1, 2])
    assert targets.disagreement_entropy == pytest.approx(0.0)
    assert math.isfinite(targets.disagreement_entropy)


@pytest.mark.parametrize(
    ("score_sum", "readers", "label", "eligible", "boundary"),
    [
        (2, 1, "benign", True, False),
        (5, 2, "boundary_zone", False, True),
        (3, 1, "indeterminate", True, False),
        (7, 2, "boundary_zone", False, True),
        (4, 1, "malignant", True, False),
    ],
)
def test_exact_paper_hard_label_and_boundary_rules(
    score_sum, readers, label, eligible, boundary
):
    assert paper_mean_label(score_sum, readers) == (label, eligible, boundary)


def test_one_reader_is_preserved_and_flagged_without_four_reader_assumption():
    row = _ledger([3])
    assert row["n_annotations"] == 1
    assert row["n_valid_malignancy_ratings"] == 1
    assert row["has_multiple_readers"] is False
    assert row["has_four_valid_readers"] is False
    assert json.loads(row["malignancy_ratings_json"]) == [3]


def test_four_valid_reader_flag_and_targets_work():
    row = _ledger([1, 2, 4, 5])
    assert row["has_multiple_readers"] is True
    assert row["has_four_valid_readers"] is True
    assert row["paper_strict_four_reader_candidate"] is True


def test_missing_and_malformed_raw_ratings_remain_in_canonical_diagnostics():
    row = _ledger([None, 6])
    assert row["n_annotations"] == 2
    assert row["n_valid_malignancy_ratings"] == 0
    assert row["n_missing_malignancy_ratings"] == 1
    assert sorted(json.loads(row["raw_malignancy_ratings_json"]), key=str) == sorted(
        [None, 6], key=str
    )
    assert json.loads(row["malignancy_ratings_json"]) == []
    assert row["source_validity_status"] == "excluded_invalid_rating"
