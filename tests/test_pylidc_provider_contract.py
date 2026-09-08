from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.lidc_records import AnnotationRecord, NoduleClusterRecord, ScanRecord
from consendxformer.phase2_config import ClusteringConfig, ConsensusConfig
from consendxformer.pylidc_provider import PylidcProvider


class _Contour:
    image_k_position = 1
    image_z_position = 1.0
    inclusion = True

    def to_matrix(self, include_k=True):
        assert include_k is True
        return np.array([[1, 1, 1], [1, 3, 1], [3, 3, 1], [3, 1, 1]])


class _Annotation:
    malignancy = 3
    contours = (_Contour(),)


class _MalformedContour:
    image_k_position = 1
    image_z_position = 1.0
    inclusion = True

    def to_matrix(self, include_k=True):
        del include_k
        return np.array([1, 2, 3])


class _RawScan:
    def __init__(self):
        self.kwargs = None

    def cluster_annotations(self, **kwargs):
        self.kwargs = kwargs
        return [[_Annotation()]]


def _provider_and_scan():
    provider = PylidcProvider(ROOT)
    provider._pylidc = object()
    raw = _RawScan()
    scan = ScanRecord(
        patient_id="LIDC-IDRI-0001",
        study_instance_uid="study",
        series_instance_uid="series",
        annotation_count=1,
        dicom_readable=True,
        annotation_source_available=True,
        sop_instance_uids=("sop-0", "sop-1", "sop-2"),
        z_positions=(0.0, 1.0, 2.0),
        provider_payload=raw,
    )
    return provider, scan, raw


def test_clustering_calls_official_api_with_every_parameter_explicit():
    provider, scan, raw = _provider_and_scan()
    clusters = provider.cluster_annotations(scan, ClusteringConfig())
    assert raw.kwargs == {
        "metric": "min",
        "tol": None,
        "factor": 0.9,
        "min_tol": 0.1,
        "return_distance_matrix": False,
        "verbose": False,
    }
    assert len(clusters) == 1
    assert clusters[0].annotations[0].malignancy == 3
    assert clusters[0].annotations[0].contours[0].sop_instance_uid == "sop-1"
    assert not hasattr(clusters[0].annotations[0], "database_id")


def test_consensus_calls_official_api_with_ret_masks_and_preserves_common_frame():
    provider, scan, _raw = _provider_and_scan()
    calls = {}

    def consensus(annotations, **kwargs):
        calls["annotations"] = annotations
        calls.update(kwargs)
        masks = [np.ones((2, 3, 1), dtype=bool), np.zeros((2, 3, 1), dtype=bool)]
        return np.ones((2, 3, 1), dtype=bool), (slice(4, 6), slice(5, 8), slice(1, 2)), masks

    provider._consensus = consensus
    annotations = (
        AnnotationRecord(1, (), provider_payload="first"),
        AnnotationRecord(5, (), provider_payload="second"),
    )
    cluster = NoduleClusterRecord(scan, annotations)
    result = provider.consensus_masks(cluster, ConsensusConfig())
    assert calls == {
        "annotations": ["first", "second"],
        "clevel": 0.5,
        "pad": None,
        "ret_masks": True,
    }
    assert result.individual_masks.shape == (2, 2, 3, 1)
    assert result.bbox == ((4, 6), (5, 8), (1, 2))


def test_malformed_provider_contour_is_preserved_as_diagnostic_record():
    provider, scan, _raw = _provider_and_scan()
    raw_annotation = _Annotation()
    raw_annotation.contours = (_Contour(), _MalformedContour())
    record = provider._annotation(raw_annotation, scan)
    assert record.source_status == "malformed"
    assert "contour[1]" in record.source_error
    assert len(record.contours) == 1
