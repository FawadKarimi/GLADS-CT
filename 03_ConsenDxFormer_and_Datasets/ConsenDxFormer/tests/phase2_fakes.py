from __future__ import annotations

from pathlib import Path

import numpy as np

from consendxformer.lidc_records import (
    AnnotationRecord,
    ConsensusVolumeRecord,
    ContourRecord,
    DicomSliceRecord,
    NoduleClusterRecord,
    ScanRecord,
)


def scan_from_manifest(report, row_index: int = 0, n_slices: int = 8) -> ScanRecord:
    row = report.series_split.iloc[row_index]
    return ScanRecord(
        patient_id=str(row.patient_id),
        study_instance_uid=str(row.study_instance_uid),
        series_instance_uid=str(row.series_instance_uid),
        annotation_count=0,
        dicom_readable=True,
        annotation_source_available=True,
        sop_instance_uids=tuple(f"sop-{index}" for index in range(n_slices)),
        z_positions=tuple(float(index) for index in range(n_slices)),
    )


def square_annotation(
    malignancy,
    *,
    offset: float = 0.0,
    slice_index: int = 3,
) -> AnnotationRecord:
    points = (
        (1.0 + offset, 1.0 + offset),
        (1.0 + offset, 3.0 + offset),
        (3.0 + offset, 3.0 + offset),
        (3.0 + offset, 1.0 + offset),
    )
    return AnnotationRecord(
        malignancy=malignancy,
        contours=(
            ContourRecord(
                slice_index=slice_index,
                points=points,
                sop_instance_uid=f"sop-{slice_index}",
                z_position=float(slice_index),
            ),
        ),
    )


def cluster_with_ratings(scan: ScanRecord, ratings) -> NoduleClusterRecord:
    annotations = tuple(
        square_annotation(score, offset=index * 0.1)
        for index, score in enumerate(ratings)
    )
    return NoduleClusterRecord(scan=scan, annotations=annotations)


def consensus_for_cluster(
    cluster: NoduleClusterRecord, _config=None
) -> ConsensusVolumeRecord:
    masks = np.zeros((len(cluster.annotations), 5, 5, 3), dtype=np.uint8)
    for index in range(len(cluster.annotations)):
        masks[index, 1:4, 1:4, 1] = 1
        if index % 2 == 0:
            masks[index, 2:4, 2:4, 2] = 1
    return ConsensusVolumeRecord(
        individual_masks=masks,
        bbox=((10, 15), (20, 25), (2, 5)),
    )


def dicom_for_scan(scan: ScanRecord, slice_index: int) -> DicomSliceRecord:
    pixels = np.arange(40 * 40, dtype=np.int16).reshape(40, 40)
    return DicomSliceRecord(
        pixels=pixels,
        rescale_slope=1.0,
        rescale_intercept=-1000.0,
        patient_id=scan.patient_id,
        study_instance_uid=scan.study_instance_uid,
        series_instance_uid=scan.series_instance_uid,
        sop_instance_uid=scan.sop_instance_uids[slice_index],
        z_position=scan.z_positions[slice_index],
        source_path=str(Path("synthetic") / f"slice-{slice_index}.dcm"),
    )
