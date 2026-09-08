from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_file
from consendxformer.lidc_masks import (
    MaskValidationError,
    build_representative_masks,
    load_mask_artifact,
    representative_slice_index,
    soft_consensus,
    write_mask_artifact,
)
from consendxformer.lidc_records import ConsensusVolumeRecord, ScanRecord


def _scan() -> ScanRecord:
    return ScanRecord(
        patient_id="LIDC-IDRI-0001",
        study_instance_uid="study",
        series_instance_uid="series",
        annotation_count=2,
        dicom_readable=True,
        annotation_source_available=True,
        sop_instance_uids=tuple(f"sop-{index}" for index in range(10)),
        z_positions=tuple(float(index) for index in range(10)),
    )


def test_soft_consensus_is_arithmetic_mean_and_bounded():
    masks = np.array(
        [
            [[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 1]]],
        ],
        dtype=np.uint8,
    )
    soft = soft_consensus(masks)
    assert np.array_equal(soft, masks.mean(axis=0))
    assert soft.dtype == np.float32
    assert float(soft.min()) >= 0 and float(soft.max()) <= 1


def test_representative_slice_uses_maximum_area():
    soft = np.zeros((3, 3, 3), dtype=np.float32)
    soft[0, 0, 0] = 1
    soft[:2, :2, 1] = 1
    soft[0, :2, 2] = 1
    assert representative_slice_index(soft) == 1


def test_representative_slice_tie_uses_lowest_common_index():
    soft = np.zeros((2, 2, 4), dtype=np.float32)
    soft[:, :, 1] = 0.5
    soft[:, :, 3] = 0.5
    assert representative_slice_index(soft) == 1


def test_representative_masks_preserve_reader_alignment_and_slice_identity():
    masks = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    masks[0, 0:2, 0:2, 1] = 1
    masks[1, 1:3, 1:3, 1] = 1
    record = build_representative_masks(
        ConsensusVolumeRecord(masks, ((10, 14), (20, 25), (3, 6))),
        _scan(),
        ("a" * 64, "b" * 64),
    )
    assert record.representative_slice_index_common == 1
    assert record.representative_slice_index == 4
    assert record.representative_sop_instance_uid == "sop-4"
    assert record.representative_z == 4.0
    assert record.individual_masks.shape == (2, 4, 5)
    assert np.array_equal(record.soft_consensus_mask, record.individual_masks.mean(axis=0))


def test_mask_artifact_is_portable_shape_checked_and_byte_deterministic(tmp_path):
    masks = np.zeros((2, 3, 3, 2), dtype=np.uint8)
    masks[:, 1:, 1:, 0] = 1
    record = build_representative_masks(
        ConsensusVolumeRecord(masks, ((4, 7), (5, 8), (2, 4))),
        _scan(),
        ("1" * 64, "2" * 64),
    )
    left, left_sha = write_mask_artifact(tmp_path / "left.npz", record)
    right, right_sha = write_mask_artifact(tmp_path / "right.npz", record)
    assert left.read_bytes() == right.read_bytes()
    assert left_sha == right_sha == sha256_file(left)
    loaded = load_mask_artifact(left)
    assert loaded.individual_masks.dtype == np.uint8
    assert loaded.soft_consensus_mask.dtype == np.float32
    assert np.array_equal(loaded.individual_masks, record.individual_masks)
    assert np.array_equal(loaded.soft_consensus_mask, record.soft_consensus_mask)
    assert loaded.annotation_sha256s == record.annotation_sha256s


def test_invalid_mask_values_and_shape_fail_clearly():
    with pytest.raises(MaskValidationError, match="shape"):
        soft_consensus(np.zeros((2, 3, 3), dtype=np.uint8))
    bad = np.zeros((1, 2, 2, 2), dtype=np.uint8)
    bad[0, 0, 0, 0] = 2
    with pytest.raises(MaskValidationError, match="outside"):
        soft_consensus(bad)
