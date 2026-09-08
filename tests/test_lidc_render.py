from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.lidc_masks import RepresentativeMaskRecord
from consendxformer.lidc_records import DicomSliceRecord
from consendxformer.lidc_render import (
    DicomValidationError,
    crop_with_padding,
    dicom_pixels_to_hu,
    render_nodule_roi,
    square_crop_box,
    window_hu,
)
from consendxformer.phase2_config import CropConfig, HUWindow


def _dicom(pixels) -> DicomSliceRecord:
    return DicomSliceRecord(
        pixels=np.asarray(pixels),
        rescale_slope=2.0,
        rescale_intercept=-1000.0,
        patient_id="LIDC-IDRI-0001",
        study_instance_uid="study",
        series_instance_uid="series",
        sop_instance_uid="sop",
        z_position=0.0,
    )


def _masks_at_border() -> RepresentativeMaskRecord:
    individual = np.zeros((2, 4, 4), dtype=np.uint8)
    individual[0, 0:2, 0:2] = 1
    individual[1, 0:3, 0:2] = 1
    soft = individual.mean(axis=0).astype(np.float32)
    return RepresentativeMaskRecord(
        individual_masks=individual,
        soft_consensus_mask=soft,
        binary_consensus_50=(soft >= 0.5).astype(np.uint8),
        full_image_bbox=((0, 4), (0, 4), (0, 1)),
        representative_slice_index_common=0,
        representative_slice_index=0,
        representative_sop_instance_uid="sop",
        representative_z=0.0,
        annotation_sha256s=("a" * 64, "b" * 64),
    )


def test_hu_rescale_slope_intercept_is_applied_exactly():
    pixels = np.array([[0, 100], [-100, 50]], dtype=np.int16)
    hu = dicom_pixels_to_hu(pixels, 2.0, -1000.0)
    assert hu.dtype == np.float32
    assert np.array_equal(hu, np.array([[-1000, -800], [-1200, -900]], dtype=np.float32))


def test_explicit_hu_clipping_and_unit_interval_mapping():
    hu = np.array([[-1200, -1000, -300, 400, 700]], dtype=np.float32)
    rendered = window_hu(hu, HUWindow(-1000, 400, "user_supplied"))
    assert np.array_equal(rendered[[0], [0, 1, 3, 4]], np.array([0, 0, 1, 1], dtype=np.float32))
    assert rendered[0, 2] == pytest.approx(0.5)
    assert float(rendered.min()) >= 0 and float(rendered.max()) <= 1


def test_crop_geometry_is_square_and_identical_for_image_and_masks():
    dicom = _dicom(np.arange(100, dtype=np.int16).reshape(10, 10))
    masks = _masks_at_border()
    config = CropConfig(context_fraction=0.5, min_side_pixels=6, pad_hu=-1000)
    result = render_nodule_roi(
        dicom,
        masks,
        hu_window=HUWindow(-1000, 400),
        crop_config=config,
        output_resolution=64,
    )
    assert result.crop_box.side == 6
    assert result.crop_box.row_start < 0 or result.crop_box.column_start < 0
    assert result.image.shape == (64, 64)
    assert result.individual_masks.shape == (2, 64, 64)
    assert result.soft_consensus_mask.shape == result.image.shape
    assert np.array_equal(result.soft_consensus_mask, result.individual_masks.mean(axis=0))


def test_border_padding_is_constant_and_preserves_requested_square():
    source = np.arange(9, dtype=np.int16).reshape(3, 3)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    box = square_crop_box(mask, CropConfig(context_fraction=0, min_side_pixels=3))
    cropped = crop_with_padding(source, box, -999)
    assert cropped.shape == (3, 3)
    assert (cropped == -999).any()
    assert source[0, 0] in cropped


@pytest.mark.parametrize("resolution", [64, 96])
def test_same_canonical_nodule_supports_both_required_resolutions(resolution):
    result = render_nodule_roi(
        _dicom(np.zeros((8, 8), dtype=np.int16)),
        _masks_at_border(),
        hu_window=HUWindow(-1000, 400),
        crop_config=CropConfig(context_fraction=0, min_side_pixels=4),
        output_resolution=resolution,
    )
    assert result.image.shape == (resolution, resolution)
    assert result.individual_masks.shape[1:] == (resolution, resolution)


def test_invalid_dicom_and_unsupported_resolution_fail_clearly():
    with pytest.raises(DicomValidationError, match="2D"):
        dicom_pixels_to_hu(np.zeros((2, 2, 2)), 1, 0)
    with pytest.raises(ValueError, match="only 64x64 and 96x96"):
        render_nodule_roi(
            _dicom(np.zeros((8, 8))),
            _masks_at_border(),
            hu_window=HUWindow(-1000, 400),
            crop_config=CropConfig(),
            output_resolution=32,
        )
