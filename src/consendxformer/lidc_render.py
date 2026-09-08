"""Explicit-HU, deterministic 2D ROI rendering for canonical LIDC nodules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .lidc_masks import RepresentativeMaskRecord, write_deterministic_npz
from .lidc_records import DicomSliceRecord
from .phase2_config import CropConfig, HUWindow


@dataclass(frozen=True)
class CropBox:
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int

    @property
    def side(self) -> int:
        if self.row_stop - self.row_start != self.column_stop - self.column_start:
            raise ValueError("CropBox is not square")
        return self.row_stop - self.row_start

    def to_list(self) -> list[int]:
        return [self.row_start, self.row_stop, self.column_start, self.column_stop]


@dataclass(frozen=True)
class RenderedROI:
    image: np.ndarray
    individual_masks: np.ndarray
    soft_consensus_mask: np.ndarray
    binary_consensus_50: np.ndarray
    crop_box: CropBox
    output_resolution: int
    hu_window: HUWindow
    crop_config: CropConfig


class DicomValidationError(ValueError):
    pass


def dicom_pixels_to_hu(
    pixels: np.ndarray, rescale_slope: float, rescale_intercept: float
) -> np.ndarray:
    array = np.asarray(pixels)
    if array.ndim != 2 or any(size <= 0 for size in array.shape):
        raise DicomValidationError("DICOM pixel array must be a non-empty 2D array")
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise DicomValidationError("DICOM pixels must be finite numeric values")
    slope = float(rescale_slope)
    intercept = float(rescale_intercept)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise DicomValidationError("DICOM rescale slope/intercept must be finite")
    return (array.astype(np.float32) * np.float32(slope) + np.float32(intercept)).astype(
        np.float32
    )


def window_hu(hu: np.ndarray, window: HUWindow) -> np.ndarray:
    values = np.asarray(hu, dtype=np.float32)
    if values.ndim != 2 or not bool(np.isfinite(values).all()):
        raise ValueError("HU image must be a finite 2D array")
    clipped = np.clip(values, window.minimum, window.maximum)
    normalized = (clipped - window.minimum) / (window.maximum - window.minimum)
    return normalized.astype(np.float32)


def square_crop_box(mask: np.ndarray, config: CropConfig) -> CropBox:
    union = np.asarray(mask).astype(bool)
    if union.ndim != 2 or not bool(union.any()):
        raise ValueError("A non-empty 2D nodule mask is required to derive a crop")
    rows, columns = np.nonzero(union)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(columns.min()), int(columns.max())
    base_side = max(row_max - row_min + 1, col_max - col_min + 1)
    side = max(
        config.min_side_pixels,
        int(math.ceil(base_side * (1.0 + 2.0 * config.context_fraction))),
    )
    row_center = (row_min + row_max) / 2.0
    col_center = (col_min + col_max) / 2.0
    row_start = int(math.floor(row_center - (side - 1) / 2.0))
    col_start = int(math.floor(col_center - (side - 1) / 2.0))
    return CropBox(row_start, row_start + side, col_start, col_start + side)


def crop_with_padding(array: np.ndarray, box: CropBox, pad_value: float | int) -> np.ndarray:
    source = np.asarray(array)
    if source.ndim != 2:
        raise ValueError("crop_with_padding expects a 2D array")
    side = box.side
    output = np.full((side, side), pad_value, dtype=source.dtype)
    src_r0 = max(0, box.row_start)
    src_r1 = min(source.shape[0], box.row_stop)
    src_c0 = max(0, box.column_start)
    src_c1 = min(source.shape[1], box.column_stop)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        raise ValueError("Crop box does not intersect the source image")
    dst_r0 = src_r0 - box.row_start
    dst_c0 = src_c0 - box.column_start
    output[
        dst_r0 : dst_r0 + (src_r1 - src_r0),
        dst_c0 : dst_c0 + (src_c1 - src_c0),
    ] = source[src_r0:src_r1, src_c0:src_c1]
    return output


def _resize_nearest(array: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(array)
    if source.ndim != 2 or size <= 0:
        raise ValueError("Nearest-neighbor resize requires a 2D array and positive size")
    row_indices = np.minimum(
        ((np.arange(size) + 0.5) * source.shape[0] / size).astype(int),
        source.shape[0] - 1,
    )
    col_indices = np.minimum(
        ((np.arange(size) + 0.5) * source.shape[1] / size).astype(int),
        source.shape[1] - 1,
    )
    return source[np.ix_(row_indices, col_indices)]


def _resize_bilinear(array: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(array, dtype=np.float32)
    if source.ndim != 2 or size <= 0:
        raise ValueError("Bilinear resize requires a 2D array and positive size")
    if source.shape == (size, size):
        return source.copy()
    row_coords = np.clip(
        (np.arange(size, dtype=np.float64) + 0.5) * source.shape[0] / size - 0.5,
        0,
        source.shape[0] - 1,
    )
    col_coords = np.clip(
        (np.arange(size, dtype=np.float64) + 0.5) * source.shape[1] / size - 0.5,
        0,
        source.shape[1] - 1,
    )
    r0 = np.floor(row_coords).astype(int)
    c0 = np.floor(col_coords).astype(int)
    r1 = np.minimum(r0 + 1, source.shape[0] - 1)
    c1 = np.minimum(c0 + 1, source.shape[1] - 1)
    rw = (row_coords - r0).astype(np.float32)[:, None]
    cw = (col_coords - c0).astype(np.float32)[None, :]
    top = source[np.ix_(r0, c0)] * (1 - cw) + source[np.ix_(r0, c1)] * cw
    bottom = source[np.ix_(r1, c0)] * (1 - cw) + source[np.ix_(r1, c1)] * cw
    return (top * (1 - rw) + bottom * rw).astype(np.float32)


def _place_common_mask(
    common: np.ndarray,
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    full_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.asarray(common)
    expected = (bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0])
    if mask.shape != expected:
        raise ValueError(f"Common-frame mask shape {mask.shape} does not match bbox {expected}")
    if bbox[0][1] > full_shape[0] or bbox[1][1] > full_shape[1]:
        raise ValueError("Common-frame bbox exceeds the DICOM image")
    full = np.zeros(full_shape, dtype=mask.dtype)
    full[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1]] = mask
    return full


def render_nodule_roi(
    dicom_slice: DicomSliceRecord,
    masks: RepresentativeMaskRecord,
    *,
    hu_window: HUWindow,
    crop_config: CropConfig,
    output_resolution: int,
) -> RenderedROI:
    if output_resolution not in {64, 96}:
        raise ValueError("Phase 2 renderer supports only 64x64 and 96x96 outputs")
    hu = dicom_pixels_to_hu(
        dicom_slice.pixels, dicom_slice.rescale_slope, dicom_slice.rescale_intercept
    )
    full_individual = np.stack(
        [
            _place_common_mask(mask, masks.full_image_bbox, hu.shape)
            for mask in masks.individual_masks
        ],
        axis=0,
    ).astype(np.uint8)
    union = full_individual.any(axis=0)
    box = square_crop_box(union, crop_config)
    image_crop = crop_with_padding(hu, box, crop_config.pad_hu)
    rendered_image = _resize_bilinear(window_hu(image_crop, hu_window), output_resolution)
    rendered_individual = np.stack(
        [
            _resize_nearest(crop_with_padding(mask, box, 0), output_resolution)
            for mask in full_individual
        ],
        axis=0,
    ).astype(np.uint8)
    rendered_soft = rendered_individual.astype(np.float32).mean(axis=0, dtype=np.float32)
    rendered_binary = (rendered_soft >= 0.5).astype(np.uint8)
    if not np.allclose(rendered_soft, rendered_individual.mean(axis=0), atol=1e-7):
        raise AssertionError("Rendered soft mask lost alignment with individual masks")
    return RenderedROI(
        image=rendered_image.astype(np.float32),
        individual_masks=rendered_individual,
        soft_consensus_mask=rendered_soft,
        binary_consensus_50=rendered_binary,
        crop_box=box,
        output_resolution=output_resolution,
        hu_window=hu_window,
        crop_config=crop_config,
    )


def write_rendered_roi(
    output_dir: str | Path, nodule_id: str, rendered: RenderedROI
) -> dict[str, str]:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{nodule_id}.npy"
    mask_path = directory / f"{nodule_id}_masks.npz"
    if image_path.exists() or mask_path.exists():
        raise FileExistsError(f"Refusing to overwrite rendered ROI for {nodule_id}")
    np.save(image_path, rendered.image.astype(np.float32), allow_pickle=False)
    metadata = np.frombuffer(
        str(rendered.crop_box.to_list()).encode("ascii"), dtype=np.uint8
    )
    write_deterministic_npz(
        mask_path,
        {
            "binary_consensus_50": rendered.binary_consensus_50,
            "crop_box_ascii": metadata,
            "individual_masks": rendered.individual_masks,
            "soft_consensus_mask": rendered.soft_consensus_mask,
        },
    )
    return {"image_path": str(image_path), "mask_path": str(mask_path)}


def load_dicom_file(path: str | Path) -> DicomSliceRecord:
    """Load one CT slice through optional pydicom with strict required identifiers."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required for real DICOM rendering; install requirements-phase2.txt"
        ) from exc
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DICOM file does not exist: {source}")
    try:
        dataset = pydicom.dcmread(source)
        pixels = dataset.pixel_array
    except Exception as exc:
        raise DicomValidationError(f"Cannot decode DICOM file {source}: {exc}") from exc
    required = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
    missing = [name for name in required if not str(getattr(dataset, name, "")).strip()]
    if missing:
        raise DicomValidationError(f"DICOM file lacks required identifiers: {missing}")
    z_position = None
    if hasattr(dataset, "ImagePositionPatient") and len(dataset.ImagePositionPatient) >= 3:
        z_position = float(dataset.ImagePositionPatient[2])
    return DicomSliceRecord(
        pixels=np.asarray(pixels),
        rescale_slope=float(getattr(dataset, "RescaleSlope", 1.0)),
        rescale_intercept=float(getattr(dataset, "RescaleIntercept", 0.0)),
        patient_id=str(dataset.PatientID),
        study_instance_uid=str(dataset.StudyInstanceUID),
        series_instance_uid=str(dataset.SeriesInstanceUID),
        sop_instance_uid=str(dataset.SOPInstanceUID),
        z_position=z_position,
        source_path=str(source),
    )
