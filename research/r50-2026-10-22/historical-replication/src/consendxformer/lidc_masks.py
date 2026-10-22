"""Common-frame mask consensus and deterministic representative-slice artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .hashing import sha256_file
from .lidc_records import ConsensusVolumeRecord, ScanRecord


REPRESENTATIVE_RULE = "maximum_soft_consensus_area_lowest_index_v1"
MASK_ARTIFACT_SCHEMA = 1


class MaskValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RepresentativeMaskRecord:
    individual_masks: np.ndarray
    soft_consensus_mask: np.ndarray
    binary_consensus_50: np.ndarray
    full_image_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    representative_slice_index_common: int
    representative_slice_index: int
    representative_sop_instance_uid: str | None
    representative_z: float | None
    annotation_sha256s: tuple[str, ...]
    selection_rule: str = REPRESENTATIVE_RULE


def _validated_individual_masks(value: np.ndarray) -> np.ndarray:
    masks = np.asarray(value)
    if masks.ndim != 4:
        raise MaskValidationError(
            "Individual masks must have shape (annotations, rows, columns, slices)"
        )
    if any(size <= 0 for size in masks.shape):
        raise MaskValidationError("Individual mask dimensions must all be non-zero")
    if not np.issubdtype(masks.dtype, np.bool_) and not np.issubdtype(masks.dtype, np.integer):
        raise MaskValidationError("Individual masks must be boolean or integer 0/1 arrays")
    if bool(((masks != 0) & (masks != 1)).any()):
        raise MaskValidationError("Individual masks contain values outside {0,1}")
    return masks.astype(bool, copy=False)


def _validated_bbox(
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    mask_shape: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if len(bbox) != 3:
        raise MaskValidationError("Consensus bbox must contain row, column, and slice ranges")
    normalized: list[tuple[int, int]] = []
    for axis, (bounds, expected_size) in enumerate(zip(bbox, mask_shape)):
        if len(bounds) != 2 or any(isinstance(value, bool) for value in bounds):
            raise MaskValidationError(f"Invalid bbox bounds for axis {axis}")
        start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop <= start:
            raise MaskValidationError(f"Invalid bbox range {(start, stop)} for axis {axis}")
        if stop - start != expected_size:
            raise MaskValidationError(
                f"BBox axis {axis} size {stop - start} does not match mask size {expected_size}"
            )
        normalized.append((start, stop))
    return tuple(normalized)  # type: ignore[return-value]


def soft_consensus(individual_masks: np.ndarray) -> np.ndarray:
    masks = _validated_individual_masks(individual_masks)
    soft = masks.astype(np.float32).mean(axis=0, dtype=np.float32)
    if bool((soft < 0).any()) or bool((soft > 1).any()):
        raise AssertionError("Soft consensus left the [0,1] range")
    return soft.astype(np.float32, copy=False)


def representative_slice_index(soft_mask_3d: np.ndarray) -> int:
    soft = np.asarray(soft_mask_3d, dtype=np.float32)
    if soft.ndim != 3 or any(size <= 0 for size in soft.shape):
        raise MaskValidationError("Soft consensus must be a non-empty 3D array")
    if not bool(np.isfinite(soft).all()) or bool((soft < 0).any()) or bool((soft > 1).any()):
        raise MaskValidationError("Soft consensus values must be finite and in [0,1]")
    areas = soft.sum(axis=(0, 1), dtype=np.float64)
    if float(areas.max()) <= 0:
        raise MaskValidationError("Cannot select a representative slice from an empty consensus")
    # np.argmax returns the first index, which is the required lowest-index tie rule.
    return int(np.argmax(areas))


def build_representative_masks(
    consensus: ConsensusVolumeRecord,
    scan: ScanRecord,
    annotation_sha256s: tuple[str, ...],
) -> RepresentativeMaskRecord:
    masks = _validated_individual_masks(consensus.individual_masks)
    if len(annotation_sha256s) != masks.shape[0]:
        raise MaskValidationError("Annotation signatures and individual masks are not aligned")
    bbox = _validated_bbox(consensus.bbox, masks.shape[1:])
    soft_3d = soft_consensus(masks)
    common_index = representative_slice_index(soft_3d)
    full_index = bbox[2][0] + common_index
    sop_uid = None
    z_position = None
    if scan.sop_instance_uids:
        if full_index >= len(scan.sop_instance_uids):
            raise MaskValidationError("Representative slice index exceeds scan SOP inventory")
        sop_uid = scan.sop_instance_uids[full_index]
    if scan.z_positions:
        if full_index >= len(scan.z_positions):
            raise MaskValidationError("Representative slice index exceeds scan z-position inventory")
        z_position = scan.z_positions[full_index]
    individual_2d = masks[:, :, :, common_index].astype(np.uint8, copy=True)
    soft_2d = soft_3d[:, :, common_index].astype(np.float32, copy=True)
    # This diagnostic artifact is definitionally the 50% view, independent
    # of a provider's separately provenance-tracked consensus clevel.
    binary_2d = (soft_2d >= 0.5).astype(np.uint8)
    if not np.allclose(soft_2d, individual_2d.mean(axis=0), rtol=0.0, atol=1e-7):
        raise AssertionError("Representative soft mask is not the individual-mask mean")
    return RepresentativeMaskRecord(
        individual_masks=individual_2d,
        soft_consensus_mask=soft_2d,
        binary_consensus_50=binary_2d,
        full_image_bbox=bbox,
        representative_slice_index_common=common_index,
        representative_slice_index=full_index,
        representative_sop_instance_uid=sop_uid,
        representative_z=z_position,
        annotation_sha256s=annotation_sha256s,
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write NPZ members with stable order, timestamps, permissions, and compression."""
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite mask artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe NPZ member name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])), compresslevel=9)
    return destination


def write_mask_artifact(path: str | Path, record: RepresentativeMaskRecord) -> tuple[Path, str]:
    metadata = {
        "schema_version": MASK_ARTIFACT_SCHEMA,
        "full_image_bbox": [list(axis) for axis in record.full_image_bbox],
        "representative_slice_index_common": record.representative_slice_index_common,
        "representative_slice_index": record.representative_slice_index,
        "representative_sop_instance_uid": record.representative_sop_instance_uid,
        "representative_z": record.representative_z,
        "selection_rule": record.selection_rule,
        "annotation_sha256s": list(record.annotation_sha256s),
    }
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    arrays = {
        "binary_consensus_50": record.binary_consensus_50.astype(np.uint8),
        "individual_masks": record.individual_masks.astype(np.uint8),
        "metadata_json_utf8": np.frombuffer(metadata_bytes, dtype=np.uint8),
        "soft_consensus_mask": record.soft_consensus_mask.astype(np.float32),
    }
    destination = write_deterministic_npz(path, arrays)
    return destination, sha256_file(destination)


def load_mask_artifact(path: str | Path) -> RepresentativeMaskRecord:
    source = Path(path).resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            expected = {
                "binary_consensus_50",
                "individual_masks",
                "metadata_json_utf8",
                "soft_consensus_mask",
            }
            if set(archive.files) != expected:
                raise MaskValidationError(
                    f"Mask artifact members differ from schema: {sorted(archive.files)}"
                )
            individual_source = archive["individual_masks"]
            soft_source = archive["soft_consensus_mask"]
            binary_source = archive["binary_consensus_50"]
            if individual_source.dtype != np.uint8:
                raise MaskValidationError("Individual masks must use uint8 storage")
            if soft_source.dtype != np.float32:
                raise MaskValidationError("Soft consensus must use float32 storage")
            if binary_source.dtype != np.uint8:
                raise MaskValidationError("Binary consensus must use uint8 storage")
            individual = individual_source.copy()
            soft = soft_source.copy()
            binary = binary_source.copy()
            metadata = json.loads(archive["metadata_json_utf8"].tobytes().decode("utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, MaskValidationError):
            raise
        raise MaskValidationError(f"Cannot read mask artifact {source}: {exc}") from exc
    if individual.ndim != 3 or soft.shape != individual.shape[1:] or binary.shape != soft.shape:
        raise MaskValidationError("Mask artifact array shapes are inconsistent")
    if any(size <= 0 for size in individual.shape):
        raise MaskValidationError("Mask artifact dimensions must be non-zero")
    if bool(((individual != 0) & (individual != 1)).any()):
        raise MaskValidationError("Individual masks contain values outside {0,1}")
    if not bool(np.isfinite(soft).all()) or bool((soft < 0).any()) or bool((soft > 1).any()):
        raise MaskValidationError("Soft consensus contains non-finite or out-of-range values")
    if bool(((binary != 0) & (binary != 1)).any()):
        raise MaskValidationError("Binary consensus contains values outside {0,1}")
    if not np.allclose(soft, individual.mean(axis=0), rtol=0.0, atol=1e-6):
        raise MaskValidationError("Stored soft consensus is not the individual-mask mean")
    if not np.array_equal(binary, (soft >= 0.5).astype(np.uint8)):
        raise MaskValidationError("Stored binary consensus is not the >=50% soft-mask view")
    required_metadata = {
        "schema_version",
        "full_image_bbox",
        "representative_slice_index_common",
        "representative_slice_index",
        "representative_sop_instance_uid",
        "representative_z",
        "selection_rule",
        "annotation_sha256s",
    }
    if not isinstance(metadata, dict) or set(metadata) != required_metadata:
        raise MaskValidationError("Mask artifact metadata differs from the schema")
    if metadata["schema_version"] != MASK_ARTIFACT_SCHEMA:
        raise MaskValidationError("Unsupported mask artifact schema version")
    try:
        bbox = tuple(tuple(int(v) for v in axis) for axis in metadata["full_image_bbox"])
        bbox = _validated_bbox(bbox, (individual.shape[1], individual.shape[2], int(bbox[2][1]) - int(bbox[2][0])))  # type: ignore[arg-type]
        common_index = int(metadata["representative_slice_index_common"])
        full_index = int(metadata["representative_slice_index"])
    except (TypeError, ValueError, IndexError) as exc:
        raise MaskValidationError("Mask artifact geometry metadata is malformed") from exc
    if not 0 <= common_index < bbox[2][1] - bbox[2][0]:
        raise MaskValidationError("Representative common-frame slice index is outside the bbox")
    if full_index != bbox[2][0] + common_index:
        raise MaskValidationError("Representative full/common slice indices disagree")
    if metadata["selection_rule"] != REPRESENTATIVE_RULE:
        raise MaskValidationError("Unsupported representative-slice rule")
    signatures = metadata["annotation_sha256s"]
    if (
        not isinstance(signatures, list)
        or len(signatures) != individual.shape[0]
        or any(not isinstance(value, str) or len(value) != 64 for value in signatures)
    ):
        raise MaskValidationError("Annotation signatures do not align with individual masks")
    z_position = metadata["representative_z"]
    if z_position is not None and (
        isinstance(z_position, bool)
        or not isinstance(z_position, (int, float))
        or not np.isfinite(z_position)
    ):
        raise MaskValidationError("Representative z position is invalid")
    return RepresentativeMaskRecord(
        individual_masks=individual,
        soft_consensus_mask=soft,
        binary_consensus_50=binary,
        full_image_bbox=bbox,  # type: ignore[arg-type]
        representative_slice_index_common=common_index,
        representative_slice_index=full_index,
        representative_sop_instance_uid=metadata["representative_sop_instance_uid"],
        representative_z=metadata["representative_z"],
        annotation_sha256s=tuple(signatures),
        selection_rule=metadata["selection_rule"],
    )
