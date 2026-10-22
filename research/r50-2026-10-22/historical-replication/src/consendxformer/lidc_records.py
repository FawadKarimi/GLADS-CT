"""Provider-neutral LIDC records used by the Phase 2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .phase2_config import ClusteringConfig, ConsensusConfig


@dataclass(frozen=True)
class ContourRecord:
    """One source contour in image row/column coordinates."""

    slice_index: int | None
    points: tuple[tuple[float, float], ...]
    sop_instance_uid: str | None = None
    z_position: float | None = None
    inclusion: bool = True


@dataclass(frozen=True)
class AnnotationRecord:
    """One local-to-scan radiologist annotation; no global reader identity."""

    malignancy: Any
    contours: tuple[ContourRecord, ...]
    source_status: str = "available"
    source_error: str | None = None
    provider_payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ScanRecord:
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    annotation_count: int
    dicom_readable: bool
    annotation_source_available: bool
    dicom_directory: str | None = None
    sop_instance_uids: tuple[str | None, ...] = ()
    z_positions: tuple[float | None, ...] = ()
    provider_payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class NoduleClusterRecord:
    scan: ScanRecord
    annotations: tuple[AnnotationRecord, ...]


@dataclass(frozen=True)
class ConsensusVolumeRecord:
    """Individual masks in one common (row, column, axial-slice) frame."""

    individual_masks: np.ndarray
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class DicomSliceRecord:
    pixels: np.ndarray
    rescale_slope: float
    rescale_intercept: float
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str | None
    z_position: float | None
    source_path: str | None = None


class LIDCProvider(Protocol):
    """Minimal provider contract; pylidc and tests implement this interface."""

    name: str

    def versions(self) -> Mapping[str, Any]: ...

    def iter_scans(self) -> Iterable[ScanRecord]: ...

    def cluster_annotations(
        self, scan: ScanRecord, config: ClusteringConfig
    ) -> Sequence[NoduleClusterRecord]: ...

    def consensus_masks(
        self,
        cluster: NoduleClusterRecord,
        config: ConsensusConfig,
    ) -> ConsensusVolumeRecord: ...

    def load_dicom_slice(self, scan: ScanRecord, slice_index: int) -> DicomSliceRecord: ...


class InMemoryLIDCProvider:
    """Explicit fake provider for offline tests; never used as a real-data fallback."""

    name = "in_memory_test_provider"

    def __init__(
        self,
        scans: Sequence[ScanRecord],
        *,
        clusters: Mapping[str, Sequence[NoduleClusterRecord]] | None = None,
        consensus_factory: Callable[[NoduleClusterRecord, ConsensusConfig], ConsensusVolumeRecord]
        | None = None,
        dicom_slices: Mapping[tuple[str, int], DicomSliceRecord] | None = None,
    ) -> None:
        self._scans = tuple(scans)
        self._clusters = {key: tuple(value) for key, value in (clusters or {}).items()}
        self._consensus_factory = consensus_factory
        self._dicom_slices = dict(dicom_slices or {})

    def versions(self) -> Mapping[str, Any]:
        return {
            "provider": "in_memory_test_provider",
            "pylidc": None,
            "pydicom": None,
            "numpy": np.__version__,
            "pylidc_numpy_compatibility": {
                "required": False,
                "aliases": [],
                "algorithm_modified": False,
                "numpy_module_modified": False,
                "pylidc_source_modified": False,
                "strategy": "not_applicable_in_memory_provider",
            },
        }

    def iter_scans(self) -> Iterable[ScanRecord]:
        return iter(self._scans)

    def cluster_annotations(
        self, scan: ScanRecord, config: ClusteringConfig
    ) -> Sequence[NoduleClusterRecord]:
        del config
        return self._clusters.get(scan.series_instance_uid, ())

    def consensus_masks(
        self, cluster: NoduleClusterRecord, config: ConsensusConfig
    ) -> ConsensusVolumeRecord:
        if self._consensus_factory is None:
            raise RuntimeError("In-memory provider has no consensus_factory")
        return self._consensus_factory(cluster, config)

    def load_dicom_slice(self, scan: ScanRecord, slice_index: int) -> DicomSliceRecord:
        try:
            return self._dicom_slices[(scan.series_instance_uid, slice_index)]
        except KeyError as exc:
            raise FileNotFoundError(
                f"No in-memory DICOM slice for {scan.series_instance_uid} index {slice_index}"
            ) from exc


def ensure_existing_directory(path: str | Path, role: str) -> Path:
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{role} directory does not exist: {directory}")
    try:
        next(directory.iterdir(), None)
    except OSError as exc:
        raise RuntimeError(f"{role} directory is not readable: {directory}") from exc
    return directory
