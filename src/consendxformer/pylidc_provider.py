"""Lazy adapter for the official pylidc==0.2.3 release."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .lidc_records import (
    AnnotationRecord,
    ConsensusVolumeRecord,
    ContourRecord,
    DicomSliceRecord,
    NoduleClusterRecord,
    ScanRecord,
)
from .phase2_config import ClusteringConfig, ConsensusConfig, PYLIDC_VERSION
from .pylidc_numpy_compat import (
    NumpyCompatibilityError,
    PylidcNumpyCompatibility,
    pylidc_numpy_compatibility_report,
)


class ProviderDependencyError(RuntimeError):
    pass


class ProviderCompatibilityError(RuntimeError):
    pass


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class PylidcProvider:
    """Translate pylidc objects without exposing SQLAlchemy primary keys."""

    name = "official_pylidc"

    def __init__(self, raw_root: str | Path | None) -> None:
        self.raw_root = None if raw_root is None else Path(raw_root).resolve()
        self._pylidc: Any = None
        self._consensus: Any = None
        self._pydicom: Any = None
        self._numpy_compatibility: PylidcNumpyCompatibility | None = None
        self._raw_scan_cache: tuple[Any, ...] | None = None
        self._scan_cache: tuple[ScanRecord, ...] | None = None
        self._annotation_scan_cache: tuple[ScanRecord, ...] | None = None
        # Keep at most one decoded series resident.  A cache containing all 1018
        # scans can approach the size of the source collection.
        self._dicom_cache_series: str | None = None
        self._dicom_cache_images: list[Any] | None = None

    def _load(self) -> None:
        if self._pylidc is not None:
            if self._numpy_compatibility is None:
                raise ProviderCompatibilityError(
                    "Official pylidc runtime is only partially initialized"
                )
            return
        try:
            installed = importlib.metadata.version("pylidc")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProviderDependencyError(
                "Official pylidc==0.2.3 is not installed. Install "
                "requirements-phase2-pylidc.txt for real LIDC access; no fork is accepted."
            ) from exc
        if installed != PYLIDC_VERSION:
            raise ProviderDependencyError(
                f"Unsupported pylidc version {installed!r}; exactly {PYLIDC_VERSION} is required"
            )
        try:
            pylidc_module = importlib.import_module("pylidc")
            utils = importlib.import_module("pylidc.utils")
            consensus = utils.consensus
            pydicom_module = importlib.import_module("pydicom")
            compatibility_modules = (
                importlib.import_module("pylidc.Annotation"),
                importlib.import_module("pylidc.Contour"),
            )
        except Exception as exc:
            raise ProviderCompatibilityError(
                f"Official pylidc=={PYLIDC_VERSION} could not be imported in this runtime: {exc!r}"
            ) from exc
        defaults = inspect.signature(pylidc_module.Scan.cluster_annotations).parameters
        expected = {"metric": "min", "tol": None, "factor": 0.9, "min_tol": 0.1}
        conflicts: dict[str, dict[str, Any]] = {}
        for name, value in expected.items():
            installed_default = defaults[name].default if name in defaults else "<missing>"
            if installed_default != value:
                conflicts[name] = {"expected": value, "installed": installed_default}
        if conflicts:
            raise ProviderCompatibilityError(
                "Installed official pylidc clustering defaults conflict with the documented "
                f"0.2.3 API: {conflicts!r}"
            )
        consensus_parameters = inspect.signature(consensus).parameters
        required_consensus_parameters = {"anns", "clevel", "pad", "ret_masks"}
        if not required_consensus_parameters.issubset(consensus_parameters):
            raise ProviderCompatibilityError(
                "Installed official pylidc consensus API lacks required parameters: "
                f"{sorted(required_consensus_parameters - set(consensus_parameters))}"
            )

        try:
            compatibility = PylidcNumpyCompatibility(np, compatibility_modules)
            compatibility.verify()
        except NumpyCompatibilityError as exc:
            raise ProviderCompatibilityError(
                "Could not establish scoped NumPy compatibility for official "
                f"pylidc=={PYLIDC_VERSION}: {exc}"
            ) from exc

        # Assign only after all imports, API checks, and compatibility checks
        # succeed, so a failed setup cannot be bypassed on a later call.
        self._pylidc = pylidc_module
        self._consensus = consensus
        self._pydicom = pydicom_module
        self._numpy_compatibility = compatibility

    def versions(self) -> Mapping[str, Any]:
        compatibility = (
            self._numpy_compatibility.report
            if self._numpy_compatibility is not None
            else pylidc_numpy_compatibility_report(np)
        )
        values: dict[str, Any] = {
            "provider": self.name,
            "numpy": np.__version__,
            "pylidc": None,
            "pydicom": None,
            "pylidc_numpy_compatibility": compatibility.to_dict(),
        }
        for package in ("pylidc", "pydicom"):
            try:
                values[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        return values

    def _dicom_directory(self, raw_scan: Any) -> tuple[str | None, bool]:
        if self.raw_root is None:
            raise ProviderCompatibilityError(
                "Annotation-database-only provider mode cannot inspect DICOM directories"
            )
        try:
            directory = Path(raw_scan.get_path_to_dicom_files()).resolve()
        except Exception:
            return None, False
        readable = directory.is_dir() and _is_within(directory, self.raw_root)
        if readable:
            # Validate an actual header without decoding pixel data. Directory
            # existence alone is not evidence that a DICOM series is readable.
            try:
                candidates = sorted(
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() == ".dcm"
                )
            except OSError:
                candidates = []
            if not candidates:
                readable = False
            else:
                try:
                    header = self._pydicom.dcmread(candidates[0], stop_before_pixels=True)
                    readable = all(
                        str(getattr(header, field, "")).strip()
                        == str(getattr(raw_scan, attribute, "")).strip()
                        for field, attribute in (
                            ("PatientID", "patient_id"),
                            ("StudyInstanceUID", "study_instance_uid"),
                            ("SeriesInstanceUID", "series_instance_uid"),
                        )
                    )
                except Exception:
                    readable = False
        return str(directory), readable

    def _load_dicoms(self, scan: ScanRecord) -> list[Any]:
        self._load()
        if self.raw_root is None:
            raise ProviderCompatibilityError(
                "Annotation-database-only provider mode cannot load DICOM pixels"
            )
        if (
            self._dicom_cache_series == scan.series_instance_uid
            and self._dicom_cache_images is not None
        ):
            return self._dicom_cache_images
        raw_scan = scan.provider_payload
        try:
            images = list(raw_scan.load_all_dicom_images(verbose=False))
        except Exception as exc:
            raise ProviderCompatibilityError(
                f"Could not load DICOM series {scan.series_instance_uid}: {exc}"
            ) from exc
        images.sort(
            key=lambda image: (
                float(image.ImagePositionPatient[2])
                if hasattr(image, "ImagePositionPatient")
                else float("inf"),
                str(getattr(image, "SOPInstanceUID", "")),
            )
        )
        self._dicom_cache_series = scan.series_instance_uid
        self._dicom_cache_images = images
        return images

    def _query_raw_scans(self) -> tuple[Any, ...]:
        self._load()
        if self._raw_scan_cache is not None:
            return self._raw_scan_cache
        try:
            raw_scans = list(self._pylidc.query(self._pylidc.Scan).all())
        except Exception as exc:
            raise ProviderCompatibilityError(
                f"Could not query the pylidc annotation database: {exc}"
            ) from exc
        raw_scans.sort(
            key=lambda scan: (
                str(getattr(scan, "patient_id", "")),
                str(getattr(scan, "study_instance_uid", "")),
                str(getattr(scan, "series_instance_uid", "")),
            )
        )
        self._raw_scan_cache = tuple(raw_scans)
        return self._raw_scan_cache

    def iter_scans(self) -> Iterable[ScanRecord]:
        self._load()
        if self.raw_root is None:
            raise ProviderCompatibilityError(
                "Use iter_annotation_scans() for a no-DICOM runtime smoke test"
            )
        if self._scan_cache is not None:
            return iter(self._scan_cache)
        records: list[ScanRecord] = []
        for raw in self._query_raw_scans():
            directory, readable = self._dicom_directory(raw)
            records.append(
                ScanRecord(
                    patient_id=str(getattr(raw, "patient_id", "")),
                    study_instance_uid=str(getattr(raw, "study_instance_uid", "")),
                    series_instance_uid=str(getattr(raw, "series_instance_uid", "")),
                    annotation_count=len(getattr(raw, "annotations", ()) or ()),
                    dicom_readable=readable,
                    annotation_source_available=True,
                    dicom_directory=directory,
                    provider_payload=raw,
                )
            )
        records.sort(
            key=lambda scan: (
                scan.patient_id,
                scan.study_instance_uid,
                scan.series_instance_uid,
            )
        )
        self._scan_cache = tuple(records)
        return iter(self._scan_cache)

    def iter_annotation_scans(self) -> Iterable[ScanRecord]:
        """Yield bundled-database records without claiming or touching DICOM readiness."""
        self._load()
        if self.raw_root is not None:
            raise ProviderCompatibilityError(
                "Annotation-only scan iteration requires PylidcProvider(raw_root=None)"
            )
        if self._annotation_scan_cache is not None:
            return iter(self._annotation_scan_cache)
        records: list[ScanRecord] = []
        for raw in self._query_raw_scans():
            try:
                z_positions = tuple(float(value) for value in raw.slice_zvals)
            except Exception as exc:
                raise ProviderCompatibilityError(
                    "Could not read slice z positions from the bundled pylidc database for "
                    f"{getattr(raw, 'series_instance_uid', '')}: {exc}"
                ) from exc
            records.append(
                ScanRecord(
                    patient_id=str(getattr(raw, "patient_id", "")),
                    study_instance_uid=str(getattr(raw, "study_instance_uid", "")),
                    series_instance_uid=str(getattr(raw, "series_instance_uid", "")),
                    annotation_count=len(getattr(raw, "annotations", ()) or ()),
                    dicom_readable=False,
                    annotation_source_available=True,
                    dicom_directory=None,
                    sop_instance_uids=(None,) * len(z_positions),
                    z_positions=z_positions,
                    provider_payload=raw,
                )
            )
        self._annotation_scan_cache = tuple(records)
        return iter(self._annotation_scan_cache)

    def _with_slice_inventory(self, scan: ScanRecord) -> ScanRecord:
        if (
            scan.z_positions
            and len(scan.sop_instance_uids) == len(scan.z_positions)
        ):
            return scan
        images = self._load_dicoms(scan)
        return ScanRecord(
            patient_id=scan.patient_id,
            study_instance_uid=scan.study_instance_uid,
            series_instance_uid=scan.series_instance_uid,
            annotation_count=scan.annotation_count,
            dicom_readable=scan.dicom_readable,
            annotation_source_available=scan.annotation_source_available,
            dicom_directory=scan.dicom_directory,
            sop_instance_uids=tuple(str(getattr(image, "SOPInstanceUID", "")) or None for image in images),
            z_positions=tuple(
                float(image.ImagePositionPatient[2])
                if hasattr(image, "ImagePositionPatient")
                else None
                for image in images
            ),
            provider_payload=scan.provider_payload,
        )

    def _annotation(self, raw: Any, scan: ScanRecord) -> AnnotationRecord:
        contours: list[ContourRecord] = []
        source_errors: list[str] = []
        for contour_index, contour in enumerate(getattr(raw, "contours", ()) or ()):
            try:
                matrix = np.asarray(contour.to_matrix(include_k=False))
                if matrix.ndim != 2 or matrix.shape[1] != 2:
                    raise ValueError("pylidc contour matrix is malformed")
                raw_index = getattr(contour, "image_k_position", None)
                slice_index = int(raw_index) if raw_index is not None else None
                if slice_index is not None and not 0 <= slice_index < len(scan.sop_instance_uids):
                    raise ValueError(f"slice index {slice_index} is outside the DICOM inventory")
                sop = scan.sop_instance_uids[slice_index] if slice_index is not None else None
                raw_z = getattr(contour, "image_z_position", None)
                z_position = None if raw_z is None else float(raw_z)
                if z_position is not None and not math.isfinite(z_position):
                    raise ValueError("contour z position is not finite")
                contours.append(
                    ContourRecord(
                        slice_index=slice_index,
                        points=tuple((float(row[0]), float(row[1])) for row in matrix),
                        sop_instance_uid=sop,
                        z_position=z_position,
                        inclusion=bool(getattr(contour, "inclusion", True)),
                    )
                )
            except Exception as exc:
                # Retain the annotation in the canonical ledger. Core validation
                # will assign a deterministic malformed signature and exclusion.
                source_errors.append(f"contour[{contour_index}]: {exc}")
        return AnnotationRecord(
            malignancy=getattr(raw, "malignancy", None),
            contours=tuple(contours),
            source_status="malformed" if source_errors else "available",
            source_error="; ".join(source_errors) if source_errors else None,
            provider_payload=raw,
        )

    def cluster_annotations(
        self, scan: ScanRecord, config: ClusteringConfig
    ) -> Sequence[NoduleClusterRecord]:
        self._load()
        scan_with_slices = self._with_slice_inventory(scan)
        if self._numpy_compatibility is None:
            raise ProviderCompatibilityError("pylidc NumPy compatibility is not initialized")
        try:
            with self._numpy_compatibility.activated():
                raw_clusters = scan.provider_payload.cluster_annotations(
                    metric=config.metric,
                    tol=config.tol,
                    factor=config.factor,
                    min_tol=config.min_tol,
                    return_distance_matrix=False,
                    verbose=False,
                )
        except NumpyCompatibilityError as exc:
            raise ProviderCompatibilityError(
                f"pylidc NumPy compatibility failed during clustering: {exc}"
            ) from exc
        except Exception as exc:
            raise ProviderCompatibilityError(
                f"pylidc clustering failed for {scan.series_instance_uid}: {exc}"
            ) from exc
        return tuple(
            NoduleClusterRecord(
                scan=scan_with_slices,
                annotations=tuple(self._annotation(annotation, scan_with_slices) for annotation in group),
            )
            for group in raw_clusters
        )

    def consensus_masks(
        self, cluster: NoduleClusterRecord, config: ConsensusConfig
    ) -> ConsensusVolumeRecord:
        self._load()
        raw_annotations = [annotation.provider_payload for annotation in cluster.annotations]
        if any(annotation is None for annotation in raw_annotations):
            raise ProviderCompatibilityError("pylidc annotation payload is unavailable")
        if self._numpy_compatibility is None:
            raise ProviderCompatibilityError("pylidc NumPy compatibility is not initialized")
        try:
            with self._numpy_compatibility.activated():
                _binary, bbox, masks = self._consensus(
                    raw_annotations,
                    clevel=config.clevel,
                    pad=config.pad,
                    ret_masks=True,
                )
        except NumpyCompatibilityError as exc:
            raise ProviderCompatibilityError(
                f"pylidc NumPy compatibility failed during consensus: {exc}"
            ) from exc
        except Exception as exc:
            raise ProviderCompatibilityError(
                f"pylidc consensus failed for {cluster.scan.series_instance_uid}: {exc}"
            ) from exc
        bounds = tuple((int(axis.start), int(axis.stop)) for axis in bbox)
        return ConsensusVolumeRecord(
            individual_masks=np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0),
            bbox=bounds,  # type: ignore[arg-type]
        )

    def load_dicom_slice(self, scan: ScanRecord, slice_index: int) -> DicomSliceRecord:
        images = self._load_dicoms(scan)
        if slice_index < 0 or slice_index >= len(images):
            raise IndexError(
                f"DICOM slice index {slice_index} is outside series with {len(images)} slices"
            )
        dataset = images[slice_index]
        try:
            pixels = np.asarray(dataset.pixel_array)
        except Exception as exc:
            raise ProviderCompatibilityError(
                f"Could not decode DICOM pixels for {scan.series_instance_uid}: {exc}"
            ) from exc
        return DicomSliceRecord(
            pixels=pixels,
            rescale_slope=float(getattr(dataset, "RescaleSlope", 1.0)),
            rescale_intercept=float(getattr(dataset, "RescaleIntercept", 0.0)),
            patient_id=str(getattr(dataset, "PatientID", "")),
            study_instance_uid=str(getattr(dataset, "StudyInstanceUID", "")),
            series_instance_uid=str(getattr(dataset, "SeriesInstanceUID", "")),
            sop_instance_uid=str(getattr(dataset, "SOPInstanceUID", "")) or None,
            z_position=(
                float(dataset.ImagePositionPatient[2])
                if hasattr(dataset, "ImagePositionPatient")
                else None
            ),
            source_path=str(getattr(dataset, "filename", "")) or None,
        )
