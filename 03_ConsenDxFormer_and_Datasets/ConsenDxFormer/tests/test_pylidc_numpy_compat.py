from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.manifest_guard import validate_frozen_manifest_bundle
from consendxformer.lidc_records import ScanRecord
from consendxformer.phase2_config import (
    ClusteringConfig,
    ConsensusConfig,
    load_phase2_config,
    phase2_config_sha256,
)
from consendxformer.phase2_provenance import build_phase2_provenance, phase2_versions
from consendxformer.pylidc_numpy_compat import (
    NumpyCompatibilityError,
    PylidcNumpyCompatibility,
)
from consendxformer.pylidc_provider import (
    ProviderCompatibilityError,
    ProviderDependencyError,
    PylidcProvider,
)


def _numpy_without_legacy_aliases() -> ModuleType:
    facade = ModuleType("numpy_without_legacy_aliases")
    for name in ("array", "zeros", "ones", "asarray", "stack", "float32", "uint8"):
        setattr(facade, name, getattr(np, name))
    facade.__version__ = np.__version__
    return facade


def _pylidc_owner(name: str, numpy_module: ModuleType) -> ModuleType:
    owner = ModuleType(name)
    owner.np = numpy_module
    return owner


def test_missing_int_and_bool_are_scoped_python_aliases_and_deterministic():
    facade = _numpy_without_legacy_aliases()
    contour = _pylidc_owner("pylidc.Contour", facade)
    annotation = _pylidc_owner("pylidc.Annotation", facade)
    before_numpy_namespace = dict(vars(facade))
    first = PylidcNumpyCompatibility(facade, (contour, annotation))
    second = PylidcNumpyCompatibility(facade, (annotation, contour))
    assert first.report.to_dict() == second.report.to_dict() == {
        "required": True,
        "aliases": ["int", "bool"],
        "algorithm_modified": False,
        "numpy_module_modified": False,
        "pylidc_source_modified": False,
        "strategy": "temporary_pylidc_module_numpy_proxy",
    }

    with first.activated():
        assert contour.np.int is int
        assert annotation.np.bool is bool
        assert contour.np.array is np.array
        assert contour.np.array([1.2]).astype(contour.np.int).dtype.kind in "iu"
        assert annotation.np.zeros(2, dtype=annotation.np.bool).dtype == np.dtype(bool)
        with pytest.raises(AttributeError):
            _ = annotation.np.float

    assert contour.np is facade
    assert annotation.np is facade
    assert vars(facade) == before_numpy_namespace


def test_compatibility_context_restores_modules_after_official_call_failure():
    facade = _numpy_without_legacy_aliases()
    contour = _pylidc_owner("pylidc.Contour", facade)
    compatibility = PylidcNumpyCompatibility(facade, (contour,))
    with pytest.raises(RuntimeError, match="official failure"):
        with compatibility.activated():
            assert contour.np.int is int
            raise RuntimeError("official failure")
    assert contour.np is facade


def test_provider_supplies_legacy_aliases_during_unchanged_official_calls(tmp_path):
    facade = _numpy_without_legacy_aliases()
    contour_owner = _pylidc_owner("pylidc.Contour", facade)
    annotation_owner = _pylidc_owner("pylidc.Annotation", facade)
    compatibility = PylidcNumpyCompatibility(
        facade, (contour_owner, annotation_owner)
    )

    class Contour:
        image_k_position = 0
        image_z_position = 0.0
        inclusion = True

        def to_matrix(self, include_k=True):
            assert include_k is False
            return np.array([[0, 0], [0, 1], [1, 1]])

    class Annotation:
        malignancy = 3
        contours = (Contour(),)

    class RawScan:
        def __init__(self):
            self.kwargs = None
            self.int_alias_seen = None

        def cluster_annotations(self, **kwargs):
            self.kwargs = kwargs
            self.int_alias_seen = contour_owner.np.int
            return [[Annotation()]]

    raw_scan = RawScan()
    scan = ScanRecord(
        patient_id="LIDC-IDRI-0001",
        study_instance_uid="study",
        series_instance_uid="series",
        annotation_count=1,
        dicom_readable=False,
        annotation_source_available=True,
        sop_instance_uids=(None,),
        z_positions=(0.0,),
        provider_payload=raw_scan,
    )
    provider = PylidcProvider(tmp_path)
    provider._pylidc = object()
    provider._numpy_compatibility = compatibility
    clusters = provider.cluster_annotations(scan, ClusteringConfig())
    assert raw_scan.int_alias_seen is int
    assert raw_scan.kwargs == {
        "metric": "min",
        "tol": None,
        "factor": 0.9,
        "min_tol": 0.1,
        "return_distance_matrix": False,
        "verbose": False,
    }

    def consensus(annotations, **kwargs):
        assert len(annotations) == 1
        assert annotation_owner.np.bool is bool
        mask = annotation_owner.np.ones((2, 2, 1), dtype=annotation_owner.np.bool)
        return mask, (slice(0, 2), slice(0, 2), slice(0, 1)), [mask]

    provider._consensus = consensus
    result = provider.consensus_masks(clusters[0], ConsensusConfig())
    assert result.individual_masks.shape == (1, 2, 2, 1)
    assert contour_owner.np is facade
    assert annotation_owner.np is facade


class _LockedPylidcModule(ModuleType):
    def __init__(self, name: str, numpy_module: ModuleType) -> None:
        super().__init__(name)
        ModuleType.__setattr__(self, "np", numpy_module)
        ModuleType.__setattr__(self, "locked", True)

    def __setattr__(self, name, value):
        if name == "np" and getattr(self, "locked", False):
            raise RuntimeError("module namespace is locked")
        ModuleType.__setattr__(self, name, value)


def test_compatibility_setup_failure_is_explicit_and_repeatable():
    facade = _numpy_without_legacy_aliases()
    locked = _LockedPylidcModule("pylidc.Contour", facade)
    compatibility = PylidcNumpyCompatibility(facade, (locked,))
    for _attempt in range(2):
        with pytest.raises(NumpyCompatibilityError, match="scoped pylidc"):
            compatibility.verify()
        assert locked.np is facade


@pytest.mark.parametrize("version", ["0.2.2", "0.2.3.post1", "0.2.3+fork"])
def test_provider_rejects_every_nonexact_pylidc_distribution_without_fork_fallback(
    monkeypatch, tmp_path, version
):
    imported: list[str] = []

    monkeypatch.setattr(importlib.metadata, "version", lambda package: version)

    def forbidden_import(name):
        imported.append(name)
        raise AssertionError("No import is allowed after rejecting the distribution version")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    provider = PylidcProvider(tmp_path)
    with pytest.raises(ProviderDependencyError, match="exactly 0.2.3"):
        provider._load()
    assert imported == []


def test_provider_load_wraps_compatibility_failure_transactionally(monkeypatch, tmp_path):
    class Scan:
        def cluster_annotations(
            self,
            metric="min",
            tol=None,
            factor=0.9,
            min_tol=0.1,
            return_distance_matrix=False,
            verbose=True,
        ):
            del metric, tol, factor, min_tol, return_distance_matrix, verbose

    pylidc = ModuleType("pylidc")
    pylidc.Scan = Scan
    utils = ModuleType("pylidc.utils")

    def consensus(anns, clevel=0.5, pad=None, ret_masks=True):
        del anns, clevel, pad, ret_masks

    utils.consensus = consensus
    annotation = _pylidc_owner("pylidc.Annotation", np)
    contour = _LockedPylidcModule("pylidc.Contour", np)
    pydicom = ModuleType("pydicom")
    modules = {
        "pylidc": pylidc,
        "pylidc.utils": utils,
        "pydicom": pydicom,
        "pylidc.Annotation": annotation,
        "pylidc.Contour": contour,
    }
    real_version = importlib.metadata.version

    def version(package):
        return "0.2.3" if package == "pylidc" else real_version(package)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])
    provider = PylidcProvider(tmp_path)
    for _attempt in range(2):
        with pytest.raises(ProviderCompatibilityError, match="scoped NumPy compatibility"):
            provider._load()
        assert provider._pylidc is None
        assert provider._numpy_compatibility is None


def test_compatibility_metadata_is_promoted_into_runtime_versions(tmp_path):
    provider = PylidcProvider(tmp_path)
    metadata = provider.versions()["pylidc_numpy_compatibility"]
    runtime = phase2_versions(provider.versions())
    assert runtime["numpy"] == np.__version__
    assert runtime["pylidc_numpy_compatibility"] == metadata
    assert metadata["algorithm_modified"] is False
    assert metadata["numpy_module_modified"] is False
    assert metadata["pylidc_source_modified"] is False
    config = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    manifest = validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")
    provenance = build_phase2_provenance(
        project_root=ROOT,
        run_type="compatibility_test",
        config=config,
        configuration_sha256=phase2_config_sha256(config),
        manifest=manifest,
        source_root=tmp_path,
        source_report={"source_mode": "test"},
        provider_versions=provider.versions(),
        output_path=tmp_path,
    )
    assert provenance["versions"]["pylidc_numpy_compatibility"] == metadata
