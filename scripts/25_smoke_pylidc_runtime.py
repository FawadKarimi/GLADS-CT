#!/usr/bin/env python3
"""Exercise official pylidc clustering/consensus without a DICOM collection."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.lidc_masks import soft_consensus  # noqa: E402
from consendxformer.phase2_config import (  # noqa: E402
    clustering_config,
    consensus_config,
    load_phase2_config,
)
from consendxformer.pylidc_provider import (  # noqa: E402
    ProviderDependencyError,
    PylidcProvider,
)


def _distribution_version() -> str | None:
    try:
        return importlib.metadata.version("pylidc")
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test real official pylidc annotation clustering and consensus "
            "using its bundled database; no DICOM/source-readiness claim is made"
        )
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase2.yaml"))
    args = parser.parse_args(argv)

    print(f"NUMPY_VERSION={np.__version__}")
    print(f"PYLIDC_VERSION={_distribution_version() or 'NOT_INSTALLED'}")
    print("DICOM_ARCHIVE_TESTED=false")
    print("SOURCE_READINESS_TESTED=false")
    try:
        config = load_phase2_config(args.config)
        provider = PylidcProvider(raw_root=None)
        scans = tuple(provider.iter_annotation_scans())
        candidates = [scan for scan in scans if scan.annotation_count >= 2]
        if not candidates:
            raise RuntimeError(
                "The bundled official pylidc database has no scan with multiple annotations"
            )
        scan = candidates[0]
        clusters = tuple(provider.cluster_annotations(scan, clustering_config(config)))
        if not clusters:
            raise RuntimeError("Real Scan.cluster_annotations() returned no physical cluster")
        cluster = next((value for value in clusters if value.annotations), None)
        if cluster is None:
            raise RuntimeError("Real clustering returned no non-empty annotation cluster")
        consensus = provider.consensus_masks(cluster, consensus_config(config))
        masks = np.asarray(consensus.individual_masks)
        if masks.ndim != 4:
            raise RuntimeError(f"Individual masks have unexpected shape {masks.shape}")
        if masks.shape[0] != len(cluster.annotations):
            raise RuntimeError(
                "Individual-mask count does not match the real clustered annotations"
            )
        if any(size <= 0 for size in masks.shape) or not bool(masks.any()):
            raise RuntimeError("Real consensus returned empty individual-mask data")
        if any(not bool(mask.any()) for mask in masks):
            raise RuntimeError("At least one real individual annotation mask is empty")
        soft = soft_consensus(masks)
        if soft.shape != masks.shape[1:] or not bool(np.isfinite(soft).all()):
            raise RuntimeError("Soft consensus shape or values are invalid")
        if not bool(soft.any()):
            raise RuntimeError("Soft consensus is empty")
        annotation_module = importlib.import_module("pylidc.Annotation")
        contour_module = importlib.import_module("pylidc.Contour")
        if annotation_module.np is not np or contour_module.np is not np:
            raise RuntimeError(
                "Scoped compatibility did not restore official pylidc NumPy bindings"
            )
        compatibility = provider.versions()["pylidc_numpy_compatibility"]
        print(f"SCAN_PATIENT_ID={scan.patient_id}")
        print(f"SCAN_SERIES_INSTANCE_UID={scan.series_instance_uid}")
        print(f"CLUSTER_ANNOTATIONS={len(cluster.annotations)}")
        print("INDIVIDUAL_MASK_SHAPE=" + "x".join(str(value) for value in masks.shape))
        print(
            "PYLIDC_NUMPY_COMPATIBILITY_ALIASES="
            + ",".join(compatibility["aliases"])
        )
        print("PYLIDC_ALGORITHM_MODIFIED=false")
        print("PYLIDC_MODULE_BINDINGS_RESTORED=true")
        print("PYLIDC_RUNTIME_SMOKE=PASS")
        return 0
    except ProviderDependencyError as exc:
        print(f"PYLIDC_RUNTIME_ERROR={type(exc).__module__}.{type(exc).__qualname__}: {exc}")
        print("PYLIDC_RUNTIME_SMOKE=NOT_RUN")
        return 2
    except Exception as exc:
        print(f"PYLIDC_RUNTIME_ERROR={type(exc).__module__}.{type(exc).__qualname__}: {exc}")
        traceback.print_exc()
        print("PYLIDC_RUNTIME_SMOKE=FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
