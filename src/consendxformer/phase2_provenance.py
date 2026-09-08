"""Machine-readable provenance for Phase 2 source, extraction, and rendering runs."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np

from .baseline_provenance import project_git_state, utc_timestamp
from .manifest_guard import ManifestGuardReport


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def phase2_versions(provider_versions: Mapping[str, str | None] | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydicom": _version("pydicom"),
        "pylidc": _version("pylidc"),
    }
    if provider_versions:
        values["provider"] = dict(provider_versions)
    return values


def build_phase2_provenance(
    *,
    project_root: str | Path,
    run_type: str,
    config: Mapping[str, Any],
    configuration_sha256: str,
    manifest: ManifestGuardReport,
    source_root: str | Path,
    source_report: Mapping[str, Any],
    provider_versions: Mapping[str, str | None],
    output_path: str | Path,
    output_resolution: int | None = None,
    hu_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": utc_timestamp(),
        "run_type": run_type,
        "project_git": project_git_state(project_root),
        "phase0_manifest": manifest.to_dict(),
        "source_root": str(Path(source_root).resolve()),
        "source_validation": dict(source_report),
        "raw_source_verification_level": "identity_and_header_inventory",
        "full_dicom_file_cryptographic_hashing_performed": False,
        "versions": phase2_versions(provider_versions),
        "clustering": dict(config["clustering"]),
        "consensus": dict(config["consensus"]),
        "representative_slice": dict(config["representative_slice"]),
        "crop": dict(config["crop"]),
        "hu": dict(hu_configuration if hu_configuration is not None else config["hu"]),
        "output_resolution": output_resolution,
        "configuration_sha256": configuration_sha256,
        "output_path": str(Path(output_path).resolve()),
        "performance_claimed": False,
    }
