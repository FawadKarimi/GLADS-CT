"""Shared real-source setup for Phase 2 command-line entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .lidc_reconciliation import assess_source_readiness
from .manifest_guard import validate_frozen_manifest_bundle
from .phase2_config import load_phase2_config, validate_phase2_config
from .pylidc_provider import PylidcProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def prepare_real_runtime(
    *,
    config_path: str | Path,
    raw_root: str | Path,
    source_mode: str,
    permit_incomplete_report: bool = False,
):
    config = deepcopy(load_phase2_config(project_path(config_path)))
    config["source"]["raw_root"] = str(Path(raw_root).resolve())
    config["source"]["mode"] = source_mode
    validate_phase2_config(config)
    manifest = validate_frozen_manifest_bundle(
        project_path(config["frozen_manifest"]["directory"])
    )
    provider = PylidcProvider(raw_root)
    scans = tuple(provider.iter_scans())
    source_report = assess_source_readiness(
        raw_root,
        scans,
        manifest,
        mode=source_mode,
        raise_on_incomplete=not permit_incomplete_report,
    )
    return config, manifest, provider, scans, source_report
