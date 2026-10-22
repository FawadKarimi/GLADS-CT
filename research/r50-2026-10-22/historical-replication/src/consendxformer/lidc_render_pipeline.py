"""Render canonical Phase 2 nodules into explicit 2D model-ready views."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from .hashing import sha256_file, sha256_json
from .lidc_masks import load_mask_artifact
from .lidc_records import LIDCProvider, ScanRecord
from .lidc_reconciliation import (
    SourceReadinessReport,
    frozen_series_index,
    validate_source_report_for_run,
)
from .lidc_render import render_nodule_roi, write_rendered_roi
from .manifest_guard import ManifestGuardReport
from .phase2_config import CropConfig, HUWindow, crop_config, validate_phase2_config
from .phase2_io import (
    mark_completed,
    prepare_resumable_run,
    write_checksum_ledger,
    write_csv_or_verify,
    write_provenance_or_verify,
)
from .phase2_provenance import build_phase2_provenance


RENDERED_COLUMNS = (
    "nodule_id",
    "nodule_sha256",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "split",
    "resolution",
    "image_path",
    "rendered_mask_path",
    "paper_label",
    "paper_primary_eligible",
    "boundary_zone",
    "n_valid_malignancy_ratings",
    "has_four_valid_readers",
    "crop_box_json",
    "hu_window_min",
    "hu_window_max",
    "hu_provenance",
)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", ""}:
        return False
    raise ValueError(f"Invalid boolean value in canonical ledger: {value!r}")


def _resolve_inside(path: Path, root: Path) -> Path:
    target = (root / path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Canonical artifact path escapes dataset root: {path}") from exc
    return target


def run_rendering(
    *,
    provider: LIDCProvider,
    scans: tuple[ScanRecord, ...],
    manifest: ManifestGuardReport,
    source_report: SourceReadinessReport,
    config: dict,
    canonical_ledger_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    hu_window: HUWindow,
    resolution: int,
) -> pd.DataFrame:
    validate_phase2_config(config)
    validate_source_report_for_run(source_report, config["source"])
    if resolution not in config["rendering"]["supported_resolutions"]:
        raise ValueError(f"Unsupported output resolution: {resolution}")
    ledger_path = Path(canonical_ledger_path).resolve()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Canonical nodule ledger does not exist: {ledger_path}")
    canonical_root = ledger_path.parent
    ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    required = {
        "nodule_id",
        "nodule_sha256",
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "split",
        "representative_slice_index",
        "representative_sop_instance_uid",
        "mask_path",
        "mask_sha256",
        "paper_label",
        "paper_primary_eligible",
        "boundary_zone",
        "n_valid_malignancy_ratings",
        "has_four_valid_readers",
        "source_validity_status",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Canonical ledger is missing required columns: {missing}")
    render_config = {
        "phase2_configuration": config,
        "canonical_ledger_sha256": sha256_file(ledger_path),
        "hu": hu_window.to_dict(),
        "crop": crop_config(config).to_dict(),
        "resolution": resolution,
    }
    render_hash = sha256_json(render_config)
    directory, _ = prepare_resumable_run(
        output_dir,
        project_root=project_root,
        configuration_sha256=render_hash,
        allow_resume=False,
    )
    scan_by_series = {scan.series_instance_uid: scan for scan in scans}
    if len(scan_by_series) != len(scans):
        raise ValueError("Provider returned duplicate series during rendering")
    frozen = frozen_series_index(manifest)
    rows: list[dict] = []
    generated_files: list[str] = []
    crop = crop_config(config)
    for row in ledger.sort_values("nodule_sha256", kind="stable").itertuples(index=False):
        if not str(row.source_validity_status).startswith("valid"):
            continue
        identity = frozen.get(row.series_instance_uid)
        if identity is None or identity.patient_id != row.patient_id or identity.split != row.split:
            raise ValueError(f"Canonical row no longer matches frozen identity: {row.nodule_id}")
        scan = scan_by_series.get(row.series_instance_uid)
        if scan is None:
            raise FileNotFoundError(f"Raw source lacks series needed by {row.nodule_id}")
        mask_path = _resolve_inside(Path(row.mask_path), canonical_root)
        if sha256_file(mask_path) != row.mask_sha256:
            raise ValueError(f"Canonical mask checksum mismatch: {row.nodule_id}")
        masks = load_mask_artifact(mask_path)
        slice_index = int(row.representative_slice_index)
        dicom = provider.load_dicom_slice(scan, slice_index)
        if (
            dicom.patient_id != row.patient_id
            or dicom.study_instance_uid != row.study_instance_uid
            or dicom.series_instance_uid != row.series_instance_uid
        ):
            raise ValueError(f"DICOM identity mismatch while rendering {row.nodule_id}")
        if row.representative_sop_instance_uid and (
            dicom.sop_instance_uid != row.representative_sop_instance_uid
        ):
            raise ValueError(f"DICOM SOPInstanceUID mismatch while rendering {row.nodule_id}")
        rendered = render_nodule_roi(
            dicom,
            masks,
            hu_window=hu_window,
            crop_config=crop,
            output_resolution=resolution,
        )
        relative_dir = Path(f"resolution_{resolution}")
        paths = write_rendered_roi(directory / relative_dir, row.nodule_id, rendered)
        image_relative = Path(paths["image_path"]).relative_to(directory).as_posix()
        mask_relative = Path(paths["mask_path"]).relative_to(directory).as_posix()
        generated_files.extend((image_relative, mask_relative))
        rows.append(
            {
                "nodule_id": row.nodule_id,
                "nodule_sha256": row.nodule_sha256,
                "patient_id": row.patient_id,
                "study_instance_uid": row.study_instance_uid,
                "series_instance_uid": row.series_instance_uid,
                "split": row.split,
                "resolution": resolution,
                "image_path": image_relative,
                "rendered_mask_path": mask_relative,
                "paper_label": row.paper_label,
                "paper_primary_eligible": _bool(row.paper_primary_eligible),
                "boundary_zone": _bool(row.boundary_zone),
                "n_valid_malignancy_ratings": int(row.n_valid_malignancy_ratings),
                "has_four_valid_readers": _bool(row.has_four_valid_readers),
                "crop_box_json": json.dumps(rendered.crop_box.to_list(), separators=(",", ":")),
                "hu_window_min": hu_window.minimum,
                "hu_window_max": hu_window.maximum,
                "hu_provenance": hu_window.provenance,
            }
        )
    rendered_ledger = pd.DataFrame(rows, columns=RENDERED_COLUMNS)
    rendered_path = write_csv_or_verify(directory / "rendered_rois.csv", rendered_ledger)
    provenance_config = deepcopy(config)
    provenance = build_phase2_provenance(
        project_root=project_root,
        run_type="model_ready_2d_rendering",
        config=provenance_config,
        configuration_sha256=render_hash,
        manifest=manifest,
        source_root=source_report.raw_root,
        source_report=source_report.to_dict(),
        provider_versions=provider.versions(),
        output_path=directory,
        output_resolution=resolution,
        hu_configuration=hu_window.to_dict(),
    )
    provenance_path = write_provenance_or_verify(directory / "provenance.json", provenance)
    checksum = write_checksum_ledger(
        directory,
        [
            "run_configuration.json",
            rendered_path.relative_to(directory).as_posix(),
            provenance_path.relative_to(directory).as_posix(),
            *generated_files,
        ],
    )
    mark_completed(directory, checksum)
    return rendered_ledger
