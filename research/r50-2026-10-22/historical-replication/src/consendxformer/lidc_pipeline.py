"""Phase 2 inventory and canonical extraction orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .lidc_inventory import inventory_lidc
from .lidc_ledger import LEDGER_COLUMNS, analyze_cluster, build_ledger_row, reordered_cluster
from .lidc_masks import build_representative_masks, load_mask_artifact, write_mask_artifact
from .lidc_records import LIDCProvider, NoduleClusterRecord, ScanRecord
from .lidc_reconciliation import (
    ReconciliationError,
    SourceReadinessReport,
    frozen_series_index,
    leakage_check,
    reconcile_scan_against_index,
    validate_source_report_for_run,
)
from .manifest_guard import ManifestGuardReport
from .phase2_config import (
    clustering_config,
    consensus_config,
    phase2_config_sha256,
)
from .phase2_io import (
    mark_completed,
    prepare_resumable_run,
    write_checksum_ledger,
    write_csv_or_verify,
    write_json_or_verify,
    write_provenance_or_verify,
)
from .phase2_provenance import build_phase2_provenance
from .pylidc_provider import ProviderCompatibilityError


def run_inventory(
    *,
    provider: LIDCProvider,
    scans: Iterable[ScanRecord],
    manifest: ManifestGuardReport,
    source_report: SourceReadinessReport,
    config: dict,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict:
    validate_source_report_for_run(source_report, config["source"])
    config_hash = phase2_config_sha256(config)
    directory, _ = prepare_resumable_run(
        output_dir,
        project_root=project_root,
        configuration_sha256=config_hash,
        allow_resume=False,
    )
    summary, frame = inventory_lidc(
        provider, scans, manifest, clustering_config(config)
    )
    summary["source"] = source_report.to_dict()
    inventory_json = write_json_or_verify(directory / "inventory.json", summary)
    inventory_csv = write_csv_or_verify(directory / "nodule_inventory.csv", frame)
    provenance = build_phase2_provenance(
        project_root=project_root,
        run_type="lidc_inventory",
        config=config,
        configuration_sha256=config_hash,
        manifest=manifest,
        source_root=source_report.raw_root,
        source_report=source_report.to_dict(),
        provider_versions=provider.versions(),
        output_path=directory,
    )
    provenance_path = write_provenance_or_verify(directory / "provenance.json", provenance)
    checksum_path = write_checksum_ledger(
        directory,
        [
            inventory_json.relative_to(directory),
            inventory_csv.relative_to(directory),
            provenance_path.relative_to(directory),
            "run_configuration.json",
        ],
    )
    mark_completed(directory, checksum_path)
    return summary


def _existing_mask(
    path: Path, annotation_sha256s: tuple[str, ...]
) -> tuple[object | None, str | None]:
    if not path.exists():
        return None, None
    record = load_mask_artifact(path)
    if record.annotation_sha256s != annotation_sha256s:
        raise RuntimeError(
            f"Existing mask artifact has different annotation signatures: {path}"
        )
    from .hashing import sha256_file

    return record, sha256_file(path)


def run_extraction(
    *,
    provider: LIDCProvider,
    scans: Iterable[ScanRecord],
    manifest: ManifestGuardReport,
    source_report: SourceReadinessReport,
    config: dict,
    output_dir: str | Path,
    project_root: str | Path,
    allow_resume: bool = False,
) -> pd.DataFrame:
    validate_source_report_for_run(source_report, config["source"])
    config_hash = phase2_config_sha256(config)
    directory, _resumed = prepare_resumable_run(
        output_dir,
        project_root=project_root,
        configuration_sha256=config_hash,
        allow_resume=allow_resume,
    )
    masks_dir = directory / config["output"]["mask_directory"]
    masks_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_series_index(manifest)
    cluster_cfg = clustering_config(config)
    consensus_cfg = consensus_config(config)
    ledger_rows: list[dict] = []
    mask_relative_paths: list[str] = []
    seen_nodule_ids: set[str] = set()

    for scan in scans:
        try:
            identity = reconcile_scan_against_index(scan, frozen)
            reconciliation_error = None
        except ReconciliationError as exc:
            identity = None
            reconciliation_error = str(exc)
        clusters: list[tuple[str, NoduleClusterRecord, object]] = []
        for cluster in provider.cluster_annotations(scan, cluster_cfg):
            analysis = analyze_cluster(cluster)
            clusters.append((analysis.identity.nodule_sha256, cluster, analysis))
        for _digest, cluster, analysis_obj in sorted(clusters, key=lambda item: item[0]):
            analysis = analysis_obj
            if analysis.identity.nodule_id in seen_nodule_ids:
                raise RuntimeError(f"Duplicate stable nodule ID: {analysis.identity.nodule_id}")
            seen_nodule_ids.add(analysis.identity.nodule_id)
            ordered_cluster = reordered_cluster(cluster, analysis)
            mask_path = masks_dir / f"{analysis.identity.nodule_sha256}.npz"
            representative = None
            mask_sha = None
            mask_error = None
            try:
                representative, mask_sha = _existing_mask(
                    mask_path, analysis.identity.annotation_sha256s
                )
                if representative is None:
                    consensus = provider.consensus_masks(ordered_cluster, consensus_cfg)
                    representative = build_representative_masks(
                        consensus,
                        ordered_cluster.scan,
                        analysis.identity.annotation_sha256s,
                    )
                    _written, mask_sha = write_mask_artifact(mask_path, representative)
            except ProviderCompatibilityError:
                # A broken official runtime/compatibility layer is systemic,
                # never a per-annotation malformed-data exclusion.
                raise
            except Exception as exc:
                mask_error = str(exc)
                representative = None
                mask_sha = None
            relative_mask = (
                mask_path.relative_to(directory).as_posix() if representative is not None else None
            )
            if relative_mask is not None:
                mask_relative_paths.append(relative_mask)
            ledger_rows.append(
                build_ledger_row(
                    ordered_cluster,
                    analysis,
                    frozen_identity=identity,
                    reconciliation_error=reconciliation_error,
                    representative_masks=representative,
                    mask_path=relative_mask,
                    mask_sha256=mask_sha,
                    mask_error=mask_error,
                )
            )

    ledger = pd.DataFrame(ledger_rows, columns=LEDGER_COLUMNS)
    if not ledger.empty:
        ledger = ledger.sort_values("nodule_sha256", kind="stable").reset_index(drop=True)
    ledger_path = write_csv_or_verify(
        directory / config["output"]["canonical_ledger"], ledger
    )
    leakage = leakage_check(ledger_rows, manifest)
    leakage["passed"] = not leakage["violations"]
    leakage_path = write_json_or_verify(directory / "leakage_check.json", leakage)
    provenance = build_phase2_provenance(
        project_root=project_root,
        run_type="canonical_lidc_extraction",
        config=config,
        configuration_sha256=config_hash,
        manifest=manifest,
        source_root=source_report.raw_root,
        source_report=source_report.to_dict(),
        provider_versions=provider.versions(),
        output_path=directory,
    )
    provenance_path = write_provenance_or_verify(directory / "provenance.json", provenance)
    checksum_members = [
        ledger_path.relative_to(directory).as_posix(),
        leakage_path.relative_to(directory).as_posix(),
        provenance_path.relative_to(directory).as_posix(),
        "run_configuration.json",
        *sorted(set(mask_relative_paths)),
    ]
    checksum_path = write_checksum_ledger(directory, checksum_members)
    mark_completed(directory, checksum_path)
    return ledger
