from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from consendxformer.lidc_reconciliation import (
    ReconciliationError,
    SourceReadinessError,
    assess_source_readiness,
    leakage_check,
    reconcile_scan,
)
from consendxformer.lidc_records import ScanRecord
from consendxformer.manifest_guard import FROZEN_BUNDLE_SHA256, validate_frozen_manifest_bundle
from phase2_fakes import scan_from_manifest


@pytest.fixture(scope="module")
def manifest():
    return validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")


def test_phase0_manifest_remains_frozen_and_patient_split_propagates(manifest):
    assert manifest.bundle_sha256 == FROZEN_BUNDLE_SHA256
    scan = scan_from_manifest(manifest, 0)
    identity = reconcile_scan(scan, manifest)
    assert identity.patient_id == scan.patient_id
    assert identity.series_instance_uid == scan.series_instance_uid
    assert identity.split == manifest.series_split.iloc[0]["split"]


def test_multiple_series_for_same_patient_preserve_one_split(manifest):
    duplicate_patients = manifest.series_split[
        manifest.series_split.duplicated("patient_id", keep=False)
    ]
    group = next(iter(duplicate_patients.groupby("patient_id")))[1]
    assert len(group) >= 2
    assert group["split"].nunique() == 1
    identities = []
    for index in group.index[:2]:
        identities.append(reconcile_scan(scan_from_manifest(manifest, index), manifest))
    assert identities[0].patient_id == identities[1].patient_id
    assert identities[0].split == identities[1].split
    assert identities[0].series_instance_uid != identities[1].series_instance_uid


def test_unknown_series_and_patient_mismatch_fail_reconciliation(manifest):
    scan = scan_from_manifest(manifest, 0)
    with pytest.raises(ReconciliationError, match="Unknown SeriesInstanceUID"):
        reconcile_scan(replace(scan, series_instance_uid="unknown-series"), manifest)
    with pytest.raises(ReconciliationError, match="Series/patient mismatch"):
        reconcile_scan(replace(scan, patient_id="LIDC-IDRI-9999"), manifest)


def test_partial_source_is_explicitly_labeled_and_never_final(tmp_path, manifest):
    (tmp_path / "source-marker").write_text("partial\n", encoding="utf-8")
    scan = scan_from_manifest(manifest, 0)
    report = assess_source_readiness(
        tmp_path, [scan], manifest, mode="partial_development"
    )
    assert report.source_mode == "partial_development"
    assert report.final_dataset_eligible is False
    assert len(report.missing_frozen_series) == 1017
    assert report.source_patient_count == 1


def test_strict_full_source_rejects_incomplete_coverage(tmp_path, manifest):
    (tmp_path / "source-marker").write_text("partial\n", encoding="utf-8")
    with pytest.raises(SourceReadinessError, match="STRICT FULL SOURCE.*missing frozen"):
        assess_source_readiness(
            tmp_path,
            [scan_from_manifest(manifest, 0)],
            manifest,
            mode="strict_full",
        )


def test_strict_full_source_accepts_exact_frozen_identity_inventory(tmp_path, manifest):
    (tmp_path / "source-marker").write_text("full\n", encoding="utf-8")
    scans = [
        ScanRecord(
            patient_id=str(row.patient_id),
            study_instance_uid=str(row.study_instance_uid),
            series_instance_uid=str(row.series_instance_uid),
            annotation_count=1 if index == 0 else 0,
            dicom_readable=True,
            annotation_source_available=True,
        )
        for index, row in enumerate(manifest.series_split.itertuples(index=False))
    ]
    report = assess_source_readiness(tmp_path, scans, manifest, mode="strict_full")
    assert report.final_dataset_eligible is True
    assert report.source_patient_count == 1010
    assert report.source_series_count == 1018
    assert report.completeness_errors == ()


def test_duplicate_series_are_reported_in_partial_mode(tmp_path, manifest):
    (tmp_path / "source-marker").write_text("partial\n", encoding="utf-8")
    scan = scan_from_manifest(manifest, 0)
    report = assess_source_readiness(
        tmp_path, [scan, scan], manifest, mode="partial_development"
    )
    assert report.duplicate_series == (scan.series_instance_uid,)
    assert any("duplicate series" in error for error in report.completeness_errors)


def test_leakage_report_detects_and_accepts_frozen_assignments(manifest):
    scan = scan_from_manifest(manifest, 0)
    identity = reconcile_scan(scan, manifest)
    good = leakage_check(
        [
            {
                "patient_id": scan.patient_id,
                "series_instance_uid": scan.series_instance_uid,
                "split": identity.split,
            }
        ],
        manifest,
    )
    assert good["passed"] is True
    bad = leakage_check(
        [
            {
                "patient_id": scan.patient_id,
                "series_instance_uid": scan.series_instance_uid,
                "split": "wrong",
            }
        ],
        manifest,
    )
    assert bad["passed"] is False
