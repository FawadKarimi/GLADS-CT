from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from consendxformer.baseline_metadata_export import export_baseline_metadata
from consendxformer.lidc_pipeline import run_extraction
from consendxformer.lidc_reconciliation import assess_source_readiness
from consendxformer.lidc_records import InMemoryLIDCProvider, ScanRecord
from consendxformer.lidc_render_pipeline import run_rendering
from consendxformer.manifest_guard import validate_frozen_manifest_bundle
from consendxformer.phase2_config import HUWindow, load_phase2_config
from consendxformer.phase2_io import write_json_new
from consendxformer.phase3_readiness import (
    PHASE3_CANONICAL_COLUMNS,
    Phase3ReadinessError,
    validate_phase3_readiness,
)
from phase2_fakes import cluster_with_ratings, consensus_for_cluster, dicom_for_scan


AVAILABLE_VIEW = "paper_aligned_available_reader_mean"
SYNTHETIC_VIEW = "synthetic_tiny_fixture"
SYNTHETIC_COLUMNS = (
    "patient_id",
    "image_path",
    "label",
    "nodule_id",
    "series_instance_uid",
    "split",
    "resolution",
    "n_valid_malignancy_ratings",
    "cohort_view",
    "data_origin",
)


@pytest.fixture(scope="module")
def manifest():
    return validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")


def _synthetic_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "SYNTHETIC-0002",
                "image_path": "images/two.npy",
                "label": "malignant",
                "nodule_id": "synthetic-nodule-2",
                "series_instance_uid": "synthetic-series-2",
                "split": "val",
                "resolution": "64",
                "n_valid_malignancy_ratings": "4",
                "cohort_view": SYNTHETIC_VIEW,
                "data_origin": "synthetic_non_research",
            },
            {
                "patient_id": "SYNTHETIC-0001",
                "image_path": "images/one.npy",
                "label": "benign",
                "nodule_id": "synthetic-nodule-1",
                "series_instance_uid": "synthetic-series-1",
                "split": "train",
                "resolution": "64",
                "n_valid_malignancy_ratings": "2",
                "cohort_view": SYNTHETIC_VIEW,
                "data_origin": "synthetic_non_research",
            },
        ],
        columns=SYNTHETIC_COLUMNS,
    )


def _synthetic_config(metadata: Path) -> dict:
    return {
        "mode": "synthetic",
        "metadata_path": str(metadata),
        "frozen_split_dir": str(ROOT / "artifacts" / "phase0" / "manifests"),
        "source_readiness_report": None,
        "canonical_dir": None,
        "rendered_dir": None,
        "cohort_view": SYNTHETIC_VIEW,
        "synthetic_marker": "SYNTHETIC_NON_RESEARCH_ONLY",
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def test_synthetic_mode_requires_explicit_non_research_dry_run(tmp_path, manifest):
    metadata = _write_csv(tmp_path / "synthetic.csv", _synthetic_rows())
    config = _synthetic_config(metadata)
    with pytest.raises(Phase3ReadinessError, match="synthetic_dry_run=True"):
        validate_phase3_readiness(
            config,
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=False,
        )
    bad_marker = dict(config, synthetic_marker="real")
    with pytest.raises(Phase3ReadinessError, match="synthetic_marker"):
        validate_phase3_readiness(
            bad_marker,
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=True,
        )


def test_synthetic_mode_is_usable_but_never_claims_real_readiness(tmp_path, manifest):
    metadata = _write_csv(tmp_path / "synthetic.csv", _synthetic_rows())
    report, canonical = validate_phase3_readiness(
        _synthetic_config(metadata),
        manifest=manifest,
        expected_resolution=64,
        project_root=ROOT,
        synthetic_dry_run=True,
    )
    assert report.ready_for_requested_mode is True
    assert report.synthetic_non_research is True
    assert report.real_research_ready is False
    assert report.software_annotation_runtime_ready is False
    assert report.dicom_source_ready is False
    assert report.extracted_cohort_ready is False
    assert report.source_readiness_validated is False
    assert report.phase2_artifacts_validated is False
    assert report.cohort_projection_validated is False
    assert tuple(canonical.columns) == PHASE3_CANONICAL_COLUMNS
    assert canonical["label_id"].tolist() == [2, 0]
    assert canonical["nodule_id"].tolist() == [
        "synthetic-nodule-2",
        "synthetic-nodule-1",
    ]
    assert report.sample_count == 2
    assert report.fingerprint.source_readiness_sha256 is None


def test_synthetic_fingerprints_separate_exact_ordered_and_logical_identity(
    tmp_path, manifest
):
    rows = _synthetic_rows()
    first = _write_csv(tmp_path / "first.csv", rows)
    second = _write_csv(tmp_path / "second.csv", rows.iloc[::-1].reset_index(drop=True))
    report_a, _ = validate_phase3_readiness(
        _synthetic_config(first),
        manifest=manifest,
        expected_resolution=64,
        project_root=ROOT,
        synthetic_dry_run=True,
    )
    report_b, _ = validate_phase3_readiness(
        _synthetic_config(second),
        manifest=manifest,
        expected_resolution=64,
        project_root=ROOT,
        synthetic_dry_run=True,
    )
    assert report_a.fingerprint.metadata_sha256 != report_b.fingerprint.metadata_sha256
    assert (
        report_a.fingerprint.logical_cohort_sha256
        == report_b.fingerprint.logical_cohort_sha256
    )
    assert (
        report_a.fingerprint.ordered_samples_sha256
        != report_b.fingerprint.ordered_samples_sha256
    )

    path_only = rows.copy()
    path_only.loc[0, "image_path"] = "different-machine/path.npy"
    third = _write_csv(tmp_path / "path-only.csv", path_only)
    report_c, _ = validate_phase3_readiness(
        _synthetic_config(third),
        manifest=manifest,
        expected_resolution=64,
        project_root=ROOT,
        synthetic_dry_run=True,
    )
    assert report_a.fingerprint.metadata_sha256 != report_c.fingerprint.metadata_sha256
    assert (
        report_a.fingerprint.logical_cohort_sha256
        == report_c.fingerprint.logical_cohort_sha256
    )
    assert (
        report_a.fingerprint.ordered_samples_sha256
        == report_c.fingerprint.ordered_samples_sha256
    )


def test_synthetic_origin_and_known_frozen_split_mismatches_fail(tmp_path, manifest):
    rows = _synthetic_rows()
    rows.loc[0, "data_origin"] = "real"
    metadata = _write_csv(tmp_path / "wrong-origin.csv", rows)
    with pytest.raises(Phase3ReadinessError, match="Every synthetic metadata row"):
        validate_phase3_readiness(
            _synthetic_config(metadata),
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=True,
        )

    frozen = manifest.series_split.iloc[0]
    rows = _synthetic_rows().iloc[[0]].copy()
    rows.loc[rows.index[0], "patient_id"] = str(frozen.patient_id)
    rows.loc[rows.index[0], "series_instance_uid"] = str(frozen.series_instance_uid)
    rows.loc[rows.index[0], "split"] = next(
        split for split in ("train", "val", "test") if split != str(frozen.split)
    )
    metadata = _write_csv(tmp_path / "wrong-split.csv", rows)
    with pytest.raises(Phase3ReadinessError, match="frozen series identity"):
        validate_phase3_readiness(
            _synthetic_config(metadata),
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=True,
        )


def test_invalid_cohort_view_and_synthetic_metadata_in_real_mode_fail(tmp_path, manifest):
    metadata = _write_csv(tmp_path / "synthetic.csv", _synthetic_rows())
    invalid = dict(_synthetic_config(metadata), cohort_view="implicit-cohort")
    with pytest.raises(Phase3ReadinessError, match="cohort_view"):
        validate_phase3_readiness(
            invalid,
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=True,
        )
    real = dict(
        _synthetic_config(metadata),
        mode="real",
        cohort_view=AVAILABLE_VIEW,
        synthetic_marker="SYNTHETIC_NON_RESEARCH_ONLY",
    )
    with pytest.raises(Phase3ReadinessError, match="forbids a synthetic marker"):
        validate_phase3_readiness(
            real,
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=False,
        )


def _official_versions() -> dict:
    return {
        "provider": "official_pylidc",
        "numpy": "2.5.2",
        "pylidc": "0.2.3",
        "pydicom": "3.0.2",
        "pylidc_numpy_compatibility": {
            "required": True,
            "aliases": ["int", "bool"],
            "algorithm_modified": False,
            "numpy_module_modified": False,
            "pylidc_source_modified": False,
            "strategy": "temporary_pylidc_module_numpy_proxy",
        },
    }


class _OfficialFixtureProvider(InMemoryLIDCProvider):
    name = "official_pylidc"

    def versions(self):
        return _official_versions()


def _full_real_fixture(tmp_path: Path, manifest):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "source-present").write_text("fixture\n", encoding="utf-8")
    scans: list[ScanRecord] = []
    for index, row in enumerate(manifest.series_split.itertuples(index=False)):
        scans.append(
            ScanRecord(
                patient_id=str(row.patient_id),
                study_instance_uid=str(row.study_instance_uid),
                series_instance_uid=str(row.series_instance_uid),
                annotation_count=3 if index == 0 else 0,
                dicom_readable=True,
                annotation_source_available=True,
                sop_instance_uids=tuple(f"sop-{value}" for value in range(8)),
                z_positions=tuple(float(value) for value in range(8)),
            )
        )
    first = scans[0]
    cluster = cluster_with_ratings(first, [1, 3, 5])
    provider = _OfficialFixtureProvider(
        scans,
        clusters={first.series_instance_uid: (cluster,)},
        consensus_factory=consensus_for_cluster,
        dicom_slices={(first.series_instance_uid, 3): dicom_for_scan(first, 3)},
    )
    readiness = assess_source_readiness(
        raw_root,
        scans,
        manifest,
        mode="strict_full",
    )
    source_payload = readiness.to_dict()
    source_payload.update(
        {
            "configuration_mode": "strict_full",
            "manifest_bundle_sha256": manifest.bundle_sha256,
            "provider_versions": _official_versions(),
            "full_dicom_file_cryptographic_hashing_performed": False,
        }
    )
    source_path = write_json_new(tmp_path / "source-readiness.json", source_payload)
    config = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    config["source"]["raw_root"] = str(raw_root.resolve())
    config["source"]["mode"] = "strict_full"
    canonical_dir = tmp_path / "canonical"
    run_extraction(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=readiness,
        config=config,
        output_dir=canonical_dir,
        project_root=ROOT,
    )
    rendered_dir = tmp_path / "rendered"
    rendered = run_rendering(
        provider=provider,
        scans=tuple(scans),
        manifest=manifest,
        source_report=readiness,
        config=config,
        canonical_ledger_path=canonical_dir / "canonical_nodule_ledger.csv",
        output_dir=rendered_dir,
        project_root=ROOT,
        hu_window=HUWindow(-1000, 400, "user_supplied"),
        resolution=64,
    )
    metadata_path = tmp_path / "baseline-metadata.csv"
    export_baseline_metadata(
        rendered,
        manifest,
        cohort_view=AVAILABLE_VIEW,
        rendered_root=rendered_dir,
        output_csv=metadata_path,
    )
    data_config = {
        "mode": "real",
        "metadata_path": str(metadata_path),
        "frozen_split_dir": str(ROOT / "artifacts" / "phase0" / "manifests"),
        "source_readiness_report": str(source_path),
        "canonical_dir": str(canonical_dir),
        "rendered_dir": str(rendered_dir),
        "cohort_view": AVAILABLE_VIEW,
        "synthetic_marker": None,
    }
    return data_config, rendered_dir


def test_real_mode_requires_complete_source_and_existing_phase2_chain(tmp_path, manifest):
    source = tmp_path / "incomplete.json"
    source.write_text("{}\n", encoding="utf-8")
    config = {
        "mode": "real",
        "metadata_path": str(tmp_path / "missing.csv"),
        "frozen_split_dir": str(ROOT / "artifacts" / "phase0" / "manifests"),
        "source_readiness_report": str(source),
        "canonical_dir": str(tmp_path / "missing-canonical"),
        "rendered_dir": str(tmp_path / "missing-rendered"),
        "cohort_view": AVAILABLE_VIEW,
        "synthetic_marker": None,
    }
    with pytest.raises(Phase3ReadinessError, match="lacks required fields"):
        validate_phase3_readiness(
            config,
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=False,
        )


def test_real_gate_validates_full_chain_projection_and_detects_tampering(tmp_path, manifest):
    config, rendered_dir = _full_real_fixture(tmp_path, manifest)
    report, canonical = validate_phase3_readiness(
        config,
        manifest=manifest,
        expected_resolution=64,
        project_root=ROOT,
        synthetic_dry_run=False,
    )
    assert report.real_research_ready is True
    assert report.software_annotation_runtime_ready is True
    assert report.dicom_source_ready is True
    assert report.extracted_cohort_ready is True
    assert report.source_readiness_validated is True
    assert report.phase2_artifacts_validated is True
    assert report.cohort_projection_validated is True
    assert report.sample_count == 1
    assert tuple(canonical.columns) == PHASE3_CANONICAL_COLUMNS
    assert canonical.loc[0, "nodule_id"].startswith(canonical.loc[0, "patient_id"])
    assert canonical.loc[0, "series_instance_uid"]
    assert report.fingerprint.source_readiness_sha256 is not None
    assert report.fingerprint.canonical_bundle_sha256 is not None
    assert report.fingerprint.rendered_bundle_sha256 is not None

    image_path = next((rendered_dir / "resolution_64").glob("*.npy"))
    payload = bytearray(image_path.read_bytes())
    payload[-1] ^= 1
    image_path.write_bytes(payload)
    with pytest.raises(Phase3ReadinessError, match="checksummed member was modified"):
        validate_phase3_readiness(
            config,
            manifest=manifest,
            expected_resolution=64,
            project_root=ROOT,
            synthetic_dry_run=False,
        )
