from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from consendxformer.hashing import sha256_file
from consendxformer.lidc_inventory import inventory_lidc
from consendxformer.lidc_pipeline import run_extraction
from consendxformer.lidc_records import InMemoryLIDCProvider
from consendxformer.lidc_reconciliation import assess_source_readiness
from consendxformer.lidc_render_pipeline import run_rendering
from consendxformer.manifest_guard import validate_frozen_manifest_bundle
from consendxformer.phase2_config import HUWindow, clustering_config, load_phase2_config
from phase2_fakes import (
    cluster_with_ratings,
    consensus_for_cluster,
    dicom_for_scan,
    scan_from_manifest,
)


@pytest.fixture(scope="module")
def manifest():
    return validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")


def _runtime(tmp_path, manifest, ratings_groups):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "identity-only-marker").write_text("not DICOM\n", encoding="utf-8")
    scan = replace(scan_from_manifest(manifest, 0), annotation_count=sum(map(len, ratings_groups)))
    clusters = tuple(cluster_with_ratings(scan, ratings) for ratings in ratings_groups)
    provider = InMemoryLIDCProvider(
        [scan],
        clusters={scan.series_instance_uid: clusters},
        consensus_factory=consensus_for_cluster,
        dicom_slices={(scan.series_instance_uid, 3): dicom_for_scan(scan, 3)},
    )
    source_report = assess_source_readiness(
        raw_root, [scan], manifest, mode="partial_development"
    )
    config = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    config["source"]["raw_root"] = str(raw_root.resolve())
    return config, provider, (scan,), source_report


def test_inventory_reports_reader_and_label_distributions_without_4095_gate(tmp_path, manifest):
    config, provider, scans, _ = _runtime(
        tmp_path, manifest, [[1], [1, 2], [1, 3, 5], [1, 2, 4, 5]]
    )
    summary, rows = inventory_lidc(
        provider, scans, manifest, clustering_config(config)
    )
    assert summary["physical_nodule_cluster_count"] == 4
    assert summary["clusters_by_number_of_annotations"] == {"1": 1, "2": 1, "3": 1, "4": 1}
    assert summary["clusters_by_number_of_valid_malignancy_ratings"] == {
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
    }
    assert summary["paper_cohort_size_4095_enforced"] is False
    assert summary["multi_reader_candidate_count"] == 3
    assert len(rows) == 4


def test_canonical_extraction_is_deterministic_and_preserves_boundary(tmp_path, manifest):
    config, provider, scans, source_report = _runtime(
        tmp_path, manifest, [[2, 3], [1, 3, 5]]
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_extraction(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=source_report,
        config=config,
        output_dir=first_dir,
        project_root=ROOT,
    )
    second = run_extraction(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=source_report,
        config=config,
        output_dir=second_dir,
        project_root=ROOT,
    )
    assert (first_dir / "canonical_nodule_ledger.csv").read_bytes() == (
        second_dir / "canonical_nodule_ledger.csv"
    ).read_bytes()
    assert first["nodule_id"].tolist() == second["nodule_id"].tolist()
    boundary = first.loc[first["paper_label"] == "boundary_zone"].iloc[0]
    assert bool(boundary["boundary_zone"]) is True
    assert bool(boundary["paper_primary_eligible"]) is False
    assert boundary["source_validity_status"] == "valid"
    assert json.loads(boundary["malignancy_ratings_json"]) == [2, 3]
    for left_relative, right_relative in zip(first["mask_path"], second["mask_path"]):
        left = first_dir / left_relative
        right = second_dir / right_relative
        assert left.read_bytes() == right.read_bytes()
        assert sha256_file(left) == sha256_file(right)
    assert (first_dir / "COMPLETED.json").is_file()
    leakage = json.loads((first_dir / "leakage_check.json").read_text(encoding="utf-8"))
    assert leakage["passed"] is True


def test_completed_extraction_cannot_be_silently_overwritten(tmp_path, manifest):
    config, provider, scans, source_report = _runtime(tmp_path, manifest, [[3]])
    output = tmp_path / "complete"
    kwargs = dict(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=source_report,
        config=config,
        output_dir=output,
        project_root=ROOT,
    )
    run_extraction(**kwargs)
    with pytest.raises(FileExistsError, match="Completed"):
        run_extraction(**kwargs, allow_resume=True)


def test_incomplete_extraction_resumes_with_identical_provenance(tmp_path, manifest):
    config, provider, scans, source_report = _runtime(tmp_path, manifest, [[1, 3, 5]])
    output = tmp_path / "interrupted"
    kwargs = dict(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=source_report,
        config=config,
        output_dir=output,
        project_root=ROOT,
    )
    original = run_extraction(**kwargs)
    provenance_before = (output / "provenance.json").read_bytes()
    (output / "COMPLETED.json").unlink()
    resumed = run_extraction(**kwargs, allow_resume=True)
    assert original.equals(resumed)
    assert (output / "provenance.json").read_bytes() == provenance_before
    assert (output / "COMPLETED.json").is_file()


def test_one_canonical_ledger_feeds_both_64_and_96_renderings(tmp_path, manifest):
    config, provider, scans, source_report = _runtime(tmp_path, manifest, [[1, 3, 5]])
    canonical_dir = tmp_path / "canonical"
    canonical = run_extraction(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=source_report,
        config=config,
        output_dir=canonical_dir,
        project_root=ROOT,
    )
    outputs = {}
    for resolution in (64, 96):
        outputs[resolution] = run_rendering(
            provider=provider,
            scans=scans,
            manifest=manifest,
            source_report=source_report,
            config=config,
            canonical_ledger_path=canonical_dir / "canonical_nodule_ledger.csv",
            output_dir=tmp_path / f"rendered-{resolution}",
            project_root=ROOT,
            hu_window=HUWindow(-1000, 400, "user_supplied"),
            resolution=resolution,
        )
        image = Path(tmp_path / f"rendered-{resolution}" / outputs[resolution].iloc[0]["image_path"])
        assert image.is_file()
    assert canonical["nodule_id"].tolist() == outputs[64]["nodule_id"].tolist()
    assert outputs[64]["nodule_id"].tolist() == outputs[96]["nodule_id"].tolist()
    assert outputs[64]["resolution"].tolist() == [64]
    assert outputs[96]["resolution"].tolist() == [96]
