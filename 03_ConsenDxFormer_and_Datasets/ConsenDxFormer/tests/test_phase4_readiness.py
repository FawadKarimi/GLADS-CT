from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_json
from consendxformer.phase3_readiness import (
    Phase3InputFingerprint,
    Phase3ReadinessError,
    Phase3ReadinessReport,
)
import consendxformer.phase4_readiness as phase4_readiness
from consendxformer.phase4_readiness import (
    Phase4ReadinessError,
    validate_phase4_readiness,
)


COHORT_VIEW = "paper_aligned_multi_reader_2_to_4"


def _primary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "LIDC-IDRI-0002",
                "image_path": "resolution_64/nodule-b.npy",
                "label": "indeterminate",
                "label_id": 1,
                "split": "val",
                "nodule_id": "LIDC-IDRI-0002__nodule-b",
                "series_instance_uid": "series-b",
            },
            {
                "patient_id": "LIDC-IDRI-0001",
                "image_path": "resolution_64/nodule-a.npy",
                "label": "benign",
                "label_id": 0,
                "split": "train",
                "nodule_id": "LIDC-IDRI-0001__nodule-a",
                "series_instance_uid": "series-a",
            },
            {
                "patient_id": "LIDC-IDRI-0003",
                "image_path": "resolution_64/nodule-c.npy",
                "label": "malignant",
                "label_id": 2,
                "split": "test",
                "nodule_id": "LIDC-IDRI-0003__nodule-c",
                "series_instance_uid": "series-c",
            },
        ]
    )


def _rendered_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nodule_id": "LIDC-IDRI-0001__nodule-a",
                "nodule_sha256": "sha-a",
                "series_instance_uid": "series-a",
                "split": "train",
                "resolution": 64,
                "rendered_mask_path": "resolution_64/nodule-a_masks.npz",
                "n_valid_malignancy_ratings": 2,
            },
            {
                "nodule_id": "LIDC-IDRI-0003__nodule-c",
                "nodule_sha256": "sha-c",
                "series_instance_uid": "series-c",
                "split": "test",
                "resolution": 64,
                "rendered_mask_path": "resolution_64/nodule-c_masks.npz",
                "n_valid_malignancy_ratings": 4,
            },
            {
                "nodule_id": "LIDC-IDRI-0002__nodule-b",
                "nodule_sha256": "sha-b",
                "series_instance_uid": "series-b",
                "split": "val",
                "resolution": 64,
                "rendered_mask_path": "resolution_64/nodule-b_masks.npz",
                "n_valid_malignancy_ratings": 3,
            },
        ]
    )


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nodule_sha256": "sha-c",
                "series_instance_uid": "series-c",
                "split": "test",
                "mask_path": "masks/sha-c.npz",
                "q_benign": 0.0,
                "q_indeterminate": 0.0,
                "q_malignant": 1.0,
                "disagreement_entropy": 0.0,
                "ordinal_t1": 1.0,
                "ordinal_t2": 1.0,
                "ordinal_t3": 1.0,
                "ordinal_t4": 1.0,
                "source_validity_status": "valid",
                "n_valid_malignancy_ratings": 4,
            },
            {
                "nodule_sha256": "sha-a",
                "series_instance_uid": "series-a",
                "split": "train",
                "mask_path": "masks/sha-a.npz",
                "q_benign": 1.0,
                "q_indeterminate": 0.0,
                "q_malignant": 0.0,
                "disagreement_entropy": 0.0,
                "ordinal_t1": 0.0,
                "ordinal_t2": 0.0,
                "ordinal_t3": 0.0,
                "ordinal_t4": 0.0,
                "source_validity_status": "valid",
                "n_valid_malignancy_ratings": 2,
            },
            {
                "nodule_sha256": "sha-b",
                "series_instance_uid": "series-b",
                "split": "val",
                "mask_path": "masks/sha-b.npz",
                "q_benign": 1.0 / 3.0,
                "q_indeterminate": 1.0 / 3.0,
                "q_malignant": 1.0 / 3.0,
                "disagreement_entropy": 1.0,
                "ordinal_t1": 1.0,
                "ordinal_t2": 2.0 / 3.0,
                "ordinal_t3": 1.0 / 3.0,
                "ordinal_t4": 0.0,
                "source_validity_status": "valid_with_missing_ratings",
                "n_valid_malignancy_ratings": 3,
            },
            {
                "nodule_sha256": "excluded-sha-1",
                "series_instance_uid": "series-excluded-1",
                "split": "train",
                "mask_path": "",
                "q_benign": "",
                "q_indeterminate": "",
                "q_malignant": "",
                "disagreement_entropy": "",
                "ordinal_t1": "",
                "ordinal_t2": "",
                "ordinal_t3": "",
                "ordinal_t4": "",
                "source_validity_status": "excluded_invalid_rating",
                "n_valid_malignancy_ratings": 0,
            },
            {
                "nodule_sha256": "excluded-sha-2",
                "series_instance_uid": "series-excluded-2",
                "split": "val",
                "mask_path": "",
                "q_benign": "",
                "q_indeterminate": "",
                "q_malignant": "",
                "disagreement_entropy": "",
                "ordinal_t1": "",
                "ordinal_t2": "",
                "ordinal_t3": "",
                "ordinal_t4": "",
                "source_validity_status": "excluded_mask_unavailable",
                "n_valid_malignancy_ratings": 0,
            },
        ]
    )


def _phase3_report(
    primary: pd.DataFrame,
    **changes: object,
) -> Phase3ReadinessReport:
    ordered_identity = primary.loc[
        :, ["patient_id", "series_instance_uid", "nodule_id", "split", "label"]
    ].to_dict(orient="records")
    fingerprint = Phase3InputFingerprint(
        metadata_sha256="1" * 64,
        logical_cohort_sha256="2" * 64,
        ordered_samples_sha256=sha256_json(ordered_identity),
        source_readiness_sha256="3" * 64,
        canonical_bundle_sha256="4" * 64,
        canonical_checksum_ledger_sha256="5" * 64,
        rendered_bundle_sha256="6" * 64,
        rendered_checksum_ledger_sha256="7" * 64,
    )
    report = Phase3ReadinessReport(
        schema_version=1,
        mode="real",
        ready_for_requested_mode=True,
        real_research_ready=True,
        software_annotation_runtime_ready=True,
        dicom_source_ready=True,
        extracted_cohort_ready=True,
        source_readiness_validated=True,
        phase2_artifacts_validated=True,
        cohort_projection_validated=True,
        frozen_split_validated=True,
        synthetic_non_research=False,
        cohort_view=COHORT_VIEW,
        expected_resolution=64,
        metadata_path="validated/primary.csv",
        sample_count=len(primary),
        split_counts=(
            ("train", int((primary["split"] == "train").sum())),
            ("val", int((primary["split"] == "val").sum())),
            ("test", int((primary["split"] == "test").sum())),
        ),
        manifest_bundle_sha256="8" * 64,
        fingerprint=fingerprint,
        limitations=("Phase 3 fixture limitation.",),
    )
    return replace(report, **changes)


def _write_ledgers(
    tmp_path: Path,
    *,
    rendered: pd.DataFrame | None = None,
    canonical: pd.DataFrame | None = None,
) -> dict[str, object]:
    canonical_dir = tmp_path / "canonical"
    rendered_dir = tmp_path / "rendered"
    canonical_dir.mkdir(exist_ok=True)
    rendered_dir.mkdir(exist_ok=True)
    (canonical if canonical is not None else _canonical_frame()).to_csv(
        canonical_dir / "canonical_nodule_ledger.csv",
        index=False,
    )
    (rendered if rendered is not None else _rendered_frame()).to_csv(
        rendered_dir / "rendered_rois.csv",
        index=False,
    )
    return {
        "mode": "real",
        "metadata_path": "validated/primary.csv",
        "frozen_split_dir": "validated/manifest",
        "source_readiness_report": "validated/source_readiness.json",
        "canonical_dir": str(canonical_dir),
        "rendered_dir": str(rendered_dir),
        "cohort_view": COHORT_VIEW,
        "synthetic_marker": None,
    }


def _mock_phase3(
    monkeypatch: pytest.MonkeyPatch,
    primary: pd.DataFrame,
    report: Phase3ReadinessReport | None = None,
) -> Phase3ReadinessReport:
    selected = report if report is not None else _phase3_report(primary)

    def validated(*args, **kwargs):
        return selected, primary.copy(deep=True)

    monkeypatch.setattr(
        phase4_readiness,
        "validate_phase3_readiness",
        validated,
    )
    return selected


def _validate(
    data_config: dict[str, object],
    tmp_path: Path,
):
    return validate_phase4_readiness(
        data_config,
        manifest=object(),
        expected_resolution=64,
        project_root=tmp_path,
    )


def test_readiness_extends_phase3_and_preserves_primary_order(
    tmp_path,
    monkeypatch,
):
    primary = _primary_frame()
    phase3_report = _mock_phase3(monkeypatch, primary)
    report, supervision = _validate(_write_ledgers(tmp_path), tmp_path)

    assert report.ready_for_requested_mode is True
    assert report.real_research_ready is True
    assert report.phase3_readiness_validated is True
    assert report.multirater_supervision_validated is True
    assert report.sample_count == 3
    assert report.cohort_view == COHORT_VIEW
    assert report.expected_resolution == 64
    assert report.manifest_bundle_sha256 == "8" * 64
    assert dict(report.split_counts) == {"train": 1, "val": 1, "test": 1}
    assert dict(report.reader_count_counts) == {2: 1, 3: 1, 4: 1}
    assert report.phase3_readiness_report == phase3_report
    assert "Phase 4 readiness validates supervision inputs" in report.limitations[-1]
    assert len(report.supervision_sha256) == 64
    assert len(report.ordered_supervision_sha256) == 64
    int(report.supervision_sha256, 16)
    int(report.ordered_supervision_sha256, 16)
    assert supervision["nodule_id"].tolist() == primary["nodule_id"].tolist()
    assert supervision["n_valid_malignancy_ratings"].tolist() == [3, 2, 4]

    payload = report.to_dict()
    assert payload["phase3_input_fingerprint"] == phase3_report.fingerprint.to_dict()
    assert payload["phase3_readiness_evidence"] == phase3_report.to_dict()
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_reader_count_outside_two_to_four_fails(tmp_path, monkeypatch):
    primary = _primary_frame()
    rendered = _rendered_frame()
    canonical = _canonical_frame()
    rendered.loc[rendered["nodule_sha256"] == "sha-b", "n_valid_malignancy_ratings"] = 1
    canonical.loc[canonical["mask_path"] == "masks/sha-b.npz", "n_valid_malignancy_ratings"] = 1
    _mock_phase3(monkeypatch, primary)

    with pytest.raises(Phase4ReadinessError, match="2, 3, or 4"):
        _validate(
            _write_ledgers(tmp_path, rendered=rendered, canonical=canonical),
            tmp_path,
        )


def test_missing_canonical_multirater_supervision_fails(tmp_path, monkeypatch):
    _mock_phase3(monkeypatch, _primary_frame())
    canonical = _canonical_frame().drop(columns="q_malignant")

    with pytest.raises(Phase4ReadinessError, match="canonical supervision"):
        _validate(_write_ledgers(tmp_path, canonical=canonical), tmp_path)


def test_invalid_canonical_multirater_supervision_fails(tmp_path, monkeypatch):
    _mock_phase3(monkeypatch, _primary_frame())
    canonical = _canonical_frame()
    canonical.loc[
        canonical["mask_path"] == "masks/sha-b.npz",
        ["q_benign", "q_indeterminate", "q_malignant"],
    ] = [0.5, 0.5, 0.5]

    with pytest.raises(Phase4ReadinessError, match="probabilities must sum to 1"):
        _validate(_write_ledgers(tmp_path, canonical=canonical), tmp_path)


def test_one_to_one_linkage_failure_fails_closed(tmp_path, monkeypatch):
    _mock_phase3(monkeypatch, _primary_frame())
    rendered = pd.concat(
        [_rendered_frame(), _rendered_frame().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(Phase4ReadinessError, match="one-to-one"):
        _validate(_write_ledgers(tmp_path, rendered=rendered), tmp_path)


def test_relevant_canonical_duplicate_fails_one_to_one(tmp_path, monkeypatch):
    _mock_phase3(monkeypatch, _primary_frame())
    canonical = pd.concat(
        [_canonical_frame(), _canonical_frame().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(Phase4ReadinessError, match="duplicate nodule SHA"):
        _validate(_write_ledgers(tmp_path, canonical=canonical), tmp_path)


def test_supervision_fingerprints_are_deterministic(tmp_path, monkeypatch):
    _mock_phase3(monkeypatch, _primary_frame())
    config = _write_ledgers(tmp_path)

    first, _ = _validate(config, tmp_path)
    second, _ = _validate(config, tmp_path)

    assert first.supervision_sha256 == second.supervision_sha256
    assert first.ordered_supervision_sha256 == second.ordered_supervision_sha256


def test_scientific_supervision_change_changes_fingerprint(tmp_path, monkeypatch):
    _mock_phase3(monkeypatch, _primary_frame())
    config = _write_ledgers(tmp_path)
    original, _ = _validate(config, tmp_path)

    canonical = _canonical_frame()
    row = canonical["mask_path"] == "masks/sha-b.npz"
    canonical.loc[row, ["q_benign", "q_indeterminate", "q_malignant"]] = [
        0.5,
        0.25,
        0.25,
    ]
    changed, _ = _validate(
        _write_ledgers(tmp_path, canonical=canonical),
        tmp_path,
    )

    assert changed.supervision_sha256 != original.supervision_sha256
    assert changed.ordered_supervision_sha256 != original.ordered_supervision_sha256


def test_ledger_row_reordering_does_not_change_logical_fingerprint(
    tmp_path,
    monkeypatch,
):
    primary = _primary_frame()
    _mock_phase3(monkeypatch, primary)
    config = _write_ledgers(tmp_path)
    original, original_frame = _validate(config, tmp_path)

    reordered, reordered_frame = _validate(
        _write_ledgers(
            tmp_path,
            rendered=_rendered_frame().iloc[::-1].reset_index(drop=True),
            canonical=_canonical_frame().iloc[::-1].reset_index(drop=True),
        ),
        tmp_path,
    )

    assert reordered.supervision_sha256 == original.supervision_sha256
    assert reordered.ordered_supervision_sha256 == original.ordered_supervision_sha256
    assert reordered_frame["nodule_id"].tolist() == original_frame["nodule_id"].tolist()


def test_authoritative_primary_reordering_changes_only_ordered_fingerprint(
    tmp_path,
    monkeypatch,
):
    primary = _primary_frame()
    _mock_phase3(monkeypatch, primary)
    config = _write_ledgers(tmp_path)
    original, _ = _validate(config, tmp_path)

    reversed_primary = primary.iloc[::-1].reset_index(drop=True)
    _mock_phase3(monkeypatch, reversed_primary)
    reordered, supervision = _validate(config, tmp_path)

    assert reordered.supervision_sha256 == original.supervision_sha256
    assert reordered.ordered_supervision_sha256 != original.ordered_supervision_sha256
    assert supervision["nodule_id"].tolist() == reversed_primary["nodule_id"].tolist()


def test_phase3_failure_is_wrapped_with_cause_and_runs_before_ledger_reads(
    tmp_path,
    monkeypatch,
):
    def fail_phase3(*args, **kwargs):
        raise Phase3ReadinessError("sentinel Phase 3 failure")

    monkeypatch.setattr(
        phase4_readiness,
        "validate_phase3_readiness",
        fail_phase3,
    )
    config = {
        "canonical_dir": str(tmp_path / "does-not-exist"),
        "rendered_dir": str(tmp_path / "also-does-not-exist"),
    }

    with pytest.raises(Phase4ReadinessError, match="Phase 3 readiness") as exc_info:
        _validate(config, tmp_path)

    assert isinstance(exc_info.value.__cause__, Phase3ReadinessError)
    assert "sentinel Phase 3 failure" in str(exc_info.value)


def test_real_phase4_cannot_accept_synthetic_nonresearch_phase3(
    tmp_path,
    monkeypatch,
):
    primary = _primary_frame()
    synthetic_report = _phase3_report(
        primary,
        mode="synthetic",
        real_research_ready=False,
        synthetic_non_research=True,
    )
    _mock_phase3(monkeypatch, primary, synthetic_report)

    with pytest.raises(Phase4ReadinessError, match="real research readiness"):
        _validate(_write_ledgers(tmp_path), tmp_path)


@pytest.mark.parametrize(
    ("report_change", "expected_resolution", "message"),
    [
        ({"cohort_view": "paper_aligned_available_reader_mean"}, 64, "cohort"),
        ({"expected_resolution": 96}, 96, "64"),
    ],
)
def test_phase4_scientific_cohort_and_resolution_are_locked(
    tmp_path,
    monkeypatch,
    report_change,
    expected_resolution,
    message,
):
    primary = _primary_frame()
    report = _phase3_report(primary, **report_change)
    _mock_phase3(monkeypatch, primary, report)
    config = _write_ledgers(tmp_path)
    if "cohort_view" in report_change:
        config["cohort_view"] = report_change["cohort_view"]

    with pytest.raises(Phase4ReadinessError, match=message):
        validate_phase4_readiness(
            config,
            manifest=object(),
            expected_resolution=expected_resolution,
            project_root=tmp_path,
        )
