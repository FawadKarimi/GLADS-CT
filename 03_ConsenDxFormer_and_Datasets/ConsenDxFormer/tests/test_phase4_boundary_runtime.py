from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.lidc_ledger import LEDGER_COLUMNS
from consendxformer.lidc_render_pipeline import RENDERED_COLUMNS
import consendxformer.phase4_boundary as boundary_runtime
from consendxformer.phase4_boundary import (
    Phase4BoundaryError,
    assert_boundary_source_snapshot_unchanged,
    isolate_phase2_boundary_cohort,
    load_validated_phase2_boundary_source,
    validate_boundary_provenance_record,
)
from consendxformer.phase4_boundary_dataset import (
    Phase4BoundaryDataset,
    Phase4BoundaryDatasetError,
)


COHORT_VIEW = "paper_aligned_multi_reader_2_to_4"


def _identity(patient: str, digest: str) -> str:
    return f"{patient}__nodule-{digest[:16]}"


def _canonical_row(
    *,
    patient: str,
    digest: str,
    split: str,
    mean: str,
    score_sum: str,
    readers: str,
    paper_label: str,
    eligible: str,
    boundary: str,
) -> dict[str, str]:
    row = {field: "" for field in LEDGER_COLUMNS}
    row.update(
        {
            "patient_id": patient,
            "nodule_id": _identity(patient, digest),
            "nodule_sha256": digest,
            "study_instance_uid": f"study-{patient}",
            "series_instance_uid": f"series-{patient}",
            "split": split,
            "mean_malignancy": mean,
            "malignancy_score_sum": score_sum,
            "n_valid_malignancy_ratings": readers,
            "paper_label": paper_label,
            "paper_primary_eligible": eligible,
            "boundary_zone": boundary,
            "source_validity_status": "valid",
            "mask_path": f"masks/{digest}.npz",
            "mask_sha256": "f" * 64,
        }
    )
    return row


def _rendered_row(canonical: dict[str, str]) -> dict[str, str]:
    row = {field: "" for field in RENDERED_COLUMNS}
    row.update(
        {
            field: canonical[field]
            for field in (
                "patient_id",
                "nodule_id",
                "nodule_sha256",
                "study_instance_uid",
                "series_instance_uid",
                "split",
                "paper_label",
                "paper_primary_eligible",
                "boundary_zone",
                "n_valid_malignancy_ratings",
            )
        }
    )
    row.update(
        {
            "resolution": "64",
            "image_path": f"resolution_64/{canonical['nodule_sha256']}.npy",
            "rendered_mask_path": (
                f"resolution_64/{canonical['nodule_sha256']}_masks.npz"
            ),
            "has_four_valid_readers": "False",
        }
    )
    return row


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _complete_artifact(
    directory: Path, *, members: list[str]
) -> tuple[str, str]:
    files = {
        name: sha256_file(directory.joinpath(*name.split("/"))) for name in members
    }
    ledger = {
        "schema_version": 1,
        "files": files,
        "bundle_sha256": sha256_json(files),
        "hash_scope": "listed generated files; raw DICOM bytes are not hashed",
    }
    _write_json(directory / "checksums.json", ledger)
    ledger_sha = sha256_file(directory / "checksums.json")
    _write_json(
        directory / "COMPLETED.json",
        {
            "schema_version": 1,
            "checksum_ledger": "checksums.json",
            "checksum_ledger_sha256": ledger_sha,
        },
    )
    return ledger["bundle_sha256"], ledger_sha


def _fixture_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
                    extra_canonical_rows=(), canonical_edit=None, rendered_edit=None):
    canonical_dir = tmp_path / "canonical"
    rendered_dir = tmp_path / "rendered"
    manifest_dir = tmp_path / "manifest"
    (canonical_dir / "masks").mkdir(parents=True)
    (rendered_dir / "resolution_64").mkdir(parents=True)
    manifest_dir.mkdir()

    primary_digest = "a" * 64
    boundary_digest = "b" * 64
    primary_row = _canonical_row(
        patient="p-primary",
        digest=primary_digest,
        split="train",
        mean="3.0",
        score_sum="6",
        readers="2",
        paper_label="indeterminate",
        eligible="True",
        boundary="False",
    )
    boundary_row = _canonical_row(
        patient="p-boundary",
        digest=boundary_digest,
        split="test",
        mean="2.5",
        score_sum="5",
        readers="2",
        paper_label="boundary_zone",
        eligible="False",
        boundary="True",
    )
    canonical = pd.DataFrame([primary_row, boundary_row, *extra_canonical_rows], columns=LEDGER_COLUMNS)
    rendered = pd.DataFrame(
        [_rendered_row(primary_row), _rendered_row(boundary_row)],
        columns=RENDERED_COLUMNS,
    )
    if canonical_edit is not None:
        canonical = canonical_edit(canonical)
    if rendered_edit is not None:
        rendered = rendered_edit(rendered)
    canonical.to_csv(canonical_dir / "canonical_nodule_ledger.csv", index=False)
    rendered.to_csv(rendered_dir / "rendered_rois.csv", index=False)
    _write_json(canonical_dir / "run_configuration.json", {"fixture": True})
    _write_json(canonical_dir / "provenance.json", {"fixture": True})
    _write_json(rendered_dir / "run_configuration.json", {"fixture": True})
    _write_json(rendered_dir / "provenance.json", {"fixture": True})

    for digest in (primary_digest, boundary_digest):
        np.savez_compressed(
            canonical_dir / "masks" / f"{digest}.npz",
            fixture=np.ones((1,), dtype=np.uint8),
        )
        image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        np.save(rendered_dir / "resolution_64" / f"{digest}.npy", image)
        np.savez_compressed(
            rendered_dir / "resolution_64" / f"{digest}_masks.npz",
            soft_consensus_mask=np.ones((64, 64), dtype=np.float32),
        )

    canonical_members = [
        "canonical_nodule_ledger.csv",
        "run_configuration.json",
        "provenance.json",
        *(f"masks/{digest}.npz" for digest in (primary_digest, boundary_digest)),
    ]
    rendered_members = [
        "rendered_rois.csv",
        "run_configuration.json",
        "provenance.json",
        *(
            member
            for digest in (primary_digest, boundary_digest)
            for member in (
                f"resolution_64/{digest}.npy",
                f"resolution_64/{digest}_masks.npz",
            )
        ),
    ]
    canonical_bundle, canonical_ledger = _complete_artifact(
        canonical_dir, members=canonical_members
    )
    rendered_bundle, rendered_ledger = _complete_artifact(
        rendered_dir, members=rendered_members
    )

    metadata_path = tmp_path / "primary.csv"
    source_readiness_path = tmp_path / "source_readiness.json"
    metadata_path.write_text("fixture-primary\n", encoding="utf-8")
    _write_json(source_readiness_path, {"fixture": "strict_full"})
    for name in (
        "patient_split.csv",
        "series_split.csv",
        "train_patient_ids.txt",
        "val_patient_ids.txt",
        "test_patient_ids.txt",
        "split_summary.json",
        "manifest_checksums.json",
    ):
        (manifest_dir / name).write_text(f"{name}\n", encoding="utf-8")

    primary = pd.DataFrame(
        [
            {
                "patient_id": primary_row["patient_id"],
                "nodule_id": primary_row["nodule_id"],
                "nodule_sha256": primary_digest,
                "series_instance_uid": primary_row["series_instance_uid"],
                "split": "train",
                "label": "indeterminate",
                "n_valid_malignancy_ratings": "2",
            }
        ]
    )
    phase3_identities = primary.loc[
        :, ["patient_id", "series_instance_uid", "nodule_id", "split", "label"]
    ].to_dict(orient="records")
    fingerprint = {
        "metadata_sha256": sha256_file(metadata_path),
        "logical_cohort_sha256": sha256_json(phase3_identities),
        "ordered_samples_sha256": sha256_json(phase3_identities),
        "source_readiness_sha256": sha256_file(source_readiness_path),
        "canonical_bundle_sha256": canonical_bundle,
        "canonical_checksum_ledger_sha256": canonical_ledger,
        "rendered_bundle_sha256": rendered_bundle,
        "rendered_checksum_ledger_sha256": rendered_ledger,
    }
    readiness = {
        "schema_version": 1,
        "ready_for_requested_mode": True,
        "real_research_ready": True,
        "phase3_readiness_validated": True,
        "multirater_supervision_validated": True,
        "cohort_view": COHORT_VIEW,
        "expected_resolution": 64,
        "sample_count": 1,
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "reader_count_counts": {"2": 1, "3": 0, "4": 0},
        "manifest_bundle_sha256": "9" * 64,
        "phase3_input_fingerprint": fingerprint,
        "supervision_sha256": "c" * 64,
        "ordered_supervision_sha256": "d" * 64,
    }
    calls: dict[str, object] = {}

    def validated_readiness(*args, **kwargs):
        calls["readiness"] = kwargs
        return readiness, primary.copy(deep=True)

    def validated_lock(lock, report):
        calls["lock"] = (lock, report)

    monkeypatch.setattr(
        boundary_runtime, "validate_phase4_readiness", validated_readiness
    )
    monkeypatch.setattr(
        boundary_runtime, "validate_phase4_input_lock", validated_lock
    )
    data_config = {
        "mode": "real",
        "metadata_path": str(metadata_path),
        "frozen_split_dir": str(manifest_dir),
        "source_readiness_report": str(source_readiness_path),
        "canonical_dir": str(canonical_dir),
        "rendered_dir": str(rendered_dir),
        "cohort_view": COHORT_VIEW,
        "synthetic_marker": None,
    }
    manifest = SimpleNamespace(manifest_dir=manifest_dir)
    lock = {"schema_version": 1, "fixture": "locked"}
    source = load_validated_phase2_boundary_source(
        data_config,
        manifest=manifest,
        input_lock=lock,
        expected_resolution=64,
        project_root=tmp_path,
    )
    return source, primary, calls


def test_runtime_loader_binds_completed_phase2_chain_and_primary_cohort(
    tmp_path, monkeypatch
):
    source, primary, calls = _fixture_source(tmp_path, monkeypatch)

    assert calls["readiness"]["synthetic_dry_run"] is False
    assert calls["lock"][1] is source.readiness_report
    assert source.frame["paper_label"].tolist() == ["boundary_zone"]
    assert source.frame["image_path"].tolist() == [
        f"resolution_64/{'b' * 64}.npy"
    ]
    record = source.provenance
    assert record["schema_version"] == 3
    assert record["sample_count"] == 1
    assert record["primary_cohort_binding"]["sample_count"] == 1
    assert record["primary_cohort_binding"]["split_counts"] == {
        "train": 1,
        "val": 0,
        "test": 0,
    }
    identity = record["ordered_identities"][0]
    assert set(identity) == {
        "patient_id",
        "nodule_id",
        "nodule_sha256",
        "study_instance_uid",
        "series_instance_uid",
        "split",
        "image_path",
        "rendered_mask_path",
        "mean_malignancy",
        "malignancy_score_sum",
        "n_valid_malignancy_ratings",
        "boundary_stratum",
    }
    assert validate_boundary_provenance_record(
        record, primary_supervision=primary
    ) == record
    assert_boundary_source_snapshot_unchanged(source.source_snapshot)

    image_path = source.rendered_root / identity["image_path"]
    original = image_path.read_bytes()
    image_path.write_bytes(b"mutated")
    with pytest.raises(Phase4BoundaryError, match="changed after validation"):
        assert_boundary_source_snapshot_unchanged(source.source_snapshot)
    image_path.write_bytes(original)


def test_runtime_provenance_rejects_primary_or_boundary_tampering(
    tmp_path, monkeypatch
):
    source, primary, _calls = _fixture_source(tmp_path, monkeypatch)
    changed_primary = primary.copy()
    changed_primary.loc[0, "patient_id"] = "different-patient"
    with pytest.raises(Phase4BoundaryError, match="Phase 3 logical cohort"):
        validate_boundary_provenance_record(
            source.provenance, primary_supervision=changed_primary
        )

    changed_record = json.loads(json.dumps(source.provenance))
    changed_record["ordered_identities"][0]["image_path"] = "../escape.npy"
    with pytest.raises(Phase4BoundaryError, match="canonical relative path"):
        validate_boundary_provenance_record(
            changed_record, primary_supervision=primary
        )


def test_boundary_isolation_rejects_whitespace_status_prefix_and_decimal_integer():
    digest = "e" * 64
    base = pd.DataFrame(
        [
            _canonical_row(
                patient="p-boundary",
                digest=digest,
                split="test",
                mean="2.5",
                score_sum="5",
                readers="2",
                paper_label="boundary_zone",
                eligible="False",
                boundary="True",
            )
        ]
    )
    whitespace = base.copy()
    whitespace.loc[0, "nodule_id"] += " "
    with pytest.raises(Phase4BoundaryError, match="without surrounding whitespace"):
        isolate_phase2_boundary_cohort(
            whitespace,
            primary_nodule_ids=set(),
            source_provenance={"phase": 2},
        )

    prefixed = base.copy()
    prefixed.loc[0, "source_validity_status"] = "valid_but_unapproved"
    with pytest.raises(Phase4BoundaryError, match="exact renderable allowlist"):
        isolate_phase2_boundary_cohort(
            prefixed,
            primary_nodule_ids=set(),
            source_provenance={"phase": 2},
        )

    decimal = base.copy()
    decimal.loc[0, "n_valid_malignancy_ratings"] = "2.0"
    with pytest.raises(Phase4BoundaryError, match="exact canonical integer"):
        isolate_phase2_boundary_cohort(
            decimal,
            primary_nodule_ids=set(),
            source_provenance={"phase": 2},
        )


def test_boundary_dataset_uses_frozen_eval_preprocessing_and_has_no_target(
    tmp_path, monkeypatch
):
    source, _primary, _calls = _fixture_source(tmp_path, monkeypatch)
    dataset = Phase4BoundaryDataset(
        source.frame,
        rendered_root=source.rendered_root,
    )

    sample = dataset[0]
    assert sample["image"].shape == (1, 64, 64)
    assert sample["image"].dtype == torch.float32
    assert sample["image"].min().item() == pytest.approx(0.0)
    assert sample["image"].max().item() == pytest.approx(1.0)
    assert sample["boundary_stratum"] == "lower_boundary"
    assert "label" not in sample
    assert "true_class" not in sample

    with_target = source.frame.assign(label="benign")
    with pytest.raises(Phase4BoundaryDatasetError, match="primary target columns"):
        Phase4BoundaryDataset(
            with_target,
            rendered_root=source.rendered_root,
        )

    escaping = source.frame.copy()
    escaping.loc[0, "image_path"] = "../escape.npy"
    with pytest.raises(Phase4BoundaryDatasetError, match="canonical relative path"):
        Phase4BoundaryDataset(
            escaping,
            rendered_root=source.rendered_root,
        )
