"""Synthetic source-validity regression tests. No models or inference."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_phase4_boundary_runtime import _canonical_row, _fixture_source
from consendxformer import phase4_boundary as boundary
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase4_boundary_dataset import Phase4BoundaryDataset, Phase4BoundaryDatasetError


def excluded_row():
    row = _canonical_row(patient="synthetic-excluded", digest="e" * 64,
        split="train", mean="2.6666666666666665", score_sum="8", readers="3",
        paper_label="boundary_zone", eligible="False", boundary="True")
    row["source_validity_status"] = "excluded_malformed_geometry"
    return row


def valid_row():
    row = _canonical_row(patient="synthetic-valid", digest="b" * 64,
        split="test", mean="2.5", score_sum="5", readers="2",
        paper_label="boundary_zone", eligible="False", boundary="True")
    return row


def isolate(rows, primary=()):
    return boundary.isolate_phase2_boundary_cohort(pd.DataFrame(rows),
        primary_nodule_ids=primary, source_provenance={"phase": 2, "synthetic": True})


def test_malformed_exclusion_preserves_complete_population_and_precision(tmp_path, monkeypatch):
    excluded = excluded_row()
    source, primary, _ = _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded])
    assert boundary.BOUNDARY_SOURCE_VALIDITY_STATUSES == {"valid", "valid_with_missing_ratings"}
    record = source.provenance
    assert record["schema_version"] == 3
    assert record["evaluated_population"] == "source_valid_boundary_subset"
    assert record["sample_count"] == len(source.frame) == 1
    account = record["source_validity_accounting"]
    assert account["full_boundary_membership_count"] == 2
    assert account["full_boundary_split_counts"] == {"train": 1, "val": 0, "test": 1}
    assert account["source_invalid_exclusion_count"] == 1
    assert account["source_invalid_exclusion_counts_by_reason"] == {"pre_existing_phase2_malformed_geometry": 1}
    assert account["source_invalid_exclusion_split_counts"] == {"train": 1, "val": 0, "test": 0}
    assert account["source_invalid_exclusion_counts_by_reason_and_split"] == {
        "pre_existing_phase2_malformed_geometry": {"train": 1, "val": 0, "test": 0}}
    assert account["eligible_subset_count"] == 1
    assert account["eligible_subset_split_counts"] == {"train": 0, "val": 0, "test": 1}
    assert account["exclusion_records"] == [{
        "patient_id": excluded["patient_id"], "nodule_id": excluded["nodule_id"],
        "split": "train", "mean_malignancy": "2.6666666666666665",
        "source_validity_status": "excluded_malformed_geometry",
        "exclusion_reason": "pre_existing_phase2_malformed_geometry"}]
    assert account["exclusion_records_sha256"] == sha256_json(account["exclusion_records"])
    assert account["ordered_full_boundary_membership_sha256"] == sha256_json(account["full_boundary_membership"])
    assert all(row["nodule_id"] != excluded["nodule_id"] for row in record["ordered_identities"])
    canonical_path = tmp_path / "canonical/canonical_nodule_ledger.csv"
    assert source.source_snapshot[str(canonical_path)] == sha256_file(canonical_path)
    assert record["source_provenance"]["canonical_artifact"]["primary_ledger_sha256"] == sha256_file(canonical_path)
    assert boundary.validate_boundary_provenance_record(record, primary_supervision=primary) == record
    # Missing excluded renders are acceptable ONLY because the canonical status
    # is already source-invalid; no eligible render is removed by this rule.
    assert not (source.rendered_root / f"resolution_64/{'e' * 64}.npy").exists()


@pytest.mark.parametrize("status", ["", None, pd.NA, "excluded_other", "valid_but_unapproved",
                                   " excluded_malformed_geometry", "excluded_malformed_geometry "])
def test_unknown_or_missing_status_fails_closed(status):
    excluded = excluded_row()
    excluded["source_validity_status"] = status
    with pytest.raises(boundary.Phase4BoundaryError):
        isolate([valid_row(), excluded])


@pytest.mark.parametrize("violation", ["split", "identity", "digest", "duplicate_id", "duplicate_digest", "overlap", "mean", "primary_eligible"])
def test_exclusion_cannot_hide_full_membership_violation(violation):
    valid, excluded = valid_row(), excluded_row()
    primary = []
    if violation == "split":
        excluded["split"] = "development"
    elif violation == "identity":
        excluded["patient_id"] = " "
    elif violation == "digest":
        excluded["nodule_sha256"] = "bad"
    elif violation == "duplicate_id":
        excluded["nodule_id"] = valid["nodule_id"]
    elif violation == "duplicate_digest":
        excluded["nodule_sha256"] = valid["nodule_sha256"]
    elif violation == "overlap":
        primary = [excluded["nodule_id"]]
    elif violation == "mean":
        excluded["mean_malignancy"] = "2.75"
    else:
        excluded["paper_primary_eligible"] = "True"
    with pytest.raises(boundary.Phase4BoundaryError):
        isolate([valid, excluded], primary)


@pytest.mark.parametrize(("mean", "total", "readers", "label", "member"), [
    ("2.5", "5", "2", "boundary_zone", True),
    ("3.5", "7", "2", "boundary_zone", True),
    ("3.0", "6", "2", "indeterminate", False),
])
def test_exact_boundary_rule_before_status_partition(mean, total, readers, label, member):
    excluded = excluded_row()
    excluded.update(mean_malignancy=mean, malignancy_score_sum=total,
        n_valid_malignancy_ratings=readers, paper_label=label,
        boundary_zone=str(member), paper_primary_eligible=str(not member))
    _, record = isolate([valid_row(), excluded])
    account = record["source_validity_accounting"]
    assert account["full_boundary_membership_count"] == 1 + int(member)
    assert account["source_invalid_exclusion_count"] == int(member)
    assert account["eligible_subset_count"] == 1


def test_no_source_valid_rows_fails_before_rendering_or_inference(tmp_path, monkeypatch):
    def invalidate(frame):
        frame.loc[frame.boundary_zone == "True", "source_validity_status"] = "excluded_malformed_geometry"
        return frame
    monkeypatch.setattr(boundary, "_join_boundary_rendered_rows",
                        lambda *a, **kw: pytest.fail("Must fail before rendering/inference"))
    with pytest.raises(boundary.Phase4BoundaryError, match="No source-valid"):
        _fixture_source(tmp_path, monkeypatch, canonical_edit=invalidate)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "split", "resolution", "path", "primary_eligible", "reader_count"])
def test_eligible_render_contract_remains_fail_closed(tmp_path, monkeypatch, failure):
    def edit(frame):
        if failure == "missing":
            return frame.iloc[[0]].copy()
        if failure == "duplicate":
            return pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
        field, value = {"split": ("split", "train"), "resolution": ("resolution", "32"),
            "path": ("image_path", "resolution_64/missing.npy"),
            "primary_eligible": ("paper_primary_eligible", "True"),
            "reader_count": ("n_valid_malignancy_ratings", "4")}[failure]
        frame.loc[1, field] = value
        return frame
    with pytest.raises(boundary.Phase4BoundaryError):
        _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded_row()], rendered_edit=edit)


def test_unusable_eligible_image_still_fails_without_model_inference(tmp_path, monkeypatch):
    source, _, _ = _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded_row()])
    path = source.rendered_root / source.frame.iloc[0].image_path
    np.save(path, np.full((64, 64), np.nan, dtype=np.float32))
    with pytest.raises(boundary.Phase4BoundaryError, match="changed after validation"):
        boundary.assert_boundary_source_snapshot_unchanged(source.source_snapshot)
    dataset = Phase4BoundaryDataset(source.frame, rendered_root=source.rendered_root)
    with pytest.raises((ValueError, Phase4BoundaryDatasetError)):
        dataset[0]


@pytest.mark.parametrize("field", ["full_boundary_membership_count", "eligible_subset_count",
    "source_invalid_exclusion_count", "eligible_subset_split_counts", "exclusion_records",
    "full_boundary_membership_sha256"])
def test_provenance_rejects_inconsistent_accounting(tmp_path, monkeypatch, field):
    source, primary, _ = _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded_row()])
    record = copy.deepcopy(source.provenance)
    record["source_validity_accounting"][field] = None
    with pytest.raises(boundary.Phase4BoundaryError, match="accounting differs"):
        boundary.validate_boundary_provenance_record(record, primary_supervision=primary)


def test_self_consistent_fabricated_exclusions_do_not_override_frozen_ledger(tmp_path, monkeypatch):
    source, primary, _ = _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded_row()])
    record = copy.deepcopy(source.provenance)
    fabricated = pd.DataFrame(record["source_validity_accounting"]["full_boundary_membership"])
    fabricated.loc[1, "patient_id"] = "fabricated-patient"
    # Recompute all internal hashes; the immutable ledger must still win.
    record["source_validity_accounting"] = boundary._source_validity_accounting(fabricated)
    with pytest.raises(boundary.Phase4BoundaryError, match="accounting differs"):
        boundary.validate_boundary_provenance_record(record, primary_supervision=primary)


def test_changed_canonical_ledger_cannot_validate_old_provenance(tmp_path, monkeypatch):
    source, primary, _ = _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded_row()])
    path = tmp_path / "canonical/canonical_nodule_ledger.csv"
    path.write_text("tampered synthetic ledger", encoding="utf-8")
    with pytest.raises(boundary.Phase4BoundaryError, match="canonical ledger changed"):
        boundary.validate_boundary_provenance_record(source.provenance, primary_supervision=primary)


def test_rehashed_canonical_edit_cannot_escape_frozen_checksum_fingerprint(tmp_path, monkeypatch):
    source, primary, _ = _fixture_source(tmp_path, monkeypatch, extra_canonical_rows=[excluded_row()])
    record = copy.deepcopy(source.provenance)
    path = tmp_path / "canonical/canonical_nodule_ledger.csv"
    canonical = pd.read_csv(path, dtype=str, keep_default_na=False)
    canonical.loc[2, "patient_id"] = "changed-excluded-patient"
    canonical.to_csv(path, index=False)
    record["source_provenance"]["canonical_artifact"]["primary_ledger_sha256"] = sha256_file(path)
    with pytest.raises(boundary.Phase4BoundaryError, match="frozen checksum ledger"):
        boundary.validate_boundary_provenance_record(record, primary_supervision=primary)


def test_posthoc_reporting_exports_accounting_without_inference(tmp_path, monkeypatch):
    from test_phase4_posthoc_runner import _install
    from consendxformer import phase4_posthoc_runner as runner

    sources = _install(tmp_path, monkeypatch)
    source, _, _ = _fixture_source(tmp_path / "boundary", monkeypatch,
                                   extra_canonical_rows=[excluded_row()])
    monkeypatch.setattr(runner, "load_repository_phase4_boundary_source", lambda **kw: source)
    by_path = {s.source_dir: s for s in sources}
    # Neither boundary nor attention inference is executed: only synthetic
    # report rows exercise this consumer's export and provenance plumbing.
    monkeypatch.setattr(runner, "load_frozen_final_best_model",
                        lambda path, device: SimpleNamespace(source=by_path[path]))
    monkeypatch.setattr(runner, "build_frozen_attention_loader",
                        lambda frozen: SimpleNamespace(input_snapshot={}))
    monkeypatch.setattr(runner, "evaluate_attention_alignment_from_loader",
        lambda frozen, loader: {"seed": frozen.source.descriptor["seed"],
            "experiment_id": frozen.source.descriptor["experiment_id"],
            "metrics": {"per_sample": [], "aggregate": {}}})
    monkeypatch.setattr(runner, "aggregate_attention_seed_results", lambda results: {"aggregate": {}})
    def report_rows(frozen, verified):
        return {"seed": frozen.source.descriptor["seed"],
                "experiment_id": frozen.source.descriptor["experiment_id"],
                "rows": [{**row, "prob_benign": .2, "prob_indeterminate": .5,
                          "prob_malignant": .3} for row in verified.provenance["ordered_identities"]]}
    monkeypatch.setattr(runner, "predict_validated_boundary_cohort", report_rows)
    result = runner.run_phase4_posthoc_evaluation(
        phase4_source_dirs=[s.source_dir for s in sources], output_root=tmp_path / "reports", include_boundary=True)
    data = json.loads(result.machine_results.read_text())
    stress = data["boundary_stress"]
    assert stress["evaluated_population"] == "source_valid_boundary_subset"
    assert stress["source_validity_accounting"] == source.provenance["source_validity_accounting"]
    path = result.output_dir / "tables/boundary_source_invalid_exclusions.csv"
    assert pd.read_csv(path, dtype=str).mean_malignancy.tolist() == ["2.6666666666666665"]
    manifest = json.loads(result.artifact_manifest.read_text())
    assert manifest["files"]["tables/boundary_source_invalid_exclusions.csv"] == sha256_file(path)
    assert "source-valid boundary subset" in (result.output_dir / "summary.md").read_text()
