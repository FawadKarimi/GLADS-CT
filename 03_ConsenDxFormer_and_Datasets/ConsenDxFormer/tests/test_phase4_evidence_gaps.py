"""CPU-only synthetic evidence exports; never use university datasets/models."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer import phase4_evidence_gaps as gaps
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase4_evaluation_metrics import predictive_entropy, risk_coverage_curve
from consendxformer.phase4_ablation_runner import reference_record
from consendxformer.phase4_evaluation_runner import Phase4EvaluationError


def predictions(seed, off=False):
    probabilities = np.array([[.8, .1, .1], [.3, .4, .3], [.1, .2, .7]])
    if off:
        probabilities = probabilities[::-1].copy()
    if seed == 43:
        probabilities = probabilities[:, ::-1].copy()
    if seed == 44:
        probabilities = probabilities * .7 + .1
    rows = []
    for index, values in enumerate(probabilities):
        rows.append({"patient_id": f"p{index // 2}", "nodule_id": f"n{index}",
            "series_instance_uid": f"s{index}", "split": "test", "seed": seed,
            "true_class": gaps.CLASS_ORDER[index],
            "predicted_class": gaps.CLASS_ORDER[int(values.argmax())],
            "experiment_id": f"{'off' if off else 'full'}-{seed}",
            **{f"prob_{name}": float(values[i]) for i, name in enumerate(gaps.CLASS_ORDER)}})
    return pd.DataFrame(rows)


def supervision():
    frame = predictions(42).rename(columns={"true_class": "label"})
    frame["disagreement_entropy"] = [.1, .5, .9]
    train = frame.iloc[[0]].copy()
    train["patient_id"], train["nodule_id"], train["split"] = "training-p", "training-n", "train"
    train["disagreement_entropy"] = .5
    return pd.concat([train, frame], ignore_index=True)


def test_saved_uncertainty_reuses_definitions_pooling_pairing_and_text_dtypes():
    full = {s: predictions(s) for s in gaps.FINAL_SEEDS}
    off = {s: predictions(s, True).iloc[::-1].reset_index(drop=True) for s in gaps.FINAL_SEEDS}
    metadata = supervision()
    metadata["label"] = metadata.label.astype(pd.StringDtype(storage="python"))
    before = metadata.copy(deep=True)
    result, tables, curves = gaps.uncertainty_comparison(full, off, metadata)
    pd.testing.assert_frame_equal(metadata, before)
    assert result["unique_nodules"] == 3 and result["seed_nodule_pairs"] == 9
    assert result["unique_patients"] == 2
    assert result["threshold"] == .5
    assert result["pooled_spearman"]["full"]["n"] == 3
    pooled = tables["uncertainty_per_nodule_seed_mean.csv"]
    means = pooled.loc[pooled.model == "full", "mean_predictive_entropy_nats"].to_numpy()
    expected = np.mean([predictive_entropy(gaps._probabilities(full[s])) for s in gaps.FINAL_SEEDS], axis=0)
    np.testing.assert_allclose(means, expected)
    assert not np.allclose(means, predictive_entropy(np.mean([gaps._probabilities(full[s]) for s in gaps.FINAL_SEEDS], axis=0)))
    assert pooled.loc[pooled.nodule_id == "n1", "disagreement_group"].eq("high").all()
    for ranking in ("entropy", "confidence"):
        expected_curve = risk_coverage_curve(full[42].true_class, gaps._probabilities(full[42]),
            ranking=ranking, identifiers=["p0|s0|n0", "p0|s1|n1", "p1|s2|n2"])
        assert curves["full/42"][ranking] == expected_curve
    paired = tables["paired_per_nodule_per_seed.csv"]
    np.testing.assert_allclose(paired.entropy_delta_full_minus_off,
                               paired.predictive_entropy_nats_full - paired.predictive_entropy_nats_off)
    assert len(tables["paired_per_nodule_seed_mean.csv"]) == 3


@pytest.mark.parametrize("mutation", ["seed", "patient", "label", "duplicate", "probabilities", "supervision"])
def test_uncertainty_fails_closed(mutation):
    full = {s: predictions(s) for s in gaps.FINAL_SEEDS}
    off = {s: predictions(s, True) for s in gaps.FINAL_SEEDS}
    metadata = supervision()
    if mutation == "seed":
        del off[44]
    elif mutation == "patient":
        off[42].loc[0, "patient_id"] = "wrong"
    elif mutation == "label":
        off[42].loc[0, "true_class"] = "malignant"
    elif mutation == "duplicate":
        off[42] = pd.concat([off[42], off[42].iloc[[0]]])
    elif mutation == "probabilities":
        off[42].loc[0, "prob_benign"] = float("nan")
    else:
        metadata.loc[metadata.split == "test", "label"] = "malignant"
    with pytest.raises((ValueError, Phase4EvaluationError)):
        gaps.uncertainty_comparison(full, off, metadata)


def boundary_results():
    results = []
    for seed in gaps.FINAL_SEEDS:
        frame = predictions(seed).drop(columns=["true_class"])
        frame["mean_malignancy"] = [2.5, 2.75, 3.5]
        frame["malignancy_score_sum"] = [5, 11, 7]
        frame["n_valid_malignancy_ratings"] = [2, 4, 2]
        frame["boundary_stratum"] = ["lower_boundary", "lower_boundary", "upper_boundary"]
        results.append({"seed": seed, "experiment_id": f"full-{seed}",
                        "checkpoint_sha256": str(seed) * 32, "rows": frame.to_dict("records")})
    return results


def test_boundary_descriptive_counts_overlap_and_no_primary_metrics():
    primary = pd.DataFrame({"patient_id": ["p0", "p1", "other"], "split": ["train", "val", "test"]})
    result, tables = gaps.boundary_summary(boundary_results(), primary)
    assert result["n_nodules"] == 3 and result["n_patients"] == 2
    assert result["patient_overlap_with_primary"]["train"] == {
        "patient_ids": ["p0"], "patient_count": 1, "boundary_nodule_count": 2}
    assert result["patient_overlap_with_primary"]["val"]["patient_count"] == 1
    assert result["patient_overlap_with_primary"]["test"]["patient_count"] == 0
    assert not result["primary_metrics_computed"] and not result["independent_held_out_test"]
    assert "NOT an independent held-out test" in result["warning"]
    assert len(tables["boundary_predictions_per_seed.csv"]) == 9
    assert len(tables["boundary_per_nodule_seed_mean.csv"]) == 3
    assert len(result["per_seed_regions"]) == 6
    assert sum(row["n_nodules"] for row in result["aggregate_regions"]) == 3


@pytest.mark.parametrize("mutation", ["seed", "identity", "label", "exact_three", "outside", "region", "class"])
def test_boundary_summary_rejects_mismatches(mutation):
    results = boundary_results()
    if mutation == "seed":
        results[2]["seed"] = 42
    elif mutation == "identity":
        results[2]["rows"][0]["nodule_id"] = "changed"
    elif mutation == "label":
        results[0]["rows"][0]["label"] = "indeterminate"
    elif mutation == "exact_three":
        results[0]["rows"][0]["mean_malignancy"] = 3.0
    elif mutation == "outside":
        results[0]["rows"][0]["mean_malignancy"] = 2.49
    elif mutation == "region":
        results[0]["rows"][0]["boundary_stratum"] = "upper_boundary"
    else:
        results[0]["rows"][0]["predicted_class"] = "indeterminate"
    with pytest.raises(ValueError):
        gaps.boundary_summary(results, supervision())


def install_sources(tmp_path, monkeypatch):
    """Mock historical loader boundary, not metric/file integrity implementations."""
    full, off = {}, {}
    metadata = supervision()
    monkeypatch.setattr(gaps, "NAMESPACE", tmp_path / "artifacts/phase4/evidence_gaps")
    monkeypatch.setattr(gaps, "validate_evaluator_repository", lambda root: {"commit": "a" * 40})
    monkeypatch.setattr(gaps, "validate_final_source_descriptors", lambda records: records)
    monkeypatch.setattr(gaps, "assert_supervision", lambda frame, descriptor: pd.testing.assert_frame_equal(frame, metadata))
    for seed in gaps.FINAL_SEEDS:
        for target, name in ((full, "full"), (off, "off")):
            directory = tmp_path / f"{name}-{seed}"
            (directory / "checkpoints").mkdir(parents=True)
            (directory / "predictions").mkdir()
            (directory / "checkpoints/best.pt").write_bytes(f"synthetic-{name}-{seed}".encode())
            frame = predictions(seed, name == "off")
            frame.to_csv(directory / "predictions/test.csv", index=False)
            snapshot = {str(path): sha256_file(path) for path in directory.rglob("*") if path.is_file()}
            source = SimpleNamespace(source_dir=directory, source_snapshot=snapshot, supervision=metadata.copy())
            if name == "full":
                source.descriptor = {"seed": seed, "experiment_id": f"full-{seed}"}
                source.predictions = frame
            else:
                source.identity = {"seed": seed, "variant": "disagreement_weighting",
                                   "full_model_reference": reference_record(full[seed])}
                source.predictions = {"test": frame}
            target[seed] = source
    monkeypatch.setattr(gaps, "load_phase4_final_prediction_source", lambda path: next(s for s in full.values() if s.source_dir == Path(path)))
    monkeypatch.setattr(gaps, "load_ablation_source", lambda path: next(s for s in off.values() if s.source_dir == Path(path)))
    return full, off


def run_uncertainty(full, off):
    return gaps.run_evidence_gap(analysis="uncertainty", full_source_dirs=[s.source_dir for s in full.values()],
                                 off_source_dirs=[s.source_dir for s in off.values()])


def test_export_manifest_provenance_no_inference_and_no_overwrite(tmp_path, monkeypatch):
    full, off = install_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(gaps, "load_frozen_final_best_model", lambda *a, **kw: pytest.fail("Saved uncertainty must not infer"))
    output = run_uncertainty(full, off)
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    completion = json.loads((output / "COMPLETED.json").read_text())
    assert manifest["bundle_sha256"] == sha256_json(manifest["files"])
    assert completion["artifact_manifest_sha256"] == sha256_file(output / "artifact_manifest.json")
    for relative, digest in manifest["files"].items():
        assert sha256_file(output / relative) == digest
    provenance = json.loads((output / "provenance/provenance.json").read_text())
    assert output.name == f"phase4-evaluation-{sha256_json(provenance)[:16]}"
    assert len(provenance["sources"]) == 6
    assert all(record["best_checkpoint_sha256"] and record["test_predictions_sha256"] for record in provenance["sources"])
    assert provenance["seeds"] == [42, 43, 44]
    with pytest.raises(FileExistsError):
        run_uncertainty(full, off)


@pytest.mark.parametrize("mismatch", ["variant", "reference", "seed", "checksum"])
def test_export_source_guard_failures(tmp_path, monkeypatch, mismatch):
    full, off = install_sources(tmp_path, monkeypatch)
    if mismatch == "variant":
        off[42].identity["variant"] = "attention_alignment"
    elif mismatch == "reference":
        off[42].identity["full_model_reference"]["best_checkpoint_sha256"] = "bad"
    elif mismatch == "seed":
        off[44].identity["seed"] = 42
    else:
        (off[42].source_dir / "predictions/test.csv").write_text("changed")
    with pytest.raises((ValueError, Phase4EvaluationError)):
        run_uncertainty(full, off)
    assert not list(tmp_path.rglob("COMPLETED.json"))


@pytest.mark.parametrize("destination", ["outside", "completed"])
def test_output_namespace_is_guarded(tmp_path, monkeypatch, destination):
    full, off = install_sources(tmp_path, monkeypatch)
    root = tmp_path / "outside" if destination == "outside" else gaps.NAMESPACE / "uncertainty"
    if destination == "completed":
        root.mkdir(parents=True)
        (root / "COMPLETED.json").write_text("{}")
    with pytest.raises(ValueError):
        gaps.run_evidence_gap(analysis="uncertainty", full_source_dirs=[s.source_dir for s in full.values()],
            off_source_dirs=[s.source_dir for s in off.values()], output_root=root)
    assert not list(root.glob("phase4-evaluation-*"))


def test_changed_source_during_export_does_not_complete(tmp_path, monkeypatch):
    full, off = install_sources(tmp_path, monkeypatch)
    calculate = gaps.uncertainty_comparison
    def mutate(*args):
        result = calculate(*args)
        (full[42].source_dir / "checkpoints/best.pt").write_bytes(b"changed")
        return result
    monkeypatch.setattr(gaps, "uncertainty_comparison", mutate)
    with pytest.raises((ValueError, Phase4EvaluationError)):
        run_uncertainty(full, off)
    assert not list(tmp_path.rglob("COMPLETED.json"))


def test_boundary_export_uses_only_frozen_cpu_best_loader(tmp_path, monkeypatch):
    full, _ = install_sources(tmp_path, monkeypatch)
    accounting = {"eligible_subset_count": 3, "full_boundary_membership_count": 4,
                  "source_invalid_exclusion_count": 1, "exclusion_records": [{
                      "patient_id": "excluded", "nodule_id": "excluded", "split": "train",
                      "mean_malignancy": "2.6666666666666665",
                      "source_validity_status": "excluded_malformed_geometry",
                      "exclusion_reason": "pre_existing_phase2_malformed_geometry"}]}
    boundary = SimpleNamespace(source_snapshot={}, provenance={"synthetic": True,
                              "source_validity_accounting": accounting})
    monkeypatch.setattr(gaps, "load_repository_phase4_boundary_source", lambda **kw: boundary)
    monkeypatch.setattr(gaps, "assert_boundary_source_snapshot_unchanged", lambda snapshot: None)
    called = []
    def frozen(path, device):
        assert device == "cpu"
        source = next(s for s in full.values() if s.source_dir == path)
        called.append(source.descriptor["seed"])
        return SimpleNamespace(source=source)
    monkeypatch.setattr(gaps, "load_frozen_final_best_model", frozen)
    monkeypatch.setattr(gaps, "predict_validated_boundary_cohort",
        lambda model, cohort: boundary_results()[model.source.descriptor["seed"] - 42])
    output = gaps.run_evidence_gap(analysis="boundary", full_source_dirs=[s.source_dir for s in full.values()])
    assert called == [42, 43, 44]
    result = json.loads((output / "evaluation.json").read_text())
    assert result["summary"]["n_nodules"] == 3
    assert result["summary"]["source_validity_accounting"] == accounting
    assert result["summary"]["evaluated_population"] == "source_valid_boundary_subset"
    assert "source-valid boundary subset" in (output / "README.md").read_text()
    exclusions = pd.read_csv(output / "tables/boundary_source_invalid_exclusions.csv", dtype=str)
    assert exclusions.mean_malignancy.tolist() == ["2.6666666666666665"]
    assert (output / "COMPLETED.json").exists()
