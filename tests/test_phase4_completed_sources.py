"""Full persisted-source loader tests, with synthetic-only trust pins.

The real Phase 4 artifact writer is exercised with an injected, non-training
executor. Nothing here reads LIDC images or trains a model.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("torch")

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.manifest_guard import validate_frozen_manifest_bundle
from consendxformer.phase3_artifacts import atomic_write_json
from consendxformer.phase4_config import load_phase4_config
from consendxformer.phase4_readiness import phase4_supervision_hashes
import consendxformer.phase4_evaluation_runner as evaluator
import consendxformer.phase4_input_lock as input_lock_module
import consendxformer.phase4_runner as experiment_runner
from test_phase4_runner import _install_guards, _readiness, _supervision, _fake_execution


def _completed(tmp_path, monkeypatch):
    _, lock = _install_guards(monkeypatch, tmp_path)
    bundle = load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml")
    frame = _supervision()
    frame["nodule_sha256"] = [f"{index:064x}" for index in range(len(frame))]
    frame["cohort_view"] = lock.cohort_view
    frame["source_validity_status"] = "valid"
    frame["rendered_mask_path"] = [f"rendered/n{index}.npz" for index in range(len(frame))]
    frame["mask_path"] = [f"canonical/n{index}.npz" for index in range(len(frame))]
    frame["resolution"] = 64
    frame["n_valid_malignancy_ratings"] = [2] * 307 + [3] * 305 + [4] * 521
    for name in ("benign", "indeterminate", "malignant"):
        frame[f"q_{name}"] = (frame["label"] == name).astype(float)
    frame["disagreement_entropy"] = 0.0
    for level in range(1, 5):
        frame[f"ordinal_t{level}"] = 0.0
    logical, ordered = phase4_supervision_hashes(frame)
    lock = replace(lock, supervision_sha256=logical, ordered_supervision_sha256=ordered)
    # Replace only synthetic trust anchors, never the production hash function.
    monkeypatch.setattr(input_lock_module, "EXPECTED_PHASE4_INPUT_LOCK", lock.to_dict())
    monkeypatch.setattr(experiment_runner, "load_phase4_input_lock", lambda path: lock)
    monkeypatch.setattr(evaluator, "_SUPERVISION_SHA256", logical)
    monkeypatch.setattr(evaluator, "_ORDERED_SUPERVISION_SHA256", ordered)
    monkeypatch.setattr(experiment_runner, "load_phase4_config", lambda path: bundle)
    report = _readiness(lock)
    monkeypatch.setattr(experiment_runner, "validate_phase4_readiness", lambda *args, **kwargs: (report, frame.copy()))
    manifest = validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")
    # Mount paths are not immutable manifest identity; verify cross-host loading.
    monkeypatch.setattr(experiment_runner, "validate_frozen_manifest_bundle", lambda path: replace(manifest, manifest_dir=tmp_path / "server_mount"))
    monkeypatch.setattr(experiment_runner, "project_git_state", lambda root: {
        "commit": evaluator.SOURCE_FINAL_MODEL_COMMIT,
        "branch": "phase4-consendxformer", "working_tree_clean": True,
    })
    monkeypatch.setattr(experiment_runner, "execute_phase4_single_run", _fake_execution)
    result = experiment_runner.run_phase4_experiment(
        config_path=bundle.config_path, project_root=tmp_path, mode="final",
        input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
    )
    return result.output_dir


def _reseal(source):
    """Attack fixture: checksums alone must not bless altered semantics."""
    path = source / "artifact_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {relative: sha256_file(source / relative) for relative in manifest["files"]}
    manifest["bundle_sha256"] = sha256_json(manifest["files"])
    atomic_write_json(path, manifest, overwrite=True)
    marker = source / "COMPLETED.json"
    completion = json.loads(marker.read_text())
    completion["artifact_manifest_sha256"] = sha256_file(path)
    atomic_write_json(marker, completion, overwrite=True)


def test_full_completed_source_loads_with_recomputed_supervision(tmp_path, monkeypatch):
    source = _completed(tmp_path, monkeypatch)
    loaded = evaluator.load_phase4_final_prediction_source(source)
    assert len(loaded.predictions) == 183
    assert len(loaded.supervision) == 1133
    assert loaded.descriptor["lambda_ordinal"] == 0.5
    assert loaded.descriptor["lambda_attention"] == 0.05
    ledger = evaluator.relative_source_snapshot(loaded)
    assert {"COMPLETED.json", "artifact_manifest.json", "status.json"} <= ledger.keys()


@pytest.mark.parametrize("mutation", ["supervision", "effective_config", "history", "last_epoch", "events"])
def test_resealed_semantic_source_drift_fails_closed(tmp_path, monkeypatch, mutation):
    source = _completed(tmp_path, monkeypatch)
    if mutation == "supervision":
        path = source / "config" / "input_supervision.csv"
        text = path.read_text()
        path.write_text(text.replace("rendered/n0.npz", "rendered/altered.npz"), encoding="utf-8")
        match = "supervision logical fingerprint"
    else:
        paths = {
            "effective_config": "config/effective_config.json",
            "history": "history/epoch_history.json",
            "last_epoch": "checkpoints/last.metadata.json",
            "events": "provenance/run.json",
        }
        path = source / paths[mutation]
        value = json.loads(path.read_text())
        if mutation == "effective_config":
            value["training"]["lr"] *= 2
            match = "exact final effective configuration"
        elif mutation == "history":
            value[0]["validation"]["test_loss"] = 0.0
            match = "selection history is invalid"
        elif mutation == "last_epoch":
            value["epoch"] = 2
            match = "last epoch"
        else:
            value["events"].reverse()
            match = "events are inconsistent"
        atomic_write_json(path, value, overwrite=True)
    _reseal(source)
    with pytest.raises(evaluator.Phase4EvaluationError, match=match):
        evaluator.load_phase4_final_prediction_source(source)


def _completed_b1(tmp_path, monkeypatch):
    from test_phase3_runner_orchestrator import (
        _install_fake_real_preflight, _fake_readiness,
        _offline_locked_source_guard, _successful_fake_executor,
    )
    from consendxformer.phase3_predictions import build_prediction_table
    import consendxformer.phase3_runner as b1_runner

    _offline_locked_source_guard.__wrapped__(monkeypatch)
    _install_fake_real_preflight(monkeypatch, tmp_path)
    metadata_path = tmp_path / "synthetic-approved.csv"
    report, _ = _fake_readiness(metadata_path)
    metadata = _supervision()
    metadata["image_path"] = "unused-synthetic.npy"
    metadata["label_id"] = metadata["label"].map({"benign": 0, "indeterminate": 1, "malignant": 2})
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    report = replace(
        report, cohort_view="paper_aligned_multi_reader_2_to_4",
        expected_resolution=64, sample_count=1133,
        split_counts=(("train", 807), ("val", 143), ("test", 183)),
        fingerprint=replace(report.fingerprint, metadata_sha256=sha256_file(metadata_path)),
    )
    monkeypatch.setattr(b1_runner, "validate_phase3_readiness", lambda *args, **kwargs: (report, metadata.copy()))
    manifest = validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")
    monkeypatch.setattr(b1_runner, "validate_frozen_manifest_bundle", lambda path: manifest)

    def fake(context):
        tiny = context.metadata.groupby("split", sort=False).head(1)
        result = _successful_fake_executor(replace(context, metadata=tiny))
        predictions = {
            split: build_prediction_table(
                context.metadata.loc[context.metadata["split"] == split],
                [[0.0, 1.0, 0.0]] * int((context.metadata["split"] == split).sum()),
                seed=context.seed, baseline_variant="B1", experiment_id=context.experiment_id,
            ) for split in ("val", "test")
        }
        provenance = {field: getattr(context, field) for field in (
            "baseline_source_commit", "project_commit", "metadata_sha256",
            "cohort_sha256", "ordered_sample_sha256", "frozen_split_sha256",
        )}
        return replace(
            result, predictions=predictions,
            best_checkpoint_record={**result.best_checkpoint_record, **provenance},
            last_checkpoint_record={**result.last_checkpoint_record, **provenance},
        )

    monkeypatch.setattr(b1_runner, "execute_locked_lungdxformer", fake)
    result = b1_runner.run_phase3_experiment(
        config_path=ROOT / "configs" / "phase3_b1.yaml", project_root=tmp_path, seed=42, device="cpu",
    )
    return result.output_dir


def test_full_b1_source_and_resealed_selection_policy(tmp_path, monkeypatch):
    source = _completed_b1(tmp_path, monkeypatch)
    loaded = evaluator.load_b1_prediction_source(source)
    assert len(loaded.predictions) == 183
    path = source / "provenance" / "run.json"
    run = json.loads(path.read_text())
    run["selection"]["criterion"] = "test_accuracy"
    atomic_write_json(path, run, overwrite=True)
    _reseal(source)
    with pytest.raises(evaluator.Phase4EvaluationError, match="selection policy"):
        evaluator.load_b1_prediction_source(source)
