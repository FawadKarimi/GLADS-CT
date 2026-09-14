from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_checkpoints import (
    Phase4CheckpointContext,
    build_phase4_checkpoint,
    save_phase4_checkpoint,
)
from consendxformer.phase4_config import Phase4ConfigBundle, load_phase4_config
from consendxformer.phase4_execution import Phase4ExecutionResult
from consendxformer.phase4_input_lock import Phase4InputLockError, load_phase4_input_lock
from consendxformer.phase4_predictions import build_phase4_prediction_table
import consendxformer.phase4_runner as phase4_runner
from consendxformer.phase4_runner import Phase4RunError, run_phase4_experiment


def _bundle(tmp_path: Path) -> Phase4ConfigBundle:
    original = load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml")
    config = deepcopy(original.phase4_config)
    config["experiment"]["output_root"] = str(tmp_path / "phase4-runs")
    config["training"]["epochs"] = 1
    return Phase4ConfigBundle(
        config_path=original.config_path,
        b1_anchor_path=original.b1_anchor_path,
        phase4_config=config,
        b1_anchor_config=original.b1_anchor_config,
        combined_config_sha256=original.combined_config_sha256,
    )


def _supervision() -> pd.DataFrame:
    rows = []
    labels = ("benign", "indeterminate", "malignant")
    for split, count in (("train", 807), ("val", 143), ("test", 183)):
        for split_index in range(count):
            index = len(rows)
            rows.append(
                {
                    "patient_id": f"p{index}",
                    "nodule_id": f"n{index}",
                    "series_instance_uid": f"s{index}",
                    "split": split,
                    "label": labels[split_index % 3],
                }
            )
    return pd.DataFrame(rows)


def _readiness(lock, *, ordered=None):
    payload = {
        "schema_version": 1,
        "ready_for_requested_mode": True,
        "real_research_ready": True,
        "phase3_readiness_validated": True,
        "multirater_supervision_validated": True,
        "cohort_view": lock.cohort_view,
        "expected_resolution": lock.expected_resolution,
        "sample_count": lock.sample_count,
        "split_counts": dict(lock.split_counts),
        "reader_count_counts": dict(lock.reader_count_counts),
        "manifest_bundle_sha256": lock.manifest_bundle_sha256,
        "phase3_input_fingerprint": {"ordered_samples_sha256": "2" * 64},
        "phase3_readiness_evidence": {"real_research_ready": True},
        "supervision_sha256": lock.supervision_sha256,
        "ordered_supervision_sha256": (
            ordered if ordered is not None else lock.ordered_supervision_sha256
        ),
        "limitations": [],
    }
    fingerprint = SimpleNamespace(to_dict=lambda: payload["phase3_input_fingerprint"])
    return SimpleNamespace(
        **{key: value for key, value in payload.items() if key != "phase3_input_fingerprint"},
        phase3_input_fingerprint=fingerprint,
        to_dict=lambda: deepcopy(payload),
    )


def _manifest(lock):
    return SimpleNamespace(
        bundle_sha256=lock.manifest_bundle_sha256,
        to_dict=lambda: {"bundle_sha256": lock.manifest_bundle_sha256},
    )


def _fake_execution(context):
    if context.run_mode == "tuning":
        assert "test" not in set(context.supervision["split"])
    checkpoint_context = Phase4CheckpointContext.from_execution_context(context)
    per_class = {
        name: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
        for name in ("benign", "indeterminate", "malignant")
    }
    metrics = {
        "loss": 0.3,
        "total_loss": 0.3,
        "diagnostic_loss": 0.25,
        "ordinal_loss": 0.1,
        "attention_loss": 0.25,
        "accuracy": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1": 1.0,
        "per_class": per_class,
    }
    history = ({"epoch": 1, "train": dict(metrics), "validation": dict(metrics)},)
    state = {"weight": __import__("torch").tensor([1.0])}
    best = build_phase4_checkpoint(
        checkpoint_kind="best",
        context=checkpoint_context,
        model_state=state,
        optimizer_state={},
        scheduler_state=None,
        epoch=1,
        best_epoch=1,
        best_validation_loss=0.3,
        validation_metrics=metrics,
    )
    last = build_phase4_checkpoint(
        checkpoint_kind="last",
        context=checkpoint_context,
        model_state=state,
        optimizer_state={},
        scheduler_state=None,
        epoch=1,
        best_epoch=1,
        best_validation_loss=0.3,
        validation_metrics=metrics,
    )
    best_path = save_phase4_checkpoint(best, context.layout.checkpoints_dir / "best.pt")
    last_path = save_phase4_checkpoint(last, context.layout.checkpoints_dir / "last.pt")
    predictions = {}
    evaluated = ("val",) if context.run_mode == "tuning" else ("val", "test")
    for split in evaluated:
        frame = context.supervision.loc[context.supervision["split"] == split]
        logits = [[3.0, 0.0, -1.0] for _ in range(len(frame))]
        predictions[split] = build_phase4_prediction_table(
            frame,
            logits,
            seed=context.seed,
            lambda_ordinal=context.lambda_ordinal,
            lambda_attention=context.lambda_attention,
            run_mode=context.run_mode,
            experiment_id=context.experiment_id,
        )
    events = [
        "best_checkpoint_selected_from_validation_total_loss",
        "persisted_checkpoints_reloaded_and_validated",
    ]
    if context.run_mode == "final":
        events.append("best_checkpoint_reloaded_for_final_validation_and_test")
        events.append("test_evaluation_started_after_final_selection_and_best_reload")
    else:
        events.append("best_checkpoint_reloaded_for_validation")
    return Phase4ExecutionResult(
        history=history,
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        best_checkpoint_record=best,
        last_checkpoint_record=last,
        predictions=predictions,
        metrics={split: {"macro_f1": 1.0} for split in evaluated},
        best_epoch=1,
        last_epoch=1,
        class_weights=(1.0, 1.0, 1.0),
        events=tuple(events),
    )


def _install_guards(monkeypatch, tmp_path, *, readiness_sequence=None, git_sequence=None):
    lock = load_phase4_input_lock(ROOT / "configs" / "phase4_input_lock.json")
    bundle = _bundle(tmp_path)
    frame = _supervision()
    reports = list(readiness_sequence or [_readiness(lock), _readiness(lock)])
    states = list(
        git_sequence
        or [
            {"commit": "a" * 40, "branch": "phase4-consendxformer", "working_tree_clean": True},
            {"commit": "a" * 40, "branch": "phase4-consendxformer", "working_tree_clean": True},
        ]
    )
    monkeypatch.setattr(phase4_runner, "load_phase4_config", lambda path: bundle)
    monkeypatch.setattr(
        phase4_runner, "_require_tracked_phase4_inputs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(phase4_runner, "validate_frozen_manifest_bundle", lambda path: _manifest(lock))
    monkeypatch.setattr(
        phase4_runner,
        "validate_phase4_readiness",
        lambda *args, **kwargs: (reports.pop(0), frame.copy(deep=True)),
    )
    monkeypatch.setattr(phase4_runner, "project_git_state", lambda root: states.pop(0))
    monkeypatch.setattr(
        phase4_runner,
        "set_global_seed",
        lambda *args, **kwargs: {"seed": 42, "torch_seeded": True},
    )
    monkeypatch.setattr(
        phase4_runner,
        "environment_audit",
        lambda *args, **kwargs: {"torch": {"cuda_available": False}},
    )
    return bundle, lock


def test_readiness_failure_precedes_executor_and_output_creation(tmp_path, monkeypatch):
    bundle, lock = _install_guards(monkeypatch, tmp_path)

    def fail_readiness(*args, **kwargs):
        raise RuntimeError("readiness sentinel")

    called = False

    def forbidden_executor(context):
        nonlocal called
        called = True
        raise AssertionError("executor must not run")

    monkeypatch.setattr(phase4_runner, "validate_phase4_readiness", fail_readiness)
    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", forbidden_executor)
    with pytest.raises(RuntimeError, match="readiness sentinel"):
        run_phase4_experiment(
            config_path=bundle.config_path,
            project_root=tmp_path,
            mode="tuning",
            lambda_ordinal=0.1,
            lambda_attention=0.05,
            input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
        )
    assert called is False
    assert not Path(bundle.phase4_config["experiment"]["output_root"]).exists()


def test_successful_tuning_run_writes_validation_only_and_completion(tmp_path, monkeypatch):
    bundle, lock = _install_guards(monkeypatch, tmp_path)
    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", _fake_execution)

    result = run_phase4_experiment(
        config_path=bundle.config_path,
        project_root=tmp_path,
        mode="tuning",
        lambda_ordinal=0.1,
        lambda_attention=0.05,
        input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
    )
    assert result.state == "completed"
    assert result.run_mode == "tuning"
    assert result.completion_marker.is_file()
    assert result.artifact_manifest.is_file()
    assert set(result.prediction_paths) == {"val"}
    assert set(result.metrics_paths) == {"val"}
    assert not (result.output_dir / "predictions" / "test.csv").exists()
    assert not (result.output_dir / "metrics" / "test.json").exists()
    snapshot = pd.read_csv(result.output_dir / "config" / "input_supervision.csv")
    assert set(snapshot["split"]) == {"train", "val"}


def test_future_frozen_final_run_writes_val_then_test_artifacts(tmp_path, monkeypatch):
    bundle, _ = _install_guards(monkeypatch, tmp_path)
    bundle.phase4_config["losses"]["selected_lambda_ordinal"] = 0.1
    bundle.phase4_config["losses"]["selected_lambda_attention"] = 0.05
    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", _fake_execution)

    result = run_phase4_experiment(
        config_path=bundle.config_path,
        project_root=tmp_path,
        mode="final",
        input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
    )
    assert result.state == "completed"
    assert result.run_mode == "final"
    assert set(result.prediction_paths) == {"val", "test"}
    assert set(result.metrics_paths) == {"val", "test"}
    snapshot = pd.read_csv(result.output_dir / "config" / "input_supervision.csv")
    assert set(snapshot["split"]) == {"train", "val", "test"}


def test_post_training_git_drift_fails_without_completion(tmp_path, monkeypatch):
    lock = load_phase4_input_lock(ROOT / "configs" / "phase4_input_lock.json")
    _install_guards(
        monkeypatch,
        tmp_path,
        git_sequence=[
            {"commit": "a" * 40, "branch": "phase4-consendxformer", "working_tree_clean": True},
            {"commit": "b" * 40, "branch": "phase4-consendxformer", "working_tree_clean": True},
        ],
    )
    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", _fake_execution)

    with pytest.raises(Phase4RunError, match="Git state changed"):
        run_phase4_experiment(
            config_path=ROOT / "configs" / "phase4_consensdxformer.yaml",
            project_root=tmp_path,
            mode="tuning",
            lambda_ordinal=0.1,
            lambda_attention=0.05,
            input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
        )
    output_root = tmp_path / "phase4-runs"
    directories = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(directories) == 1
    assert not (directories[0] / "COMPLETED.json").exists()
    assert '"state":"failed"' in (directories[0] / "status.json").read_text()


def test_post_training_readiness_drift_fails_without_completion(tmp_path, monkeypatch):
    lock = load_phase4_input_lock(ROOT / "configs" / "phase4_input_lock.json")
    _install_guards(
        monkeypatch,
        tmp_path,
        readiness_sequence=[_readiness(lock), _readiness(lock, ordered="0" * 64)],
    )
    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", _fake_execution)

    with pytest.raises(Phase4InputLockError, match="ordered_supervision_sha256"):
        run_phase4_experiment(
            config_path=ROOT / "configs" / "phase4_consensdxformer.yaml",
            project_root=tmp_path,
            mode="tuning",
            lambda_ordinal=0.1,
            lambda_attention=0.05,
            input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
        )
    directory = next((tmp_path / "phase4-runs").iterdir())
    assert not (directory / "COMPLETED.json").exists()
    assert '"state":"failed"' in (directory / "status.json").read_text()


def test_executor_failure_never_claims_completion(tmp_path, monkeypatch):
    _install_guards(monkeypatch, tmp_path)

    def fail(context):
        raise RuntimeError("training interrupted")

    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", fail)
    with pytest.raises(RuntimeError, match="training interrupted"):
        run_phase4_experiment(
            config_path=ROOT / "configs" / "phase4_consensdxformer.yaml",
            project_root=tmp_path,
            mode="tuning",
            lambda_ordinal=0.1,
            lambda_attention=0.05,
            input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
        )
    directory = next((tmp_path / "phase4-runs").iterdir())
    assert not (directory / "COMPLETED.json").exists()
    assert '"state":"failed"' in (directory / "status.json").read_text()


def test_missing_required_epoch_metrics_fail_without_completion(tmp_path, monkeypatch):
    _install_guards(monkeypatch, tmp_path)

    def incomplete(context):
        result = _fake_execution(context)
        history = [deepcopy(record) for record in result.history]
        del history[0]["validation"]["per_class"]["indeterminate"]["recall"]
        return replace(result, history=tuple(history))

    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", incomplete)
    with pytest.raises(Phase4RunError, match="indeterminate.recall"):
        run_phase4_experiment(
            config_path=ROOT / "configs" / "phase4_consensdxformer.yaml",
            project_root=tmp_path,
            mode="tuning",
            lambda_ordinal=0.1,
            lambda_attention=0.05,
            input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
        )
    directory = next((tmp_path / "phase4-runs").iterdir())
    assert not (directory / "COMPLETED.json").exists()


def test_prediction_reordering_fails_exact_authoritative_membership(tmp_path, monkeypatch):
    _install_guards(monkeypatch, tmp_path)

    def reordered(context):
        result = _fake_execution(context)
        predictions = dict(result.predictions)
        predictions["val"] = predictions["val"].iloc[::-1].reset_index(drop=True)
        return replace(result, predictions=predictions)

    monkeypatch.setattr(phase4_runner, "execute_phase4_single_run", reordered)
    with pytest.raises(Phase4RunError, match="exact readiness-approved samples"):
        run_phase4_experiment(
            config_path=ROOT / "configs" / "phase4_consensdxformer.yaml",
            project_root=tmp_path,
            mode="tuning",
            lambda_ordinal=0.1,
            lambda_attention=0.05,
            input_lock_path=ROOT / "configs" / "phase4_input_lock.json",
        )


def test_noncanonical_config_cannot_unlock_final_mode(tmp_path):
    with pytest.raises(Phase4RunError, match="canonical tracked configuration"):
        run_phase4_experiment(
            config_path=tmp_path / "copied-config-with-selected-lambdas.yaml",
            project_root=tmp_path,
            mode="final",
        )


def test_canonical_config_and_input_lock_must_match_tracked_head(monkeypatch):
    monkeypatch.setattr(
        phase4_runner,
        "_git_head_blob",
        lambda path, *, project_root: b"changed\n",
    )
    with pytest.raises(Phase4RunError, match="tracked HEAD blob"):
        phase4_runner._require_tracked_phase4_inputs(
            ROOT / "configs" / "phase4_consensdxformer.yaml",
            ROOT / "configs" / "phase4_input_lock.json",
            project_root=ROOT,
        )


def test_tracked_inputs_must_belong_to_recorded_project_root(tmp_path):
    with pytest.raises(Phase4RunError, match="outside the repository"):
        phase4_runner._require_tracked_phase4_inputs(
            ROOT / "configs" / "phase4_consensdxformer.yaml",
            ROOT / "configs" / "phase4_input_lock.json",
            project_root=tmp_path,
        )
