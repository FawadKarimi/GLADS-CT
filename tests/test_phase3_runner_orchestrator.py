from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import consendxformer.phase3_runner as runner_module
import consendxformer.phase3_orchestrator as orchestrator_module
from consendxformer.config import load_config
from consendxformer.hashing import sha256_file
from consendxformer.manifest_guard import FROZEN_BUNDLE_SHA256
from consendxformer.baseline_source import PINNED_BASELINE_SHA, SourceGuardReport
from consendxformer.phase3_checkpoints import (
    build_best_checkpoint,
    build_history_record,
    build_last_checkpoint,
)
from consendxformer.phase3_execution import Phase3ExecutionResult
from consendxformer.phase3_orchestrator import (
    Phase3OrchestrationError,
    run_phase3_matrix,
)
from consendxformer.phase3_predictions import build_prediction_table
from consendxformer.phase3_readiness import (
    Phase3InputFingerprint,
    Phase3ReadinessReport,
)
from consendxformer.phase3_runner import (
    Phase3RunError,
    Phase3RunResult,
    run_phase3_experiment,
)


@pytest.fixture(autouse=True)
def _offline_locked_source_guard(monkeypatch):
    """Keep lifecycle tests independent of the ignored external checkout."""

    report = SourceGuardReport(
        repository_path=str(ROOT / "external" / "LungDxFormer"),
        lock_file=str(ROOT / "locks" / "baseline_sources.yaml"),
        repository_url="https://github.com/Srinivasa459/LungDxFormer.git",
        expected_commit=PINNED_BASELINE_SHA,
        head_commit=PINNED_BASELINE_SHA,
        working_tree_clean=True,
        source_tree_sha256="9" * 64,
    )
    monkeypatch.setattr(runner_module, "verify_baseline_source", lambda *args: report)
    monkeypatch.setattr(
        runner_module, "snapshot_source_tree", lambda *args: {"locked": "9" * 64}
    )
    monkeypatch.setattr(
        runner_module, "assert_source_snapshot_unchanged", lambda *args: None
    )


def _write_config(source_name: str, destination: Path, output_root: Path) -> Path:
    config = load_config(ROOT / "configs" / source_name)
    config["experiment"]["output_root"] = str(output_root.resolve())
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def _fake_readiness(metadata_path: Path) -> tuple[Phase3ReadinessReport, pd.DataFrame]:
    metadata = pd.DataFrame(
        {
            "patient_id": ["P-train", "P-val", "P-test"],
            "image_path": ["train.npy", "val.npy", "test.npy"],
            "label": ["benign", "indeterminate", "malignant"],
            "label_id": [0, 1, 2],
            "split": ["train", "val", "test"],
            "nodule_id": ["N-train", "N-val", "N-test"],
            "series_instance_uid": ["S-train", "S-val", "S-test"],
        }
    )
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    fingerprint = Phase3InputFingerprint(
        metadata_sha256=sha256_file(metadata_path),
        logical_cohort_sha256="2" * 64,
        ordered_samples_sha256="3" * 64,
        source_readiness_sha256="4" * 64,
        canonical_bundle_sha256="5" * 64,
        canonical_checksum_ledger_sha256="6" * 64,
        rendered_bundle_sha256="7" * 64,
        rendered_checksum_ledger_sha256="8" * 64,
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
        cohort_view="paper_aligned_available_reader_mean",
        expected_resolution=96,
        metadata_path=str(metadata_path.resolve()),
        sample_count=3,
        split_counts=(("train", 1), ("val", 1), ("test", 1)),
        manifest_bundle_sha256=FROZEN_BUNDLE_SHA256,
        fingerprint=fingerprint,
        limitations=(),
    )
    return report, metadata


def _install_fake_real_preflight(monkeypatch, directory: Path):
    report, metadata = _fake_readiness(directory / "fake-approved-metadata.csv")
    monkeypatch.setattr(
        runner_module,
        "validate_phase3_readiness",
        lambda *args, **kwargs: (report, metadata.copy(deep=True)),
    )
    monkeypatch.setattr(
        runner_module,
        "_require_project_git_state",
        lambda *args, **kwargs: {
            "commit": "7" * 40,
            "branch": "phase3-baseline-experiments",
            "working_tree_clean": True,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "set_global_seed",
        lambda *args, **kwargs: {
            "seed": args[0],
            "bitwise_reproducibility_across_environments_claimed": False,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "environment_audit",
        lambda **kwargs: {
            "python": {"version": "test"},
            "torch": {"available": False, "gpu_names": []},
            "bitwise_reproducibility_across_environments_claimed": False,
        },
    )


def _successful_fake_executor(context):
    history = (
        build_history_record(epoch=1, train_loss=0.8, validation_loss=0.4),
        build_history_record(epoch=2, train_loss=0.6, validation_loss=0.2),
    )
    common = {
        "model_state": {"weight": [1.0]},
        "optimizer_state": {"step": 2},
        "scheduler_state": {"last_epoch": 2},
        "best_validation_loss": 0.2,
        "seed": context.seed,
        "baseline_variant": context.baseline_variant,
        "config_sha256": context.configuration_sha256,
        "experiment_id": context.experiment_id,
    }
    best = build_best_checkpoint(epoch=2, **common)
    last = build_last_checkpoint(epoch=2, best_epoch=2, **common)
    best_path = context.layout.checkpoints_dir / "best.pt"
    last_path = context.layout.checkpoints_dir / "last.pt"
    best_path.write_bytes(b"fake-best-checkpoint\n")
    last_path.write_bytes(b"fake-last-checkpoint\n")

    predictions = {}
    for split, logits in (("val", [[0.0, 2.0, 0.0]]), ("test", [[0.0, 0.0, 2.0]])):
        frame = context.metadata.loc[context.metadata["split"] == split].reset_index(
            drop=True
        )
        predictions[split] = build_prediction_table(
            frame,
            logits,
            seed=context.seed,
            baseline_variant=context.baseline_variant,
            experiment_id=context.experiment_id,
        )
    return Phase3ExecutionResult(
        history=history,
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        best_checkpoint_record=best,
        last_checkpoint_record=last,
        predictions=predictions,
        best_epoch=2,
        last_epoch=2,
        events=(
            "best_checkpoint_selected_from_validation_loss",
            "best_checkpoint_reloaded",
            "test_evaluation_started_after_checkpoint_selection",
        ),
    )


def test_real_runner_fails_before_output_while_source_gate_is_incomplete():
    with pytest.raises(RuntimeError, match="source readiness report does not exist"):
        run_phase3_experiment(config_path=ROOT / "configs" / "phase3_b0.yaml")


def test_synthetic_dry_run_is_explicit_and_emits_no_result_artifacts(
    tmp_path, monkeypatch
):
    config = _write_config(
        "phase3_synthetic_b0.yaml", tmp_path / "synthetic-b0.yaml", tmp_path / "out"
    )
    called = False

    def forbidden_executor(context):
        nonlocal called
        called = True
        raise AssertionError("synthetic dry-run called the trainer")

    monkeypatch.setattr(
        runner_module, "execute_locked_lungdxformer", forbidden_executor
    )

    with pytest.raises(Phase3RunError, match="explicit synthetic_dry_run"):
        run_phase3_experiment(config_path=config)

    result = run_phase3_experiment(
        config_path=config,
        synthetic_dry_run=True,
    )

    assert called is False
    assert result.state == "synthetic_dry_run_completed"
    assert result.research_results_eligible is False
    assert result.completion_marker.name == "DRY_RUN_COMPLETED.json"
    assert not (result.output_dir / "COMPLETED.json").exists()
    assert list((result.output_dir / "checkpoints").iterdir()) == []
    assert list((result.output_dir / "predictions").iterdir()) == []
    assert list((result.output_dir / "metrics").iterdir()) == []
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["trainer_called"] is False
    assert provenance["performance_claimed"] is False
    assert provenance["synthetic_non_research"] is True

    manifest = json.loads(result.artifact_manifest.read_text(encoding="utf-8"))
    assert manifest["research_results_eligible"] is False
    for relative, digest in manifest["files"].items():
        assert sha256_file(result.output_dir / relative) == digest

    with pytest.raises(FileExistsError, match="completed Phase 3 experiment"):
        run_phase3_experiment(config_path=config, synthetic_dry_run=True)


def test_failed_real_execution_is_marked_failed_and_never_completed(
    tmp_path, monkeypatch
):
    config = _write_config(
        "phase3_b0.yaml", tmp_path / "real-b0.yaml", tmp_path / "failed-out"
    )
    _install_fake_real_preflight(monkeypatch, tmp_path)

    def fail(context):
        raise RuntimeError("intentional executor failure")

    monkeypatch.setattr(runner_module, "execute_locked_lungdxformer", fail)
    with pytest.raises(RuntimeError, match="intentional executor failure"):
        run_phase3_experiment(config_path=config)

    experiments = list((tmp_path / "failed-out" / "experiments").iterdir())
    assert len(experiments) == 1
    experiment = experiments[0]
    status = json.loads((experiment / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["details"]["error_type"] == "RuntimeError"
    assert not (experiment / "COMPLETED.json").exists()
    assert not (experiment / "DRY_RUN_COMPLETED.json").exists()
    assert not (experiment / "artifact_manifest.json").exists()


def test_terminal_status_failure_removes_synthetic_completion_marker(
    tmp_path, monkeypatch
):
    config = _write_config(
        "phase3_synthetic_b0.yaml",
        tmp_path / "terminal-failure.yaml",
        tmp_path / "terminal-failure-out",
    )
    original_transition = runner_module.transition_status

    def fail_terminal(layout, state, **kwargs):
        if state == "synthetic_dry_run_completed":
            raise OSError("simulated terminal status failure")
        return original_transition(layout, state, **kwargs)

    monkeypatch.setattr(runner_module, "transition_status", fail_terminal)
    with pytest.raises(OSError, match="terminal status failure"):
        run_phase3_experiment(config_path=config, synthetic_dry_run=True)

    experiment = next(
        (tmp_path / "terminal-failure-out" / "experiments").iterdir()
    )
    status = json.loads((experiment / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert not (experiment / "DRY_RUN_COMPLETED.json").exists()


def test_real_result_contract_writes_best_last_predictions_and_shared_metrics(
    tmp_path, monkeypatch
):
    config = _write_config(
        "phase3_b0.yaml", tmp_path / "real-b0.yaml", tmp_path / "completed-out"
    )
    _install_fake_real_preflight(monkeypatch, tmp_path)

    monkeypatch.setattr(
        runner_module, "execute_locked_lungdxformer", _successful_fake_executor
    )
    result = run_phase3_experiment(config_path=config)

    assert result.state == "completed"
    assert result.completion_marker.name == "COMPLETED.json"
    assert set(result.prediction_paths) == {"val", "test"}
    assert set(result.metrics_paths) == {"val", "test"}
    assert (result.output_dir / "checkpoints" / "best.pt").is_file()
    assert (result.output_dir / "checkpoints" / "last.pt").is_file()
    assert (result.output_dir / "history" / "epoch_history.json").is_file()
    test_metrics = json.loads(result.metrics_paths["test"].read_text(encoding="utf-8"))
    assert test_metrics["class_order"] == ["benign", "indeterminate", "malignant"]
    assert test_metrics["auroc_ovr"]["macro"] is None
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["selection"]["split"] == "validation"
    assert provenance["selection"]["test_metrics_influenced_selection"] is False


def test_synthetic_matrix_creates_independent_runs_and_correct_index(tmp_path):
    output_root = tmp_path / "matrix-out"
    b0 = _write_config(
        "phase3_synthetic_b0.yaml", tmp_path / "matrix-b0.yaml", output_root
    )
    b1 = _write_config(
        "phase3_synthetic_b1.yaml", tmp_path / "matrix-b1.yaml", output_root
    )

    result = run_phase3_matrix(
        config_paths={"B0": b0, "B1": b1},
        seeds=[101],
        synthetic_dry_run=True,
    )

    assert result.state == "completed"
    assert result.run_count == 2
    assert result.succeeded == 2
    assert result.failed == 0
    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["summary"] == {
        "planned": 2,
        "succeeded": 2,
        "failed": 0,
        "comparison_failures": 0,
    }
    assert len({run["experiment_id"] for run in index["runs"]}) == 2
    assert {run["baseline_variant"] for run in index["runs"]} == {"B0", "B1"}
    assert all(run["state"] == "synthetic_dry_run_completed" for run in index["runs"])
    assert index["postflight_fair_pairs"][0]["eligible"] is True

    with pytest.raises(FileExistsError, match="run index"):
        run_phase3_matrix(
            config_paths={"B0": b0, "B1": b1},
            seeds=[101],
            synthetic_dry_run=True,
        )


def test_matrix_records_one_failure_without_corrupting_another_run(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "isolated-out"
    b0 = _write_config(
        "phase3_synthetic_b0.yaml", tmp_path / "isolated-b0.yaml", output_root
    )
    b1 = _write_config(
        "phase3_synthetic_b1.yaml", tmp_path / "isolated-b1.yaml", output_root
    )

    def fake_runner(**kwargs):
        config_name = Path(kwargs["config_path"]).name
        variant = "B0" if "b0" in config_name else "B1"
        if variant == "B0":
            raise RuntimeError("isolated B0 failure")
        directory = output_root / "experiments" / "fake-b1"
        directory.mkdir(parents=True)
        marker = directory / "DRY_RUN_COMPLETED.json"
        marker.write_text("{}\n", encoding="utf-8")
        provenance = directory / "run.json"
        provenance.write_text(
            json.dumps(
                {
                    "identity": {
                        "experiment_id": "fake-b1",
                        "baseline_variant": "B1",
                        "seed": kwargs["seed"],
                        "cohort_sha256": "2" * 64,
                        "frozen_split_bundle_sha256": FROZEN_BUNDLE_SHA256,
                        "ordered_sample_sha256": "3" * 64,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return Phase3RunResult(
            experiment_id="fake-b1",
            baseline_variant="B1",
            seed=kwargs["seed"],
            mode="synthetic",
            state="synthetic_dry_run_completed",
            synthetic_dry_run=True,
            research_results_eligible=False,
            output_dir=directory,
            status_path=directory / "status.json",
            provenance_path=provenance,
            completion_marker=marker,
            artifact_manifest=directory / "artifact_manifest.json",
            prediction_paths={},
            metrics_paths={},
        )

    monkeypatch.setattr(
        orchestrator_module, "run_phase3_experiment", fake_runner
    )
    result = run_phase3_matrix(
        config_paths={"B0": b0, "B1": b1},
        seeds=[202],
        synthetic_dry_run=True,
    )

    assert result.state == "completed_with_failures"
    assert result.succeeded == 1
    assert result.failed == 1
    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["summary"]["planned"] == 2
    assert index["summary"]["succeeded"] == 1
    assert index["summary"]["failed"] == 1
    assert (output_root / "experiments" / "fake-b1" / "DRY_RUN_COMPLETED.json").is_file()


def test_matrix_requires_nonempty_unique_explicit_seeds(tmp_path):
    configs = {
        "B0": ROOT / "configs" / "phase3_synthetic_b0.yaml",
        "B1": ROOT / "configs" / "phase3_synthetic_b1.yaml",
    }
    with pytest.raises(Phase3OrchestrationError, match="At least one"):
        run_phase3_matrix(config_paths=configs, seeds=[], synthetic_dry_run=True)
    with pytest.raises(Phase3OrchestrationError, match="duplicates"):
        run_phase3_matrix(config_paths=configs, seeds=[1, 1], synthetic_dry_run=True)
