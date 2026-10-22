"""Transactional Phase 3 baseline experiment lifecycle.

The runner has two deliberately disjoint paths. Real mode cannot reach model
construction until the complete Phase 2 readiness chain has passed. Synthetic
mode performs an infrastructure-only dry-run and never calls a trainer or
creates research-result-shaped artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import argparse
import traceback

import pandas as pd
import yaml

from .baseline_evaluation import CLASS_ORDER
from .baseline_provenance import project_git_state, utc_timestamp
from .baseline_source import (
    assert_source_snapshot_unchanged,
    snapshot_source_tree,
    verify_baseline_source,
)
from .hashing import sha256_file
from .manifest_guard import validate_frozen_manifest_bundle
from .phase3_artifacts import (
    ExperimentLayout,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    transition_status,
    write_artifact_manifest,
    write_completion_marker,
)
from .phase3_checkpoints import (
    select_best_validation_epoch,
    validate_checkpoint_pair,
    validate_epoch_history,
)
from .phase3_config import Phase3ConfigBundle, load_phase3_config
from .phase3_execution import (
    Phase3ExecutionContext,
    Phase3ExecutionResult,
    execute_locked_lungdxformer,
)
from .phase3_identity import ExperimentIdentity, build_experiment_identity
from .phase3_predictions import (
    evaluate_phase3_predictions,
    load_phase3_prediction_csv,
    ordered_sample_checksum,
    validate_prediction_table,
    write_prediction_csv,
)
from .phase3_readiness import Phase3ReadinessReport, validate_phase3_readiness
from .reproducibility import environment_audit, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Phase3RunError(RuntimeError):
    """Raised when a run cannot be completed without weakening a guard."""


@dataclass(frozen=True)
class Phase3RunResult:
    experiment_id: str
    baseline_variant: str
    seed: int
    mode: str
    state: str
    synthetic_dry_run: bool
    research_results_eligible: bool
    output_dir: Path
    status_path: Path
    provenance_path: Path
    completion_marker: Path
    artifact_manifest: Path
    prediction_paths: Mapping[str, Path]
    metrics_paths: Mapping[str, Path]

    def to_index_record(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "baseline_variant": self.baseline_variant,
            "seed": self.seed,
            "mode": self.mode,
            "state": self.state,
            "synthetic_dry_run": self.synthetic_dry_run,
            "research_results_eligible": self.research_results_eligible,
            "output_dir": str(self.output_dir),
            "status_path": str(self.status_path),
            "provenance_path": str(self.provenance_path),
            "artifact_manifest": str(self.artifact_manifest),
            "completion_marker": str(self.completion_marker),
        }


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _yaml_text(value: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(value), sort_keys=False, allow_unicode=True, default_flow_style=False
    )


def _require_project_git_state(project_root: Path, *, real: bool) -> dict[str, Any]:
    state = project_git_state(project_root)
    commit = state.get("commit")
    if not isinstance(commit, str) or len(commit) not in {40, 64}:
        raise Phase3RunError(
            "Phase 3 requires an auditable Git commit for the ConsenDxFormer repository"
        )
    if real and state.get("working_tree_clean") is not True:
        raise Phase3RunError(
            "Real Phase 3 execution requires a clean ConsenDxFormer working tree so "
            "the recorded project commit fully identifies the running code"
        )
    return state


def _build_identity(
    bundle: Phase3ConfigBundle,
    readiness: Phase3ReadinessReport,
    *,
    project_commit: str,
    baseline_commit: str,
) -> ExperimentIdentity:
    fingerprint = readiness.fingerprint
    return build_experiment_identity(
        baseline_variant=bundle.phase3_config["experiment"]["baseline_variant"],
        seed=bundle.phase3_config["experiment"]["seed"],
        baseline_source_commit=baseline_commit,
        project_commit=project_commit,
        combined_config_sha256=bundle.combined_config_sha256,
        metadata_sha256=fingerprint.metadata_sha256,
        cohort_sha256=fingerprint.logical_cohort_sha256,
        ordered_sample_sha256=fingerprint.ordered_samples_sha256,
        frozen_split_bundle_sha256=readiness.manifest_bundle_sha256,
    )


def _checkpoint_sidecar(record: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    excluded = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "effective_baseline_config",
        "validation_metrics",
    }
    return {
        **{key: value for key, value in record.items() if key not in excluded},
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
    }


def _assert_prediction_matches_context(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    split: str,
    identity: ExperimentIdentity,
) -> pd.DataFrame:
    canonical = validate_prediction_table(predictions)
    if set(canonical["split"]) != {split}:
        raise Phase3RunError(
            f"The {split} prediction artifact contains another split"
        )
    if set(canonical["seed"].astype(int)) != {identity.seed}:
        raise Phase3RunError(f"The {split} prediction artifact has the wrong seed")
    if set(canonical["baseline_variant"]) != {identity.baseline_variant}:
        raise Phase3RunError(
            f"The {split} prediction artifact has the wrong baseline variant"
        )
    if set(canonical["experiment_id"]) != {identity.experiment_id}:
        raise Phase3RunError(
            f"The {split} prediction artifact has the wrong experiment identity"
        )

    expected = metadata.loc[
        metadata["split"] == split,
        ["patient_id", "series_instance_uid", "nodule_id", "label"],
    ].copy()
    expected = expected.rename(columns={"label": "true_class"}).sort_values(
        ["patient_id", "series_instance_uid", "nodule_id"],
        kind="mergesort",
        ignore_index=True,
    )
    observed = canonical.loc[
        :, ["patient_id", "series_instance_uid", "nodule_id", "true_class"]
    ].reset_index(drop=True)
    if not observed.equals(expected):
        raise Phase3RunError(
            f"The {split} prediction rows are not the exact readiness-approved samples"
        )
    return canonical


def _validate_execution_result(
    result: Phase3ExecutionResult,
    context: Phase3ExecutionContext,
    identity: ExperimentIdentity,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, pd.DataFrame],
]:
    if not isinstance(result, Phase3ExecutionResult):
        raise Phase3RunError("The Phase 3 executor returned an unsupported result type")
    history = validate_epoch_history(result.history)
    selection = select_best_validation_epoch(history)
    if result.best_epoch != selection["epoch"]:
        raise Phase3RunError(
            "Executor best_epoch disagrees with validation-loss-only selection"
        )
    if result.last_epoch != history[-1]["epoch"]:
        raise Phase3RunError("Executor last_epoch disagrees with epoch history")

    best_path = Path(result.best_checkpoint_path).resolve()
    last_path = Path(result.last_checkpoint_path).resolve()
    if best_path != context.layout.checkpoints_dir / "best.pt":
        raise Phase3RunError("Best checkpoint is outside its assigned experiment path")
    if last_path != context.layout.checkpoints_dir / "last.pt":
        raise Phase3RunError("Last checkpoint is outside its assigned experiment path")
    if not best_path.is_file() or not last_path.is_file():
        raise Phase3RunError("Executor did not create both best.pt and last.pt")

    best, last = validate_checkpoint_pair(
        result.best_checkpoint_record, result.last_checkpoint_record
    )
    for record, kind in ((best, "best"), (last, "last")):
        expected = {
            "checkpoint_kind": kind,
            "seed": context.seed,
            "baseline_variant": context.baseline_variant,
            "config_sha256": context.configuration_sha256,
            "experiment_id": context.experiment_id,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise Phase3RunError(
                    f"{kind.capitalize()} checkpoint field {field!r} disagrees with the run"
                )
    if (
        best["epoch"] != selection["epoch"]
        or best["best_validation_loss"] != selection["value"]
    ):
        raise Phase3RunError(
            "Best checkpoint does not encode the selected validation-loss minimum"
        )

    if set(result.predictions) != {"val", "test"}:
        raise Phase3RunError("Executor must return exactly val and test predictions")
    predictions = {
        split: _assert_prediction_matches_context(
            result.predictions[split],
            context.metadata,
            split=split,
            identity=identity,
        )
        for split in ("val", "test")
    }
    required_events = {
        "best_checkpoint_selected_from_validation_loss",
        "best_checkpoint_reloaded",
        "test_evaluation_started_after_checkpoint_selection",
    }
    if not required_events.issubset(set(result.events)):
        raise Phase3RunError(
            "Executor did not attest that test evaluation began only after validation selection"
        )
    return history, best, last, predictions


def _assert_source_unchanged(
    source_root: Path,
    lock_file: Path,
    before: Mapping[str, str],
) -> None:
    after = snapshot_source_tree(source_root)
    assert_source_snapshot_unchanged(dict(before), after)
    verify_baseline_source(source_root, lock_file)


def _write_prepared_artifacts(
    layout: ExperimentLayout,
    bundle: Phase3ConfigBundle,
    identity: ExperimentIdentity,
    readiness: Phase3ReadinessReport,
    *,
    project_state: Mapping[str, Any],
    source_report: Mapping[str, Any],
    manifest_report: Mapping[str, Any],
    environment: Mapping[str, Any],
    seed_report: Mapping[str, Any],
    synthetic_dry_run: bool,
) -> tuple[list[Path], Path, dict[str, Any]]:
    phase3_path = atomic_write_text(
        layout.config_dir / "phase3_effective.yaml", _yaml_text(bundle.phase3_config)
    )
    phase1_path = atomic_write_text(
        layout.config_dir / "phase1_effective.yaml",
        _yaml_text(bundle.effective_phase1_config),
    )
    identity_path = atomic_write_json(
        layout.config_dir / "experiment_identity.json", identity.to_dict()
    )
    input_metadata = Path(readiness.metadata_path)
    metadata_path = atomic_write_bytes(
        layout.config_dir / "input_metadata.csv", input_metadata.read_bytes()
    )
    if sha256_file(metadata_path) != readiness.fingerprint.metadata_sha256:
        raise Phase3RunError(
            "Persisted metadata bytes disagree with the readiness-approved checksum"
        )
    readiness_path = atomic_write_json(
        layout.provenance_dir / "readiness.json", readiness.to_dict()
    )
    environment_path = atomic_write_json(
        layout.provenance_dir / "environment.json", dict(environment)
    )
    run_path = layout.provenance_dir / "run.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc_timestamp(),
        "finished_utc": None,
        "identity": identity.to_dict(),
        "mode": readiness.mode,
        "synthetic_dry_run": synthetic_dry_run,
        "synthetic_non_research": readiness.synthetic_non_research,
        "input_real_research_ready": readiness.real_research_ready,
        "research_results_eligible": False,
        "real_training_started": False,
        "performance_claimed": False,
        "project_git": dict(project_state),
        "baseline_source": dict(source_report),
        "phase0_manifest": dict(manifest_report),
        "readiness": readiness.to_dict(),
        "seed_controls": dict(seed_report),
        "device": {
            "requested": bundle.phase3_config["device"],
            "resolved": (
                "cuda"
                if bundle.phase3_config["device"] == "auto"
                and bool(environment.get("torch", {}).get("cuda_available", False))
                else "cpu"
            ),
        },
        "environment_record": layout.relative(environment_path),
        "configuration": {
            "phase3_input_path": str(bundle.config_path),
            "phase1_input_path": str(bundle.phase1_config_path),
            "combined_sha256": bundle.combined_config_sha256,
            "phase3_effective_path": layout.relative(phase3_path),
            "phase1_effective_path": layout.relative(phase1_path),
            "input_metadata_snapshot": layout.relative(metadata_path),
            "input_metadata_sha256": readiness.fingerprint.metadata_sha256,
        },
        "checkpoint_policy": deepcopy(bundle.phase3_config["checkpoint"]),
        "prediction_policy": deepcopy(bundle.phase3_config["predictions"]),
        "inference_policy": deepcopy(bundle.phase3_config["inference"]),
        "class_order": list(CLASS_ORDER),
        "limitations": [
            "Deterministic controls do not promise bitwise-identical results across "
            "GPU, driver, CUDA, cuDNN, PyTorch, or OS environments."
        ],
    }
    atomic_write_json(run_path, record)
    return [
        phase3_path,
        phase1_path,
        identity_path,
        metadata_path,
        readiness_path,
        environment_path,
        run_path,
    ], run_path, record


def run_phase3_experiment(
    *,
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    seed: int | None = None,
    device: str | None = None,
    synthetic_dry_run: bool = False,
) -> Phase3RunResult:
    """Validate, identify, and execute one isolated B0 or B1 experiment."""

    root = Path(project_root).resolve()
    bundle = load_phase3_config(config_path, seed=seed, device=device)
    phase3 = bundle.phase3_config
    phase1 = bundle.effective_phase1_config
    mode = str(phase3["data"]["mode"])
    real = mode == "real"
    if real and synthetic_dry_run:
        raise Phase3RunError("A real Phase 3 config cannot run as a synthetic dry-run")
    if not real and not synthetic_dry_run:
        raise Phase3RunError(
            "Synthetic Phase 3 configs require explicit synthetic_dry_run=True"
        )

    source_root = _project_path(phase1["baseline"]["repository_dir"], root)
    lock_file = _project_path(phase1["baseline"]["source_lock_file"], root)
    manifest_root = _project_path(phase3["data"]["frozen_split_dir"], root)
    output_root = _project_path(phase3["experiment"]["output_root"], root)

    # Every guard precedes output creation, model construction, and data-loader
    # construction. There is intentionally no bypass parameter.
    source_report = verify_baseline_source(source_root, lock_file)
    source_before = snapshot_source_tree(source_root)
    manifest = validate_frozen_manifest_bundle(manifest_root)
    readiness, metadata = validate_phase3_readiness(
        phase3["data"],
        manifest=manifest,
        expected_resolution=int(phase1["dataset"]["image_size"]),
        project_root=root,
        synthetic_dry_run=synthetic_dry_run,
    )
    if real and not readiness.real_research_ready:
        raise Phase3RunError("Real Phase 3 readiness was not established")
    project_state = _require_project_git_state(root, real=real)
    identity = _build_identity(
        bundle,
        readiness,
        project_commit=str(project_state["commit"]),
        baseline_commit=source_report.head_commit,
    )

    layout = ExperimentLayout.create(
        output_root,
        identity.experiment_id,
        protected_roots=(source_root, manifest_root),
    )
    transition_status(
        layout,
        "prepared",
        experiment_id=identity.experiment_id,
        details={
            "mode": mode,
            "synthetic_dry_run": synthetic_dry_run,
            "input_real_research_ready": readiness.real_research_ready,
            "research_results_eligible": False,
        },
    )

    artifacts: list[Path] = []
    run_path = layout.provenance_dir / "run.json"
    run_record: dict[str, Any] = {}
    completion_written: Path | None = None
    try:
        seed_settings = phase3["determinism"]
        seed_report = set_global_seed(
            identity.seed,
            deterministic_algorithms=bool(seed_settings["deterministic_algorithms"]),
            cudnn_benchmark=bool(seed_settings["cudnn_benchmark"]),
            warn_only=bool(seed_settings["warn_only"]),
            require_torch=real,
        )
        environment = environment_audit(require_torch=real)
        artifacts, run_path, run_record = _write_prepared_artifacts(
            layout,
            bundle,
            identity,
            readiness,
            project_state=project_state,
            source_report=source_report.to_dict(),
            manifest_report=manifest.to_dict(),
            environment=environment,
            seed_report=seed_report,
            synthetic_dry_run=synthetic_dry_run,
        )

        if synthetic_dry_run:
            _assert_source_unchanged(source_root, lock_file, source_before)
            run_record.update(
                {
                    "finished_utc": utc_timestamp(),
                    "execution_state": "synthetic_dry_run_completed",
                    "research_results_eligible": False,
                    "real_training_started": False,
                    "trainer_called": False,
                    "checkpoints_created": False,
                    "predictions_created": False,
                    "metrics_created": False,
                    "synthetic_statement": (
                        "Infrastructure-only synthetic dry-run; not a training, "
                        "evaluation, readiness, or performance result."
                    ),
                    "baseline_unchanged_after_execution": True,
                }
            )
            atomic_write_json(run_path, run_record, overwrite=True)
            artifact_manifest = write_artifact_manifest(
                layout,
                artifacts,
                experiment_id=identity.experiment_id,
                research_results_eligible=False,
            )
            completion_path = layout.experiment_dir / "DRY_RUN_COMPLETED.json"
            completed_result = Phase3RunResult(
                experiment_id=identity.experiment_id,
                baseline_variant=identity.baseline_variant,
                seed=identity.seed,
                mode=mode,
                state="synthetic_dry_run_completed",
                synthetic_dry_run=True,
                research_results_eligible=False,
                output_dir=layout.experiment_dir,
                status_path=layout.status_path,
                provenance_path=run_path,
                completion_marker=completion_path,
                artifact_manifest=artifact_manifest,
                prediction_paths={},
                metrics_paths={},
            )
            completion_written = write_completion_marker(
                layout,
                artifact_manifest,
                experiment_id=identity.experiment_id,
                synthetic_dry_run=True,
            )
            transition_status(
                layout,
                "synthetic_dry_run_completed",
                experiment_id=identity.experiment_id,
                details={
                    "completion_marker": completion_written.name,
                    "research_results_eligible": False,
                },
            )
            return completed_result

        transition_status(
            layout,
            "running",
            experiment_id=identity.experiment_id,
            details={"selection_split": "validation", "test_used_for_selection": False},
        )
        run_record["real_training_started"] = True
        run_record["execution_state"] = "running"
        atomic_write_json(run_path, run_record, overwrite=True)

        execution_config = deepcopy(phase1)
        execution_config["phase3_reproducibility"] = deepcopy(phase3["determinism"])
        context = Phase3ExecutionContext(
            experiment_id=identity.experiment_id,
            baseline_variant=identity.baseline_variant,
            seed=identity.seed,
            configuration_sha256=identity.combined_config_sha256,
            baseline_source_commit=identity.baseline_source_commit,
            metadata_sha256=identity.metadata_sha256,
            cohort_sha256=identity.cohort_sha256,
            ordered_sample_sha256=identity.ordered_sample_sha256,
            frozen_split_sha256=identity.frozen_split_bundle_sha256,
            project_commit=identity.project_commit,
            baseline_source_root=source_root,
            effective_baseline_config=execution_config,
            metadata=metadata.copy(deep=True),
            layout=layout,
        )
        result = execute_locked_lungdxformer(context)
        history, best, last, predictions = _validate_execution_result(
            result, context, identity
        )
        transition_status(
            layout,
            "finalizing",
            experiment_id=identity.experiment_id,
            details={"best_epoch": result.best_epoch, "last_epoch": result.last_epoch},
        )

        history_path = atomic_write_json(
            layout.history_dir / "epoch_history.json", list(history)
        )
        artifacts.append(history_path)
        best_metadata_path = atomic_write_json(
            layout.checkpoints_dir / "best.metadata.json",
            _checkpoint_sidecar(best, result.best_checkpoint_path),
        )
        last_metadata_path = atomic_write_json(
            layout.checkpoints_dir / "last.metadata.json",
            _checkpoint_sidecar(last, result.last_checkpoint_path),
        )
        artifacts.extend(
            [
                Path(result.best_checkpoint_path),
                Path(result.last_checkpoint_path),
                best_metadata_path,
                last_metadata_path,
            ]
        )

        prediction_paths: dict[str, Path] = {}
        metrics_paths: dict[str, Path] = {}
        sample_checksums: dict[str, str] = {}
        for split in ("val", "test"):
            prediction_path = write_prediction_csv(
                predictions[split], layout.predictions_dir / f"{split}.csv"
            )
            reloaded = load_phase3_prediction_csv(prediction_path)
            metrics = evaluate_phase3_predictions(reloaded)
            metrics_path = atomic_write_json(
                layout.metrics_dir / f"{split}.json", metrics
            )
            prediction_paths[split] = prediction_path
            metrics_paths[split] = metrics_path
            sample_checksums[split] = ordered_sample_checksum(reloaded)
            artifacts.extend([prediction_path, metrics_path])

        final_readiness, _ = validate_phase3_readiness(
            phase3["data"],
            manifest=manifest,
            expected_resolution=int(phase1["dataset"]["image_size"]),
            project_root=root,
            synthetic_dry_run=False,
        )
        if final_readiness.to_dict() != readiness.to_dict():
            raise Phase3RunError(
                "Phase 2 readiness or input fingerprints changed during training"
            )
        final_project_state = _require_project_git_state(root, real=True)
        if final_project_state.get("commit") != identity.project_commit:
            raise Phase3RunError("The project Git commit changed during training")
        _assert_source_unchanged(source_root, lock_file, source_before)
        run_record.update(
            {
                "finished_utc": utc_timestamp(),
                "execution_state": "completed",
                "research_results_eligible": True,
                "baseline_unchanged_after_execution": True,
                "readiness_revalidated_after_training": True,
                "project_commit_unchanged_after_training": True,
                "trainer_called": True,
                "selection": {
                    "criterion": "validation_loss",
                    "mode": "min",
                    "split": "validation",
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": best["best_validation_loss"],
                    "test_metrics_influenced_selection": False,
                },
                "checkpoint_files": {
                    "best": layout.relative(result.best_checkpoint_path),
                    "last": layout.relative(result.last_checkpoint_path),
                },
                "prediction_files": {
                    split: layout.relative(path)
                    for split, path in prediction_paths.items()
                },
                "metrics_files": {
                    split: layout.relative(path) for split, path in metrics_paths.items()
                },
                "prediction_sample_checksums": sample_checksums,
                "events": list(result.events),
                "performance_claimed": False,
            }
        )
        atomic_write_json(run_path, run_record, overwrite=True)
        artifact_manifest = write_artifact_manifest(
            layout,
            artifacts,
            experiment_id=identity.experiment_id,
            research_results_eligible=True,
        )
        completion_path = layout.experiment_dir / "COMPLETED.json"
        completed_result = Phase3RunResult(
            experiment_id=identity.experiment_id,
            baseline_variant=identity.baseline_variant,
            seed=identity.seed,
            mode=mode,
            state="completed",
            synthetic_dry_run=False,
            research_results_eligible=True,
            output_dir=layout.experiment_dir,
            status_path=layout.status_path,
            provenance_path=run_path,
            completion_marker=completion_path,
            artifact_manifest=artifact_manifest,
            prediction_paths=prediction_paths,
            metrics_paths=metrics_paths,
        )
        completion_written = write_completion_marker(
            layout,
            artifact_manifest,
            experiment_id=identity.experiment_id,
            synthetic_dry_run=False,
        )
        transition_status(
            layout,
            "completed",
            experiment_id=identity.experiment_id,
            details={
                "completion_marker": completion_written.name,
                "best_epoch": result.best_epoch,
                "test_used_for_selection": False,
            },
        )
        return completed_result
    except BaseException as exc:
        completion_cleanup_error: OSError | None = None
        if completion_written is not None:
            try:
                completion_written.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                completion_cleanup_error = cleanup_exc
        source_error: BaseException | None = None
        try:
            _assert_source_unchanged(source_root, lock_file, source_before)
        except BaseException as integrity_exc:
            source_error = integrity_exc
        try:
            transition_status(
                layout,
                "failed",
                experiment_id=identity.experiment_id,
                details={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "source_integrity_error": (
                        None if source_error is None else repr(source_error)
                    ),
                    "completion_cleanup_error": (
                        None
                        if completion_cleanup_error is None
                        else repr(completion_cleanup_error)
                    ),
                },
            )
        except BaseException:
            pass
        if run_record:
            run_record.update(
                {
                    "finished_utc": utc_timestamp(),
                    "execution_state": "failed",
                    "research_results_eligible": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        ),
                    },
                    "baseline_unchanged_after_execution": source_error is None,
                    "source_integrity_error": (
                        None if source_error is None else repr(source_error)
                    ),
                    "completion_cleanup_error": (
                        None
                        if completion_cleanup_error is None
                        else repr(completion_cleanup_error)
                    ),
                }
            )
            try:
                atomic_write_json(run_path, run_record, overwrite=True)
            except BaseException:
                pass
        if source_error is not None:
            raise Phase3RunError(
                "Locked baseline source integrity failed during Phase 3 execution"
            ) from source_error
        if completion_cleanup_error is not None:
            raise Phase3RunError(
                "Could not revoke a completion marker after a terminal write failure"
            ) from completion_cleanup_error
        raise


def phase3_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one guarded, content-addressed Phase 3 LungDxFormer baseline."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu"), default=None)
    parser.add_argument(
        "--synthetic-dry-run",
        action="store_true",
        help=(
            "Explicitly authorize an infrastructure-only synthetic run. This flag "
            "cannot bypass a real-data readiness failure."
        ),
    )
    args = parser.parse_args(argv)
    result = run_phase3_experiment(
        config_path=args.config,
        seed=args.seed,
        device=args.device,
        synthetic_dry_run=args.synthetic_dry_run,
    )
    print(f"PHASE3_STATE={result.state}")
    print(f"EXPERIMENT_ID={result.experiment_id}")
    print(f"OUTPUT_DIR={result.output_dir}")
    print(f"COMPLETION_MARKER={result.completion_marker}")
    return 0


__all__ = [
    "PROJECT_ROOT",
    "Phase3RunError",
    "Phase3RunResult",
    "phase3_cli",
    "run_phase3_experiment",
]
