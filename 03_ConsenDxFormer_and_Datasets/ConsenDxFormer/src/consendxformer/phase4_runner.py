"""Fail-closed orchestration for one Phase 4 tuning or final run."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .baseline_provenance import project_git_state, utc_timestamp
from .hashing import sha256_file, sha256_json
from .manifest_guard import validate_frozen_manifest_bundle
from .phase3_artifacts import atomic_write_csv, atomic_write_json
from .phase3_checkpoints import validate_epoch_history
from .phase4_artifacts import (
    Phase4ExperimentLayout,
    transition_phase4_status,
    write_phase4_artifact_manifest,
    write_phase4_completion_marker,
)
from .phase4_checkpoints import (
    Phase4CheckpointContext,
    load_and_validate_phase4_checkpoints,
    select_phase4_best_epoch,
)
from .phase4_config import Phase4ConfigBundle, load_phase4_config
from .phase4_execution import (
    Phase4ExecutionContext,
    Phase4ExecutionResult,
    execute_phase4_single_run,
)
from .phase4_input_lock import (
    Phase4InputLock,
    load_phase4_input_lock,
    validate_phase4_input_lock,
)
from .phase4_predictions import (
    evaluate_phase4_predictions,
    load_phase4_prediction_csv,
    validate_phase4_prediction_table,
    write_phase4_prediction_csv,
)
from .phase4_protocol import (
    Phase4RunSpec,
    resolve_phase4_run_spec,
    tuning_protocol_record,
    tuning_protocol_sha256,
)
from .phase4_readiness import Phase4ReadinessReport, validate_phase4_readiness
from .phase4_training import compute_training_class_weights, split_phase4_supervision
from .reproducibility import environment_audit, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase4_consensdxformer.yaml"
DEFAULT_INPUT_LOCK = PROJECT_ROOT / "configs" / "phase4_input_lock.json"


class Phase4RunError(RuntimeError):
    """Raised when Phase 4 execution cannot make an auditable completion claim."""


@dataclass(frozen=True)
class Phase4RunResult:
    experiment_id: str
    run_mode: str
    seed: int
    lambda_ordinal: float
    lambda_attention: float
    state: str
    output_dir: Path
    status_path: Path
    provenance_path: Path
    completion_marker: Path
    artifact_manifest: Path
    prediction_paths: Mapping[str, Path]
    metrics_paths: Mapping[str, Path]


def _project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_project_state(root: Path) -> dict[str, Any]:
    state = project_git_state(root)
    if not state.get("commit") or not state.get("branch"):
        raise Phase4RunError("Phase 4 requires an identifiable Git commit and branch")
    if state.get("working_tree_clean") is not True:
        raise Phase4RunError("Phase 4 requires a clean project working tree")
    if state.get("branch") != "phase4-consendxformer":
        raise Phase4RunError("Phase 4 execution is locked to branch phase4-consendxformer")
    return state


def _git_head_blob(path: Path, *, project_root: Path) -> bytes:
    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise Phase4RunError(f"Locked Phase 4 input is outside the repository: {path}") from exc
    process = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise Phase4RunError(f"Locked Phase 4 input is not tracked at HEAD: {relative}")
    return bytes(process.stdout)


def _require_tracked_phase4_inputs(
    config_path: Path,
    input_lock_path: Path,
    *,
    project_root: Path,
) -> None:
    """Bind methodology and selected lambdas to the recorded project commit."""

    for path, role in (
        (config_path, "configuration"),
        (input_lock_path, "input lock"),
    ):
        try:
            working = path.read_bytes().replace(b"\r\n", b"\n")
        except OSError as exc:
            raise Phase4RunError(f"Cannot read canonical Phase 4 {role}: {path}") from exc
        committed = _git_head_blob(path, project_root=project_root).replace(
            b"\r\n", b"\n"
        )
        if working != committed:
            raise Phase4RunError(
                f"Canonical Phase 4 {role} does not match its tracked HEAD blob"
            )


def _effective_config(
    bundle: Phase4ConfigBundle,
    run_spec: Phase4RunSpec,
    *,
    input_lock_sha256: str,
) -> tuple[dict[str, Any], str]:
    effective = deepcopy(bundle.phase4_config)
    effective["experiment"]["seed"] = run_spec.seed
    effective["execution"] = {
        "single_run_only": True,
        "run_mode": run_spec.mode,
        "lambda_ordinal": run_spec.lambda_ordinal,
        "lambda_attention": run_spec.lambda_attention,
        "evaluate_test": run_spec.evaluate_test,
        "input_lock_sha256": input_lock_sha256,
        "tuning_protocol_sha256": tuning_protocol_sha256(),
    }
    digest = sha256_json(
        {
            "schema_version": 1,
            "base_combined_config_sha256": bundle.combined_config_sha256,
            "effective_config": effective,
            "input_lock_sha256": input_lock_sha256,
            "tuning_protocol_sha256": tuning_protocol_sha256(),
        }
    )
    return effective, digest


def _identity(
    *,
    run_spec: Phase4RunSpec,
    configuration_sha256: str,
    project_commit: str,
    readiness: Phase4ReadinessReport | Any,
    input_lock_sha256: str,
) -> dict[str, Any]:
    report = readiness.to_dict()
    payload = {
        "schema_version": 1,
        "run_mode": run_spec.mode,
        "seed": run_spec.seed,
        "lambda_ordinal": run_spec.lambda_ordinal,
        "lambda_attention": run_spec.lambda_attention,
        "project_commit": project_commit,
        "configuration_sha256": configuration_sha256,
        "manifest_bundle_sha256": report["manifest_bundle_sha256"],
        "phase3_input_fingerprint": report["phase3_input_fingerprint"],
        "supervision_sha256": report["supervision_sha256"],
        "ordered_supervision_sha256": report["ordered_supervision_sha256"],
        "input_lock_sha256": input_lock_sha256,
        "cohort_view": report["cohort_view"],
        "expected_resolution": report["expected_resolution"],
        "sample_count": report["sample_count"],
        "split_counts": report["split_counts"],
    }
    digest = sha256_json(payload)
    return {
        "experiment_id": f"phase4-{run_spec.mode}-s{run_spec.seed}-{digest[:16]}",
        "digest": digest,
        **payload,
    }


def _validate_supervision_against_readiness(
    supervision: pd.DataFrame,
    readiness: Phase4ReadinessReport | Any,
) -> None:
    splits = split_phase4_supervision(supervision)
    report = readiness.to_dict()
    if len(supervision) != int(report["sample_count"]):
        raise Phase4RunError("Readiness-approved supervision row count is inconsistent")
    actual_counts = {split: len(frame) for split, frame in splits.items()}
    if actual_counts != {key: int(value) for key, value in report["split_counts"].items()}:
        raise Phase4RunError("Readiness-approved supervision split counts are inconsistent")


def _checkpoint_sidecar(record: Mapping[str, Any], path: Path) -> dict[str, Any]:
    excluded = {"model_state", "optimizer_state", "scheduler_state"}
    return {
        **{key: value for key, value in record.items() if key not in excluded},
        "checkpoint_file": path.name,
        "checkpoint_sha256": sha256_file(path),
    }


def _assert_predictions_match(
    predictions: pd.DataFrame,
    supervision: pd.DataFrame,
    *,
    split: str,
    identity: Mapping[str, Any],
) -> pd.DataFrame:
    validated = validate_phase4_prediction_table(predictions)
    expected = supervision.loc[
        supervision["split"] == split,
        ["patient_id", "nodule_id", "series_instance_uid", "split", "label"],
    ].reset_index(drop=True)
    observed = validated.loc[
        :, ["patient_id", "nodule_id", "series_instance_uid", "split", "true_class"]
    ].reset_index(drop=True)
    expected = expected.rename(columns={"label": "true_class"})
    if observed.to_dict("records") != expected.to_dict("records"):
        raise Phase4RunError(
            f"{split} predictions do not preserve the exact readiness-approved samples"
        )
    constant_expectations = {
        "seed": identity["seed"],
        "lambda_ordinal": identity["lambda_ordinal"],
        "lambda_attention": identity["lambda_attention"],
        "run_mode": identity["run_mode"],
        "experiment_id": identity["experiment_id"],
    }
    for field, expected_value in constant_expectations.items():
        if validated[field].nunique(dropna=False) != 1 or validated[field].iloc[0] != expected_value:
            raise Phase4RunError(f"Prediction provenance field {field!r} is inconsistent")
    return validated


def _validate_epoch_classification_metrics(
    metrics: Mapping[str, Any], *, location: str
) -> None:
    for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if field not in metrics:
            raise Phase4RunError(f"Epoch history lacks {location}.{field}")
        try:
            value = float(metrics[field])
        except (TypeError, ValueError) as exc:
            raise Phase4RunError(f"Epoch metric {location}.{field} is not numeric") from exc
        if not 0.0 <= value <= 1.0:
            raise Phase4RunError(f"Epoch metric {location}.{field} is outside [0, 1]")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping) or set(per_class) != set(CLASS_ORDER):
        raise Phase4RunError(
            f"Epoch history {location}.per_class must contain the fixed class order"
        )
    for class_name in CLASS_ORDER:
        class_metrics = per_class[class_name]
        if not isinstance(class_metrics, Mapping):
            raise Phase4RunError(
                f"Epoch history {location}.per_class.{class_name} is invalid"
            )
        for field in ("precision", "recall", "f1"):
            if field not in class_metrics:
                raise Phase4RunError(
                    f"Epoch history lacks {location}.per_class.{class_name}.{field}"
                )
            try:
                value = float(class_metrics[field])
            except (TypeError, ValueError) as exc:
                raise Phase4RunError(
                    f"Epoch metric {location}.per_class.{class_name}.{field} is not numeric"
                ) from exc
            if not 0.0 <= value <= 1.0:
                raise Phase4RunError(
                    f"Epoch metric {location}.per_class.{class_name}.{field} is outside [0, 1]"
                )


def _validate_execution(
    result: Phase4ExecutionResult,
    context: Phase4ExecutionContext,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, pd.DataFrame],
]:
    if not isinstance(result, Phase4ExecutionResult):
        raise Phase4RunError("Phase 4 executor returned an invalid result type")
    try:
        history = validate_epoch_history(result.history)
        selection = select_phase4_best_epoch(history)
    except (TypeError, ValueError) as exc:
        raise Phase4RunError("Phase 4 history violates validation-only selection") from exc
    for record in history:
        for role in ("train", "validation"):
            metrics = record[role]
            if "total_loss" not in metrics or float(metrics["loss"]) != float(
                metrics["total_loss"]
            ):
                raise Phase4RunError(f"{role} loss must explicitly equal total_loss")
            for field in ("diagnostic_loss", "ordinal_loss", "attention_loss"):
                if field not in metrics:
                    raise Phase4RunError(f"Epoch history lacks {role}.{field}")
            _validate_epoch_classification_metrics(
                metrics, location=f"epoch_{record['epoch']}.{role}"
            )
    checkpoint_context = Phase4CheckpointContext.from_execution_context(context)
    expected_checkpoint_paths = {
        "best": (context.layout.checkpoints_dir / "best.pt").resolve(),
        "last": (context.layout.checkpoints_dir / "last.pt").resolve(),
    }
    if Path(result.best_checkpoint_path).resolve() != expected_checkpoint_paths[
        "best"
    ] or Path(result.last_checkpoint_path).resolve() != expected_checkpoint_paths["last"]:
        raise Phase4RunError("Executor checkpoint paths escaped the experiment layout")
    try:
        best, last = load_and_validate_phase4_checkpoints(
            result.best_checkpoint_path,
            result.last_checkpoint_path,
            context=checkpoint_context,
            history=history,
            map_location="cpu",
        )
    except (TypeError, ValueError) as exc:
        raise Phase4RunError("Persisted Phase 4 checkpoints failed validation") from exc
    if result.best_epoch != selection["epoch"] or result.last_epoch != history[-1]["epoch"]:
        raise Phase4RunError("Execution epoch summary disagrees with validation history")
    expected_splits = {"val"} if context.run_mode == "tuning" else {"val", "test"}
    if set(result.predictions) != expected_splits or set(result.metrics) != expected_splits:
        raise Phase4RunError("Execution outputs disagree with run-mode split policy")
    predictions = {
        split: _assert_predictions_match(
            result.predictions[split],
            context.supervision,
            split=split,
            identity={
                "seed": context.seed,
                "lambda_ordinal": context.lambda_ordinal,
                "lambda_attention": context.lambda_attention,
                "run_mode": context.run_mode,
                "experiment_id": context.experiment_id,
            },
        )
        for split in expected_splits
    }
    expected_weights = compute_training_class_weights(
        context.supervision, num_classes=len(CLASS_ORDER)
    )
    if len(result.class_weights) != len(expected_weights) or any(
        abs(float(actual) - float(expected)) > 1e-12
        for actual, expected in zip(result.class_weights, expected_weights)
    ):
        raise Phase4RunError("Executor class weights differ from train-only B1 weights")
    events = list(result.events)
    required_events = {
        "best_checkpoint_selected_from_validation_total_loss",
        "persisted_checkpoints_reloaded_and_validated",
    }
    if not required_events.issubset(events):
        raise Phase4RunError("Execution lacks checkpoint-selection audit events")
    test_event = "test_evaluation_started_after_final_selection_and_best_reload"
    if context.run_mode == "tuning" and (
        test_event in events or any("test_evaluation" in event for event in events)
    ):
        raise Phase4RunError("Tuning execution contains a test-evaluation event")
    selection_event = "best_checkpoint_selected_from_validation_total_loss"
    persisted_event = "persisted_checkpoints_reloaded_and_validated"
    if events.index(selection_event) >= events.index(persisted_event):
        raise Phase4RunError("Persisted checkpoints were audited before selection")
    if context.run_mode == "tuning":
        reload_event = "best_checkpoint_reloaded_for_validation"
        if reload_event not in events or events.index(persisted_event) >= events.index(
            reload_event
        ):
            raise Phase4RunError("Tuning validation did not follow persisted best reload")
    if context.run_mode == "final":
        reload_event = "best_checkpoint_reloaded_for_final_validation_and_test"
        if reload_event not in events or test_event not in events:
            raise Phase4RunError("Final execution lacks best-reload/test-order events")
        if not (
            events.index(selection_event)
            < events.index(persisted_event)
            < events.index(reload_event)
            < events.index(test_event)
        ):
            raise Phase4RunError("Final test started before persisted best reload")
    return history, best, last, predictions


def _write_prepared(
    layout: Phase4ExperimentLayout,
    *,
    bundle: Phase4ConfigBundle,
    effective_config: Mapping[str, Any],
    identity: Mapping[str, Any],
    input_lock: Phase4InputLock,
    supervision: pd.DataFrame,
    readiness: Phase4ReadinessReport | Any,
    manifest: Any,
    environment: Mapping[str, Any],
    seed_controls: Mapping[str, Any],
    project_state: Mapping[str, Any],
    run_mode: str,
) -> tuple[list[Path], Path, dict[str, Any]]:
    supervision_snapshot = (
        supervision
        if run_mode == "final"
        else supervision.loc[supervision["split"].isin(("train", "val"))]
    ).reset_index(drop=True)
    artifacts = [
        atomic_write_json(layout.config_dir / "phase4_config.json", bundle.phase4_config),
        atomic_write_json(layout.config_dir / "b1_anchor_config.json", bundle.b1_anchor_config),
        atomic_write_json(layout.config_dir / "effective_config.json", effective_config),
        atomic_write_json(layout.config_dir / "run_identity.json", identity),
        atomic_write_json(layout.config_dir / "input_lock.json", input_lock.to_dict()),
        atomic_write_csv(
            layout.config_dir / "input_supervision.csv", supervision_snapshot
        ),
        atomic_write_json(
            layout.provenance_dir / "readiness_before.json", readiness.to_dict()
        ),
        atomic_write_json(
            layout.provenance_dir / "phase0_manifest.json", manifest.to_dict()
        ),
        atomic_write_json(layout.provenance_dir / "environment.json", environment),
        atomic_write_json(
            layout.provenance_dir / "tuning_protocol.json", tuning_protocol_record()
        ),
    ]
    run_path = layout.provenance_dir / "run.json"
    record = {
        "schema_version": 1,
        "created_utc": utc_timestamp(),
        "finished_utc": None,
        "execution_state": "prepared",
        "identity": dict(identity),
        "run_mode": identity["run_mode"],
        "lambda_ordinal": identity["lambda_ordinal"],
        "lambda_attention": identity["lambda_attention"],
        "project_git_before": dict(project_state),
        "project_git_after": None,
        "configuration_sha256": identity["configuration_sha256"],
        "input_lock_sha256": identity["input_lock_sha256"],
        "manifest_bundle_sha256": identity["manifest_bundle_sha256"],
        "phase3_input_fingerprint": identity["phase3_input_fingerprint"],
        "supervision_sha256": identity["supervision_sha256"],
        "ordered_supervision_sha256": identity["ordered_supervision_sha256"],
        "input_supervision_snapshot_splits": (
            ["train", "val", "test"] if run_mode == "final" else ["train", "val"]
        ),
        "effective_config_path": "config/effective_config.json",
        "seed_controls": dict(seed_controls),
        "environment_path": "provenance/environment.json",
        "architecture_toggles": {
            "model": {
                field: deepcopy(effective_config["model"][field])
                for field in (
                    "use_transformer",
                    "use_positional_encoding",
                    "use_spatial_attention",
                    "fusion_type",
                    "use_raw_transformer_in_fusion",
                    "use_hierarchical_fusion",
                )
            },
            "ablation": deepcopy(effective_config["ablation"]),
        },
        "determinism": deepcopy(effective_config["determinism"]),
        "selection": {
            "criterion": "validation_loss",
            "metric": "total_loss",
            "mode": "min",
            "split": "val",
            "test_influenced_selection": False,
        },
        "real_training_started": False,
        "research_results_eligible": False,
        "test_results_eligible": False,
        "performance_claimed": False,
    }
    artifacts.append(atomic_write_json(run_path, record))
    return artifacts, run_path, record


def run_phase4_experiment(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    project_root: str | Path = PROJECT_ROOT,
    mode: str,
    seed: int | None = None,
    lambda_ordinal: float | None = None,
    lambda_attention: float | None = None,
    input_lock_path: str | Path = DEFAULT_INPUT_LOCK,
) -> Phase4RunResult:
    """Run one guarded experiment; this function never executes a tuning matrix."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise Phase4RunError(f"project_root does not exist: {root}")
    requested_config = Path(config_path).resolve()
    if requested_config != DEFAULT_CONFIG.resolve():
        raise Phase4RunError(
            "Phase 4 execution requires the canonical tracked configuration"
        )
    requested_lock = Path(input_lock_path).resolve()
    if requested_lock != DEFAULT_INPUT_LOCK.resolve():
        raise Phase4RunError("Phase 4 execution requires the canonical tracked input lock")
    bundle = load_phase4_config(requested_config)
    run_spec = resolve_phase4_run_spec(
        bundle.phase4_config,
        mode=mode,
        seed=seed,
        lambda_ordinal=lambda_ordinal,
        lambda_attention=lambda_attention,
    )
    _require_tracked_phase4_inputs(
        requested_config,
        requested_lock,
        project_root=root,
    )
    input_lock = load_phase4_input_lock(requested_lock)
    input_lock_sha256 = sha256_json(input_lock.to_dict())
    phase4 = bundle.phase4_config
    manifest_root = _project_path(phase4["data"]["frozen_split_dir"], root)
    output_root = _project_path(phase4["experiment"]["output_root"], root)
    rendered_root = _project_path(phase4["data"]["rendered_dir"], root)

    # These guards are intentionally before layout, model, Dataset, or DataLoader creation.
    manifest = validate_frozen_manifest_bundle(manifest_root)
    readiness, supervision = validate_phase4_readiness(
        phase4["data"],
        manifest=manifest,
        expected_resolution=int(phase4["dataset"]["image_size"]),
        project_root=root,
        synthetic_dry_run=False,
    )
    validate_phase4_input_lock(input_lock, readiness)
    if not readiness.ready_for_requested_mode or not readiness.real_research_ready:
        raise Phase4RunError("Phase 4 real research readiness was not established")
    _validate_supervision_against_readiness(supervision, readiness)
    project_before = _require_project_state(root)
    effective_config, config_sha256 = _effective_config(
        bundle, run_spec, input_lock_sha256=input_lock_sha256
    )
    identity = _identity(
        run_spec=run_spec,
        configuration_sha256=config_sha256,
        project_commit=str(project_before["commit"]),
        readiness=readiness,
        input_lock_sha256=input_lock_sha256,
    )
    layout = Phase4ExperimentLayout.create(
        output_root,
        identity["experiment_id"],
        protected_roots=(manifest_root, rendered_root),
    )
    transition_phase4_status(
        layout,
        "prepared",
        experiment_id=identity["experiment_id"],
        details={"run_mode": run_spec.mode, "test_evaluation": run_spec.evaluate_test},
    )
    run_path = layout.provenance_dir / "run.json"
    run_record: dict[str, Any] = {}
    completion: Path | None = None
    try:
        deterministic = effective_config["determinism"]
        seed_controls = set_global_seed(
            run_spec.seed,
            deterministic_algorithms=bool(deterministic["deterministic_algorithms"]),
            cudnn_benchmark=bool(deterministic["cudnn_benchmark"]),
            warn_only=bool(deterministic["warn_only"]),
            require_torch=True,
        )
        environment = environment_audit(require_torch=True)
        artifacts, run_path, run_record = _write_prepared(
            layout,
            bundle=bundle,
            effective_config=effective_config,
            identity=identity,
            input_lock=input_lock,
            supervision=supervision,
            readiness=readiness,
            manifest=manifest,
            environment=environment,
            seed_controls=seed_controls,
            project_state=project_before,
            run_mode=run_spec.mode,
        )
        transition_phase4_status(
            layout,
            "running",
            experiment_id=identity["experiment_id"],
            details={"selection_split": "val", "test_used_for_selection": False},
        )
        run_record.update(
            {"execution_state": "running", "real_training_started": True}
        )
        atomic_write_json(run_path, run_record, overwrite=True)
        execution_supervision = (
            supervision
            if run_spec.mode == "final"
            else supervision.loc[supervision["split"].isin(("train", "val"))]
        ).copy(deep=True).reset_index(drop=True)
        context = Phase4ExecutionContext(
            experiment_id=identity["experiment_id"],
            run_mode=run_spec.mode,
            seed=run_spec.seed,
            lambda_ordinal=run_spec.lambda_ordinal,
            lambda_attention=run_spec.lambda_attention,
            configuration_sha256=config_sha256,
            project_commit=str(project_before["commit"]),
            manifest_bundle_sha256=readiness.manifest_bundle_sha256,
            phase3_input_fingerprint=readiness.phase3_input_fingerprint.to_dict(),
            supervision_sha256=readiness.supervision_sha256,
            ordered_supervision_sha256=readiness.ordered_supervision_sha256,
            input_lock_sha256=input_lock_sha256,
            cohort_view=readiness.cohort_view,
            expected_resolution=readiness.expected_resolution,
            sample_count=readiness.sample_count,
            split_counts=dict(readiness.split_counts),
            effective_config=effective_config,
            # Tuning receives no test identities, labels, paths, or targets. The
            # complete frame exists only in the mandatory readiness boundary.
            supervision=execution_supervision,
            rendered_root=rendered_root,
            layout=layout,
        )
        result = execute_phase4_single_run(context)
        history, best, last, predictions = _validate_execution(result, context)
        transition_phase4_status(
            layout,
            "finalizing",
            experiment_id=identity["experiment_id"],
            details={"best_epoch": result.best_epoch, "last_epoch": result.last_epoch},
        )
        history_path = atomic_write_json(
            layout.history_dir / "epoch_history.json", list(history)
        )
        best_sidecar = atomic_write_json(
            layout.checkpoints_dir / "best.metadata.json",
            _checkpoint_sidecar(best, Path(result.best_checkpoint_path)),
        )
        last_sidecar = atomic_write_json(
            layout.checkpoints_dir / "last.metadata.json",
            _checkpoint_sidecar(last, Path(result.last_checkpoint_path)),
        )
        artifacts.extend(
            [
                history_path,
                Path(result.best_checkpoint_path),
                Path(result.last_checkpoint_path),
                best_sidecar,
                last_sidecar,
            ]
        )
        prediction_paths: dict[str, Path] = {}
        metrics_paths: dict[str, Path] = {}
        for split in (("val",) if run_spec.mode == "tuning" else ("val", "test")):
            prediction_path = write_phase4_prediction_csv(
                predictions[split], layout.predictions_dir / f"{split}.csv"
            )
            reloaded = load_phase4_prediction_csv(prediction_path)
            classification = evaluate_phase4_predictions(reloaded)
            reported = dict(result.metrics[split])
            metrics_record = {
                **classification,
                "run_mode": run_spec.mode,
                "seed": run_spec.seed,
                "lambda_ordinal": run_spec.lambda_ordinal,
                "lambda_attention": run_spec.lambda_attention,
                "experiment_id": identity["experiment_id"],
                "evaluation_losses": {
                    field: reported[field]
                    for field in (
                        "loss",
                        "total_loss",
                        "diagnostic_loss",
                        "ordinal_loss",
                        "attention_loss",
                    )
                    if field in reported
                },
            }
            metrics_path = atomic_write_json(
                layout.metrics_dir / f"{split}.json", metrics_record
            )
            prediction_paths[split] = prediction_path
            metrics_paths[split] = metrics_path
            artifacts.extend([prediction_path, metrics_path])

        post_manifest = validate_frozen_manifest_bundle(manifest_root)
        if post_manifest.to_dict() != manifest.to_dict():
            raise Phase4RunError("Frozen Phase 0 manifest changed during training")
        post_readiness, _ = validate_phase4_readiness(
            phase4["data"],
            manifest=post_manifest,
            expected_resolution=int(phase4["dataset"]["image_size"]),
            project_root=root,
            synthetic_dry_run=False,
        )
        validate_phase4_input_lock(input_lock, post_readiness)
        if post_readiness.to_dict() != readiness.to_dict():
            raise Phase4RunError("Phase 4 readiness changed during training")
        readiness_after_path = atomic_write_json(
            layout.provenance_dir / "readiness_after.json", post_readiness.to_dict()
        )
        artifacts.append(readiness_after_path)
        project_after = _require_project_state(root)
        if project_after != project_before:
            raise Phase4RunError("Project Git state changed during Phase 4 training")
        run_record.update(
            {
                "finished_utc": utc_timestamp(),
                "execution_state": "completed",
                "project_git_after": project_after,
                "readiness_revalidated_after_training": True,
                "manifest_revalidated_after_training": True,
                "project_git_state_unchanged_after_training": True,
                "research_results_eligible": True,
                "test_results_eligible": run_spec.mode == "final",
                "best_epoch": result.best_epoch,
                "last_epoch": result.last_epoch,
                "best_validation_loss": best["best_validation_loss"],
                "class_weights_from_train_hard_labels": list(result.class_weights),
                "events": list(result.events),
                "checkpoint_files": {
                    "best": layout.relative(result.best_checkpoint_path),
                    "last": layout.relative(result.last_checkpoint_path),
                },
                "prediction_files": {
                    split: layout.relative(path) for split, path in prediction_paths.items()
                },
                "metrics_files": {
                    split: layout.relative(path) for split, path in metrics_paths.items()
                },
                "performance_claimed": False,
            }
        )
        atomic_write_json(run_path, run_record, overwrite=True)
        manifest_path = write_phase4_artifact_manifest(
            layout,
            artifacts,
            experiment_id=identity["experiment_id"],
            run_mode=run_spec.mode,
        )
        completion = write_phase4_completion_marker(
            layout,
            manifest_path,
            experiment_id=identity["experiment_id"],
            run_mode=run_spec.mode,
        )
        transition_phase4_status(
            layout,
            "completed",
            experiment_id=identity["experiment_id"],
            details={
                "completion_marker": completion.name,
                "test_used_for_selection": False,
                "test_evaluated": run_spec.mode == "final",
            },
        )
        return Phase4RunResult(
            experiment_id=identity["experiment_id"],
            run_mode=run_spec.mode,
            seed=run_spec.seed,
            lambda_ordinal=run_spec.lambda_ordinal,
            lambda_attention=run_spec.lambda_attention,
            state="completed",
            output_dir=layout.experiment_dir,
            status_path=layout.status_path,
            provenance_path=run_path,
            completion_marker=completion,
            artifact_manifest=manifest_path,
            prediction_paths=prediction_paths,
            metrics_paths=metrics_paths,
        )
    except BaseException as exc:
        if completion is not None:
            try:
                completion.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            transition_phase4_status(
                layout,
                "failed",
                experiment_id=identity["experiment_id"],
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
        except BaseException:
            pass
        if run_record:
            run_record.update(
                {
                    "finished_utc": utc_timestamp(),
                    "execution_state": "failed",
                    "research_results_eligible": False,
                    "test_results_eligible": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        ),
                    },
                }
            )
            try:
                atomic_write_json(run_path, run_record, overwrite=True)
            except BaseException:
                pass
        raise


def phase4_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one guarded Phase 4 ConsenDxFormer experiment (never a matrix)."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=("tuning", "final"), required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-ordinal", type=float, default=None)
    parser.add_argument("--lambda-attention", type=float, default=None)
    args = parser.parse_args(argv)
    result = run_phase4_experiment(
        config_path=args.config,
        mode=args.mode,
        seed=args.seed,
        lambda_ordinal=args.lambda_ordinal,
        lambda_attention=args.lambda_attention,
    )
    print(f"PHASE4_STATE={result.state}")
    print(f"RUN_MODE={result.run_mode}")
    print(f"EXPERIMENT_ID={result.experiment_id}")
    print(f"OUTPUT_DIR={result.output_dir}")
    print(f"COMPLETION_MARKER={result.completion_marker}")
    return 0


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_INPUT_LOCK",
    "PROJECT_ROOT",
    "Phase4RunError",
    "Phase4RunResult",
    "phase4_cli",
    "run_phase4_experiment",
]
