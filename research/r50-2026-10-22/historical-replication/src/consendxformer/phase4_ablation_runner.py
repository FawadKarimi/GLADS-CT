"""Fail-closed, explicitly authorized ablations in a separate artifact namespace."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .baseline_source import verify_baseline_source, snapshot_source_tree
from .manifest_guard import validate_frozen_manifest_bundle
from .phase3_artifacts import atomic_write_csv, atomic_write_json
from .phase4_artifacts import Phase4ExperimentLayout
from .phase4_config import load_phase4_config
from .phase4_input_lock import load_phase4_input_lock, validate_phase4_input_lock
from .phase4_readiness import validate_phase4_readiness, phase4_supervision_hashes
from .phase4_runner import _require_project_state, _require_tracked_phase4_inputs, _validate_supervision_against_readiness
from .phase4_evaluation_runner import load_phase4_final_prediction_source, snapshot_source_files, assert_source_snapshot_unchanged, _safe_source_member
from .phase4_ablation_contracts import (
    RUN_KIND, NAMESPACE, AblationRunSpec, effective_ablation_config, execution_protocol,
    make_identity, validate_identity, identity_rows, validate_predictions, prediction_metrics,
    load_checkpoint_pair, checkpoint_metadata, validate_loss_metrics, FROZEN_PHASE4_INPUT_LOCK_SHA256,
)
from .reproducibility import environment_audit
from .phase4_ablations import _values_are_exactly_equal as exactly_equal

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guard_new_namespace(path):
    path = Path(path).absolute()
    if path.resolve() != path or any((parent / "COMPLETED.json").exists() for parent in path.parents):
        raise ValueError("Ablation output cannot traverse a symlink or an existing completed bundle")
    return path


def guard_image_paths(frame):
    # The unchanged dataset reads image paths relative to the process cwd.
    # Reject an invocation that would read different bytes than our ROOT seal.
    if any(Path(value).resolve() != (ROOT / value).resolve() for value in frame["image_path"]):
        raise ValueError("Run from the repository root: relative image paths resolve elsewhere")


def frozen_bundle():
    config, lock = ROOT / "configs/phase4_consensdxformer.yaml", ROOT / "configs/phase4_input_lock.json"
    _require_tracked_phase4_inputs(config, lock, project_root=ROOT)
    bundle = load_phase4_config(config)
    input_lock = load_phase4_input_lock(lock)
    if sha256_json(input_lock.to_dict()) != FROZEN_PHASE4_INPUT_LOCK_SHA256:
        raise ValueError("Frozen ablation input lock changed")
    return bundle, input_lock


def revalidate_inputs(config, lock):
    manifest = validate_frozen_manifest_bundle(ROOT / config["data"]["frozen_split_dir"])
    readiness, supervision = validate_phase4_readiness(config["data"], manifest=manifest,
        expected_resolution=config["dataset"]["image_size"], project_root=ROOT, synthetic_dry_run=False)
    validate_phase4_input_lock(lock, readiness)
    if not readiness.ready_for_requested_mode or not readiness.real_research_ready:
        raise ValueError("Ablation requires real frozen readiness")
    _validate_supervision_against_readiness(supervision, readiness)
    return manifest, readiness, supervision


def reference_record(source):
    return {"source_dir": str(source.source_dir), "descriptor": dict(source.descriptor),
            "best_checkpoint_sha256": sha256_file(source.source_dir / "checkpoints/best.pt")}


def assert_supervision(frame, expected):
    hashes = phase4_supervision_hashes(frame)
    if hashes != (expected["supervision_sha256"], expected["ordered_supervision_sha256"]):
        raise ValueError("Ablation supervision differs from frozen source")


@dataclass(frozen=True)
class PreparedAblation:
    spec: AblationRunSpec
    config: dict
    identity: dict
    supervision: pd.DataFrame
    readiness: dict
    manifest: dict
    input_lock: Any
    project_state: dict
    baseline_state: dict
    baseline_snapshot: dict
    source_snapshot: dict
    output_dir: Path


def prepare_ablation(spec: AblationRunSpec, full_source: str | Path) -> PreparedAblation:
    """Read-only preflight. No model, dataset, loader, directory or checkpoint construction."""
    spec.specification()
    project = _require_project_state(ROOT)
    bundle, lock = frozen_bundle()
    config = effective_ablation_config(bundle, spec)
    # All real-input guards precede allocation of any training objects or layout.
    manifest, readiness, supervision = revalidate_inputs(config, lock)
    guard_image_paths(supervision)
    reference = load_phase4_final_prediction_source(full_source)
    if reference.descriptor["seed"] != spec.seed:
        raise ValueError("Full-model reference seed differs from ablation seed")
    report = readiness.to_dict()
    for field in ("manifest_bundle_sha256", "supervision_sha256", "ordered_supervision_sha256", "phase3_input_fingerprint"):
        if reference.descriptor[field] != report[field]:
            raise ValueError(f"Ablation/full-model inputs differ: {field}")
    assert_supervision(supervision, reference.descriptor)
    if identity_rows(supervision) != identity_rows(reference.supervision):
        raise ValueError("Full-model and live sample identities/labels differ")
    test = supervision.loc[supervision["split"] == "test"]
    if len(test) != 183 or test["patient_id"].nunique() != 99:
        raise ValueError("Frozen test cohort must contain 183 nodules from 99 patients")
    baseline = verify_baseline_source(ROOT / "external/LungDxFormer", ROOT / "locks/baseline_sources.yaml").to_dict()
    baseline_snapshot = snapshot_source_tree(ROOT / "external/LungDxFormer")
    paths = [ROOT / "configs/phase4_consensdxformer.yaml", ROOT / "configs/phase4_input_lock.json"]
    rendered = ROOT / config["data"]["rendered_dir"]
    for row in supervision.to_dict("records"):
        paths.extend([ROOT / row["image_path"], rendered / row["rendered_mask_path"]])
    snapshot = {**reference.source_snapshot, **snapshot_source_files(paths)}
    identity = make_identity(spec, config, str(project["commit"]), report, reference_record(reference))
    output = ROOT / NAMESPACE / "runs" / identity["experiment_id"]
    guard_new_namespace(output)
    if output.exists():
        raise FileExistsError(f"Ablation identity already exists; never overwrite or auto-resume: {output}")
    prepared = PreparedAblation(spec, config, identity, supervision, report, manifest.to_dict(), lock,
                               project, baseline, baseline_snapshot, snapshot, output)
    recheck_prepared(prepared)
    return prepared


def recheck_prepared(prepared):
    assert_source_snapshot_unchanged(prepared.source_snapshot)
    if _require_project_state(ROOT) != prepared.project_state:
        raise ValueError("Ablation project Git state changed")
    if snapshot_source_tree(ROOT / "external/LungDxFormer") != prepared.baseline_snapshot:
        raise ValueError("External baseline tree changed")
    if verify_baseline_source(ROOT / "external/LungDxFormer", ROOT / "locks/baseline_sources.yaml").to_dict() != prepared.baseline_state:
        raise ValueError("External baseline source state changed")


def verify_manifest(directory, *, completed=True):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "artifact_manifest.json")
    if manifest.get("schema") != "phase4_ablation_artifacts_v1" or manifest.get("run_kind") != RUN_KIND:
        raise ValueError("Not a separately authorized ablation source")
    files = manifest.get("files")
    if not isinstance(files, dict) or sha256_json(files) != manifest.get("bundle_sha256"):
        raise ValueError("Invalid ablation artifact manifest")
    snapshot = snapshot_source_files([directory / "artifact_manifest.json"])
    for relative, digest in files.items():
        canonical, member = _safe_source_member(directory, relative)
        if canonical != relative or sha256_file(member) != digest:
            raise ValueError("Ablation artifact checksum mismatch")
        snapshot[str(member)] = digest
    required = {"config/run_identity.json", "config/effective_config.json", "config/input_supervision.csv",
                "config/input_lock.json", "config/execution_protocol.json", "provenance/run.json",
                "provenance/readiness_before.json", "provenance/readiness_after.json", "provenance/phase0_manifest.json",
                "history/epoch_history.json", "checkpoints/best.pt", "checkpoints/last.pt",
                "checkpoints/best.metadata.json", "checkpoints/last.metadata.json",
                "predictions/val.csv", "predictions/test.csv", "metrics/val.json", "metrics/test.json"}
    if set(files) != required:
        raise ValueError("Ablation manifest lacks exact required artifacts")
    if completed:
        marker = read_json(directory / "COMPLETED.json")
        if marker != {"schema": "phase4_ablation_completion_v1", "run_kind": RUN_KIND,
                       "experiment_id": manifest["experiment_id"], "artifact_manifest_sha256": sha256_file(directory / "artifact_manifest.json"),
                       "runtime_training_explicitly_authorized": True}:
            raise ValueError("Ablation completion/authorization is invalid")
        snapshot.update(snapshot_source_files([directory / "COMPLETED.json"]))
    return manifest, snapshot


@dataclass(frozen=True)
class AblationSource:
    source_dir: Path
    identity: dict
    config: dict
    supervision: pd.DataFrame
    predictions: dict
    source_snapshot: dict
    best_record: dict


def _load_ablation_source(directory, *, completed):
    directory = Path(directory).resolve()
    manifest, snapshot = verify_manifest(directory, completed=completed)
    identity = read_json(directory / "config/run_identity.json")
    config = read_json(directory / "config/effective_config.json")
    bundle, lock = frozen_bundle()
    validate_identity(identity, config, bundle)
    if directory.name != identity["experiment_id"] or manifest["experiment_id"] != identity["experiment_id"]:
        raise ValueError("Ablation directory/manifest identity differs")
    before = read_json(directory / "provenance/readiness_before.json")
    after = read_json(directory / "provenance/readiness_after.json")
    if before != after or read_json(directory / "config/input_lock.json") != lock.to_dict():
        raise ValueError("Ablation readiness or frozen input lock changed")
    validate_phase4_input_lock(lock, before)
    for field, value in lock.to_dict().items():
        if field in identity["inputs"] and not exactly_equal(identity["inputs"][field], value):
            raise ValueError("Ablation identity differs from exact frozen input fields")
    if any(before.get(field) != value for field, value in identity["inputs"].items()):
        raise ValueError("Ablation readiness identity mismatch")
    if read_json(directory / "config/execution_protocol.json") != execution_protocol():
        raise ValueError("Ablation protocol changed")
    reference = load_phase4_final_prediction_source(identity["full_model_reference"]["source_dir"])
    if reference_record(reference) != identity["full_model_reference"] or reference.descriptor["seed"] != identity["seed"]:
        raise ValueError("Frozen complete-model reference changed")
    for field in ("manifest_bundle_sha256", "supervision_sha256", "ordered_supervision_sha256", "phase3_input_fingerprint"):
        if reference.descriptor[field] != identity["inputs"][field]:
            raise ValueError("Ablation source inputs differ from full model")
    supervision = pd.read_csv(directory / "config/input_supervision.csv", dtype=str, keep_default_na=False)
    assert_supervision(supervision, identity["inputs"])
    history = read_json(directory / "history/epoch_history.json")
    for epoch in history:
        for split in ("train", "validation"):
            validate_loss_metrics(epoch[split], identity)
    best, last = load_checkpoint_pair(directory / "checkpoints", identity, history)
    for name, checkpoint in (("best", best), ("last", last)):
        if not exactly_equal(read_json(directory / f"checkpoints/{name}.metadata.json"), checkpoint_metadata(checkpoint, directory / f"checkpoints/{name}.pt")):
            raise ValueError("Ablation checkpoint metadata differs from persisted snapshot")
    run = read_json(directory / "provenance/run.json")
    if (run.get("runtime_training_explicitly_authorized") is not True or run.get("fresh_initialization") is not True
        or run.get("completed") is not True or run.get("identity") != identity
        or run.get("events") != ["fresh_model_initialized", "best_selected_from_validation", "persisted_best_last_validated",
                                 "best_weights_reloaded", "validation_predictions_exported", "test_loader_created_after_best_reload"]):
        raise ValueError("Ablation execution authorization/events invalid")
    rng = run.get("seed_controls", {})
    expected_rng = {"seed": identity["seed"], "pythonhashseed": str(identity["seed"]), "python_random_seeded": True,
                    "numpy_seeded": True, "torch_seeded": True, "torch_cpu_seeded": True,
                    "deterministic_algorithms_requested": config["determinism"]["deterministic_algorithms"],
                    "deterministic_algorithms_warn_only_requested": config["determinism"]["warn_only"],
                    "cudnn_benchmark_requested": config["determinism"]["cudnn_benchmark"]}
    if any(not exactly_equal(rng.get(field), value) for field, value in expected_rng.items()):
        raise ValueError("Ablation RNG provenance differs from frozen policy")
    project = run.get("project_state_before_after", {})
    if project.get("commit") != identity["ablation_code_commit"] or project.get("working_tree_clean") is not True or project.get("branch") != "phase4-consendxformer":
        raise ValueError("Ablation code provenance differs")
    if not isinstance(run.get("environment"), dict) or not run["environment"] or run.get("source_snapshot_sha256") != sha256_json(run.get("source_snapshot")):
        raise ValueError("Ablation environment/input provenance missing or inconsistent")
    from .phase4_training import compute_training_class_weights
    if not np.array_equal(run["class_weights"], compute_training_class_weights(supervision)):
        raise ValueError("Ablation class weights are not training-only hard-label weights")
    predictions = {}
    for split in ("val", "test"):
        frame = pd.read_csv(directory / f"predictions/{split}.csv", float_precision="round_trip")
        predictions[split] = validate_predictions(frame, supervision.loc[supervision["split"] == split], identity)
        reported = read_json(directory / f"metrics/{split}.json")
        validate_loss_metrics(reported["within_variant_evaluation_losses"], identity)
        if reported["prediction_metrics"] != prediction_metrics(frame):
            raise ValueError("Ablation metrics differ from exact saved predictions")
    assert_source_snapshot_unchanged(snapshot)
    assert_source_snapshot_unchanged(reference.source_snapshot)
    return AblationSource(directory, identity, config, supervision, predictions, snapshot, best)


def load_ablation_source(directory) -> AblationSource:
    """Only the distinct completed-ablation contract is accepted; never a final source."""
    return _load_ablation_source(directory, completed=True)


def run_ablation(spec, full_source, *, authorize_training=False, device="cpu", gpu_assignment=None):
    if authorize_training is not True:
        raise ValueError("Separate explicit runtime training authorization is required")
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda device required")
    if device == "cuda" and (not gpu_assignment or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_assignment):
        raise ValueError("CUDA requires an explicit isolated GPU assignment before process startup")
    prepared = prepare_ablation(spec, full_source)
    from .phase4_ablation_execution import execute_ablation
    protected = [ROOT / path for path in ("data", "external", "artifacts/phase0", "artifacts/phase2", "artifacts/phase3",
                                           "artifacts/phase4/experiments", "artifacts/phase4/evaluation")]
    protected.append(Path(prepared.identity["full_model_reference"]["source_dir"]))
    guard_new_namespace(prepared.output_dir)
    layout = Phase4ExperimentLayout.create(prepared.output_dir.parent, prepared.identity["experiment_id"], protected_roots=protected)
    artifacts = []
    marker = layout.experiment_dir / "COMPLETED.json"
    try:
        for name, record in (("config/run_identity.json", prepared.identity), ("config/effective_config.json", prepared.config),
                             ("config/input_lock.json", prepared.input_lock.to_dict()), ("config/execution_protocol.json", execution_protocol()),
                             ("provenance/readiness_before.json", prepared.readiness), ("provenance/phase0_manifest.json", prepared.manifest)):
            artifacts.append(atomic_write_json(layout.experiment_dir / name, record))
        artifacts.append(atomic_write_csv(layout.config_dir / "input_supervision.csv", prepared.supervision))
        recheck_prepared(prepared)
        result = execute_ablation(prepared, layout, device=device)
        # Reload again for runner provenance, never trusting returned tensor references.
        best, last = load_checkpoint_pair(layout.checkpoints_dir, prepared.identity, result["history"])
        artifacts.append(atomic_write_json(layout.history_dir / "epoch_history.json", result["history"]))
        for kind, record in (("best", best), ("last", last)):
            path = layout.checkpoints_dir / f"{kind}.pt"
            artifacts.extend([path, atomic_write_json(layout.checkpoints_dir / f"{kind}.metadata.json", checkpoint_metadata(record, path))])
        for split in ("val", "test"):
            frame = validate_predictions(result["predictions"][split], prepared.supervision.loc[prepared.supervision["split"] == split], prepared.identity)
            path = atomic_write_csv(layout.predictions_dir / f"{split}.csv", frame)
            frame = validate_predictions(pd.read_csv(path, float_precision="round_trip"), prepared.supervision.loc[prepared.supervision["split"] == split], prepared.identity)
            artifacts.extend([path, atomic_write_json(layout.metrics_dir / f"{split}.json", {
                "prediction_metrics": prediction_metrics(frame), "within_variant_evaluation_losses": result["metrics"][split]})])
        manifest_after, readiness_after, live = revalidate_inputs(prepared.config, prepared.input_lock)
        if readiness_after.to_dict() != prepared.readiness or manifest_after.to_dict() != prepared.manifest:
            raise ValueError("Readiness/manifest changed during ablation")
        assert_supervision(live, prepared.identity["inputs"])
        recheck_prepared(prepared)
        artifacts.append(atomic_write_json(layout.provenance_dir / "readiness_after.json", readiness_after.to_dict()))
        artifacts.append(atomic_write_json(layout.provenance_dir / "run.json", {
            "identity": prepared.identity, "completed": True, "runtime_training_explicitly_authorized": True,
            "fresh_initialization": True, "events": result["events"], "class_weights": result["class_weights"],
            "seed_controls": result["seed_controls"], "environment": environment_audit(require_torch=True),
            "device": device, "gpu_assignment": gpu_assignment, "project_state_before_after": prepared.project_state,
            "baseline_state": prepared.baseline_state, "source_snapshot": prepared.source_snapshot,
            "source_snapshot_sha256": sha256_json(prepared.source_snapshot), "performance_claimed": False}))
        files = {layout.relative(path): sha256_file(path) for path in sorted(artifacts)}
        manifest_path = atomic_write_json(layout.experiment_dir / "artifact_manifest.json", {
            "schema": "phase4_ablation_artifacts_v1", "run_kind": RUN_KIND, "experiment_id": prepared.identity["experiment_id"],
            "files": files, "bundle_sha256": sha256_json(files)})
        _load_ablation_source(layout.experiment_dir, completed=False)
        recheck_prepared(prepared)
        atomic_write_json(marker, {"schema": "phase4_ablation_completion_v1", "run_kind": RUN_KIND,
            "experiment_id": prepared.identity["experiment_id"], "artifact_manifest_sha256": sha256_file(manifest_path),
            "runtime_training_explicitly_authorized": True})
        recheck_prepared(prepared)
        return layout.experiment_dir
    except BaseException as error:
        if marker.exists():
            marker.unlink()  # Only this newly created run's marker, never an existing source.
        atomic_write_json(layout.experiment_dir / "FAILED.json", {"error_type": type(error).__name__, "message": str(error),
                                                                 "completion_valid": False})
        raise
