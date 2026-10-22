"""Frozen-checkpoint-only inference for Phase 4 spatial and boundary analyses.

This module contains no optimizer, backward pass, checkpoint writer, or training
entry point.  It loads one manifest-verified final ``best.pt`` and keeps the
source experiment immutable throughout inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .hashing import sha256_file, sha256_json
from .phase4_evaluation_metrics import (
    ATTENTION_ALIGNMENT_DEFINITION,
    attention_alignment_metrics,
)
from .phase4_evaluation_runner import (
    FINAL_SEEDS,
    PROJECT_ROOT,
    SOURCE_FINAL_MODEL_COMMIT,
    Phase4EvaluationError,
    SavedPredictionSource,
    assert_source_snapshot_unchanged,
    load_phase4_final_prediction_source,
    snapshot_source_files,
)
from .manifest_guard import ManifestIntegrityError, validate_frozen_manifest_bundle
from .phase4_readiness import Phase4ReadinessError, validate_phase4_readiness


class Phase4PosthocError(RuntimeError):
    """Raised when post-hoc inference is not provably frozen and source-safe."""


@dataclass(frozen=True)
class FrozenFinalModel:
    """A final model loaded exclusively from a persisted, validated best file."""

    model: Any
    source: SavedPredictionSource
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_record: Mapping[str, Any]
    device: str


@dataclass(frozen=True)
class FrozenAttentionLoader:
    """An exact test loader sealed to revalidated Phase 2 input bytes."""

    loader: Iterable[Mapping[str, Any]]
    input_snapshot: Mapping[str, str]
    ordered_test_cohort_sha256: str

    def __iter__(self):
        return iter(self.loader)


def _read_history(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4PosthocError(f"Cannot read persisted epoch history: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise Phase4PosthocError("Persisted epoch history must be a list of records")
    return [dict(row) for row in value]


def load_frozen_final_best_model(
    experiment_dir: str | Path,
    *,
    device: str = "cpu",
    model_factory: Any | None = None,
) -> FrozenFinalModel:
    """Load and freeze only the persisted final ``best.pt`` for inference.

    ``last.pt`` is covered by the completed source manifest but is never loaded
    into the model.  The best checkpoint is reconciled to the persisted epoch
    history and exact final-run provenance before model construction.
    """

    if device not in {"cpu", "cuda"}:
        raise Phase4PosthocError("device must be 'cpu' or 'cuda'")
    try:
        import torch
        from .phase4_checkpoints import (
            Phase4CheckpointContext,
            select_phase4_best_epoch,
            validate_phase4_checkpoint,
        )
    except (ImportError, OSError) as exc:
        raise Phase4PosthocError("PyTorch Phase 4 runtime is unavailable") from exc
    if model_factory is None:
        try:
            from .phase4_execution import _model as locked_model_factory
        except (ImportError, OSError) as exc:
            raise Phase4PosthocError("Phase 4 model runtime is unavailable") from exc
    else:
        locked_model_factory = model_factory

    source = load_phase4_final_prediction_source(experiment_dir)
    descriptor = dict(source.descriptor)
    if descriptor.get("project_commit") != SOURCE_FINAL_MODEL_COMMIT:
        raise Phase4PosthocError("Final model source commit is not the frozen commit")
    if descriptor.get("seed") not in FINAL_SEEDS:
        raise Phase4PosthocError("Final model seed is not one of 42, 43, and 44")
    assert_source_snapshot_unchanged(source.source_snapshot)
    identity_path = source.source_dir / "config" / "run_identity.json"
    effective_path = source.source_dir / "config" / "effective_config.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4PosthocError("Cannot read final model configuration") from exc
    if not isinstance(identity, dict) or not isinstance(effective, dict):
        raise Phase4PosthocError("Final model identity/configuration must be JSON objects")
    context = Phase4CheckpointContext(
        experiment_id=identity["experiment_id"],
        run_mode=identity["run_mode"],
        seed=identity["seed"],
        lambda_ordinal=identity["lambda_ordinal"],
        lambda_attention=identity["lambda_attention"],
        project_commit=identity["project_commit"],
        config_sha256=identity["configuration_sha256"],
        manifest_bundle_sha256=identity["manifest_bundle_sha256"],
        phase3_input_fingerprint=identity["phase3_input_fingerprint"],
        supervision_sha256=identity["supervision_sha256"],
        ordered_supervision_sha256=identity["ordered_supervision_sha256"],
        input_lock_sha256=identity["input_lock_sha256"],
        cohort_view=identity["cohort_view"],
        expected_resolution=identity["expected_resolution"],
        sample_count=identity["sample_count"],
        split_counts=identity["split_counts"],
    )
    best_path = (source.source_dir / "checkpoints" / "best.pt").resolve()
    try:
        raw = torch.load(best_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise Phase4PosthocError(f"Cannot load persisted final best checkpoint: {best_path}") from exc
    try:
        checkpoint = validate_phase4_checkpoint(
            raw,
            expected_kind="best",
            context=context,
        )
        selection = select_phase4_best_epoch(
            _read_history(source.source_dir / "history" / "epoch_history.json")
        )
    except (TypeError, ValueError) as exc:
        raise Phase4PosthocError("Final best checkpoint fails history/provenance validation") from exc
    if checkpoint["epoch"] != selection["epoch"] or checkpoint[
        "best_validation_loss"
    ] != selection["value"]:
        raise Phase4PosthocError(
            "Final best checkpoint does not equal the validation-selected epoch"
        )
    factory = locked_model_factory
    if not callable(factory):
        raise TypeError("model_factory must be callable")
    model = factory(effective)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Phase4PosthocError("Loaded final model could not be frozen for inference")
    assert_source_snapshot_unchanged(source.source_snapshot)
    return FrozenFinalModel(
        model=model,
        source=source,
        checkpoint_path=best_path,
        checkpoint_sha256=sha256_file(best_path),
        checkpoint_record=checkpoint,
        device=device,
    )


def _batch_text(batch: Mapping[str, Any], field: str, count: int) -> list[str]:
    if field not in batch:
        raise Phase4PosthocError(f"Inference batch lacks identifier {field!r}")
    raw = batch[field]
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError as exc:
            raise Phase4PosthocError(f"Inference identifier {field!r} is not iterable") from exc
    text_values = [str(value) for value in values]
    normalized = [value.strip() for value in text_values]
    if (
        len(normalized) != count
        or any(not value for value in normalized)
        or normalized != text_values
    ):
        raise Phase4PosthocError(
            f"Inference identifier {field!r} is empty, noncanonical, or does not "
            "match batch membership"
        )
    return normalized


_ATTENTION_SOURCE_TO_BATCH_FIELDS = {
    "patient_id": "patient_id",
    "nodule_id": "nodule_id",
    "nodule_sha256": "nodule_sha256",
    "series_instance_uid": "series_instance_uid",
    "split": "split",
    "image_path": "path",
    "rendered_mask_path": "rendered_mask_path",
}
_ATTENTION_IDENTITY_FIELDS = tuple(_ATTENTION_SOURCE_TO_BATCH_FIELDS.values())


def _sha256_text(value: Any, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise Phase4PosthocError(f"{role} must be a lowercase SHA-256 digest")
    return text


def _validated_attention_source_cohort(
    frozen: FrozenFinalModel,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Rebuild the exact frozen test cohort accepted by the attention loader."""

    source = frozen.source
    descriptor = dict(source.descriptor)
    if source.model != "ConsenDxFormer":
        raise Phase4PosthocError("Attention alignment requires a ConsenDxFormer source")
    expected_count = descriptor.get("test_sample_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise Phase4PosthocError("Attention source test_sample_count is invalid")
    for field in (
        "manifest_bundle_sha256",
        "supervision_sha256",
        "ordered_supervision_sha256",
        "input_lock_sha256",
        "artifact_manifest_sha256",
        "artifact_bundle_sha256",
    ):
        _sha256_text(descriptor.get(field), f"Attention source {field}")
    fingerprint = descriptor.get("phase3_input_fingerprint")
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        raise Phase4PosthocError("Attention source Phase 3 input fingerprint is missing")

    supervision = source.supervision
    if not isinstance(supervision, pd.DataFrame) or supervision.empty:
        raise Phase4PosthocError("Attention source lacks frozen supervision")
    required = set(_ATTENTION_SOURCE_TO_BATCH_FIELDS)
    missing = sorted(required - set(supervision.columns))
    if missing:
        raise Phase4PosthocError(
            f"Attention source supervision lacks cohort fields: {missing}"
        )
    cohort = supervision.loc[
        supervision["split"] == "test", list(_ATTENTION_SOURCE_TO_BATCH_FIELDS)
    ].reset_index(drop=True)
    if len(cohort) != expected_count:
        raise Phase4PosthocError(
            "Attention source test cohort count differs from final-run provenance"
        )
    if cohort.isna().any().any():
        raise Phase4PosthocError("Attention source test cohort contains missing values")
    for field in cohort.columns:
        values = cohort[field].astype(str)
        if bool((values == "").any()) or not values.equals(values.str.strip()):
            raise Phase4PosthocError(
                f"Attention source test cohort contains noncanonical {field}"
            )
        cohort[field] = values
    identity = ["patient_id", "nodule_id", "series_instance_uid", "split"]
    if cohort.duplicated(subset=identity).any() or cohort["nodule_sha256"].duplicated().any():
        raise Phase4PosthocError("Attention source test cohort contains duplicate samples")

    predictions = source.predictions
    prediction_fields = ["patient_id", "nodule_id", "series_instance_uid", "split"]
    if not isinstance(predictions, pd.DataFrame) or any(
        field not in predictions.columns for field in prediction_fields
    ):
        raise Phase4PosthocError("Attention source lacks validated test predictions")
    # Saved CSVs use the prediction writer's canonical sort, whereas inference
    # must preserve the readiness-approved supervision order. Reconcile exact
    # membership here, then enforce supervision order on every inferred row.
    expected_prediction_identity = cohort.loc[:, prediction_fields].sort_values(prediction_fields)
    actual_prediction_identity = predictions.loc[:, prediction_fields].sort_values(prediction_fields)
    if actual_prediction_identity.to_dict("records") != expected_prediction_identity.to_dict("records"):
        raise Phase4PosthocError(
            "Attention source predictions and frozen supervision have different test rows"
        )

    records = [
        {
            batch_field: str(row[source_field])
            for source_field, batch_field in _ATTENTION_SOURCE_TO_BATCH_FIELDS.items()
        }
        for row in cohort.to_dict(orient="records")
    ]
    cohort_record = {
        "sample_count": len(records),
        "identity_fields": list(_ATTENTION_IDENTITY_FIELDS),
        "ordered_test_cohort_sha256": sha256_json(records),
        "manifest_bundle_sha256": descriptor["manifest_bundle_sha256"],
        "supervision_sha256": descriptor["supervision_sha256"],
        "ordered_supervision_sha256": descriptor["ordered_supervision_sha256"],
        "input_lock_sha256": descriptor["input_lock_sha256"],
        "phase3_input_fingerprint": dict(fingerprint),
        "source_artifact_manifest_sha256": descriptor["artifact_manifest_sha256"],
        "source_artifact_bundle_sha256": descriptor["artifact_bundle_sha256"],
    }
    return records, cohort_record


def _assert_frozen(value: FrozenFinalModel) -> None:
    if not isinstance(value, FrozenFinalModel):
        raise TypeError("frozen must be a FrozenFinalModel")
    if value.model.training or any(
        parameter.requires_grad for parameter in value.model.parameters()
    ):
        raise Phase4PosthocError("Post-hoc inference requires an eval-mode frozen model")
    if value.checkpoint_path.name != "best.pt":
        raise Phase4PosthocError("Post-hoc inference may use only best.pt")
    if sha256_file(value.checkpoint_path) != value.checkpoint_sha256:
        raise Phase4PosthocError("Persisted final best checkpoint changed after loading")
    assert_source_snapshot_unchanged(value.source.source_snapshot)


def evaluate_attention_alignment_from_loader(
    frozen: FrozenFinalModel,
    loader: FrozenAttentionLoader,
) -> dict[str, Any]:
    """Run no-grad DSA/soft-consensus spatial-correspondence evaluation."""

    _assert_frozen(frozen)
    if not isinstance(loader, FrozenAttentionLoader):
        raise TypeError("loader must be produced by build_frozen_attention_loader")
    expected_cohort, cohort_provenance = _validated_attention_source_cohort(frozen)
    if (
        loader.ordered_test_cohort_sha256
        != cohort_provenance["ordered_test_cohort_sha256"]
    ):
        raise Phase4PosthocError("Attention loader cohort seal is inconsistent")
    assert_source_snapshot_unchanged(loader.input_snapshot)
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise Phase4PosthocError("PyTorch Phase 4 runtime is unavailable") from exc
    attention_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    cohort_rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in loader:
            if "image" not in batch or "soft_consensus_mask" not in batch:
                raise Phase4PosthocError(
                    "Attention inference requires image and soft_consensus_mask"
                )
            images = batch["image"].to(frozen.device)
            masks = batch["soft_consensus_mask"]
            count = int(images.shape[0])
            outputs = frozen.model(images)
            if not isinstance(outputs, Mapping) or "attention_map" not in outputs:
                raise Phase4PosthocError("Frozen model output lacks DSA attention_map")
            attention = outputs["attention_map"].detach().cpu().numpy()
            consensus = masks.detach().cpu().numpy()
            if attention.shape[0] != count or consensus.shape[0] != count:
                raise Phase4PosthocError("Attention/mask batch membership is inconsistent")
            attention_rows.extend(attention[:, 0, :, :])
            mask_rows.extend(consensus[:, 0, :, :])
            fields = {
                name: _batch_text(batch, name, count)
                for name in _ATTENTION_IDENTITY_FIELDS
            }
            if any(value != "test" for value in fields["split"]):
                raise Phase4PosthocError("Attention alignment final evaluation is test-only")
            cohort_rows.extend(
                {
                    name: fields[name][position]
                    for name in _ATTENTION_IDENTITY_FIELDS
                }
                for position in range(count)
            )
    if not attention_rows:
        raise Phase4PosthocError("Attention evaluation loader yielded no samples")
    if cohort_rows != expected_cohort:
        raise Phase4PosthocError(
            "Attention inference rows do not exactly match the frozen test cohort order"
        )
    metrics = attention_alignment_metrics(
        np.stack(attention_rows),
        np.stack(mask_rows),
        identifiers=[
            {
                name: record[name]
                for name in ("patient_id", "nodule_id", "series_instance_uid")
            }
            for record in cohort_rows
        ],
    )
    for record, cohort_record in zip(metrics["per_sample"], cohort_rows):
        record.update(
            {
                name: cohort_record[name]
                for name in (
                    "nodule_sha256",
                    "split",
                    "path",
                    "rendered_mask_path",
                )
            }
        )
    assert_source_snapshot_unchanged(loader.input_snapshot)
    assert_source_snapshot_unchanged(frozen.source.source_snapshot)
    return {
        "schema_version": 1,
        "analysis": "phase4_attention_alignment_spatial_correspondence",
        "model": "ConsenDxFormer",
        "seed": frozen.source.descriptor["seed"],
        "experiment_id": frozen.source.descriptor["experiment_id"],
        "checkpoint_kind": "best",
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "causal_explanation_claimed": False,
        "training_performed": False,
        "source_cohort": {
            **cohort_provenance,
            "input_file_snapshot_sha256": sha256_json(dict(loader.input_snapshot)),
            "phase2_inputs_revalidated_before_inference": True,
            "input_files_unchanged_after_inference": True,
        },
        "metrics": metrics,
        "definition": ATTENTION_ALIGNMENT_DEFINITION,
    }


def build_frozen_attention_loader(frozen: FrozenFinalModel) -> Any:
    """Build the locked, unaugmented test loader from source supervision."""

    _assert_frozen(frozen)
    expected_cohort, cohort_provenance = _validated_attention_source_cohort(frozen)
    try:
        from torch.utils.data import DataLoader
        from .phase4_dataset import ConsenDxDataset
    except (ImportError, OSError) as exc:
        raise Phase4PosthocError("PyTorch Phase 4 dataset runtime is unavailable") from exc
    effective_path = frozen.source.source_dir / "config" / "effective_config.json"
    try:
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4PosthocError("Cannot read final model effective configuration") from exc
    if not isinstance(effective, Mapping):
        raise Phase4PosthocError("Final model effective configuration must be an object")
    dataset_config = effective.get("dataset")
    data_config = effective.get("data")
    augmentation = effective.get("augmentation")
    if not all(
        isinstance(value, Mapping)
        for value in (dataset_config, data_config, augmentation)
    ):
        raise Phase4PosthocError("Final model data configuration is incomplete")
    if augmentation.get("eval") != {"enabled": False}:
        raise Phase4PosthocError("Attention evaluation preprocessing must be unaugmented")
    rendered_reference = data_config.get("rendered_dir")
    manifest_reference = data_config.get("frozen_split_dir")
    if not isinstance(rendered_reference, str) or not rendered_reference.strip():
        raise Phase4PosthocError("Final model rendered-data root is missing")
    if not isinstance(manifest_reference, str) or not manifest_reference.strip():
        raise Phase4PosthocError("Final model frozen-manifest directory is missing")
    rendered_root = Path(rendered_reference)
    if not rendered_root.is_absolute():
        rendered_root = PROJECT_ROOT / rendered_root
    rendered_root = rendered_root.resolve()
    manifest_root = Path(manifest_reference)
    if not manifest_root.is_absolute():
        manifest_root = PROJECT_ROOT / manifest_root
    try:
        manifest = validate_frozen_manifest_bundle(manifest_root.resolve())
        readiness, live_supervision = validate_phase4_readiness(
            data_config,
            manifest=manifest,
            expected_resolution=int(dataset_config["image_size"]),
            project_root=PROJECT_ROOT,
            synthetic_dry_run=False,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        ManifestIntegrityError,
        Phase4ReadinessError,
    ) as exc:
        raise Phase4PosthocError(
            "Attention inputs fail current Phase 0/2/3/4 readiness validation"
        ) from exc
    descriptor = dict(frozen.source.descriptor)
    readiness_expectations = {
        "cohort_view": readiness.cohort_view,
        "sample_count": readiness.sample_count,
        "manifest_bundle_sha256": readiness.manifest_bundle_sha256,
        "supervision_sha256": readiness.supervision_sha256,
        "ordered_supervision_sha256": readiness.ordered_supervision_sha256,
        "phase3_input_fingerprint": readiness.phase3_input_fingerprint.to_dict(),
    }
    for field, actual in readiness_expectations.items():
        expected = descriptor.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise Phase4PosthocError(
                f"Revalidated attention input {field} differs from the final source"
            )
    supervision = frozen.source.supervision
    assert isinstance(supervision, pd.DataFrame)
    test_frame = supervision.loc[supervision["split"] == "test"].reset_index(drop=True)
    live_test = live_supervision.loc[live_supervision["split"] == "test"].reset_index(
        drop=True
    )
    source_fields = list(_ATTENTION_SOURCE_TO_BATCH_FIELDS)
    if any(field not in live_test for field in source_fields):
        raise Phase4PosthocError("Revalidated attention cohort lacks source fields")
    persisted_identity = test_frame.loc[:, source_fields].astype(str)
    live_identity = live_test.loc[:, source_fields].astype(str)
    if not persisted_identity.equals(live_identity):
        raise Phase4PosthocError(
            "Revalidated Phase 2 inputs differ from persisted source supervision"
        )
    try:
        dataset = ConsenDxDataset(
            test_frame,
            image_size=int(dataset_config["image_size"]),
            in_channels=int(dataset_config["in_channels"]),
            rendered_root=rendered_root,
            augmenter=None,
        )
        raw_loader = DataLoader(
            dataset,
            batch_size=int(dataset_config["batch_size"]),
            num_workers=int(dataset_config["num_workers"]),
            shuffle=False,
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise Phase4PosthocError("Cannot construct the frozen attention test loader") from exc
    input_paths: list[Path] = []
    for record in expected_cohort:
        image_path = Path(record["path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        mask_path = Path(record["rendered_mask_path"])
        if not mask_path.is_absolute():
            mask_path = rendered_root / mask_path
        input_paths.extend((image_path.resolve(), mask_path.resolve()))
    try:
        input_snapshot = snapshot_source_files(input_paths)
    except (OSError, Phase4EvaluationError) as exc:
        raise Phase4PosthocError("Cannot seal revalidated attention input files") from exc
    return FrozenAttentionLoader(
        loader=raw_loader,
        input_snapshot=input_snapshot,
        ordered_test_cohort_sha256=cohort_provenance["ordered_test_cohort_sha256"],
    )


def evaluate_attention_alignment_from_source(
    experiment_dir: str | Path,
    *,
    device: str = "cpu",
    model_factory: Any | None = None,
) -> dict[str, Any]:
    """Load persisted best.pt and run the exact frozen attention test cohort."""

    frozen = load_frozen_final_best_model(
        experiment_dir,
        device=device,
        model_factory=model_factory,
    )
    return evaluate_attention_alignment_from_loader(
        frozen,
        build_frozen_attention_loader(frozen),
    )


def aggregate_attention_seed_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-seed mean alignment metrics over exactly seeds 42/43/44."""

    if isinstance(results, (str, bytes)) or not all(
        isinstance(result, Mapping) for result in results
    ):
        raise TypeError("results must be a sequence of attention result mappings")
    by_seed = {result.get("seed"): result for result in results}
    if set(by_seed) != set(FINAL_SEEDS) or len(results) != len(FINAL_SEEDS):
        raise Phase4PosthocError("Attention aggregation requires exactly seeds 42, 43, 44")
    experiment_ids: list[str] = []
    checkpoint_digests: list[str] = []
    reference_rows: list[tuple[str, ...]] | None = None
    reference_cohort_sha256: str | None = None
    reference_supervision: tuple[str, ...] | None = None
    for seed in FINAL_SEEDS:
        result = by_seed[seed]
        checks = {
            "schema_version": 1,
            "analysis": "phase4_attention_alignment_spatial_correspondence",
            "model": "ConsenDxFormer",
            "checkpoint_kind": "best",
            "causal_explanation_claimed": False,
            "training_performed": False,
        }
        for field, expected in checks.items():
            if result.get(field) != expected:
                raise Phase4PosthocError(
                    f"Attention seed {seed} has invalid {field!r} provenance"
                )
        experiment_id = str(result.get("experiment_id", ""))
        if not experiment_id or experiment_id != experiment_id.strip():
            raise Phase4PosthocError("Attention source experiment_id is invalid")
        experiment_ids.append(experiment_id)
        checkpoint_digests.append(
            _sha256_text(result.get("checkpoint_sha256"), "Attention best checkpoint")
        )
        source_cohort = result.get("source_cohort")
        if not isinstance(source_cohort, Mapping):
            raise Phase4PosthocError("Attention seed result lacks source cohort provenance")
        for field in (
            "source_artifact_manifest_sha256",
            "source_artifact_bundle_sha256",
        ):
            _sha256_text(source_cohort.get(field), f"Attention {field}")
        cohort_sha256 = _sha256_text(
            source_cohort.get("ordered_test_cohort_sha256"),
            "Attention ordered test cohort",
        )
        supervision_binding = tuple(
            _sha256_text(source_cohort.get(field), f"Attention source {field}")
            for field in (
                "manifest_bundle_sha256",
                "supervision_sha256",
                "ordered_supervision_sha256",
                "input_lock_sha256",
                "input_file_snapshot_sha256",
            )
        )
        fingerprint = source_cohort.get("phase3_input_fingerprint")
        if not isinstance(fingerprint, Mapping) or not fingerprint:
            raise Phase4PosthocError("Attention source fingerprint is missing")
        supervision_binding = (*supervision_binding, sha256_json(dict(fingerprint)))
        if source_cohort.get("identity_fields") != list(_ATTENTION_IDENTITY_FIELDS):
            raise Phase4PosthocError("Attention source identity fields are invalid")
        for field in (
            "phase2_inputs_revalidated_before_inference",
            "input_files_unchanged_after_inference",
        ):
            if source_cohort.get(field) is not True:
                raise Phase4PosthocError(
                    f"Attention seed result lacks {field} evidence"
                )
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise Phase4PosthocError("Attention seed result lacks metrics")
        per_sample = metrics.get("per_sample")
        if not isinstance(per_sample, list) or not per_sample:
            raise Phase4PosthocError("Attention seed result lacks per-sample rows")
        if source_cohort.get("sample_count") != len(per_sample):
            raise Phase4PosthocError("Attention result count differs from source cohort")
        row_keys: list[tuple[str, ...]] = []
        cohort_records: list[dict[str, str]] = []
        for position, row in enumerate(per_sample):
            if not isinstance(row, Mapping):
                raise Phase4PosthocError(
                    f"Attention per-sample row {position} is not a mapping"
                )
            try:
                key = tuple(str(row[field]) for field in _ATTENTION_IDENTITY_FIELDS)
            except KeyError as exc:
                raise Phase4PosthocError(
                    f"Attention per-sample row {position} lacks cohort identity"
                ) from exc
            if any(not value or value != value.strip() for value in key):
                raise Phase4PosthocError(
                    f"Attention per-sample row {position} has noncanonical identity"
                )
            row_keys.append(key)
            cohort_records.append(
                {
                    field: value
                    for field, value in zip(_ATTENTION_IDENTITY_FIELDS, key)
                }
            )
        if len(set(row_keys)) != len(row_keys):
            raise Phase4PosthocError("Attention seed result contains duplicate samples")
        if sha256_json(cohort_records) != cohort_sha256:
            raise Phase4PosthocError(
                "Attention per-sample identities differ from the recorded cohort hash"
            )
        if reference_rows is None:
            reference_rows = row_keys
            reference_cohort_sha256 = cohort_sha256
            reference_supervision = supervision_binding
        elif (
            row_keys != reference_rows
            or cohort_sha256 != reference_cohort_sha256
            or supervision_binding != reference_supervision
        ):
            raise Phase4PosthocError(
                "Attention test cohort/order/provenance differs across final seeds"
            )
    if len(set(experiment_ids)) != len(FINAL_SEEDS):
        raise Phase4PosthocError(
            "Attention aggregation requires a unique source experiment per seed"
        )
    if len(set(checkpoint_digests)) != len(FINAL_SEEDS):
        raise Phase4PosthocError(
            "Attention aggregation requires a unique persisted best checkpoint per seed"
        )
    names = (
        "kl_consensus_to_attention",
        "spatial_correlation",
        "distribution_overlap",
        "dice",
        "iou",
    )
    aggregate: dict[str, Any] = {}
    for name in names:
        values: list[float] = []
        for seed in FINAL_SEEDS:
            metrics = by_seed[seed]["metrics"]
            summaries = metrics.get("aggregate")
            if not isinstance(summaries, Mapping) or not isinstance(
                summaries.get(name), Mapping
            ):
                raise Phase4PosthocError(f"Attention seed result lacks {name}")
            mean = summaries[name].get("mean")
            if mean is None:
                continue
            values.append(float(mean))
        aggregate[name] = {
            "mean_of_seed_means": float(np.mean(values)) if values else None,
            "standard_deviation_of_seed_means": (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            ),
            "n_defined_seeds": len(values),
        }
    return {
        "schema_version": 1,
        "seeds": list(FINAL_SEEDS),
        "per_seed": [dict(by_seed[seed]) for seed in FINAL_SEEDS],
        "aggregate": aggregate,
        "standard_deviation_ddof": 1,
        "causal_explanation_claimed": False,
    }


def predict_validated_boundary_cohort(
    frozen: FrozenFinalModel,
    boundary_source: Any,
) -> dict[str, Any]:
    """Infer the verified boundary source using the frozen evaluation Dataset."""

    from torch.utils.data import DataLoader
    import torch
    from .phase4_boundary import (
        Phase4BoundarySource,
        assert_boundary_source_snapshot_unchanged,
        validate_boundary_provenance_record,
    )
    from .phase4_boundary_dataset import Phase4BoundaryDataset

    _assert_frozen(frozen)
    if not isinstance(boundary_source, Phase4BoundarySource):
        raise TypeError("boundary_source must come from the verified Phase 2 loader")
    assert_boundary_source_snapshot_unchanged(boundary_source.source_snapshot)
    provenance = validate_boundary_provenance_record(
        boundary_source.provenance,
        primary_supervision=frozen.source.supervision,
    )
    if provenance.get("source_snapshot_sha256") != sha256_json(
        dict(boundary_source.source_snapshot)
    ):
        raise Phase4PosthocError("Boundary source snapshot seal differs")
    binding = provenance["primary_cohort_binding"]
    for field in (
        "cohort_view", "sample_count", "manifest_bundle_sha256",
        "supervision_sha256", "ordered_supervision_sha256", "phase3_input_fingerprint",
    ):
        if binding.get(field) != frozen.source.descriptor.get(field):
            raise Phase4PosthocError(
                f"Boundary primary binding {field} differs from the frozen final model"
            )
    expected = provenance["ordered_identities"]
    dataset = Phase4BoundaryDataset(
        boundary_source.frame, rendered_root=boundary_source.rendered_root,
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    identities = (
        "patient_id", "nodule_id", "nodule_sha256", "study_instance_uid",
        "series_instance_uid", "split", "path", "rendered_mask_path",
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            if {"label", "true_class"} & set(batch):
                raise Phase4PosthocError("Boundary inference received a primary target")
            images = batch["image"].to(frozen.device)
            outputs = frozen.model(images)
            if not isinstance(outputs, Mapping) or "probs" not in outputs:
                raise Phase4PosthocError("Frozen model lacks boundary probabilities")
            probabilities = outputs["probs"].detach().cpu().numpy().astype(np.float64)
            count = len(images)
            if (
                probabilities.shape != (count, len(CLASS_ORDER))
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
            ):
                raise Phase4PosthocError("Boundary probabilities are invalid")
            fields = {name: _batch_text(batch, name, count) for name in identities}
            for position in range(count):
                rows.append({
                    **{name: fields[name][position] for name in identities},
                    **{
                        f"prob_{name}": float(probabilities[position, index])
                        for index, name in enumerate(CLASS_ORDER)
                    },
                    "predicted_class": CLASS_ORDER[int(np.argmax(probabilities[position]))],
                })
    actual_ids = [{key: row[key] for key in identities} for row in rows]
    expected_ids = [
        {
            key: row["image_path" if key == "path" else key]
            for key in identities
        }
        for row in expected
    ]
    if actual_ids != expected_ids:
        raise Phase4PosthocError("Boundary inference differs from the verified cohort order")
    for row, evidence in zip(rows, expected):
        for field in (
            "mean_malignancy", "malignancy_score_sum",
            "n_valid_malignancy_ratings", "boundary_stratum",
        ):
            row[field] = evidence[field]
    assert_boundary_source_snapshot_unchanged(boundary_source.source_snapshot)
    assert_source_snapshot_unchanged(frozen.source.source_snapshot)
    return {
        "schema_version": 1,
        "analysis": "phase4_boundary_stress_predictions",
        "seed": frozen.source.descriptor["seed"],
        "experiment_id": frozen.source.descriptor["experiment_id"],
        "checkpoint_kind": "best",
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "sample_count": len(rows),
        "evaluated_population": provenance["evaluated_population"],
        "source_validity_accounting": provenance["source_validity_accounting"],
        "source_validity_note": provenance["source_validity_note"],
        "rows": rows,
        "primary_three_class_targets_assigned": False,
        "primary_metrics_computed": False,
        "primary_headline_inclusion_permitted": False,
        "boundary_driven_tuning": False,
        "training_performed": False,
        "boundary_provenance": provenance,
    }


__all__ = [
    "FrozenFinalModel",
    "FrozenAttentionLoader",
    "Phase4PosthocError",
    "aggregate_attention_seed_results",
    "build_frozen_attention_loader",
    "evaluate_attention_alignment_from_loader",
    "evaluate_attention_alignment_from_source",
    "load_frozen_final_best_model",
    "predict_validated_boundary_cohort",
]
