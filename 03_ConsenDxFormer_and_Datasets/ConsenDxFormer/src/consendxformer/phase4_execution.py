"""Guarded Phase 4 single-run trainer with structural test-set isolation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .phase4_artifacts import Phase4ExperimentLayout
from .phase4_checkpoints import (
    Phase4CheckpointContext,
    build_phase4_checkpoint,
    load_and_validate_phase4_checkpoints,
    save_phase4_checkpoint,
)
from .phase4_dataset import ConsenDxDataset
from .phase4_model import ConsenDxFormer
from .phase4_predictions import (
    build_phase4_prediction_table,
    evaluate_phase4_predictions,
)
from .phase4_protocol import resolve_phase4_run_spec
from .phase4_supervision import PairedAugmenter
from .phase4_training import (
    classification_metrics_from_logits,
    compute_phase4_loss,
    compute_training_class_weights,
    split_phase4_supervision,
)
from .reproducibility import (
    make_dataloader_generator,
    seed_dataloader_worker,
    set_global_seed,
)


@dataclass(frozen=True)
class Phase4ExecutionContext:
    experiment_id: str
    run_mode: str
    seed: int
    lambda_ordinal: float
    lambda_attention: float
    configuration_sha256: str
    project_commit: str
    manifest_bundle_sha256: str
    phase3_input_fingerprint: Mapping[str, Any]
    supervision_sha256: str
    ordered_supervision_sha256: str
    input_lock_sha256: str
    cohort_view: str
    expected_resolution: int
    sample_count: int
    split_counts: Mapping[str, int]
    effective_config: Mapping[str, Any]
    supervision: pd.DataFrame
    rendered_root: Path
    layout: Phase4ExperimentLayout

    @property
    def config_sha256(self) -> str:
        return self.configuration_sha256


@dataclass(frozen=True)
class Phase4ExecutionResult:
    history: tuple[Mapping[str, Any], ...]
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_checkpoint_record: Mapping[str, Any]
    last_checkpoint_record: Mapping[str, Any]
    predictions: Mapping[str, pd.DataFrame]
    metrics: Mapping[str, Mapping[str, Any]]
    best_epoch: int
    last_epoch: int
    class_weights: tuple[float, ...]
    events: tuple[str, ...]


class Phase4ExecutionError(RuntimeError):
    """Raised when a guarded Phase 4 training run cannot complete safely."""


def build_phase4_initial_datasets(
    supervision: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    rendered_root: str | Path,
    dataset_factory: Callable[..., Any] = ConsenDxDataset,
    run_mode: str = "final",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Construct train/val only; retain test metadata without touching test images."""

    if run_mode not in {"tuning", "final"}:
        raise Phase4ExecutionError("run_mode must be tuning or final")
    required_splits = ("train", "val") if run_mode == "tuning" else (
        "train",
        "val",
        "test",
    )
    frames = split_phase4_supervision(
        supervision, required_splits=required_splits
    )
    dataset = config.get("dataset")
    augmentation = config.get("augmentation")
    if not isinstance(dataset, Mapping) or not isinstance(augmentation, Mapping):
        raise Phase4ExecutionError("Phase 4 dataset/augmentation config is missing")
    train_policy = augmentation.get("train")
    eval_policy = augmentation.get("eval")
    if not isinstance(train_policy, Mapping) or not isinstance(eval_policy, Mapping):
        raise Phase4ExecutionError("Phase 4 augmentation policy is incomplete")
    if dict(eval_policy) != {"enabled": False}:
        raise Phase4ExecutionError("Phase 4 validation/test augmentation must be disabled")
    train_augmenter = PairedAugmenter(**dict(train_policy), enabled=True)
    common = {
        "image_size": int(dataset["image_size"]),
        "in_channels": int(dataset["in_channels"]),
        "rendered_root": rendered_root,
    }
    datasets = {
        "train": dataset_factory(frames["train"], augmenter=train_augmenter, **common),
        "val": dataset_factory(frames["val"], augmenter=None, **common),
    }
    return frames, datasets


def _build_test_dataset(
    test_frame: pd.DataFrame,
    config: Mapping[str, Any],
    rendered_root: Path,
) -> Any:
    dataset = config["dataset"]
    return ConsenDxDataset(
        test_frame,
        image_size=int(dataset["image_size"]),
        in_channels=int(dataset["in_channels"]),
        rendered_root=rendered_root,
        augmenter=None,
    )


def _loader(
    dataset: Any,
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    settings = config["dataset"]
    determinism = config["determinism"]
    if determinism.get("dataloader_worker_seeding") is not True:
        raise Phase4ExecutionError("DataLoader worker seeding must remain enabled")
    if determinism.get("seeded_dataloader_generator") is not True:
        raise Phase4ExecutionError("Seeded DataLoader generators must remain enabled")
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        num_workers=int(settings["num_workers"]),
        shuffle=shuffle,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(seed, require_torch=True),
    )


def _model(config: Mapping[str, Any]) -> torch.nn.Module:
    dataset = config["dataset"]
    model = config["model"]
    ablation = config["ablation"]
    if model.get("use_transformer") is not True:
        raise Phase4ExecutionError("The locked Phase 4 Transformer cannot be disabled")
    if model.get("fusion_type") != "concat" or model.get(
        "use_raw_transformer_in_fusion"
    ) is not False:
        raise Phase4ExecutionError("The locked Phase 4 fusion configuration changed")
    if type(model.get("use_hierarchical_fusion")) is not bool or model[
        "use_hierarchical_fusion"
    ] != ablation.get("use_hierarchical_fusion"):
        raise Phase4ExecutionError(
            "Configured and ablation hierarchical-fusion controls disagree"
        )
    return ConsenDxFormer(
        image_size=int(dataset["image_size"]),
        in_channels=int(dataset["in_channels"]),
        num_classes=int(dataset["num_classes"]),
        cnn_channels=tuple(int(value) for value in model["cnn_channels"]),
        embed_dim=int(model["embed_dim"]),
        transformer_heads=int(model["transformer_heads"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_mlp_ratio=float(model["transformer_mlp_ratio"]),
        dropout=float(model["dropout"]),
        classifier_hidden_dim=int(model["classifier_hidden_dim"]),
        use_positional_encoding=bool(model["use_positional_encoding"]),
        use_spatial_attention=bool(model["use_spatial_attention"]),
        use_hierarchical_fusion=model["use_hierarchical_fusion"],
    )


def _move_loss_batch(batch: Mapping[str, Any], device: str) -> dict[str, torch.Tensor]:
    required = (
        "label",
        "soft_label",
        "disagreement",
        "ordinal_target",
        "soft_consensus_mask",
    )
    missing = [field for field in required if field not in batch]
    if missing:
        raise Phase4ExecutionError(f"Phase 4 batch lacks supervision: {missing}")
    return {field: batch[field].to(device) for field in required}


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    training: bool,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    lambda_ordinal: float,
    lambda_attention: float,
    ablation: Mapping[str, Any],
    grad_clip_norm: float | None,
) -> tuple[dict[str, Any], np.ndarray]:
    model.train(training)
    totals = {name: 0.0 for name in ("total", "diagnostic", "ordinal", "attention")}
    labels: list[int] = []
    logits_rows: list[list[float]] = []
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        loss_batch = _move_loss_batch(batch, device)
        batch_size = int(images.shape[0])
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = compute_phase4_loss(
                outputs,
                loss_batch,
                lambda_ordinal=lambda_ordinal,
                lambda_attention=lambda_attention,
                class_weights=class_weights,
                ablation=ablation,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        for name in totals:
            totals[name] += float(getattr(loss, name).detach().item()) * batch_size
        sample_count += batch_size
        detached = outputs["logits"].detach().cpu()
        logits_rows.extend(detached.numpy().tolist())
        labels.extend(loss_batch["label"].detach().cpu().tolist())
    if sample_count != len(loader.dataset) or sample_count == 0:
        raise Phase4ExecutionError("DataLoader did not yield its exact dataset membership")
    logits = np.asarray(logits_rows, dtype=np.float64)
    metrics = classification_metrics_from_logits(labels, logits)
    averages = {name: float(value / sample_count) for name, value in totals.items()}
    metrics.update(
        {
            "loss": averages["total"],
            "total_loss": averages["total"],
            "diagnostic_loss": averages["diagnostic"],
            "ordinal_loss": averages["ordinal"],
            "attention_loss": averages["attention"],
        }
    )
    return metrics, logits


def execute_phase4_single_run(context: Phase4ExecutionContext) -> Phase4ExecutionResult:
    """Train one run; tuning mode has no code path that constructs a test loader."""

    if context.run_mode not in {"tuning", "final"}:
        raise Phase4ExecutionError("run_mode must be tuning or final")
    config = dict(context.effective_config)
    resolved_spec = resolve_phase4_run_spec(
        config,
        mode=context.run_mode,
        seed=context.seed,
        lambda_ordinal=(context.lambda_ordinal if context.run_mode == "tuning" else None),
        lambda_attention=(
            context.lambda_attention if context.run_mode == "tuning" else None
        ),
    )
    if (
        resolved_spec.lambda_ordinal != context.lambda_ordinal
        or resolved_spec.lambda_attention != context.lambda_attention
    ):
        raise Phase4ExecutionError(
            "Execution lambda values do not match the mode-authorized configuration"
        )
    deterministic = config["determinism"]
    set_global_seed(
        context.seed,
        deterministic_algorithms=bool(deterministic["deterministic_algorithms"]),
        cudnn_benchmark=bool(deterministic["cudnn_benchmark"]),
        warn_only=bool(deterministic["warn_only"]),
        require_torch=True,
    )
    frames, datasets = build_phase4_initial_datasets(
        context.supervision,
        config,
        rendered_root=context.rendered_root,
        dataset_factory=ConsenDxDataset,
        run_mode=context.run_mode,
    )
    train_loader = _loader(datasets["train"], config, seed=context.seed, shuffle=True)
    val_loader = _loader(datasets["val"], config, seed=context.seed, shuffle=False)

    requested_device = str(config.get("device", "auto"))
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu"
    class_weight_values = compute_training_class_weights(
        context.supervision, num_classes=int(config["dataset"]["num_classes"])
    )
    if config["training"].get("class_weights") != "auto":
        raise Phase4ExecutionError("Phase 4 automatic B1 class weighting changed")
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    model = _model(config).to(device)
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    if training.get("scheduler") != "reduce_on_plateau":
        raise Phase4ExecutionError("Phase 4 scheduler must be reduce_on_plateau")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )
    patience = int(training["early_stopping_patience"])
    if patience <= 0:
        raise Phase4ExecutionError("early_stopping_patience must be positive")
    clip = training.get("grad_clip_norm")
    grad_clip = None if clip is None else float(clip)

    checkpoint_context = Phase4CheckpointContext.from_execution_context(context)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    no_improvement = 0
    best_path = context.layout.checkpoints_dir / "best.pt"
    last_path = context.layout.checkpoints_dir / "last.pt"
    for epoch in range(1, int(training["epochs"]) + 1):
        train_metrics, _ = _run_epoch(
            model,
            train_loader,
            device=device,
            training=True,
            optimizer=optimizer,
            class_weights=class_weights,
            lambda_ordinal=context.lambda_ordinal,
            lambda_attention=context.lambda_attention,
            ablation=config["ablation"],
            grad_clip_norm=grad_clip,
        )
        validation_metrics, _ = _run_epoch(
            model,
            val_loader,
            device=device,
            training=False,
            optimizer=optimizer,
            class_weights=class_weights,
            lambda_ordinal=context.lambda_ordinal,
            lambda_attention=context.lambda_attention,
            ablation=config["ablation"],
            grad_clip_norm=grad_clip,
        )
        validation_loss = float(validation_metrics["total_loss"])
        if not math.isfinite(validation_loss):
            raise Phase4ExecutionError("Validation total loss is non-finite")
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            no_improvement = 0
            save_phase4_checkpoint(
                build_phase4_checkpoint(
                    checkpoint_kind="best",
                    context=checkpoint_context,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    epoch=epoch,
                    best_epoch=epoch,
                    best_validation_loss=best_loss,
                    validation_metrics=validation_metrics,
                ),
                best_path,
            )
        else:
            no_improvement += 1
        save_phase4_checkpoint(
            build_phase4_checkpoint(
                checkpoint_kind="last",
                context=checkpoint_context,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                validation_metrics=validation_metrics,
            ),
            last_path,
        )
        if no_improvement >= patience:
            break
    if not history or best_epoch <= 0 or not best_path.is_file() or not last_path.is_file():
        raise Phase4ExecutionError("Training produced no validation-selected checkpoints")

    persisted_best, persisted_last = load_and_validate_phase4_checkpoints(
        best_path,
        last_path,
        context=checkpoint_context,
        history=history,
        map_location=device,
    )
    events = [
        "best_checkpoint_selected_from_validation_total_loss",
        "persisted_checkpoints_reloaded_and_validated",
    ]
    model.load_state_dict(persisted_best["model_state"])
    if context.run_mode == "final":
        events.append("best_checkpoint_reloaded_for_final_validation_and_test")
    else:
        events.append("best_checkpoint_reloaded_for_validation")

    val_metrics, val_logits = _run_epoch(
        model,
        val_loader,
        device=device,
        training=False,
        optimizer=optimizer,
        class_weights=class_weights,
        lambda_ordinal=context.lambda_ordinal,
        lambda_attention=context.lambda_attention,
        ablation=config["ablation"],
        grad_clip_norm=grad_clip,
    )
    val_predictions = build_phase4_prediction_table(
        frames["val"],
        val_logits,
        seed=context.seed,
        lambda_ordinal=context.lambda_ordinal,
        lambda_attention=context.lambda_attention,
        run_mode=context.run_mode,
        experiment_id=context.experiment_id,
    )
    # Validate classification metrics independently from the epoch accumulator.
    evaluate_phase4_predictions(val_predictions)
    predictions: dict[str, pd.DataFrame] = {"val": val_predictions}
    metrics: dict[str, Mapping[str, Any]] = {"val": val_metrics}

    if context.run_mode == "final":
        # This is the first construction of anything that can read a test image.
        test_dataset = _build_test_dataset(frames["test"], config, context.rendered_root)
        test_loader = _loader(test_dataset, config, seed=context.seed, shuffle=False)
        events.append("test_evaluation_started_after_final_selection_and_best_reload")
        test_metrics, test_logits = _run_epoch(
            model,
            test_loader,
            device=device,
            training=False,
            optimizer=optimizer,
            class_weights=class_weights,
            lambda_ordinal=context.lambda_ordinal,
            lambda_attention=context.lambda_attention,
            ablation=config["ablation"],
            grad_clip_norm=grad_clip,
        )
        test_predictions = build_phase4_prediction_table(
            frames["test"],
            test_logits,
            seed=context.seed,
            lambda_ordinal=context.lambda_ordinal,
            lambda_attention=context.lambda_attention,
            run_mode=context.run_mode,
            experiment_id=context.experiment_id,
        )
        evaluate_phase4_predictions(test_predictions)
        predictions["test"] = test_predictions
        metrics["test"] = test_metrics

    return Phase4ExecutionResult(
        history=tuple(history),
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        best_checkpoint_record=persisted_best,
        last_checkpoint_record=persisted_last,
        predictions=predictions,
        metrics=metrics,
        best_epoch=best_epoch,
        last_epoch=int(persisted_last["epoch"]),
        class_weights=tuple(float(value) for value in class_weight_values),
        events=tuple(events),
    )


__all__ = [
    "Phase4ExecutionContext",
    "Phase4ExecutionError",
    "Phase4ExecutionResult",
    "build_phase4_initial_datasets",
    "execute_phase4_single_run",
]
