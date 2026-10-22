"""Locked-source LungDxFormer execution adapter for Phase 3.

The adapter imports the immutable official implementation lazily, constructs
the released model/loss/augmentation objects, and owns only the experiment
lifecycle features missing from the released trainer: seeded workers, complete
best/last checkpoints, and identity-preserving validation/test inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .phase3_artifacts import ExperimentLayout
from .phase3_checkpoints import (
    build_best_checkpoint,
    build_last_checkpoint,
    select_best_validation_epoch,
    validate_checkpoint_pair,
    validate_epoch_history,
)
from .phase3_predictions import build_prediction_table
from .reproducibility import make_dataloader_generator, seed_dataloader_worker, set_global_seed


@dataclass(frozen=True)
class Phase3ExecutionContext:
    experiment_id: str
    baseline_variant: str
    seed: int
    configuration_sha256: str
    baseline_source_commit: str
    project_commit: str
    metadata_sha256: str
    cohort_sha256: str
    ordered_sample_sha256: str
    frozen_split_sha256: str
    baseline_source_root: Path
    effective_baseline_config: Mapping[str, Any]
    metadata: pd.DataFrame
    layout: ExperimentLayout


@dataclass(frozen=True)
class Phase3ExecutionResult:
    history: tuple[Mapping[str, Any], ...]
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_checkpoint_record: Mapping[str, Any]
    last_checkpoint_record: Mapping[str, Any]
    predictions: Mapping[str, pd.DataFrame]
    best_epoch: int
    last_epoch: int
    events: tuple[str, ...]


class Phase3ExecutionError(RuntimeError):
    """Raised when the locked real baseline cannot execute safely."""


class IdentityPreservingDataset:
    """Attach stable Phase 3 identities to a released baseline dataset.

    Composition keeps this adapter importable and pickleable for DataLoader
    workers that use the ``spawn`` start method, including Windows workers.
    """

    def __init__(self, dataset: Any, metadata: pd.DataFrame) -> None:
        if len(dataset) != len(metadata):
            raise Phase3ExecutionError(
                "Baseline dataset and identity metadata have different lengths"
            )
        self.dataset = dataset
        self.identities = metadata.loc[
            :, ["series_instance_uid", "nodule_id", "label"]
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        row = self.identities.iloc[index]
        sample["series_instance_uid"] = str(row["series_instance_uid"])
        sample["nodule_id"] = str(row["nodule_id"])
        sample["true_class"] = str(row["label"])
        return sample


def _module_is_within_source(module: Any, source_path: Path) -> bool:
    location = getattr(module, "__file__", None)
    if not isinstance(location, str):
        return False
    try:
        Path(location).resolve().relative_to(source_path)
        return True
    except (OSError, ValueError):
        return False


def _reject_cached_baseline_modules() -> None:
    cached = sorted(
        name
        for name in sys.modules
        if name == "lungdxformer" or name.startswith("lungdxformer.")
    )
    if cached:
        raise Phase3ExecutionError(
            "A LungDxFormer module is already cached in this interpreter; refusing "
            f"ambiguous execution: {cached[:10]!r}"
        )


def _atomic_torch_save(torch: Any, payload: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def _checkpoint_provenance(
    context: Phase3ExecutionContext,
) -> dict[str, Any]:
    return {
        "baseline_source_commit": context.baseline_source_commit,
        "project_commit": context.project_commit,
        "metadata_sha256": context.metadata_sha256,
        "cohort_sha256": context.cohort_sha256,
        "ordered_sample_sha256": context.ordered_sample_sha256,
        "frozen_split_sha256": context.frozen_split_sha256,
        "class_order": list(CLASS_ORDER),
    }


def _load_persisted_checkpoint(
    torch: Any,
    path: Path,
    *,
    device: str,
) -> Mapping[str, Any]:
    """Load one authoritative checkpoint snapshot from its persisted file."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise Phase3ExecutionError(
            f"Could not load persisted Phase 3 checkpoint: {path}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise Phase3ExecutionError(
            f"Persisted Phase 3 checkpoint is not a mapping: {path}"
        )
    return checkpoint


def _load_and_validate_persisted_checkpoints(
    torch: Any,
    context: Phase3ExecutionContext,
    *,
    best_path: Path,
    last_path: Path,
    device: str,
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reload and validate the authoritative best/last checkpoint snapshots."""

    loaded_best = _load_persisted_checkpoint(torch, best_path, device=device)
    loaded_last = _load_persisted_checkpoint(torch, last_path, device=device)
    try:
        validated_history = validate_epoch_history(history)
        selection = select_best_validation_epoch(validated_history)
        best, last = validate_checkpoint_pair(loaded_best, loaded_last)
    except (TypeError, ValueError) as exc:
        raise Phase3ExecutionError(
            "Persisted Phase 3 best/last checkpoint contract validation failed"
        ) from exc

    expected_context = {
        "experiment_id": context.experiment_id,
        "baseline_variant": context.baseline_variant,
        "seed": context.seed,
        "config_sha256": context.configuration_sha256,
        **_checkpoint_provenance(context),
        "effective_baseline_config": dict(context.effective_baseline_config),
    }
    for kind, checkpoint in (("best", best), ("last", last)):
        for field, expected in expected_context.items():
            if checkpoint.get(field) != expected:
                raise Phase3ExecutionError(
                    f"Persisted {kind} checkpoint field {field!r} disagrees "
                    "with its execution context"
                )

    selected_epoch = int(selection["epoch"])
    selected_loss = float(selection["value"])
    final_epoch = int(validated_history[-1]["epoch"])
    if best["epoch"] != selected_epoch or best["best_epoch"] != selected_epoch:
        raise Phase3ExecutionError(
            "Persisted best checkpoint epoch disagrees with validation selection"
        )
    if last["epoch"] != final_epoch or last["best_epoch"] != selected_epoch:
        raise Phase3ExecutionError(
            "Persisted last checkpoint epoch metadata disagrees with training history"
        )
    if (
        best["best_validation_loss"] != selected_loss
        or last["best_validation_loss"] != selected_loss
    ):
        raise Phase3ExecutionError(
            "Persisted checkpoint validation loss disagrees with validation selection"
        )
    history_by_epoch = {
        int(record["epoch"]): record for record in validated_history
    }
    for kind, checkpoint in (("best", best), ("last", last)):
        validation_metrics = checkpoint.get("validation_metrics")
        expected_loss = history_by_epoch[int(checkpoint["epoch"])]["validation"][
            "loss"
        ]
        if (
            not isinstance(validation_metrics, Mapping)
            or validation_metrics.get("loss") != expected_loss
        ):
            raise Phase3ExecutionError(
                f"Persisted {kind} checkpoint validation metrics disagree "
                "with training history"
            )
    return best, last


def _require_real_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "series_instance_uid",
        "nodule_id",
        "split",
        "label",
        "image_path",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise Phase3ExecutionError(f"Real Phase 3 metadata lacks columns: {missing}")
    frame = metadata.copy(deep=True)
    if frame.empty:
        raise Phase3ExecutionError("Real Phase 3 metadata is empty")
    if frame["nodule_id"].astype(str).duplicated().any():
        raise Phase3ExecutionError("Real Phase 3 metadata has duplicate nodule IDs")
    if not frame["split"].isin(("train", "val", "test")).all():
        raise Phase3ExecutionError("Real Phase 3 metadata has an invalid split")
    for split in ("train", "val", "test"):
        if not bool((frame["split"] == split).any()):
            raise Phase3ExecutionError(f"Real Phase 3 metadata has no {split} samples")
    label_to_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    if not frame["label"].isin(CLASS_ORDER).all():
        raise Phase3ExecutionError("Real Phase 3 metadata has a noncanonical class label")
    frame["label_id"] = frame["label"].map(label_to_index).astype("int64")
    return frame


def execute_locked_lungdxformer(
    context: Phase3ExecutionContext,
) -> Phase3ExecutionResult:
    """Train/evaluate the exact locked architecture without editing its checkout."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from sklearn.utils.class_weight import compute_class_weight
    except Exception as exc:  # pragma: no cover - real runtime only
        raise Phase3ExecutionError(
            "Real Phase 3 execution requires the locked LungDxFormer runtime dependencies"
        ) from exc

    config = dict(context.effective_baseline_config)
    deterministic = config.get("phase3_reproducibility", {})
    set_global_seed(
        context.seed,
        deterministic_algorithms=bool(deterministic.get("deterministic_algorithms", True)),
        cudnn_benchmark=bool(deterministic.get("cudnn_benchmark", False)),
        warn_only=bool(deterministic.get("warn_only", True)),
        require_torch=True,
    )

    project_subdir = str(config["baseline"]["project_subdir"])
    source_path = (context.baseline_source_root / project_subdir / "src").resolve()
    if not source_path.is_dir():
        raise Phase3ExecutionError(f"Locked baseline Python source is missing: {source_path}")
    _reject_cached_baseline_modules()

    previous_bytecode = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    previous_env_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    previous_env_pycache = os.environ.get("PYTHONPYCACHEPREFIX")
    redirected_pycache = str(
        (context.layout.provenance_dir / ".disabled_pycache").resolve()
    )
    inserted = False
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
        inserted = True
    sys.dont_write_bytecode = True
    # Redirect cache lookup as well as cache writes, so ignored bytecode in the
    # locked checkout can never supersede the verified source files. Export the
    # same controls for spawned DataLoader workers.
    sys.pycache_prefix = redirected_pycache
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = redirected_pycache
    try:
        from lungdxformer.data.augmentation import BasicAugmenter
        from lungdxformer.data.dataset import LungNoduleDataset
        from lungdxformer.evaluation.metrics import classification_metrics
        from lungdxformer.models.lungdxformer import LungDxFormer
        from lungdxformer.training.early_stopping import EarlyStopping
        from lungdxformer.training.losses import build_loss
        from lungdxformer.training.scheduler import build_scheduler

        imported_baseline_modules = (
            BasicAugmenter,
            LungNoduleDataset,
            classification_metrics,
            LungDxFormer,
            EarlyStopping,
            build_loss,
            build_scheduler,
        )
        for imported in imported_baseline_modules:
            module = sys.modules.get(imported.__module__)
            if module is None or not _module_is_within_source(module, source_path):
                raise Phase3ExecutionError(
                    "A baseline component was imported from outside the locked source: "
                    f"{imported.__module__}"
                )

        frame = _require_real_metadata(context.metadata)

        train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
        val_frame = frame.loc[frame["split"] == "val"].sort_values(
            "nodule_id", kind="stable"
        ).reset_index(drop=True)

        train_aug = BasicAugmenter(**config["augmentation"]["train"], enabled=True)
        train_dataset = LungNoduleDataset(
            train_frame,
            config["dataset"]["image_size"],
            config["dataset"]["in_channels"],
            augmenter=train_aug,
        )
        val_dataset = IdentityPreservingDataset(
            LungNoduleDataset(
                val_frame,
                config["dataset"]["image_size"],
                config["dataset"]["in_channels"],
                augmenter=None,
            ),
            val_frame,
        )
        loader_kwargs = {
            "batch_size": config["dataset"]["batch_size"],
            "num_workers": config["dataset"]["num_workers"],
            "worker_init_fn": seed_dataloader_worker,
        }
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=make_dataloader_generator(context.seed, require_torch=True),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            generator=make_dataloader_generator(context.seed, require_torch=True),
            **loader_kwargs,
        )

        cuda_available = bool(torch.cuda.is_available())
        requested_device = config.get("device", "auto")
        device = "cuda" if requested_device == "auto" and cuda_available else "cpu"

        y_train = train_frame["label_id"].astype(int).to_numpy()
        class_weights = None
        if config["training"]["class_weights"] == "auto":
            try:
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=np.arange(config["dataset"]["num_classes"]),
                    y=y_train,
                ).tolist()
            except ValueError as exc:
                raise Phase3ExecutionError(
                    "Training split must contain every class for automatic class weights"
                ) from exc
        elif isinstance(config["training"]["class_weights"], list):
            class_weights = list(config["training"]["class_weights"])

        model = LungDxFormer(
            image_size=config["dataset"]["image_size"],
            in_channels=config["dataset"]["in_channels"],
            num_classes=config["dataset"]["num_classes"],
            cnn_channels=tuple(config["model"]["cnn_channels"]),
            embed_dim=config["model"]["embed_dim"],
            transformer_heads=config["model"]["transformer_heads"],
            transformer_layers=config["model"]["transformer_layers"],
            transformer_mlp_ratio=config["model"]["transformer_mlp_ratio"],
            dropout=config["model"]["dropout"],
            use_transformer=config["model"]["use_transformer"],
            use_positional_encoding=config["model"]["use_positional_encoding"],
            use_spatial_attention=config["model"]["use_spatial_attention"],
            fusion_type=config["model"]["fusion_type"],
            use_raw_transformer_in_fusion=config["model"]["use_raw_transformer_in_fusion"],
            classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )
        training_config = dict(config["training"])
        training_config["_computed_class_weights"] = class_weights
        criterion = build_loss(
            class_weights=class_weights,
            label_smoothing=training_config.get("label_smoothing", 0.0),
            device=device,
        )
        scheduler = build_scheduler(optimizer, training_config)
        early_stopping = EarlyStopping(
            patience=training_config["early_stopping_patience"], mode="min"
        )

        def run_epoch(loader: Any, *, training: bool) -> tuple[dict[str, Any], np.ndarray]:
            model.train(training)
            total_loss = 0.0
            labels: list[int] = []
            predicted: list[int] = []
            probabilities: list[list[float]] = []
            logits_rows: list[list[float]] = []
            for batch in loader:
                inputs = batch["image"].to(device)
                target = batch["label"].to(device)
                with torch.set_grad_enabled(training):
                    output = model(inputs)
                    logits = output["logits"]
                    loss = criterion(logits, target)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        clip = training_config.get("grad_clip_norm")
                        if clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                        optimizer.step()
                total_loss += float(loss.item()) * int(inputs.size(0))
                detached = logits.detach().cpu()
                probs = torch.softmax(detached, dim=1).numpy()
                labels.extend(target.detach().cpu().tolist())
                predicted.extend(detached.argmax(dim=1).tolist())
                probabilities.extend(probs.tolist())
                logits_rows.extend(detached.numpy().tolist())
            metrics = classification_metrics(
                labels,
                predicted,
                np.asarray(probabilities),
                num_classes=len(CLASS_ORDER),
            )
            metrics["loss"] = float(total_loss / max(1, len(loader.dataset)))
            return metrics, np.asarray(logits_rows, dtype=np.float64)

        history: list[dict[str, Any]] = []
        best_loss = math.inf
        best_epoch = 0
        best_path = context.layout.checkpoints_dir / "best.pt"
        last_path = context.layout.checkpoints_dir / "last.pt"
        best_checkpoint_written = False
        last_checkpoint_written = False
        for epoch in range(1, training_config["epochs"] + 1):
            train_metrics, _ = run_epoch(train_loader, training=True)
            validation_metrics, _ = run_epoch(val_loader, training=False)
            validation_loss = float(validation_metrics["loss"])
            if not math.isfinite(validation_loss):
                raise Phase3ExecutionError("Validation loss is non-finite")
            if scheduler is not None:
                scheduler.step(validation_loss)
            history.append(
                {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            )
            improved = early_stopping.step(validation_loss)
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                checkpoint_to_persist = build_best_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=(
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    epoch=epoch,
                    best_validation_loss=best_loss,
                    seed=context.seed,
                    baseline_variant=context.baseline_variant,
                    config_sha256=context.configuration_sha256,
                    experiment_id=context.experiment_id,
                )
                checkpoint_to_persist.update(_checkpoint_provenance(context))
                checkpoint_to_persist["effective_baseline_config"] = config
                checkpoint_to_persist["validation_metrics"] = validation_metrics
                _atomic_torch_save(
                    torch,
                    checkpoint_to_persist,
                    best_path,
                )
                best_checkpoint_written = True
            checkpoint_to_persist = build_last_checkpoint(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=(scheduler.state_dict() if scheduler is not None else None),
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                seed=context.seed,
                baseline_variant=context.baseline_variant,
                config_sha256=context.configuration_sha256,
                experiment_id=context.experiment_id,
            )
            checkpoint_to_persist.update(_checkpoint_provenance(context))
            checkpoint_to_persist["effective_baseline_config"] = config
            checkpoint_to_persist["validation_metrics"] = validation_metrics
            _atomic_torch_save(
                torch,
                checkpoint_to_persist,
                last_path,
            )
            last_checkpoint_written = True
            if early_stopping.should_stop:
                break

        if (
            not best_checkpoint_written
            or not last_checkpoint_written
            or best_epoch <= 0
            or not history
        ):
            raise Phase3ExecutionError("Training produced no validation-selected checkpoint")
        persisted_best, persisted_last = _load_and_validate_persisted_checkpoints(
            torch,
            context,
            best_path=best_path,
            last_path=last_path,
            device=device,
            history=history,
        )
        model.load_state_dict(persisted_best["model_state"])

        _, val_logits = run_epoch(val_loader, training=False)
        val_predictions = build_prediction_table(
            val_frame,
            val_logits,
            seed=context.seed,
            baseline_variant=context.baseline_variant,
            experiment_id=context.experiment_id,
        )

        # The test split is not instantiated or evaluated until the best epoch
        # has been fixed exclusively from validation loss and reloaded.
        test_frame = frame.loc[frame["split"] == "test"].sort_values(
            "nodule_id", kind="stable"
        ).reset_index(drop=True)
        test_dataset = IdentityPreservingDataset(
            LungNoduleDataset(
                test_frame,
                config["dataset"]["image_size"],
                config["dataset"]["in_channels"],
                augmenter=None,
            ),
            test_frame,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            generator=make_dataloader_generator(context.seed, require_torch=True),
            **loader_kwargs,
        )
        _, test_logits = run_epoch(test_loader, training=False)
        test_predictions = build_prediction_table(
            test_frame,
            test_logits,
            seed=context.seed,
            baseline_variant=context.baseline_variant,
            experiment_id=context.experiment_id,
        )
        return Phase3ExecutionResult(
            history=tuple(history),
            best_checkpoint_path=best_path,
            last_checkpoint_path=last_path,
            best_checkpoint_record=persisted_best,
            last_checkpoint_record=persisted_last,
            predictions={"val": val_predictions, "test": test_predictions},
            best_epoch=best_epoch,
            last_epoch=int(persisted_last["epoch"]),
            events=(
                "best_checkpoint_selected_from_validation_loss",
                "persisted_checkpoints_reloaded_and_validated",
                "best_checkpoint_reloaded",
                "test_evaluation_started_after_checkpoint_selection",
            ),
        )
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.pycache_prefix = previous_pycache_prefix
        if previous_env_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous_env_bytecode
        if previous_env_pycache is None:
            os.environ.pop("PYTHONPYCACHEPREFIX", None)
        else:
            os.environ["PYTHONPYCACHEPREFIX"] = previous_env_pycache
        for name in tuple(sys.modules):
            if name == "lungdxformer" or name.startswith("lungdxformer."):
                sys.modules.pop(name, None)
        if inserted:
            try:
                sys.path.remove(str(source_path))
            except ValueError:
                pass


__all__ = [
    "IdentityPreservingDataset",
    "Phase3ExecutionContext",
    "Phase3ExecutionError",
    "Phase3ExecutionResult",
    "execute_locked_lungdxformer",
]
