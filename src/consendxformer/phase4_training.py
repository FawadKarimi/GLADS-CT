"""Small, testable Phase 4 training contracts shared by the executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .baseline_evaluation import CLASS_ORDER, evaluate_predictions
from .phase4_losses import (
    attention_alignment_loss,
    diagnostic_disagreement_loss,
    ordinal_distribution_loss,
)


class Phase4TrainingError(ValueError):
    """Raised when training inputs or loss controls violate Phase 4."""


@dataclass(frozen=True)
class Phase4Loss:
    total: torch.Tensor
    diagnostic: torch.Tensor
    ordinal: torch.Tensor
    attention: torch.Tensor


def split_phase4_supervision(
    frame: pd.DataFrame,
    *,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, pd.DataFrame]:
    """Return exact authoritative split subsequences without sorting or sampling."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise Phase4TrainingError("Phase 4 supervision must be a non-empty DataFrame")
    required = {
        "patient_id",
        "nodule_id",
        "series_instance_uid",
        "split",
        "label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Phase4TrainingError(f"Phase 4 supervision is missing columns: {missing}")
    if frame["nodule_id"].astype(str).duplicated().any():
        raise Phase4TrainingError("Phase 4 supervision contains duplicate nodule_id values")
    if not required_splits or not set(required_splits).issubset(
        {"train", "val", "test"}
    ):
        raise Phase4TrainingError("required_splits must be a non-empty Phase 4 subset")
    if not frame["split"].isin(required_splits).all():
        raise Phase4TrainingError("Phase 4 supervision contains an invalid split")
    if not frame["label"].isin(CLASS_ORDER).all():
        raise Phase4TrainingError("Phase 4 supervision contains a noncanonical hard label")
    splits = {
        split: frame.loc[frame["split"] == split].copy(deep=True).reset_index(drop=True)
        for split in required_splits
    }
    for split, subset in splits.items():
        if subset.empty:
            raise Phase4TrainingError(f"Phase 4 supervision has no {split} samples")
    if sum(len(subset) for subset in splits.values()) != len(frame):
        raise Phase4TrainingError("Phase 4 split partition changed supervision membership")
    return splits


def compute_training_class_weights(
    supervision: pd.DataFrame,
    *,
    num_classes: int = 3,
) -> np.ndarray:
    """Match B1 balanced weights using only training-split hard labels."""

    if num_classes != len(CLASS_ORDER):
        raise Phase4TrainingError("Phase 4 class weights are locked to three classes")
    if not isinstance(supervision, pd.DataFrame) or supervision.empty:
        raise Phase4TrainingError("supervision must be a non-empty DataFrame")
    missing = sorted({"split", "label"} - set(supervision.columns))
    if missing:
        raise Phase4TrainingError(f"class-weight inputs are missing columns: {missing}")
    train = supervision.loc[supervision["split"] == "train"]
    if train.empty:
        raise Phase4TrainingError("class-weight inputs contain no training samples")
    if not train["label"].isin(CLASS_ORDER).all():
        raise Phase4TrainingError("training hard labels are noncanonical")
    label_to_id = {name: index for index, name in enumerate(CLASS_ORDER)}
    labels = train["label"].map(label_to_id).to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts == 0):
        raise Phase4TrainingError(
            "Training split must contain every class for automatic class weights"
        )
    # This is sklearn.utils.class_weight.compute_class_weight('balanced', ...),
    # expressed directly so the provenance-critical rule stays visible here.
    weights = len(labels) / (num_classes * counts.astype(np.float64))
    return weights.astype(np.float64)


def _ablation_flag(ablation: Mapping[str, Any], field: str) -> bool:
    value = ablation.get(field)
    if type(value) is not bool:
        raise Phase4TrainingError(f"ablation.{field} must be boolean")
    return value


def _lambda(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4TrainingError(f"{field} must be numeric")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise Phase4TrainingError(f"{field} must be finite and non-negative")
    return number


def compute_phase4_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    lambda_ordinal: float,
    lambda_attention: float,
    class_weights: torch.Tensor | None,
    ablation: Mapping[str, Any],
) -> Phase4Loss:
    """Compose enabled losses, explicitly skipping zero/ablated auxiliaries."""

    ordinal_weight = _lambda(lambda_ordinal, "lambda_ordinal")
    attention_weight = _lambda(lambda_attention, "lambda_attention")
    use_disagreement = _ablation_flag(ablation, "use_disagreement_weighting")
    use_ordinal = _ablation_flag(ablation, "use_ordinal_head")
    use_attention = _ablation_flag(ablation, "use_attention_alignment")
    _ablation_flag(ablation, "use_hierarchical_fusion")

    required_outputs = {"logits"}
    required_batch = {"label"}
    if use_disagreement:
        required_batch |= {"soft_label", "disagreement"}
    missing_outputs = sorted(required_outputs - set(outputs))
    missing_batch = sorted(required_batch - set(batch))
    if missing_outputs or missing_batch:
        raise Phase4TrainingError(
            f"Loss inputs are incomplete: outputs={missing_outputs}, batch={missing_batch}"
        )
    logits = outputs["logits"]
    if use_disagreement:
        diagnostic = diagnostic_disagreement_loss(
            logits,
            batch["label"],
            batch["soft_label"],
            batch["disagreement"],
            class_weights=class_weights,
        )
    else:
        weights = (
            None
            if class_weights is None
            else class_weights.to(device=logits.device, dtype=logits.dtype)
        )
        diagnostic = F.cross_entropy(logits, batch["label"], weight=weights)
    zero = logits.sum() * 0.0

    if use_ordinal and ordinal_weight > 0.0:
        if "ordinal_logits" not in outputs or "ordinal_target" not in batch:
            raise Phase4TrainingError("Enabled ordinal loss inputs are missing")
        ordinal = ordinal_distribution_loss(
            outputs["ordinal_logits"], batch["ordinal_target"]
        )
    else:
        ordinal = zero

    if use_attention and attention_weight > 0.0:
        if "attention_map" not in outputs or "soft_consensus_mask" not in batch:
            raise Phase4TrainingError("Enabled attention loss inputs are missing")
        attention = attention_alignment_loss(
            outputs["attention_map"], batch["soft_consensus_mask"]
        )
    else:
        attention = zero

    total = diagnostic
    if use_ordinal:
        total = total + ordinal_weight * ordinal
    if use_attention:
        total = total + attention_weight * attention
    if not bool(torch.isfinite(total)):
        raise Phase4TrainingError("Phase 4 total loss is non-finite")
    return Phase4Loss(
        total=total,
        diagnostic=diagnostic,
        ordinal=ordinal,
        attention=attention,
    )


def classification_metrics_from_logits(
    labels: Sequence[int], logits: np.ndarray
) -> dict[str, Any]:
    """Return fixed-order accuracy/macro/per-class metrics for an epoch."""

    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape != (len(targets), len(CLASS_ORDER)):
        raise Phase4TrainingError("Epoch logits must have shape [N, 3]")
    if len(targets) == 0:
        raise Phase4TrainingError("Cannot calculate metrics for an empty epoch")
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predicted = values.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "nodule_id": [f"epoch-{index}" for index in range(len(targets))],
            "true_class": [CLASS_ORDER[int(value)] for value in targets],
            "predicted_class": [CLASS_ORDER[int(value)] for value in predicted],
            **{
                f"prob_{name}": probabilities[:, index]
                for index, name in enumerate(CLASS_ORDER)
            },
        }
    )
    return evaluate_predictions(
        frame,
        identifier_column="nodule_id",
        true_label_column="true_class",
        predicted_class_column="predicted_class",
        probability_columns=tuple(f"prob_{name}" for name in CLASS_ORDER),
    )


__all__ = [
    "Phase4Loss",
    "Phase4TrainingError",
    "classification_metrics_from_logits",
    "compute_phase4_loss",
    "compute_training_class_weights",
    "split_phase4_supervision",
]
