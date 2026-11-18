"""Validation-only checkpoint selection and Phase 3 checkpoint contracts.

This module deliberately has no PyTorch dependency.  Framework-native state
objects can be carried opaquely in the returned dictionaries and serialized by
the training adapter that owns the framework runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any


VALIDATION_CRITERION = "validation_loss"
VALIDATION_MODE = "min"
SELECTION_SPLIT = "validation"
CHECKPOINT_KINDS = ("best", "last")
BASELINE_VARIANTS = ("B0", "B1")
HISTORY_KEYS = ("epoch", "train", "validation")

_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


class CheckpointContractError(ValueError):
    """Raised when history or checkpoint metadata can compromise selection."""


def _positive_epoch(value: Any, *, field: str = "epoch") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointContractError(f"{field} must be a positive integer; got {value!r}")
    if value < 1:
        raise CheckpointContractError(f"{field} must be at least 1; got {value}")
    return int(value)


def _seed(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 2**32 - 1
    ):
        raise CheckpointContractError(
            f"seed must be an integer in the range 0..{2**32 - 1}; got {value!r}"
        )
    return int(value)


def _finite_loss(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise CheckpointContractError(f"{field} must be a finite number, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError(f"{field} must be a finite number; got {value!r}") from exc
    if not math.isfinite(numeric):
        raise CheckpointContractError(f"{field} must be finite; got {value!r}")
    return numeric


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointContractError(f"{field} must be a non-blank string")
    return value.strip()


def _config_sha256(value: Any) -> str:
    text = _required_text(value, field="config_sha256")
    if not _SHA256.fullmatch(text):
        raise CheckpointContractError(
            "config_sha256 must contain exactly 64 hexadecimal characters"
        )
    return text.lower()


def _variant(value: Any) -> str:
    text = _required_text(value, field="baseline_variant")
    if text not in BASELINE_VARIANTS:
        raise CheckpointContractError(
            f"baseline_variant must be one of {list(BASELINE_VARIANTS)!r}; got {text!r}"
        )
    return text


def _reject_test_fields(value: Any, *, location: str) -> None:
    """Reject test-derived fields anywhere in a selection-history record."""

    if not isinstance(value, Mapping):
        return
    for key, nested in value.items():
        key_text = str(key).casefold()
        if key_text == "test" or key_text.startswith("test_"):
            raise CheckpointContractError(
                f"test-derived field {key!r} is forbidden in checkpoint-selection "
                f"history ({location})"
            )
        _reject_test_fields(nested, location=f"{location}.{key}")


def build_history_record(
    *,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    train_metrics: Mapping[str, Any] | None = None,
    validation_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one epoch record containing train and validation information only."""

    train = dict(train_metrics or {})
    validation = dict(validation_metrics or {})
    if "loss" in train or "loss" in validation:
        raise CheckpointContractError(
            "train_metrics/validation_metrics must not override the explicit loss values"
        )
    train["loss"] = _finite_loss(train_loss, field="train.loss")
    validation["loss"] = _finite_loss(
        validation_loss, field="validation.loss"
    )
    record = {
        "epoch": _positive_epoch(epoch),
        "train": train,
        "validation": validation,
    }
    _reject_test_fields(record, location="history record")
    return record


def validate_epoch_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate ordered train/validation records used for model selection.

    Top-level fields are exact, and any nested test-derived key is forbidden.
    Other train/validation metrics may be retained for diagnostics, but only
    ``validation.loss`` participates in checkpoint selection.
    """

    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence of mappings")
    if len(history) == 0:
        raise CheckpointContractError("history must contain at least one epoch")

    canonical: list[dict[str, Any]] = []
    previous_epoch = 0
    for position, raw_record in enumerate(history):
        if not isinstance(raw_record, Mapping):
            raise CheckpointContractError(
                f"history record {position} must be a mapping"
            )
        _reject_test_fields(raw_record, location=f"history[{position}]")
        actual_keys = tuple(raw_record.keys())
        if set(actual_keys) != set(HISTORY_KEYS):
            missing = sorted(set(HISTORY_KEYS) - set(actual_keys))
            extra = sorted(set(actual_keys) - set(HISTORY_KEYS), key=str)
            raise CheckpointContractError(
                f"history record {position} must contain exactly "
                f"{list(HISTORY_KEYS)!r}; missing={missing!r}, extra={extra!r}"
            )

        epoch = _positive_epoch(raw_record["epoch"])
        if epoch <= previous_epoch:
            raise CheckpointContractError(
                "history epochs must be strictly increasing; "
                f"record {position} has epoch {epoch} after {previous_epoch}"
            )
        previous_epoch = epoch

        train = raw_record["train"]
        validation = raw_record["validation"]
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise CheckpointContractError(
                f"history record {position} train and validation fields must be mappings"
            )
        if "loss" not in train or "loss" not in validation:
            raise CheckpointContractError(
                f"history record {position} must include train.loss and validation.loss"
            )
        canonical.append(
            {
                "epoch": epoch,
                "train": {
                    **dict(train),
                    "loss": _finite_loss(
                        train["loss"], field=f"history[{position}].train.loss"
                    ),
                },
                "validation": {
                    **dict(validation),
                    "loss": _finite_loss(
                        validation["loss"],
                        field=f"history[{position}].validation.loss",
                    ),
                },
            }
        )
    return tuple(canonical)


def select_best_validation_epoch(
    history: Sequence[Mapping[str, Any]],
    *,
    criterion: str = VALIDATION_CRITERION,
    mode: str = VALIDATION_MODE,
) -> dict[str, Any]:
    """Select the earliest epoch attaining minimum validation loss."""

    if criterion != VALIDATION_CRITERION:
        raise CheckpointContractError(
            f"checkpoint criterion is locked to {VALIDATION_CRITERION!r}; "
            f"got {criterion!r}"
        )
    if mode != VALIDATION_MODE:
        raise CheckpointContractError(
            f"checkpoint mode is locked to {VALIDATION_MODE!r}; got {mode!r}"
        )
    canonical = validate_epoch_history(history)
    best_record = min(
        canonical,
        key=lambda record: (record["validation"]["loss"], record["epoch"]),
    )
    return {
        "epoch": int(best_record["epoch"]),
        "value": float(best_record["validation"]["loss"]),
        "criterion": VALIDATION_CRITERION,
        "mode": VALIDATION_MODE,
        "selection_split": SELECTION_SPLIT,
    }


def build_checkpoint_record(
    *,
    checkpoint_kind: str,
    model_state: Any,
    optimizer_state: Any,
    scheduler_state: Any,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    seed: int,
    baseline_variant: str,
    config_sha256: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Build a self-describing best/last checkpoint payload.

    Presence flags are derived from state values; callers cannot assert that a
    missing state exists.  Model and optimizer state are mandatory.  Scheduler
    state is explicit but may be unavailable when the locked trainer has no
    scheduler.
    """

    record = {
        "schema_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "model_state_present": model_state is not None,
        "optimizer_state_present": optimizer_state is not None,
        "scheduler_state_present": scheduler_state is not None,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "seed": seed,
        "baseline_variant": baseline_variant,
        "config_sha256": config_sha256,
        "experiment_id": experiment_id,
        "criterion": VALIDATION_CRITERION,
        "mode": VALIDATION_MODE,
        "selection_split": SELECTION_SPLIT,
    }
    return validate_checkpoint_record(record, expected_kind=checkpoint_kind)


def build_best_checkpoint(
    *,
    model_state: Any,
    optimizer_state: Any,
    scheduler_state: Any,
    epoch: int,
    best_validation_loss: float,
    seed: int,
    baseline_variant: str,
    config_sha256: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Build a checkpoint captured at the validation-selected best epoch."""

    return build_checkpoint_record(
        checkpoint_kind="best",
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        epoch=epoch,
        best_epoch=epoch,
        best_validation_loss=best_validation_loss,
        seed=seed,
        baseline_variant=baseline_variant,
        config_sha256=config_sha256,
        experiment_id=experiment_id,
    )


def build_last_checkpoint(
    *,
    model_state: Any,
    optimizer_state: Any,
    scheduler_state: Any,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    seed: int,
    baseline_variant: str,
    config_sha256: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Build the final-epoch checkpoint while retaining best-validation metadata."""

    return build_checkpoint_record(
        checkpoint_kind="last",
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        epoch=epoch,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        seed=seed,
        baseline_variant=baseline_variant,
        config_sha256=config_sha256,
        experiment_id=experiment_id,
    )


def validate_checkpoint_record(
    checkpoint: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Validate a best/last payload and return a canonical shallow copy."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    required = {
        "schema_version",
        "checkpoint_kind",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "model_state_present",
        "optimizer_state_present",
        "scheduler_state_present",
        "epoch",
        "best_epoch",
        "best_validation_loss",
        "seed",
        "baseline_variant",
        "config_sha256",
        "experiment_id",
        "criterion",
        "mode",
        "selection_split",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise CheckpointContractError(
            f"checkpoint is missing required fields: {missing!r}"
        )
    if checkpoint["schema_version"] != 1:
        raise CheckpointContractError(
            f"unsupported checkpoint schema_version {checkpoint['schema_version']!r}"
        )

    kind = checkpoint["checkpoint_kind"]
    if kind not in CHECKPOINT_KINDS:
        raise CheckpointContractError(
            f"checkpoint_kind must be one of {list(CHECKPOINT_KINDS)!r}; got {kind!r}"
        )
    if expected_kind is not None and expected_kind not in CHECKPOINT_KINDS:
        raise CheckpointContractError(
            f"expected_kind must be one of {list(CHECKPOINT_KINDS)!r}"
        )
    if expected_kind is not None and kind != expected_kind:
        raise CheckpointContractError(
            f"expected a {expected_kind!r} checkpoint, got {kind!r}"
        )

    if checkpoint["criterion"] != VALIDATION_CRITERION:
        raise CheckpointContractError(
            f"criterion must be {VALIDATION_CRITERION!r}"
        )
    if checkpoint["mode"] != VALIDATION_MODE:
        raise CheckpointContractError(f"mode must be {VALIDATION_MODE!r}")
    if checkpoint["selection_split"] != SELECTION_SPLIT:
        raise CheckpointContractError(
            f"selection_split must be {SELECTION_SPLIT!r}; test selection is forbidden"
        )

    epoch = _positive_epoch(checkpoint["epoch"])
    best_epoch = _positive_epoch(checkpoint["best_epoch"], field="best_epoch")
    if kind == "best" and epoch != best_epoch:
        raise CheckpointContractError(
            "best checkpoint epoch must equal best_epoch"
        )
    if kind == "last" and epoch < best_epoch:
        raise CheckpointContractError(
            "last checkpoint epoch cannot precede best_epoch"
        )

    for state_name in ("model", "optimizer", "scheduler"):
        flag_name = f"{state_name}_state_present"
        state_key = f"{state_name}_state"
        flag = checkpoint[flag_name]
        if not isinstance(flag, bool):
            raise CheckpointContractError(f"{flag_name} must be boolean")
        actual = checkpoint[state_key] is not None
        if flag != actual:
            raise CheckpointContractError(
                f"{flag_name}={flag!r} disagrees with {state_key} presence"
            )
    if not checkpoint["model_state_present"]:
        raise CheckpointContractError("model_state is required in every checkpoint")
    if not checkpoint["optimizer_state_present"]:
        raise CheckpointContractError("optimizer_state is required in every checkpoint")

    canonical = dict(checkpoint)
    canonical.update(
        {
            "checkpoint_kind": kind,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": _finite_loss(
                checkpoint["best_validation_loss"], field="best_validation_loss"
            ),
            "seed": _seed(checkpoint["seed"]),
            "baseline_variant": _variant(checkpoint["baseline_variant"]),
            "config_sha256": _config_sha256(checkpoint["config_sha256"]),
            "experiment_id": _required_text(
                checkpoint["experiment_id"], field="experiment_id"
            ),
        }
    )
    return canonical


def validate_checkpoint_pair(
    best_checkpoint: Mapping[str, Any],
    last_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ensure best and last payloads describe the same experiment."""

    best = validate_checkpoint_record(best_checkpoint, expected_kind="best")
    last = validate_checkpoint_record(last_checkpoint, expected_kind="last")
    for field in (
        "experiment_id",
        "seed",
        "baseline_variant",
        "config_sha256",
        "best_epoch",
        "best_validation_loss",
        "criterion",
        "mode",
        "selection_split",
    ):
        if best[field] != last[field]:
            raise CheckpointContractError(
                f"best/last checkpoint mismatch for {field}: "
                f"{best[field]!r} != {last[field]!r}"
            )
    return best, last


__all__ = [
    "BASELINE_VARIANTS",
    "CHECKPOINT_KINDS",
    "HISTORY_KEYS",
    "SELECTION_SPLIT",
    "VALIDATION_CRITERION",
    "VALIDATION_MODE",
    "CheckpointContractError",
    "build_best_checkpoint",
    "build_checkpoint_record",
    "build_history_record",
    "build_last_checkpoint",
    "select_best_validation_epoch",
    "validate_checkpoint_pair",
    "validate_checkpoint_record",
    "validate_epoch_history",
]
