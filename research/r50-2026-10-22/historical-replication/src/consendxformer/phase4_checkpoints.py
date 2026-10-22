"""Persisted-authoritative checkpoint contracts for Phase 4 single runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .baseline_evaluation import CLASS_ORDER
from .phase3_checkpoints import select_best_validation_epoch, validate_epoch_history
from .phase4_protocol import ATTENTION_CANDIDATES, ORDINAL_CANDIDATES, RUN_MODES


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_KINDS = ("best", "last")


class Phase4CheckpointError(ValueError):
    """Raised when a Phase 4 checkpoint is incomplete or inconsistent."""


@dataclass(frozen=True)
class Phase4CheckpointContext:
    experiment_id: str
    run_mode: str
    seed: int
    lambda_ordinal: float
    lambda_attention: float
    project_commit: str
    config_sha256: str
    manifest_bundle_sha256: str
    phase3_input_fingerprint: Mapping[str, Any]
    supervision_sha256: str
    ordered_supervision_sha256: str
    input_lock_sha256: str
    cohort_view: str
    expected_resolution: int
    sample_count: int
    split_counts: Mapping[str, int]

    @classmethod
    def from_execution_context(cls, context: Any) -> "Phase4CheckpointContext":
        return cls(
            **{
                field: getattr(context, field)
                for field in cls.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase3_input_fingerprint"] = dict(self.phase3_input_fingerprint)
        value["split_counts"] = dict(self.split_counts)
        return value


def _finite(value: Any, role: str) -> float:
    if isinstance(value, bool):
        raise Phase4CheckpointError(f"{role} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase4CheckpointError(f"{role} must be finite numeric") from exc
    if not math.isfinite(number):
        raise Phase4CheckpointError(f"{role} must be finite numeric")
    return number


def _epoch(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase4CheckpointError(f"{role} must be a positive integer")
    return value


def _validate_context(context: Phase4CheckpointContext) -> dict[str, Any]:
    if not isinstance(context, Phase4CheckpointContext):
        raise TypeError("context must be a Phase4CheckpointContext")
    values = context.to_dict()
    if not isinstance(context.experiment_id, str) or not context.experiment_id.strip():
        raise Phase4CheckpointError("experiment_id must be non-empty")
    if context.run_mode not in RUN_MODES:
        raise Phase4CheckpointError("run_mode is invalid")
    if isinstance(context.seed, bool) or not isinstance(context.seed, int):
        raise Phase4CheckpointError("seed must be an integer")
    if not 0 <= context.seed <= 2**32 - 1:
        raise Phase4CheckpointError("seed is outside the unsigned 32-bit range")
    if isinstance(context.lambda_ordinal, bool) or not isinstance(
        context.lambda_ordinal, (int, float)
    ) or float(context.lambda_ordinal) not in ORDINAL_CANDIDATES:
        raise Phase4CheckpointError("lambda_ordinal is not a locked candidate")
    if isinstance(context.lambda_attention, bool) or not isinstance(
        context.lambda_attention, (int, float)
    ) or float(context.lambda_attention) not in ATTENTION_CANDIDATES:
        raise Phase4CheckpointError("lambda_attention is not a locked candidate")
    if _COMMIT_RE.fullmatch(context.project_commit) is None:
        raise Phase4CheckpointError("project_commit must be an exact Git commit")
    for role in (
        "config_sha256",
        "manifest_bundle_sha256",
        "supervision_sha256",
        "ordered_supervision_sha256",
        "input_lock_sha256",
    ):
        if _SHA256_RE.fullmatch(str(values[role])) is None:
            raise Phase4CheckpointError(f"{role} must be a lowercase SHA-256 digest")
    if not isinstance(context.phase3_input_fingerprint, Mapping) or not dict(
        context.phase3_input_fingerprint
    ):
        raise Phase4CheckpointError("phase3_input_fingerprint must be a non-empty mapping")
    if not isinstance(context.cohort_view, str) or not context.cohort_view.strip():
        raise Phase4CheckpointError("cohort_view must be non-empty")
    if context.expected_resolution != 64:
        raise Phase4CheckpointError("expected_resolution must be the locked 64")
    if isinstance(context.sample_count, bool) or not isinstance(
        context.sample_count, int
    ) or context.sample_count <= 0:
        raise Phase4CheckpointError("sample_count must be a positive integer")
    if not isinstance(context.split_counts, Mapping):
        raise Phase4CheckpointError("split_counts must be a mapping")
    canonical_counts = {
        str(key): int(value) for key, value in context.split_counts.items()
    }
    if set(canonical_counts) != {"train", "val", "test"} or any(
        value <= 0 for value in canonical_counts.values()
    ) or sum(canonical_counts.values()) != context.sample_count:
        raise Phase4CheckpointError("split_counts are inconsistent with sample_count")
    values["split_counts"] = canonical_counts
    return values


def select_phase4_best_epoch(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select minimum validation total loss and reject ambiguous loss aliases."""

    canonical = validate_epoch_history(history)
    for record in canonical:
        validation = record["validation"]
        if "total_loss" not in validation or _finite(
            validation["total_loss"], "validation.total_loss"
        ) != float(validation["loss"]):
            raise Phase4CheckpointError(
                "validation.loss must explicitly equal validation.total_loss"
            )
    return select_best_validation_epoch(canonical)


def build_phase4_checkpoint(
    *,
    checkpoint_kind: str,
    context: Phase4CheckpointContext,
    model_state: Any,
    optimizer_state: Any,
    scheduler_state: Any,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint_kind not in _KINDS:
        raise Phase4CheckpointError(f"checkpoint_kind must be one of {_KINDS!r}")
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
        "selection_criterion": "validation_loss",
        "selection_metric": "total_loss",
        "selection_mode": "min",
        "selection_split": "val",
        "validation_metrics": dict(validation_metrics),
        "class_order": list(CLASS_ORDER),
        **_validate_context(context),
    }
    return validate_phase4_checkpoint(record, expected_kind=checkpoint_kind)


def validate_phase4_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    context: Phase4CheckpointContext | None = None,
) -> dict[str, Any]:
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
        "selection_criterion",
        "selection_metric",
        "selection_mode",
        "selection_split",
        "validation_metrics",
        "class_order",
        *Phase4CheckpointContext.__dataclass_fields__,
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise Phase4CheckpointError(f"checkpoint is missing fields: {missing}")
    if checkpoint["schema_version"] != 1:
        raise Phase4CheckpointError("unsupported Phase 4 checkpoint schema")
    kind = checkpoint["checkpoint_kind"]
    if kind not in _KINDS or (expected_kind is not None and kind != expected_kind):
        raise Phase4CheckpointError(
            f"checkpoint kind {kind!r} does not match expected {expected_kind!r}"
        )
    epoch = _epoch(checkpoint["epoch"], "epoch")
    best_epoch = _epoch(checkpoint["best_epoch"], "best_epoch")
    if kind == "best" and epoch != best_epoch:
        raise Phase4CheckpointError("best checkpoint epoch must equal best_epoch")
    if kind == "last" and epoch < best_epoch:
        raise Phase4CheckpointError("last checkpoint cannot precede best_epoch")
    for state in ("model", "optimizer", "scheduler"):
        present = checkpoint[f"{state}_state_present"]
        if type(present) is not bool or present != (
            checkpoint[f"{state}_state"] is not None
        ):
            raise Phase4CheckpointError(f"{state} state presence flag is inconsistent")
    if not checkpoint["model_state_present"] or not checkpoint["optimizer_state_present"]:
        raise Phase4CheckpointError("model and optimizer state are mandatory")
    if (
        checkpoint["selection_criterion"] != "validation_loss"
        or checkpoint["selection_metric"] != "total_loss"
        or checkpoint["selection_mode"] != "min"
        or checkpoint["selection_split"] != "val"
    ):
        raise Phase4CheckpointError(
            "checkpoint selection must be minimum validation total loss"
        )
    if checkpoint["class_order"] != list(CLASS_ORDER):
        raise Phase4CheckpointError("checkpoint class order is invalid")
    validation_metrics = checkpoint["validation_metrics"]
    if not isinstance(validation_metrics, Mapping) or "loss" not in validation_metrics:
        raise Phase4CheckpointError("checkpoint validation_metrics must contain loss")
    loss = _finite(validation_metrics["loss"], "validation_metrics.loss")
    if "total_loss" in validation_metrics and _finite(
        validation_metrics["total_loss"], "validation_metrics.total_loss"
    ) != loss:
        raise Phase4CheckpointError("validation loss and total_loss disagree")

    embedded_context = Phase4CheckpointContext(
        **{
            field: checkpoint[field]
            for field in Phase4CheckpointContext.__dataclass_fields__
        }
    )
    canonical_context = _validate_context(embedded_context)
    if context is not None and canonical_context != _validate_context(context):
        changed = sorted(
            field
            for field in canonical_context
            if canonical_context[field] != _validate_context(context)[field]
        )
        raise Phase4CheckpointError(
            f"checkpoint provenance differs from execution context: {changed}"
        )
    result = dict(checkpoint)
    result.update(canonical_context)
    result.update(
        {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": _finite(
                checkpoint["best_validation_loss"], "best_validation_loss"
            ),
            "validation_metrics": {**dict(validation_metrics), "loss": loss},
        }
    )
    return result


def save_phase4_checkpoint(
    checkpoint: Mapping[str, Any], destination: str | Path
) -> Path:
    """Atomically serialize one checkpoint; the file becomes authoritative."""

    import torch

    payload = validate_phase4_checkpoint(checkpoint)
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def _load(path: Path, map_location: str) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise Phase4CheckpointError(f"Could not load persisted checkpoint: {path}") from exc
    if not isinstance(value, Mapping):
        raise Phase4CheckpointError(f"Persisted checkpoint is not a mapping: {path}")
    return value


def load_and_validate_phase4_checkpoints(
    best_path: str | Path,
    last_path: str | Path,
    *,
    context: Phase4CheckpointContext,
    history: Sequence[Mapping[str, Any]],
    map_location: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reload authoritative best/last files and reconcile them with history."""

    try:
        canonical_history = validate_epoch_history(history)
        selection = select_phase4_best_epoch(canonical_history)
        best = validate_phase4_checkpoint(
            _load(Path(best_path).resolve(), map_location),
            expected_kind="best",
            context=context,
        )
        last = validate_phase4_checkpoint(
            _load(Path(last_path).resolve(), map_location),
            expected_kind="last",
            context=context,
        )
    except Phase4CheckpointError:
        raise
    except (TypeError, ValueError) as exc:
        raise Phase4CheckpointError(
            "Persisted Phase 4 checkpoints disagree with training history"
        ) from exc

    selected_epoch = int(selection["epoch"])
    selected_loss = float(selection["value"])
    final_epoch = int(canonical_history[-1]["epoch"])
    if best["epoch"] != selected_epoch or best["best_epoch"] != selected_epoch:
        raise Phase4CheckpointError("persisted best epoch disagrees with selection")
    if last["epoch"] != final_epoch or last["best_epoch"] != selected_epoch:
        raise Phase4CheckpointError("persisted last epoch disagrees with history")
    if best["best_validation_loss"] != selected_loss or last[
        "best_validation_loss"
    ] != selected_loss:
        raise Phase4CheckpointError("persisted best loss disagrees with selection")
    by_epoch = {int(record["epoch"]): record for record in canonical_history}
    for kind, checkpoint in (("best", best), ("last", last)):
        expected_loss = float(
            by_epoch[int(checkpoint["epoch"])]["validation"]["loss"]
        )
        if float(checkpoint["validation_metrics"]["loss"]) != expected_loss:
            raise Phase4CheckpointError(
                f"persisted {kind} validation metrics disagree with history"
            )
    pair_fields = (
        *Phase4CheckpointContext.__dataclass_fields__,
        "best_epoch",
        "best_validation_loss",
        "selection_criterion",
        "selection_metric",
        "selection_mode",
        "selection_split",
        "class_order",
    )
    for field in pair_fields:
        if best[field] != last[field]:
            raise Phase4CheckpointError(f"best/last checkpoint mismatch for {field}")
    return best, last


__all__ = [
    "Phase4CheckpointContext",
    "Phase4CheckpointError",
    "build_phase4_checkpoint",
    "load_and_validate_phase4_checkpoints",
    "save_phase4_checkpoint",
    "select_phase4_best_epoch",
    "validate_phase4_checkpoint",
]
