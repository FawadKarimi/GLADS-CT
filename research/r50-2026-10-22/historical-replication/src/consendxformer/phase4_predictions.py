"""Strict Phase 4 validation/final prediction artifact contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER, evaluate_predictions
from .phase3_artifacts import atomic_write_csv
from .phase3_predictions import stable_softmax
from .phase4_protocol import ATTENTION_CANDIDATES, ORDINAL_CANDIDATES, RUN_MODES


LOGIT_COLUMNS: tuple[str, str, str] = tuple(
    f"logit_{name}" for name in CLASS_ORDER
)  # type: ignore[assignment]
PROBABILITY_COLUMNS: tuple[str, str, str] = tuple(
    f"prob_{name}" for name in CLASS_ORDER
)  # type: ignore[assignment]
PHASE4_PREDICTION_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "nodule_id",
    "series_instance_uid",
    "split",
    "true_class",
    "predicted_class",
    *LOGIT_COLUMNS,
    *PROBABILITY_COLUMNS,
    "seed",
    "lambda_ordinal",
    "lambda_attention",
    "run_mode",
    "experiment_id",
)


class Phase4PredictionError(ValueError):
    """Raised when a Phase 4 prediction table violates its fixed schema."""


def _text(value: Any, field: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise Phase4PredictionError(f"{field} must be non-empty")
    return str(value).strip()


def _lambda(value: Any, candidates: tuple[float, ...], field: str) -> float:
    if isinstance(value, bool):
        raise Phase4PredictionError(f"{field} must be a locked numeric candidate")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase4PredictionError(f"{field} must be numeric") from exc
    if not np.isfinite(number) or number not in candidates:
        raise Phase4PredictionError(
            f"{field} must be one of {list(candidates)!r}; got {value!r}"
        )
    return number


def build_phase4_prediction_table(
    metadata: pd.DataFrame,
    logits: Any,
    *,
    seed: int,
    lambda_ordinal: float,
    lambda_attention: float,
    run_mode: str,
    experiment_id: str,
) -> pd.DataFrame:
    """Join aligned logits to authoritative identities without changing row order."""

    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise Phase4PredictionError("metadata must be a non-empty pandas DataFrame")
    required = {
        "patient_id",
        "nodule_id",
        "series_instance_uid",
        "split",
        "label",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise Phase4PredictionError(f"metadata is missing required columns: {missing}")
    values = np.asarray(logits, dtype=np.float64)
    probabilities = stable_softmax(values)
    if len(metadata) != len(values):
        raise Phase4PredictionError(
            f"metadata/logit row-count mismatch: {len(metadata)} != {len(values)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise Phase4PredictionError("seed must be an integer")
    canonical_seed = int(seed)
    if not 0 <= canonical_seed <= 2**32 - 1:
        raise Phase4PredictionError("seed is outside the unsigned 32-bit range")
    if run_mode not in RUN_MODES:
        raise Phase4PredictionError(f"run_mode must be one of {list(RUN_MODES)!r}")
    ordinal = _lambda(lambda_ordinal, ORDINAL_CANDIDATES, "lambda_ordinal")
    attention = _lambda(lambda_attention, ATTENTION_CANDIDATES, "lambda_attention")
    identity = _text(experiment_id, "experiment_id")

    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(metadata.iterrows()):
        split = _text(row["split"], "split")
        if split not in {"val", "test"}:
            raise Phase4PredictionError("Prediction exports may contain only val or test")
        if run_mode == "tuning" and split != "val":
            raise Phase4PredictionError("Tuning prediction exports are validation-only")
        true_class = _text(row["label"], "true_class").lower()
        if true_class not in CLASS_ORDER:
            raise Phase4PredictionError(f"Noncanonical true class: {true_class!r}")
        predicted = CLASS_ORDER[int(np.argmax(probabilities[position]))]
        record: dict[str, Any] = {
            "patient_id": _text(row["patient_id"], "patient_id"),
            "nodule_id": _text(row["nodule_id"], "nodule_id"),
            "series_instance_uid": _text(
                row["series_instance_uid"], "series_instance_uid"
            ),
            "split": split,
            "true_class": true_class,
            "predicted_class": predicted,
            "seed": canonical_seed,
            "lambda_ordinal": ordinal,
            "lambda_attention": attention,
            "run_mode": run_mode,
            "experiment_id": identity,
        }
        record.update(
            {
                column: float(values[position, index])
                for index, column in enumerate(LOGIT_COLUMNS)
            }
        )
        record.update(
            {
                column: float(probabilities[position, index])
                for index, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        rows.append(record)
    return validate_phase4_prediction_table(
        pd.DataFrame(rows, columns=PHASE4_PREDICTION_COLUMNS)
    )


def validate_phase4_prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise Phase4PredictionError("prediction table must be non-empty")
    if tuple(str(column) for column in frame.columns) != PHASE4_PREDICTION_COLUMNS:
        raise Phase4PredictionError(
            "prediction columns must exactly match PHASE4_PREDICTION_COLUMNS in order"
        )
    result = frame.copy(deep=True)
    for column in ("patient_id", "nodule_id", "series_instance_uid", "experiment_id"):
        result[column] = [_text(value, column) for value in result[column]]
    if result["nodule_id"].duplicated().any():
        raise Phase4PredictionError("nodule_id must be unique in a prediction table")
    if not result["split"].isin(("val", "test")).all():
        raise Phase4PredictionError("prediction split must be val or test")
    if result["split"].nunique() != 1:
        raise Phase4PredictionError("each prediction artifact must contain exactly one split")
    if not result["true_class"].isin(CLASS_ORDER).all() or not result[
        "predicted_class"
    ].isin(CLASS_ORDER).all():
        raise Phase4PredictionError("prediction labels must use the fixed class order")

    numeric = result.loc[:, [*LOGIT_COLUMNS, *PROBABILITY_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise Phase4PredictionError("prediction scores must be finite numbers")
    logits = numeric.loc[:, list(LOGIT_COLUMNS)].to_numpy(dtype=np.float64)
    probabilities = numeric.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    expected = stable_softmax(logits)
    if not np.allclose(probabilities, expected, rtol=1e-12, atol=1e-15):
        raise Phase4PredictionError("stored probabilities do not equal logit softmax")
    predicted = np.asarray(CLASS_ORDER, dtype=object)[np.argmax(logits, axis=1)]
    if not np.array_equal(predicted, result["predicted_class"].to_numpy(dtype=object)):
        raise Phase4PredictionError("predicted_class does not equal logit argmax")
    result.loc[:, [*LOGIT_COLUMNS, *PROBABILITY_COLUMNS]] = numeric

    for field, candidates in (
        ("lambda_ordinal", ORDINAL_CANDIDATES),
        ("lambda_attention", ATTENTION_CANDIDATES),
    ):
        result[field] = [_lambda(value, candidates, field) for value in result[field]]
        if result[field].nunique(dropna=False) != 1:
            raise Phase4PredictionError(f"{field} must be constant within an artifact")
    seeds: list[int] = []
    for value in result["seed"]:
        try:
            numeric_seed = float(value)
        except (TypeError, ValueError) as exc:
            raise Phase4PredictionError("seed must be an integer") from exc
        if not numeric_seed.is_integer() or not 0 <= numeric_seed <= 2**32 - 1:
            raise Phase4PredictionError("seed must be an unsigned 32-bit integer")
        seeds.append(int(numeric_seed))
    result["seed"] = seeds
    if result["seed"].nunique() != 1:
        raise Phase4PredictionError("seed must be constant within an artifact")
    if result["experiment_id"].nunique() != 1:
        raise Phase4PredictionError("experiment_id must be constant within an artifact")
    if not result["run_mode"].isin(RUN_MODES).all() or result["run_mode"].nunique() != 1:
        raise Phase4PredictionError("run_mode must be one constant supported mode")
    if result["run_mode"].iloc[0] == "tuning" and result["split"].iloc[0] != "val":
        raise Phase4PredictionError("Tuning prediction artifacts must be validation-only")
    return result.loc[:, list(PHASE4_PREDICTION_COLUMNS)]


def evaluate_phase4_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    validated = validate_phase4_prediction_table(frame)
    return evaluate_predictions(
        validated,
        identifier_column="nodule_id",
        true_label_column="true_class",
        predicted_class_column="predicted_class",
        probability_columns=PROBABILITY_COLUMNS,
    )


def write_phase4_prediction_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    return atomic_write_csv(path, validate_phase4_prediction_table(frame))


def load_phase4_prediction_csv(path: str | Path) -> pd.DataFrame:
    return validate_phase4_prediction_table(pd.read_csv(path, dtype=str))


__all__ = [
    "LOGIT_COLUMNS",
    "PHASE4_PREDICTION_COLUMNS",
    "PROBABILITY_COLUMNS",
    "Phase4PredictionError",
    "build_phase4_prediction_table",
    "evaluate_phase4_predictions",
    "load_phase4_prediction_csv",
    "validate_phase4_prediction_table",
    "write_phase4_prediction_csv",
]
