"""Evaluation utilities for three-class LungDxFormer prediction exports.

The confusion-matrix convention is rows=true class and columns=predicted class.
Class indices are fixed for the lifetime of the project:

0. benign
1. indeterminate
2. malignant

This module intentionally depends only on NumPy and pandas.  In particular, it
does not rely on scikit-learn being installed in a baseline execution
environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_ORDER: tuple[str, str, str] = ("benign", "indeterminate", "malignant")
"""Canonical label/index order used by every reported metric."""

_IDENTIFIER_CANDIDATES = (
    "identifier",
    "sample_id",
    "patient_id",
    "nodule_id",
    "id",
)
_TRUE_LABEL_CANDIDATES = ("true_label", "label", "y_true")
_PREDICTED_CLASS_CANDIDATES = (
    "predicted_class",
    "predicted_label",
    "prediction",
    "y_pred",
)

_PROBABILITY_COLUMN_FAMILIES = (
    tuple(f"prob_{name}" for name in CLASS_ORDER),
    tuple(f"probability_{name}" for name in CLASS_ORDER),
    tuple(f"{name}_probability" for name in CLASS_ORDER),
    tuple(f"{name}_prob" for name in CLASS_ORDER),
)
_LOGIT_COLUMN_FAMILIES = (
    tuple(f"logit_{name}" for name in CLASS_ORDER),
    tuple(f"{name}_logit" for name in CLASS_ORDER),
)


def load_prediction_csv(path: str | Path) -> pd.DataFrame:
    """Read a prediction CSV while preserving identifier text such as ``001``."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction CSV does not exist: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Prediction CSV is empty: {csv_path}") from exc
    return frame


def _resolve_required_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(
                f"Requested {role} column {requested!r} is absent. "
                f"Available columns: {list(frame.columns)!r}"
            )
        return requested

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not find a {role} column. Expected one of {list(candidates)!r}; "
        f"available columns: {list(frame.columns)!r}"
    )


def _ordered_score_columns(
    columns: Mapping[str, str] | Sequence[str] | None,
    kind: str,
) -> tuple[str, str, str] | None:
    if columns is None:
        return None
    if isinstance(columns, Mapping):
        supplied = set(columns)
        expected = set(CLASS_ORDER)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"{kind} column mapping must have exactly the class keys "
                f"{list(CLASS_ORDER)!r}; missing={missing!r}, extra={extra!r}"
            )
        ordered = tuple(str(columns[name]) for name in CLASS_ORDER)
    else:
        if isinstance(columns, (str, bytes)):
            raise TypeError(
                f"{kind} columns must be a three-item sequence in CLASS_ORDER, not a string"
            )
        ordered = tuple(str(column) for column in columns)
        if len(ordered) != len(CLASS_ORDER):
            raise ValueError(
                f"{kind} columns must contain exactly {len(CLASS_ORDER)} names in "
                f"{list(CLASS_ORDER)!r} order; got {len(ordered)}"
            )
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{kind} column names must be distinct; got {ordered!r}")
    return ordered  # type: ignore[return-value]


def _auto_detect_score_columns(
    frame: pd.DataFrame,
) -> tuple[str | None, tuple[str, str, str] | None]:
    complete: list[tuple[str, tuple[str, str, str]]] = []
    partial: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    for kind, families in (
        ("probabilities", _PROBABILITY_COLUMN_FAMILIES),
        ("logits", _LOGIT_COLUMN_FAMILIES),
    ):
        for family in families:
            present = tuple(column for column in family if column in frame.columns)
            if len(present) == len(CLASS_ORDER):
                complete.append((kind, family))
            elif present:
                missing = tuple(column for column in family if column not in frame.columns)
                partial.append((kind, present, missing))

    if len(complete) > 1:
        descriptions = [f"{kind}: {list(columns)!r}" for kind, columns in complete]
        raise ValueError(
            "Multiple complete probability/logit column sets were detected; select exactly "
            "one explicitly. Detected " + "; ".join(descriptions)
        )
    if complete:
        return complete[0]
    if partial:
        details = "; ".join(
            f"{kind} present={list(present)!r}, missing={list(missing)!r}"
            for kind, present, missing in partial
        )
        raise ValueError(
            "Detected an incomplete class probability/logit column set. Scores must be "
            f"provided for every class in {list(CLASS_ORDER)!r}. {details}"
        )
    return None, None


def _resolve_score_columns(
    frame: pd.DataFrame,
    probability_columns: Mapping[str, str] | Sequence[str] | None,
    logit_columns: Mapping[str, str] | Sequence[str] | None,
) -> tuple[str | None, tuple[str, str, str] | None]:
    probability_columns = _ordered_score_columns(probability_columns, "Probability")
    logit_columns = _ordered_score_columns(logit_columns, "Logit")
    if probability_columns is not None and logit_columns is not None:
        raise ValueError("Select probability columns or logit columns, not both")

    if probability_columns is not None:
        kind, columns = "probabilities", probability_columns
    elif logit_columns is not None:
        kind, columns = "logits", logit_columns
    else:
        return _auto_detect_score_columns(frame)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"The selected {kind} columns are incomplete; missing columns: {missing!r}"
        )
    return kind, columns


def _label_index(value: Any) -> int:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        raise ValueError("missing value")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean labels are not valid class indices")
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < len(CLASS_ORDER):
            return index
        raise ValueError(f"class index {index} is outside 0..{len(CLASS_ORDER) - 1}")
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return _label_index(int(numeric))
        raise ValueError(f"numeric label {value!r} is not an integer class index")

    text = str(value).strip()
    if not text:
        raise ValueError("blank value")
    lowered = text.casefold()
    if lowered in CLASS_ORDER:
        return CLASS_ORDER.index(lowered)
    try:
        numeric = float(text)
    except ValueError:
        numeric = float("nan")
    if np.isfinite(numeric) and numeric.is_integer():
        return _label_index(int(numeric))
    raise ValueError(f"unknown class {value!r}")


def _encode_label_column(series: pd.Series, role: str) -> np.ndarray:
    encoded = np.empty(len(series), dtype=np.int64)
    for position, (row_index, value) in enumerate(series.items()):
        try:
            encoded[position] = _label_index(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {role} at CSV/DataFrame row {row_index!r}: {value!r} ({exc}). "
                f"Allowed labels are {list(CLASS_ORDER)!r} or indices 0, 1, 2 in that order."
            ) from exc
    return encoded


def _validate_identifiers(series: pd.Series, column: str) -> None:
    missing = series.isna()
    text = series.astype("string").str.strip()
    invalid = missing | text.eq("")
    if bool(invalid.any()):
        rows = series.index[invalid].tolist()[:5]
        raise ValueError(
            f"Identifier column {column!r} contains missing or blank values at rows {rows!r}"
        )


def _score_matrix(
    frame: pd.DataFrame,
    kind: str,
    columns: tuple[str, str, str],
) -> np.ndarray:
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    bad = ~np.isfinite(values)
    if bool(bad.any()):
        row, column = np.argwhere(bad)[0]
        original = frame.iloc[int(row)][columns[int(column)]]
        raise ValueError(
            f"{kind.capitalize()} column {columns[int(column)]!r} contains a missing, "
            f"non-numeric, or non-finite value {original!r} at row {frame.index[int(row)]!r}"
        )

    if kind == "probabilities":
        if bool(((values < 0.0) | (values > 1.0)).any()):
            row, column = np.argwhere((values < 0.0) | (values > 1.0))[0]
            raise ValueError(
                f"Probability {values[int(row), int(column)]!r} in column "
                f"{columns[int(column)]!r} at row {frame.index[int(row)]!r} is outside [0, 1]"
            )
        row_sums = values.sum(axis=1)
        bad_sums = ~np.isclose(row_sums, 1.0, rtol=1e-6, atol=1e-3)
        if bool(bad_sums.any()):
            row = int(np.flatnonzero(bad_sums)[0])
            raise ValueError(
                "Class probabilities must sum to 1 for every row (absolute tolerance "
                f"1e-3); row {frame.index[row]!r} sums to {float(row_sums[row])!r}"
            )
        return values

    # Stable softmax turns logits into comparable one-vs-rest class scores.
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _binary_auroc(target: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        return None

    # The Mann-Whitney form of AUROC is exact and average ranks handle ties.
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_positive = float(ranks[target.astype(bool)].sum())
    auc = (
        rank_sum_positive - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    # Guard only against insignificant floating point drift at the boundaries.
    return float(min(1.0, max(0.0, auc)))


def _classification_metrics(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
) -> tuple[dict[str, dict[str, float | int]], list[list[int]], dict[str, float]]:
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    np.add.at(confusion, (true_indices, predicted_indices), 1)

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for class_index, class_name in enumerate(CLASS_ORDER):
        true_positive = int(confusion[class_index, class_index])
        predicted_total = int(confusion[:, class_index].sum())
        support = int(confusion[class_index, :].sum())
        precision = float(true_positive / predicted_total) if predicted_total else 0.0
        recall = float(true_positive / support) if support else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
    }
    confusion_json = [[int(value) for value in row] for row in confusion.tolist()]
    return per_class, confusion_json, macro


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    identifier_column: str | None = None,
    true_label_column: str | None = None,
    predicted_class_column: str | None = None,
    probability_columns: Mapping[str, str] | Sequence[str] | None = None,
    logit_columns: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a prediction table and return JSON-safe fixed-order metrics.

    String class labels and integer indices 0/1/2 are accepted.  Probability or
    logit columns must be supplied for all three classes, either through one of
    the recognized naming families or explicitly in :data:`CLASS_ORDER` order.
    Precision, recall, and F1 use zero for a zero denominator.  AUROC instead
    uses ``None`` plus a warning whenever the required positive/negative labels
    are unavailable.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if predictions.empty:
        raise ValueError("Prediction table contains no rows; metrics are undefined")
    if not predictions.columns.is_unique:
        duplicates = predictions.columns[predictions.columns.duplicated()].tolist()
        raise ValueError(f"Prediction table has duplicate columns: {duplicates!r}")

    identifier_column = _resolve_required_column(
        predictions, identifier_column, _IDENTIFIER_CANDIDATES, "identifier"
    )
    true_label_column = _resolve_required_column(
        predictions, true_label_column, _TRUE_LABEL_CANDIDATES, "true-label"
    )
    predicted_class_column = _resolve_required_column(
        predictions,
        predicted_class_column,
        _PREDICTED_CLASS_CANDIDATES,
        "predicted-class",
    )
    required_columns = (identifier_column, true_label_column, predicted_class_column)
    if len(set(required_columns)) != len(required_columns):
        raise ValueError(
            "Identifier, true-label, and predicted-class columns must be distinct; "
            f"resolved columns: {required_columns!r}"
        )

    _validate_identifiers(predictions[identifier_column], identifier_column)
    true_indices = _encode_label_column(predictions[true_label_column], "true label")
    predicted_indices = _encode_label_column(
        predictions[predicted_class_column], "predicted class"
    )

    score_source, score_columns = _resolve_score_columns(
        predictions, probability_columns, logit_columns
    )
    class_scores = (
        _score_matrix(predictions, score_source, score_columns)
        if score_source is not None and score_columns is not None
        else None
    )

    per_class, confusion, macro = _classification_metrics(
        true_indices, predicted_indices
    )
    accuracy = float(np.mean(true_indices == predicted_indices))

    warnings: list[str] = []
    auroc_by_class: dict[str, float | None] = {}
    if class_scores is None:
        auroc_by_class.update({name: None for name in CLASS_ORDER})
        warnings.append(
            "One-vs-rest AUROC was not computed because no complete class probability "
            "or logit column set was supplied."
        )
    else:
        for class_index, class_name in enumerate(CLASS_ORDER):
            target = (true_indices == class_index).astype(np.int64)
            auc = _binary_auroc(target, class_scores[:, class_index])
            auroc_by_class[class_name] = auc
            if auc is None:
                positives = int(target.sum())
                negatives = int(len(target) - positives)
                warnings.append(
                    f"One-vs-rest AUROC is undefined for class {class_name!r}: "
                    f"positives={positives}, negatives={negatives}."
                )

    undefined_classes = [name for name in CLASS_ORDER if auroc_by_class[name] is None]
    if undefined_classes:
        macro_auroc: float | None = None
        if class_scores is not None:
            warnings.append(
                "Macro one-vs-rest AUROC is undefined because class AUROC is undefined "
                f"for {undefined_classes!r}."
            )
    else:
        macro_auroc = float(
            np.mean([auroc_by_class[name] for name in CLASS_ORDER], dtype=np.float64)
        )

    score_column_record = (
        {name: score_columns[index] for index, name in enumerate(CLASS_ORDER)}
        if score_columns is not None
        else None
    )
    return {
        "schema_version": 1,
        "class_order": list(CLASS_ORDER),
        "n_samples": int(len(predictions)),
        "identifier_column": identifier_column,
        "true_label_column": true_label_column,
        "predicted_class_column": predicted_class_column,
        "score_source": score_source,
        "score_columns": score_column_record,
        "accuracy": accuracy,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "per_class": per_class,
        "confusion_matrix": confusion,
        "auroc_ovr": {
            **auroc_by_class,
            "macro": macro_auroc,
        },
        "warnings": warnings,
    }


def evaluate_prediction_csv(
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read and evaluate a CSV prediction export."""

    return evaluate_predictions(load_prediction_csv(path), **kwargs)


# A plural alias reads naturally at call sites and preserves one canonical
# implementation.
evaluate_predictions_csv = evaluate_prediction_csv


def write_evaluation_json(metrics: Mapping[str, Any], path: str | Path) -> Path:
    """Write metrics as strict JSON, rejecting NaN and Infinity."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return output_path
