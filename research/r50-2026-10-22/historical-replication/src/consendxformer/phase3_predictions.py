"""Deterministic Phase 3 prediction-table contracts.

The Phase 3 baselines emit both logits and probabilities for auditability.  A
consumer must therefore select the probability family explicitly when invoking
the shared evaluator; automatic score-column detection is intentionally
ambiguous for this schema.
"""

from __future__ import annotations

from collections.abc import Mapping
import io
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER, evaluate_predictions
from .hashing import sha256_json


IDENTITY_COLUMNS: tuple[str, str, str] = (
    "patient_id",
    "series_instance_uid",
    "nodule_id",
)
PROBABILITY_COLUMNS: tuple[str, str, str] = tuple(
    f"prob_{class_name}" for class_name in CLASS_ORDER
)  # type: ignore[assignment]
LOGIT_COLUMNS: tuple[str, str, str] = tuple(
    f"logit_{class_name}" for class_name in CLASS_ORDER
)  # type: ignore[assignment]
PREDICTION_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "series_instance_uid",
    "nodule_id",
    "split",
    "true_class",
    "predicted_class",
    *PROBABILITY_COLUMNS,
    *LOGIT_COLUMNS,
    "seed",
    "baseline_variant",
    "experiment_id",
)

_ALLOWED_SPLITS = ("val", "test")
_SPLIT_RANK = {name: index for index, name in enumerate(_ALLOWED_SPLITS)}
_VARIANTS = ("B0", "B1")
_INTEGER_TEXT = re.compile(r"0|[1-9][0-9]*")


class PredictionContractError(ValueError):
    """Raised when a Phase 3 prediction artifact violates its fixed contract."""


def stable_softmax(logits: Any) -> np.ndarray:
    """Return row-wise float64 softmax values for an ``N x 3`` logit matrix."""

    try:
        values = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictionContractError("logits must be a numeric N x 3 matrix") from exc
    if values.ndim != 2 or values.shape[1] != len(CLASS_ORDER):
        raise PredictionContractError(
            f"logits must have shape (N, {len(CLASS_ORDER)}); got {values.shape!r}"
        )
    if values.shape[0] == 0:
        raise PredictionContractError("logits must contain at least one row")
    if not bool(np.isfinite(values).all()):
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise PredictionContractError(
            "logits contain a non-finite value at "
            f"row {int(row)}, class {CLASS_ORDER[int(column)]!r}"
        )

    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _required_text(value: Any, *, field: str, row: Any | None = None) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        where = f" at row {row!r}" if row is not None else ""
        raise PredictionContractError(f"{field} is missing{where}")
    text = str(value).strip()
    if not text:
        where = f" at row {row!r}" if row is not None else ""
        raise PredictionContractError(f"{field} is blank{where}")
    return text


def _canonical_class(value: Any, *, field: str, row: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        raise PredictionContractError(f"{field} is missing at row {row!r}")
    if isinstance(value, (bool, np.bool_)):
        raise PredictionContractError(f"{field} cannot be boolean at row {row!r}")
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < len(CLASS_ORDER):
            return CLASS_ORDER[index]
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return _canonical_class(int(numeric), field=field, row=row)
    else:
        text = str(value).strip().casefold()
        if text in CLASS_ORDER:
            return text
        if _INTEGER_TEXT.fullmatch(text):
            return _canonical_class(int(text), field=field, row=row)
    raise PredictionContractError(
        f"{field} has invalid class {value!r} at row {row!r}; expected "
        f"one of {list(CLASS_ORDER)!r} or indices 0, 1, 2"
    )


def _canonical_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise PredictionContractError("seed must be a non-negative integer, not boolean")
    if isinstance(value, (int, np.integer)):
        seed = int(value)
    elif isinstance(value, str) and _INTEGER_TEXT.fullmatch(value.strip()):
        seed = int(value.strip())
    else:
        raise PredictionContractError(f"seed must be a non-negative integer; got {value!r}")
    if not 0 <= seed <= 2**32 - 1:
        raise PredictionContractError(
            f"seed must be in the range 0..{2**32 - 1}; got {seed}"
        )
    return seed


def _resolve_true_class_column(
    metadata: pd.DataFrame,
    requested: str | None,
) -> str:
    if requested is not None:
        if requested not in metadata.columns:
            raise PredictionContractError(
                f"true-class column {requested!r} is absent from metadata"
            )
        return requested
    for candidate in ("true_class", "label"):
        if candidate in metadata.columns:
            return candidate
    raise PredictionContractError(
        "metadata must contain a true-class column named 'true_class' or 'label', "
        "or true_class_column must be supplied explicitly"
    )


def _deterministic_sort(frame: pd.DataFrame) -> pd.DataFrame:
    sortable = frame.assign(_split_rank=frame["split"].map(_SPLIT_RANK))
    sortable = sortable.sort_values(
        ["_split_rank", *IDENTITY_COLUMNS],
        kind="mergesort",
        ignore_index=True,
    )
    return sortable.drop(columns="_split_rank").loc[:, list(PREDICTION_COLUMNS)]


def build_prediction_table(
    metadata: pd.DataFrame,
    logits: Any,
    *,
    seed: int,
    baseline_variant: str,
    experiment_id: str,
    true_class_column: str | None = None,
) -> pd.DataFrame:
    """Build and validate the fixed Phase 3 table from aligned metadata/logits.

    ``metadata`` row order is the model-inference order.  The returned artifact
    is sorted deterministically only after each logit row has been joined to its
    three-part identity.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if metadata.empty:
        raise PredictionContractError("metadata must contain at least one row")
    if not metadata.columns.is_unique:
        duplicates = metadata.columns[metadata.columns.duplicated()].tolist()
        raise PredictionContractError(f"metadata has duplicate columns: {duplicates!r}")

    true_column = _resolve_true_class_column(metadata, true_class_column)
    required = {*IDENTITY_COLUMNS, "split", true_column}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise PredictionContractError(f"metadata is missing required columns: {missing!r}")

    values = np.asarray(logits, dtype=np.float64)
    probabilities = stable_softmax(values)
    if len(metadata) != values.shape[0]:
        raise PredictionContractError(
            "metadata/logit row-count mismatch: "
            f"metadata={len(metadata)}, logits={values.shape[0]}"
        )

    canonical_seed = _canonical_seed(seed)
    variant = _required_text(baseline_variant, field="baseline_variant")
    if variant not in _VARIANTS:
        raise PredictionContractError(
            f"baseline_variant must be one of {list(_VARIANTS)!r}; got {variant!r}"
        )
    canonical_experiment_id = _required_text(experiment_id, field="experiment_id")

    rows: list[dict[str, Any]] = []
    for position, (row_index, metadata_row) in enumerate(metadata.iterrows()):
        identity = {
            column: _required_text(metadata_row[column], field=column, row=row_index)
            for column in IDENTITY_COLUMNS
        }
        split = _required_text(metadata_row["split"], field="split", row=row_index)
        if split not in _ALLOWED_SPLITS:
            raise PredictionContractError(
                f"split must be one of {list(_ALLOWED_SPLITS)!r}; "
                f"got {split!r} at row {row_index!r}"
            )
        true_class = _canonical_class(
            metadata_row[true_column], field="true_class", row=row_index
        )
        predicted_index = int(np.argmax(probabilities[position]))
        record: dict[str, Any] = {
            **identity,
            "split": split,
            "true_class": true_class,
            "predicted_class": CLASS_ORDER[predicted_index],
            "seed": canonical_seed,
            "baseline_variant": variant,
            "experiment_id": canonical_experiment_id,
        }
        record.update(
            {
                column: float(probabilities[position, class_index])
                for class_index, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        record.update(
            {
                column: float(values[position, class_index])
                for class_index, column in enumerate(LOGIT_COLUMNS)
            }
        )
        rows.append(record)

    return validate_prediction_table(pd.DataFrame(rows, columns=PREDICTION_COLUMNS))


def validate_prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate all identities, labels, scores, and provenance fields.

    The returned copy has canonical scalar types and deterministic row order.
    Extra columns are rejected so downstream metrics cannot silently select an
    unintended score family or identity.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if predictions.empty:
        raise PredictionContractError("prediction table must contain at least one row")
    if not predictions.columns.is_unique:
        duplicates = predictions.columns[predictions.columns.duplicated()].tolist()
        raise PredictionContractError(
            f"prediction table has duplicate columns: {duplicates!r}"
        )
    actual_columns = tuple(str(column) for column in predictions.columns)
    if actual_columns != PREDICTION_COLUMNS:
        missing = [column for column in PREDICTION_COLUMNS if column not in actual_columns]
        extra = [column for column in actual_columns if column not in PREDICTION_COLUMNS]
        raise PredictionContractError(
            "prediction columns must exactly match PREDICTION_COLUMNS in order; "
            f"missing={missing!r}, extra={extra!r}, actual={list(actual_columns)!r}"
        )

    canonical = predictions.copy(deep=True)
    for column in IDENTITY_COLUMNS:
        canonical[column] = [
            _required_text(value, field=column, row=row_index)
            for row_index, value in canonical[column].items()
        ]
    duplicated = canonical.duplicated(subset=list(IDENTITY_COLUMNS), keep=False)
    if bool(duplicated.any()):
        examples = canonical.loc[duplicated, list(IDENTITY_COLUMNS)].head(5).to_dict("records")
        raise PredictionContractError(
            f"three-part sample identities must be unique; duplicates={examples!r}"
        )
    if bool(canonical["nodule_id"].duplicated().any()):
        duplicates = sorted(
            canonical.loc[canonical["nodule_id"].duplicated(False), "nodule_id"]
            .unique()
            .tolist()
        )
        raise PredictionContractError(
            f"stable nodule_id values must be globally unique; duplicates={duplicates[:5]!r}"
        )

    canonical["split"] = [
        _required_text(value, field="split", row=row_index)
        for row_index, value in canonical["split"].items()
    ]
    invalid_splits = ~canonical["split"].isin(_ALLOWED_SPLITS)
    if bool(invalid_splits.any()):
        values = sorted(canonical.loc[invalid_splits, "split"].unique().tolist())
        raise PredictionContractError(
            f"split must contain only {list(_ALLOWED_SPLITS)!r}; got {values!r}"
        )

    for column in ("true_class", "predicted_class"):
        # Fixed exports use names, not numeric class aliases.
        invalid = ~canonical[column].isin(CLASS_ORDER)
        if bool(invalid.any()):
            row_index = canonical.index[invalid][0]
            raise PredictionContractError(
                f"{column} must use exact canonical names {list(CLASS_ORDER)!r}; "
                f"got {canonical.loc[row_index, column]!r} at row {row_index!r}"
            )

    score_columns = [*PROBABILITY_COLUMNS, *LOGIT_COLUMNS]
    numeric = canonical.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    scores = numeric.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(scores).all()):
        row, column = np.argwhere(~np.isfinite(scores))[0]
        raise PredictionContractError(
            f"score {score_columns[int(column)]!r} is non-numeric or non-finite "
            f"at row {canonical.index[int(row)]!r}"
        )
    canonical.loc[:, score_columns] = numeric

    probabilities = numeric.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    logits = numeric.loc[:, list(LOGIT_COLUMNS)].to_numpy(dtype=np.float64)
    expected_probabilities = stable_softmax(logits)
    if not np.allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-12,
        atol=1e-15,
    ):
        row, class_index = np.argwhere(
            ~np.isclose(
                probabilities,
                expected_probabilities,
                rtol=1e-12,
                atol=1e-15,
            )
        )[0]
        raise PredictionContractError(
            "probability/logit mismatch at row "
            f"{canonical.index[int(row)]!r}, class {CLASS_ORDER[int(class_index)]!r}: "
            f"stored={probabilities[int(row), int(class_index)]!r}, "
            f"softmax={expected_probabilities[int(row), int(class_index)]!r}"
        )

    expected_classes = np.asarray(CLASS_ORDER, dtype=object)[
        np.argmax(expected_probabilities, axis=1)
    ]
    predicted = canonical["predicted_class"].to_numpy(dtype=object)
    mismatched = predicted != expected_classes
    if bool(mismatched.any()):
        row = int(np.flatnonzero(mismatched)[0])
        raise PredictionContractError(
            f"predicted_class does not equal logit argmax at row "
            f"{canonical.index[row]!r}: stored={predicted[row]!r}, "
            f"expected={expected_classes[row]!r}"
        )

    canonical["seed"] = [_canonical_seed(value) for value in canonical["seed"]]
    if canonical["seed"].nunique(dropna=False) != 1:
        raise PredictionContractError("a prediction table must contain exactly one seed")
    canonical["baseline_variant"] = [
        _required_text(value, field="baseline_variant", row=row_index)
        for row_index, value in canonical["baseline_variant"].items()
    ]
    invalid_variants = ~canonical["baseline_variant"].isin(_VARIANTS)
    if bool(invalid_variants.any()):
        raise PredictionContractError(
            f"baseline_variant must be one of {list(_VARIANTS)!r}"
        )
    if canonical["baseline_variant"].nunique(dropna=False) != 1:
        raise PredictionContractError(
            "a prediction table must contain exactly one baseline_variant"
        )
    canonical["experiment_id"] = [
        _required_text(value, field="experiment_id", row=row_index)
        for row_index, value in canonical["experiment_id"].items()
    ]
    if canonical["experiment_id"].nunique(dropna=False) != 1:
        raise PredictionContractError(
            "a prediction table must contain exactly one experiment_id"
        )

    return _deterministic_sort(canonical)


def evaluate_phase3_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate a Phase 3 table using its probability columns explicitly."""

    canonical = validate_prediction_table(predictions)
    probability_mapping: Mapping[str, str] = {
        class_name: PROBABILITY_COLUMNS[index]
        for index, class_name in enumerate(CLASS_ORDER)
    }
    return evaluate_predictions(
        canonical,
        identifier_column="nodule_id",
        true_label_column="true_class",
        predicted_class_column="predicted_class",
        probability_columns=probability_mapping,
    )


def ordered_sample_checksum(predictions: pd.DataFrame) -> str:
    """Hash canonical export order, identities, split, and reference class."""

    canonical = validate_prediction_table(predictions)
    records = canonical.loc[
        :, [*IDENTITY_COLUMNS, "split", "true_class"]
    ].to_dict(orient="records")
    return sha256_json(
        {
            "schema_version": 1,
            "class_order": list(CLASS_ORDER),
            "ordered_samples": records,
        }
    )


def write_prediction_csv(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write canonical UTF-8 CSV bytes with deterministic ordering."""

    canonical = validate_prediction_table(predictions)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prediction artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO(newline="")
    canonical.to_csv(
        buffer,
        index=False,
        columns=list(PREDICTION_COLUMNS),
        lineterminator="\n",
    )
    payload = buffer.getvalue().encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite prediction artifact: {destination}"
                ) from exc
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_phase3_prediction_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a Phase 3 prediction CSV without losing text identities."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction CSV does not exist: {source}")
    text_columns = {
        column: "string"
        for column in (
            *IDENTITY_COLUMNS,
            "split",
            "true_class",
            "predicted_class",
            "baseline_variant",
            "experiment_id",
        )
    }
    try:
        frame = pd.read_csv(source, dtype=text_columns)
    except pd.errors.EmptyDataError as exc:
        raise PredictionContractError(f"Prediction CSV is empty: {source}") from exc
    return validate_prediction_table(frame)


__all__ = [
    "CLASS_ORDER",
    "IDENTITY_COLUMNS",
    "LOGIT_COLUMNS",
    "PREDICTION_COLUMNS",
    "PROBABILITY_COLUMNS",
    "PredictionContractError",
    "build_prediction_table",
    "evaluate_phase3_predictions",
    "load_phase3_prediction_csv",
    "ordered_sample_checksum",
    "stable_softmax",
    "validate_prediction_table",
    "write_prediction_csv",
]
