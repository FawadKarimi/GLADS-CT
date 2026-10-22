"""Deterministic patient-cluster bootstrap utilities for Phase 4 evaluation.

The sampling unit is a patient, not a nodule.  Each bootstrap replicate draws
``n_patients`` patient identifiers with replacement and materializes *all*
nodules belonging to every draw.  Drawing one patient twice therefore includes
that patient's complete nodule cluster twice.  Paired B1/ConsenDxFormer
comparisons use one shared patient draw for both models.

Intervals are nonparametric percentile intervals with NumPy's fixed ``linear``
quantile rule.  The returned records are descriptive uncertainty summaries;
they intentionally contain no p-values or declarations of statistical
significance.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .hashing import sha256_json


SAMPLE_ID_COLUMNS: tuple[str, str, str, str] = (
    "patient_id",
    "nodule_id",
    "series_instance_uid",
    "split",
)
MATCH_COLUMNS: tuple[str, str, str, str, str] = (
    *SAMPLE_ID_COLUMNS,
    "true_class",
)
BOOTSTRAP_METRICS: tuple[str, str, str, str, str] = (
    "macro_f1",
    "benign_f1",
    "indeterminate_f1",
    "indeterminate_recall",
    "malignant_f1",
)
BOOTSTRAP_METHOD = "patient_cluster_percentile_v1"
BOOTSTRAP_METRIC_DEFINITION = {
    "class_order": list(CLASS_ORDER),
    "precision": "TP / (TP + FP)",
    "recall": "TP / (TP + FN)",
    "f1": "2 * precision * recall / (precision + recall)",
    "zero_denominator_value": 0.0,
    "macro_f1": "arithmetic mean over all three fixed classes, including absent classes",
    "replicate_metric_unit": "nodule; patient clusters are the sampling unit",
}


class Phase4BootstrapError(ValueError):
    """Raised when the patient-bootstrap or pairing contract is violated."""


def _required_text(value: Any, *, field: str, model_name: str, row: Any) -> str:
    if value is None or (
        not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))
    ):
        raise Phase4BootstrapError(
            f"{model_name} {field!r} is missing at row {row!r}"
        )
    text = str(value).strip()
    if not text:
        raise Phase4BootstrapError(
            f"{model_name} {field!r} is blank at row {row!r}"
        )
    return text


def _validate_prediction_rows(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    expected_split: str,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(f"{model_name} predictions must be a pandas DataFrame")
    if predictions.empty:
        raise Phase4BootstrapError(f"{model_name} prediction table is empty")
    if not predictions.columns.is_unique:
        duplicates = predictions.columns[predictions.columns.duplicated()].tolist()
        raise Phase4BootstrapError(
            f"{model_name} prediction table has duplicate columns: {duplicates!r}"
        )

    required = {*MATCH_COLUMNS, "predicted_class"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise Phase4BootstrapError(
            f"{model_name} prediction table is missing required columns: {missing!r}"
        )

    canonical = predictions.copy(deep=True)
    for column in (*SAMPLE_ID_COLUMNS, "true_class", "predicted_class"):
        canonical[column] = [
            _required_text(
                value,
                field=column,
                model_name=model_name,
                row=row_index,
            )
            for row_index, value in canonical[column].items()
        ]

    if not isinstance(expected_split, str) or not expected_split.strip():
        raise Phase4BootstrapError("expected_split must be a non-blank string")
    expected_split = expected_split.strip()
    invalid_split = canonical["split"] != expected_split
    if bool(invalid_split.any()):
        actual = sorted(canonical.loc[invalid_split, "split"].unique().tolist())
        raise Phase4BootstrapError(
            f"{model_name} predictions must contain expected split "
            f"{expected_split!r} only; got {actual!r}"
        )

    for column in ("true_class", "predicted_class"):
        invalid = ~canonical[column].isin(CLASS_ORDER)
        if bool(invalid.any()):
            row_index = canonical.index[invalid][0]
            raise Phase4BootstrapError(
                f"{model_name} {column} must use the fixed class order "
                f"{list(CLASS_ORDER)!r}; got {canonical.loc[row_index, column]!r} "
                f"at row {row_index!r}"
            )

    duplicated = canonical.duplicated(subset=list(SAMPLE_ID_COLUMNS), keep=False)
    if bool(duplicated.any()):
        examples = canonical.loc[
            duplicated, list(SAMPLE_ID_COLUMNS)
        ].head(5).to_dict("records")
        raise Phase4BootstrapError(
            f"{model_name} prediction table has duplicate sample identities: "
            f"{examples!r}"
        )
    duplicated_nodules = canonical["nodule_id"].duplicated(keep=False)
    if bool(duplicated_nodules.any()):
        examples = sorted(
            canonical.loc[duplicated_nodules, "nodule_id"].unique().tolist()
        )[:5]
        raise Phase4BootstrapError(
            f"{model_name} prediction table has duplicate nodule_id values: "
            f"{examples!r}"
        )

    return canonical.sort_values(
        list(SAMPLE_ID_COLUMNS),
        kind="mergesort",
        ignore_index=True,
    )


def validate_paired_prediction_rows(
    b1_predictions: pd.DataFrame,
    consendxformer_predictions: pd.DataFrame,
    *,
    expected_split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and align an exact frozen B1/ConsenDxFormer prediction pair.

    Rows may arrive in different orders, but their four-part sample identities
    and true classes must match exactly.  Duplicate identities are rejected
    before alignment so a set comparison cannot hide multiplicity errors.
    """

    b1 = _validate_prediction_rows(
        b1_predictions,
        model_name="B1",
        expected_split=expected_split,
    )
    consen = _validate_prediction_rows(
        consendxformer_predictions,
        model_name="ConsenDxFormer",
        expected_split=expected_split,
    )

    b1_identities = [
        tuple(row)
        for row in b1.loc[:, list(SAMPLE_ID_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]
    consen_identities = [
        tuple(row)
        for row in consen.loc[:, list(SAMPLE_ID_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]
    if b1_identities != consen_identities:
        b1_set = set(b1_identities)
        consen_set = set(consen_identities)
        b1_only = sorted(b1_set - consen_set)[:5]
        consen_only = sorted(consen_set - b1_set)[:5]
        raise Phase4BootstrapError(
            "B1/ConsenDxFormer sample identity mismatch: "
            f"B1_only={b1_only!r}, ConsenDxFormer_only={consen_only!r}"
        )

    b1_true = b1["true_class"].to_numpy(dtype=object)
    consen_true = consen["true_class"].to_numpy(dtype=object)
    unequal = b1_true != consen_true
    if bool(unequal.any()):
        position = int(np.flatnonzero(unequal)[0])
        identity = {
            column: b1.iloc[position][column] for column in SAMPLE_ID_COLUMNS
        }
        raise Phase4BootstrapError(
            "B1/ConsenDxFormer true-class mismatch for sample "
            f"{identity!r}: B1={b1_true[position]!r}, "
            f"ConsenDxFormer={consen_true[position]!r}"
        )

    matching_records = b1.loc[:, list(MATCH_COLUMNS)].to_dict(orient="records")
    report: dict[str, Any] = {
        "schema_version": 1,
        "matched": True,
        "match_columns": list(MATCH_COLUMNS),
        "expected_split": expected_split,
        "n_patients": int(b1["patient_id"].nunique()),
        "n_nodules": int(len(b1)),
        "matched_rows_sha256": sha256_json(
            {
                "schema_version": 1,
                "class_order": list(CLASS_ORDER),
                "match_columns": list(MATCH_COLUMNS),
                "rows": matching_records,
            }
        ),
    }
    return b1, consen, report


def _canonical_patient_id(value: Any, *, role: str, position: int) -> str:
    if value is None or (
        not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))
    ):
        raise Phase4BootstrapError(
            f"{role} patient identifier is missing at position {position}"
        )
    text = str(value).strip()
    if not text:
        raise Phase4BootstrapError(
            f"{role} patient identifier is blank at position {position}"
        )
    return text


def patient_cluster_resample_indices(
    row_patient_ids: Iterable[Any],
    sampled_patient_ids: Iterable[Any],
) -> np.ndarray:
    """Materialize row positions for an explicit ordered patient draw.

    For every sampled patient occurrence, every source row for that patient is
    appended in source order.  Repeated patient draws deliberately repeat the
    entire cluster.
    """

    if isinstance(row_patient_ids, (str, bytes)) or not isinstance(
        row_patient_ids, Iterable
    ):
        raise TypeError("row_patient_ids must be an iterable of patient identifiers")
    if isinstance(sampled_patient_ids, (str, bytes)) or not isinstance(
        sampled_patient_ids, Iterable
    ):
        raise TypeError(
            "sampled_patient_ids must be an iterable of patient identifiers"
        )
    source = [
        _canonical_patient_id(value, role="source", position=position)
        for position, value in enumerate(row_patient_ids)
    ]
    draws = [
        _canonical_patient_id(value, role="sampled", position=position)
        for position, value in enumerate(sampled_patient_ids)
    ]
    if not source:
        raise Phase4BootstrapError("row_patient_ids must contain at least one row")
    if not draws:
        raise Phase4BootstrapError(
            "sampled_patient_ids must contain at least one patient draw"
        )

    positions_by_patient: dict[str, list[int]] = {}
    for position, patient_id in enumerate(source):
        positions_by_patient.setdefault(patient_id, []).append(position)
    unknown = sorted(set(draws) - set(positions_by_patient))
    if unknown:
        raise Phase4BootstrapError(
            f"sampled patient identifiers are absent from source rows: {unknown!r}"
        )

    materialized = [
        row_position
        for patient_id in draws
        for row_position in positions_by_patient[patient_id]
    ]
    return np.asarray(materialized, dtype=np.int64)


def _validate_bootstrap_parameters(
    *,
    seed: int,
    n_resamples: int,
    confidence_level: float,
) -> tuple[int, int, float]:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise Phase4BootstrapError("bootstrap seed must be an unsigned 32-bit integer")
    canonical_seed = int(seed)
    if not 0 <= canonical_seed <= 2**32 - 1:
        raise Phase4BootstrapError("bootstrap seed must be an unsigned 32-bit integer")
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise Phase4BootstrapError("n_resamples must be a positive integer")
    canonical_resamples = int(n_resamples)
    if canonical_resamples <= 0:
        raise Phase4BootstrapError("n_resamples must be a positive integer")
    if isinstance(confidence_level, (bool, np.bool_)):
        raise Phase4BootstrapError("confidence_level must be strictly between 0 and 1")
    try:
        canonical_confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise Phase4BootstrapError(
            "confidence_level must be strictly between 0 and 1"
        ) from exc
    if not np.isfinite(canonical_confidence) or not 0.0 < canonical_confidence < 1.0:
        raise Phase4BootstrapError("confidence_level must be strictly between 0 and 1")
    return canonical_seed, canonical_resamples, canonical_confidence


def _metric_vector(
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
) -> np.ndarray:
    class_to_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    true_indices = np.fromiter(
        (class_to_index[str(value)] for value in true_classes),
        dtype=np.int64,
        count=len(true_classes),
    )
    predicted_indices = np.fromiter(
        (class_to_index[str(value)] for value in predicted_classes),
        dtype=np.int64,
        count=len(predicted_classes),
    )
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    np.add.at(confusion, (true_indices, predicted_indices), 1)

    f1_scores: list[float] = []
    recalls: list[float] = []
    for class_index in range(len(CLASS_ORDER)):
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
        recalls.append(recall)
        f1_scores.append(f1)

    return np.asarray(
        [
            float(np.mean(f1_scores, dtype=np.float64)),
            f1_scores[0],
            f1_scores[1],
            recalls[1],
            f1_scores[2],
        ],
        dtype=np.float64,
    )


def _metric_record(values: np.ndarray) -> dict[str, float]:
    return {
        metric: float(values[index])
        for index, metric in enumerate(BOOTSTRAP_METRICS)
    }


def _estimate_record(values: np.ndarray) -> dict[str, list[float]]:
    return {
        metric: [float(value) for value in values[:, index].tolist()]
        for index, metric in enumerate(BOOTSTRAP_METRICS)
    }


def _interval_record(
    estimates: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, dict[str, float | str]]:
    tail = (1.0 - confidence_level) / 2.0
    bounds = np.quantile(
        estimates,
        [tail, 1.0 - tail],
        axis=0,
        method="linear",
    )
    return {
        metric: {
            "lower": float(bounds[0, index]),
            "upper": float(bounds[1, index]),
            "confidence_level": confidence_level,
            "method": "percentile",
        }
        for index, metric in enumerate(BOOTSTRAP_METRICS)
    }


def _bootstrap_metadata(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int,
    confidence_level: float,
    paired: bool,
) -> dict[str, Any]:
    return {
        "method": BOOTSTRAP_METHOD,
        "metric_definition": dict(BOOTSTRAP_METRIC_DEFINITION),
        "sampling_unit": "patient_id",
        "patient_grouping_preserved": True,
        "paired_resampling": paired,
        "rng": "numpy.random.Generator(PCG64)",
        "seed": seed,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "ci_method": "percentile",
        "quantile_method": "linear",
        "n_patients": int(frame["patient_id"].nunique()),
        "n_nodules": int(len(frame)),
        "draws_per_resample": int(frame["patient_id"].nunique()),
    }


def _draw_matrix(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int,
) -> list[np.ndarray]:
    patients = sorted(frame["patient_id"].unique().tolist())
    rng = np.random.Generator(np.random.PCG64(seed))
    result: list[np.ndarray] = []
    for _ in range(n_resamples):
        draw_positions = rng.integers(0, len(patients), size=len(patients))
        sampled_patients = [patients[int(position)] for position in draw_positions]
        result.append(
            patient_cluster_resample_indices(
                frame["patient_id"].tolist(),
                sampled_patients,
            )
        )
    return result


def bootstrap_model_metrics(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
    n_resamples: int,
    confidence_level: float = 0.95,
    expected_split: str = "test",
) -> dict[str, Any]:
    """Return fixed primary metrics and patient-cluster percentile intervals."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise Phase4BootstrapError("model_name must be a non-blank string")
    canonical_seed, canonical_resamples, canonical_confidence = (
        _validate_bootstrap_parameters(
            seed=seed,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
    )
    canonical = _validate_prediction_rows(
        predictions,
        model_name=model_name.strip(),
        expected_split=expected_split,
    )
    true_classes = canonical["true_class"].to_numpy(dtype=object)
    predicted_classes = canonical["predicted_class"].to_numpy(dtype=object)
    point = _metric_vector(true_classes, predicted_classes)
    estimates = np.empty(
        (canonical_resamples, len(BOOTSTRAP_METRICS)), dtype=np.float64
    )
    for replicate, row_indices in enumerate(
        _draw_matrix(
            canonical,
            seed=canonical_seed,
            n_resamples=canonical_resamples,
        )
    ):
        estimates[replicate] = _metric_vector(
            true_classes[row_indices], predicted_classes[row_indices]
        )

    return {
        "schema_version": 1,
        "analysis": "per_model_patient_cluster_bootstrap",
        "model": model_name.strip(),
        "class_order": list(CLASS_ORDER),
        "metrics": list(BOOTSTRAP_METRICS),
        "point_estimates": _metric_record(point),
        "confidence_intervals": _interval_record(
            estimates,
            confidence_level=canonical_confidence,
        ),
        "bootstrap": _bootstrap_metadata(
            canonical,
            seed=canonical_seed,
            n_resamples=canonical_resamples,
            confidence_level=canonical_confidence,
            paired=False,
        ),
        "bootstrap_estimates": _estimate_record(estimates),
        "interpretation": (
            "Descriptive patient-cluster percentile confidence intervals; no "
            "parametric significance claim is made."
        ),
    }


def bootstrap_paired_differences(
    b1_predictions: pd.DataFrame,
    consendxformer_predictions: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int,
    confidence_level: float = 0.95,
    expected_split: str = "test",
) -> dict[str, Any]:
    """Bootstrap paired ConsenDxFormer-minus-B1 primary-metric differences."""

    canonical_seed, canonical_resamples, canonical_confidence = (
        _validate_bootstrap_parameters(
            seed=seed,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
    )
    b1, consen, matching = validate_paired_prediction_rows(
        b1_predictions,
        consendxformer_predictions,
        expected_split=expected_split,
    )
    true_classes = b1["true_class"].to_numpy(dtype=object)
    b1_predicted = b1["predicted_class"].to_numpy(dtype=object)
    consen_predicted = consen["predicted_class"].to_numpy(dtype=object)

    b1_point = _metric_vector(true_classes, b1_predicted)
    consen_point = _metric_vector(true_classes, consen_predicted)
    b1_estimates = np.empty(
        (canonical_resamples, len(BOOTSTRAP_METRICS)), dtype=np.float64
    )
    consen_estimates = np.empty_like(b1_estimates)
    for replicate, row_indices in enumerate(
        _draw_matrix(
            b1,
            seed=canonical_seed,
            n_resamples=canonical_resamples,
        )
    ):
        sampled_true = true_classes[row_indices]
        b1_estimates[replicate] = _metric_vector(
            sampled_true, b1_predicted[row_indices]
        )
        consen_estimates[replicate] = _metric_vector(
            sampled_true, consen_predicted[row_indices]
        )
    difference_estimates = consen_estimates - b1_estimates

    return {
        "schema_version": 1,
        "analysis": "paired_model_patient_cluster_bootstrap",
        "comparison_direction": "ConsenDxFormer_minus_B1",
        "class_order": list(CLASS_ORDER),
        "metrics": list(BOOTSTRAP_METRICS),
        "matching": matching,
        "point_estimates": {
            "B1": _metric_record(b1_point),
            "ConsenDxFormer": _metric_record(consen_point),
            "difference": _metric_record(consen_point - b1_point),
        },
        "confidence_intervals": {
            "B1": _interval_record(
                b1_estimates,
                confidence_level=canonical_confidence,
            ),
            "ConsenDxFormer": _interval_record(
                consen_estimates,
                confidence_level=canonical_confidence,
            ),
            "difference": _interval_record(
                difference_estimates,
                confidence_level=canonical_confidence,
            ),
        },
        "bootstrap": _bootstrap_metadata(
            b1,
            seed=canonical_seed,
            n_resamples=canonical_resamples,
            confidence_level=canonical_confidence,
            paired=True,
        ),
        "bootstrap_estimates": {
            "B1": _estimate_record(b1_estimates),
            "ConsenDxFormer": _estimate_record(consen_estimates),
            "difference": _estimate_record(difference_estimates),
        },
        "interpretation": (
            "Descriptive paired patient-cluster percentile confidence intervals; "
            "no parametric significance claim is made."
        ),
    }


# Readable aliases for callers that name the statistical design explicitly.
patient_aware_bootstrap = bootstrap_model_metrics
paired_patient_aware_bootstrap = bootstrap_paired_differences
validate_paired_prediction_cohort = validate_paired_prediction_rows


__all__ = [
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_METRICS",
    "MATCH_COLUMNS",
    "SAMPLE_ID_COLUMNS",
    "Phase4BootstrapError",
    "bootstrap_model_metrics",
    "bootstrap_paired_differences",
    "paired_patient_aware_bootstrap",
    "patient_aware_bootstrap",
    "patient_cluster_resample_indices",
    "validate_paired_prediction_cohort",
    "validate_paired_prediction_rows",
]
