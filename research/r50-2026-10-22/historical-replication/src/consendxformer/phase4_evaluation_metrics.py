"""Pure, prespecified metrics for Phase 4 post-hoc evaluation.

This module intentionally performs no file I/O, model inference, fitting, or
threshold selection.  Every class-dependent calculation uses the immutable
order ``benign, indeterminate, malignant``.  The public definition records are
JSON-compatible so an evaluation runner can copy them verbatim into provenance.

Attention results are spatial-correspondence measurements.  They must not be
interpreted as causal explanations of a model prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .baseline_evaluation import CLASS_ORDER


ECE_BIN_COUNT = 15
ECE_BIN_EDGES: tuple[float, ...] = tuple(
    float(index / ECE_BIN_COUNT) for index in range(ECE_BIN_COUNT + 1)
)
NLL_EPSILON = 1e-15
ATTENTION_QUANTILE = 0.90
ATTENTION_QUANTILE_METHOD = "linear"
SOFT_MASK_THRESHOLD = 0.5
ATTENTION_NORMALIZATION_ATOL = 1e-6
_KL_EPSILON = 1e-8
_IDENTIFIER_FIELDS = ("patient_id", "nodule_id", "series_instance_uid")


CALIBRATION_DEFINITION: dict[str, Any] = {
    "schema_version": 1,
    "class_order": list(CLASS_ORDER),
    "nll": {
        "formula": "-mean_i(log(max(p_i,true_i, 1e-15)))",
        "logarithm": "natural",
        "zero_probability_floor": NLL_EPSILON,
    },
    "brier": {
        "formula": "mean_i(sum_c((p_i,c - 1[true_i=c])^2))",
        "class_normalization": "no division by number of classes",
    },
    "ece": {
        "variant": "top-label (maximum-confidence) multiclass ECE",
        "formula": "sum_b (n_b / N) * abs(accuracy_b - mean_confidence_b)",
        "n_bins": ECE_BIN_COUNT,
        "bin_edges": list(ECE_BIN_EDGES),
        "bin_membership": (
            "[edge_i, edge_(i+1)) except the final bin [edge_14, 1.0]"
        ),
        "prediction": "argmax in fixed class order; first class wins an exact tie",
        "fitting": "none",
    },
}

UNCERTAINTY_DEFINITION: dict[str, Any] = {
    "schema_version": 1,
    "class_order": list(CLASS_ORDER),
    "predictive_entropy": {
        "formula": "-sum_c(p_c * ln(p_c)), with 0*ln(0) defined as 0",
        "logarithm": "natural",
    },
    "association": {
        "statistic": "Spearman rank correlation",
        "ties": "average ranks",
        "degenerate_result": "null coefficient with explicit reason",
    },
    "disagreement_groups": {
        "recommended_threshold": "median of the frozen training-split disagreement entropy",
        "standalone_default": "median of supplied disagreement entropy values only",
        "rule": "low: disagreement < threshold; high: disagreement >= threshold",
        "tie_policy": "high_if_greater_than_or_equal_to_threshold",
        "outcome_based_selection": False,
    },
}

RISK_COVERAGE_DEFINITION: dict[str, Any] = {
    "schema_version": 1,
    "class_order": list(CLASS_ORDER),
    "risk": "mean 0/1 classification error among retained samples",
    "coverage": "retained sample count divided by total sample count",
    "entropy_ranking": "ascending natural-log predictive entropy",
    "confidence_ranking": "descending maximum class probability",
    "ties": "ascending supplied stable identity; input position only if identities are omitted",
    "aurc": "right-endpoint discrete integral: mean of risks at coverages k/N, k=1..N",
    "threshold_fitting": "none",
}

ATTENTION_ALIGNMENT_DEFINITION: dict[str, Any] = {
    "schema_version": 1,
    "interpretation": (
        "attention-alignment / spatial-correspondence analysis; not a causal explanation"
    ),
    "mask_resize": "deterministic area averaging onto the native attention grid",
    "normalization": (
        "DSA attention must already have unit spatial mass within absolute tolerance "
        f"{ATTENTION_NORMALIZATION_ATOL}; resized consensus is normalized to unit mass"
    ),
    "kl": {
        "direction": "KL(consensus spatial distribution || DSA attention distribution)",
        "epsilon": _KL_EPSILON,
        "relationship_to_training": "same direction as Phase 4 attention alignment loss",
    },
    "continuous_correlation": (
        "Pearson correlation across flattened spatial cells; null for a constant map"
    ),
    "continuous_overlap": "sum of elementwise minima of the two unit-mass distributions",
    "dice_iou": {
        "attention_foreground": (
            "attention >= its per-map 0.90 quantile using NumPy linear interpolation"
        ),
        "soft_mask_foreground": "area-resized soft consensus mask >= 0.5",
        "attention_quantile": ATTENTION_QUANTILE,
        "attention_quantile_method": ATTENTION_QUANTILE_METHOD,
        "soft_mask_threshold": SOFT_MASK_THRESHOLD,
        "threshold_fitting": "none",
    },
}


def _probability_matrix(probabilities: Any) -> np.ndarray:
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a numeric matrix") from exc
    if values.ndim != 2 or values.shape[1] != len(CLASS_ORDER):
        raise ValueError(
            f"probabilities must have shape [N, {len(CLASS_ORDER)}] with three columns "
            f"in class order {list(CLASS_ORDER)!r}"
        )
    if values.shape[0] == 0:
        raise ValueError("probabilities must contain at least one observation")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    row_mass = values.sum(axis=1)
    if not np.allclose(row_mass, 1.0, rtol=0.0, atol=1e-8):
        row = int(np.flatnonzero(~np.isclose(row_mass, 1.0, rtol=0.0, atol=1e-8))[0])
        raise ValueError(
            "probabilities must sum to 1 for every row within absolute tolerance "
            f"1e-8; row {row} sums to {float(row_mass[row])!r}"
        )
    return values


def _class_index(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean labels are not class indices")
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < len(CLASS_ORDER):
            return index
        raise ValueError(f"class index {index} is outside 0..{len(CLASS_ORDER) - 1}")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return _class_index(int(number))
    text = str(value).strip().casefold() if value is not None else ""
    if text in CLASS_ORDER:
        return CLASS_ORDER.index(text)
    raise ValueError(
        f"Unknown class {value!r}; expected one of {list(CLASS_ORDER)!r} or indices 0, 1, 2"
    )


def _label_indices(labels: Sequence[Any], expected_length: int) -> np.ndarray:
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a sequence, not a string")
    values = list(labels)
    if len(values) != expected_length:
        raise ValueError(
            f"label/probability row-count mismatch: {len(values)} != {expected_length}"
        )
    encoded = np.empty(expected_length, dtype=np.int64)
    for position, value in enumerate(values):
        try:
            encoded[position] = _class_index(value)
        except ValueError as exc:
            raise ValueError(f"Invalid label at position {position}: {exc}") from exc
    return encoded


def multiclass_negative_log_likelihood(
    true_labels: Sequence[Any], probabilities: Any
) -> float:
    """Return mean three-class NLL using the fixed class order and log floor."""

    values = _probability_matrix(probabilities)
    labels = _label_indices(true_labels, len(values))
    true_probability = values[np.arange(len(values)), labels]
    return float(-np.log(np.maximum(true_probability, NLL_EPSILON)).mean())


def multiclass_brier_score(true_labels: Sequence[Any], probabilities: Any) -> float:
    """Return ``mean_i sum_c (p_ic - y_ic)^2`` (not divided by classes)."""

    values = _probability_matrix(probabilities)
    labels = _label_indices(true_labels, len(values))
    targets = np.zeros_like(values)
    targets[np.arange(len(values)), labels] = 1.0
    return float(np.square(values - targets).sum(axis=1).mean())


def expected_calibration_error(
    true_labels: Sequence[Any],
    probabilities: Any,
    *,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Compute frozen 15-bin top-label multiclass ECE.

    Confidence is the maximum probability and accuracy is correctness of the
    argmax class.  Bins are equal width on [0, 1], left closed/right open,
    except the last bin includes 1.  Empty bins contribute zero.
    """

    values = _probability_matrix(probabilities)
    labels = _label_indices(true_labels, len(values))
    predicted = np.argmax(values, axis=1)
    confidence = np.max(values, axis=1)
    correct = predicted == labels
    edges = np.asarray(ECE_BIN_EDGES, dtype=np.float64)
    assignments = np.searchsorted(edges, confidence, side="right") - 1
    assignments = np.clip(assignments, 0, ECE_BIN_COUNT - 1)

    ece = 0.0
    bins: list[dict[str, Any]] = []
    for bin_index in range(ECE_BIN_COUNT):
        members = assignments == bin_index
        count = int(members.sum())
        accuracy = float(correct[members].mean()) if count else None
        mean_confidence = float(confidence[members].mean()) if count else None
        absolute_gap = (
            float(abs(accuracy - mean_confidence))
            if accuracy is not None and mean_confidence is not None
            else None
        )
        weight = float(count / len(values))
        if absolute_gap is not None:
            ece += weight * absolute_gap
        bins.append(
            {
                "bin_index": bin_index,
                "lower_edge": float(edges[bin_index]),
                "upper_edge": float(edges[bin_index + 1]),
                "upper_inclusive": bin_index == ECE_BIN_COUNT - 1,
                "count": count,
                "weight": weight,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "absolute_gap": absolute_gap,
            }
        )
    if not return_details:
        return float(ece)
    return {
        "ece": float(ece),
        "n_samples": int(len(values)),
        "definition": CALIBRATION_DEFINITION["ece"],
        "bins": bins,
    }


def calibration_metrics(
    true_labels: Sequence[Any], probabilities: Any
) -> dict[str, Any]:
    """Return all prespecified calibration metrics and their provenance."""

    values = _probability_matrix(probabilities)
    return {
        "class_order": list(CLASS_ORDER),
        "n_samples": int(len(values)),
        "negative_log_likelihood": multiclass_negative_log_likelihood(
            true_labels, values
        ),
        "brier_score": multiclass_brier_score(true_labels, values),
        "expected_calibration_error": expected_calibration_error(
            true_labels, values
        ),
        "definition": CALIBRATION_DEFINITION,
    }


def predictive_entropy(probabilities: Any) -> np.ndarray:
    """Return natural-log predictive entropy for each observation."""

    values = _probability_matrix(probabilities)
    terms = np.zeros_like(values)
    positive = values > 0.0
    terms[positive] = values[positive] * np.log(values[positive])
    return -terms.sum(axis=1)


def _finite_vector(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        # Ranks are one-based. Positions start..end-1 have mean rank below.
        mean_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = mean_rank
        start = end
    return ranks


def spearman_association(left: Any, right: Any) -> dict[str, Any]:
    """Return Spearman rho with explicit nulls for small or constant input."""

    x = _finite_vector(left, "left")
    y = _finite_vector(right, "right")
    if len(x) != len(y):
        raise ValueError(f"Spearman row-count mismatch: {len(x)} != {len(y)}")
    if len(x) < 2:
        return {
            "coefficient": None,
            "n": int(len(x)),
            "status": "undefined",
            "reason": "fewer_than_two_observations",
        }
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.square(x_centered).sum() * np.square(y_centered).sum())
    )
    if denominator == 0.0:
        return {
            "coefficient": None,
            "n": int(len(x)),
            "status": "undefined",
            "reason": "constant_rank_input",
        }
    coefficient = float(np.dot(x_centered, y_centered) / denominator)
    return {
        "coefficient": float(np.clip(coefficient, -1.0, 1.0)),
        "n": int(len(x)),
        "status": "defined",
        "reason": None,
    }


def uncertainty_disagreement_association(
    probabilities: Any, disagreement_entropy: Any
) -> dict[str, Any]:
    """Associate predictive and frozen radiologist-disagreement entropies."""

    uncertainty = predictive_entropy(probabilities)
    disagreement = _finite_vector(disagreement_entropy, "disagreement_entropy")
    if len(disagreement) != len(uncertainty):
        raise ValueError(
            "disagreement/probability row-count mismatch: "
            f"{len(disagreement)} != {len(uncertainty)}"
        )
    result = spearman_association(uncertainty, disagreement)
    return {
        **result,
        "predictive_entropy_definition": UNCERTAINTY_DEFINITION["predictive_entropy"],
        "association_definition": UNCERTAINTY_DEFINITION["association"],
    }


def _stable_identifiers(identifiers: Sequence[Any] | None, count: int) -> list[str]:
    if identifiers is None:
        width = max(1, len(str(max(count - 1, 0))))
        return [f"input-position-{index:0{width}d}" for index in range(count)]
    if isinstance(identifiers, (str, bytes)):
        raise TypeError("identifiers must be a sequence, not a string")
    values = [str(value).strip() for value in identifiers]
    if len(values) != count:
        raise ValueError(f"identifier/probability row-count mismatch: {len(values)} != {count}")
    if any(not value for value in values):
        raise ValueError("identifiers must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError("identifiers must be unique for deterministic tie-breaking")
    return values


def risk_coverage_curve(
    true_labels: Sequence[Any],
    probabilities: Any,
    *,
    ranking: str = "entropy",
    identifiers: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return a selective-prediction curve and its frozen discrete AURC.

    At point ``k`` the ``k`` most certain observations are retained.  Risk is
    their mean 0/1 argmax error.  Equal selection scores are ordered by stable
    identity, never by correctness or outcome.
    """

    values = _probability_matrix(probabilities)
    labels = _label_indices(true_labels, len(values))
    stable_ids = _stable_identifiers(identifiers, len(values))
    if ranking == "entropy":
        scores = predictive_entropy(values)
        primary_order = scores
        direction = "ascending"
    elif ranking == "confidence":
        scores = np.max(values, axis=1)
        primary_order = -scores
        direction = "descending"
    else:
        raise ValueError("ranking must be 'entropy' or 'confidence'")

    identity_key = np.asarray(stable_ids, dtype=str)
    order = np.lexsort((identity_key, primary_order))
    predicted = np.argmax(values, axis=1)
    errors = (predicted != labels).astype(np.float64)[order]
    retained = np.arange(1, len(values) + 1, dtype=np.float64)
    risk = np.cumsum(errors) / retained
    coverage = retained / len(values)
    aurc = float(risk.mean())
    return {
        "ranking": ranking,
        "ranking_direction": direction,
        "n_samples": int(len(values)),
        "ordered_identifiers": [stable_ids[int(index)] for index in order],
        "selection_score": [float(scores[int(index)]) for index in order],
        "retained_count": [int(value) for value in retained],
        "coverage": [float(value) for value in coverage],
        "risk": [float(value) for value in risk],
        "aurc": aurc,
        "definition": RISK_COVERAGE_DEFINITION,
    }


def _descriptive_group(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "count": 0,
            "mean_predictive_entropy": None,
            "median_predictive_entropy": None,
            "standard_deviation": None,
        }
    return {
        "count": int(len(values)),
        "mean_predictive_entropy": float(values.mean()),
        "median_predictive_entropy": float(np.median(values)),
        "standard_deviation": (
            float(values.std(ddof=1)) if len(values) > 1 else None
        ),
    }


def high_low_disagreement_comparison(
    probabilities: Any,
    disagreement_entropy: Any,
    *,
    threshold: float | None = None,
    threshold_provenance: str | None = None,
    tie_policy: str = "high_if_greater_than_or_equal_to_threshold",
) -> dict[str, Any]:
    """Describe uncertainty around a prespecified disagreement threshold.

    Evaluation runners should pass the median computed once from the frozen
    training split and record that provenance.  The standalone default uses
    the median of the supplied disagreement values (never outcomes).  Values
    exactly at the threshold belong to the high-disagreement group under the
    single frozen tie policy.
    """

    uncertainty = predictive_entropy(probabilities)
    disagreement = _finite_vector(disagreement_entropy, "disagreement_entropy")
    if len(disagreement) != len(uncertainty):
        raise ValueError(
            "disagreement/probability row-count mismatch: "
            f"{len(disagreement)} != {len(uncertainty)}"
        )
    if np.any(disagreement < 0.0) or np.any(disagreement > 1.0):
        raise ValueError("disagreement_entropy must lie in [0, 1]")
    if tie_policy != "high_if_greater_than_or_equal_to_threshold":
        raise ValueError(
            "tie_policy must be 'high_if_greater_than_or_equal_to_threshold'"
        )
    threshold_was_supplied = threshold is not None
    if threshold is None:
        threshold_value = float(np.median(disagreement))
        if threshold_provenance is None:
            threshold_provenance = (
                "median of supplied disagreement entropy values only"
            )
    else:
        if isinstance(threshold, (bool, np.bool_)):
            raise ValueError("threshold must be a finite number in [0, 1]")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
            raise ValueError("threshold must be a finite number in [0, 1]")
        if threshold_provenance is None or not str(threshold_provenance).strip():
            raise ValueError("an explicit threshold requires non-empty threshold_provenance")
    provenance = str(threshold_provenance).strip()
    if not provenance:
        raise ValueError("threshold_provenance must be non-empty")

    high = disagreement >= threshold_value
    low = ~high
    return {
        "threshold": threshold_value,
        "rule": "low: disagreement < threshold; high: disagreement >= threshold",
        "tie_policy": tie_policy,
        "threshold_provenance": provenance,
        "threshold_was_supplied": threshold_was_supplied,
        "outcome_based_selection": False,
        "standard_deviation_definition": (
            "sample standard deviation (ddof=1); null for a group with fewer than two values"
        ),
        "groups": ["low" if member else "high" for member in low],
        "low": _descriptive_group(uncertainty[low]),
        "high": _descriptive_group(uncertainty[high]),
    }


def _map_batch(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, :, :]
    elif array.ndim != 3:
        raise ValueError(f"{name} must have shape [H, W], [N, H, W], or [N, 1, H, W]")
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_attention_normalization(attention_maps: Any) -> np.ndarray:
    """Validate nonnegative, finite DSA maps with unit spatial mass."""

    maps = _map_batch(attention_maps, "attention_maps")
    if np.any(maps < 0.0):
        raise ValueError("attention_maps must be non-negative")
    mass = maps.sum(axis=(1, 2))
    valid = np.isclose(mass, 1.0, rtol=0.0, atol=ATTENTION_NORMALIZATION_ATOL)
    if not valid.all():
        index = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            "attention_maps must have unit spatial mass within absolute tolerance "
            f"{ATTENTION_NORMALIZATION_ATOL}; map {index} sums to {float(mass[index])!r}"
        )
    return maps.copy()


def _validate_consensus_masks(consensus_masks: Any) -> np.ndarray:
    masks = _map_batch(consensus_masks, "consensus_masks")
    if np.any(masks < 0.0) or np.any(masks > 1.0):
        raise ValueError("consensus_masks must lie in [0, 1]")
    if np.any(masks.sum(axis=(1, 2)) <= 0.0):
        raise ValueError("every consensus mask must contain positive spatial mass")
    return masks


def _area_weights(input_size: int, output_size: int) -> np.ndarray:
    scale = input_size / output_size
    weights = np.zeros((output_size, input_size), dtype=np.float64)
    for output_index in range(output_size):
        start = output_index * scale
        stop = (output_index + 1) * scale
        first = int(math.floor(start))
        last = int(math.ceil(stop))
        for input_index in range(first, min(last, input_size)):
            overlap = max(
                0.0,
                min(stop, input_index + 1.0) - max(start, float(input_index)),
            )
            weights[output_index, input_index] = overlap / scale
    return weights


def _area_resize(source: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if source.shape == output_shape:
        return source.copy()
    row_weights = _area_weights(source.shape[0], output_shape[0])
    column_weights = _area_weights(source.shape[1], output_shape[1])
    return row_weights @ source @ column_weights.T


def _normalized_consensus(consensus: np.ndarray) -> np.ndarray:
    mass = float(consensus.sum())
    if mass <= 0.0:
        raise ValueError("resized consensus mask must contain positive spatial mass")
    return consensus / mass


def _spatial_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left.ravel() - float(left.mean())
    right_centered = right.ravel() - float(right.mean())
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    result = float(np.dot(left_centered, right_centered) / denominator)
    return float(np.clip(result, -1.0, 1.0))


def threshold_dice_iou(attention_map: Any, consensus_mask: Any) -> dict[str, Any]:
    """Compute fixed-threshold spatial Dice/IoU for one map pair.

    Attention foreground is at or above the per-map 0.90 quantile (NumPy
    ``linear`` method).  Consensus foreground is the area-resized soft mask at
    or above 0.5.  Neither threshold depends on labels or test performance.
    """

    attention = validate_attention_normalization(attention_map)
    consensus = _validate_consensus_masks(consensus_mask)
    if len(attention) != 1 or len(consensus) != 1:
        raise ValueError("threshold_dice_iou accepts exactly one attention/mask pair")
    attention_2d = attention[0]
    resized = _area_resize(consensus[0], attention_2d.shape)
    attention_threshold = float(
        np.quantile(
            attention_2d,
            ATTENTION_QUANTILE,
            method=ATTENTION_QUANTILE_METHOD,
        )
    )
    attention_foreground = attention_2d >= attention_threshold
    consensus_foreground = resized >= SOFT_MASK_THRESHOLD
    intersection = int(np.logical_and(attention_foreground, consensus_foreground).sum())
    attention_count = int(attention_foreground.sum())
    consensus_count = int(consensus_foreground.sum())
    union = int(np.logical_or(attention_foreground, consensus_foreground).sum())
    dice_denominator = attention_count + consensus_count
    dice = float(2 * intersection / dice_denominator) if dice_denominator else 1.0
    iou = float(intersection / union) if union else 1.0
    return {
        "dice": dice,
        "iou": iou,
        "attention_threshold": attention_threshold,
        "consensus_threshold": SOFT_MASK_THRESHOLD,
        "threshold_provenance": (
            "prespecified attention per-map 0.90 quantile (linear method) and "
            "soft-mask >= 0.5"
        ),
    }


def _identity_records(
    identifiers: Sequence[Mapping[str, Any]] | None, count: int
) -> list[dict[str, str]]:
    if identifiers is None:
        return [{} for _ in range(count)]
    if isinstance(identifiers, (str, bytes)):
        raise TypeError("identifiers must be a sequence of mappings")
    values = list(identifiers)
    if len(values) != count:
        raise ValueError(f"identifier/map row-count mismatch: {len(values)} != {count}")
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for position, record in enumerate(values):
        if not isinstance(record, Mapping):
            raise TypeError(f"identifier at position {position} must be a mapping")
        missing = [field for field in _IDENTIFIER_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"identifier at position {position} is missing fields: {missing!r}"
            )
        canonical = {
            field: str(record[field]).strip() for field in _IDENTIFIER_FIELDS
        }
        if any(not value for value in canonical.values()):
            raise ValueError(f"identifier at position {position} contains a blank value")
        key = tuple(canonical[field] for field in _IDENTIFIER_FIELDS)
        if key in seen:
            raise ValueError(f"duplicate attention identity at position {position}: {key!r}")
        seen.add(key)
        result.append(canonical)
    return result


def _aggregate(values: Sequence[float | None]) -> dict[str, Any]:
    defined = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(defined) == 0:
        return {"mean": None, "standard_deviation": None, "n_defined": 0}
    return {
        "mean": float(defined.mean()),
        "standard_deviation": (
            float(defined.std(ddof=1)) if len(defined) > 1 else None
        ),
        "n_defined": int(len(defined)),
    }


def attention_alignment_metrics(
    attention_maps: Any,
    consensus_masks: Any,
    *,
    identifiers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate continuous and prespecified-threshold alignment per nodule."""

    attention = validate_attention_normalization(attention_maps)
    # Validation rejects material normalization defects. Re-normalizing only
    # accepted numerical drift keeps KL/overlap exact probability metrics.
    attention = attention / attention.sum(axis=(1, 2), keepdims=True)
    consensus = _validate_consensus_masks(consensus_masks)
    if len(attention) != len(consensus):
        raise ValueError(f"attention/mask row-count mismatch: {len(attention)} != {len(consensus)}")
    identity = _identity_records(identifiers, len(attention))

    records: list[dict[str, Any]] = []
    for index in range(len(attention)):
        attention_map = attention[index]
        resized_mask = _area_resize(consensus[index], attention_map.shape)
        target = _normalized_consensus(resized_mask)
        positive_target = target > 0.0
        kl = float(
            (
                target[positive_target]
                * (
                    np.log(np.maximum(target[positive_target], _KL_EPSILON))
                    - np.log(np.maximum(attention_map[positive_target], _KL_EPSILON))
                )
            ).sum()
        )
        # Round-off can produce a negligible negative result for equal maps.
        if -1e-12 < kl < 0.0:
            kl = 0.0
        correlation = _spatial_correlation(attention_map, target)
        overlap = float(np.minimum(attention_map, target).sum())
        threshold = threshold_dice_iou(attention_map, resized_mask)
        records.append(
            {
                **identity[index],
                "kl_consensus_to_attention": kl,
                "spatial_correlation": correlation,
                "distribution_overlap": overlap,
                "dice": threshold["dice"],
                "iou": threshold["iou"],
                "attention_threshold": threshold["attention_threshold"],
                "consensus_threshold": threshold["consensus_threshold"],
            }
        )

    metric_names = (
        "kl_consensus_to_attention",
        "spatial_correlation",
        "distribution_overlap",
        "dice",
        "iou",
    )
    aggregate = {
        name: _aggregate([record[name] for record in records]) for name in metric_names
    }
    return {
        "n_samples": int(len(records)),
        "per_sample": records,
        "aggregate": aggregate,
        "aggregate_standard_deviation": (
            "sample standard deviation (ddof=1); null when fewer than two values are defined"
        ),
        "definition": ATTENTION_ALIGNMENT_DEFINITION,
    }


# Short aliases make the formulas easy to discover without creating alternate
# implementations.
multiclass_nll = multiclass_negative_log_likelihood
multiclass_brier = multiclass_brier_score
multiclass_ece = expected_calibration_error
spearman_disagreement_association = uncertainty_disagreement_association


__all__ = [
    "ATTENTION_ALIGNMENT_DEFINITION",
    "ATTENTION_NORMALIZATION_ATOL",
    "ATTENTION_QUANTILE",
    "ATTENTION_QUANTILE_METHOD",
    "CALIBRATION_DEFINITION",
    "CLASS_ORDER",
    "ECE_BIN_COUNT",
    "ECE_BIN_EDGES",
    "NLL_EPSILON",
    "RISK_COVERAGE_DEFINITION",
    "SOFT_MASK_THRESHOLD",
    "UNCERTAINTY_DEFINITION",
    "attention_alignment_metrics",
    "calibration_metrics",
    "expected_calibration_error",
    "high_low_disagreement_comparison",
    "multiclass_brier",
    "multiclass_brier_score",
    "multiclass_ece",
    "multiclass_negative_log_likelihood",
    "multiclass_nll",
    "predictive_entropy",
    "risk_coverage_curve",
    "spearman_association",
    "spearman_disagreement_association",
    "threshold_dice_iou",
    "uncertainty_disagreement_association",
    "validate_attention_normalization",
]
