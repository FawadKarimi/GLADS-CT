"""Boundary-only numerical preparation; never rewrite prediction evidence."""
from __future__ import annotations

from typing import Any

import numpy as np

from .baseline_evaluation import CLASS_ORDER


def boundary_entropy_policy() -> dict[str, Any]:
    """Describe the existing inference contract and entropy-only correction."""
    return {
        "schema_version": 1,
        "scope": "boundary_predictive_entropy_only",
        "class_order": list(CLASS_ORDER),
        "working_dtype": "float64",
        "input_contract": "nonempty [N,3], finite values in [0,1]",
        "raw_row_sum_rtol": 0.0,
        "raw_row_sum_atol": 1e-6,
        "acceptance_source": "unchanged predict_validated_boundary_cohort guard",
        "operation": "validate raw values first; divide a separate copy by its float64 row sums",
        "entropy_function": "unchanged phase4_evaluation_metrics.predictive_entropy",
        "entropy_logarithm": "natural",
        "shared_metric_guard_unchanged": True,
        "raw_probabilities_classes_confidence_and_probability_summaries_unchanged": True,
        "clipping_or_row_dropping_permitted": False,
        "diagnostics": "signed deviation is raw sum minus 1; factor is 1/raw sum; adjustment is max absolute normalized-minus-raw probability per row",
    }


def prepare_boundary_entropy_inputs(probabilities: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate before normalization and return private float64 entropy inputs.

    Columns must already be in CLASS_ORDER (the boundary runner uses its
    existing _probabilities extractor). This adapter is not used for primary
    or uncertainty results, or for class/confidence/probability summaries.
    """
    try:
        if np.iscomplexobj(probabilities):
            raise ValueError("complex probabilities are invalid")
        raw = np.array(probabilities, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Boundary entropy probabilities must be a real numeric matrix") from exc
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != len(CLASS_ORDER):
        raise ValueError("Boundary entropy probabilities must have nonempty shape [N,3]")
    if not np.isfinite(raw).all():
        raise ValueError("Boundary entropy probabilities must be finite")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("Boundary entropy probabilities must lie in [0,1]")
    row_sums = raw.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Boundary entropy probability sums exceed inference tolerance (rtol=0, atol=1e-6)")

    # Only accepted, nonzero-mass distributions reach division. Neither the
    # caller's array nor the private raw evidence is modified.
    normalized = raw.copy()
    normalized /= row_sums[:, None]
    deviations = row_sums - 1.0
    normalized_sums = normalized.sum(axis=1)
    adjustments = np.max(np.abs(normalized - raw), axis=1)
    return normalized, {
        "per_row": {
            "entropy_raw_probability_sum": row_sums.tolist(),
            "entropy_raw_probability_sum_deviation": deviations.tolist(),
            "entropy_raw_probability_sum_abs_deviation": np.abs(deviations).tolist(),
            "entropy_normalization_factor": (1.0 / row_sums).tolist(),
            "entropy_normalized_probability_sum": normalized_sums.tolist(),
            "entropy_max_abs_probability_adjustment": adjustments.tolist(),
        },
        "summary": {
            "row_count": len(raw),
            "rows_with_nonunit_raw_mass": int(np.count_nonzero(deviations)),
            "min_raw_probability_sum_deviation": float(deviations.min()),
            "max_raw_probability_sum_deviation": float(deviations.max()),
            "max_abs_raw_probability_sum_deviation": float(np.abs(deviations).max()),
            "max_abs_normalized_probability_sum_deviation": float(np.abs(normalized_sums - 1.0).max()),
            "max_abs_probability_adjustment": float(adjustments.max()),
        },
    }
