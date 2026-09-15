from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_evaluation_metrics import (
    ATTENTION_ALIGNMENT_DEFINITION,
    CALIBRATION_DEFINITION,
    CLASS_ORDER,
    ECE_BIN_EDGES,
    RISK_COVERAGE_DEFINITION,
    attention_alignment_metrics,
    expected_calibration_error,
    high_low_disagreement_comparison,
    multiclass_brier_score,
    multiclass_negative_log_likelihood,
    predictive_entropy,
    risk_coverage_curve,
    spearman_association,
    threshold_dice_iou,
    validate_attention_normalization,
)


def test_calibration_definitions_freeze_class_order_and_ece_bins():
    assert CLASS_ORDER == ("benign", "indeterminate", "malignant")
    assert len(ECE_BIN_EDGES) == 16
    assert ECE_BIN_EDGES[0] == 0.0
    assert ECE_BIN_EDGES[-1] == 1.0
    assert CALIBRATION_DEFINITION["ece"]["n_bins"] == 15
    assert CALIBRATION_DEFINITION["ece"]["bin_membership"] == (
        "[edge_i, edge_(i+1)) except the final bin [edge_14, 1.0]"
    )


def test_multiclass_negative_log_likelihood_formula():
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]
    )
    expected = -(math.log(0.8) + math.log(0.5) + math.log(0.7)) / 3

    assert multiclass_negative_log_likelihood(
        ["benign", "indeterminate", "malignant"], probabilities
    ) == pytest.approx(expected)


def test_multiclass_brier_is_mean_sum_of_three_squared_errors():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    one_hot = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    assert multiclass_brier_score([0, 1], probabilities) == pytest.approx(expected)


def test_top_label_ece_formula_and_exact_edge_membership():
    # 0.6 is exactly a bin edge and belongs to [0.6, ...), while 1.0 belongs
    # to the closed final bin. Correctness is [1, 0, 1].
    probabilities = np.asarray(
        [[0.60, 0.25, 0.15], [0.20, 0.60, 0.20], [0.00, 0.00, 1.00]]
    )
    details = expected_calibration_error(
        [0, 0, 2], probabilities, return_details=True
    )
    # The first two observations share one bin: accuracy=.5, confidence=.6;
    # final-bin contribution is zero.
    assert details["ece"] == pytest.approx((2 / 3) * 0.1)
    populated = [entry for entry in details["bins"] if entry["count"]]
    assert [entry["count"] for entry in populated] == [2, 1]
    assert populated[0]["lower_edge"] == pytest.approx(0.6)
    assert populated[-1]["upper_inclusive"] is True


def test_probability_contract_rejects_wrong_order_shape_and_non_unit_mass():
    with pytest.raises(ValueError, match="three columns"):
        multiclass_brier_score([0], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="sum to 1"):
        predictive_entropy([[0.4, 0.4, 0.4]])
    with pytest.raises(ValueError, match="Unknown class"):
        multiclass_negative_log_likelihood(["borderline"], [[0.4, 0.3, 0.3]])


def test_predictive_entropy_uses_natural_log_and_zero_log_zero_is_zero():
    values = predictive_entropy([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    assert values.tolist() == pytest.approx([0.0, math.log(3.0)])


def test_spearman_association_uses_average_tie_ranks():
    result = spearman_association([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0])
    assert result["coefficient"] == pytest.approx(1.0)
    assert result["status"] == "defined"
    assert result["n"] == 4


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        ([1.0], [2.0], "fewer_than_two_observations"),
        ([1.0, 1.0], [2.0, 3.0], "constant_rank_input"),
    ],
)
def test_spearman_degenerate_inputs_are_explicitly_undefined(left, right, reason):
    result = spearman_association(left, right)
    assert result == {
        "coefficient": None,
        "n": len(left),
        "status": "undefined",
        "reason": reason,
    }


def test_entropy_risk_coverage_uses_identity_tie_break_and_discrete_aurc():
    # a and b have tied entropy. Identity order retains a (correct) before b
    # (incorrect), regardless of their input order.
    probabilities = np.asarray(
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.05, 0.05, 0.9]]
    )
    result = risk_coverage_curve(
        [0, 0, 2], probabilities, ranking="entropy", identifiers=["b", "a", "c"]
    )
    assert result["ordered_identifiers"] == ["c", "a", "b"]
    assert result["risk"] == pytest.approx([0.0, 0.0, 1 / 3])
    assert result["coverage"] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert result["aurc"] == pytest.approx(1 / 9)
    assert RISK_COVERAGE_DEFINITION["risk"] == "mean 0/1 classification error among retained samples"


def test_confidence_risk_coverage_orders_descending_confidence():
    probabilities = np.asarray(
        [[0.6, 0.3, 0.1], [0.05, 0.05, 0.9], [0.2, 0.5, 0.3]]
    )
    result = risk_coverage_curve(
        [0, 2, 1], probabilities, ranking="confidence", identifiers=["a", "b", "c"]
    )
    assert result["ordered_identifiers"] == ["b", "a", "c"]
    assert result["risk"] == [0.0, 0.0, 0.0]
    assert result["aurc"] == 0.0


def test_median_disagreement_comparison_is_outcome_independent_and_records_provenance():
    probabilities = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    result = high_low_disagreement_comparison(probabilities, [0.1, 0.2, 0.8, 0.9])
    assert result["threshold"] == pytest.approx(0.5)
    assert result["rule"] == "low: disagreement < threshold; high: disagreement >= threshold"
    assert result["threshold_provenance"] == "median of supplied disagreement entropy values only"
    assert result["threshold_was_supplied"] is False
    assert result["groups"] == ["low", "low", "high", "high"]
    assert result["low"]["count"] == result["high"]["count"] == 2
    assert result["high"]["mean_predictive_entropy"] > result["low"]["mean_predictive_entropy"]


def test_disagreement_comparison_applies_frozen_train_threshold_and_high_tie_policy():
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [1 / 3, 1 / 3, 1 / 3]]
    )
    result = high_low_disagreement_comparison(
        probabilities,
        [0.49, 0.50, 0.90],
        threshold=0.50,
        threshold_provenance="median of frozen train supervision disagreement entropy",
    )
    assert result["groups"] == ["low", "high", "high"]
    assert result["threshold_was_supplied"] is True
    assert result["tie_policy"] == "high_if_greater_than_or_equal_to_threshold"
    assert result["threshold_provenance"].startswith("median of frozen train")


def test_explicit_disagreement_threshold_requires_provenance_and_fixed_tie_policy():
    probabilities = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
    with pytest.raises(ValueError, match="requires non-empty threshold_provenance"):
        high_low_disagreement_comparison(
            probabilities, [0.1, 0.9], threshold=0.5
        )
    with pytest.raises(ValueError, match="tie_policy"):
        high_low_disagreement_comparison(
            probabilities,
            [0.1, 0.9],
            threshold=0.5,
            threshold_provenance="train median",
            tie_policy="low_on_equal",
        )


def test_attention_normalization_is_validated_not_silently_repaired():
    valid = np.asarray([[[0.1, 0.2], [0.3, 0.4]]])
    assert np.array_equal(validate_attention_normalization(valid), valid)
    with pytest.raises(ValueError, match="unit spatial mass"):
        validate_attention_normalization(valid * 2)
    with pytest.raises(ValueError, match="non-negative"):
        validate_attention_normalization([[[-0.1, 0.2], [0.3, 0.6]]])


def test_attention_alignment_metrics_match_identical_spatial_distributions():
    attention = np.asarray([[0.0, 0.0], [0.5, 0.5]])
    consensus = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    result = attention_alignment_metrics(attention, consensus)
    sample = result["per_sample"][0]
    assert sample["kl_consensus_to_attention"] == pytest.approx(0.0)
    assert sample["spatial_correlation"] == pytest.approx(1.0)
    assert sample["distribution_overlap"] == pytest.approx(1.0)
    assert sample["dice"] == pytest.approx(1.0)
    assert sample["iou"] == pytest.approx(1.0)
    assert ATTENTION_ALIGNMENT_DEFINITION["interpretation"].startswith(
        "attention-alignment / spatial-correspondence"
    )


def test_attention_alignment_area_resizes_mask_and_preserves_identifiers():
    attention = np.asarray([[[0.5, 0.0], [0.0, 0.5]]])
    consensus = np.asarray(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ]
    )
    identity = {"patient_id": "p1", "nodule_id": "n1", "series_instance_uid": "s1"}
    result = attention_alignment_metrics(attention, consensus, identifiers=[identity])
    assert result["per_sample"][0] == pytest.approx(
        {
            **identity,
            "kl_consensus_to_attention": 0.0,
            "spatial_correlation": 1.0,
            "distribution_overlap": 1.0,
            "dice": 1.0,
            "iou": 1.0,
            "attention_threshold": 0.5,
            "consensus_threshold": 0.5,
        }
    )


def test_dice_iou_threshold_contract_is_fixed_and_not_outcome_selected():
    attention = np.asarray([[0.01, 0.09], [0.40, 0.50]])
    consensus = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    result = threshold_dice_iou(attention, consensus)
    expected_attention_threshold = np.quantile(attention, 0.90, method="linear")
    assert result["attention_threshold"] == pytest.approx(expected_attention_threshold)
    assert result["consensus_threshold"] == 0.5
    assert result["threshold_provenance"] == (
        "prespecified attention per-map 0.90 quantile (linear method) and soft-mask >= 0.5"
    )
    assert result["dice"] == 0.0
    assert result["iou"] == 0.0


def test_attention_correlation_degeneracy_is_reported_as_null():
    attention = np.full((2, 2), 0.25)
    consensus = np.ones((2, 2))
    result = attention_alignment_metrics(attention, consensus)
    sample = result["per_sample"][0]
    assert sample["spatial_correlation"] is None
    assert sample["kl_consensus_to_attention"] == pytest.approx(0.0)
    assert result["aggregate"]["kl_consensus_to_attention"] == {
        "mean": pytest.approx(0.0),
        "standard_deviation": None,
        "n_defined": 1,
    }
    assert result["aggregate_standard_deviation"].startswith(
        "sample standard deviation (ddof=1)"
    )
