from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_bootstrap import (
    BOOTSTRAP_METRICS,
    MATCH_COLUMNS,
    Phase4BootstrapError,
    bootstrap_model_metrics,
    bootstrap_paired_differences,
    patient_cluster_resample_indices,
    validate_paired_prediction_rows,
)


def _predictions(predicted: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p3", "p4"],
            "nodule_id": ["n1", "n2", "n3", "n4", "n5", "n6"],
            "series_instance_uid": ["s1", "s1", "s2", "s3", "s3", "s4"],
            "split": ["test"] * 6,
            "true_class": [
                "benign",
                "indeterminate",
                "malignant",
                "benign",
                "indeterminate",
                "malignant",
            ],
            "predicted_class": predicted,
        }
    )


def _b1_predictions() -> pd.DataFrame:
    return _predictions(
        [
            "benign",
            "benign",
            "malignant",
            "indeterminate",
            "malignant",
            "malignant",
        ]
    )


def _consendxformer_predictions() -> pd.DataFrame:
    return _predictions(
        [
            "benign",
            "indeterminate",
            "malignant",
            "benign",
            "indeterminate",
            "malignant",
        ]
    )


def _contains_forbidden_claim_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "p_value" in str(key).casefold()
            or "significant" in str(key).casefold()
            or _contains_forbidden_claim_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_claim_key(item) for item in value)
    return False


def test_patient_cluster_resampling_preserves_every_nodule_and_multiplicity():
    row_patients = pd.Series(["p1", "p1", "p2", "p3", "p3"])

    indices = patient_cluster_resample_indices(
        row_patients,
        sampled_patient_ids=["p3", "p1", "p3"],
    )

    assert indices.tolist() == [3, 4, 0, 1, 3, 4]
    assert [row_patients.iloc[index] for index in indices] == [
        "p3",
        "p3",
        "p1",
        "p1",
        "p3",
        "p3",
    ]


def test_bootstrap_absent_classes_keep_fixed_three_class_macro_definition():
    rows = _consendxformer_predictions().iloc[[0, 3]].copy()
    result = bootstrap_model_metrics(rows, model_name="ConsenDxFormer", seed=7, n_resamples=10)
    assert result["point_estimates"]["macro_f1"] == pytest.approx(1 / 3)
    assert result["point_estimates"]["indeterminate_recall"] == 0.0
    assert result["bootstrap"]["metric_definition"]["zero_denominator_value"] == 0.0


def test_model_bootstrap_is_deterministic_and_records_explicit_parameters():
    first = bootstrap_model_metrics(
        _b1_predictions(),
        model_name="B1",
        seed=2026,
        n_resamples=80,
        confidence_level=0.95,
    )
    second = bootstrap_model_metrics(
        _b1_predictions().sample(frac=1, random_state=9),
        model_name="B1",
        seed=2026,
        n_resamples=80,
        confidence_level=0.95,
    )
    different_seed = bootstrap_model_metrics(
        _b1_predictions(),
        model_name="B1",
        seed=2027,
        n_resamples=80,
        confidence_level=0.95,
    )

    assert first == second
    assert first["metrics"] == list(BOOTSTRAP_METRICS)
    assert first["bootstrap"]["sampling_unit"] == "patient_id"
    assert first["bootstrap"]["patient_grouping_preserved"] is True
    assert first["bootstrap"]["seed"] == 2026
    assert first["bootstrap"]["n_resamples"] == 80
    assert first["bootstrap"]["confidence_level"] == 0.95
    assert first["bootstrap_estimates"] != different_seed["bootstrap_estimates"]
    assert json.loads(json.dumps(first, allow_nan=False))["class_order"] == [
        "benign",
        "indeterminate",
        "malignant",
    ]


def test_exact_pair_matching_aligns_shuffled_rows_and_records_contract():
    b1 = _b1_predictions()
    consen = _consendxformer_predictions().iloc[[5, 2, 0, 4, 1, 3]]

    aligned_b1, aligned_consen, report = validate_paired_prediction_rows(b1, consen)

    assert aligned_b1.loc[:, list(MATCH_COLUMNS)].to_dict("records") == (
        aligned_consen.loc[:, list(MATCH_COLUMNS)].to_dict("records")
    )
    assert aligned_b1["nodule_id"].tolist() == ["n1", "n2", "n3", "n4", "n5", "n6"]
    assert report["matched"] is True
    assert report["match_columns"] == list(MATCH_COLUMNS)
    assert report["n_patients"] == 4
    assert report["n_nodules"] == 6


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("patient_id", "other-patient", "identity mismatch"),
        ("nodule_id", "other-nodule", "identity mismatch"),
        ("series_instance_uid", "other-series", "identity mismatch"),
        ("split", "val", "expected split"),
        ("true_class", "malignant", "true-class mismatch"),
    ],
)
def test_pair_matching_rejects_any_cohort_mismatch(column, replacement, message):
    b1 = _b1_predictions()
    consen = _consendxformer_predictions()
    consen.loc[0, column] = replacement

    with pytest.raises(Phase4BootstrapError, match=message):
        validate_paired_prediction_rows(b1, consen)


def test_pair_matching_rejects_duplicate_sample_identity():
    b1 = pd.concat([_b1_predictions(), _b1_predictions().iloc[[0]]], ignore_index=True)

    with pytest.raises(Phase4BootstrapError, match="duplicate sample identities"):
        validate_paired_prediction_rows(b1, _consendxformer_predictions())


def test_paired_bootstrap_uses_shared_patient_draws_and_reports_direction():
    result = bootstrap_paired_differences(
        _b1_predictions(),
        _consendxformer_predictions(),
        seed=51,
        n_resamples=100,
        confidence_level=0.90,
    )

    assert result["comparison_direction"] == "ConsenDxFormer_minus_B1"
    assert result["matching"]["matched"] is True
    assert result["bootstrap"]["paired_resampling"] is True
    assert result["bootstrap"]["sampling_unit"] == "patient_id"
    assert result["point_estimates"]["ConsenDxFormer"]["macro_f1"] == 1.0
    for metric in BOOTSTRAP_METRICS:
        assert result["point_estimates"]["difference"][metric] == pytest.approx(
            result["point_estimates"]["ConsenDxFormer"][metric]
            - result["point_estimates"]["B1"][metric]
        )
        assert result["bootstrap_estimates"]["difference"][metric] == pytest.approx(
            [
                consen - b1
                for b1, consen in zip(
                    result["bootstrap_estimates"]["B1"][metric],
                    result["bootstrap_estimates"]["ConsenDxFormer"][metric],
                )
            ]
        )
        interval = result["confidence_intervals"]["difference"][metric]
        assert interval["confidence_level"] == 0.90
        assert interval["lower"] <= interval["upper"]
    assert not _contains_forbidden_claim_key(result)


def test_bootstrap_does_not_mutate_prediction_tables():
    b1 = _b1_predictions()
    consen = _consendxformer_predictions()
    b1_before = copy.deepcopy(b1)
    consen_before = copy.deepcopy(consen)

    bootstrap_paired_differences(
        b1,
        consen,
        seed=8,
        n_resamples=10,
    )

    pd.testing.assert_frame_equal(b1, b1_before)
    pd.testing.assert_frame_equal(consen, consen_before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": True, "n_resamples": 10}, "seed"),
        ({"seed": -1, "n_resamples": 10}, "seed"),
        ({"seed": 1, "n_resamples": 0}, "n_resamples"),
        ({"seed": 1, "n_resamples": 10, "confidence_level": 1.0}, "confidence_level"),
    ],
)
def test_bootstrap_parameter_contracts_fail_closed(kwargs, message):
    with pytest.raises(Phase4BootstrapError, match=message):
        bootstrap_model_metrics(_b1_predictions(), model_name="B1", **kwargs)
