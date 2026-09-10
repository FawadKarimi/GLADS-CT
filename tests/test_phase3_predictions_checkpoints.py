from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase3_checkpoints import (
    CheckpointContractError,
    build_best_checkpoint,
    build_history_record,
    build_last_checkpoint,
    select_best_validation_epoch,
    validate_checkpoint_pair,
    validate_checkpoint_record,
    validate_epoch_history,
)
from consendxformer.phase3_predictions import (
    CLASS_ORDER,
    PREDICTION_COLUMNS,
    PredictionContractError,
    build_prediction_table,
    evaluate_phase3_predictions,
    load_phase3_prediction_csv,
    ordered_sample_checksum,
    stable_softmax,
    validate_prediction_table,
    write_prediction_csv,
)


@pytest.fixture
def prediction_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.DataFrame(
        {
            "patient_id": ["LIDC-0003", "LIDC-0001", "LIDC-0002"],
            "series_instance_uid": ["1.2.3", "1.2.1", "1.2.2"],
            "nodule_id": ["nodule-c", "nodule-a", "nodule-b"],
            "split": ["test", "val", "val"],
            "label": ["indeterminate", 0, 1],
        }
    )
    logits = np.asarray(
        [
            [0.0, 4.0, -2.0],
            [1000.0, 999.0, 998.0],
            [-4.0, -1.0, -2.0],
        ],
        dtype=np.float64,
    )
    return metadata, logits


def _prediction_table(
    prediction_inputs: tuple[pd.DataFrame, np.ndarray],
) -> pd.DataFrame:
    metadata, logits = prediction_inputs
    return build_prediction_table(
        metadata,
        logits,
        seed=17,
        baseline_variant="B0",
        experiment_id="phase3-b0-seed17",
    )


def test_prediction_table_has_fixed_schema_and_preserves_all_identities(
    prediction_inputs: tuple[pd.DataFrame, np.ndarray],
) -> None:
    table = _prediction_table(prediction_inputs)

    assert tuple(table.columns) == PREDICTION_COLUMNS
    assert table[["patient_id", "series_instance_uid", "nodule_id"]].to_dict(
        orient="records"
    ) == [
        {
            "patient_id": "LIDC-0001",
            "series_instance_uid": "1.2.1",
            "nodule_id": "nodule-a",
        },
        {
            "patient_id": "LIDC-0002",
            "series_instance_uid": "1.2.2",
            "nodule_id": "nodule-b",
        },
        {
            "patient_id": "LIDC-0003",
            "series_instance_uid": "1.2.3",
            "nodule_id": "nodule-c",
        },
    ]
    assert table["predicted_class"].tolist() == [
        "benign",
        "indeterminate",
        "indeterminate",
    ]
    assert table["true_class"].tolist() == [
        "benign",
        "indeterminate",
        "indeterminate",
    ]


def test_stable_softmax_and_prediction_validator_bind_probabilities_to_logits(
    prediction_inputs: tuple[pd.DataFrame, np.ndarray],
) -> None:
    table = _prediction_table(prediction_inputs)
    probabilities = table[
        ["prob_benign", "prob_indeterminate", "prob_malignant"]
    ].to_numpy(dtype=np.float64)
    logits = table[
        ["logit_benign", "logit_indeterminate", "logit_malignant"]
    ].to_numpy(dtype=np.float64)

    assert np.isfinite(stable_softmax(np.asarray([[1000.0, 999.0, -1000.0]]))).all()
    np.testing.assert_allclose(probabilities, stable_softmax(logits), rtol=0, atol=0)

    wrong_probability = table.copy()
    wrong_probability.loc[0, "prob_benign"] += 1e-4
    with pytest.raises(PredictionContractError, match="probability/logit mismatch"):
        validate_prediction_table(wrong_probability)

    wrong_argmax = table.copy()
    wrong_argmax.loc[0, "predicted_class"] = "malignant"
    with pytest.raises(PredictionContractError, match="does not equal logit argmax"):
        validate_prediction_table(wrong_argmax)


def test_prediction_builder_fails_instead_of_inventing_missing_identity() -> None:
    metadata = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "series_instance_uid": ["S1"],
            "split": ["val"],
            "label": ["benign"],
        }
    )
    with pytest.raises(PredictionContractError, match="nodule_id"):
        build_prediction_table(
            metadata,
            [[1.0, 0.0, 0.0]],
            seed=1,
            baseline_variant="B0",
            experiment_id="missing-identity",
        )


def test_prediction_export_is_byte_identical_across_input_order(
    prediction_inputs: tuple[pd.DataFrame, np.ndarray], tmp_path
) -> None:
    metadata, logits = prediction_inputs
    first = build_prediction_table(
        metadata,
        logits,
        seed=17,
        baseline_variant="B0",
        experiment_id="phase3-b0-seed17",
    )
    permutation = [2, 0, 1]
    second = build_prediction_table(
        metadata.iloc[permutation].reset_index(drop=True),
        logits[permutation],
        seed=17,
        baseline_variant="B0",
        experiment_id="phase3-b0-seed17",
    )

    first_path = write_prediction_csv(first, tmp_path / "first.csv")
    second_path = write_prediction_csv(second, tmp_path / "second.csv")

    assert first_path.read_bytes() == second_path.read_bytes()
    pd.testing.assert_frame_equal(
        load_phase3_prediction_csv(first_path),
        load_phase3_prediction_csv(second_path),
    )
    assert ordered_sample_checksum(first) == ordered_sample_checksum(second)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_prediction_csv(first, first_path)


def test_phase3_evaluator_explicitly_selects_probabilities_and_reports_undefined_auroc(
    prediction_inputs: tuple[pd.DataFrame, np.ndarray],
) -> None:
    # Both complete score families exist, which the shared evaluator regards as
    # ambiguous unless the Phase 3 adapter supplies probability_columns.
    metrics = evaluate_phase3_predictions(_prediction_table(prediction_inputs))

    assert metrics["class_order"] == list(CLASS_ORDER)
    assert metrics["score_source"] == "probabilities"
    assert metrics["score_columns"] == {
        "benign": "prob_benign",
        "indeterminate": "prob_indeterminate",
        "malignant": "prob_malignant",
    }
    assert metrics["auroc_ovr"]["malignant"] is None
    assert metrics["auroc_ovr"]["macro"] is None
    assert any("malignant" in warning for warning in metrics["warnings"])


def test_best_checkpoint_selection_uses_minimum_validation_loss_only() -> None:
    history = [
        build_history_record(epoch=1, train_loss=0.9, validation_loss=0.4),
        build_history_record(epoch=2, train_loss=0.7, validation_loss=0.2),
        build_history_record(epoch=3, train_loss=0.1, validation_loss=0.2),
    ]

    selection = select_best_validation_epoch(history)

    assert selection == {
        "epoch": 2,
        "value": 0.2,
        "criterion": "validation_loss",
        "mode": "min",
        "selection_split": "validation",
    }

    # Optimization loss may change without moving the validation-selected epoch.
    different_train_history = deepcopy(history)
    different_train_history[0]["train"]["loss"] = 0.001
    different_train_history[1]["train"]["loss"] = 999.0
    assert select_best_validation_epoch(different_train_history) == selection


def test_test_metrics_are_forbidden_from_checkpoint_selection_history() -> None:
    history = [
        build_history_record(epoch=1, train_loss=0.9, validation_loss=0.4),
        build_history_record(epoch=2, train_loss=0.8, validation_loss=0.3),
    ]
    history[0]["test"] = {"loss": 0.0}
    with pytest.raises(CheckpointContractError, match="test-derived field"):
        validate_epoch_history(history)

    nested_test = build_history_record(
        epoch=1,
        train_loss=0.9,
        validation_loss=0.4,
    )
    nested_test["validation"]["test_accuracy"] = 1.0
    with pytest.raises(CheckpointContractError, match="test-derived field"):
        select_best_validation_epoch([nested_test])


def test_best_and_last_checkpoint_metadata_are_complete_and_consistent() -> None:
    common = {
        "model_state": {"weight": [1.0]},
        "optimizer_state": {"step": 4},
        "scheduler_state": None,
        "best_validation_loss": 0.25,
        "seed": 17,
        "baseline_variant": "B1",
        "config_sha256": "a" * 64,
        "experiment_id": "phase3-b1-seed17",
    }
    best = build_best_checkpoint(epoch=2, **common)
    last = build_last_checkpoint(epoch=4, best_epoch=2, **common)

    validated_best, validated_last = validate_checkpoint_pair(best, last)
    assert validated_best["checkpoint_kind"] == "best"
    assert validated_last["checkpoint_kind"] == "last"
    assert validated_best["model_state_present"] is True
    assert validated_best["optimizer_state_present"] is True
    assert validated_best["scheduler_state_present"] is False
    assert validated_best["criterion"] == "validation_loss"
    assert validated_best["mode"] == "min"
    assert validated_best["selection_split"] == "validation"


def test_checkpoint_validator_rejects_false_presence_and_test_selection() -> None:
    checkpoint = build_best_checkpoint(
        model_state={"weight": [1.0]},
        optimizer_state={"step": 2},
        scheduler_state={"last_epoch": 2},
        epoch=2,
        best_validation_loss=0.25,
        seed=17,
        baseline_variant="B0",
        config_sha256="b" * 64,
        experiment_id="phase3-b0-seed17",
    )

    wrong_presence = dict(checkpoint, model_state_present=False)
    with pytest.raises(CheckpointContractError, match="disagrees"):
        validate_checkpoint_record(wrong_presence)

    test_selected = dict(checkpoint, selection_split="test")
    with pytest.raises(CheckpointContractError, match="selection_split"):
        validate_checkpoint_record(test_selected)
