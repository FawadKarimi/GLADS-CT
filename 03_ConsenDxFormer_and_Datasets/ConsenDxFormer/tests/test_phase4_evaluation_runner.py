from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_evaluation_runner import (
    EXPECTED_EVALUATION_CONFIG,
    Phase4EvaluationError,
    aggregate_seed_records,
    compute_saved_prediction_evaluation,
    create_evaluation_layout,
    load_phase4_evaluation_config,
    validate_final_source_descriptors,
    validate_seeded_test_cohorts,
)


def _descriptor(seed: int, *, model: str = "ConsenDxFormer") -> dict:
    common = {
        "model": model,
        "experiment_id": f"{model.lower()}-s{seed}-source",
        "seed": seed,
        "run_mode": "final",
        "class_order": ["benign", "indeterminate", "malignant"],
        "cohort_view": "paper_aligned_multi_reader_2_to_4",
        "sample_count": 1133,
        "test_sample_count": 183,
        "manifest_bundle_sha256": (
            "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
        ),
        "supervision_sha256": (
            "cbe4e402825665428ee890887963437d2bc480f8616bd7ec7999c667759ccc25"
        ),
        "ordered_supervision_sha256": (
            "2bae364b180f75cf2332714cc486a517540fcff3b9e94c48d83f59d0d9d6789a"
        ),
        "test_used_for_selection": False,
        "completed": True,
    }
    if model == "ConsenDxFormer":
        common.update(
            {
                "project_commit": "e1b174490245c616e355ef7c0c6f7c3c239d3acd",
                "lambda_ordinal": 0.5,
                "lambda_attention": 0.05,
            }
        )
    return common


def _predictions(seed: int, *, model: str) -> pd.DataFrame:
    rows = []
    for index, true_class in enumerate(("benign", "indeterminate", "malignant")):
        probabilities = [0.05, 0.05, 0.05]
        probabilities[index] = 0.9
        rows.append(
            {
                "patient_id": f"p{index}",
                "nodule_id": f"n{index}",
                "series_instance_uid": f"series{index}",
                "split": "test",
                "true_class": true_class,
                "predicted_class": true_class,
                "prob_benign": probabilities[0],
                "prob_indeterminate": probabilities[1],
                "prob_malignant": probabilities[2],
                "seed": seed,
                "model": model,
                "experiment_id": f"{model.lower()}-s{seed}-source",
            }
        )
    return pd.DataFrame(rows)


def test_evaluation_config_is_exact_and_prespecified(tmp_path):
    config = load_phase4_evaluation_config(
        ROOT / "configs" / "phase4_evaluation.json"
    )
    assert config == EXPECTED_EVALUATION_CONFIG
    assert config["final_seeds"] == [42, 43, 44]
    assert config["selected_lambdas"] == {
        "lambda_ordinal": 0.5,
        "lambda_attention": 0.05,
    }
    assert config["calibration"]["ece"]["bin_count"] == 15
    assert config["uncertainty"]["disagreement_threshold_rule"] == (
        "median_of_frozen_training_split"
    )
    assert config["safety"]["test_driven_tuning"] is False

    changed = deepcopy(config)
    changed["selected_lambdas"]["lambda_attention"] = 0.1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(Phase4EvaluationError, match="approved evaluation protocol"):
        load_phase4_evaluation_config(path)

    changed = deepcopy(config)
    changed["safety"]["test_driven_tuning"] = 0
    path = tmp_path / "integer-boolean.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(Phase4EvaluationError, match="approved evaluation protocol"):
        load_phase4_evaluation_config(path)


def test_final_source_descriptors_require_exact_seeds_lambdas_and_commit():
    records = [_descriptor(seed) for seed in (44, 42, 43)]
    canonical = validate_final_source_descriptors(records)
    assert [record["seed"] for record in canonical] == [42, 43, 44]

    wrong = deepcopy(records)
    wrong[0]["lambda_ordinal"] = 0.25
    with pytest.raises(Phase4EvaluationError, match="lambda_ordinal"):
        validate_final_source_descriptors(wrong)

    wrong = deepcopy(records)
    wrong[0]["project_commit"] = "0" * 40
    with pytest.raises(Phase4EvaluationError, match="project_commit"):
        validate_final_source_descriptors(wrong)

    with pytest.raises(Phase4EvaluationError, match="seeds"):
        validate_final_source_descriptors(records[:2])


def test_seeded_test_cohorts_require_exact_identity_and_label_matching():
    phase4 = {
        seed: _predictions(seed, model="ConsenDxFormer")
        for seed in (42, 43, 44)
    }
    b1 = {seed: _predictions(seed, model="B1") for seed in (42, 43, 44)}
    report = validate_seeded_test_cohorts(phase4, b1)
    assert report["matched"] is True
    assert report["seeds"] == [42, 43, 44]
    assert report["sample_count_per_seed"] == 3

    mismatched = {seed: frame.copy() for seed, frame in b1.items()}
    mismatched[43].loc[0, "nodule_id"] = "different"
    with pytest.raises(Phase4EvaluationError, match="cohort mismatch"):
        validate_seeded_test_cohorts(phase4, mismatched)


def test_aggregate_seed_records_uses_sample_standard_deviation():
    records = [
        {"seed": 42, "nll": 1.0, "brier": 0.2},
        {"seed": 43, "nll": 2.0, "brier": 0.4},
        {"seed": 44, "nll": 3.0, "brier": 0.6},
    ]
    aggregate = aggregate_seed_records(records, metric_fields=("nll", "brier"))
    assert aggregate["nll"] == {"mean": 2.0, "standard_deviation": 1.0}
    assert aggregate["brier"]["mean"] == pytest.approx(0.4)
    assert aggregate["seed_count"] == 3


def test_evaluation_layout_never_overwrites_sources_or_existing_output(tmp_path):
    source = tmp_path / "phase4-final-s42-source"
    source.mkdir()
    output_root = tmp_path / "evaluation"
    layout = create_evaluation_layout(
        output_root,
        "phase4-evaluation-01234567",
        protected_sources=(source,),
    )
    assert layout.output_dir.parent == output_root.resolve()
    assert layout.output_dir != source.resolve()

    with pytest.raises(FileExistsError, match="overwrite"):
        create_evaluation_layout(
            output_root,
            "phase4-evaluation-01234567",
            protected_sources=(source,),
        )

    with pytest.raises(Phase4EvaluationError, match="source experiment"):
        create_evaluation_layout(
            source,
            "phase4-evaluation-fedcba98",
            protected_sources=(source,),
        )


def test_saved_prediction_evaluation_uses_frozen_definitions_and_train_threshold(
    monkeypatch,
):
    phase4 = {
        seed: _predictions(seed, model="ConsenDxFormer")
        for seed in (42, 43, 44)
    }
    b1 = {seed: _predictions(seed, model="B1") for seed in (42, 43, 44)}
    supervision = pd.DataFrame(
        {
            "patient_id": ["train-a", "train-b", "train-c", "p0", "p1", "p2"],
            "nodule_id": ["tn0", "tn1", "tn2", "n0", "n1", "n2"],
            "series_instance_uid": ["ts0", "ts1", "ts2", "series0", "series1", "series2"],
            "split": ["train", "train", "train", "test", "test", "test"],
            "label": ["benign", "indeterminate", "malignant", "benign", "indeterminate", "malignant"],
            "disagreement_entropy": [0.1, 0.2, 0.9, 0.05, 0.4, 0.8],
        }
    )
    calls = []

    def fake_model(frame, *, model_name, seed, n_resamples, confidence_level):
        calls.append(("model", model_name, seed, n_resamples, confidence_level))
        return {"model": model_name, "bootstrap": {"seed": seed}}

    def fake_paired(left, right, *, seed, n_resamples, confidence_level):
        calls.append(("paired", seed, n_resamples, confidence_level))
        return {"comparison_direction": "ConsenDxFormer_minus_B1"}

    monkeypatch.setattr(
        "consendxformer.phase4_evaluation_runner.bootstrap_model_metrics",
        fake_model,
    )
    monkeypatch.setattr(
        "consendxformer.phase4_evaluation_runner.bootstrap_paired_differences",
        fake_paired,
    )
    result = compute_saved_prediction_evaluation(
        phase4,
        b1,
        supervision,
        evaluation_config=deepcopy(EXPECTED_EVALUATION_CONFIG),
    )
    assert result["calibration"]["aggregate"]["ConsenDxFormer"]["seed_count"] == 3
    assert result["calibration"]["aggregate"]["B1"]["seed_count"] == 3
    assert result["uncertainty"]["disagreement_threshold"] == pytest.approx(0.2)
    assert result["uncertainty"]["threshold_provenance"]["source_split"] == "train"
    assert result["definitions"]["test_driven_tuning"] is False
    assert all(item[3] == 10000 for item in calls if item[0] == "model")
    assert all(item[2] == 10000 for item in calls if item[0] == "paired")
    assert len([item for item in calls if item[0] == "model"]) == 6
    assert len([item for item in calls if item[0] == "paired"]) == 3
