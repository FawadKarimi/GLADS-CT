from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_evaluation import (
    CLASS_ORDER,
    evaluate_prediction_csv,
    evaluate_predictions,
    write_evaluation_json,
)
from consendxformer import baseline_evaluator
from consendxformer.baseline_source import SourceGuardReport, snapshot_source_tree
from consendxformer.hashing import sha256_json
from consendxformer.manifest_guard import validate_frozen_manifest_bundle


MANIFEST_DIR = ROOT / "artifacts" / "phase0" / "manifests"


def _install_evaluation_guards(tmp_path, monkeypatch):
    source_root = tmp_path / "locked-baseline"
    source_root.mkdir()
    (source_root / "source-marker.txt").write_text("unchanged\n", encoding="utf-8")
    manifest_report = validate_frozen_manifest_bundle(MANIFEST_DIR)
    events: list[str] = []

    def source_guard(repository_path, _lock_file):
        assert Path(repository_path).resolve() == source_root.resolve()
        events.append("source")
        return SourceGuardReport(
            repository_path=str(source_root.resolve()),
            lock_file=str(ROOT / "locks" / "baseline_sources.yaml"),
            repository_url="https://github.com/Srinivasa459/LungDxFormer.git",
            expected_commit="0e5bcadcadfce5ea99df8484d23705d7bd4d778a",
            head_commit="0e5bcadcadfce5ea99df8484d23705d7bd4d778a",
            working_tree_clean=True,
            source_tree_sha256=sha256_json(snapshot_source_tree(source_root)),
        )

    def manifest_guard(manifest_path):
        assert Path(manifest_path).resolve() == MANIFEST_DIR.resolve()
        events.append("manifest")
        return manifest_report

    monkeypatch.setattr(baseline_evaluator, "verify_baseline_source", source_guard)
    monkeypatch.setattr(
        baseline_evaluator, "validate_frozen_manifest_bundle", manifest_guard
    )
    monkeypatch.setattr(
        baseline_evaluator,
        "project_git_state",
        lambda _root: {
            "commit": "095bd166cd5ac4750197101dbd6a89b4422d6500",
            "branch": "phase1-baselines",
            "working_tree_clean": False,
        },
    )
    return source_root, manifest_report, events


def _base_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(6)],
            "true_label": [
                "benign",
                "benign",
                "indeterminate",
                "indeterminate",
                "malignant",
                "malignant",
            ],
            "predicted_class": [
                "benign",
                "indeterminate",
                "indeterminate",
                "malignant",
                "malignant",
                "malignant",
            ],
        }
    )


def test_fixed_class_order_confusion_and_classification_metrics():
    result = evaluate_predictions(_base_predictions())

    assert CLASS_ORDER == ("benign", "indeterminate", "malignant")
    assert result["class_order"] == list(CLASS_ORDER)
    assert result["n_samples"] == 6
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_precision"] == pytest.approx((1.0 + 0.5 + 2 / 3) / 3)
    assert result["macro_recall"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0.8) / 3)
    assert result["per_class"] == {
        "benign": {
            "precision": 1.0,
            "recall": 0.5,
            "f1": pytest.approx(2 / 3),
            "support": 2,
        },
        "indeterminate": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "support": 2,
        },
        "malignant": {
            "precision": pytest.approx(2 / 3),
            "recall": 1.0,
            "f1": pytest.approx(0.8),
            "support": 2,
        },
    }


def test_no_scores_has_stable_null_auroc_schema_and_warning():
    result = evaluate_predictions(_base_predictions())

    assert result["score_source"] is None
    assert result["score_columns"] is None
    assert result["auroc_ovr"] == {
        "benign": None,
        "indeterminate": None,
        "malignant": None,
        "macro": None,
    }
    assert any("no complete" in warning for warning in result["warnings"])


def test_complete_probabilities_produce_one_vs_rest_auroc():
    frame = pd.DataFrame(
        {
            "identifier": ["a", "b", "c", "d", "e", "f"],
            "true_label": ["benign", "benign", "indeterminate", "indeterminate", "malignant", "malignant"],
            "predicted_class": ["benign", "benign", "indeterminate", "indeterminate", "malignant", "malignant"],
            "prob_benign": [0.90, 0.80, 0.10, 0.10, 0.05, 0.05],
            "prob_indeterminate": [0.05, 0.10, 0.80, 0.75, 0.10, 0.05],
            "prob_malignant": [0.05, 0.10, 0.10, 0.15, 0.85, 0.90],
        }
    )
    result = evaluate_predictions(frame)

    assert result["score_source"] == "probabilities"
    assert result["score_columns"] == {
        "benign": "prob_benign",
        "indeterminate": "prob_indeterminate",
        "malignant": "prob_malignant",
    }
    assert result["auroc_ovr"] == {
        "benign": 1.0,
        "indeterminate": 1.0,
        "malignant": 1.0,
        "macro": 1.0,
    }
    assert result["warnings"] == []


def test_logits_are_softmaxed_and_custom_columns_follow_canonical_order():
    frame = pd.DataFrame(
        {
            "case": ["a", "b", "c", "d", "e", "f"],
            "truth": [0, 0, 1, 1, 2, 2],
            "choice": [0, 0, 1, 1, 2, 2],
            "score_0": [8, 7, 0, 0, 0, 0],
            "score_1": [0, 0, 8, 7, 0, 0],
            "score_2": [0, 0, 0, 0, 8, 7],
        }
    )
    result = evaluate_predictions(
        frame,
        identifier_column="case",
        true_label_column="truth",
        predicted_class_column="choice",
        logit_columns={
            "malignant": "score_2",
            "benign": "score_0",
            "indeterminate": "score_1",
        },
    )

    assert result["score_source"] == "logits"
    assert result["auroc_ovr"]["macro"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_tied_scores_use_average_rank_auroc():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f"],
            "true_label": [0, 0, 1, 1, 2, 2],
            "predicted_class": [0, 0, 1, 1, 2, 2],
            "prob_benign": [1 / 3] * 6,
            "prob_indeterminate": [1 / 3] * 6,
            "prob_malignant": [1 / 3] * 6,
        }
    )
    result = evaluate_predictions(frame)

    assert result["auroc_ovr"] == {
        "benign": 0.5,
        "indeterminate": 0.5,
        "malignant": 0.5,
        "macro": 0.5,
    }


def test_absent_class_has_null_class_and_macro_auroc():
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "true_label": ["benign", "benign", "indeterminate", "indeterminate"],
            "predicted_class": ["benign", "benign", "indeterminate", "indeterminate"],
            "prob_benign": [0.9, 0.8, 0.1, 0.2],
            "prob_indeterminate": [0.1, 0.2, 0.8, 0.7],
            "prob_malignant": [0.0, 0.0, 0.1, 0.1],
        }
    )
    result = evaluate_predictions(frame)

    assert result["auroc_ovr"]["benign"] == 1.0
    assert result["auroc_ovr"]["indeterminate"] == 1.0
    assert result["auroc_ovr"]["malignant"] is None
    assert result["auroc_ovr"]["macro"] is None
    assert result["per_class"]["malignant"]["support"] == 0
    assert len(result["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in result["confusion_matrix"])
    assert any("malignant" in warning and "positives=0" in warning for warning in result["warnings"])
    assert any("Macro" in warning for warning in result["warnings"])


@pytest.mark.parametrize("bad_label", ["probably-benign", 3, -1, 1.5, True, None])
def test_invalid_labels_are_rejected_with_class_order(bad_label):
    frame = _base_predictions()
    # pandas 3 rejects mixed scalar assignment to its inferred string dtype;
    # object dtype represents the malformed external input we want to test.
    frame["true_label"] = frame["true_label"].astype(object)
    frame.loc[0, "true_label"] = bad_label

    with pytest.raises(ValueError, match="Allowed labels.*benign.*indeterminate.*malignant"):
        evaluate_predictions(frame)


def test_blank_identifier_and_empty_table_are_rejected():
    frame = _base_predictions()
    frame.loc[2, "sample_id"] = "  "
    with pytest.raises(ValueError, match="missing or blank"):
        evaluate_predictions(frame)

    with pytest.raises(ValueError, match="no rows"):
        evaluate_predictions(frame.iloc[0:0])


def test_partial_or_ambiguous_score_sets_are_rejected():
    partial = _base_predictions().assign(prob_benign=0.5, prob_indeterminate=0.5)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_predictions(partial)

    both = _base_predictions().assign(
        prob_benign=0.6,
        prob_indeterminate=0.3,
        prob_malignant=0.1,
        logit_benign=1.0,
        logit_indeterminate=0.0,
        logit_malignant=-1.0,
    )
    with pytest.raises(ValueError, match="Multiple complete"):
        evaluate_predictions(both)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("prob_benign", 1.1, r"outside \[0, 1\]"),
        ("prob_indeterminate", "not-a-number", "non-numeric"),
    ],
)
def test_invalid_probability_values_are_rejected(column, value, message):
    frame = _base_predictions().assign(
        prob_benign=0.6,
        prob_indeterminate=0.3,
        prob_malignant=0.1,
    )
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(frame)


def test_probabilities_must_sum_to_one():
    frame = _base_predictions().assign(
        prob_benign=0.5,
        prob_indeterminate=0.3,
        prob_malignant=0.1,
    )
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate_predictions(frame)


def test_csv_reader_preserves_text_identifier_and_json_is_strict(tmp_path):
    prediction_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "metrics.json"
    frame = _base_predictions().iloc[:3].copy()
    frame["sample_id"] = ["001", "002", "003"]
    frame.to_csv(prediction_path, index=False)

    result = evaluate_prediction_csv(prediction_path)
    write_evaluation_json(result, output_path)
    parsed = json.loads(output_path.read_text(encoding="utf-8"))

    assert parsed == result
    assert "NaN" not in output_path.read_text(encoding="utf-8")
    assert "Infinity" not in output_path.read_text(encoding="utf-8")


def test_writer_rejects_non_json_finite_metrics(tmp_path):
    with pytest.raises(ValueError, match="Out of range float"):
        write_evaluation_json({"bad": np.nan}, tmp_path / "bad.json")


def test_guarded_b1_cli_writes_json_only_from_supplied_csv(
    tmp_path, monkeypatch
):
    prediction_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "nested" / "metrics.json"
    _base_predictions().to_csv(prediction_path, index=False)
    source_root, manifest_report, events = _install_evaluation_guards(
        tmp_path, monkeypatch
    )

    returncode = baseline_evaluator.baseline_evaluation_cli(
        [
            "--mode", "B1",
            "--predictions-csv", str(prediction_path),
            "--output-json", str(output_path),
            "--baseline-dir", str(source_root),
            "--manifest-dir", str(MANIFEST_DIR),
        ]
    )

    assert returncode == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["n_samples"] == 6
    assert result["baseline_context"]["experiment_type"] == "B1"
    assert result["baseline_context"]["baseline"]["head_commit"] == (
        "0e5bcadcadfce5ea99df8484d23705d7bd4d778a"
    )
    assert result["baseline_context"]["manifest"]["bundle_sha256"] == (
        manifest_report.bundle_sha256
    )
    assert result["baseline_context"]["prediction_source"] == "caller_supplied"
    assert result["baseline_context"]["training_success_claimed"] is False
    assert events == ["source", "manifest", "source"]


def test_guard_failure_prevents_prediction_evaluation_and_output(
    tmp_path, monkeypatch
):
    prediction_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "metrics.json"
    _base_predictions().to_csv(prediction_path, index=False)
    source_root = tmp_path / "bad-baseline"
    source_root.mkdir()

    def rejected_source(*_args, **_kwargs):
        raise RuntimeError("locked source rejected")

    monkeypatch.setattr(
        baseline_evaluator, "verify_baseline_source", rejected_source
    )
    monkeypatch.setattr(
        baseline_evaluator,
        "evaluate_prediction_csv",
        lambda *_args, **_kwargs: pytest.fail("predictions read before source guard"),
    )

    with pytest.raises(RuntimeError, match="locked source rejected"):
        baseline_evaluator.run_baseline_evaluation(
            "B0",
            config_path=ROOT / "configs" / "b0.yaml",
            predictions_csv=prediction_path,
            output_json=output_path,
            baseline_dir=source_root,
            manifest_dir=MANIFEST_DIR,
        )
    assert not output_path.exists()


def test_cli_cannot_run_without_supplied_predictions(tmp_path):
    output_path = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "12_evaluate_baseline.py"),
            "--mode",
            "B0",
            "--output-json",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not output_path.exists()
