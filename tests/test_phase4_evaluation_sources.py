from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase3_artifacts import atomic_write_json, atomic_write_text
from consendxformer.phase3_predictions import build_prediction_table
from consendxformer.phase4_evaluation_runner import (
    Phase4EvaluationError,
    assert_source_snapshot_unchanged,
    snapshot_source_files,
    validate_b1_prediction_source,
    validate_phase4_prediction_source,
    verify_completed_source_artifact,
)
from consendxformer.phase4_predictions import build_phase4_prediction_table


def _metadata(count: int = 183) -> pd.DataFrame:
    classes = ("benign", "indeterminate", "malignant")
    return pd.DataFrame(
        {
            "patient_id": [f"p{index // 2}" for index in range(count)],
            "nodule_id": [f"n{index}" for index in range(count)],
            "series_instance_uid": [f"series{index}" for index in range(count)],
            "split": ["test"] * count,
            "label": [classes[index % 3] for index in range(count)],
        }
    )


def _logits(metadata: pd.DataFrame) -> np.ndarray:
    by_class = {"benign": 0, "indeterminate": 1, "malignant": 2}
    result = np.full((len(metadata), 3), -2.0, dtype=np.float64)
    for index, label in enumerate(metadata["label"]):
        result[index, by_class[label]] = 2.0
    return result


def test_completed_source_manifest_is_verified_and_tampering_fails(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payload = atomic_write_text(source / "predictions.csv", "a,b\n1,2\n")
    manifest = atomic_write_json(
        source / "artifact_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": "phase4-final-s42-source",
            "run_mode": "final",
            "research_results_eligible": True,
            "test_results_eligible": True,
            "files": {"predictions.csv": sha256_file(payload)},
            "bundle_sha256": sha256_json(
                {"predictions.csv": sha256_file(payload)}
            ),
        },
    )
    atomic_write_json(
        source / "COMPLETED.json",
        {
            "schema_version": 1,
            "experiment_id": "phase4-final-s42-source",
            "run_mode": "final",
            "artifact_manifest": "artifact_manifest.json",
            "artifact_manifest_sha256": sha256_file(manifest),
            "research_results_eligible": True,
            "test_results_eligible": True,
        },
    )
    report = verify_completed_source_artifact(
        source,
        required_members=("predictions.csv",),
        expected_experiment_id="phase4-final-s42-source",
    )
    assert report["verified"] is True

    payload.write_text("tampered", encoding="utf-8")
    with pytest.raises(Phase4EvaluationError, match="checksum"):
        verify_completed_source_artifact(
            source,
            required_members=("predictions.csv",),
            expected_experiment_id="phase4-final-s42-source",
        )


def test_phase4_and_b1_prediction_sources_preserve_exact_test_cohort():
    metadata = _metadata()
    logits = _logits(metadata)
    phase4 = build_phase4_prediction_table(
        metadata,
        logits,
        seed=42,
        lambda_ordinal=0.5,
        lambda_attention=0.05,
        run_mode="final",
        experiment_id="phase4-final-s42-source",
    )
    b1 = build_prediction_table(
        metadata,
        logits,
        seed=42,
        baseline_variant="B1",
        experiment_id="phase3-b1-s42-source",
    )
    descriptor = {
        "experiment_id": "phase4-final-s42-source",
        "seed": 42,
        "test_sample_count": 183,
        "lambda_ordinal": 0.5,
        "lambda_attention": 0.05,
    }
    validated_phase4 = validate_phase4_prediction_source(
        phase4, metadata, descriptor=descriptor
    )
    validated_b1 = validate_b1_prediction_source(b1, seed=42)
    assert len(validated_phase4) == len(validated_b1) == 183

    changed = metadata.copy()
    changed.loc[0, "nodule_id"] = "different"
    with pytest.raises(Phase4EvaluationError, match="supervision"):
        validate_phase4_prediction_source(phase4, changed, descriptor=descriptor)


def test_source_file_snapshot_detects_mutation(tmp_path):
    source = tmp_path / "immutable.txt"
    source.write_text("before", encoding="utf-8")
    snapshot = snapshot_source_files((source,))
    assert_source_snapshot_unchanged(snapshot)
    source.write_text("after", encoding="utf-8")
    with pytest.raises(Phase4EvaluationError, match="changed during evaluation"):
        assert_source_snapshot_unchanged(snapshot)
