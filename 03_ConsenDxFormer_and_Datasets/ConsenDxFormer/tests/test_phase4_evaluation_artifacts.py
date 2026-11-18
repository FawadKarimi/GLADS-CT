from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase4_bootstrap import BOOTSTRAP_METRICS
from consendxformer.phase4_evaluation_runner import (
    EXPECTED_EVALUATION_CONFIG,
    FINAL_SEEDS,
    SavedPredictionInputs,
    SavedPredictionSource,
    write_saved_prediction_evaluation,
)
import consendxformer.phase4_evaluation_runner as phase4_evaluation_runner


def _inputs(tmp_path: Path):
    cohort_rows = [
        {
            "patient_id": f"p{index}",
            "nodule_id": f"n{index}",
            "series_instance_uid": f"s{index}",
            "split": "test",
            "true_class": class_name,
        }
        for index, class_name in enumerate(
            ("benign", "indeterminate", "malignant"), start=1
        )
    ]
    supervision = pd.DataFrame(
        [
            {
                "patient_id": f"train-p{index}",
                "nodule_id": f"train-n{index}",
                "series_instance_uid": f"train-s{index}",
                "split": "train",
                "label": class_name,
                "disagreement_entropy": value,
            }
            for index, (class_name, value) in enumerate(
                zip(
                    ("benign", "indeterminate", "malignant"),
                    (0.1, 0.2, 0.3),
                ),
                start=1,
            )
        ]
        + [
            {
                "patient_id": row["patient_id"],
                "nodule_id": row["nodule_id"],
                "series_instance_uid": row["series_instance_uid"],
                "split": "test",
                "label": row["true_class"],
                "disagreement_entropy": 0.1 * index,
            }
            for index, row in enumerate(cohort_rows, start=1)
        ]
    )
    phase4 = {}
    b1 = {}
    snapshot = {}
    source_ids = {"B1": {}, "ConsenDxFormer": {}}
    for model, destination in (("B1", b1), ("ConsenDxFormer", phase4)):
        for seed in FINAL_SEEDS:
            source_dir = tmp_path / "sources" / f"{model.lower()}-{seed}"
            source_dir.mkdir(parents=True)
            source_file = source_dir / "immutable.txt"
            source_file.write_text(f"{model}-{seed}", encoding="utf-8")
            digest = sha256_file(source_file)
            experiment_id = f"{model.lower()}-s{seed}-source"
            source_ids[model][str(seed)] = experiment_id
            prediction_rows = []
            for index, row in enumerate(cohort_rows):
                probabilities = [0.05, 0.05, 0.05]
                probabilities[index] = 0.9
                prediction_rows.append(
                    {
                        **row,
                        "predicted_class": row["true_class"],
                        "prob_benign": probabilities[0],
                        "prob_indeterminate": probabilities[1],
                        "prob_malignant": probabilities[2],
                        "seed": seed,
                        "model": model,
                        "experiment_id": experiment_id,
                    }
                )
            source = SavedPredictionSource(
                model=model,
                source_dir=source_dir.resolve(),
                descriptor={"experiment_id": experiment_id, "seed": seed},
                predictions=pd.DataFrame(prediction_rows),
                supervision=(
                    supervision.copy(deep=True)
                    if model == "ConsenDxFormer"
                    else None
                ),
                verification={
                    "artifact_manifest_sha256": "a" * 64,
                    "artifact_bundle_sha256": digest,
                    "files": {"immutable.txt": digest},
                },
                source_snapshot={str(source_file.resolve()): digest},
            )
            destination[seed] = source
            snapshot[str(source_file.resolve())] = digest
    return (
        SavedPredictionInputs(
            phase4=phase4,
            b1=b1,
            supervision=supervision,
            source_snapshot=snapshot,
        ),
        source_ids,
    )


def test_output_bundle_is_deterministic_complete_and_never_overwrites_sources(
    tmp_path, monkeypatch
):
    inputs, source_ids = _inputs(tmp_path)
    monkeypatch.setattr(
        phase4_evaluation_runner, "load_saved_prediction_inputs",
        lambda *_args: inputs,
    )
    monkeypatch.setattr(
        phase4_evaluation_runner,
        "validate_evaluator_repository",
        lambda _root: {
            "branch": "phase4-consendxformer",
            "commit": "b" * 40,
            "clean": True,
            "source_final_model_commit_is_ancestor": True,
        },
    )
    output_root = tmp_path / "evaluation"
    result = write_saved_prediction_evaluation(
        inputs,
        evaluation_config=EXPECTED_EVALUATION_CONFIG,
        output_root=output_root,
    )
    assert result.completion_marker.is_file()
    assert result.machine_results.is_file()
    assert result.summary.is_file()
    assert (result.output_dir / "tables" / "calibration_per_seed.csv").is_file()
    assert (result.output_dir / "curves" / "risk_coverage.csv").is_file()
    completion = json.loads(result.completion_marker.read_text(encoding="utf-8"))
    manifest = json.loads(result.artifact_manifest.read_text(encoding="utf-8"))
    assert completion["training_performed"] is False
    assert completion["artifact_manifest_sha256"] == sha256_file(
        result.artifact_manifest
    )
    assert manifest["bundle_sha256"] == sha256_json(manifest["files"])
    for path, digest in inputs.source_snapshot.items():
        assert sha256_file(path) == digest
    with pytest.raises(FileExistsError, match="overwrite"):
        write_saved_prediction_evaluation(
            inputs,
            evaluation_config=EXPECTED_EVALUATION_CONFIG,
            output_root=output_root,
        )
