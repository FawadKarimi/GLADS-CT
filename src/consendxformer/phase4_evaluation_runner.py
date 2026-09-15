"""Guarded Phase 4 post-hoc evaluation orchestration primitives.

This module is intentionally evaluation-only.  It consumes immutable completed
experiment artifacts, never creates a model optimizer, and never modifies a
source experiment directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .baseline_evaluation import CLASS_ORDER
from .baseline_source import PINNED_BASELINE_SHA
from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .phase3_config import load_phase3_config, validate_phase3_config
from .phase4_bootstrap import (
    BOOTSTRAP_METRICS,
    bootstrap_model_metrics,
    bootstrap_paired_differences,
)
from .phase4_ablations import phase4_ablation_protocol_record
from .phase4_boundary import (
    BOUNDARY_LIMITATIONS,
    BOUNDARY_RANGE_RECORDS,
    BOUNDARY_TARGET_INTERPRETATION,
)
from .phase4_config import load_phase4_config
from .phase4_input_lock import load_phase4_input_lock
from .phase4_evaluation_metrics import (
    ATTENTION_ALIGNMENT_DEFINITION,
    CALIBRATION_DEFINITION,
    RISK_COVERAGE_DEFINITION,
    UNCERTAINTY_DEFINITION,
    calibration_metrics,
    high_low_disagreement_comparison,
    predictive_entropy,
    risk_coverage_curve,
    spearman_association,
    uncertainty_disagreement_association,
)
from .phase3_identity import build_experiment_identity
from .phase3_predictions import load_phase3_prediction_csv, validate_prediction_table
from .phase4_predictions import (
    load_phase4_prediction_csv,
    validate_phase4_prediction_table,
)
from .phase4_protocol import tuning_protocol_record, tuning_protocol_sha256
from .phase4_readiness import phase4_supervision_hashes
from .manifest_guard import ManifestIntegrityError, validate_frozen_manifest_bundle


SOURCE_FINAL_MODEL_COMMIT = "e1b174490245c616e355ef7c0c6f7c3c239d3acd"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "phase4_evaluation.json"
DEFAULT_EVALUATION_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "phase4" / "evaluation"
FINAL_SEEDS: tuple[int, int, int] = (42, 43, 44)
SELECTED_LAMBDA_ORDINAL = 0.5
SELECTED_LAMBDA_ATTENTION = 0.05
_MANIFEST_SHA256 = "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
_SUPERVISION_SHA256 = "cbe4e402825665428ee890887963437d2bc480f8616bd7ec7999c667759ccc25"
_ORDERED_SUPERVISION_SHA256 = (
    "2bae364b180f75cf2332714cc486a517540fcff3b9e94c48d83f59d0d9d6789a"
)
_EVALUATION_ID = re.compile(r"phase4-evaluation-[0-9a-f]{8,64}\Z")
_MATCH_COLUMNS = (
    "patient_id",
    "nodule_id",
    "series_instance_uid",
    "split",
    "true_class",
)
_PHASE4_FINAL_REQUIRED_MEMBERS = (
    "config/phase4_config.json",
    "config/b1_anchor_config.json",
    "config/effective_config.json",
    "config/run_identity.json",
    "config/input_lock.json",
    "config/input_supervision.csv",
    "provenance/readiness_before.json",
    "provenance/readiness_after.json",
    "provenance/phase0_manifest.json",
    "provenance/environment.json",
    "provenance/tuning_protocol.json",
    "provenance/run.json",
    "history/epoch_history.json",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "checkpoints/best.metadata.json",
    "checkpoints/last.metadata.json",
    "predictions/val.csv",
    "predictions/test.csv",
    "metrics/val.json",
    "metrics/test.json",
)
_PHASE3_B1_REQUIRED_MEMBERS = (
    "config/experiment_identity.json",
    "config/input_metadata.csv",
    "config/phase1_effective.yaml",
    "config/phase3_effective.yaml",
    "provenance/environment.json",
    "provenance/readiness.json",
    "provenance/run.json",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "checkpoints/best.metadata.json",
    "checkpoints/last.metadata.json",
    "history/epoch_history.json",
    "predictions/val.csv",
    "predictions/test.csv",
    "metrics/val.json",
    "metrics/test.json",
)


class Phase4EvaluationError(RuntimeError):
    """Raised when an evaluation input or output violates the frozen protocol."""


@dataclass(frozen=True)
class SavedPredictionSource:
    """One fully verified, immutable experiment input for saved-result analysis."""

    model: str
    source_dir: Path
    descriptor: Mapping[str, Any]
    predictions: pd.DataFrame
    supervision: pd.DataFrame | None
    verification: Mapping[str, Any]
    source_snapshot: Mapping[str, str]


@dataclass(frozen=True)
class SavedPredictionInputs:
    """The exact three-seed B1/ConsenDxFormer source set."""

    phase4: Mapping[int, SavedPredictionSource]
    b1: Mapping[int, SavedPredictionSource]
    supervision: pd.DataFrame
    source_snapshot: Mapping[str, str]


@dataclass(frozen=True)
class Phase4EvaluationResult:
    """A completed content-addressed evaluation artifact bundle."""

    evaluation_id: str
    output_dir: Path
    machine_results: Path
    provenance: Path
    summary: Path
    artifact_manifest: Path
    completion_marker: Path


def _expected_evaluation_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_final_model_commit": SOURCE_FINAL_MODEL_COMMIT,
        "final_seeds": list(FINAL_SEEDS),
        "selected_lambdas": {
            "lambda_ordinal": SELECTED_LAMBDA_ORDINAL,
            "lambda_attention": SELECTED_LAMBDA_ATTENTION,
        },
        "class_order": list(CLASS_ORDER),
        "cohort": {
            "cohort_view": "paper_aligned_multi_reader_2_to_4",
            "sample_count": 1133,
            "test_sample_count": 183,
            "manifest_bundle_sha256": _MANIFEST_SHA256,
            "supervision_sha256": _SUPERVISION_SHA256,
            "ordered_supervision_sha256": _ORDERED_SUPERVISION_SHA256,
        },
        "calibration": {
            "nll_probability_floor": 1e-15,
            "brier_definition": "mean_sample_sum_class_squared_error",
            "ece": {
                "definition": "top_label_absolute_accuracy_confidence_gap",
                "binning": (
                    "15_equal_width_bins_over_[0,1]_left_closed_right_open_except_last"
                ),
                "bin_count": 15,
                "bin_edges": np.linspace(0.0, 1.0, 16).tolist(),
            },
        },
        "uncertainty": {
            "predictive_entropy": "natural_log_nats",
            "disagreement_threshold_rule": "median_of_frozen_training_split",
            "disagreement_tie_policy": (
                "high_if_greater_than_or_equal_to_threshold"
            ),
            "pooled_rule": (
                "mean_predictive_entropy_per_nodule_across_seeds_before_spearman"
            ),
            "risk_coverage": {
                "risk": "zero_one_classification_error",
                "ordering": "ascending_predictive_entropy",
                "tie_break": ["patient_id", "series_instance_uid", "nodule_id"],
                "aurc": (
                    "right_endpoint_mean_of_risk_at_coverages_k_over_n_for_k_1_to_n"
                ),
            },
        },
        "attention_alignment": {
            "interpretation": (
                "attention_alignment_spatial_correspondence_not_causal_explanation"
            ),
            "kl_direction": "soft_consensus_target_to_dsa_attention",
            "normalization": "nonnegative_unit_spatial_mass",
            "soft_mask_threshold": 0.5,
            "attention_threshold_rule": "per_map_quantile",
            "attention_quantile": 0.9,
            "quantile_method": "linear",
        },
        "boundary_stress": {
            "authoritative_rule": "phase2_paper_mean_label_boundary_zone",
            "lower_band": {
                "minimum_inclusive": 2.5,
                "maximum_exclusive": 3.0,
            },
            "upper_band": {
                "minimum_exclusive": 3.0,
                "maximum_inclusive": 3.5,
            },
            "primary_metrics_excluded": True,
            "target_interpretation": (
                "retain_boundary_zone_and_continuous_reader_targets_without_forcing_"
                "a_primary_true_class"
            ),
        },
        "bootstrap": {
            "unit": "patient",
            "seed": 20260914,
            "resamples": 10000,
            "confidence_level": 0.95,
            "interval": "percentile",
            "aggregate_standard_deviation_ddof": 1,
        },
        "safety": {
            "test_driven_tuning": False,
            "temperature_scaling": False,
            "threshold_tuning": False,
            "model_retraining": False,
            "ablation_training_in_this_milestone": False,
            "overwrite_source_experiments": False,
            "all_toggles_off_is_b1": False,
        },
    }


EXPECTED_EVALUATION_CONFIG = _expected_evaluation_config()


def _values_are_exactly_equal(actual: Any, expected: Any) -> bool:
    """Compare provenance recursively without Python's bool/int equivalence."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _values_are_exactly_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_are_exactly_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _approved_evaluation_config(value: Any) -> bool:
    return _values_are_exactly_equal(value, EXPECTED_EVALUATION_CONFIG)


def load_phase4_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load the versioned protocol only when every prespecified field is exact."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Phase 4 evaluation configuration is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4EvaluationError(
            f"Cannot read Phase 4 evaluation configuration: {source}"
        ) from exc
    if not _approved_evaluation_config(value):
        raise Phase4EvaluationError(
            "Phase 4 evaluation configuration differs from the approved evaluation protocol"
        )
    return deepcopy(EXPECTED_EVALUATION_CONFIG)


def _require_equal(record: Mapping[str, Any], field: str, expected: Any) -> None:
    if not _values_are_exactly_equal(record.get(field), expected):
        raise Phase4EvaluationError(
            f"Final source {field} differs: actual={record.get(field)!r}, "
            f"expected={expected!r}"
        )


def validate_final_source_descriptors(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the three immutable ConsenDxFormer final-run descriptors."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    canonical: list[dict[str, Any]] = []
    for source in records:
        if not isinstance(source, Mapping):
            raise TypeError("each final source descriptor must be a mapping")
        record = dict(source)
        for field, expected in (
            ("model", "ConsenDxFormer"),
            ("run_mode", "final"),
            ("class_order", list(CLASS_ORDER)),
            ("cohort_view", "paper_aligned_multi_reader_2_to_4"),
            ("sample_count", 1133),
            ("test_sample_count", 183),
            ("manifest_bundle_sha256", _MANIFEST_SHA256),
            ("supervision_sha256", _SUPERVISION_SHA256),
            ("ordered_supervision_sha256", _ORDERED_SUPERVISION_SHA256),
            ("project_commit", SOURCE_FINAL_MODEL_COMMIT),
            ("lambda_ordinal", SELECTED_LAMBDA_ORDINAL),
            ("lambda_attention", SELECTED_LAMBDA_ATTENTION),
            ("test_used_for_selection", False),
            ("completed", True),
        ):
            _require_equal(record, field, expected)
        experiment_id = record.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise Phase4EvaluationError("Final source experiment_id must be non-empty")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise Phase4EvaluationError("Final source seed must be an integer")
        canonical.append(record)
    seeds = [record["seed"] for record in canonical]
    if sorted(seeds) != list(FINAL_SEEDS) or len(set(seeds)) != len(FINAL_SEEDS):
        raise Phase4EvaluationError(
            f"Final source seeds must be exactly {list(FINAL_SEEDS)!r}; got {seeds!r}"
        )
    experiment_ids = [str(record["experiment_id"]) for record in canonical]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise Phase4EvaluationError("Final source experiment IDs must be unique")
    return sorted(canonical, key=lambda record: int(record["seed"]))


def _canonical_match_rows(frame: pd.DataFrame, *, seed: int, role: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise Phase4EvaluationError(f"{role} seed {seed} predictions are empty")
    missing = sorted(set(_MATCH_COLUMNS) - set(frame.columns))
    if missing:
        raise Phase4EvaluationError(
            f"{role} seed {seed} predictions lack matching fields: {missing}"
        )
    if not bool((frame["split"] == "test").all()):
        raise Phase4EvaluationError(f"{role} seed {seed} predictions must be test-only")
    if "seed" in frame:
        numeric_seed = pd.to_numeric(frame["seed"], errors="coerce")
        if numeric_seed.isna().any() or set(numeric_seed.astype(int)) != {seed}:
            raise Phase4EvaluationError(f"{role} prediction seed provenance is invalid")
    rows = frame.loc[:, list(_MATCH_COLUMNS)].copy()
    if rows.duplicated(subset=["patient_id", "nodule_id", "series_instance_uid"]).any():
        raise Phase4EvaluationError(f"{role} seed {seed} contains duplicate samples")
    return rows.sort_values(list(_MATCH_COLUMNS), kind="mergesort").reset_index(drop=True)


def validate_seeded_test_cohorts(
    phase4: Mapping[int, pd.DataFrame],
    b1: Mapping[int, pd.DataFrame],
) -> dict[str, Any]:
    """Require identical patient/nodule/series/test/label rows for all six tables."""

    if set(phase4) != set(FINAL_SEEDS) or set(b1) != set(FINAL_SEEDS):
        raise Phase4EvaluationError(
            f"Prediction mappings must contain exactly seeds {list(FINAL_SEEDS)!r}"
        )
    reference: pd.DataFrame | None = None
    for seed in FINAL_SEEDS:
        current_phase4 = _canonical_match_rows(
            phase4[seed], seed=seed, role="ConsenDxFormer"
        )
        current_b1 = _canonical_match_rows(b1[seed], seed=seed, role="B1")
        if not current_phase4.equals(current_b1):
            raise Phase4EvaluationError(
                f"B1/ConsenDxFormer test cohort mismatch for seed {seed}"
            )
        if reference is not None and not current_phase4.equals(reference):
            raise Phase4EvaluationError(
                f"ConsenDxFormer test cohort mismatch across seeds at seed {seed}"
            )
        reference = current_phase4
    assert reference is not None
    return {
        "schema_version": 1,
        "matched": True,
        "seeds": list(FINAL_SEEDS),
        "sample_count_per_seed": len(reference),
        "identity_fields": list(_MATCH_COLUMNS),
    }


def aggregate_seed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_fields: Iterable[str],
) -> dict[str, Any]:
    """Aggregate the exact final seeds with arithmetic mean and sample SD."""

    by_seed = {record.get("seed"): record for record in records}
    if set(by_seed) != set(FINAL_SEEDS) or len(records) != len(FINAL_SEEDS):
        raise Phase4EvaluationError(
            f"Aggregation requires exactly seeds {list(FINAL_SEEDS)!r}"
        )
    result: dict[str, Any] = {
        "seed_count": len(FINAL_SEEDS),
        "seeds": list(FINAL_SEEDS),
        "standard_deviation_ddof": 1,
    }
    for field in metric_fields:
        try:
            values = np.asarray(
                [float(by_seed[seed][field]) for seed in FINAL_SEEDS],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase4EvaluationError(
                f"Cannot aggregate nonnumeric metric {field!r}"
            ) from exc
        if not bool(np.isfinite(values).all()):
            raise Phase4EvaluationError(f"Metric {field!r} contains non-finite values")
        result[field] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
        }
    return result


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase4EvaluationError(f"{role} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4EvaluationError(f"Cannot read {role}: {path}") from exc
    if not isinstance(value, dict):
        raise Phase4EvaluationError(f"{role} must be a JSON object: {path}")
    return value


def _safe_source_member(source: Path, relative: Any) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative.strip():
        raise Phase4EvaluationError("Source artifact manifest contains an invalid path")
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise Phase4EvaluationError(
            f"Source artifact manifest contains an unsafe path: {relative!r}"
        )
    target = (source / Path(*parts)).resolve()
    if not _within(target, source):
        raise Phase4EvaluationError(
            f"Source artifact path escapes its experiment: {relative!r}"
        )
    return "/".join(parts), target


def verify_completed_source_artifact(
    source_dir: str | Path,
    *,
    required_members: Iterable[str],
    expected_experiment_id: str | None = None,
) -> dict[str, Any]:
    """Verify a source completion marker, manifest, and every checksummed member."""

    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise Phase4EvaluationError(f"Source experiment directory is missing: {source}")
    completion = _read_json(source / "COMPLETED.json", "source completion marker")
    manifest_name = completion.get("artifact_manifest")
    if manifest_name != "artifact_manifest.json":
        raise Phase4EvaluationError("Source completion marker names an invalid manifest")
    manifest_path = source / manifest_name
    manifest = _read_json(manifest_path, "source artifact manifest")
    if completion.get("artifact_manifest_sha256") != sha256_file(manifest_path):
        raise Phase4EvaluationError("Source artifact manifest checksum is inconsistent")
    if manifest.get("schema_version") != 1 or completion.get("schema_version") != 1:
        raise Phase4EvaluationError("Source artifact schema is unsupported")
    experiment_id = manifest.get("experiment_id")
    if (
        not isinstance(experiment_id, str)
        or not experiment_id.strip()
        or completion.get("experiment_id") != experiment_id
        or (
            expected_experiment_id is not None
            and experiment_id != expected_experiment_id
        )
    ):
        raise Phase4EvaluationError("Source artifact experiment identity is inconsistent")
    if manifest.get("research_results_eligible") is not True or completion.get(
        "research_results_eligible"
    ) is not True:
        raise Phase4EvaluationError("Source experiment is not research-results eligible")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise Phase4EvaluationError("Source artifact manifest contains no files")
    canonical_files = {str(key): str(value) for key, value in files.items()}
    if manifest.get("bundle_sha256") != sha256_json(canonical_files):
        raise Phase4EvaluationError("Source artifact bundle checksum is inconsistent")
    verified: dict[str, str] = {}
    for supplied_relative, expected_digest in canonical_files.items():
        relative, target = _safe_source_member(source, supplied_relative)
        if relative != supplied_relative or relative in verified:
            raise Phase4EvaluationError(
                f"Source artifact manifest path is not canonical and unique: {supplied_relative!r}"
            )
        if not target.is_file():
            raise Phase4EvaluationError(f"Source artifact is missing: {relative}")
        actual_digest = sha256_file(target)
        if actual_digest != expected_digest:
            raise Phase4EvaluationError(
                f"Source artifact checksum differs for {relative}"
            )
        verified[relative] = actual_digest
    required = {str(value).replace("\\", "/") for value in required_members}
    missing = sorted(required - set(verified))
    if missing:
        raise Phase4EvaluationError(
            f"Source artifact manifest lacks required files: {missing}"
        )
    return {
        "schema_version": 1,
        "verified": True,
        "experiment_id": experiment_id,
        "source_dir": str(source),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_bundle_sha256": manifest["bundle_sha256"],
        "files": verified,
        "manifest": manifest,
        "completion": completion,
    }


def snapshot_source_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash source files before evaluation so later mutation is detectable."""

    snapshot: dict[str, str] = {}
    for supplied in paths:
        path = Path(supplied).resolve()
        if not path.is_file():
            raise Phase4EvaluationError(f"Cannot snapshot missing source file: {path}")
        key = str(path)
        if key in snapshot:
            raise Phase4EvaluationError(f"Duplicate source file in snapshot: {path}")
        snapshot[key] = sha256_file(path)
    if not snapshot:
        raise Phase4EvaluationError("A source snapshot must contain at least one file")
    return snapshot


def assert_source_snapshot_unchanged(snapshot: Mapping[str, str]) -> None:
    """Fail if any snapshotted experiment artifact changed or disappeared."""

    if not isinstance(snapshot, Mapping) or not snapshot:
        raise Phase4EvaluationError("Source snapshot is empty or invalid")
    for supplied, expected in snapshot.items():
        path = Path(supplied)
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase4EvaluationError(
                f"Source artifact changed during evaluation: {path}"
            )


def validate_phase4_prediction_source(
    predictions: pd.DataFrame,
    supervision: pd.DataFrame,
    *,
    descriptor: Mapping[str, Any],
) -> pd.DataFrame:
    """Bind one final Phase 4 test CSV to its run and supervision snapshot."""

    try:
        canonical = validate_phase4_prediction_table(predictions)
    except (TypeError, ValueError) as exc:
        raise Phase4EvaluationError("Phase 4 prediction source is invalid") from exc
    expectations = {
        "seed": descriptor.get("seed"),
        "lambda_ordinal": descriptor.get("lambda_ordinal"),
        "lambda_attention": descriptor.get("lambda_attention"),
        "experiment_id": descriptor.get("experiment_id"),
        "run_mode": "final",
        "split": "test",
    }
    for field, expected in expectations.items():
        if canonical[field].nunique(dropna=False) != 1 or canonical[field].iloc[0] != expected:
            raise Phase4EvaluationError(
                f"Phase 4 prediction {field} differs from final-run provenance"
            )
    expected_count = descriptor.get("test_sample_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise Phase4EvaluationError("Final source test_sample_count is invalid")
    if len(canonical) != expected_count:
        raise Phase4EvaluationError(
            f"Phase 4 test prediction count differs: {len(canonical)} != {expected_count}"
        )
    if not isinstance(supervision, pd.DataFrame):
        raise TypeError("supervision must be a pandas DataFrame")
    required = {"patient_id", "nodule_id", "series_instance_uid", "split", "label"}
    missing = sorted(required - set(supervision.columns))
    if missing:
        raise Phase4EvaluationError(
            f"Phase 4 supervision snapshot lacks fields: {missing}"
        )
    expected_rows = supervision.loc[
        supervision["split"] == "test",
        ["patient_id", "nodule_id", "series_instance_uid", "split", "label"],
    ].rename(columns={"label": "true_class"}).reset_index(drop=True)
    actual_rows = canonical.loc[:, list(_MATCH_COLUMNS)].reset_index(drop=True)
    if not actual_rows.equals(expected_rows.loc[:, list(_MATCH_COLUMNS)]):
        raise Phase4EvaluationError(
            "Phase 4 predictions do not match the final supervision test rows"
        )
    return canonical


def validate_b1_prediction_source(
    predictions: pd.DataFrame,
    *,
    seed: int,
    expected_sample_count: int = 183,
) -> pd.DataFrame:
    """Validate one completed B1 test prediction artifact for paired analysis."""

    try:
        canonical = validate_prediction_table(predictions)
    except (TypeError, ValueError) as exc:
        raise Phase4EvaluationError("B1 prediction source is invalid") from exc
    checks = {
        "split": "test",
        "seed": seed,
        "baseline_variant": "B1",
    }
    for field, expected in checks.items():
        if canonical[field].nunique(dropna=False) != 1 or canonical[field].iloc[0] != expected:
            raise Phase4EvaluationError(f"B1 prediction {field} is inconsistent")
    if len(canonical) != expected_sample_count:
        raise Phase4EvaluationError(
            f"B1 test prediction count differs: {len(canonical)} != {expected_sample_count}"
        )
    return canonical


def _source_snapshot_from_verification(
    verification: Mapping[str, Any],
    *,
    include_status: bool = True,
) -> dict[str, str]:
    source = Path(str(verification["source_dir"])).resolve()
    files = verification.get("files")
    if not isinstance(files, Mapping):
        raise Phase4EvaluationError("Verified source file ledger is invalid")
    snapshot = {
        str((source / Path(*str(relative).split("/"))).resolve()): str(digest)
        for relative, digest in files.items()
    }
    for name in ("artifact_manifest.json", "COMPLETED.json"):
        path = (source / name).resolve()
        snapshot[str(path)] = sha256_file(path)
    if include_status:
        status = (source / "status.json").resolve()
        if not status.is_file():
            raise Phase4EvaluationError(f"Completed source status is missing: {status}")
        snapshot[str(status)] = sha256_file(status)
    return snapshot


def _read_csv_strings(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        raise Phase4EvaluationError(f"{role} is missing: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise Phase4EvaluationError(f"Cannot read {role}: {path}") from exc


def _require_mapping(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase4EvaluationError(f"{role} must be a mapping")
    return dict(value)


def _require_source_value(actual: Any, expected: Any, role: str) -> None:
    if not _values_are_exactly_equal(actual, expected):
        raise Phase4EvaluationError(
            f"{role} differs: actual={actual!r}, expected={expected!r}"
        )


def _validate_phase4_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(value)
    payload_fields = (
        "schema_version",
        "run_mode",
        "seed",
        "lambda_ordinal",
        "lambda_attention",
        "project_commit",
        "configuration_sha256",
        "manifest_bundle_sha256",
        "phase3_input_fingerprint",
        "supervision_sha256",
        "ordered_supervision_sha256",
        "input_lock_sha256",
        "cohort_view",
        "expected_resolution",
        "sample_count",
        "split_counts",
    )
    expected_fields = {"experiment_id", "digest", *payload_fields}
    if set(identity) != expected_fields:
        raise Phase4EvaluationError(
            "Phase 4 final identity has missing or unexpected fields"
        )
    payload = {field: identity[field] for field in payload_fields}
    digest = sha256_json(payload)
    seed = identity["seed"]
    expected_id = f"phase4-final-s{seed}-{digest[:16]}"
    _require_source_value(identity["digest"], digest, "Phase 4 identity digest")
    _require_source_value(
        identity["experiment_id"], expected_id, "Phase 4 content-addressed identity"
    )
    return identity


def _validate_persisted_selection(
    source: Path, run: Mapping[str, Any], *, model: str,
) -> None:
    """Reconcile selection, both sidecars and test timing to saved history."""

    from .phase3_checkpoints import select_best_validation_epoch
    from .phase4_checkpoints import select_phase4_best_epoch

    try:
        history = json.loads(
            (source / "history" / "epoch_history.json").read_text(encoding="utf-8")
        )
        selector = (
            select_phase4_best_epoch if model == "ConsenDxFormer"
            else select_best_validation_epoch
        )
        selected = selector(history)
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise Phase4EvaluationError(f"{model} persisted selection history is invalid") from exc
    selection = {
        "criterion": "validation_loss", "mode": "min",
        "split": "val" if model == "ConsenDxFormer" else "validation",
    }
    if model == "ConsenDxFormer":
        selection.update(metric="total_loss", test_influenced_selection=False)
        for key, expected in (
            ("best_epoch", selected["epoch"]),
            ("last_epoch", history[-1]["epoch"]),
            ("best_validation_loss", selected["value"]),
        ):
            _require_source_value(run.get(key), expected, f"{model} run {key}")
        events = [
            "best_checkpoint_selected_from_validation_total_loss",
            "persisted_checkpoints_reloaded_and_validated",
            "best_checkpoint_reloaded_for_final_validation_and_test",
            "test_evaluation_started_after_final_selection_and_best_reload",
        ]
    else:
        selection.update(
            best_epoch=selected["epoch"], best_validation_loss=selected["value"],
            test_metrics_influenced_selection=False,
        )
        events = [
            "best_checkpoint_selected_from_validation_loss",
            "best_checkpoint_reloaded",
            "test_evaluation_started_after_checkpoint_selection",
        ]
    _require_source_value(run.get("selection"), selection, f"{model} selection policy")
    actual_events = run.get("events")
    if not isinstance(actual_events, list) or any(
        actual_events.count(event) != 1 for event in events
    ) or [actual_events.index(event) for event in events] != sorted(
        actual_events.index(event) for event in events
    ):
        raise Phase4EvaluationError(f"{model} selection/reload/test events are inconsistent")
    for kind in ("best", "last"):
        sidecar = _read_json(
            source / "checkpoints" / f"{kind}.metadata.json", f"{model} {kind} metadata"
        )
        expected = {
            "epoch": selected["epoch"] if kind == "best" else history[-1]["epoch"],
            "best_epoch": selected["epoch"], "best_validation_loss": selected["value"],
            "selection_split": selection["split"],
        }
        if model == "ConsenDxFormer":
            expected.update(
                selection_criterion="validation_loss", selection_metric="total_loss",
                selection_mode="min",
            )
        else:
            expected.update(criterion="validation_loss", mode="min")
        for field, value in expected.items():
            _require_source_value(sidecar.get(field), value, f"{model} {kind} {field}")


def _validate_checkpoint_sidecar(
    sidecar: Mapping[str, Any],
    *,
    source: Path,
    identity: Mapping[str, Any],
    kind: str,
    model: str,
) -> None:
    record = dict(sidecar)
    _require_source_value(record.get("checkpoint_kind"), kind, f"{model} checkpoint kind")
    _require_source_value(
        record.get("experiment_id"), identity["experiment_id"],
        f"{model} checkpoint experiment identity",
    )
    _require_source_value(record.get("seed"), identity["seed"], f"{model} checkpoint seed")
    checkpoint = source / "checkpoints" / f"{kind}.pt"
    _require_source_value(
        record.get("checkpoint_file"), f"{kind}.pt", f"{model} checkpoint file"
    )
    checksum_field = (
        "checkpoint_sha256" if model == "ConsenDxFormer" else "checkpoint_file_sha256"
    )
    _require_source_value(
        record.get(checksum_field), sha256_file(checkpoint),
        f"{model} checkpoint checksum",
    )
    if model == "ConsenDxFormer":
        for field in (
            "run_mode",
            "lambda_ordinal",
            "lambda_attention",
            "project_commit",
            "config_sha256",
            "manifest_bundle_sha256",
            "supervision_sha256",
            "ordered_supervision_sha256",
            "input_lock_sha256",
            "cohort_view",
            "expected_resolution",
            "sample_count",
            "split_counts",
        ):
            identity_field = (
                "configuration_sha256" if field == "config_sha256" else field
            )
            _require_source_value(
                record.get(field), identity[identity_field],
                f"Phase 4 checkpoint {field}",
            )
        _require_source_value(record.get("run_mode"), "final", "Phase 4 checkpoint mode")
    else:
        _require_source_value(
            record.get("config_sha256"), identity["combined_config_sha256"],
            "B1 checkpoint configuration checksum",
        )
        for field in (
            "baseline_variant",
            "baseline_source_commit",
            "project_commit",
            "metadata_sha256",
            "cohort_sha256",
            "ordered_sample_sha256",
            "frozen_split_sha256",
        ):
            identity_field = (
                "frozen_split_bundle_sha256"
                if field == "frozen_split_sha256"
                else field
            )
            _require_source_value(
                record.get(field), identity[identity_field],
                f"B1 checkpoint {field}",
            )


def load_phase4_final_prediction_source(
    source_dir: str | Path,
) -> SavedPredictionSource:
    """Load one completed Phase 4 final seed without trusting filenames alone."""

    source = Path(source_dir).resolve()
    verification = verify_completed_source_artifact(
        source, required_members=_PHASE4_FINAL_REQUIRED_MEMBERS
    )
    snapshot = _source_snapshot_from_verification(verification)
    completion = _require_mapping(verification["completion"], "Phase 4 completion")
    manifest = _require_mapping(verification["manifest"], "Phase 4 manifest")
    for record, role in ((completion, "completion"), (manifest, "manifest")):
        _require_source_value(record.get("run_mode"), "final", f"Phase 4 {role} mode")
        _require_source_value(
            record.get("test_results_eligible"), True,
            f"Phase 4 {role} test eligibility",
        )

    identity = _validate_phase4_identity(
        _read_json(source / "config" / "run_identity.json", "Phase 4 run identity")
    )
    experiment_id = str(identity["experiment_id"])
    _require_source_value(
        verification["experiment_id"], experiment_id,
        "Phase 4 manifest/run experiment identity",
    )
    run = _read_json(source / "provenance" / "run.json", "Phase 4 run provenance")
    readiness_before = _read_json(
        source / "provenance" / "readiness_before.json", "Phase 4 readiness before"
    )
    readiness_after = _read_json(
        source / "provenance" / "readiness_after.json", "Phase 4 readiness after"
    )
    input_lock = _read_json(source / "config" / "input_lock.json", "Phase 4 input lock")
    effective = _read_json(
        source / "config" / "effective_config.json", "Phase 4 effective config"
    )
    phase4_config = _read_json(
        source / "config" / "phase4_config.json", "Phase 4 base config"
    )
    status = _read_json(source / "status.json", "Phase 4 final status")
    phase0_copy = _read_json(
        source / "provenance" / "phase0_manifest.json",
        "Phase 4 frozen Phase 0 manifest copy",
    )
    try:
        current_phase0 = validate_frozen_manifest_bundle(
            PROJECT_ROOT / "artifacts" / "phase0" / "manifests"
        ).to_dict()
    except (OSError, ValueError, ManifestIntegrityError) as exc:
        raise Phase4EvaluationError(
            "Current frozen Phase 0 manifest bundle is invalid"
        ) from exc
    _require_source_value(
        {key: value for key, value in phase0_copy.items() if key != "manifest_dir"},
        {key: value for key, value in current_phase0.items() if key != "manifest_dir"},
        "Phase 4 source frozen Phase 0 manifest report",
    )

    _require_source_value(run.get("identity"), identity, "Phase 4 run identity copy")
    for field, expected in (
        ("execution_state", "completed"),
        ("run_mode", "final"),
        ("real_training_started", True),
        ("research_results_eligible", True),
        ("test_results_eligible", True),
        ("readiness_revalidated_after_training", True),
        ("manifest_revalidated_after_training", True),
        ("project_git_state_unchanged_after_training", True),
    ):
        _require_source_value(run.get(field), expected, f"Phase 4 run {field}")
    selection = _require_mapping(run.get("selection"), "Phase 4 selection provenance")
    _require_source_value(
        selection.get("test_influenced_selection"), False,
        "Phase 4 test influence on selection",
    )
    _require_source_value(status.get("state"), "completed", "Phase 4 source status")
    _require_source_value(status.get("experiment_id"), experiment_id, "Phase 4 status identity")
    project_before = _require_mapping(run.get("project_git_before"), "Phase 4 project Git before")
    _require_source_value(
        project_before.get("commit"), SOURCE_FINAL_MODEL_COMMIT,
        "Phase 4 source project commit",
    )
    _require_source_value(
        run.get("project_git_after"), project_before,
        "Phase 4 project Git state after training",
    )
    _require_source_value(
        run.get("checkpoint_files"),
        {"best": "checkpoints/best.pt", "last": "checkpoints/last.pt"},
        "Phase 4 checkpoint paths",
    )
    prediction_files = _require_mapping(
        run.get("prediction_files"), "Phase 4 prediction paths"
    )
    _require_source_value(
        prediction_files.get("test"),
        "predictions/test.csv",
        "Phase 4 test prediction path",
    )

    _require_source_value(readiness_after, readiness_before, "Phase 4 readiness revalidation")
    _require_source_value(
        identity["phase3_input_fingerprint"],
        readiness_before.get("phase3_input_fingerprint"),
        "Phase 4 Phase 3 input fingerprint",
    )
    for field, expected in (
        ("ready_for_requested_mode", True),
        ("real_research_ready", True),
        ("phase3_readiness_validated", True),
        ("multirater_supervision_validated", True),
    ):
        _require_source_value(readiness_before.get(field), expected, f"Phase 4 readiness {field}")
    for field, expected in (
        ("cohort_view", "paper_aligned_multi_reader_2_to_4"),
        ("expected_resolution", 64),
        ("sample_count", 1133),
        ("split_counts", {"train": 807, "val": 143, "test": 183}),
        ("manifest_bundle_sha256", _MANIFEST_SHA256),
        ("supervision_sha256", _SUPERVISION_SHA256),
        ("ordered_supervision_sha256", _ORDERED_SUPERVISION_SHA256),
    ):
        _require_source_value(readiness_before.get(field), expected, f"Phase 4 readiness {field}")
        _require_source_value(identity.get(field), expected, f"Phase 4 identity {field}")
    try:
        validated_input_lock = load_phase4_input_lock(
            source / "config" / "input_lock.json"
        ).to_dict()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise Phase4EvaluationError("Phase 4 input lock is invalid") from exc
    _require_source_value(input_lock, validated_input_lock, "Phase 4 input lock schema")
    _require_source_value(sha256_json(input_lock), identity["input_lock_sha256"], "Phase 4 input lock checksum")
    for field in (
        "cohort_view",
        "expected_resolution",
        "sample_count",
        "split_counts",
        "manifest_bundle_sha256",
        "supervision_sha256",
        "ordered_supervision_sha256",
    ):
        _require_source_value(input_lock.get(field), identity[field], f"Phase 4 input lock {field}")

    execution = _require_mapping(effective.get("execution"), "Phase 4 execution config")
    _require_source_value(execution.get("run_mode"), "final", "Phase 4 effective run mode")
    if "seed" in execution:
        raise Phase4EvaluationError("Phase 4 execution config contains an unexpected seed field")
    for field, expected in (
        ("single_run_only", True),
        ("lambda_ordinal", SELECTED_LAMBDA_ORDINAL),
        ("lambda_attention", SELECTED_LAMBDA_ATTENTION),
        ("evaluate_test", True),
    ):
        _require_source_value(execution.get(field), expected, f"Phase 4 execution {field}")
    _require_source_value(
        _require_mapping(effective.get("experiment"), "Phase 4 experiment config").get("seed"),
        identity["seed"], "Phase 4 effective seed",
    )
    losses = _require_mapping(phase4_config.get("losses"), "Phase 4 loss config")
    _require_source_value(losses.get("selected_lambda_ordinal"), SELECTED_LAMBDA_ORDINAL, "selected lambda_ordinal")
    _require_source_value(losses.get("selected_lambda_attention"), SELECTED_LAMBDA_ATTENTION, "selected lambda_attention")
    _require_source_value(losses.get("test_tuning_forbidden"), True, "Phase 4 test-tuning guard")
    _require_source_value(identity["run_mode"], "final", "Phase 4 identity mode")
    _require_source_value(identity["seed"] in FINAL_SEEDS, True, "Phase 4 final seed membership")
    _require_source_value(identity["lambda_ordinal"], SELECTED_LAMBDA_ORDINAL, "Phase 4 identity lambda_ordinal")
    _require_source_value(identity["lambda_attention"], SELECTED_LAMBDA_ATTENTION, "Phase 4 identity lambda_attention")
    _require_source_value(identity["project_commit"], SOURCE_FINAL_MODEL_COMMIT, "Phase 4 identity project commit")
    _require_source_value(identity["schema_version"], 1, "Phase 4 identity schema")
    b1_anchor = _read_json(
        source / "config" / "b1_anchor_config.json", "Phase 4 B1 anchor config"
    )
    protocol = _read_json(
        source / "provenance" / "tuning_protocol.json", "Phase 4 tuning protocol"
    )
    _require_source_value(protocol, tuning_protocol_record(), "Phase 4 tuning protocol")
    canonical_bundle = load_phase4_config(
        PROJECT_ROOT / "configs" / "phase4_consensdxformer.yaml"
    )
    _require_source_value(
        phase4_config,
        canonical_bundle.phase4_config,
        "Phase 4 source base configuration",
    )
    _require_source_value(
        b1_anchor,
        canonical_bundle.b1_anchor_config,
        "Phase 4 source B1 anchor configuration",
    )
    expected_effective = deepcopy(canonical_bundle.phase4_config)
    expected_effective["experiment"]["seed"] = identity["seed"]
    expected_effective["execution"] = {
        "single_run_only": True,
        "run_mode": "final",
        "lambda_ordinal": SELECTED_LAMBDA_ORDINAL,
        "lambda_attention": SELECTED_LAMBDA_ATTENTION,
        "evaluate_test": True,
        "input_lock_sha256": identity["input_lock_sha256"],
        "tuning_protocol_sha256": tuning_protocol_sha256(),
    }
    _require_source_value(
        effective,
        expected_effective,
        "Phase 4 exact final effective configuration",
    )
    base_combined_sha256 = sha256_json(
        {"schema_version": 1, "phase4": phase4_config, "b1_anchor": b1_anchor}
    )
    expected_configuration_sha256 = sha256_json(
        {
            "schema_version": 1,
            "base_combined_config_sha256": base_combined_sha256,
            "effective_config": effective,
            "input_lock_sha256": identity["input_lock_sha256"],
            "tuning_protocol_sha256": tuning_protocol_sha256(),
        }
    )
    _require_source_value(
        identity["configuration_sha256"], expected_configuration_sha256,
        "Phase 4 effective configuration checksum",
    )
    _require_source_value(
        execution.get("input_lock_sha256"), identity["input_lock_sha256"],
        "Phase 4 execution input-lock checksum",
    )
    _require_source_value(
        execution.get("tuning_protocol_sha256"), tuning_protocol_sha256(),
        "Phase 4 execution tuning-protocol checksum",
    )

    _validate_persisted_selection(source, run, model="ConsenDxFormer")
    for kind in ("best", "last"):
        _validate_checkpoint_sidecar(
            _read_json(
                source / "checkpoints" / f"{kind}.metadata.json",
                f"Phase 4 {kind} checkpoint metadata",
            ),
            source=source,
            identity=identity,
            kind=kind,
            model="ConsenDxFormer",
        )

    supervision = _read_csv_strings(
        source / "config" / "input_supervision.csv", "Phase 4 supervision snapshot"
    )
    if "split" not in supervision:
        raise Phase4EvaluationError("Phase 4 supervision snapshot lacks split")
    if len(supervision) != 1133 or supervision["split"].value_counts().to_dict() != {
        "train": 807,
        "test": 183,
        "val": 143,
    }:
        raise Phase4EvaluationError("Phase 4 supervision snapshot counts differ")
    try:
        supervision_sha256, ordered_supervision_sha256 = (
            phase4_supervision_hashes(supervision)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Phase4EvaluationError(
            "Phase 4 supervision snapshot cannot be fingerprinted"
        ) from exc
    _require_source_value(
        supervision_sha256,
        identity["supervision_sha256"],
        "Phase 4 supervision logical fingerprint",
    )
    _require_source_value(
        ordered_supervision_sha256,
        identity["ordered_supervision_sha256"],
        "Phase 4 supervision ordered fingerprint",
    )
    descriptor = {
        "model": "ConsenDxFormer",
        "experiment_id": experiment_id,
        "seed": int(identity["seed"]),
        "run_mode": "final",
        "class_order": list(CLASS_ORDER),
        "cohort_view": identity["cohort_view"],
        "sample_count": identity["sample_count"],
        "test_sample_count": int(identity["split_counts"]["test"]),
        "manifest_bundle_sha256": identity["manifest_bundle_sha256"],
        "phase3_input_fingerprint": deepcopy(identity["phase3_input_fingerprint"]),
        "supervision_sha256": identity["supervision_sha256"],
        "ordered_supervision_sha256": identity["ordered_supervision_sha256"],
        "input_lock_sha256": identity["input_lock_sha256"],
        "project_commit": identity["project_commit"],
        "configuration_sha256": identity["configuration_sha256"],
        "lambda_ordinal": identity["lambda_ordinal"],
        "lambda_attention": identity["lambda_attention"],
        "test_used_for_selection": False,
        "completed": True,
        "artifact_manifest_sha256": verification["artifact_manifest_sha256"],
        "artifact_bundle_sha256": verification["artifact_bundle_sha256"],
    }
    predictions = validate_phase4_prediction_source(
        load_phase4_prediction_csv(source / "predictions" / "test.csv"),
        supervision,
        descriptor=descriptor,
    )
    assert_source_snapshot_unchanged(snapshot)
    return SavedPredictionSource(
        model="ConsenDxFormer",
        source_dir=source,
        descriptor=descriptor,
        predictions=predictions,
        supervision=supervision,
        verification=verification,
        source_snapshot=snapshot,
    )


def load_b1_prediction_source(source_dir: str | Path) -> SavedPredictionSource:
    """Load one completed real Phase 3 B1 experiment for paired evaluation."""

    source = Path(source_dir).resolve()
    verification = verify_completed_source_artifact(
        source, required_members=_PHASE3_B1_REQUIRED_MEMBERS
    )
    snapshot = _source_snapshot_from_verification(verification)
    completion = _require_mapping(verification["completion"], "B1 completion")
    _require_source_value(completion.get("synthetic_dry_run"), False, "B1 synthetic flag")
    identity = _read_json(
        source / "config" / "experiment_identity.json", "B1 experiment identity"
    )
    identity_fields = {
        field: identity.get(field)
        for field in (
            "baseline_variant",
            "seed",
            "baseline_source_commit",
            "project_commit",
            "combined_config_sha256",
            "metadata_sha256",
            "cohort_sha256",
            "ordered_sample_sha256",
            "frozen_split_bundle_sha256",
        )
    }
    try:
        rebuilt = build_experiment_identity(**identity_fields).to_dict()
    except (TypeError, ValueError) as exc:
        raise Phase4EvaluationError("B1 experiment identity is invalid") from exc
    _require_source_value(identity, rebuilt, "B1 content-addressed experiment identity")
    experiment_id = str(identity["experiment_id"])
    _require_source_value(verification["experiment_id"], experiment_id, "B1 manifest identity")
    run = _read_json(source / "provenance" / "run.json", "B1 run provenance")
    readiness = _read_json(source / "provenance" / "readiness.json", "B1 readiness")
    status = _read_json(source / "status.json", "B1 final status")
    _require_source_value(run.get("identity"), identity, "B1 run identity copy")
    for field, expected in (
        ("mode", "real"),
        ("synthetic_dry_run", False),
        ("synthetic_non_research", False),
        ("input_real_research_ready", True),
        ("execution_state", "completed"),
        ("research_results_eligible", True),
        ("real_training_started", True),
        ("trainer_called", True),
        ("readiness_revalidated_after_training", True),
        ("project_commit_unchanged_after_training", True),
        ("baseline_unchanged_after_execution", True),
    ):
        _require_source_value(run.get(field), expected, f"B1 run {field}")
    _require_source_value(run.get("readiness"), readiness, "B1 readiness copy")
    selection = _require_mapping(run.get("selection"), "B1 selection provenance")
    _require_source_value(
        selection.get("test_metrics_influenced_selection"), False,
        "B1 test influence on selection",
    )
    _require_source_value(status.get("state"), "completed", "B1 source status")
    _require_source_value(status.get("experiment_id"), experiment_id, "B1 status identity")
    project_git = _require_mapping(run.get("project_git"), "B1 project Git")
    _require_source_value(
        project_git.get("commit"), identity["project_commit"], "B1 project Git commit"
    )
    _require_source_value(
        run.get("checkpoint_files"),
        {"best": "checkpoints/best.pt", "last": "checkpoints/last.pt"},
        "B1 checkpoint paths",
    )
    prediction_files = _require_mapping(run.get("prediction_files"), "B1 prediction paths")
    _require_source_value(
        prediction_files.get("test"), "predictions/test.csv", "B1 test prediction path"
    )
    for field, expected in (
        ("mode", "real"),
        ("ready_for_requested_mode", True),
        ("real_research_ready", True),
        ("synthetic_non_research", False),
        ("cohort_view", "paper_aligned_multi_reader_2_to_4"),
        ("expected_resolution", 64),
        ("sample_count", 1133),
        ("split_counts", {"train": 807, "val": 143, "test": 183}),
        ("manifest_bundle_sha256", _MANIFEST_SHA256),
    ):
        _require_source_value(readiness.get(field), expected, f"B1 readiness {field}")
    fingerprint = _require_mapping(readiness.get("fingerprint"), "B1 input fingerprint")
    for identity_field, fingerprint_field in (
        ("metadata_sha256", "metadata_sha256"),
        ("cohort_sha256", "logical_cohort_sha256"),
        ("ordered_sample_sha256", "ordered_samples_sha256"),
    ):
        _require_source_value(
            identity[identity_field], fingerprint.get(fingerprint_field),
            f"B1 identity/readiness {identity_field}",
        )
    _require_source_value(
        identity["frozen_split_bundle_sha256"], readiness["manifest_bundle_sha256"],
        "B1 frozen manifest fingerprint",
    )
    _require_source_value(identity["baseline_variant"], "B1", "B1 baseline variant")
    _require_source_value(
        identity["baseline_source_commit"], PINNED_BASELINE_SHA, "B1 baseline source commit"
    )
    _require_source_value(identity["seed"] in FINAL_SEEDS, True, "B1 final seed membership")
    try:
        phase3_effective = yaml.safe_load(
            (source / "config" / "phase3_effective.yaml").read_text(encoding="utf-8")
        )
        phase1_effective = yaml.safe_load(
            (source / "config" / "phase1_effective.yaml").read_text(encoding="utf-8")
        )
        validate_phase3_config(
            phase3_effective,
            effective_phase1_config=phase1_effective,
            expected_variant="B1",
        )
        canonical_b1 = load_phase3_config(
            PROJECT_ROOT / "configs" / "phase3_b1.yaml",
            expected_variant="B1",
            seed=int(identity["seed"]),
            device=str(phase3_effective["device"]),
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise Phase4EvaluationError(
            "B1 effective configuration snapshots are invalid"
        ) from exc
    effective_experiment = _require_mapping(
        phase3_effective.get("experiment"), "B1 experiment config"
    )
    _require_source_value(
        effective_experiment.get("seed"), identity["seed"], "B1 effective seed"
    )
    _require_source_value(
        phase3_effective,
        canonical_b1.phase3_config,
        "B1 exact canonical Phase 3 effective configuration",
    )
    _require_source_value(
        phase1_effective,
        canonical_b1.effective_phase1_config,
        "B1 exact canonical Phase 1 effective configuration",
    )
    _require_source_value(
        identity["combined_config_sha256"],
        canonical_b1.combined_config_sha256,
        "B1 effective configuration checksum",
    )
    _validate_persisted_selection(source, run, model="B1")
    for kind in ("best", "last"):
        _validate_checkpoint_sidecar(
            _read_json(
                source / "checkpoints" / f"{kind}.metadata.json",
                f"B1 {kind} checkpoint metadata",
            ),
            source=source,
            identity=identity,
            kind=kind,
            model="B1",
        )
    predictions = validate_b1_prediction_source(
        load_phase3_prediction_csv(source / "predictions" / "test.csv"),
        seed=int(identity["seed"]),
    )
    _require_source_value(
        _source_experiment_id(predictions), experiment_id,
        "B1 prediction experiment identity",
    )
    metadata = _read_csv_strings(
        source / "config" / "input_metadata.csv", "B1 input metadata snapshot"
    )
    _require_source_value(
        sha256_file(source / "config" / "input_metadata.csv"),
        identity["metadata_sha256"],
        "B1 input metadata byte checksum",
    )
    required_metadata = {
        "patient_id",
        "nodule_id",
        "series_instance_uid",
        "split",
        "label",
    }
    if not required_metadata.issubset(metadata.columns):
        raise Phase4EvaluationError("B1 input metadata lacks required cohort fields")
    expected = metadata.loc[
        metadata["split"] == "test",
        ["patient_id", "nodule_id", "series_instance_uid", "split", "label"],
    ].rename(columns={"label": "true_class"})
    expected = expected.sort_values(list(_MATCH_COLUMNS), kind="mergesort").reset_index(drop=True)
    actual = predictions.loc[:, list(_MATCH_COLUMNS)].sort_values(
        list(_MATCH_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    # pandas may use StringDtype for validated class names but object strings
    # for CSV metadata. Compare exact values, not their storage representation.
    if actual.to_dict("records") != expected.loc[:, list(_MATCH_COLUMNS)].to_dict("records"):
        raise Phase4EvaluationError("B1 predictions do not match the input metadata test rows")
    descriptor = {
        "model": "B1",
        "experiment_id": experiment_id,
        "seed": int(identity["seed"]),
        "run_mode": "final",
        "class_order": list(CLASS_ORDER),
        "cohort_view": readiness["cohort_view"],
        "sample_count": readiness["sample_count"],
        "test_sample_count": int(readiness["split_counts"]["test"]),
        "manifest_bundle_sha256": readiness["manifest_bundle_sha256"],
        "phase3_input_fingerprint": deepcopy(fingerprint),
        "project_commit": identity["project_commit"],
        "baseline_source_commit": identity["baseline_source_commit"],
        "test_used_for_selection": False,
        "completed": True,
        "artifact_manifest_sha256": verification["artifact_manifest_sha256"],
        "artifact_bundle_sha256": verification["artifact_bundle_sha256"],
    }
    assert_source_snapshot_unchanged(snapshot)
    return SavedPredictionSource(
        model="B1",
        source_dir=source,
        descriptor=descriptor,
        predictions=predictions,
        supervision=None,
        verification=verification,
        source_snapshot=snapshot,
    )


def load_saved_prediction_inputs(
    phase4_source_dirs: Sequence[str | Path],
    b1_source_dirs: Sequence[str | Path],
) -> SavedPredictionInputs:
    """Load and cross-bind exactly six completed source experiments."""

    if len(phase4_source_dirs) != 3 or len(b1_source_dirs) != 3:
        raise Phase4EvaluationError("Evaluation requires exactly three Phase 4 and three B1 sources")
    all_dirs = [Path(value).resolve() for value in (*phase4_source_dirs, *b1_source_dirs)]
    if len(set(all_dirs)) != 6:
        raise Phase4EvaluationError("All six source experiment directories must be distinct")
    phase4_sources = [load_phase4_final_prediction_source(path) for path in phase4_source_dirs]
    b1_sources = [load_b1_prediction_source(path) for path in b1_source_dirs]
    validate_final_source_descriptors([source.descriptor for source in phase4_sources])
    phase4_by_seed = {int(source.descriptor["seed"]): source for source in phase4_sources}
    b1_by_seed = {int(source.descriptor["seed"]): source for source in b1_sources}
    if set(phase4_by_seed) != set(FINAL_SEEDS) or len(phase4_by_seed) != 3:
        raise Phase4EvaluationError("Phase 4 sources must contain each final seed exactly once")
    if set(b1_by_seed) != set(FINAL_SEEDS) or len(b1_by_seed) != 3:
        raise Phase4EvaluationError("B1 sources must contain each final seed exactly once")
    reference_supervision = phase4_by_seed[42].supervision
    assert reference_supervision is not None
    reference_fingerprint = dict(
        phase4_by_seed[42].descriptor["phase3_input_fingerprint"]
    )
    for seed in FINAL_SEEDS:
        supervision = phase4_by_seed[seed].supervision
        if supervision is None or not supervision.equals(reference_supervision):
            raise Phase4EvaluationError("Phase 4 supervision snapshots differ across final seeds")
        if dict(phase4_by_seed[seed].descriptor["phase3_input_fingerprint"]) != reference_fingerprint:
            raise Phase4EvaluationError("Phase 4 primary cohort fingerprint differs across seeds")
        if dict(b1_by_seed[seed].descriptor["phase3_input_fingerprint"]) != reference_fingerprint:
            raise Phase4EvaluationError(
                f"B1/ConsenDxFormer frozen cohort fingerprint mismatch for seed {seed}"
            )
    validate_seeded_test_cohorts(
        {seed: phase4_by_seed[seed].predictions for seed in FINAL_SEEDS},
        {seed: b1_by_seed[seed].predictions for seed in FINAL_SEEDS},
    )
    snapshot: dict[str, str] = {}
    for source in (*phase4_sources, *b1_sources):
        overlap = set(snapshot) & set(source.source_snapshot)
        if overlap:
            raise Phase4EvaluationError(f"Source snapshots overlap: {sorted(overlap)!r}")
        snapshot.update(source.source_snapshot)
    return SavedPredictionInputs(
        phase4=phase4_by_seed,
        b1=b1_by_seed,
        supervision=reference_supervision.copy(deep=True),
        source_snapshot=snapshot,
    )


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    columns = [f"prob_{class_name}" for class_name in CLASS_ORDER]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise Phase4EvaluationError(f"Prediction probabilities are missing: {missing}")
    try:
        values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Phase4EvaluationError("Prediction probabilities must be numeric") from exc
    return values


def _test_disagreement(
    frame: pd.DataFrame,
    supervision: pd.DataFrame,
) -> np.ndarray:
    keys = ["patient_id", "nodule_id", "series_instance_uid", "split"]
    required = {*keys, "label", "disagreement_entropy"}
    missing = sorted(required - set(supervision.columns))
    if missing:
        raise Phase4EvaluationError(
            f"Frozen supervision lacks uncertainty fields: {missing}"
        )
    source = supervision.loc[
        supervision["split"] == "test", [*keys, "label", "disagreement_entropy"]
    ].copy()
    if source.duplicated(subset=keys).any():
        raise Phase4EvaluationError("Frozen supervision contains duplicate test identities")
    aligned = frame.loc[:, [*keys, "true_class"]].merge(
        source,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(frame) or aligned["disagreement_entropy"].isna().any():
        raise Phase4EvaluationError(
            "Prediction rows do not resolve exactly to frozen disagreement metadata"
        )
    if not aligned["true_class"].equals(aligned["label"]):
        raise Phase4EvaluationError(
            "Prediction true classes disagree with frozen supervision"
        )
    try:
        values = pd.to_numeric(
            aligned["disagreement_entropy"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Phase4EvaluationError(
            "Frozen disagreement entropy must be numeric"
        ) from exc
    if not bool(np.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise Phase4EvaluationError("Frozen disagreement entropy must lie in [0, 1]")
    return values


def _training_disagreement_threshold(supervision: pd.DataFrame) -> float:
    if not isinstance(supervision, pd.DataFrame):
        raise TypeError("supervision must be a pandas DataFrame")
    if "split" not in supervision or "disagreement_entropy" not in supervision:
        raise Phase4EvaluationError(
            "Frozen supervision lacks split/disagreement threshold fields"
        )
    train = supervision.loc[supervision["split"] == "train", "disagreement_entropy"]
    if train.empty:
        raise Phase4EvaluationError("Frozen supervision has no training disagreement values")
    try:
        values = pd.to_numeric(train, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Phase4EvaluationError(
            "Training disagreement entropy must be numeric"
        ) from exc
    if not bool(np.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise Phase4EvaluationError("Training disagreement entropy must lie in [0, 1]")
    return float(np.median(values))


def _source_experiment_id(frame: pd.DataFrame) -> str:
    if "experiment_id" not in frame or frame["experiment_id"].nunique() != 1:
        raise Phase4EvaluationError("Prediction experiment identity is not constant")
    value = str(frame["experiment_id"].iloc[0]).strip()
    if not value:
        raise Phase4EvaluationError("Prediction experiment identity is blank")
    return value


def compute_saved_prediction_evaluation(
    phase4_predictions: Mapping[int, pd.DataFrame],
    b1_predictions: Mapping[int, pd.DataFrame],
    supervision: pd.DataFrame,
    *,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute all prediction-only analyses under the frozen evaluation protocol.

    No fitting occurs.  The high/low disagreement threshold is calculated once
    from frozen training supervision, while all reported model values use the
    already-frozen final test predictions.
    """

    if not _approved_evaluation_config(dict(evaluation_config)):
        raise Phase4EvaluationError(
            "Runtime evaluation configuration differs from the approved protocol"
        )
    matching = validate_seeded_test_cohorts(phase4_predictions, b1_predictions)
    threshold = _training_disagreement_threshold(supervision)
    threshold_description = (
        "median disagreement_entropy of the frozen primary training split; "
        "computed without predictions or test outcomes"
    )
    calibration_by_model: dict[str, list[dict[str, Any]]] = {
        "B1": [],
        "ConsenDxFormer": [],
    }
    source_ids: dict[str, dict[str, str]] = {"B1": {}, "ConsenDxFormer": {}}
    for model_name, frames in (
        ("B1", b1_predictions),
        ("ConsenDxFormer", phase4_predictions),
    ):
        for seed in FINAL_SEEDS:
            frame = frames[seed]
            calculated = calibration_metrics(
                frame["true_class"].tolist(), _probabilities(frame)
            )
            calibration_by_model[model_name].append(
                {
                    "model": model_name,
                    "seed": seed,
                    "experiment_id": _source_experiment_id(frame),
                    "sample_count": len(frame),
                    "negative_log_likelihood": calculated[
                        "negative_log_likelihood"
                    ],
                    "brier_score": calculated["brier_score"],
                    "expected_calibration_error": calculated[
                        "expected_calibration_error"
                    ],
                }
            )
            source_ids[model_name][str(seed)] = _source_experiment_id(frame)
    metric_fields = (
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error",
    )
    calibration_aggregate = {
        model: aggregate_seed_records(records, metric_fields=metric_fields)
        for model, records in calibration_by_model.items()
    }

    uncertainty_rows: list[dict[str, Any]] = []
    uncertainty_per_seed: list[dict[str, Any]] = []
    curves: dict[str, dict[str, Any]] = {}
    pooled_by_identity: dict[tuple[str, str, str], list[float]] = {}
    pooled_disagreement: dict[tuple[str, str, str], float] = {}
    for seed in FINAL_SEEDS:
        frame = phase4_predictions[seed]
        probabilities = _probabilities(frame)
        disagreement = _test_disagreement(frame, supervision)
        entropy = predictive_entropy(probabilities)
        identifiers = [
            "|".join(
                str(frame.iloc[position][field])
                for field in ("patient_id", "series_instance_uid", "nodule_id")
            )
            for position in range(len(frame))
        ]
        association = uncertainty_disagreement_association(
            probabilities, disagreement
        )
        comparison = high_low_disagreement_comparison(
            probabilities,
            disagreement,
            threshold=threshold,
            threshold_provenance=threshold_description,
        )
        entropy_curve = risk_coverage_curve(
            frame["true_class"].tolist(),
            probabilities,
            ranking="entropy",
            identifiers=identifiers,
        )
        confidence_curve = risk_coverage_curve(
            frame["true_class"].tolist(),
            probabilities,
            ranking="confidence",
            identifiers=identifiers,
        )
        curves[str(seed)] = {
            "entropy": entropy_curve,
            "confidence": confidence_curve,
        }
        uncertainty_per_seed.append(
            {
                "seed": seed,
                "experiment_id": _source_experiment_id(frame),
                "sample_count": len(frame),
                "spearman": association,
                "high_low_disagreement": comparison,
                "entropy_aurc": entropy_curve["aurc"],
                "confidence_aurc": confidence_curve["aurc"],
            }
        )
        groups = comparison["groups"]
        for position, row in frame.reset_index(drop=True).iterrows():
            identity = (
                str(row["patient_id"]),
                str(row["nodule_id"]),
                str(row["series_instance_uid"]),
            )
            pooled_by_identity.setdefault(identity, []).append(float(entropy[position]))
            if identity in pooled_disagreement and pooled_disagreement[identity] != float(
                disagreement[position]
            ):
                raise Phase4EvaluationError(
                    "Frozen disagreement entropy differs across final seeds"
                )
            pooled_disagreement[identity] = float(disagreement[position])
            uncertainty_rows.append(
                {
                    "patient_id": identity[0],
                    "nodule_id": identity[1],
                    "series_instance_uid": identity[2],
                    "split": "test",
                    "seed": seed,
                    "experiment_id": _source_experiment_id(frame),
                    "predictive_entropy_nats": float(entropy[position]),
                    "disagreement_entropy": float(disagreement[position]),
                    "disagreement_group": groups[position],
                    "disagreement_threshold": threshold,
                }
            )
    ordered_identities = sorted(pooled_by_identity)
    pooled_entropy_values = np.asarray(
        [np.mean(pooled_by_identity[key]) for key in ordered_identities],
        dtype=np.float64,
    )
    pooled_disagreement_values = np.asarray(
        [pooled_disagreement[key] for key in ordered_identities], dtype=np.float64
    )
    pooled_association = spearman_association(
        pooled_entropy_values, pooled_disagreement_values
    )
    uncertainty_aggregate = aggregate_seed_records(
        uncertainty_per_seed,
        metric_fields=("entropy_aurc", "confidence_aurc"),
    )

    bootstrap_config = EXPECTED_EVALUATION_CONFIG["bootstrap"]
    bootstrap_models: dict[str, dict[str, Any]] = {"B1": {}, "ConsenDxFormer": {}}
    bootstrap_paired: dict[str, Any] = {}
    for seed in FINAL_SEEDS:
        for model_name, frames in (
            ("B1", b1_predictions),
            ("ConsenDxFormer", phase4_predictions),
        ):
            bootstrap_models[model_name][str(seed)] = bootstrap_model_metrics(
                frames[seed],
                model_name=model_name,
                seed=int(bootstrap_config["seed"]),
                n_resamples=int(bootstrap_config["resamples"]),
                confidence_level=float(bootstrap_config["confidence_level"]),
            )
        bootstrap_paired[str(seed)] = bootstrap_paired_differences(
            b1_predictions[seed],
            phase4_predictions[seed],
            seed=int(bootstrap_config["seed"]),
            n_resamples=int(bootstrap_config["resamples"]),
            confidence_level=float(bootstrap_config["confidence_level"]),
        )

    return {
        "schema_version": 1,
        "analysis": "phase4_saved_final_prediction_evaluation",
        "source_experiment_ids": source_ids,
        "matching": matching,
        "calibration": {
            "per_seed": calibration_by_model,
            "aggregate": calibration_aggregate,
        },
        "uncertainty": {
            "disagreement_threshold": threshold,
            "threshold_provenance": {
                "rule": "median",
                "source_split": "train",
                "source_field": "disagreement_entropy",
                "test_performance_used": False,
                "description": threshold_description,
            },
            "per_nodule_per_seed": uncertainty_rows,
            "per_seed": uncertainty_per_seed,
            "aggregate": uncertainty_aggregate,
            "pooled_descriptive": {
                "rule": EXPECTED_EVALUATION_CONFIG["uncertainty"]["pooled_rule"],
                "sample_count": len(ordered_identities),
                "spearman": pooled_association,
            },
            "risk_coverage_curves": curves,
        },
        "bootstrap": {
            "parameters": deepcopy(bootstrap_config),
            "per_model_per_seed": bootstrap_models,
            "paired_per_seed": bootstrap_paired,
            "three_training_seed_parametric_claims": False,
        },
        "definitions": {
            "class_order": list(CLASS_ORDER),
            "calibration": CALIBRATION_DEFINITION,
            "uncertainty": UNCERTAINTY_DEFINITION,
            "risk_coverage": RISK_COVERAGE_DEFINITION,
            "attention_alignment": ATTENTION_ALIGNMENT_DEFINITION,
            "test_driven_tuning": False,
            "temperature_scaling_or_calibration_fitting": False,
            "test_threshold_tuning": False,
        },
    }


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class Phase4EvaluationLayout:
    output_root: Path
    output_dir: Path
    tables_dir: Path
    curves_dir: Path
    provenance_dir: Path


def create_evaluation_layout(
    output_root: str | Path,
    evaluation_id: str,
    *,
    protected_sources: Iterable[str | Path],
) -> Phase4EvaluationLayout:
    """Create a new evaluation directory without overwriting any source or result."""

    if _EVALUATION_ID.fullmatch(evaluation_id) is None:
        raise Phase4EvaluationError(f"Unsafe Phase 4 evaluation ID: {evaluation_id!r}")
    root = Path(output_root).resolve()
    sources = tuple(Path(source).resolve() for source in protected_sources)
    for source in sources:
        if root == source or _within(root, source):
            raise Phase4EvaluationError(
                f"Evaluation output cannot be inside source experiment {source}"
            )
    output = root / evaluation_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 4 evaluation: {output}")
    tables = output / "tables"
    curves = output / "curves"
    provenance = output / "provenance"
    for path in (tables, curves, provenance):
        path.mkdir(parents=True, exist_ok=False)
    return Phase4EvaluationLayout(
        output_root=root,
        output_dir=output,
        tables_dir=tables,
        curves_dir=curves,
        provenance_dir=provenance,
    )


def _git_output(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase4EvaluationError(
            f"Cannot establish evaluator Git state for {' '.join(arguments)}"
        ) from exc
    return completed.stdout.strip()


def validate_evaluator_repository(project_root: str | Path) -> dict[str, Any]:
    """Require a clean Phase 4 branch descended from the frozen final commit."""

    root = Path(project_root).resolve()
    if root != PROJECT_ROOT.resolve():
        raise Phase4EvaluationError(
            "Evaluator project_root must be the checkout that supplies the running code"
        )
    branch = _git_output(root, "branch", "--show-current")
    if branch != "phase4-consendxformer":
        raise Phase4EvaluationError(
            f"Evaluation must run on phase4-consendxformer, not {branch!r}"
        )
    commit = _git_output(root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise Phase4EvaluationError("Evaluator HEAD is not an exact Git commit")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                SOURCE_FINAL_MODEL_COMMIT,
                commit,
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase4EvaluationError(
            "Evaluator commit is not descended from the frozen final-model commit"
        ) from exc
    dirty = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise Phase4EvaluationError(
            "Evaluator repository must be clean before analysis artifact creation"
        )
    for relative in (
        "configs/phase4_evaluation.json",
        "configs/phase4_consensdxformer.yaml",
    ):
        working = (root / Path(*relative.split("/"))).read_bytes().replace(b"\r\n", b"\n")
        try:
            committed = subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"],
                check=True,
                capture_output=True,
            ).stdout.replace(b"\r\n", b"\n")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase4EvaluationError(
                f"Canonical evaluator input is not tracked at HEAD: {relative}"
            ) from exc
        if working != committed:
            raise Phase4EvaluationError(
                f"Canonical evaluator input differs from its HEAD blob: {relative}"
            )
    phase4 = load_phase4_config(root / "configs" / "phase4_consensdxformer.yaml")
    losses = phase4.phase4_config["losses"]
    _require_source_value(
        losses.get("selected_lambda_ordinal"), SELECTED_LAMBDA_ORDINAL,
        "repository selected lambda_ordinal",
    )
    _require_source_value(
        losses.get("selected_lambda_attention"), SELECTED_LAMBDA_ATTENTION,
        "repository selected lambda_attention",
    )
    return {
        "branch": branch,
        "commit": commit,
        "clean": True,
        "source_final_model_commit_is_ancestor": True,
    }


def relative_source_snapshot(source: SavedPredictionSource) -> dict[str, str]:
    """Reconstructable source ledger, independent of the artifact mount point."""

    root = source.source_dir.resolve()
    try:
        return {
            Path(path).resolve().relative_to(root).as_posix(): digest
            for path, digest in sorted(source.source_snapshot.items())
        }
    except ValueError as exc:
        raise Phase4EvaluationError("Source snapshot escapes its experiment") from exc


def _source_records(inputs: SavedPredictionInputs) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model, sources in (("B1", inputs.b1), ("ConsenDxFormer", inputs.phase4)):
        for seed in FINAL_SEEDS:
            source = sources[seed]
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "source_dir": str(source.source_dir),
                    "descriptor": deepcopy(dict(source.descriptor)),
                    "artifact_manifest_sha256": source.verification[
                        "artifact_manifest_sha256"
                    ],
                    "artifact_bundle_sha256": source.verification[
                        "artifact_bundle_sha256"
                    ],
                    "manifest_files": deepcopy(dict(source.verification["files"])),
                    "source_snapshot": relative_source_snapshot(source),
                }
            )
    return records


def derive_evaluation_id(
    *,
    evaluation_config: Mapping[str, Any],
    evaluator_commit: str,
    source_records: Sequence[Mapping[str, Any]],
) -> str:
    """Derive a path-stable ID from protocol, code, and immutable source bundles."""

    if not _approved_evaluation_config(dict(evaluation_config)):
        raise Phase4EvaluationError("Cannot identify a non-approved evaluation protocol")
    if re.fullmatch(r"[0-9a-f]{40}", evaluator_commit) is None:
        raise Phase4EvaluationError("evaluator_commit must be an exact 40-hex commit")
    sources = sorted(
        (
            {
                "model": record.get("model"),
                "seed": record.get("seed"),
                "experiment_id": _require_mapping(
                    record.get("descriptor"), "source descriptor"
                ).get("experiment_id"),
                "artifact_manifest_sha256": record.get(
                    "artifact_manifest_sha256"
                ),
                "artifact_bundle_sha256": record.get("artifact_bundle_sha256"),
            }
            for record in source_records
        ),
        key=lambda record: (str(record["model"]), int(record["seed"])),
    )
    if [(record["model"], record["seed"]) for record in sources] != [
        *(('B1', seed) for seed in FINAL_SEEDS),
        *(('ConsenDxFormer', seed) for seed in FINAL_SEEDS),
    ]:
        raise Phase4EvaluationError("Evaluation identity requires the exact six source seeds")
    digest = sha256_json(
        {
            "schema_version": 1,
            "evaluation_config_sha256": sha256_json(dict(evaluation_config)),
            "evaluator_commit": evaluator_commit,
            "sources": sources,
        }
    )
    return f"phase4-evaluation-{digest[:16]}"


def _calibration_tables(analysis: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = _require_mapping(analysis.get("calibration"), "calibration results")
    per_seed = _require_mapping(calibration.get("per_seed"), "per-seed calibration")
    rows: list[dict[str, Any]] = []
    for model in ("B1", "ConsenDxFormer"):
        model_rows = per_seed.get(model)
        if not isinstance(model_rows, Sequence):
            raise Phase4EvaluationError(f"Calibration rows are missing for {model}")
        rows.extend(dict(record) for record in model_rows)
    aggregate = _require_mapping(calibration.get("aggregate"), "aggregate calibration")
    aggregate_rows: list[dict[str, Any]] = []
    for model in ("B1", "ConsenDxFormer"):
        record = _require_mapping(aggregate.get(model), f"{model} aggregate calibration")
        for metric in (
            "negative_log_likelihood",
            "brier_score",
            "expected_calibration_error",
        ):
            summary = _require_mapping(record.get(metric), f"aggregate {metric}")
            aggregate_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": summary.get("mean"),
                    "standard_deviation": summary.get("standard_deviation"),
                    "standard_deviation_ddof": record.get("standard_deviation_ddof"),
                    "seed_count": record.get("seed_count"),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(aggregate_rows)


def _uncertainty_tables(analysis: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uncertainty = _require_mapping(analysis.get("uncertainty"), "uncertainty results")
    per_nodule = pd.DataFrame(uncertainty.get("per_nodule_per_seed", ()))
    seed_rows: list[dict[str, Any]] = []
    for raw in uncertainty.get("per_seed", ()):
        record = dict(raw)
        spearman = _require_mapping(record.pop("spearman"), "seed Spearman result")
        comparison = _require_mapping(
            record.pop("high_low_disagreement"), "high/low disagreement result"
        )
        record.update(
            {
                "spearman_coefficient": spearman.get("coefficient"),
                "spearman_status": spearman.get("status"),
                "disagreement_threshold": comparison.get("threshold"),
                "low_count": _require_mapping(comparison.get("low"), "low group").get("count"),
                "low_mean_predictive_entropy": _require_mapping(comparison.get("low"), "low group").get("mean_predictive_entropy"),
                "high_count": _require_mapping(comparison.get("high"), "high group").get("count"),
                "high_mean_predictive_entropy": _require_mapping(comparison.get("high"), "high group").get("mean_predictive_entropy"),
            }
        )
        seed_rows.append(record)
    curve_rows: list[dict[str, Any]] = []
    curves = _require_mapping(uncertainty.get("risk_coverage_curves"), "risk curves")
    for seed in FINAL_SEEDS:
        by_ranking = _require_mapping(curves.get(str(seed)), f"seed {seed} risk curves")
        for ranking in ("entropy", "confidence"):
            curve = _require_mapping(by_ranking.get(ranking), f"{ranking} risk curve")
            arrays = [
                curve.get("ordered_identifiers"),
                curve.get("selection_score"),
                curve.get("retained_count"),
                curve.get("coverage"),
                curve.get("risk"),
            ]
            if any(not isinstance(value, Sequence) for value in arrays) or len(
                {len(value) for value in arrays}
            ) != 1:
                raise Phase4EvaluationError("Risk-coverage arrays are inconsistent")
            for identity, score, retained, coverage, risk in zip(*arrays):
                curve_rows.append(
                    {
                        "seed": seed,
                        "ranking": ranking,
                        "ordered_identifier": identity,
                        "selection_score": score,
                        "retained_count": retained,
                        "coverage": coverage,
                        "risk": risk,
                        "aurc": curve.get("aurc"),
                    }
                )
    return per_nodule, pd.DataFrame(seed_rows), pd.DataFrame(curve_rows)


def _bootstrap_summary_table(analysis: Mapping[str, Any]) -> pd.DataFrame:
    bootstrap = _require_mapping(analysis.get("bootstrap"), "bootstrap results")
    rows: list[dict[str, Any]] = []
    per_model = _require_mapping(
        bootstrap.get("per_model_per_seed"), "per-model bootstrap results"
    )
    for model in ("B1", "ConsenDxFormer"):
        by_seed = _require_mapping(per_model.get(model), f"{model} bootstrap results")
        for seed in FINAL_SEEDS:
            result = _require_mapping(by_seed.get(str(seed)), f"{model} seed {seed} bootstrap")
            points = _require_mapping(result.get("point_estimates"), "bootstrap point estimates")
            intervals = _require_mapping(result.get("confidence_intervals"), "bootstrap intervals")
            for metric in BOOTSTRAP_METRICS:
                interval = _require_mapping(intervals.get(metric), f"{metric} interval")
                rows.append(
                    {
                        "analysis": "per_model",
                        "model_or_comparison": model,
                        "seed": seed,
                        "metric": metric,
                        "point_estimate": points.get(metric),
                        "ci_lower": interval.get("lower"),
                        "ci_upper": interval.get("upper"),
                        "confidence_level": interval.get("confidence_level"),
                        "ci_method": interval.get("method"),
                    }
                )
    paired = _require_mapping(bootstrap.get("paired_per_seed"), "paired bootstrap results")
    for seed in FINAL_SEEDS:
        result = _require_mapping(paired.get(str(seed)), f"paired seed {seed} bootstrap")
        points = _require_mapping(result.get("point_estimates"), "paired point estimates")
        differences = _require_mapping(points.get("difference"), "paired point differences")
        interval_sets = _require_mapping(result.get("confidence_intervals"), "paired intervals")
        intervals = _require_mapping(interval_sets.get("difference"), "paired difference intervals")
        for metric in BOOTSTRAP_METRICS:
            interval = _require_mapping(intervals.get(metric), f"paired {metric} interval")
            rows.append(
                {
                    "analysis": "paired_difference",
                    "model_or_comparison": "ConsenDxFormer_minus_B1",
                    "seed": seed,
                    "metric": metric,
                    "point_estimate": differences.get(metric),
                    "ci_lower": interval.get("lower"),
                    "ci_upper": interval.get("upper"),
                    "confidence_level": interval.get("confidence_level"),
                    "ci_method": interval.get("method"),
                }
            )
    return pd.DataFrame(rows)


def _summary_markdown(analysis: Mapping[str, Any], evaluation_id: str) -> str:
    _per_seed, aggregate = _calibration_tables(analysis)
    lines = [
        f"# Phase 4 saved-prediction evaluation: {evaluation_id}",
        "",
        "This bundle reports prespecified post-hoc analyses of immutable final test predictions. No model fitting, threshold tuning, temperature scaling, retraining, or ablation training occurred.",
        "",
        "## Calibration (three-seed mean ± sample SD)",
        "",
        "| Model | Metric | Mean | SD |",
        "|---|---|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['metric']} | {float(row['mean']):.8g} | {float(row['standard_deviation']):.8g} |"
        )
    uncertainty = _require_mapping(analysis.get("uncertainty"), "uncertainty results")
    lines.extend(
        [
            "",
            "## Prespecified uncertainty analysis",
            "",
            f"The high/low disagreement threshold was the frozen training-split median ({float(uncertainty['disagreement_threshold']):.8g}); test performance did not select it.",
            "",
            "Attention outputs, when generated by the separate frozen-checkpoint post-hoc path, are spatial-correspondence measurements and are not causal explanations.",
            "",
            "Boundary-stress results remain separate from the primary test headline and require an explicitly validated Phase 2 boundary cohort.",
            "",
            "Bootstrap intervals are descriptive patient-cluster percentile intervals. No parametric significance claim over three training seeds is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _relative_output(layout: Phase4EvaluationLayout, path: Path) -> str:
    resolved = path.resolve()
    if not _within(resolved, layout.output_dir):
        raise Phase4EvaluationError(f"Evaluation artifact escapes output: {resolved}")
    return resolved.relative_to(layout.output_dir).as_posix()


def write_saved_prediction_evaluation(
    inputs: SavedPredictionInputs,
    *,
    evaluation_config: Mapping[str, Any],
    output_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> Phase4EvaluationResult:
    """Compute verified inputs and write artifacts, with COMPLETED written last."""

    if not _approved_evaluation_config(dict(evaluation_config)):
        raise Phase4EvaluationError("Runtime evaluation protocol is not approved")
    evaluator_git = validate_evaluator_repository(project_root)
    # A caller's mutable DataFrames are not evidence: reread the checked files.
    inputs = load_saved_prediction_inputs(
        [inputs.phase4[seed].source_dir for seed in FINAL_SEEDS],
        [inputs.b1[seed].source_dir for seed in FINAL_SEEDS],
    )
    evaluator_commit = str(evaluator_git.get("commit", ""))
    records = _source_records(inputs)
    evaluation_id = derive_evaluation_id(
        evaluation_config=evaluation_config,
        evaluator_commit=evaluator_commit,
        source_records=records,
    )
    if (Path(output_root).resolve() / evaluation_id).exists():
        raise FileExistsError(
            "Refusing to overwrite Phase 4 evaluation: "
            f"{Path(output_root).resolve() / evaluation_id}"
        )
    analysis = compute_saved_prediction_evaluation(
        {seed: inputs.phase4[seed].predictions for seed in FINAL_SEEDS},
        {seed: inputs.b1[seed].predictions for seed in FINAL_SEEDS},
        inputs.supervision,
        evaluation_config=evaluation_config,
    )
    expected_ids = {
        model: {
            str(seed): str(sources[seed].descriptor["experiment_id"])
            for seed in FINAL_SEEDS
        }
        for model, sources in (("B1", inputs.b1), ("ConsenDxFormer", inputs.phase4))
    }
    _require_source_value(
        analysis.get("source_experiment_ids"), expected_ids,
        "analysis source experiment IDs",
    )
    assert_source_snapshot_unchanged(inputs.source_snapshot)
    layout = create_evaluation_layout(
        output_root,
        evaluation_id,
        protected_sources=(record["source_dir"] for record in records),
    )
    artifacts: list[Path] = []
    machine_results = atomic_write_json(layout.output_dir / "evaluation.json", dict(analysis))
    artifacts.append(machine_results)
    per_seed_calibration, aggregate_calibration = _calibration_tables(analysis)
    per_nodule, uncertainty_by_seed, risk_curves = _uncertainty_tables(analysis)
    bootstrap_summary = _bootstrap_summary_table(analysis)
    for name, frame in (
        ("calibration_per_seed.csv", per_seed_calibration),
        ("calibration_aggregate.csv", aggregate_calibration),
        ("uncertainty_per_nodule_per_seed.csv", per_nodule),
        ("uncertainty_per_seed.csv", uncertainty_by_seed),
        ("bootstrap_confidence_intervals.csv", bootstrap_summary),
    ):
        artifacts.append(atomic_write_csv(layout.tables_dir / name, frame))
    artifacts.append(atomic_write_csv(layout.curves_dir / "risk_coverage.csv", risk_curves))
    ablation_path = atomic_write_json(
        layout.provenance_dir / "ablation_protocol.json",
        phase4_ablation_protocol_record(),
    )
    attention_path = atomic_write_json(
        layout.provenance_dir / "attention_alignment_protocol.json",
        {
            "schema_version": 1,
            "status": "infrastructure_only_requires_persisted_final_best_checkpoint_inference",
            "results_fabricated": False,
            "definition": deepcopy(ATTENTION_ALIGNMENT_DEFINITION),
            "interpretation": "attention_alignment_spatial_correspondence_not_causal_explanation",
            "training_permitted": False,
            "checkpoint_overwrite_permitted": False,
        },
    )
    boundary_path = atomic_write_json(
        layout.provenance_dir / "boundary_stress_protocol.json",
        {
            "schema_version": 1,
            "status": "infrastructure_only_requires_explicit_validated_boundary_cohort",
            "results_fabricated": False,
            "boundary_ranges": [dict(value) for value in BOUNDARY_RANGE_RECORDS],
            "target_interpretation": BOUNDARY_TARGET_INTERPRETATION,
            "limitations": list(BOUNDARY_LIMITATIONS),
            "primary_headline_inclusion_permitted": False,
            "boundary_driven_tuning_permitted": False,
            "frozen_final_model_required": True,
        },
    )
    artifacts.extend((ablation_path, attention_path, boundary_path))
    provenance_record = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "evaluator_git": dict(evaluator_git),
        "source_final_model_commit": SOURCE_FINAL_MODEL_COMMIT,
        "source_experiments": records,
        "source_snapshot_sha256": sha256_json({
            f"{record['model']}/{record['seed']}": record["source_snapshot"]
            for record in records
        }),
        "evaluation_config": deepcopy(dict(evaluation_config)),
        "evaluation_config_sha256": sha256_json(dict(evaluation_config)),
        "selected_lambdas": {
            "lambda_ordinal": SELECTED_LAMBDA_ORDINAL,
            "lambda_attention": SELECTED_LAMBDA_ATTENTION,
        },
        "seeds": list(FINAL_SEEDS),
        "class_order": list(CLASS_ORDER),
        "source_experiments_mutation_permitted": False,
        "test_driven_tuning": False,
        "model_retraining": False,
        "ablation_training": False,
        "performance_values_origin": "computed_from_verified_saved_prediction_csvs",
    }
    provenance = atomic_write_json(
        layout.provenance_dir / "provenance.json", provenance_record
    )
    summary = atomic_write_text(
        layout.output_dir / "summary.md", _summary_markdown(analysis, evaluation_id)
    )
    artifacts.extend((provenance, summary))
    assert_source_snapshot_unchanged(inputs.source_snapshot)
    files = {
        _relative_output(layout, path): sha256_file(path)
        for path in sorted(artifacts, key=lambda item: _relative_output(layout, item))
    }
    manifest_record = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "files": files,
        "bundle_sha256": sha256_json(files),
        "source_artifacts_unchanged": True,
        "research_results_eligible": True,
        "test_results_eligible": True,
        "training_performed": False,
    }
    artifact_manifest = atomic_write_json(
        layout.output_dir / "artifact_manifest.json", manifest_record
    )
    for relative, digest in files.items():
        if sha256_file(layout.output_dir / Path(*relative.split("/"))) != digest:
            raise Phase4EvaluationError(f"Evaluation artifact changed before completion: {relative}")
    if sha256_json(files) != manifest_record["bundle_sha256"]:
        raise Phase4EvaluationError("Evaluation artifact bundle checksum is inconsistent")
    assert_source_snapshot_unchanged(inputs.source_snapshot)
    completion_marker = atomic_write_json(
        layout.output_dir / "COMPLETED.json",
        {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "artifact_manifest": artifact_manifest.name,
            "artifact_manifest_sha256": sha256_file(artifact_manifest),
            "source_artifacts_unchanged": True,
            "research_results_eligible": True,
            "test_results_eligible": True,
            "training_performed": False,
        },
    )
    try:
        assert_source_snapshot_unchanged(inputs.source_snapshot)
    except BaseException:
        completion_marker.unlink(missing_ok=True)
        raise
    return Phase4EvaluationResult(
        evaluation_id=evaluation_id,
        output_dir=layout.output_dir,
        machine_results=machine_results,
        provenance=provenance,
        summary=summary,
        artifact_manifest=artifact_manifest,
        completion_marker=completion_marker,
    )


def run_phase4_saved_evaluation(
    *,
    phase4_source_dirs: Sequence[str | Path],
    b1_source_dirs: Sequence[str | Path],
    output_root: str | Path,
    evaluation_config_path: str | Path,
    project_root: str | Path,
) -> Phase4EvaluationResult:
    """Execute only the prespecified saved-prediction analyses and write outputs."""

    evaluation_config = load_phase4_evaluation_config(evaluation_config_path)
    inputs = load_saved_prediction_inputs(phase4_source_dirs, b1_source_dirs)
    assert_source_snapshot_unchanged(inputs.source_snapshot)
    return write_saved_prediction_evaluation(
        inputs,
        evaluation_config=evaluation_config,
        output_root=output_root,
        project_root=project_root,
    )


def phase4_evaluation_cli(argv: Sequence[str] | None = None) -> int:
    """CLI for explicit, immutable saved-prediction source directories."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate three completed Phase 4 final runs and matched Phase 3 B1 "
            "runs without training or calibration fitting."
        )
    )
    parser.add_argument(
        "--phase4-source",
        action="append",
        required=True,
        help="Completed phase4-final experiment directory (pass exactly three).",
    )
    parser.add_argument(
        "--b1-source",
        action="append",
        required=True,
        help="Completed Phase 3 B1 experiment directory (pass exactly three).",
    )
    parser.add_argument("--config", default=str(DEFAULT_EVALUATION_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_EVALUATION_OUTPUT_ROOT))
    args = parser.parse_args(None if argv is None else list(argv))
    result = run_phase4_saved_evaluation(
        phase4_source_dirs=args.phase4_source,
        b1_source_dirs=args.b1_source,
        output_root=args.output_root,
        evaluation_config_path=args.config,
        project_root=PROJECT_ROOT,
    )
    print(f"PHASE4_EVALUATION_ID={result.evaluation_id}")
    print(f"OUTPUT_DIR={result.output_dir}")
    print(f"COMPLETION_MARKER={result.completion_marker}")
    return 0


__all__ = [
    "DEFAULT_EVALUATION_CONFIG",
    "DEFAULT_EVALUATION_OUTPUT_ROOT",
    "EXPECTED_EVALUATION_CONFIG",
    "FINAL_SEEDS",
    "SELECTED_LAMBDA_ATTENTION",
    "SELECTED_LAMBDA_ORDINAL",
    "SOURCE_FINAL_MODEL_COMMIT",
    "PROJECT_ROOT",
    "Phase4EvaluationError",
    "Phase4EvaluationLayout",
    "Phase4EvaluationResult",
    "SavedPredictionInputs",
    "SavedPredictionSource",
    "aggregate_seed_records",
    "assert_source_snapshot_unchanged",
    "create_evaluation_layout",
    "compute_saved_prediction_evaluation",
    "derive_evaluation_id",
    "load_b1_prediction_source",
    "load_phase4_final_prediction_source",
    "load_phase4_evaluation_config",
    "load_saved_prediction_inputs",
    "phase4_evaluation_cli",
    "run_phase4_saved_evaluation",
    "snapshot_source_files",
    "validate_b1_prediction_source",
    "validate_final_source_descriptors",
    "validate_phase4_prediction_source",
    "validate_seeded_test_cohorts",
    "verify_completed_source_artifact",
    "validate_evaluator_repository",
    "write_saved_prediction_evaluation",
]
