"""Guarded entry point for evaluating supplied B0/B1 prediction exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .baseline_config import load_baseline_config
from .baseline_evaluation import CLASS_ORDER, evaluate_prediction_csv, write_evaluation_json
from .baseline_provenance import project_git_state, utc_timestamp
from .baseline_source import (
    assert_source_snapshot_unchanged,
    snapshot_source_tree,
    verify_baseline_source,
)
from .hashing import sha256_json
from .manifest_guard import validate_frozen_manifest_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_safe_output(
    output_path: Path,
    prediction_path: Path,
    protected_roots: Sequence[Path],
) -> None:
    if output_path == prediction_path:
        raise ValueError("Evaluation output must not overwrite the prediction CSV")
    for root in protected_roots:
        if output_path == root or _is_within(output_path, root):
            raise ValueError(f"Evaluation output must not be inside protected path {root}")
    if output_path.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {output_path}. "
            "Use a fresh path; Phase 1 evaluation does not overwrite results."
        )


def run_baseline_evaluation(
    mode: str,
    *,
    config_path: str | Path,
    predictions_csv: str | Path,
    output_json: str | Path,
    baseline_dir: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    identifier_column: str | None = None,
    true_label_column: str | None = None,
    predicted_class_column: str | None = None,
    probability_columns: Sequence[str] | None = None,
    logit_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate locked inputs, compute metrics, and write guarded provenance.

    The prediction table is supplied by the caller. This function never runs a
    model or generates predictions, and it writes only the requested fresh JSON
    file outside the immutable source and manifest trees.
    """
    expected_mode = mode.upper()
    config_file = _project_path(config_path)
    config = load_baseline_config(config_file, expected_mode=expected_mode)
    source_root = (
        Path(baseline_dir).resolve()
        if baseline_dir is not None
        else _project_path(config["baseline"]["repository_dir"])
    )
    frozen_root = (
        Path(manifest_dir).resolve()
        if manifest_dir is not None
        else _project_path(config["frozen_manifest"]["directory"])
    )
    lock_file = _project_path(config["baseline"]["source_lock_file"])
    prediction_path = Path(predictions_csv).resolve()
    output_path = Path(output_json).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction CSV does not exist: {prediction_path}")
    _require_safe_output(output_path, prediction_path, (source_root, frozen_root))

    # Evaluation is a baseline entry point: both immutable inputs are verified
    # before prediction contents are read or any output directory is created.
    source_report = verify_baseline_source(source_root, lock_file)
    manifest_report = validate_frozen_manifest_bundle(frozen_root)
    source_before = snapshot_source_tree(source_root)

    metrics = evaluate_prediction_csv(
        prediction_path,
        identifier_column=identifier_column,
        true_label_column=true_label_column,
        predicted_class_column=predicted_class_column,
        probability_columns=probability_columns,
        logit_columns=logit_columns,
    )

    source_after = snapshot_source_tree(source_root)
    assert_source_snapshot_unchanged(source_before, source_after)
    verify_baseline_source(source_root, lock_file)
    metrics["baseline_context"] = {
        "schema_version": 1,
        "created_utc": utc_timestamp(),
        "experiment_type": expected_mode,
        "experiment_name": config["experiment"]["name"],
        "project_git": project_git_state(PROJECT_ROOT),
        "baseline": source_report.to_dict(),
        "configuration": {
            "path": str(config_file),
            "sha256": sha256_json(config),
        },
        "manifest": manifest_report.to_dict(),
        "manifest_checksum_reference": str(frozen_root / "manifest_checksums.json"),
        "seed": config["seed"],
        "device": config["device"],
        "input_predictions_path": str(prediction_path),
        "output_path": str(output_path),
        "prediction_source": "caller_supplied",
        "training_success_claimed": False,
    }
    write_evaluation_json(metrics, output_path)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute fixed-order benign/indeterminate/malignant metrics from a "
            "caller-supplied B0/B1 prediction CSV after verifying the locked source "
            "and frozen split. This command never fabricates prediction rows."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str.upper,
        choices=("B0", "B1"),
        help="Baseline identity associated with the supplied predictions.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Matching locked Phase 1 config (default: configs/b0.yaml or configs/b1.yaml).",
    )
    parser.add_argument("--predictions-csv", "--predictions", dest="predictions_csv", required=True)
    parser.add_argument("--output-json", "--output", dest="output_json", required=True)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--identifier-column", default=None)
    parser.add_argument("--true-label-column", default=None)
    parser.add_argument("--predicted-class-column", default=None)
    scores = parser.add_mutually_exclusive_group()
    scores.add_argument(
        "--probability-columns",
        nargs=3,
        metavar="COLUMN",
        help="Three probability columns in strict order: " + ", ".join(CLASS_ORDER),
    )
    scores.add_argument(
        "--logit-columns",
        nargs=3,
        metavar="COLUMN",
        help="Three logit columns in strict order: " + ", ".join(CLASS_ORDER),
    )
    return parser


def baseline_evaluation_cli(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config or str(PROJECT_ROOT / "configs" / f"{args.mode.lower()}.yaml")
    run_baseline_evaluation(
        args.mode,
        config_path=config_path,
        predictions_csv=args.predictions_csv,
        output_json=args.output_json,
        baseline_dir=args.baseline_dir,
        manifest_dir=args.manifest_dir,
        identifier_column=args.identifier_column,
        true_label_column=args.true_label_column,
        predicted_class_column=args.predicted_class_column,
        probability_columns=args.probability_columns,
        logit_columns=args.logit_columns,
    )
    return 0
