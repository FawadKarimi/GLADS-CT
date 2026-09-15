"""Explicit, evaluation-only attention/boundary inference artifact pipeline.

Saved-prediction analyses live in script 42. This separate opt-in path needs
the frozen rendered inputs and final best checkpoints; it never trains.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .phase4_boundary import (
    BOUNDARY_LIMITATIONS,
    assert_boundary_source_snapshot_unchanged,
    load_repository_phase4_boundary_source,
)
from .phase4_evaluation_runner import (
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_EVALUATION_OUTPUT_ROOT,
    FINAL_SEEDS,
    PROJECT_ROOT,
    Phase4EvaluationError,
    Phase4EvaluationResult,
    assert_source_snapshot_unchanged,
    create_evaluation_layout,
    load_phase4_evaluation_config,
    load_phase4_final_prediction_source,
    relative_source_snapshot,
    validate_evaluator_repository,
    validate_final_source_descriptors,
)
from .phase4_posthoc import (
    aggregate_attention_seed_results,
    build_frozen_attention_loader,
    evaluate_attention_alignment_from_loader,
    load_frozen_final_best_model,
    predict_validated_boundary_cohort,
)


def run_phase4_posthoc_evaluation(
    *,
    phase4_source_dirs: Sequence[str | Path],
    output_root: str | Path = DEFAULT_EVALUATION_OUTPUT_ROOT,
    evaluation_config_path: str | Path = DEFAULT_EVALUATION_CONFIG,
    include_boundary: bool = False,
    device: str = "cpu",
) -> Phase4EvaluationResult:
    """Run exactly three frozen seeds; seal new outputs only after revalidation."""

    if type(include_boundary) is not bool or device not in {"cpu", "cuda"}:
        raise Phase4EvaluationError("Use a boolean boundary flag and cpu/cuda device")
    if len(phase4_source_dirs) != len(FINAL_SEEDS):
        raise Phase4EvaluationError("Post-hoc inference requires exactly three final seeds")
    config = load_phase4_evaluation_config(evaluation_config_path)
    git_before = validate_evaluator_repository(PROJECT_ROOT)
    sources = [load_phase4_final_prediction_source(path) for path in phase4_source_dirs]
    descriptors = validate_final_source_descriptors([source.descriptor for source in sources])
    by_seed = {source.descriptor["seed"]: source for source in sources}
    sources = [by_seed[seed] for seed in FINAL_SEEDS]
    records = [
        {
            "descriptor": dict(source.descriptor),
            "source_dir": str(source.source_dir),
            "source_snapshot": relative_source_snapshot(source),
        }
        for source in sources
    ]
    boundary = load_repository_phase4_boundary_source(project_root=PROJECT_ROOT) if include_boundary else None
    identity = {
        "analysis": "phase4_frozen_attention_and_optional_boundary",
        "evaluator_commit": git_before["commit"],
        "evaluation_config_sha256": sha256_json(config),
        "source_snapshots": [record["source_snapshot"] for record in records],
        "source_descriptors": descriptors,
        "boundary_cohort_sha256": None if boundary is None else boundary.provenance["boundary_cohort_sha256"],
        "device": device,
    }
    evaluation_id = f"phase4-evaluation-{sha256_json(identity)[:16]}"
    protected = [source.source_dir for source in sources]
    if boundary is not None:
        protected.extend(Path(path).parent for path in boundary.source_snapshot)
    # Reserve once, before inference. A failed partial bundle is never complete
    # and must be retained/relocated explicitly before a rerun at the same ID.
    layout = create_evaluation_layout(output_root, evaluation_id, protected_sources=protected)
    all_snapshots: dict[str, str] = {}
    for source in sources:
        all_snapshots.update(source.source_snapshot)
    if boundary is not None:
        all_snapshots.update(boundary.source_snapshot)
    attention_results: list[dict[str, Any]] = []
    boundary_results: list[dict[str, Any]] = []
    import torch

    prior_determinism = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for source in sources:
            assert_source_snapshot_unchanged(all_snapshots)
            frozen = load_frozen_final_best_model(source.source_dir, device=device)
            loader = build_frozen_attention_loader(frozen)
            all_snapshots.update(loader.input_snapshot)
            attention_results.append(evaluate_attention_alignment_from_loader(frozen, loader))
            if boundary is not None:
                boundary_results.append(predict_validated_boundary_cohort(frozen, boundary))
            del loader, frozen
    finally:
        torch.use_deterministic_algorithms(prior_determinism, warn_only=prior_warn_only)
    attention = aggregate_attention_seed_results(attention_results)
    boundary_summary: list[dict[str, Any]] = []
    for result in boundary_results:
        frame = pd.DataFrame(result["rows"])
        for stratum, group in frame.groupby("boundary_stratum", sort=True):
            boundary_summary.append({
                "seed": result["seed"], "boundary_stratum": stratum,
                "sample_count": len(group), "patient_count": group["patient_id"].nunique(),
                **{f"mean_prob_{name}": float(group[f"prob_{name}"].mean()) for name in CLASS_ORDER},
            })
    analysis = {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "attention_alignment": attention,
        "boundary_stress": {
            "status": "evaluated_separately" if include_boundary else "not_requested",
            "per_seed": boundary_results, "descriptive_probability_summary": boundary_summary,
            "primary_metrics_computed": False,
            "limitations": list(BOUNDARY_LIMITATIONS),
        },
        "training_performed": False, "test_driven_tuning": False,
        "causal_explanation_claimed": False,
    }
    artifacts: list[Path] = []
    machine = atomic_write_json(layout.output_dir / "evaluation.json", analysis)
    artifacts.append(machine)
    attention_rows = [
        {"seed": result["seed"], "experiment_id": result["experiment_id"], **row}
        for result in attention_results for row in result["metrics"]["per_sample"]
    ]
    seed_rows = [
        {"seed": result["seed"], "metric": metric, **values}
        for result in attention_results for metric, values in result["metrics"]["aggregate"].items()
    ]
    aggregate_rows = [{"metric": metric, **values} for metric, values in attention["aggregate"].items()]
    tables = {
        "attention_per_nodule_per_seed.csv": attention_rows,
        "attention_per_seed.csv": seed_rows,
        "attention_aggregate.csv": aggregate_rows,
    }
    if include_boundary:
        tables["boundary_predictions_separate.csv"] = [
            {"seed": result["seed"], "experiment_id": result["experiment_id"], **row}
            for result in boundary_results for row in result["rows"]
        ]
        tables["boundary_descriptive_per_seed.csv"] = boundary_summary
    for name, rows in tables.items():
        artifacts.append(atomic_write_csv(layout.tables_dir / name, pd.DataFrame(rows)))
    provenance = atomic_write_json(layout.provenance_dir / "provenance.json", {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "identity": identity, "evaluator_git": git_before,
        "evaluation_config": config, "source_experiments": records,
        "input_file_snapshot": all_snapshots,
        "input_file_snapshot_sha256": sha256_json(all_snapshots),
        "boundary_provenance": None if boundary is None else dict(boundary.provenance),
        "seeds": list(FINAL_SEEDS), "class_order": list(CLASS_ORDER),
        "selected_lambdas": config["selected_lambdas"],
        "checkpoint_kind": "best", "deterministic_algorithms": True,
        "training_performed": False, "source_overwrite_permitted": False,
    })
    lines = [
        f"# {evaluation_id}", "", "Frozen best-checkpoint inference only; no training or tuning.", "",
        "Attention alignment measures spatial correspondence, not causal explanation.", "",
        "| Metric | Mean of seed means | Sample SD | Defined seeds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(f"| {row['metric']} | {row['mean_of_seed_means']} | {row['standard_deviation_of_seed_means']} | {row['n_defined_seeds']} |")
    lines.extend(["", f"Boundary analysis: {analysis['boundary_stress']['status']}.",
                  "Boundary rows retain their original split and reader targets; no primary class is invented.",
                  "Boundary results are descriptive and excluded from the primary test headline.", ""])
    for row in boundary_summary:
        lines.append(f"Seed {row['seed']}, {row['boundary_stratum']}: {row['sample_count']} nodules / {row['patient_count']} patients.")
    summary = atomic_write_text(layout.output_dir / "summary.md", "\n".join(lines) + "\n")
    artifacts.extend((provenance, summary))
    assert_source_snapshot_unchanged(all_snapshots)
    if boundary is not None:
        assert_boundary_source_snapshot_unchanged(boundary.source_snapshot)
    if validate_evaluator_repository(PROJECT_ROOT) != git_before:
        raise Phase4EvaluationError("Evaluator Git state changed during inference")
    files = {path.relative_to(layout.output_dir).as_posix(): sha256_file(path) for path in sorted(artifacts)}
    manifest = atomic_write_json(layout.output_dir / "artifact_manifest.json", {
        "schema_version": 1, "evaluation_id": evaluation_id, "files": files,
        "bundle_sha256": sha256_json(files), "research_results_eligible": True,
        "source_artifacts_unchanged": True, "training_performed": False,
    })
    for relative, digest in files.items():
        if sha256_file(layout.output_dir / relative) != digest:
            raise Phase4EvaluationError(f"Output changed before completion: {relative}")
    assert_source_snapshot_unchanged(all_snapshots)
    completion = atomic_write_json(layout.output_dir / "COMPLETED.json", {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "artifact_manifest": manifest.name, "artifact_manifest_sha256": sha256_file(manifest),
        "research_results_eligible": True, "source_artifacts_unchanged": True,
        "training_performed": False,
    })
    try:
        assert_source_snapshot_unchanged(all_snapshots)
    except BaseException:
        completion.unlink(missing_ok=True)
        raise
    return Phase4EvaluationResult(evaluation_id, layout.output_dir, machine, provenance, summary, manifest, completion)


def phase4_posthoc_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen final attention/boundary inference; never trains.")
    parser.add_argument("--phase4-source", action="append", required=True, help="Pass exactly three completed final experiment directories.")
    parser.add_argument("--config", default=str(DEFAULT_EVALUATION_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_EVALUATION_OUTPUT_ROOT))
    parser.add_argument("--include-boundary", action="store_true", help="Explicitly include verified Phase 2 excluded-boundary inference, separately.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = run_phase4_posthoc_evaluation(
        phase4_source_dirs=args.phase4_source, output_root=args.output_root,
        evaluation_config_path=args.config, include_boundary=args.include_boundary, device=args.device,
    )
    print(f"PHASE4_EVALUATION_ID={result.evaluation_id}")
    print(f"OUTPUT_DIR={result.output_dir}")
    print(f"COMPLETION_MARKER={result.completion_marker}")
    return 0
