"""Explicit multi-seed B0/B1 orchestration with fair-pair preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baseline_evaluation import CLASS_ORDER
from .baseline_provenance import utc_timestamp
from .hashing import sha256_json
from .manifest_guard import validate_frozen_manifest_bundle
from .phase3_artifacts import atomic_write_json
from .phase3_comparison import (
    build_comparison_descriptor,
    validate_fair_baseline_pair,
)
from .phase3_config import Phase3ConfigBundle, load_phase3_config
from .phase3_readiness import validate_phase3_readiness
from .phase3_runner import PROJECT_ROOT, Phase3RunResult, run_phase3_experiment
from .reproducibility import validate_seed


class Phase3OrchestrationError(RuntimeError):
    """Raised when a matrix is ambiguous, unsafe, or not fair-comparable."""


@dataclass(frozen=True)
class Phase3MatrixResult:
    matrix_id: str
    state: str
    index_path: Path
    run_count: int
    succeeded: int
    failed: int


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be an explicit sequence of integers")
    if not seeds:
        raise Phase3OrchestrationError("At least one explicit seed is required")
    canonical = tuple(validate_seed(seed) for seed in seeds)
    if len(canonical) != len(set(canonical)):
        raise Phase3OrchestrationError("The explicit seed list contains duplicates")
    return canonical


def _load_matrix_bundles(
    config_paths: Mapping[str, str | Path], seeds: tuple[int, ...]
) -> dict[str, Phase3ConfigBundle]:
    if set(config_paths) != {"B0", "B1"}:
        raise Phase3OrchestrationError(
            "config_paths must contain exactly one explicit B0 and one explicit B1 config"
        )
    return {
        variant: load_phase3_config(
            config_paths[variant], expected_variant=variant, seed=seeds[0]
        )
        for variant in ("B0", "B1")
    }


def _preflight_fair_pairs(
    bundles: Mapping[str, Phase3ConfigBundle],
    seeds: tuple[int, ...],
    *,
    project_root: Path,
    synthetic_dry_run: bool,
) -> list[dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}
    for variant in ("B0", "B1"):
        bundle = bundles[variant]
        phase3 = bundle.phase3_config
        phase1 = bundle.effective_phase1_config
        manifest_root = _project_path(
            phase3["data"]["frozen_split_dir"], project_root
        )
        manifest = validate_frozen_manifest_bundle(manifest_root)
        readiness, _ = validate_phase3_readiness(
            phase3["data"],
            manifest=manifest,
            expected_resolution=int(phase1["dataset"]["image_size"]),
            project_root=project_root,
            synthetic_dry_run=synthetic_dry_run,
        )
        fingerprint = readiness.fingerprint
        descriptors[variant] = {
            "semantic_cohort_checksum": fingerprint.logical_cohort_sha256,
            "frozen_split_checksum": readiness.manifest_bundle_sha256,
            "ordered_sample_checksum": fingerprint.ordered_samples_sha256,
            "class_order": list(phase3["predictions"]["class_order"]),
            "tta_enabled": bool(phase3["inference"]["tta"]),
        }

    reports: list[dict[str, Any]] = []
    for seed in seeds:
        reports.append(
            validate_fair_baseline_pair(
                build_comparison_descriptor(
                    baseline_variant="B0", seed=seed, **descriptors["B0"]
                ),
                build_comparison_descriptor(
                    baseline_variant="B1", seed=seed, **descriptors["B1"]
                ),
            )
        )
    return reports


def _read_identity(result: Phase3RunResult) -> Mapping[str, Any]:
    try:
        record = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase3OrchestrationError(
            f"Cannot read completed run provenance: {result.provenance_path}"
        ) from exc
    identity = record.get("identity") if isinstance(record, dict) else None
    if not isinstance(identity, dict):
        raise Phase3OrchestrationError("Completed run provenance lacks its identity")
    return identity


def _postflight_pair(
    b0: Phase3RunResult, b1: Phase3RunResult
) -> dict[str, Any]:
    if b0.experiment_id == b1.experiment_id or b0.output_dir.resolve() == b1.output_dir.resolve():
        raise Phase3OrchestrationError(
            "B0 and B1 must have independent experiment identities and directories"
        )
    descriptors: list[dict[str, Any]] = []
    for result in (b0, b1):
        identity = _read_identity(result)
        descriptors.append(
            build_comparison_descriptor(
                baseline_variant=result.baseline_variant,
                seed=result.seed,
                semantic_cohort_checksum=str(identity["cohort_sha256"]),
                frozen_split_checksum=str(
                    identity["frozen_split_bundle_sha256"]
                ),
                ordered_sample_checksum=str(identity["ordered_sample_sha256"]),
                class_order=CLASS_ORDER,
                tta_enabled=False,
                experiment_id=result.experiment_id,
            )
        )
    return validate_fair_baseline_pair(descriptors[0], descriptors[1])


def _validate_run_result(
    result: Phase3RunResult,
    *,
    variant: str,
    seed: int,
    mode: str,
    synthetic_dry_run: bool,
) -> Phase3RunResult:
    if not isinstance(result, Phase3RunResult):
        raise Phase3OrchestrationError("Matrix runner returned an unsupported result")
    expected_state = "synthetic_dry_run_completed" if synthetic_dry_run else "completed"
    if (
        result.baseline_variant != variant
        or result.seed != seed
        or result.mode != mode
        or result.synthetic_dry_run is not synthetic_dry_run
        or result.state != expected_state
        or result.research_results_eligible != (not synthetic_dry_run)
    ):
        raise Phase3OrchestrationError(
            "Matrix runner result disagrees with the planned variant, seed, mode, or state"
        )
    if not result.completion_marker.is_file():
        raise Phase3OrchestrationError(
            "Matrix runner reported success without a completion marker"
        )
    return result


def run_phase3_matrix(
    *,
    config_paths: Mapping[str, str | Path],
    seeds: Sequence[int],
    project_root: str | Path = PROJECT_ROOT,
    synthetic_dry_run: bool = False,
) -> Phase3MatrixResult:
    """Run every explicit seed for B0 and B1 in independent directories.

    Each ordinary run failure is recorded and orchestration continues. Guard or
    fairness preflight failures occur before any trainer is called.
    """

    canonical_seeds = _validate_seeds(seeds)
    root = Path(project_root).resolve()
    bundles = _load_matrix_bundles(config_paths, canonical_seeds)
    modes = {bundle.phase3_config["data"]["mode"] for bundle in bundles.values()}
    if len(modes) != 1:
        raise Phase3OrchestrationError("B0 and B1 must use the same data mode")
    mode = str(next(iter(modes)))
    if synthetic_dry_run != (mode == "synthetic"):
        raise Phase3OrchestrationError(
            "synthetic_dry_run must be true exactly when both configs use synthetic mode"
        )

    output_roots = {
        _project_path(bundle.phase3_config["experiment"]["output_root"], root)
        for bundle in bundles.values()
    }
    if len(output_roots) != 1:
        raise Phase3OrchestrationError("B0 and B1 must use the same output root")
    output_root = next(iter(output_roots))
    for bundle in bundles.values():
        protected = (
            _project_path(
                bundle.effective_phase1_config["baseline"]["repository_dir"], root
            ),
            _project_path(bundle.phase3_config["data"]["frozen_split_dir"], root),
        )
        for protected_root in protected:
            if output_root == protected_root or _is_within(output_root, protected_root):
                raise Phase3OrchestrationError(
                    f"Phase 3 matrix output cannot be inside protected path {protected_root}"
                )

    preflight = _preflight_fair_pairs(
        bundles,
        canonical_seeds,
        project_root=root,
        synthetic_dry_run=synthetic_dry_run,
    )
    matrix_payload = {
        "schema_version": 1,
        "mode": mode,
        "synthetic_dry_run": synthetic_dry_run,
        "seeds": list(canonical_seeds),
        "configs": {
            variant: {
                "path": str(bundles[variant].config_path),
                "combined_sha256_at_first_seed": bundles[
                    variant
                ].combined_config_sha256,
            }
            for variant in ("B0", "B1")
        },
        "preflight_fair_pairs": preflight,
    }
    matrix_digest = sha256_json(matrix_payload)
    matrix_id = f"phase3-matrix-{matrix_digest[:16]}"
    index_path = output_root / f"run_index.{matrix_id}.json"
    if index_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Phase 3 run index: {index_path}"
        )

    index: dict[str, Any] = {
        **matrix_payload,
        "matrix_id": matrix_id,
        "matrix_digest": matrix_digest,
        "state": "running",
        "created_utc": utc_timestamp(),
        "updated_utc": utc_timestamp(),
        "runs": [],
        "postflight_fair_pairs": [],
        "summary": {
            "planned": len(canonical_seeds) * 2,
            "succeeded": 0,
            "failed": 0,
            "comparison_failures": 0,
        },
    }
    atomic_write_json(index_path, index)

    successes: dict[tuple[str, int], Phase3RunResult] = {}
    for seed in canonical_seeds:
        for variant in ("B0", "B1"):
            started = utc_timestamp()
            try:
                result = _validate_run_result(
                    run_phase3_experiment(
                        config_path=bundles[variant].config_path,
                        project_root=root,
                        seed=seed,
                        synthetic_dry_run=synthetic_dry_run,
                    ),
                    variant=variant,
                    seed=seed,
                    mode=mode,
                    synthetic_dry_run=synthetic_dry_run,
                )
                identity_record = dict(_read_identity(result))
                if (
                    identity_record.get("experiment_id") != result.experiment_id
                    or identity_record.get("baseline_variant") != variant
                    or identity_record.get("seed") != seed
                ):
                    raise Phase3OrchestrationError(
                        "Completed run provenance identity disagrees with its matrix result"
                    )
                successes[(variant, seed)] = result
                run_record = {
                    **result.to_index_record(),
                    "identity": identity_record,
                    "started_utc": started,
                    "finished_utc": utc_timestamp(),
                }
                index["summary"]["succeeded"] += 1
            except Exception as exc:
                run_record = {
                    "baseline_variant": variant,
                    "seed": seed,
                    "state": "failed",
                    "started_utc": started,
                    "finished_utc": utc_timestamp(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                index["summary"]["failed"] += 1
            index["runs"].append(run_record)
            index["updated_utc"] = utc_timestamp()
            atomic_write_json(index_path, index, overwrite=True)

        b0 = successes.get(("B0", seed))
        b1 = successes.get(("B1", seed))
        if b0 is not None and b1 is not None:
            try:
                pair = _postflight_pair(b0, b1)
                index["postflight_fair_pairs"].append(pair)
            except Exception as exc:
                index["postflight_fair_pairs"].append(
                    {
                        "eligible": False,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                index["summary"]["comparison_failures"] += 1
        index["updated_utc"] = utc_timestamp()
        atomic_write_json(index_path, index, overwrite=True)

    failed = int(index["summary"]["failed"])
    comparison_failures = int(index["summary"]["comparison_failures"])
    succeeded = int(index["summary"]["succeeded"])
    state = (
        "completed"
        if failed == 0 and comparison_failures == 0
        else "completed_with_failures"
    )
    index["state"] = state
    index["updated_utc"] = utc_timestamp()
    atomic_write_json(index_path, index, overwrite=True)
    return Phase3MatrixResult(
        matrix_id=matrix_id,
        state=state,
        index_path=index_path,
        run_count=len(index["runs"]),
        succeeded=succeeded,
        failed=failed,
    )


def phase3_matrix_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an explicit, fair-comparable Phase 3 B0/B1 seed matrix."
    )
    parser.add_argument("--b0-config", default="configs/phase3_b0.yaml")
    parser.add_argument("--b1-config", default="configs/phase3_b1.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase3_matrix(
        config_paths={"B0": args.b0_config, "B1": args.b1_config},
        seeds=args.seeds,
        synthetic_dry_run=args.synthetic_dry_run,
    )
    print(f"PHASE3_MATRIX_STATE={result.state}")
    print(f"MATRIX_ID={result.matrix_id}")
    print(f"RUN_INDEX={result.index_path}")
    return 0 if result.state == "completed" else 1


__all__ = [
    "Phase3MatrixResult",
    "Phase3OrchestrationError",
    "phase3_matrix_cli",
    "run_phase3_matrix",
]
