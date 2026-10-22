"""Opt-in exploratory attention comparison and evidence-bundle orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .phase4_evaluation_runner import (
    PROJECT_ROOT, FINAL_SEEDS, DEFAULT_EVALUATION_OUTPUT_ROOT,
    _safe_source_member, _values_are_exactly_equal, create_evaluation_layout,
    load_saved_prediction_inputs, relative_source_snapshot, snapshot_source_files,
    assert_source_snapshot_unchanged, validate_evaluator_repository,
)
from .phase4_posthoc import build_frozen_attention_loader, load_frozen_final_best_model
from .phase4_attention_runtime import pinned_b1_api, load_frozen_b1, infer_matched_attention
from .phase4_attention_evidence import (
    FLAGGED_NODULE, IDENTITY, METRICS, build_location_prior, map_evidence,
    paired_alignment_bootstrap, select_panel_cases, supplementary_protocol,
    write_map_archive, write_overlay_panel,
)

PROTOCOL_PATH = PROJECT_ROOT / "configs" / "phase4_attention_evidence_v1.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_bundle(path: Path, phase4_sources: dict) -> tuple[dict, dict[str, str]]:
    """Verify the earlier attention bundle and bind it to all three same models."""
    root = path.resolve()
    completion, manifest = _json(root / "COMPLETED.json"), _json(root / "artifact_manifest.json")
    if (completion.get("artifact_manifest") != "artifact_manifest.json"
        or type(completion.get("schema_version")) is not int or completion["schema_version"] != 1
        or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
        or not isinstance(completion.get("evaluation_id"), str) or not completion["evaluation_id"]
        or completion.get("artifact_manifest_sha256") != sha256_file(root / "artifact_manifest.json")
        or completion.get("evaluation_id") != manifest.get("evaluation_id")
        or completion.get("research_results_eligible") is not True
        or manifest.get("research_results_eligible") is not True):
        raise ValueError("Previous attention bundle completion is inconsistent")
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("bundle_sha256") != sha256_json(files):
        raise ValueError("Previous attention bundle checksum ledger is invalid")
    snapshot = snapshot_source_files([root / "COMPLETED.json", root / "artifact_manifest.json"])
    for relative, expected in files.items():
        canonical, member = _safe_source_member(root, relative)
        if canonical != relative or sha256_file(member) != expected:
            raise ValueError("Previous attention bundle member checksum differs")
        snapshot[str(member)] = expected
    if not {"evaluation.json", "provenance/provenance.json"}.issubset(files):
        raise ValueError("Previous attention bundle lacks results/provenance")
    data = _json(root / "evaluation.json")
    rows = data.get("attention_alignment", {}).get("per_seed", [])
    if sorted(row.get("seed", -1) for row in rows) != list(FINAL_SEEDS):
        raise ValueError("Previous attention bundle must contain each final seed once")
    for row in rows:
        source = phase4_sources[row["seed"]]
        if row.get("experiment_id") != source.descriptor["experiment_id"] or row.get("checkpoint_kind") != "best" or row.get("checkpoint_sha256") != sha256_file(source.source_dir / "checkpoints" / "best.pt"):
            raise ValueError("Previous attention bundle refers to another frozen checkpoint")
        if row.get("definition") != supplementary_protocol()["alignment_definition"]:
            raise ValueError("Previous attention metric definitions differ")
        for field in ("manifest_bundle_sha256", "supervision_sha256", "ordered_supervision_sha256", "input_lock_sha256"):
            if row.get("source_cohort", {}).get(field) != source.descriptor[field]:
                raise ValueError(f"Previous attention source binding differs: {field}")
        old_ids = pd.DataFrame(row["metrics"]["per_sample"])
        fields = ["patient_id", "nodule_id", "series_instance_uid"]
        if old_ids["nodule_id"].duplicated().any() or old_ids[fields].sort_values(fields).to_dict("records") != source.predictions[fields].sort_values(fields).to_dict("records"):
            raise ValueError("Previous attention bundle cohort differs")
    assert_source_snapshot_unchanged(snapshot)
    return {row["seed"]: row for row in rows}, snapshot


def _training_prior(frozen, grid: tuple[int, int]):
    from .phase4_dataset import _read_soft_mask
    from .manifest_guard import validate_frozen_manifest_bundle
    from .phase4_readiness import validate_phase4_readiness

    config = _json(frozen.source.source_dir / "config" / "effective_config.json")
    rendered = Path(config["data"]["rendered_dir"])
    if not rendered.is_absolute():
        rendered = PROJECT_ROOT / rendered
    training = frozen.source.supervision.loc[frozen.source.supervision["split"] == "train"].copy()
    training = training.rename(columns={"label": "true_class"})
    paths = [(rendered / row["rendered_mask_path"]).resolve() for row in training.to_dict("records")]
    snapshot = snapshot_source_files(paths)
    manifest_path = Path(config["data"]["frozen_split_dir"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    # Seal before revalidation: a changed training mask must not be accepted
    # merely because it changed between test inference and prior construction.
    readiness, live = validate_phase4_readiness(
        config["data"], manifest=validate_frozen_manifest_bundle(manifest_path),
        expected_resolution=int(config["dataset"]["image_size"]),
        project_root=PROJECT_ROOT, synthetic_dry_run=False,
    )
    for field in ("manifest_bundle_sha256", "supervision_sha256", "ordered_supervision_sha256"):
        if getattr(readiness, field) != frozen.source.descriptor[field]:
            raise ValueError(f"Training prior readiness differs from final source: {field}")
    if readiness.phase3_input_fingerprint.to_dict() != frozen.source.descriptor["phase3_input_fingerprint"]:
        raise ValueError("Training prior Phase 3 input fingerprint differs")
    live_train = live.loc[live["split"] == "train"].rename(columns={"label": "true_class"})
    fields = [*IDENTITY, "rendered_mask_path"]
    if live_train[fields].to_dict("records") != training[fields].to_dict("records"):
        raise ValueError("Training prior is not the exact readiness-approved training cohort")
    masks = np.stack([_read_soft_mask(path, int(config["dataset"]["image_size"])) for path in paths])
    prior, provenance = build_location_prior(training, masks, grid)
    provenance["mask_sources"] = [
        {"nodule_id": row["nodule_id"], "path": str(path), "sha256": snapshot[str(path)]}
        for row, path in zip(training.to_dict("records"), paths)
    ]
    provenance["source_supervision_sha256"] = frozen.source.descriptor["supervision_sha256"]
    provenance["readiness_revalidated"] = readiness.to_dict()
    provenance["mask_source_snapshot_sha256"] = sha256_json(snapshot)
    assert_source_snapshot_unchanged(snapshot)
    return prior, provenance, snapshot


def compare_scalar_records(actual: list[dict], expected: list[dict], tolerance: dict) -> dict[str, Any]:
    """Report discrepancies; never rewrite an earlier scalar bundle."""
    a, b = pd.DataFrame(actual), pd.DataFrame(expected)
    fields = ["patient_id", "nodule_id", "series_instance_uid"]
    a = a.sort_values(fields).reset_index(drop=True)
    b = b.sort_values(fields).reset_index(drop=True)
    if a[fields].to_dict("records") != b[fields].to_dict("records") or a["nodule_id"].duplicated().any():
        raise ValueError("Scalar comparison cohort mismatch")
    differences = []
    for metric in METRICS:
        for i, (left, right) in enumerate(zip(a[metric], b[metric])):
            missing_a, missing_b = pd.isna(left), pd.isna(right)
            match = bool(missing_a and missing_b) or (not missing_a and not missing_b and bool(np.isclose(left, right, **tolerance)))
            if not match:
                differences.append({"nodule_id": a.iloc[i]["nodule_id"], "metric": metric,
                                    "reconstructed": None if missing_a else float(left),
                                    "reference": None if missing_b else float(right)})
    return {"consistent": not differences, "tolerance": tolerance, "discrepancies": differences}


def _aggregate_replay(actual: dict, expected: dict, tolerance: dict) -> dict:
    differences = []
    for metric in METRICS:
        for field in ("mean", "standard_deviation", "n_defined"):
            a, b = actual[metric][field], expected[metric][field]
            matches = a is None and b is None
            if a is not None and b is not None:
                matches = a == b if field == "n_defined" else bool(np.isclose(a, b, **tolerance))
            if not matches:
                differences.append({"metric": metric, "field": field, "reconstructed": a, "reference": b})
    return {"consistent": not differences, "tolerance": tolerance, "discrepancies": differences}


def run_attention_comparison(*, phase4_source_dirs: Sequence[str | Path], b1_source_dirs: Sequence[str | Path],
                             reference_attention_bundle: str | Path, output_root: str | Path = DEFAULT_EVALUATION_OUTPUT_ROOT,
                             device: str = "cpu") -> Path:
    """New evidence only; requires real frozen inputs but is never called by unit fixtures unguarded."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    requested_output = Path(output_root).resolve()
    if any((parent / "COMPLETED.json").is_file() for parent in (requested_output, *requested_output.parents)):
        raise ValueError("New evidence cannot be placed inside an existing completed bundle")
    protocol = supplementary_protocol()
    import torch
    import cv2

    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "torch": str(torch.__version__), "opencv": cv2.__version__, "device": device,
                   "torch_cuda_version": torch.version.cuda, "torch_threads": torch.get_num_threads(),
                   "deterministic_algorithms": True, "autocast": False}
    if not _values_are_exactly_equal(_json(PROTOCOL_PATH), protocol):
        raise ValueError("Supplementary protocol differs from version 1")
    git_before = validate_evaluator_repository(PROJECT_ROOT)
    inputs = load_saved_prediction_inputs(phase4_source_dirs, b1_source_dirs)
    test = inputs.supervision.loc[inputs.supervision["split"] == "test"]
    if len(test) != protocol["test_nodules"] or test["patient_id"].nunique() != protocol["test_patients"]:
        raise ValueError("Evidence requires the frozen 183-nodule / 99-patient test cohort")
    if int(inputs.supervision["split"].eq("train").sum()) != protocol["training_nodules"] or FLAGGED_NODULE not in set(test["nodule_id"]):
        raise ValueError("Frozen training count or flagged test nodule differs")
    reference_root = Path(reference_attention_bundle).resolve()
    reference, reference_snapshot = load_reference_bundle(reference_root, inputs.phase4)
    records = [{"model": model, "seed": seed, "descriptor": dict(sources[seed].descriptor),
                "source_snapshot": relative_source_snapshot(sources[seed])}
               for model, sources in (("B1", inputs.b1), ("ConsenDxFormer", inputs.phase4)) for seed in FINAL_SEEDS]
    identity = {"protocol_sha256": sha256_json(protocol), "evaluator_commit": git_before["commit"],
                "sources": records, "reference_manifest_sha256": sha256_file(reference_root / "artifact_manifest.json"),
                "runtime_environment": environment, "device": device}
    evaluation_id = f"phase4-evaluation-{sha256_json(identity)[:16]}"
    protected = [source.source_dir for source in (*inputs.b1.values(), *inputs.phase4.values())]
    protected.extend([reference_root, PROJECT_ROOT / "external", PROJECT_ROOT / "artifacts" / "phase0",
                      PROJECT_ROOT / "artifacts" / "phase2", PROJECT_ROOT / "data"])
    layout = create_evaluation_layout(output_root, evaluation_id, protected_sources=protected)
    snapshot = {**inputs.source_snapshot, **reference_snapshot, str(PROTOCOL_PATH): sha256_file(PROTOCOL_PATH)}
    archives, rows, seed_results, paired, paired_rows = [], [], [], [], []
    prior = None
    shared = None
    panel_arrays = None
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with pinned_b1_api(layout.provenance_dir / ".disabled_pycache") as (model_class, baseline_source):
            for seed in FINAL_SEEDS:
                assert_source_snapshot_unchanged(snapshot)
                cdx = load_frozen_final_best_model(inputs.phase4[seed].source_dir, device=device)
                b1 = load_frozen_b1(inputs.b1[seed].source_dir, model_class, device=device)
                loader = build_frozen_attention_loader(cdx)
                snapshot.update(loader.input_snapshot)
                result = infer_matched_attention(cdx, b1, loader)
                arrays, identities = result["arrays"], result["identities"]
                cohort_records = identities.to_dict("records")
                if shared is None:
                    shared = {"identities": cohort_records, "ct": arrays["ct"], "masks": arrays["masks"]}
                elif (shared["identities"] != cohort_records or not np.array_equal(shared["ct"], arrays["ct"]) or not np.array_equal(shared["masks"], arrays["masks"])):
                    raise ValueError("Frozen images/targets/identities differ across seeds")
                grid = tuple(int(n) for n in arrays["cdx_attention"].shape[-2:])
                if prior is None:
                    prior, prior_provenance, prior_snapshot = _training_prior(cdx, grid)
                    snapshot.update(prior_snapshot)
                    prior_evidence = map_evidence(np.broadcast_to(prior, (len(identities), *grid)), arrays["masks"], cohort_records)
                    archives.append(write_map_archive(layout.output_dir / "location_prior.npz", {"prior": prior}))
                    archives.append(write_map_archive(layout.output_dir / "test_inputs.npz", {
                        **{field: np.asarray(identities[field].tolist(), dtype=np.str_) for field in IDENTITY},
                        "ct_crops": arrays["ct"], "soft_masks_input": arrays["masks"],
                        "soft_masks_native_grid": prior_evidence["resized_soft_masks"],
                        "target_distributions": prior_evidence["target_distributions"],
                    }))
                if prior.shape != grid:
                    raise ValueError("Native attention grid changed across seeds")
                evidence = {name: map_evidence(arrays[f"{key}_attention"], arrays["masks"], cohort_records)
                            for name, key in (("B1", "b1"), ("ConsenDxFormer", "cdx"))}
                archive_path = layout.output_dir / f"seed_{seed}_model_arrays.npz"
                exported = {name: array for name, array in arrays.items() if name not in {"ct", "masks"}}
                archives.append(write_map_archive(archive_path, exported))
                with np.load(archive_path, allow_pickle=False) as reloaded:
                    for model, key in (("B1", "b1"), ("ConsenDxFormer", "cdx")):
                        replayed = map_evidence(reloaded[f"{key}_attention"], arrays["masks"], cohort_records)
                        if not compare_scalar_records(replayed["rows"], evidence[model]["rows"], protocol["exported_map_scalar_replay"])["consistent"]:
                            raise ValueError("Exported map scalar round-trip differs")
                prior_rows = pd.DataFrame(prior_evidence["rows"])
                for model in ("B1", "ConsenDxFormer"):
                    rows.extend({**row, "model": model, "seed": seed} for row in evidence[model]["rows"])
                for left_name, right_name in (("ConsenDxFormer", "B1"), ("ConsenDxFormer", "prior"), ("B1", "prior")):
                    left = pd.DataFrame(evidence[left_name]["rows"])
                    right = prior_rows if right_name == "prior" else pd.DataFrame(evidence[right_name]["rows"])
                    comparison = f"{left_name}_minus_{right_name}"
                    paired.append({"seed": seed, "comparison": comparison, **paired_alignment_bootstrap(left, right)})
                    for a, b in zip(left.to_dict("records"), right.to_dict("records")):
                        paired_rows.append({**{field: a[field] for field in IDENTITY}, "seed": seed, "comparison": comparison,
                                            **{metric: None if pd.isna(a[metric]) or pd.isna(b[metric]) else a[metric] - b[metric] for metric in METRICS}})
                scalar_replay = compare_scalar_records(evidence["ConsenDxFormer"]["rows"], reference[seed]["metrics"]["per_sample"], protocol["previous_scalar_replay"])
                aggregate_replay = _aggregate_replay(evidence["ConsenDxFormer"]["metrics"]["aggregate"], reference[seed]["metrics"]["aggregate"], protocol["previous_scalar_replay"])
                seed_results.append({"seed": seed, "classification_replay": result["replay"], "previous_scalar_replay": scalar_replay,
                                     "previous_aggregate_replay": aggregate_replay,
                                     "actual_attention_shapes": result["actual_attention_shapes"], "source_cohort": result["source_cohort"],
                                     "best_checkpoints": {"B1": b1.checkpoint_sha256, "ConsenDxFormer": cdx.checkpoint_sha256},
                                     "alignment": {name: value["metrics"]["aggregate"] for name, value in evidence.items()}})
                if seed == 42:
                    panel_arrays = (arrays["ct"], prior_evidence["resized_soft_masks"], arrays["b1_attention"][:, 0], arrays["cdx_attention"][:, 0])
                del cdx, b1, loader
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn)
    frame = pd.DataFrame(rows)
    cases = select_panel_cases(frame.loc[frame["model"] == "ConsenDxFormer"])
    artifacts = [layout.output_dir / record["file"] for record in archives]
    artifacts.append(write_overlay_panel(layout.output_dir / "overlay_panel.png", cases, pd.DataFrame(shared["identities"]), *panel_arrays, prior))
    for name, table in (("alignment_per_nodule.csv", frame), ("paired_differences_per_nodule.csv", pd.DataFrame(paired_rows)),
                        ("prior_per_nodule.csv", pd.DataFrame(prior_evidence["rows"])),
                        ("flagged_case.csv", pd.concat([frame.loc[frame["nodule_id"] == FLAGGED_NODULE],
                                                       pd.DataFrame(prior_evidence["rows"]).query("nodule_id == @FLAGGED_NODULE").assign(model="prior", seed="fixed")], ignore_index=True))):
        artifacts.append(atomic_write_csv(layout.tables_dir / name, table))
    consistent = all(row["previous_scalar_replay"]["consistent"] and row["previous_aggregate_replay"]["consistent"]
                     and all(replay["consistent"] for replay in row["classification_replay"].values()) for row in seed_results)
    aggregate_rows = []
    for model in ("B1", "ConsenDxFormer"):
        for metric in METRICS:
            values = [row["alignment"][model][metric]["mean"] for row in seed_results]
            values = [value for value in values if value is not None]
            aggregate_rows.append({"model": model, "metric": metric, "n_defined_seeds": len(values),
                                   "mean_of_seed_means": float(np.mean(values)) if values else None,
                                   "sample_sd_of_seed_means": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    artifacts.append(atomic_write_csv(layout.tables_dir / "alignment_aggregate.csv", pd.DataFrame(aggregate_rows)))
    artifacts.append(atomic_write_csv(layout.tables_dir / "alignment_per_seed.csv", pd.DataFrame([
        {"seed": row["seed"], "model": model, "metric": metric, **stats}
        for row in seed_results for model, metrics in row["alignment"].items() for metric, stats in metrics.items()
    ])))
    artifacts.append(atomic_write_csv(layout.tables_dir / "paired_patient_confidence_intervals.csv", pd.DataFrame([
        {"seed": row["seed"], "comparison": row["comparison"], "metric": metric, **stats}
        for row in paired for metric, stats in row["metrics"].items()
    ])))
    analysis = {"schema_version": 1, "evaluation_id": evaluation_id, "protocol": protocol,
                "status": "complete" if consistent else "complete_with_replay_discrepancies",
                "replay_consistent": consistent, "comparison_interpretation_permitted": consistent,
                "seed_results": seed_results, "patient_bootstrap": paired, "prior_alignment": prior_evidence["metrics"]["aggregate"],
                "aggregate_alignment": aggregate_rows,
                "panel_cases": cases, "training_performed": False, "causal_explanation_claimed": False}
    provenance = {"identity": identity, "evaluator_git": git_before, "baseline_source": baseline_source,
                  "source_experiments": records, "prior": prior_provenance, "archives": archives,
                  "input_file_snapshot": snapshot, "input_file_snapshot_sha256": sha256_json(snapshot),
                  "previous_attention_bundle": str(reference_root), "training_performed": False}
    artifacts.extend([atomic_write_json(layout.output_dir / "evaluation.json", analysis),
                      atomic_write_json(layout.provenance_dir / "provenance.json", provenance),
                      atomic_write_json(layout.provenance_dir / "supplementary_protocol.json", protocol)])
    summary = [f"# {evaluation_id}", "", "Exploratory post-hoc evidence, proposed after original attention results; not preregistered.",
               "No training or tuning. Attention alignment is spatial correspondence, not causal explanation or an attention-loss ablation.",
               f"Replay status: {analysis['status']}. See evaluation.json for every discrepancy; tolerances were not adjusted.", "",
               "Per-seed mean alignment (prior is identical across test nodules and seeds):", ""]
    for seed in seed_results:
        for model, metrics in seed["alignment"].items():
            summary.append(f"Seed {seed['seed']} {model}: " + "; ".join(f"{name}={values['mean']}" for name, values in metrics.items()))
    summary.extend(["", "Paired patient-cluster percentile CIs are per trained seed, not pooled across 549 nodule-seed observations.",
                    "The flagged case exports foreground counts and an explicit empty-target versus disjoint-foreground diagnosis.",
                    "Existing bundles, model weights, predictions and metric definitions were not modified."])
    artifacts.append(atomic_write_text(layout.output_dir / "summary.md", "\n".join(summary) + "\n"))
    assert_source_snapshot_unchanged(snapshot)
    if validate_evaluator_repository(PROJECT_ROOT) != git_before:
        raise ValueError("Evaluator Git state changed during supplementary inference")
    files = {path.relative_to(layout.output_dir).as_posix(): sha256_file(path) for path in sorted(artifacts)}
    manifest_path = atomic_write_json(layout.output_dir / "artifact_manifest.json", {
        "schema_version": 1, "evaluation_id": evaluation_id, "files": files, "bundle_sha256": sha256_json(files),
        "exploratory_post_hoc": True, "replay_consistent": consistent, "training_performed": False,
    })
    for relative, digest in files.items():
        if sha256_file(layout.output_dir / relative) != digest:
            raise ValueError("Evidence output changed before completion")
    assert_source_snapshot_unchanged(snapshot)
    marker = atomic_write_json(layout.output_dir / "COMPLETED.json", {
        "schema_version": 1, "evaluation_id": evaluation_id, "artifact_manifest": manifest_path.name,
        "artifact_manifest_sha256": sha256_file(manifest_path), "exploratory_post_hoc": True,
        "replay_consistent": consistent, "comparison_interpretation_permitted": consistent, "training_performed": False,
    })
    try:
        assert_source_snapshot_unchanged(snapshot)
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    return layout.output_dir


def attention_comparison_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory frozen B1/CDX attention comparison, train-only prior and evidence export; never trains.")
    parser.add_argument("--phase4-source", action="append", required=True)
    parser.add_argument("--b1-source", action="append", required=True)
    parser.add_argument("--reference-attention-bundle", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_EVALUATION_OUTPUT_ROOT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = run_attention_comparison(phase4_source_dirs=args.phase4_source, b1_source_dirs=args.b1_source,
                                      reference_attention_bundle=args.reference_attention_bundle,
                                      output_root=args.output_root, device=args.device)
    print(f"OUTPUT_DIR={result}")
    print(f"COMPLETION_MARKER={result / 'COMPLETED.json'}")
    return 0
