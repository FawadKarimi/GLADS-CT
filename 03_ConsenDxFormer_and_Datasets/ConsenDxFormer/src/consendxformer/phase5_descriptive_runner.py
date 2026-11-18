"""Opt-in saved-artifact-only Phase 5 subset. No inference or formal testing."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .phase4_ablation_analysis import load_fixed_evidence
from .phase4_ablation_contracts import identity_rows
from .phase4_ablation_runner import guard_new_namespace, load_ablation_source, reference_record
from .phase4_attention_evidence import array_record, write_map_archive
from .phase4_attention_runtime import _saved_replay
from .phase4_evaluation_metrics import _area_resize
from .phase4_evaluation_runner import (
    PROJECT_ROOT, _safe_source_member, assert_source_snapshot_unchanged,
    load_saved_prediction_inputs, relative_source_snapshot, snapshot_source_files,
    validate_evaluator_repository,
)
from .phase5_descriptive import (
    IDENTITY, FLAGGED_NODULE, MODELS, SEEDS, SCOPE, VARIANTS, PROPOSAL_SHA256,
    accuracy_intervals, canonical_predictions, descriptive_protocol, identities,
    load_protocol, pointing_tables, unresolved_hypotheses, validate_native_inputs,
)

PROTOCOL_PATH = PROJECT_ROOT / "configs/phase5_descriptive_v1.json"
PROPOSAL_PATH = PROJECT_ROOT / "docs/phase5_supplemental_evaluation_proposal.md"
NAMESPACE = PROJECT_ROOT / "artifacts/phase5/pointing_and_accuracy"
WARNING = (
    "Exploratory post-hoc, pointing_and_accuracy_only. Pointing is coarse spatial "
    "correspondence, not causal faithfulness. Accuracy intervals are descriptive, "
    "unadjusted and conditional on fitted models, not simultaneous coverage or "
    "superiority tests. Repeated seeds are not independent patients. Formal tests "
    "and multiplicity remain unresolved; insertion/deletion and calibration were not executed."
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _merge_snapshot(target, incoming):
    for name, digest in incoming.items():
        if name in target and target[name] != digest:
            raise ValueError("Conflicting source snapshot hashes")
        target[name] = digest


@dataclass(frozen=True)
class DescriptiveInputs:
    frames: dict
    ids: pd.DataFrame
    maps: dict
    soft: np.ndarray
    prior: np.ndarray
    snapshot: dict
    source_records: list
    evidence_provenance: dict
    source_directories: tuple


def load_descriptive_inputs(full_dirs, b1_dirs, ablation_dirs, attention_dir):
    """All 18 strict source contracts and archived evidence before any scoring."""
    paths = [Path(p).resolve() for p in (*full_dirs, *b1_dirs, *ablation_dirs)]
    if (len(full_dirs), len(b1_dirs), len(ablation_dirs)) != (3, 3, 12) or len(set(paths)) != 18:
        raise ValueError("Exactly 18 distinct source directories (3 full, 3 B1, 12 OFF) required")
    inputs = load_saved_prediction_inputs(full_dirs, b1_dirs)
    snapshot = dict(inputs.source_snapshot)
    frames, records, ablations = {}, [], {}
    for model, sources in (("ConsenDxFormer", inputs.phase4), ("B1", inputs.b1)):
        for seed in SEEDS:
            source = sources[seed]
            frames[(model, seed)] = source.predictions
            records.append({"model": model, "seed": seed, **reference_record(source),
                            "source_snapshot": relative_source_snapshot(source)})
    for path in ablation_dirs:
        source = load_ablation_source(path)
        variant, seed = source.identity["variant"], source.identity["seed"]
        key = (variant, seed)
        if variant not in VARIANTS or type(seed) is not int or seed not in SEEDS or key in ablations:
            raise ValueError("Ablation variant/seed set is not exact")
        if source.identity["full_model_reference"] != reference_record(inputs.phase4[seed]):
            raise ValueError("Ablation binds a different complete-model source")
        if identity_rows(source.supervision) != identity_rows(inputs.supervision):
            raise ValueError("Ablation supervision identities differ from full sources")
        ablations[key] = source
        frames[(f"{variant}_off", seed)] = source.predictions["test"]
        _merge_snapshot(snapshot, source.source_snapshot)
        records.append({"model": f"{variant}_off", "seed": seed,
                        "source_dir": str(source.source_dir), "identity": source.identity,
                        "best_checkpoint_sha256": sha256_file(source.source_dir / "checkpoints/best.pt"),
                        "source_snapshot": dict(source.source_snapshot)})
    if set(ablations) != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError("All four OFF variants and all three seeds required")
    frames = canonical_predictions(frames)
    reference = frames[(MODELS[0], SEEDS[0])]
    if len(reference) != 183 or reference.patient_id.nunique() != 99:
        raise ValueError("Frozen primary test cohort must contain 183 nodules / 99 patients")
    expected = pd.DataFrame(identity_rows(inputs.supervision.loc[inputs.supervision.split == "test"]))
    if identities(expected.sort_values(list(IDENTITY))) != identities(reference.sort_values(list(IDENTITY))):
        raise ValueError("Prediction membership differs from frozen test supervision")

    attention_dir = Path(attention_dir).resolve()
    archives, provenance, fixed_snapshot = load_fixed_evidence(
        attention_dir, {s: ablations[(VARIANTS[0], s)] for s in SEEDS})
    _merge_snapshot(snapshot, fixed_snapshot)
    required = {"evaluation.json", "provenance/provenance.json", "location_prior.npz", "test_inputs.npz",
                "provenance/supplementary_protocol.json", *(f"seed_{s}_model_arrays.npz" for s in SEEDS)}
    for name in required:
        _, member = _safe_source_member(attention_dir, name)
        if str(member) not in fixed_snapshot:
            raise ValueError("Required attention evidence member is not manifest-bound")
    if (sha256_json(provenance["input_file_snapshot"]) != provenance["input_file_snapshot_sha256"]
            or _json(attention_dir / "evaluation.json").get("comparison_interpretation_permitted") is not True):
        raise ValueError("Attention snapshot or comparison permission differs")
    archived_sources = provenance["source_experiments"]
    if len(archived_sources) != 6:
        raise ValueError("Archived attention needs exactly six bound model sources")
    seen = set()
    for record in archived_sources:
        model, seed = record["model"], record["seed"]
        key = (model, seed)
        if model not in MODELS[:2] or type(seed) is not int or seed not in SEEDS or key in seen:
            raise ValueError("Archived attention model/seed identity mismatch")
        seen.add(key)
        source = (inputs.b1 if model == "B1" else inputs.phase4)[seed]
        if record["descriptor"] != dict(source.descriptor) or record["source_snapshot"] != relative_source_snapshot(source):
            raise ValueError("Archived attention descriptor/checkpoint/prediction source changed")
    saved = archives["test_inputs.npz"]
    ids = pd.DataFrame({field: saved[field] for field in IDENTITY})
    if identities(ids.sort_values(list(IDENTITY))) != identities(reference.sort_values(list(IDENTITY))):
        raise ValueError("Archived native map identities differ from the exact test cohort")
    soft, prior = saved["soft_masks_native_grid"], archives["location_prior.npz"]["prior"]
    if soft.shape != (183, 8, 8) or prior.shape != (8, 8):
        raise ValueError("Frozen native mask/prior grid must be 8x8")
    input_masks = saved["soft_masks_input"]
    if (input_masks.shape != (183, 1, 64, 64) or not np.isfinite(input_masks).all()
            or (input_masks < 0).any() or (input_masks > 1).any()):
        raise ValueError("Archived input soft masks differ from the frozen input contract")
    # Validate the archived native-mask key, not an alignment metric or a new target.
    if not np.array_equal(soft, np.stack([_area_resize(m[0], (8, 8)) for m in input_masks])):
        raise ValueError("Native soft masks must be the unnormalized archived area averages")
    maps = {}
    for model, prefix in (("ConsenDxFormer", "cdx"), ("B1", "b1")):
        for seed in SEEDS:
            arrays = archives[f"seed_{seed}_model_arrays.npz"]
            value = arrays[f"{prefix}_attention"]
            if value.shape != (183, 1, 8, 8):
                raise ValueError("Frozen native model map shape must be [183,1,8,8]")
            validate_native_inputs(ids, value, soft)
            # Compare saved arrays to saved predictions only: never reconstruct a model.
            replay = _saved_replay(frames[(model, seed)], ids, arrays[f"{prefix}_logits"], arrays[f"{prefix}_probabilities"])
            if not replay["consistent"]:
                raise ValueError("Archived classification differs from the bound saved predictions")
            maps[(model, seed)] = value
    validate_native_inputs(ids, np.broadcast_to(prior, (183, 1, 8, 8)), soft)
    flagged = np.flatnonzero(ids.nodule_id.eq(FLAGGED_NODULE))
    if len(flagged) != 1 or (soft[flagged[0]] >= 0.5).any():
        raise ValueError("Known flagged empty native target requires review; never repair it")
    assert_source_snapshot_unchanged(snapshot)
    return DescriptiveInputs(frames, ids, maps, soft, prior, snapshot,
                             sorted(records, key=lambda r: (r["model"], r["seed"])), provenance,
                             tuple([*paths, attention_dir]))


def _output_path(root, evaluation_id, inputs):
    root = Path(root).absolute()
    guard_new_namespace(root)
    if not root.resolve().is_relative_to(NAMESPACE.resolve()):
        raise ValueError("Subset outputs must remain in the distinct Phase 5 namespace")
    output = root / evaluation_id
    guard_new_namespace(output)
    protected = [*inputs.source_directories, PROJECT_ROOT / "data", PROJECT_ROOT / "external",
                 PROJECT_ROOT / "artifacts/phase0", PROJECT_ROOT / "artifacts/phase2"]
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
        raise ValueError("Output overlaps a protected source")
    if output.exists():
        raise FileExistsError("Existing completed or partial subset output is never overwritten")
    return output


def _write_csv(path, frame):
    atomic_write_csv(path, frame)
    loaded = pd.read_csv(path, dtype={c: str for c in frame if frame[c].dtype == object},
                         keep_default_na=False, float_precision="round_trip")
    for column in frame:
        expected = frame[column]
        if pd.api.types.is_numeric_dtype(expected) and not expected.isna().any():
            if not np.array_equal(loaded[column].to_numpy(), expected.to_numpy()):
                raise ValueError("Numeric CSV round-trip differs")
        elif loaded[column].astype(str).tolist() != ["" if pd.isna(v) else str(v) for v in expected.tolist()]:
            raise ValueError(f"Text/nullable CSV round-trip differs: {column}")


def validate_descriptive_completion(directory):
    """Read-only manifest/completion verification; never certify the full supplement."""
    directory = Path(directory).resolve()
    manifest, marker = _json(directory / "artifact_manifest.json"), _json(directory / "COMPLETED.json")
    if (marker.get("schema") != "phase5_descriptive_completion_v1"
            or manifest.get("schema") != "phase5_descriptive_artifacts_v1"
            or marker.get("artifact_manifest_sha256") != sha256_file(directory / "artifact_manifest.json")
            or marker.get("subset_completed") is not True
            or marker.get("evaluation_id") != manifest.get("evaluation_id")):
        raise ValueError("Invalid descriptive subset completion binding")
    files = manifest.get("files")
    expected_tables = {"pointing_per_nodule", "pointing_per_seed", "pointing_aggregate",
                       "pointing_paired_descriptive", "pointing_paired_per_nodule", "flagged_case",
                       "accuracy_intervals", "accuracy_per_nodule", "hypothesis_register"}
    required = {*(f"tables/{n}.csv" for n in expected_tables), "accuracy_bootstrap.npz", "protocol.json",
                "evaluation.json", "provenance/provenance.json", "summary.md"}
    if not isinstance(files, dict) or set(files) != required or sha256_json(files) != manifest.get("bundle_sha256"):
        raise ValueError("Incomplete subset manifest")
    for name, digest in files.items():
        _, member = _safe_source_member(directory, name)
        if sha256_file(member) != digest:
            raise ValueError("Subset output hash mismatch")
    evaluation = _json(directory / "evaluation.json")
    provenance = _json(directory / "provenance/provenance.json")
    policy = load_protocol(directory / "protocol.json")
    for record in (manifest, marker, evaluation, provenance):
        expected = {"scope": SCOPE, "full_supplemental_completion": False, "model_forwards": 0,
                    "training_performed": False, "formal_tests_computed": False,
                    "unresolved_hypotheses": unresolved_hypotheses(), "excluded": policy["excluded"]}
        if sha256_json({k: record.get(k) for k in expected}) != sha256_json(expected):
            raise ValueError("Completion may certify only the approved descriptive subset")
    if evaluation["identity"] != provenance["identity"] or evaluation["identity"]["protocol"] != policy:
        raise ValueError("Subset identity/protocol disagreement")
    expected_id = f"phase5-descriptive-{sha256_json(evaluation['identity'])[:16]}"
    if evaluation.get("evaluation_id") != expected_id or marker["evaluation_id"] != expected_id:
        raise ValueError("Subset identity hash mismatch")
    if provenance["source_snapshot_sha256"] != sha256_json(provenance["source_snapshot"]):
        raise ValueError("Subset source snapshot binding differs")
    archive = provenance["bootstrap_archive"]
    if archive["file"] != "accuracy_bootstrap.npz" or archive["sha256"] != files["accuracy_bootstrap.npz"]:
        raise ValueError("Subset bootstrap archive binding differs")
    with np.load(directory / "accuracy_bootstrap.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(archive["arrays"]) or any(array_record(saved[k]) != v for k, v in archive["arrays"].items()):
            raise ValueError("Subset bootstrap array hashes differ")
        if saved["draw_indices"].shape != (10000, 99) or saved["replicate_estimates"].shape != (10000, 33):
            raise ValueError("Subset bootstrap dimensions differ")
    return marker


def run_descriptive(*, full_source_dirs, b1_source_dirs, ablation_source_dirs, attention_source,
                    authorize_analysis=False, output_root=None):
    if authorize_analysis is not True:
        raise ValueError("Explicit --authorize-analysis required; saved-artifact scoring is opt-in")
    git = validate_evaluator_repository(PROJECT_ROOT)
    protocol = load_protocol(PROTOCOL_PATH)
    if hashlib.sha256(PROPOSAL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PROPOSAL_SHA256:
        raise ValueError("Reviewed proposal content changed")
    inputs = load_descriptive_inputs(full_source_dirs, b1_source_dirs, ablation_source_dirs, attention_source)
    snapshot = dict(inputs.snapshot)
    _merge_snapshot(snapshot, snapshot_source_files([PROTOCOL_PATH, PROPOSAL_PATH]))
    identity = {"protocol": protocol, "scope": SCOPE, "implementation_commit": git["commit"],
                "sources": inputs.source_records, "source_snapshot_sha256": sha256_json(snapshot),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    evaluation_id = f"phase5-descriptive-{sha256_json(identity)[:16]}"
    output = _output_path(NAMESPACE if output_root is None else output_root, evaluation_id, inputs)
    assert_source_snapshot_unchanged(snapshot)
    output.mkdir(parents=True, exist_ok=False)
    tables = pointing_tables(inputs.ids, inputs.maps, inputs.soft, inputs.prior)
    intervals, arrays, predictions = accuracy_intervals(inputs.frames)
    if len(tables["pointing_per_nodule.csv"]) != 1281 or len(tables["flagged_case.csv"]) != 7 or len(intervals) != 33:
        raise ValueError("Subset output accounting differs")
    tables["accuracy_intervals.csv"] = intervals
    tables["accuracy_per_nodule.csv"] = predictions
    tables["hypothesis_register.csv"] = pd.DataFrame(unresolved_hypotheses())
    for name, table in tables.items():
        _write_csv(output / "tables" / name, table)
    archive = write_map_archive(output / "accuracy_bootstrap.npz", arrays)
    status = {"scope": SCOPE, "full_supplemental_completion": False, "model_forwards": 0,
              "training_performed": False, "formal_tests_computed": False,
              "unresolved_hypotheses": unresolved_hypotheses(), "excluded": protocol["excluded"]}
    atomic_write_json(output / "protocol.json", protocol)
    atomic_write_json(output / "evaluation.json", {"evaluation_id": evaluation_id, "identity": identity,
                      **status, "warning": WARNING, "n_nodules": 183, "n_patients": 99,
                      "pointing_records": 1281, "individual_intervals": 18, "paired_intervals": 15})
    atomic_write_json(output / "provenance/provenance.json", {
        "identity": identity, "source_snapshot": snapshot, "source_snapshot_sha256": sha256_json(snapshot),
        "evidence_provenance": inputs.evidence_provenance, "bootstrap_archive": archive,
        "patient_draw_sha256": array_record(arrays["draw_indices"])["array_sha256"],
        "prior_recomputed": False, "maps_regenerated": False, "new_inference_replay_performed": False,
        "archived_predictions_compared_to_saved_predictions": True, **status})
    atomic_write_text(output / "summary.md", f"# {evaluation_id}\n\n{WARNING}\n\n"
        "Completion certifies only pointing-game and 33 descriptive accuracy intervals. "
        "The complete supplement, Stage A, McNemar and multiplicity are not completed.\n"
        "All 183 nodules/99 patients remain; empty targets are explicit misses. "
        "See per-seed tables, empty-target counts, flagged case and common draw ledger.\n")
    assert_source_snapshot_unchanged(snapshot)
    if validate_evaluator_repository(PROJECT_ROOT) != git:
        raise ValueError("Evaluator repository changed during subset scoring")
    files = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = atomic_write_json(output / "artifact_manifest.json", {
        "schema": "phase5_descriptive_artifacts_v1", "evaluation_id": evaluation_id, **status,
        "files": files, "bundle_sha256": sha256_json(files)})
    for name, digest in files.items():
        _, member = _safe_source_member(output, name)
        if sha256_file(member) != digest:
            raise ValueError("Subset output checksum changed before completion")
    assert_source_snapshot_unchanged(snapshot)
    marker = atomic_write_json(output / "COMPLETED.json", {
        "schema": "phase5_descriptive_completion_v1", "evaluation_id": evaluation_id,
        "subset_completed": True, **status, "artifact_manifest_sha256": sha256_file(manifest)})
    try:
        validate_descriptive_completion(output)
        assert_source_snapshot_unchanged(snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Repository changed before subset completion")
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    return output


def descriptive_cli(argv=None):
    parser = argparse.ArgumentParser(description="Opt-in Phase 5 pointing/accuracy only; zero model forwards")
    parser.add_argument("--full-source", action="append", required=True)
    parser.add_argument("--b1-source", action="append", required=True)
    parser.add_argument("--ablation-source", action="append", required=True)
    parser.add_argument("--attention-source", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--authorize-analysis", action="store_true")
    args = parser.parse_args(argv)
    output = run_descriptive(full_source_dirs=args.full_source, b1_source_dirs=args.b1_source,
                             ablation_source_dirs=args.ablation_source, attention_source=args.attention_source,
                             authorize_analysis=args.authorize_analysis, output_root=args.output_root)
    print(f"OUTPUT_DIR={output}")
    print(f"COMPLETION_MARKER={output / 'COMPLETED.json'}")
    return 0
