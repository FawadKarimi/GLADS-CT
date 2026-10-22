"""Separately authorized insertion/deletion runner; never executes another analysis."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_json, atomic_write_text
from .phase4_ablation_runner import guard_new_namespace
from .phase4_attention_evidence import array_record, write_map_archive
from .phase4_evaluation_runner import (PROJECT_ROOT, _safe_source_member, snapshot_source_files,
    assert_source_snapshot_unchanged, validate_evaluator_repository)
from .phase5_descriptive import PROPOSAL_SHA256, unresolved_hypotheses
from .phase5_perturbation import (SCOPE, KEYS, WARNING, CallLedger, perturbation_protocol, execute_curves)
from .phase5_perturbation_sources import (read_json, merge_snapshot, load_perturbation_inputs, frozen_predictors)
from .phase5_perturbation_exports import write_result, validate_result_exports
from .phase5_timing_runtime import architecture_contract, runtime_environment
from .phase5_timing_runner import (BUNDLE_FILES as TIMING_FILES, validate_timing_bundle, copy_timing_evidence)

PROTOCOL_PATH = PROJECT_ROOT / "configs/phase5_insertion_deletion_v1.json"
PROPOSAL_PATH = PROJECT_ROOT / "docs/phase5_supplemental_evaluation_proposal.md"
NAMESPACE = PROJECT_ROOT / "artifacts/phase5/insertion_deletion"
TABLES = {"units.csv", "perturbation_curves.csv", "perturbation_auc_per_nodule.csv",
          "perturbation_paired_per_nodule.csv", "perturbation_per_seed.csv",
          "perturbation_paired_descriptive.csv", "perturbation_mean_curves.csv", "hypothesis_register.csv"}
PAYLOADS = {*(f"tables/{name}" for name in TABLES), "protocol.json", "evaluation.json", "summary.md",
            "provenance/provenance.json", "provenance/reviewed_proposal.md", "orders.npz", "curves.npz",
            "baseline_cache.npz", "replay_cache.npz", "replay_barrier.json", "replay_checks.json", "call_ledger.json"}
LEGACY_PAYLOADS = set(PAYLOADS)
PAYLOADS |= {f"provenance/timing/{name}" for name in TIMING_FILES}


def subset_status():
    return {"scope": SCOPE, "full_supplemental_completion": False, "stage_a_complete": False,
            "formal_tests_computed": False, "training_performed": False,
            "hypotheses": unresolved_hypotheses(), "analysis_status": "exploratory_post_hoc",
            "excluded": perturbation_protocol()["excluded"]}


def load_protocol():
    protocol = read_json(PROTOCOL_PATH)
    if sha256_json(protocol) != sha256_json(perturbation_protocol()):
        raise ValueError("Insertion/deletion protocol differs from reviewed method")
    document = PROPOSAL_PATH.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(document).hexdigest() != PROPOSAL_SHA256:
        raise ValueError("Reviewed proposal content differs")
    return protocol, document.decode("utf-8")


def output_directory(root, evaluation_id, inputs):
    root = Path(root).absolute()
    guard_new_namespace(root)
    if not root.resolve().is_relative_to(NAMESPACE.resolve()):
        raise ValueError("Outputs must stay inside the distinct insertion/deletion namespace")
    output = root / evaluation_id
    guard_new_namespace(output)
    protected = [*inputs.source_directories, PROJECT_ROOT / "data", PROJECT_ROOT / "external",
                 PROJECT_ROOT / "artifacts/phase0", PROJECT_ROOT / "artifacts/phase2"]
    # Snapshots can include external data paths as well as project-local inputs.
    protected.extend(Path(p) for p in inputs.snapshot)
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
        raise ValueError("Output overlaps protected source/input files")
    if output.exists():
        raise FileExistsError("Never overwrite/resume an existing completed or partial output")
    return output


def _ledger_from_export(directory):
    row = read_json(directory / "call_ledger.json")
    ledger = CallLedger(attempted=row["attempted"], accepted=row["accepted"],
                        replays=read_json(directory / "replay_checks.json"),
                        barrier_passed=row["global_replay_barrier_passed"],
                        last_attempt=row["last_attempt"], failure=row["failure"])
    if row != ledger.record():
        raise ValueError("Call ledger totals/policy differ")
    return ledger


def validate_completed_perturbation(directory):
    """Inference-free read-back; no live-source access or old metric suite calls."""
    directory = Path(directory).resolve()
    if (directory / "FAILED.json").exists():
        raise ValueError("A recorded failed attempt cannot certify subset completion")
    manifest, marker = read_json(directory / "artifact_manifest.json"), read_json(directory / "COMPLETED.json")
    # Historical v1 validation remains read-only; only v2 can be newly executed.
    version = 2 if marker.get("schema") == "phase5_insertion_deletion_completion_v2" else 1
    if (manifest.get("schema") != f"phase5_insertion_deletion_artifacts_v{version}"
            or marker.get("schema") != f"phase5_insertion_deletion_completion_v{version}"
            or marker.get("subset_completed") is not True
            or marker.get("artifact_manifest_sha256") != sha256_file(directory / "artifact_manifest.json")
            or marker.get("evaluation_id") != manifest.get("evaluation_id")
            or set(manifest.get("files", {})) != (PAYLOADS if version == 2 else LEGACY_PAYLOADS)
            or manifest.get("bundle_sha256") != sha256_json(manifest["files"])):
        raise ValueError("Invalid/incomplete insertion-deletion subset completion")
    for name, digest in manifest["files"].items():
        _, member = _safe_source_member(directory, name)
        if sha256_file(member) != digest:
            raise ValueError("Insertion/deletion output hash differs")
    protocol, evaluation, provenance = (read_json(directory / name) for name in
                                      ("protocol.json", "evaluation.json", "provenance/provenance.json"))
    if sha256_json(protocol) != sha256_json(perturbation_protocol()):
        raise ValueError("Exported perturbation policy differs")
    document = (directory / "provenance/reviewed_proposal.md").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(document).hexdigest() != PROPOSAL_SHA256:
        raise ValueError("Exported reviewed proposal hash differs")
    expected = subset_status()
    for record in (manifest, marker, evaluation, provenance):
        if sha256_json({k: record.get(k) for k in expected}) != sha256_json(expected):
            raise ValueError("Completion must certify insertion_deletion_only; formal family remains unresolved")
    identity = evaluation["identity"]
    if version == 2:
        binding = identity["timing_evidence"]
        timing = validate_timing_bundle(directory / "provenance/timing", binding["manifest_sha256"],
            expected_architecture=identity["timing_architecture"], expected_runtime=identity["environment"])
        if ({k: v for k, v in binding.items() if k != "source_directory"} != timing["binding"]
                or provenance.get("timing_evidence") != binding):
            raise ValueError("Research run timing-evidence binding differs")
        original_timing = provenance["timing_source_snapshot"]
        expected_members = {binding["source_directory"].rstrip("/") + "/" + name:
                            sha256_file(directory / "provenance/timing" / name) for name in TIMING_FILES}
        if ({p.replace("\\", "/"): v for p, v in original_timing.items()} != expected_members
                or any(provenance["source_snapshot"].get(p) != v for p, v in original_timing.items())):
            raise ValueError("Timing source snapshot is not bound to embedded evidence")
        execution = provenance["research_execution"]
        duration = execution["curve_execution_wall_seconds"]
        if (evaluation.get("research_execution") != execution or type(duration) not in (float, int)
                or not np.isfinite(duration) or duration < 0 or execution["end_to_end_runtime_measured"] is not False
                or execution["measurement_scope"] != "execute_curves including replays, construction, validation and replay-cache barrier I/O"
                or provenance.get("historical_protocol_timing_field", "missing") != protocol["synthetic_timing_evidence"]
                or execution["synthetic_adapter_estimate_seconds"] != timing["binding"]["estimated_adapter_seconds"]):
            raise ValueError("Research execution/estimate distinction differs")
    if (identity != provenance["identity"] or identity["protocol"] != protocol
            or identity.get("execution_commit") != identity["implementation_commit"]
            or identity.get("proposal_published_commit") != "75b5d7ce895338e4b21747c05b039850eae2b936"
            or evaluation["evaluation_id"] != f"phase5-perturbation-{sha256_json(identity)[:16]}"
            or marker["evaluation_id"] != evaluation["evaluation_id"]
            or sha256_json(provenance["source_snapshot"]) != identity["source_snapshot_sha256"]):
        raise ValueError("Identity/source snapshot binding differs")
    ledger = _ledger_from_export(directory)
    expected_counts = {"n_nodules": 183, "n_patients": 99, "per_seed_summaries": 36, "within_model_paired_summaries": 24}
    if any(evaluation.get(k) != v for k, v in expected_counts.items()):
        raise ValueError("Evaluation summary counts differ from the complete frozen subset")
    if evaluation["calls"] != ledger.record() or evaluation["successful_run_budget"] != protocol["successful_run_budget"]:
        raise ValueError("Evaluation call accounting differs")
    records = provenance["archives"]
    for record in records:
        if manifest["files"].get(record["file"]) != record["sha256"]:
            raise ValueError("Archive not manifest-bound")
    result = validate_result_exports(directory, records, ledger)
    barrier = read_json(directory / "replay_barrier.json")
    binding = {"source_binding": sha256_json(identity),
               "orders_sha256": sha256_json({k: array_record(v) for k, v in result.orders.items()}),
               "units_sha256": sha256_json(result.units.to_dict("records")),
               "arrays": {k: array_record(v) for k, v in result.replay.items()}}
    if barrier != {**binding, "cache_binding_sha256": sha256_json(binding),
                   "replays_passed": 1098, "baseline_calls_at_barrier": 0, "interior_calls_at_barrier": 0}:
        raise ValueError("Global replay cache/source binding differs")
    return marker


def run_insertion_deletion(*, full_source_dirs, b1_source_dirs, attention_source, attention_manifest_sha256,
                          authorize_execution=False, device="cpu", output_root=None,
                          timing_bundle=None, timing_manifest_sha256=None):
    if authorize_execution is not True:
        raise ValueError("Explicit --authorize-execution required for insertion/deletion")
    if timing_bundle is None or timing_manifest_sha256 is None:
        raise ValueError("Explicit timing bundle path and approved manifest SHA-256 required before research access")
    git = validate_evaluator_repository(PROJECT_ROOT)
    protocol, proposal = load_protocol()
    environment = runtime_environment(device)
    timing_architecture, code_snapshot = architecture_contract()
    timing = validate_timing_bundle(timing_bundle, timing_manifest_sha256,
        expected_architecture=timing_architecture, expected_runtime=environment)
    timing_binding = {**timing["binding"], "source_directory": Path(timing_bundle).resolve().as_posix()}
    # This entire timing compatibility gate precedes research data/checkpoint loading.
    inputs = load_perturbation_inputs(full_source_dirs, b1_source_dirs, attention_source, attention_manifest_sha256)
    snapshot = dict(inputs.snapshot)
    merge_snapshot(snapshot, timing["snapshot"])
    merge_snapshot(snapshot, code_snapshot)
    merge_snapshot(snapshot, snapshot_source_files([PROTOCOL_PATH, PROPOSAL_PATH]))
    ledger, output, created = CallLedger(), None, False
    status = subset_status()
    try:
        # Runtime preparation performs only frozen loading/readiness/input checks, no forwards.
        with frozen_predictors(inputs, device=device, cache_dir=NAMESPACE / ".disabled_pycache") as (predictors, runtime):
            merge_snapshot(snapshot, runtime["live_input_snapshot"])
            identity = {"scope": SCOPE, "protocol": protocol, "implementation_commit": git["commit"],
                        "execution_commit": git["commit"],
                        "proposal_published_commit": "75b5d7ce895338e4b21747c05b039850eae2b936",
                        "sources": [inputs.source_records[k] for k in KEYS],
                        "source_snapshot_sha256": sha256_json(snapshot),
                        "attention_manifest_sha256": attention_manifest_sha256,
                        "environment": environment, "runtime": runtime,
                        "timing_evidence": timing_binding, "timing_architecture": timing_architecture}
            evaluation_id = f"phase5-perturbation-{sha256_json(identity)[:16]}"
            output = output_directory(NAMESPACE if output_root is None else output_root, evaluation_id, inputs)
            assert_source_snapshot_unchanged(snapshot)
            output.mkdir(parents=True, exist_ok=False)
            created = True
            atomic_write_json(output / "protocol.json", protocol)
            atomic_write_text(output / "provenance/reviewed_proposal.md", proposal)
            copy_timing_evidence(timing_bundle, output / "provenance/timing", timing_manifest_sha256)
            archive_records = []

            def barrier(cache, binding, log):
                if log.attempted != {"unperturbed": 1098, "baseline": 0, "interior": 0} or len(log.replays) != 1098:
                    raise ValueError("All 1098 unperturbed replays must pass globally")
                assert_source_snapshot_unchanged(snapshot)
                if validate_evaluator_repository(PROJECT_ROOT) != git:
                    raise ValueError("Repository changed before perturbation barrier")
                archive_records.append(write_map_archive(output / "replay_cache.npz", cache))
                atomic_write_json(output / "replay_barrier.json", {**binding, "replays_passed": 1098,
                    "baseline_calls_at_barrier": 0, "interior_calls_at_barrier": 0})
                # All cache bytes were read back before granting the first nonoriginal call.
                assert_source_snapshot_unchanged(snapshot)

            research_start = time.perf_counter()
            result = execute_curves(inputs.ids, inputs.images, inputs.maps, inputs.prior, inputs.predictions,
                predictors, source_binding=sha256_json(identity), source_records=inputs.source_records,
                ledger=ledger, on_barrier=barrier)
            research_execution = {"curve_execution_wall_seconds": time.perf_counter()-research_start,
                "measurement_scope": "execute_curves including replays, construction, validation and replay-cache barrier I/O",
                "end_to_end_runtime_measured": False,
                "synthetic_adapter_estimate_seconds": timing["binding"]["estimated_adapter_seconds"]}
            assert_source_snapshot_unchanged(snapshot)
            archive_records.extend(write_result(output, result))
            atomic_write_json(output / "replay_checks.json", ledger.replays)
            atomic_write_json(output / "call_ledger.json", ledger.record())
            atomic_write_json(output / "evaluation.json", {"evaluation_id": evaluation_id, "identity": identity,
                **status, "calls": ledger.record(), "successful_run_budget": protocol["successful_run_budget"],
                "research_execution": research_execution,
                "warning": WARNING, "n_nodules": 183, "n_patients": 99,
                "per_seed_summaries": 36, "within_model_paired_summaries": 24})
            atomic_write_json(output / "provenance/provenance.json", {"identity": identity, **status,
                "source_snapshot": snapshot, "archives": archive_records,
                "evidence_provenance": inputs.evidence_provenance, "prior_recomputed": False,
                "rankings_recomputed_after_perturbation": False, "masks_in_forward_or_ranking": False,
                "timing_evidence": timing_binding, "timing_source_snapshot": timing["snapshot"],
                "research_execution": research_execution,
                "historical_protocol_timing_field": protocol["synthetic_timing_evidence"]})
            atomic_write_text(output / "summary.md", f"# {evaluation_id}\n\n{WARNING}\n\n"
                "All 183 nodules / 99 patients and all three seeds retained. See 36 per-seed "
                "summaries, 24 within-model paired summaries and all 65 positions of 6,588 curves. "
                "417,240 batch-one forwards; no retries or additional deduplication. "
                f"Approved synthetic timing evidence: {timing_binding['timing_id']}. "
                f"Synthetic adapter-only estimate: {timing_binding['estimated_adapter_seconds']:.9g} seconds. "
                f"Observed curve-execution wall time: {research_execution['curve_execution_wall_seconds']:.9g} seconds "
                "(not end-to-end runtime). Synthetic measurements/estimates are distinct from research execution; "
                "the frozen protocol's historical null timing field is unchanged.\n")
            validate_result_exports(output, archive_records, ledger)
        # Includes pinned-source and in-memory frozen-weight checks on context exit.
        assert_source_snapshot_unchanged(snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Repository changed during insertion/deletion execution")
        files = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
        if set(files) != PAYLOADS:
            raise ValueError("Incomplete or unexpected insertion/deletion exports")
        manifest = atomic_write_json(output / "artifact_manifest.json", {
            "schema": "phase5_insertion_deletion_artifacts_v2", "evaluation_id": evaluation_id,
            **status, "files": files, "bundle_sha256": sha256_json(files)})
        assert_source_snapshot_unchanged(snapshot)
        atomic_write_json(output / "COMPLETED.json", {
            "schema": "phase5_insertion_deletion_completion_v2", "evaluation_id": evaluation_id,
            "subset_completed": True, **status, "artifact_manifest_sha256": sha256_file(manifest)})
        validate_completed_perturbation(output)
        assert_source_snapshot_unchanged(snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Repository changed before insertion/deletion completion")
        return output
    except BaseException as exc:
        # Catch Python interruptions too, but touch only the directory this call
        # created. Independent best-effort actions must never mask the original.
        if created and output is not None:
            try:
                (output / "COMPLETED.json").unlink(missing_ok=True)
            except BaseException:
                pass  # Unwritable filesystems/repeated interrupts cannot guarantee cleanup.
            try:
                if output.is_dir() and not (output / "FAILED.json").exists():
                    atomic_write_json(output / "FAILED.json", {**status, "subset_completed": False,
                        "error_type": type(exc).__name__, "message": str(exc), "calls": ledger.record(),
                        "replay_checks": ledger.replays, "automatic_resume_permitted": False})
            except BaseException:
                pass  # No retry, overwrite or replacement of the original exception.
        raise


def insertion_deletion_cli(argv=None):
    parser = argparse.ArgumentParser(description="Separately authorized insertion_deletion_only; frozen inputs/models")
    parser.add_argument("--full-source", action="append", required=True)
    parser.add_argument("--b1-source", action="append", required=True)
    parser.add_argument("--attention-source", required=True)
    parser.add_argument("--attention-manifest-sha256", required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--timing-bundle", type=Path)
    parser.add_argument("--timing-manifest-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output = run_insertion_deletion(full_source_dirs=args.full_source, b1_source_dirs=args.b1_source,
        attention_source=args.attention_source, attention_manifest_sha256=args.attention_manifest_sha256,
        authorize_execution=args.authorize_execution, device=args.device, output_root=args.output_root,
        timing_bundle=args.timing_bundle, timing_manifest_sha256=args.timing_manifest_sha256)
    print(f"OUTPUT_DIR={output}")
    print(f"COMPLETION_MARKER={output / 'COMPLETED.json'}")
    return 0
