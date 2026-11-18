"""Opt-in synthetic timing and independently validated timing-evidence bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_json, atomic_write_text, atomic_write_bytes
from .phase4_ablation_runner import guard_new_namespace
from .phase4_attention_evidence import array_record, write_map_archive
from .phase4_evaluation_runner import (PROJECT_ROOT, _safe_source_member, snapshot_source_files,
    assert_source_snapshot_unchanged, validate_evaluator_repository)
from .phase5_descriptive import PROPOSAL_SHA256, unresolved_hypotheses
from .phase5_device import bootstrap_record, validate_device_evidence
from .phase5_perturbation import KEYS, perturbation_protocol, validate_probabilities
from .phase5_timing import (SCOPE, TOTAL, PerfClock, TimingLedger, timing_specification, make_plan,
                            measure_schedule, elapsed, extrapolate, validate_loop_envelope)
from .phase5_timing_runtime import architecture_contract, runtime_environment, synthetic_predictors, memory_snapshot, synchronize

NAMESPACE = PROJECT_ROOT / "artifacts/phase5/synthetic_timing"
SPEC_PATH = PROJECT_ROOT / "configs/phase5_synthetic_timing_v1.json"
PAYLOADS = {"specification.json", "plan.json", "identity.json", "bootstrap.json", "models.json", "measurements.json",
            "synthetic_inputs.npz", "synthetic_outputs.npz", "source_snapshot.json", "summary.md"}
BUNDLE_FILES = PAYLOADS | {"artifact_manifest.json", "COMPLETED.json"}
WARNING = ("Synthetic initialized weights/inputs only. Measurements and adapter-only extrapolations are not actual research runtime. "
           "Contention, loading/startup, input construction, validation, disk I/O, six-model residency and synthetic weights limit estimates. "
           "Block minima/maxima are not confidence intervals. Completion does not authorize research execution, close any formal hypothesis "
           "or complete the full supplement/Stage A.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status():
    return {"scope": SCOPE, "research_execution_authorized": False, "research_forwards": 0,
            "formal_tests_computed": False, "full_supplemental_completion": False, "stage_a_complete": False,
            "hypotheses": unresolved_hypotheses()}


def load_specification():
    spec = read_json(SPEC_PATH)
    if sha256_json(spec) != sha256_json(timing_specification()):
        raise ValueError("Synthetic timing specification differs")
    if read_json(PROJECT_ROOT / "configs/phase5_insertion_deletion_v1.json") != perturbation_protocol():
        raise ValueError("Frozen scientific protocol changed")
    proposal = (PROJECT_ROOT / "docs/phase5_supplemental_evaluation_proposal.md").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(proposal).hexdigest() != PROPOSAL_SHA256:
        raise ValueError("Frozen proposal changed")
    return spec


def validate_memory(record, device):
    fields = ["host_peak_rss_bytes"]
    if record["host_current_rss_bytes"] is not None:
        fields.append("host_current_rss_bytes")
    if device == "cuda":
        fields.extend(["cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes",
                       "cuda_peak_reserved_bytes", "cuda_free_bytes", "cuda_total_bytes"])
    if (any(type(record.get(k)) is not int or record[k] < 0 for k in fields)
            or not isinstance(record.get("host_measurement"), str) or not record["host_measurement"]
            or (record["host_current_rss_bytes"] is not None and record["host_current_rss_bytes"] > record["host_peak_rss_bytes"])):
        raise ValueError("Invalid memory diagnostics")
    if device == "cuda" and not (record["cuda_allocated_bytes"] <= record["cuda_reserved_bytes"]
            <= record["cuda_peak_reserved_bytes"] <= record["cuda_total_bytes"]
            and record["cuda_allocated_bytes"] <= record["cuda_peak_allocated_bytes"] <= record["cuda_peak_reserved_bytes"]
            and record["cuda_free_bytes"] <= record["cuda_total_bytes"]):
        raise ValueError("Inconsistent CUDA memory diagnostics")


def validate_timing_bundle(directory, expected_sha256, *, expected_architecture=None, expected_runtime=None):
    """No model/data loading. Suitable for offline validation of an embedded copy."""
    root = Path(directory).resolve()
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("An explicit timing manifest SHA-256 is required")
    if (root / "FAILED.json").exists():
        raise ValueError("A failed timing attempt is not evidence")
    manifest_path = root / "artifact_manifest.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("Timing manifest hash differs")
    manifest, marker = read_json(manifest_path), read_json(root / "COMPLETED.json")
    if (manifest.get("schema") != "phase5_synthetic_timing_artifacts_v1"
            or marker.get("schema") != "phase5_synthetic_timing_completion_v1"
            or marker.get("artifact_manifest_sha256") != expected_sha256 or marker.get("subset_completed") is not True
            or marker.get("timing_id") != manifest.get("timing_id") or set(manifest.get("files", {})) != PAYLOADS
            or manifest.get("bundle_sha256") != sha256_json(manifest["files"])):
        raise ValueError("Incomplete/invalid timing completion")
    for item in (marker, manifest):
        if {k: item.get(k) for k in status()} != status():
            raise ValueError("Timing completion cannot authorize research or close formal hypotheses")
    if {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} != BUNDLE_FILES:
        raise ValueError("Unexpected/missing timing bundle files")
    snapshot = {}
    for name in sorted(BUNDLE_FILES):
        _, path = _safe_source_member(root, name)
        if path.stat().st_size > 16*1024**2:
            raise ValueError("Timing evidence exceeds bounded file size")
        digest = sha256_file(path)
        if name in manifest["files"] and digest != manifest["files"][name]:
            raise ValueError("Timing payload hash differs")
        snapshot[str(path)] = digest
    identity, plan, measured = (read_json(root / f) for f in ("identity.json", "plan.json", "measurements.json"))
    images, expected_plan = make_plan()
    if (read_json(root / "specification.json") != timing_specification() or plan != expected_plan
            or identity["specification"] != timing_specification() or identity["plan_sha256"] != sha256_json(plan)
            or identity["scope"] != SCOPE or manifest["timing_id"] != f"phase5-timing-{sha256_json(identity)[:16]}"):
        raise ValueError("Timing specification/sequence/identity differs")
    if expected_architecture is not None and identity["architecture"] != expected_architecture:
        raise ValueError("Timing architecture/configuration/source incompatibility")
    if expected_runtime is not None and identity["runtime"] != expected_runtime:
        raise ValueError("Timing runtime/device compatibility mismatch")
    runtime = identity["runtime"]
    validate_device_evidence(read_json(root / "bootstrap.json"), runtime)
    policy = {"schema": "phase5_timing_runtime_v1", "batch_size": 1, "dtype": "float32", "autocast": False,
              "deterministic_algorithms": True, "deterministic_warn_only": False,
              "cudnn_benchmark": False, "cudnn_deterministic": True}
    if {k: runtime.get(k) for k in policy} != policy:
        raise ValueError("Timing runtime execution policy differs")
    source_snapshot = read_json(root / "source_snapshot.json")
    if (not source_snapshot or identity["source_snapshot_sha256"] != sha256_json(source_snapshot)
            or any(not isinstance(p, str) or re.fullmatch(r"[0-9a-f]{64}", str(d)) is None for p, d in source_snapshot.items())):
        raise ValueError("Timing source snapshot binding differs")
    with np.load(root / "synthetic_inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(images) or any(array_record(archive[k]) != array_record(v) for k, v in images.items()):
            raise ValueError("Synthetic input archive differs")
    with np.load(root / "synthetic_outputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"logits", "probabilities"}:
            raise ValueError("Invalid timing output archive")
        logits, probabilities = archive["logits"].copy(), archive["probabilities"].copy()
    if logits.shape != (TOTAL, 3) or logits.dtype != np.float32 or not np.isfinite(logits).all() or probabilities.shape != logits.shape:
        raise ValueError("Invalid raw synthetic outputs")
    _, mass, deviation = validate_probabilities(probabilities)
    expected_archive = {"file": "synthetic_outputs.npz", "sha256": sha256_file(root / "synthetic_outputs.npz"),
                        "arrays": {"logits": array_record(logits), "probabilities": array_record(probabilities)}}
    if (measured["output_archive"] != expected_archive
            or measured["bounded_export_file_bytes"] != (root / "synthetic_outputs.npz").stat().st_size):
        raise ValueError("Bounded export/read-back evidence differs")
    if measured["clock"] != "time.perf_counter_ns" or len(measured["calls"]) != TOTAL:
        raise ValueError("Fake/missing clock or wrong call count is not timing evidence")
    durations = ("input_construction_seconds", "adapter_seconds", "validation_hashing_seconds")
    for i, row in enumerate(measured["calls"]):
        model, seed = KEYS[i//416]
        position = expected_plan["sequence"][i % 416]
        if any(row.get(k) != v for k, v in {"call_index": i+1, "model": model, "seed": seed, **position}.items()):
            raise ValueError("Synthetic timing call order/count differs")
        if any(type(row.get(k)) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0 for k in durations):
            raise ValueError("Invalid call duration")
        if (row["logits"] != array_record(logits[i:i+1]) or row["probabilities"] != array_record(probabilities[i:i+1])
                or row["raw_row_sum"] != float(mass[i]) or row["raw_mass_deviation"] != float(deviation[i])):
            raise ValueError("Timing raw output/diagnostic binding differs")
    models = read_json(root / "models.json")
    if (models["resident_model_count"] != 6 or models["research_artifacts_accessed"] is not False or models["probe_calls"] != 0
            or [(m["model"], m["seed"]) for m in models["loaded_models"]] != list(KEYS)
            or any(m["synthetic_initialization"] is not True or m["checkpoint_loaded"] is not False
                   or m["parameter_count"] <= 0 or m["state_bytes"] <= 0
                   or any(re.fullmatch(r"[0-9a-f]{64}", m[k]) is None for k in ("state_sha256", "structure_sha256")) for m in models["loaded_models"])):
        raise ValueError("Synthetic model residency/initialization evidence differs")
    ledger = measured["ledger"]
    if (ledger["attempted"] != {"warmup": 192, "measured": 2304} or ledger["accepted"] != ledger["attempted"]
            or ledger["failure"] is not None or ledger["automatic_retries"] != 0
            or ledger["last_attempt"] != {"model": KEYS[-1][0], "seed": KEYS[-1][1],
                                           **plan["sequence"][-1], "stage": "output_validation"}):
        raise ValueError("Synthetic timing call ledger differs")
    for model, _ in KEYS[::3]:
        records = [m for m in models["loaded_models"] if m["model"] == model]
        if any(len({m[k] for m in records}) != 1 for k in ("structure_sha256", "parameter_count", "state_bytes")):
            raise ValueError("Architecture differs across synthetic seeds")
    for block in measured["blocks"]:
        rows = [r for r in measured["calls"] if (r["model"], r["seed"], r["block"]) == (block["model"], block["seed"], block["block"])]
        if len(rows) != 128 or any(block[k] != math.fsum(r[k] for r in rows) for k in durations):
            raise ValueError("Measured block aggregates differ")
        validate_memory(block["memory"], runtime["device"])
    for key in ("memory_before", "memory_after"):
        validate_memory(measured[key], runtime["device"])
    estimates = extrapolate(measured["blocks"])
    if estimates != measured["estimates"]:
        raise ValueError("Timing extrapolation differs")
    for key in ("preparation_wall_seconds", "loop_wall_seconds", "bounded_export_readback_seconds"):
        if type(measured[key]) not in (float, int) or not math.isfinite(measured[key]) or measured[key] < 0:
            raise ValueError("Invalid timing phase envelope")
    validate_loop_envelope(measured)
    binding = {"timing_id": manifest["timing_id"], "manifest_sha256": expected_sha256,
               "bundle_sha256": manifest["bundle_sha256"], "architecture_sha256": sha256_json(identity["architecture"]),
               "runtime_sha256": sha256_json(identity["runtime"]), "plan_sha256": identity["plan_sha256"],
               "estimated_adapter_seconds": estimates["estimated_adapter_seconds"],
               "synthetic_measurements_available": True, "research_execution_authorized": False}
    assert_source_snapshot_unchanged(snapshot)
    return {"binding": binding, "snapshot": snapshot, "identity": identity, "measurements": measured}


def copy_timing_evidence(directory, destination, expected_sha256):
    evidence = validate_timing_bundle(directory, expected_sha256)
    for name in sorted(BUNDLE_FILES):
        _, source = _safe_source_member(Path(directory).resolve(), name)
        atomic_write_bytes(Path(destination) / name, source.read_bytes())
    copied = validate_timing_bundle(destination, expected_sha256)
    if copied["binding"] != evidence["binding"]:
        raise ValueError("Copied timing evidence differs")
    assert_source_snapshot_unchanged(evidence["snapshot"])
    return copied


def run_synthetic_timing(*, authorize_synthetic_timing=False, attempt_id=None, device="cpu", output_root=None):
    if authorize_synthetic_timing is not True:
        raise ValueError("Explicit --authorize-synthetic-timing required")
    if not isinstance(attempt_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", attempt_id) is None:
        raise ValueError("An explicit safe --attempt-id is required; no automatic retries")
    clock, ledger = PerfClock(), TimingLedger()
    preparation_start = clock()
    git = validate_evaluator_repository(PROJECT_ROOT)
    spec = load_specification()
    environment = runtime_environment(device)
    validate_device_evidence(bootstrap_record(), environment)
    architecture, snapshot = architecture_contract()
    images, plan = make_plan()
    identity = {"scope": SCOPE, "attempt_id": attempt_id, "implementation_commit": git["commit"], "specification": spec,
                "architecture": architecture, "runtime": environment, "plan_sha256": sha256_json(plan),
                "source_snapshot_sha256": sha256_json(snapshot)}
    timing_id = f"phase5-timing-{sha256_json(identity)[:16]}"
    root = Path(NAMESPACE if output_root is None else output_root).absolute()
    guard_new_namespace(root)
    if not root.resolve().is_relative_to(NAMESPACE.resolve()):
        raise ValueError("Timing outputs must remain inside their separate protected namespace")
    output, created = root / timing_id, False
    guard_new_namespace(output)
    if output.exists(): raise FileExistsError("Never overwrite/reuse a completed or partial timing attempt")
    try:
        output.mkdir(parents=True, exist_ok=False); created = True
        for name, value in (("identity.json", identity), ("specification.json", spec), ("plan.json", plan),
                            ("bootstrap.json", bootstrap_record()), ("source_snapshot.json", snapshot)):
            atomic_write_json(output / name, value)
        write_map_archive(output / "synthetic_inputs.npz", images)
        # Persist/hash the complete sequence before model preparation and all forward calls.
        with synthetic_predictors(device, NAMESPACE / ".disabled_pycache") as (predictors, models):
            atomic_write_json(output / "models.json", models)
            before = memory_snapshot(device)
            preparation = elapsed(clock, preparation_start)
            result = measure_schedule(predictors, images, plan, clock=clock,
                synchronize=lambda: synchronize(device), ledger=ledger, memory=lambda: memory_snapshot(device))
            start = clock()
            archive = write_map_archive(output / "synthetic_outputs.npz", result.pop("raw"))
            bounded_export = elapsed(clock, start)
            result.update(preparation_wall_seconds=preparation, bounded_export_readback_seconds=bounded_export,
                          bounded_export_file_bytes=(output / "synthetic_outputs.npz").stat().st_size,
                          output_archive=archive, ledger=ledger.record(), memory_before=before, memory_after=memory_snapshot(device),
                          estimates=extrapolate(result["blocks"]))
        assert_source_snapshot_unchanged(snapshot)
        if architecture_contract()[0] != architecture or runtime_environment(device) != environment or validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Timing source/runtime/repository changed")
        atomic_write_json(output / "measurements.json", result)
        atomic_write_text(output / "summary.md", f"# {timing_id}\n\n{WARNING}\n\n"
            f"2496 calls: 192 warm-up and 2304 measured; 18 measured blocks. "
            f"Adapter-only estimate: {result['estimates']['estimated_adapter_seconds']:.9g} seconds for 417240 research calls. "
            "Input construction and external validation/hashing are disjoint from adapter timing. "
            "Preparation/loop envelopes are not added again. Bounded synthetic NPZ export is not a research-I/O estimate.\n")
        files = {name: sha256_file(output / name) for name in sorted(PAYLOADS)}
        manifest = atomic_write_json(output / "artifact_manifest.json", {"schema": "phase5_synthetic_timing_artifacts_v1",
            "timing_id": timing_id, **status(), "files": files, "bundle_sha256": sha256_json(files)})
        atomic_write_json(output / "COMPLETED.json", {"schema": "phase5_synthetic_timing_completion_v1",
            "timing_id": timing_id, **status(), "subset_completed": True, "artifact_manifest_sha256": sha256_file(manifest)})
        validate_timing_bundle(output, sha256_file(manifest), expected_architecture=architecture, expected_runtime=environment)
        assert_source_snapshot_unchanged(snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git: raise ValueError("Repository changed before timing completion")
        return output
    except BaseException as exc:
        if created:
            try: (output / "COMPLETED.json").unlink(missing_ok=True)
            except BaseException: pass
            try:
                atomic_write_json(output / "FAILED.json", {**status(), "subset_completed": False,
                    "error_type": type(exc).__name__, "message": str(exc), "ledger": ledger.record()})
            except BaseException: pass
        raise


def timing_cli(argv=None):
    parser = argparse.ArgumentParser(description="Separately authorized synthetic timing; never research evidence")
    parser.add_argument("--authorize-synthetic-timing", action="store_true")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output = run_synthetic_timing(authorize_synthetic_timing=args.authorize_synthetic_timing,
        attempt_id=args.attempt_id, device=args.device, output_root=args.output_root)
    print(f"TIMING_BUNDLE={output}")
    print(f"TIMING_MANIFEST_SHA256={sha256_file(output / 'artifact_manifest.json')}")
    return 0
