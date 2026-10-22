"""Opt-in Candidate C saved-prediction integration; preflight never scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import mpmath
import pandas as pd
import numpy as np

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_csv, atomic_write_json, atomic_write_text
from .phase4_ablation_runner import guard_new_namespace
from .phase4_evaluation_runner import (
    PROJECT_ROOT, validate_evaluator_repository,
    snapshot_source_files,
)
from .phase5_bounded_mean import HYPOTHESIS_IDS
from .phase5_inference_evidence import load_evidence, validate_evidence_payload, document_snapshot
from .phase5_inference_exports import compute_family, validate_arithmetic
from .phase5_inference_protocol import SCOPE, load_protocol
from .phase5_inference_sources import (
    load_inference_inputs, prediction_records, validate_records, strict_path, merge_snapshot, MODELS, SEEDS,
    assert_inference_snapshot_unchanged as assert_source_snapshot_unchanged,
)

PROTOCOL_PATH = PROJECT_ROOT / "configs/phase5_inference_v1.json"
NAMESPACE = PROJECT_ROOT / "artifacts/phase5/paired_correctness"
IMPLEMENTATION_FILES = (
    "src/consendxformer/phase5_bounded_mean.py", "src/consendxformer/phase5_inference_sources.py",
    "src/consendxformer/phase5_inference_evidence.py", "src/consendxformer/phase5_inference_exports.py",
    "src/consendxformer/phase5_inference_protocol.py", "src/consendxformer/phase5_inference_runner.py",
    "scripts/51_run_phase5_inference.py", "requirements-phase5-inference.txt",
)
WARNING = (
    "Exploratory post-hoc. Candidate C explicitly substitutes for McNemar/Durkalski; "
    "the latter's calibration failure is not repaired. Conditional independent heterogeneous "
    "patient clusters are an assumption, not established by source checks or tests. "
    "Equal-patient correctness is secondary; macro-F1 remains primary. Existing nodule-weighted "
    "accuracy and unadjusted intervals are unchanged. Non-rejection is not equivalence. "
    "No general classification-superiority or clinical/SOTA claim follows. "
    "The full supplement and Stage A remain incomplete."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flags(preflight):
    return {"scope": SCOPE, "preflight_only": preflight, "model_forwards": 0,
            "training_performed": False, "formal_tests_computed": not preflight,
            "full_supplemental_completion": False, "stage_a_complete": False,
            "research_readiness_certified": False,
            "analysis_status": "exploratory_post_hoc"}


def _output_path(root, eid, directories, snapshot):
    root = strict_path(root)
    guard_new_namespace(root)
    if not root.is_relative_to(NAMESPACE.resolve()):
        raise ValueError("Outputs must stay in the separate paired-correctness namespace")
    output = strict_path(root / eid)
    guard_new_namespace(output)
    if any((p / "PREFLIGHT.json").exists() for p in (root, *root.parents)):
        raise ValueError("Cannot nest into a completed preflight")
    protected = [*directories, *(PROJECT_ROOT / p for p in ("data", "external", "artifacts/phase0",
                  "artifacts/phase2", "artifacts/phase3", "artifacts/phase4"))]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ValueError("Output overlaps protected sources")
    if any(Path(p).is_relative_to(output) for p in snapshot):
        raise ValueError("Output overlaps evidence or implementation inputs")
    if output.exists():
        raise FileExistsError("Never overwrite a completed, partial or failed attempt")
    return output


def _check_csv(path, records):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(records[0]):
            raise ValueError("CSV fields changed")
        if list(reader) != [{k: str(v) for k, v in row.items()} for row in records]:
            raise ValueError("CSV read-back differs from exact JSON records")


def _source_binding(identity, provenance):
    snapshot = provenance["source_snapshot"]
    if not snapshot or identity["source_snapshot_sha256"] != sha256_json(snapshot):
        raise ValueError("Source snapshot binding mismatch")
    records = identity["sources"]
    if (len(records) != 18 or len({r["source_dir"] for r in records}) != 18
            or {(r["model"], r["seed"]) for r in records} != {(m, s) for m in MODELS for s in SEEDS}):
        raise ValueError("Source record set mismatch")
    by_key = {(r["model"], r["seed"]): r for r in records}
    portable = {p.replace("\\", "/"): h for p, h in snapshot.items()}
    for record in records:
        if type(record["seed"]) is not int:
            raise ValueError("Noninteger seed")
        # Paths originate on the execution host; use portable slash comparisons
        # for offline Windows/Linux read-back, without resolving live sources.
        root = record["source_dir"].replace("\\", "/").rstrip("/")
        for relative, digest in record["source_snapshot"].items():
            if (relative.startswith("/") or "\\" in relative or ":" in relative
                    or any(p in ("", ".", "..") for p in relative.split("/"))
                    or portable.get(root + "/" + relative) != digest):
                raise ValueError("Per-source ledger differs from global source snapshot")
        if record["model"] in ("ConsenDxFormer", "B1"):
            if record["descriptor"]["seed"] != record["seed"]:
                raise ValueError("Descriptor seed binding mismatch")
        else:
            full = by_key[("ConsenDxFormer", record["seed"])]
            expected_reference = {k: full[k] for k in ("source_dir", "descriptor", "best_checkpoint_sha256")}
            if (record["identity"]["seed"] != record["seed"]
                    or record["identity"]["variant"] + "_off" != record["model"]
                    or record["identity"]["full_model_reference"] != expected_reference):
                raise ValueError("Ablation same-seed full reference binding mismatch")
        for field, name in (("prediction_sha256", "predictions/test.csv"),
                            ("best_checkpoint_sha256", "checkpoints/best.pt")):
            if portable.get(root + "/" + name) != record[field]:
                raise ValueError("Original prediction/checkpoint hash not bound")


def validate_completion(directory):
    """Offline integrity + independent numerical read-back. No live source load."""
    directory = strict_path(directory)
    if (directory / "FAILED.json").exists():
        raise ValueError("Failed attempts cannot be completed")
    manifest = read_json(directory / "artifact_manifest.json")
    preflight = manifest.get("preflight_only")
    if type(preflight) is not bool:
        raise ValueError("Invalid preflight status")
    marker_name = "PREFLIGHT.json" if preflight else "COMPLETED.json"
    marker = read_json(directory / marker_name)
    required = {"protocol.json", "evaluation.json", "provenance.json", "evidence.json", "saved_predictions.json",
                "inference.json", "summary.md", "tables/hypotheses.csv"}
    if not preflight:
        required.add("tables/patient_counts.csv")
    files = manifest.get("files", {})
    actual = set()
    for path in directory.rglob("*"):
        strict_path(path)
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if (set(files) != required or actual != required | {"artifact_manifest.json", marker_name}
            or sha256_json(files) != manifest.get("bundle_sha256")
            or marker.get("artifact_manifest_sha256") != sha256_file(directory / "artifact_manifest.json")):
        raise ValueError("Incomplete/extra/corrupt artifact manifest")
    for name, expected in files.items():
        if sha256_file(directory / name) != expected:
            raise ValueError("Output checksum mismatch")
    evaluation, provenance = read_json(directory / "evaluation.json"), read_json(directory / "provenance.json")
    policy = load_protocol(directory / "protocol.json")
    for record in (manifest, marker, evaluation, provenance):
        if sha256_json({k: record.get(k) for k in flags(preflight)}) != sha256_json(flags(preflight)):
            raise ValueError("Subset-only status mismatch")
    if (marker.get("schema") != "phase5_inference_completion_v1"
            or manifest.get("schema") != "phase5_inference_artifacts_v1"
            or marker.get("subset_completed") is not (not preflight)
            or marker.get("preflight_completed") is not preflight):
        raise ValueError("Completion cannot certify a different stage")
    identity = evaluation["identity"]
    evidence = read_json(directory / "evidence.json")
    payload = read_json(directory / "saved_predictions.json")
    if (identity != provenance["identity"] or identity["protocol"] != policy
            or identity["evidence_sha256"] != sha256_json(evidence)
            or identity["predictions_sha256"] != sha256_json(payload)
            or identity["preflight_only"] is not preflight):
        raise ValueError("Identity/input policy mismatch")
    eid = f"phase5-inference-{'preflight' if preflight else 'results'}-{sha256_json(identity)[:16]}"
    if any(r.get("evaluation_id") != eid for r in (manifest, marker, evaluation, provenance)):
        raise ValueError("Content-addressed identity mismatch")
    if evaluation["warning"] != WARNING:
        raise ValueError("Interpretation warning changed")
    if provenance["reference_evidence"] != validate_evidence_payload(evidence):
        raise ValueError("Evidence validation status mismatch")
    _source_binding(identity, provenance)
    validate_records(payload["records"], payload["expected_identities"])
    result = read_json(directory / "inference.json")
    if preflight:
        if result != {"hypotheses": [{"hypothesis_id": k, "status": "unresolved_not_computed"}
                                      for k in HYPOTHESIS_IDS], "patients": []}:
            raise ValueError("Preflight cannot contain research statistics")
    else:
        validate_arithmetic(payload["records"], payload["expected_identities"], result)
        _check_csv(directory / "tables/patient_counts.csv", result["patients"])
    _check_csv(directory / "tables/hypotheses.csv", result["hypotheses"])
    return marker


def _record_failure(output, exc):
    # Independent best-effort cleanup preserves R1's original exception.
    try:
        atomic_write_json(output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc),
                         "scope": SCOPE, "completed": False, "model_forwards": 0})
    except BaseException:
        pass
    for name in ("COMPLETED.json", "PREFLIGHT.json"):
        try:
            (output / name).unlink(missing_ok=True)
        except BaseException:
            pass


def run_inference(*, full_source_dirs, b1_source_dirs, ablation_source_dirs,
                  core_evidence_zip, r_reference_dir, r_reference_hashes, r_reference_hashes_sha256,
                  preflight_only=True, authorize_preflight=False, authorize_research_statistics=False,
                  output_root=None):
    if type(preflight_only) is not bool:
        raise ValueError("Explicit preflight mode required")
    if ((preflight_only and (authorize_preflight is not True or authorize_research_statistics is not False))
            or (not preflight_only and (authorize_research_statistics is not True or authorize_preflight is not False))):
        raise ValueError("Explicit mode-specific authorization required before source access")
    git = validate_evaluator_repository(PROJECT_ROOT)
    policy = load_protocol(PROTOCOL_PATH)
    snapshot = document_snapshot(PROJECT_ROOT)
    merge_snapshot(snapshot, snapshot_source_files([PROTOCOL_PATH, *(PROJECT_ROOT / p for p in IMPLEMENTATION_FILES)]))
    evidence, evidence_snapshot = load_evidence(core_evidence_zip, r_reference_dir, r_reference_hashes,
                                                r_reference_hashes_sha256, PROJECT_ROOT)
    merge_snapshot(snapshot, evidence_snapshot)
    inputs = load_inference_inputs(full_source_dirs, b1_source_dirs, ablation_source_dirs)
    merge_snapshot(snapshot, inputs.snapshot)
    payload = {"expected_identities": inputs.expected_identities, "records": prediction_records(inputs.frames)}
    validate_records(payload["records"], payload["expected_identities"])
    identity = {"scope": SCOPE, "protocol": policy, "implementation": git, "preflight_only": preflight_only,
                "sources": inputs.records, "source_snapshot_sha256": sha256_json(snapshot),
                "predictions_sha256": sha256_json(payload), "evidence_sha256": sha256_json(evidence),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "mpmath": mpmath.__version__}}
    eid = f"phase5-inference-{'preflight' if preflight_only else 'results'}-{sha256_json(identity)[:16]}"
    output = _output_path(NAMESPACE if output_root is None else output_root, eid, inputs.directories, snapshot)
    assert_source_snapshot_unchanged(snapshot)
    output.mkdir(parents=True, exist_ok=False)
    try:
        # The entire 18-source/cohort/evidence barrier has passed. Only this
        # explicit branch can compute Candidate C; preflight never calls it.
        result = ({"hypotheses": [{"hypothesis_id": k, "status": "unresolved_not_computed"}
                                   for k in HYPOTHESIS_IDS], "patients": []} if preflight_only else
                  compute_family(payload["records"], payload["expected_identities"]))
        if not preflight_only:
            validate_arithmetic(payload["records"], payload["expected_identities"], result)
        status = flags(preflight_only)
        for name, value in (("protocol.json", policy), ("evidence.json", evidence),
                            ("saved_predictions.json", payload), ("inference.json", result),
                            ("evaluation.json", {"evaluation_id": eid, "identity": identity, "warning": WARNING, **status}),
                            ("provenance.json", {"evaluation_id": eid, "identity": identity,
                                "source_snapshot": snapshot, "reference_evidence": validate_evidence_payload(evidence), **status})):
            atomic_write_json(output / name, value)
        atomic_write_csv(output / "tables/hypotheses.csv",
                         pd.DataFrame(result["hypotheses"], columns=sorted(result["hypotheses"][0])))
        if not preflight_only:
            atomic_write_csv(output / "tables/patient_counts.csv",
                             pd.DataFrame(result["patients"], columns=sorted(result["patients"][0])))
        atomic_write_text(output / "summary.md", f"# {eid}\n\n{WARNING}\n\n"
            + ("Preflight only: all 15 hypotheses remain unresolved; no Candidate C statistics.\n"
               if preflight_only else "Only the 15 paired-correctness hypotheses are computed in this new bundle.\n")
            + "All 18 sources, 183 nodules and 99 patients retained. "
            + ("No p-values or decisions are exported. " if preflight_only else
               "Raw and adjusted p-values are exported as numerical enclosures, not falsely exact decimal scalars. ")
            + "Historical evidence/status is "
              "immutable; the later R pass is bound additively. No R process or campaign was rerun. "
              "This completion does not authorize another execution or certify full-supplement/Stage A readiness.\n")
        assert_source_snapshot_unchanged(snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Implementation repository changed during execution")
        files = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
        manifest = atomic_write_json(output / "artifact_manifest.json", {"schema": "phase5_inference_artifacts_v1",
            "evaluation_id": eid, "files": files, "bundle_sha256": sha256_json(files), **status})
        assert_source_snapshot_unchanged(snapshot)
        atomic_write_json(output / ("PREFLIGHT.json" if preflight_only else "COMPLETED.json"), {
            "schema": "phase5_inference_completion_v1", "evaluation_id": eid,
            "preflight_completed": preflight_only, "subset_completed": not preflight_only,
            "artifact_manifest_sha256": sha256_file(manifest), **status})
        validate_completion(output)
        assert_source_snapshot_unchanged(snapshot)
        if validate_evaluator_repository(PROJECT_ROOT) != git:
            raise ValueError("Repository changed during completion read-back")
    except BaseException as exc:
        _record_failure(output, exc)
        raise
    return output


def inference_cli(argv=None):
    parser = argparse.ArgumentParser(description="Saved-prediction Candidate C; zero model forwards")
    for name in ("full-source", "b1-source", "ablation-source"):
        parser.add_argument("--" + name, action="append", required=True)
    for name in ("core-evidence-zip", "r-reference-dir", "r-reference-hashes", "r-reference-hashes-sha256"):
        parser.add_argument("--" + name, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--authorize-preflight", action="store_true")
    mode.add_argument("--authorize-research-statistics", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output = run_inference(full_source_dirs=args.full_source, b1_source_dirs=args.b1_source,
        ablation_source_dirs=args.ablation_source, core_evidence_zip=args.core_evidence_zip,
        r_reference_dir=args.r_reference_dir, r_reference_hashes=args.r_reference_hashes,
        r_reference_hashes_sha256=args.r_reference_hashes_sha256,
        preflight_only=args.authorize_preflight, authorize_preflight=args.authorize_preflight,
        authorize_research_statistics=args.authorize_research_statistics, output_root=args.output_root)
    print(f"OUTPUT_DIR={output}\nMODEL_FORWARDS=0\nFULL_SUPPLEMENTAL_COMPLETION=False")
    return 0
