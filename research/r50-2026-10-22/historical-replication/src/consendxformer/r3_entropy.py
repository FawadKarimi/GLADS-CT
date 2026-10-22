"""Opt-in descriptive stored-entropy exports; no model or metric-loader imports."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess

from .hashing import sha256_file, sha256_json
from . import r3_entropy_source as source

SCOPE = "r3_entropy_descriptive_only"
PROTOCOL = {
    "schema_version": 1, "scope": SCOPE, "validation_scope": "archived_saved_records",
    "proposal_sha256": source.PROPOSAL_SHA256,
    "archive_sha256": source.ARCHIVE_SHA256, "member_pins": source.PINS,
    "payload_digest": source.PAYLOAD_DIGEST, "cohort_sha256": source.COHORT_SHA256,
    "cohort_schema": "r3_archived_cohort_v1", "cohort_fields": list(source.COHORT_FIELDS),
    "cohort_order": "lexical tuple of exact CSV strings in cohort_fields order",
    "hash_convention": "UTF-8 JSON, sorted keys, compact separators, ensure_ascii=False, no newline",
    "source_row_index": "zero-based CSV data-record index, excluding header (not physical line)",
    "seeds": list(source.SEEDS), "models": list(source.MODELS),
    "quantiles": "linear interpolation at (n-1)*q; q=0.25,0.5,0.75",
    "sd": "sample standard deviation, ddof=1; null for n<2",
    "weighting": "nodule-weighted within each model/seed/group; no pooling",
    "axes": {"correctness": ["correct", "incorrect"], "true_class": ["indeterminate", "definitive"]},
    "contrasts": ["incorrect_minus_correct", "indeterminate_minus_definitive"],
    "entropy": "stored predictive_entropy_nats only; no recomputation, clipping or normalization",
    "invalid_policy": "fail closed on missing/nonfinite values or range anomaly outside [0,ln(3)]",
    "empty_policy": "retain empty groups with null statistics; null contrast if either group empty",
}
LIMITATIONS = {
    "validation_scope": "archived_saved_records",
    "original_prediction_bytes_reverified": False, "original_supervision_bytes_reverified": False,
    "checkpoint_bytes_reverified": False, "live_sources_reverified": False,
    "model_forwards": 0, "training_performed": False, "entropy_recomputed": False,
    "formal_tests_performed": False, "exploratory_post_hoc": True,
    "r3_review_closed": False, "r4_closed": False, "r5_closed": False,
    "stage_a_certified": False, "full_supplement_complete": False,
    "interpretation": "Descriptive conditional on reviewed archived records; macro-F1 remains primary. "
        "No independence across nodules within patients or repeated seeds, causal superiority, calibration "
        "or population inference. Correctness is outcome-defined; definitive pools benign and malignant. "
        "S7 is historical attributed evidence, not fresh original-file validation.",
}


def _quantile(values, q):
    position = (len(values) - 1) * q
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def summarize(rows):
    """Pure arithmetic for already validated records, also usable with invented fixtures."""
    groups = {(model, seed): [] for model in source.MODELS for seed in source.SEEDS}
    seen = set()
    for index, row in enumerate(rows):
        source.require(row.get("seed") in {str(s) for s in source.SEEDS}, f"source row {index}: seed")
        key = (row.get("model"), int(row["seed"]))
        source.require(key in groups, f"source row {index}: model")
        identity = tuple(row.get(f) for f in source.IDENTITY)
        source.require(all(identity) and row.get("split") == "test", f"source row {index}: identity")
        source.require((key, identity) not in seen, f"source row {index}: duplicate")
        seen.add((key, identity))
        source.require(row.get("true_class") in source.CLASSES and row.get("predicted_class") in source.CLASSES,
                       f"source row {index}: class")
        source.require(bool(row.get("experiment_id")), f"source row {index}: experiment")
        groups[key].append((index, row, source.entropy_value(row, index)))
    summaries, contrasts, memberships, overlaps = [], [], [], []
    for (model, seed), records in groups.items():
        source.require(bool(records), f"missing model/seed: {model}/{seed}")
        experiments = {r[1]["experiment_id"] for r in records}
        source.require(len(experiments) == 1, "mixed experiment")
        base = dict(model=model, seed=seed, experiment_id=next(iter(experiments)))
        for axis, names in PROTOCOL["axes"].items():
            selected = {name: [] for name in names}
            for index, row, value in records:
                name = ("correct" if row["predicted_class"] == row["true_class"] else "incorrect") if axis == "correctness" else (
                    "indeterminate" if row["true_class"] == "indeterminate" else "definitive")
                selected[name].append((row, value))
                memberships.append({**base, **{f: row[f] for f in source.IDENTITY},
                    "true_class": row["true_class"], "predicted_class": row["predicted_class"],
                    "axis": axis, "group": name, "source_row_index": index,
                    "stored_entropy_text": row["predictive_entropy_nats"]})
            stats, patients = {}, {}
            for name in names:
                values = sorted(value for _, value in selected[name])
                patients[name] = {row["patient_id"] for row, _ in selected[name]}
                n = len(values)
                statrow = {**base, "axis": axis, "group": name, "eligible_total": len(records),
                    "nodule_count": n, "patient_count": len(patients[name]), "finite_entropy_count": n,
                    "missing_invalid_count": 0, "status": "ok" if n else "empty_group",
                    "sd_reason": None if n >= 2 else "insufficient_n_for_sample_sd",
                    "mean_nats": statistics.mean(values) if n else None,
                    "sample_sd_nats": statistics.stdev(values) if n >= 2 else None,
                    "q1_nats": _quantile(values, .25) if n else None,
                    "median_nats": _quantile(values, .5) if n else None,
                    "q3_nats": _quantile(values, .75) if n else None,
                    "min_nats": min(values) if n else None, "max_nats": max(values) if n else None}
                summaries.append(statrow)
                stats[name] = statrow
            first, second = ("incorrect", "correct") if axis == "correctness" else ("indeterminate", "definitive")
            defined = bool(stats[first]["nodule_count"] and stats[second]["nodule_count"])
            contrasts.append({**base, "axis": axis, "contrast": first + "_minus_" + second,
                "first_nodule_count": stats[first]["nodule_count"], "second_nodule_count": stats[second]["nodule_count"],
                "first_patient_count": stats[first]["patient_count"], "second_patient_count": stats[second]["patient_count"],
                "mean_difference_nats": stats[first]["mean_nats"] - stats[second]["mean_nats"] if defined else None,
                "reason": None if defined else "empty_comparison_group"})
            shared = sorted(patients[first] & patients[second])
            overlaps.append({**base, "axis": axis, "patient_union_count": len(patients[first] | patients[second]),
                "patient_overlap_count": len(shared), "patient_overlap_ids": shared})
    return dict(summaries=summaries, contrasts=contrasts, memberships=memberships, patient_overlap=overlaps)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def run(archive_path, namespace, proposal_path, *, authorized=False):
    """Authorization must be separately granted for real execution; CLI has no pin overrides."""
    source.require(authorized, "explicit execution authorization required")
    proposal = Path(proposal_path).read_bytes()
    source.require(source.digest(proposal) == source.PROPOSAL_SHA256, "proposal hash")
    validated = source.load_archive(archive_path)
    namespace = Path(namespace)
    source.require(namespace.parts[-3:] == ("artifacts", "phase5", "r3_entropy"), "protected output namespace required")
    source.require(not namespace.is_symlink(), "symlink namespace")
    source.require(all(not p.is_symlink() for p in namespace.parents), "symlink namespace ancestor")
    identity = {"protocol_sha256": sha256_json(PROTOCOL), "archive_sha256": source.ARCHIVE_SHA256,
                "adapter_sha256": sha256_file(__file__), "source_adapter_sha256": sha256_file(source.__file__)}
    run_id = "r3-entropy-" + sha256_json(identity)[:16]
    output = namespace / run_id
    source.require(not output.exists(), "output exists; no overwrite")
    result = summarize(validated["rows"])
    source.require(len(result["summaries"]) == 24 and len(result["contrasts"]) == 12, "output shape")
    payloads = {
        "protocol.json": _json_bytes(PROTOCOL), "proposal.md": proposal,
        "source/archive.zip": validated["archive"], "archived_cohort.json": _json_bytes(validated["cohort"]),
        "provenance.json": _json_bytes({"scope": SCOPE, "identity": identity,
            "archived_provenance": validated["provenance"], "limitations": LIMITATIONS}),
        "limitations.json": _json_bytes(LIMITATIONS), "patient_overlap.json": _json_bytes(result["patient_overlap"]),
        "tables/group_summaries.csv": _csv_bytes(result["summaries"]),
        "tables/descriptive_contrasts.csv": _csv_bytes(result["contrasts"]),
        "tables/membership.csv": _csv_bytes(result["memberships"]),
        "validation.json": _json_bytes({"source_records": 1098, "model_seed_groups": 6,
            "nodules_per_group": 183, "patients_per_group": 99, "snapshot_entries": 132,
            "summary_rows": 24, "contrast_rows": 12, "membership_rows": 2196,
            "archive_before_sha256": source.digest(validated["archive"])}),
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for relative, data in payloads.items():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        source.require(sha256_file(archive_path) == source.ARCHIVE_SHA256, "source archive changed during export")
        source.require(sha256_file(proposal_path) == source.PROPOSAL_SHA256, "proposal changed during export")
        source.require(sha256_file(__file__) == identity["adapter_sha256"] and
                       sha256_file(source.__file__) == identity["source_adapter_sha256"],
                       "adapter changed during export")
        hashes = {name: source.digest(data) for name, data in payloads.items()}
        source.require(all(sha256_file(output / name) == value for name, value in hashes.items()), "export readback")
        manifest = {"scope": SCOPE, "run_id": run_id, "files": hashes, "bundle_sha256": sha256_json(hashes)}
        with (output / "artifact_manifest.json").open("xb") as stream:
            stream.write(_json_bytes(manifest))
        with (output / "COMPLETED.json").open("xb") as stream:
            stream.write(_json_bytes({"scope": SCOPE, "run_id": run_id,
                "artifact_manifest_sha256": sha256_file(output / "artifact_manifest.json"),
                "source_archive_unchanged": True, **LIMITATIONS}))
    except BaseException:
        # Partial output is retained for review. Never allow an interruption to certify completion.
        (output / "COMPLETED.json").unlink(missing_ok=True)
        raise
    return output


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    # Output location is fixed, never an arbitrary existing bundle supplied by the caller.
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    source.require(branch == "phase4-consendxformer" and not status, "clean approved branch required")
    print(run(args.archive, repo / "artifacts/phase5/r3_entropy",
              repo / "docs/r3_entropy_resolution_proposal.md", authorized=args.authorize_execution))
    return 0
