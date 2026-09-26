"""Pinned archive-only validation. Never dereference archived source paths."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from .hashing import sha256_json

PROPOSAL_SHA256 = "e767983361beada298f76fbc6be12c940f21da1d02be35ae3d34af0dbb125c29"
ARCHIVE_SHA256 = "6530307c59cf9f8a58d41982b64eed8aaeff8b68c39e1ff24a55dde1543282bd"
EVALUATION_ID = "phase4-evaluation-8c5876a39d46aaa7"
PINS = {
    "artifact_manifest.json": "d2eaec89a15dda91ddbd7efddb977c90fc5dacf5e6b36b0d844b2d48fbe0c02a",
    "provenance/provenance.json": "79a90082dc31db738a56dcfbb1775a2f603bebf2e2590952567064cc48e1ac26",
    "tables/uncertainty_per_nodule_per_seed.csv": "7c2b8790a27342e8b03a40f2297b15991b1d539553f408798ccbb5be5c2235b8",
}
PAYLOAD_DIGEST = "a7bc955d56fcd8e5a55f2fdebcda24fcf3c0f3118d69418fd792cd88c35997b8"
COHORT_SHA256 = "8b836f06e8289000073ce0df9a82d6542c8b11fed4faa9e4d02fc6a5b1162f04"
MODELS = ("full", "disagreement_weighting_off")
SEEDS = (42, 43, 44)
CLASSES = ("benign", "indeterminate", "malignant")
IDENTITY = ("patient_id", "series_instance_uid", "nodule_id", "split")
COHORT_FIELDS = (*IDENTITY, "true_class")
EXPERIMENTS = {
    ("full", 42): "phase4-final-s42-c1e650d77585ed9c",
    ("full", 43): "phase4-final-s43-946220804796df93",
    ("full", 44): "phase4-final-s44-8c74c3dd1e666bb5",
    ("disagreement_weighting_off", 42): "phase4-ablation-disagreement_weighting-s42-e1a00dcc25479c12",
    ("disagreement_weighting_off", 43): "phase4-ablation-disagreement_weighting-s43-d21e9a6a3beaa01e",
    ("disagreement_weighting_off", 44): "phase4-ablation-disagreement_weighting-s44-d89020abb1db5cfe",
}
MEMBERS = {
    "COMPLETED.json", "README.md", "artifact_manifest.json", "curves/risk_coverage.json",
    "evaluation.json", "provenance/provenance.json", "tables/paired_per_nodule_per_seed.csv",
    "tables/paired_per_nodule_seed_mean.csv", "tables/uncertainty_per_nodule_per_seed.csv",
    "tables/uncertainty_per_nodule_seed_mean.csv",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def read_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "CSV headers")
    rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), "CSV row width")
    return rows


def cohort_ledger(rows):
    """Exact strings; lexical tuple order; canonical JSON via hashing.sha256_json."""
    return {"schema": "r3_archived_cohort_v1", "fields": list(COHORT_FIELDS), "rows": [
        {field: row[field] for field in COHORT_FIELDS}
        for row in sorted(rows, key=lambda row: tuple(row[field] for field in COHORT_FIELDS))
    ]}


def entropy_value(row, index):
    location = f"source row {index} ({tuple(row.get(field) for field in IDENTITY)})"
    try:
        value = float(row["predictive_entropy_nats"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{location}: missing/malformed stored entropy") from exc
    require(math.isfinite(value), f"{location}: nonfinite stored entropy")
    require(0 <= value <= math.log(3), f"{location}: entropy range anomaly; source review required")
    return value


def validate_rows(rows, paired):
    require(len(rows) == 1098, "raw row count")
    groups = {key: {} for key in EXPERIMENTS}
    for index, row in enumerate(rows):
        key = (row.get("model"), row.get("seed"))
        require(key[1] in {str(seed) for seed in SEEDS}, f"source row {index}: seed")
        key = (key[0], int(key[1]))
        require(key in groups, f"source row {index}: model")
        identity = tuple(row.get(field) for field in IDENTITY)
        require(all(identity) and row.get("split") == "test", f"source row {index}: identity/split")
        require(identity not in groups[key], f"source row {index}: duplicate identity")
        require(row.get("experiment_id") == EXPERIMENTS[key], f"source row {index}: experiment")
        require(row.get("true_class") in CLASSES and row.get("predicted_class") in CLASSES,
                f"source row {index}: class")
        entropy_value(row, index)
        groups[key][identity] = row
    reference = groups[("full", 42)]
    ledger = cohort_ledger(list(reference.values()))
    require(sha256_json(ledger) == COHORT_SHA256, "archived cohort hash")
    for key, group in groups.items():
        require(len(group) == 183 and len({i[0] for i in group}) == 99, f"cohort counts: {key}")
        require(set(group) == set(reference), f"cohort identities: {key}")
        require(len({r['nodule_id'] for r in group.values()}) == 183, "duplicate nodule")
        for identity, row in group.items():
            for field in ("true_class", "disagreement_entropy", "disagreement_group", "disagreement_threshold"):
                require(row.get(field) not in (None, "") and row[field] == reference[identity][field],
                        f"cohort frozen field: {key}/{identity}/{field}")
    require(len(paired) == 549, "paired row count")
    seen = set()
    for row in paired:
        require(row.get("seed") in {str(s) for s in SEEDS}, "paired seed")
        seed = int(row["seed"])
        identity = tuple(row.get(field) for field in IDENTITY)
        require((seed, identity) not in seen and identity in reference, "paired duplicate/identity")
        seen.add((seed, identity))
        for suffix, model in (("full", "full"), ("off", "disagreement_weighting_off")):
            original = groups[(model, seed)][identity]
            for field in ("true_class", "predicted_class", "experiment_id", "predictive_entropy_nats"):
                column = field if field == "true_class" else field + "_" + suffix
                require(row.get(column) == original[field], f"paired/raw mismatch: {column}")
    return ledger


def validate_bindings(provenance):
    require(provenance["analysis"] == "uncertainty" and provenance["seeds"] == list(SEEDS), "analysis/seeds")
    require(len(provenance["sources"]) == 6, "source count")
    sources, merged = {}, {}
    for source in provenance["sources"]:
        ident = source["identity"]
        model = "full" if "model" in ident else "disagreement_weighting_off"
        key = (model, ident["seed"])
        require(key in EXPERIMENTS and key not in sources, "source model/seed")
        require(ident["experiment_id"] == EXPERIMENTS[key], "source experiment")
        sources[key] = source
        require(len(source["snapshot"]) == (24 if model == "full" else 20), "snapshot count")
        for path, value in source["snapshot"].items():
            require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "snapshot hash")
            require(path not in merged or merged[path] == value, "snapshot collision")
            merged[path] = value
        for relative, field in (("checkpoints/best.pt", "best_checkpoint_sha256"),
                                ("predictions/test.csv", "test_predictions_sha256")):
            require(source["snapshot"][source["source_dir"] + "/" + relative] == source[field], "source binding")
    require(len(merged) == 132 and merged == provenance["input_snapshot"], "snapshot union")
    for seed in SEEDS:
        full, off = sources[("full", seed)], sources[("disagreement_weighting_off", seed)]
        fi, oi = full["identity"], off["identity"]
        require(fi["project_commit"] == oi["full_model_training_commit"] ==
                "e1b174490245c616e355ef7c0c6f7c3c239d3acd", "training source")
        require(fi["completed"] is True and fi["run_mode"] == "final" and fi["test_used_for_selection"] is False,
                "full completion/selection")
        require(fi["model"] == "ConsenDxFormer" and fi["class_order"] == list(CLASSES), "full model/classes")
        require(fi["lambda_ordinal"] == oi["lambda_ordinal"] == .5 and
                fi["lambda_attention"] == oi["lambda_attention"] == .05, "lambdas")
        require(oi["variant"] == "disagreement_weighting" and oi["toggles"] == {
            "use_attention_alignment": True, "use_disagreement_weighting": False,
            "use_hierarchical_fusion": True, "use_ordinal_head": True}, "OFF variant/toggles")
        require(oi["full_model_reference"] == dict(source_dir=full["source_dir"], descriptor=fi,
                best_checkpoint_sha256=full["best_checkpoint_sha256"]), "OFF full reference")
        for field in ("cohort_view", "manifest_bundle_sha256", "supervision_sha256",
                      "ordered_supervision_sha256", "sample_count", "phase3_input_fingerprint"):
            require(fi[field] == oi["inputs"][field], f"source cohort binding: {field}")
        require(fi["cohort_view"] == "paper_aligned_multi_reader_2_to_4" and fi["sample_count"] == 1133,
                "source cohort")
        require(fi["input_lock_sha256"] == oi["input_lock_sha256"], "input lock")
        require(fi["test_sample_count"] == oi["inputs"]["split_counts"]["test"] == 183, "source test count")
        require(fi["artifact_manifest_sha256"] == full["snapshot"][full["source_dir"] + "/artifact_manifest.json"],
                "source manifest")


def load_archive(path):
    """Integrity only: no new groups/statistics; pins cannot be overridden by callers."""
    archive = Path(path).read_bytes()
    require(digest(archive) == ARCHIVE_SHA256, "archive SHA-256")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        entries = zipped.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)) == 10, "archive duplicate/member count")
        for entry in entries:
            parts = PurePosixPath(entry.filename)
            require(not parts.is_absolute() and parts.parts[0] == EVALUATION_ID and
                    ".." not in parts.parts and "\\" not in entry.filename and
                    not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16), "unsafe archive member")
        require(set(names) == {EVALUATION_ID + "/" + member for member in MEMBERS}, "archive inventory")
        require(zipped.testzip() is None, "archive CRC")
        data = {member: zipped.read(EVALUATION_ID + "/" + member) for member in MEMBERS}
    for member, expected in PINS.items():
        require(digest(data[member]) == expected, f"pinned member: {member}")
    manifest = read_json(data["artifact_manifest.json"])
    completion = read_json(data["COMPLETED.json"])
    provenance = read_json(data["provenance/provenance.json"])
    evaluation = read_json(data["evaluation.json"])
    require(set(manifest["files"]) == MEMBERS - {"artifact_manifest.json", "COMPLETED.json"}, "manifest inventory")
    for name, expected in manifest["files"].items():
        require(digest(data[name]) == expected, f"payload hash: {name}")
    require(sha256_json(manifest["files"]) == manifest["bundle_sha256"] == PAYLOAD_DIGEST, "payload digest")
    require(digest(data["artifact_manifest.json"]) == completion["artifact_manifest_sha256"], "completion linkage")
    require(completion["evaluation_id"] == manifest["evaluation_id"] == evaluation["evaluation_id"] ==
            EVALUATION_ID == "phase4-evaluation-" + sha256_json(provenance)[:16], "evaluation identity")
    require(completion["source_artifacts_unchanged"] is True and completion["training_performed"] is False
            and completion["exploratory_post_hoc"] is True, "completion flags")
    require(evaluation["analysis"] == "uncertainty", "evaluation analysis")
    validate_bindings(provenance)
    rows = read_csv(data["tables/uncertainty_per_nodule_per_seed.csv"])
    ledger = validate_rows(rows, read_csv(data["tables/paired_per_nodule_per_seed.csv"]))
    return {"archive": archive, "rows": rows, "cohort": ledger, "provenance": provenance}
