#!/usr/bin/env python3
"""Standalone standard-library, allowlisted TRAIN/VALIDATION diagnostic extraction.

Never imports project code, model libraries, YAML evaluators or network clients.
Exit 0 = completed diagnostic; 2 = partial/blocked diagnostic; 1 = fatal error.
No audit outcome authorizes training. Mixed CSV bytes are hashed, test rows ignored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

VERSION = "performance_v2_diagnostics_v1"
CLASSES = ("benign", "indeterminate", "malignant")
SEEDS = (42, 43, 44)
VARIANTS = ("B1", "full", "attention_alignment", "disagreement_weighting",
            "ordinal_auxiliary", "hierarchical_fusion")
ROOTS = ("artifacts/phase3/experiments", "artifacts/phase4/experiments",
         "artifacts/phase4/ablations/runs")
ALLOW = frozenset({"COMPLETED.json", "artifact_manifest.json", "history/epoch_history.json",
    "provenance/run.json", "config/experiment_identity.json", "config/input_metadata.csv",
    "config/phase1_effective.yaml", "config/phase3_effective.yaml", "metrics/val.json",
    "config/phase4_config.json", "config/b1_anchor_config.json", "config/effective_config.json",
    "config/run_identity.json", "config/input_lock.json", "config/input_supervision.csv"})
SHA = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
MAX_BYTES = 32 * 1024 * 1024
NOT_RECORDED = "NOT_RECORDED"
FROZEN_SPLIT = "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
FROZEN_BASELINE = "0e5bcadcadfce5ea99df8484d23705d7bd4d778a"


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_load(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(AuditError("Nonfinite JSON")))


def safe_path(path):
    """Reject every existing symlink/reparse ancestor, before resolving/reading."""
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents), path]:
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not
                    (getattr(info, "st_file_attributes", 0) & 0x400), "Symlink/reparse path: " + str(part))
    return path


def number(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), "Expected finite numeric metric")
    return float(value)


class Reader:
    """Only this class opens source payloads. No manifest-driven arbitrary traversal."""
    def __init__(self):
        self.records = {}

    def read(self, root, relative, expected=None):
        require(relative in ALLOW, "Forbidden source read: " + relative)
        root = safe_path(root)
        path = safe_path(root / relative)
        require(path.is_relative_to(root) and path.is_file(), "Missing/unsafe source: " + str(path))
        require(path.stat().st_size <= MAX_BYTES, "Source exceeds bounded size limit")
        data = path.read_bytes()
        require(len(data) <= MAX_BYTES, "Source exceeds bounded size limit")
        actual = digest(data)
        if expected is not None:
            require(isinstance(expected, str) and SHA.fullmatch(expected) and actual == expected,
                    "Allowlisted payload hash mismatch: " + str(path))
        record = {"path": str(path), "relative": relative, "sha256": actual,
                  "size_bytes": len(data), "recorded_manifest_sha256": expected,
                  "fresh_bytes_checked": True, "full_bundle_verified": False}
        key = str(path)
        if key in self.records:
            require(self.records[key]["sha256"] == actual, "Input changed during read: " + key)
        else:
            self.records[key] = record
        return data

    def recheck(self):
        for record in list(self.records.values()):
            root = Path(record["path"])
            for _ in PurePosixPath(record["relative"]).parts:
                root = root.parent
            self.read(root, record["relative"], record["sha256"])


def git_state(repo):
    result = {}
    for field, args in (("branch", ["branch", "--show-current"]),
                        ("head", ["rev-parse", "HEAD"]),
                        ("status", ["status", "--porcelain=v1", "--untracked-files=all"])):
        command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo), *args]
        p = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        result[field] = p.stdout.strip()
    return result


def discover(repo):
    """Three fixed roots, one level, <=500 children each; never recurse into runs."""
    candidates = []
    for relative in ROOTS:
        root = safe_path(repo / relative)
        if not root.exists():
            continue
        entries = sorted(root.iterdir())
        require(len(entries) <= 500, "Inventory root exceeds bound")
        for entry in entries:
            safe_path(entry)
            if entry.is_dir():
                candidates.append(entry)
    return candidates


def load_run(reader, directory):
    """Validate receipts, identity and subset hashes without visiting tensor paths."""
    marker = json_load(reader.read(directory, "COMPLETED.json"))
    manifest_bytes = reader.read(directory, "artifact_manifest.json")
    manifest = json_load(manifest_bytes)
    require(marker.get("artifact_manifest_sha256") == digest(manifest_bytes), "Completion/manifest binding differs")
    require(marker.get("artifact_manifest", "artifact_manifest.json") == "artifact_manifest.json", "Unsafe manifest pointer")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Missing manifest files map")
    for key, value in files.items():
        p = PurePosixPath(key)
        require(not p.is_absolute() and ".." not in p.parts and "\\" not in key
                and ":" not in key and str(p) == key and SHA.fullmatch(str(value)), "Unsafe manifest entry")
    require(digest(canonical(files)) == manifest.get("bundle_sha256"), "Canonical files-map digest differs")
    exp = marker.get("experiment_id")
    require(exp == manifest.get("experiment_id") == directory.name, "Directory/receipt identity mismatch")
    def read(name):
        require(name in files, "Unbound required payload: " + name)
        return reader.read(directory, name, files[name])
    off = manifest.get("schema") == "phase4_ablation_artifacts_v1"
    baseline = "config/experiment_identity.json" in files
    config_records = {}
    id_path = "config/experiment_identity.json" if baseline else "config/run_identity.json"
    identity = json_load(read(id_path))
    run = json_load(read("provenance/run.json"))
    require(identity.get("experiment_id") == exp and run.get("identity") == identity,
            "Run/identity binding differs")
    payload = {k: v for k, v in identity.items() if k not in ("experiment_id", "digest")}
    if baseline:
        payload["schema_version"] = 1
    require(digest(canonical(payload)) == identity.get("digest"), "Content-addressed identity mismatch")
    seed = identity.get("seed")
    require(type(seed) is int, "Invalid seed")
    variant = identity.get("baseline_variant") if baseline else identity.get("variant", "full")
    commit = identity.get("ablation_code_commit" if off else "project_commit")
    require(isinstance(commit, str) and GIT_SHA.fullmatch(commit), "Missing execution commit")
    if baseline:
        require(manifest.get("schema_version") == 1 and marker.get("schema_version") == 1,
                "Unknown B1 manifest schema")
        require(marker.get("synthetic_dry_run") is False and run.get("mode") == "real"
                and run.get("synthetic_non_research") is False, "Not a real baseline run")
        require(run.get("research_results_eligible") is True and manifest.get("research_results_eligible") is True
                and marker.get("research_results_eligible") is True, "Baseline not research eligible")
        require(run.get("execution_state") == "completed", "Baseline execution incomplete")
        order = run.get("prediction_policy", {}).get("class_order")
        config = None
        for name in ("config/phase1_effective.yaml", "config/phase3_effective.yaml"):
            config_records[name] = read(name).decode("utf-8")  # Opaque, never evaluated as YAML.
        metadata_path = "config/input_metadata.csv"
        require(files.get(metadata_path) == identity.get("metadata_sha256"), "Metadata byte binding differs")
    else:
        config = json_load(read("config/effective_config.json"))
        config_records["config/effective_config.json"] = config
        order = config.get("predictions", {}).get("class_order")
        require(config.get("data", {}).get("mode") == "real", "Not real Phase 4 data")
        require(config.get("experiment", {}).get("seed") == seed, "Config seed mismatch")
        require(config.get("dataset", {}).get("class_names") == list(CLASSES), "Dataset class order mismatch")
        if off:
            require(variant in VARIANTS[2:] and marker.get("schema") == "phase4_ablation_completion_v1"
                    and manifest.get("run_kind") == marker.get("run_kind") == identity.get("run_kind")
                    == "phase4_component_ablation", "Unknown OFF identity")
            require(run.get("completed") is True and run.get("fresh_initialization") is True
                    and marker.get("runtime_training_explicitly_authorized") is True, "Incomplete OFF run")
            require(digest(canonical(config)) == identity.get("configuration_sha256"), "OFF config binding differs")
            toggles = {"attention_alignment": "use_attention_alignment", "disagreement_weighting": "use_disagreement_weighting",
                       "ordinal_auxiliary": "use_ordinal_head", "hierarchical_fusion": "use_hierarchical_fusion"}
            require(config.get("ablation") == identity.get("toggles") and
                    set(identity["toggles"]) == set(toggles.values()) and
                    all(v is (k != toggles[variant]) for k, v in identity["toggles"].items()), "OFF toggles differ")
        else:
            require(manifest.get("schema_version") == marker.get("schema_version") == 1, "Unknown full schema")
            require(identity.get("run_mode") == manifest.get("run_mode") == marker.get("run_mode") == "final",
                    "Not a final model (tuning excluded)")
            require(all(record.get("research_results_eligible") is True and record.get("test_results_eligible") is True
                        for record in (marker, manifest)), "Final receipt eligibility flags differ")
            require(run.get("execution_state") == "completed" and run.get("research_results_eligible") is True,
                    "Full run incomplete")
            require(config.get("ablation") == {"use_attention_alignment": True, "use_disagreement_weighting": True,
                    "use_ordinal_head": True, "use_hierarchical_fusion": True}, "Full model toggles differ")
            config_records["config/phase4_config.json"] = json_load(read("config/phase4_config.json"))
            config_records["config/b1_anchor_config.json"] = json_load(read("config/b1_anchor_config.json"))
            base_hash = digest(canonical({"schema_version": 1, "phase4": config_records["config/phase4_config.json"],
                                         "b1_anchor": config_records["config/b1_anchor_config.json"]}))
            configuration_hash = digest(canonical({"schema_version": 1, "base_combined_config_sha256": base_hash,
                "effective_config": config, "input_lock_sha256": identity.get("input_lock_sha256"),
                "tuning_protocol_sha256": config.get("execution", {}).get("tuning_protocol_sha256")}))
            require(configuration_hash == identity.get("configuration_sha256"), "Full effective configuration binding differs")
        lock_bytes = read("config/input_lock.json")
        require(digest(canonical(json_load(lock_bytes))) == identity.get("input_lock_sha256"), "Input lock canonical binding differs")
        config_records["config/input_lock.json"] = json_load(lock_bytes)
        metadata_path = "config/input_supervision.csv"
    require(order == list(CLASSES), "Missing/wrong/ambiguous class order")
    require(variant in (*VARIANTS, "B0"), "Unknown model variant")
    if baseline:
        require(identity.get("frozen_split_bundle_sha256") == FROZEN_SPLIT
                and identity.get("baseline_source_commit") == FROZEN_BASELINE, "Frozen baseline/split binding differs")
    else:
        inputs = identity.get("inputs", {}) if off else identity
        require(inputs.get("manifest_bundle_sha256") == FROZEN_SPLIT
                and inputs.get("cohort_view") == "paper_aligned_multi_reader_2_to_4"
                and inputs.get("expected_resolution") == 64, "Frozen Phase 4 input contract differs")
    history = json_load(read("history/epoch_history.json"))
    val = json_load(read("metrics/val.json"))
    metadata = read(metadata_path)
    return {"experiment_id": exp, "variant": variant, "seed": seed, "execution_commit": commit,
            "adapter": "phase3_v1" if baseline else ("phase4_off_v1" if off else "phase4_full_v1"),
            "identity": identity, "run": run, "config": config, "history": history,
            "val": val, "metadata": metadata, "metadata_path": metadata_path,
            "files": files, "config_records": config_records, "directory": str(directory), "manifest_sha256": digest(manifest_bytes)}


def metric_fields(metrics, baseline):
    require(isinstance(metrics, dict), "Metrics must be objects")
    if "class_order" in metrics:
        require(metrics["class_order"] == list(CLASSES), "Metric class order mismatch")
    mappings = {"accuracy": "accuracy", "loss": "loss",
                "macro_f1": "f1_macro" if baseline else "macro_f1",
                "macro_precision": "precision_macro" if baseline else "macro_precision",
                "macro_recall": "recall_macro" if baseline else "macro_recall"}
    if not baseline:
        mappings.update({k: k for k in ("total_loss", "diagnostic_loss", "ordinal_loss", "attention_loss")})
    fields = []
    for normalized, original in mappings.items():
        require(original in metrics, "Missing metric: " + original)
        value = number(metrics[original])
        require(value >= 0 and ("loss" in normalized or value <= 1), "Metric out of range")
        fields.append((normalized, original, value))
    for metric in ("precision", "recall", "f1", "support"):
        if baseline:
            values = metrics.get(metric + "_per_class")
            require(isinstance(values, list) and len(values) == 3, "Missing ordered class array")
        else:
            per_class = metrics.get("per_class", {})
            require(set(per_class) == set(CLASSES), "Unknown per-class schema")
            values = [per_class[c].get(metric) for c in CLASSES]
        for index, (c, value) in enumerate(zip(CLASSES, values)):
            value = number(value)
            require(value >= 0 and (value.is_integer() if metric == "support" else value <= 1), "Bad class metric")
            key = f"{metric}_per_class[{index}]" if baseline else f"per_class.{c}.{metric}"
            fields.append((f"{c}.{metric}", key, value))
    if not baseline:
        require(metrics["loss"] == metrics["total_loss"], "Loss aliases disagree")
    return fields


def histories(source):
    records = source["history"]
    require(isinstance(records, list) and records, "Empty/invalid history")
    require(all(type(r.get("epoch")) is int for r in records) and
            [r.get("epoch") for r in records] == list(range(1, len(records) + 1)), "Nonsequential history epochs")
    baseline = source["adapter"] == "phase3_v1"
    loss = "loss" if baseline else "total_loss"
    f1 = "f1_macro" if baseline else "macro_f1"
    rows = []
    for record in records:
        for split in ("train", "validation"):
            for normalized, key, value in metric_fields(record.get(split), baseline):
                rows.append({"experiment_id": source["experiment_id"], "adapter": source["adapter"],
                             "epoch": record["epoch"], "split": split, "metric": normalized,
                             "original_key": f"{split}.{key}", "value": value})
    low = min(records, key=lambda r: (r["validation"][loss], r["epoch"]))
    high = min(records, key=lambda r: (-r["validation"][f1], r["epoch"]))
    selected = (source["run"].get("selection", {}).get("best_epoch") if baseline else source["run"].get("best_epoch"))
    if selected is not None:
        require(type(selected) is int and selected == low["epoch"], "Receipt selection differs from first minimum history loss")
    val = source["val"]
    if source["adapter"] == "phase4_off_v1":
        require(set(val) == {"prediction_metrics", "within_variant_evaluation_losses"}, "Unknown nested OFF val schema")
        val = val["within_variant_evaluation_losses"]
    elif source["adapter"] == "phase4_full_v1":
        require("evaluation_losses" in val, "Missing full-model nested evaluation losses")
        val = val["evaluation_losses"]
    require(isinstance(val, dict), "Invalid validation receipt")
    # Full Phase 4 metrics are the stored epoch-style metrics; B1 may export
    # classification under a different metrics wrapper, so compare only known loss.
    if loss in val:
        require(math.isclose(number(val[loss]), number(low["validation"][loss]), rel_tol=1e-6, abs_tol=1e-8),
                "Validation receipt loss differs from history-selected epoch")
    selection = {"experiment_id": source["experiment_id"], "variant": source["variant"], "seed": source["seed"],
        "completed_epochs": len(records), "receipt_selected_epoch": selected,
        "selection_evidence": "receipt_and_history_not_tensor" if selected else "history_derived_criterion_only",
        "minimum_loss_epoch": low["epoch"], "maximum_macro_f1_epoch": high["epoch"],
        "minimum_loss_macro_f1": low["validation"][f1], "maximum_macro_f1": high["validation"][f1],
        "maximum_f1_epoch_loss": high["validation"][loss], "initial_validation_loss": records[0]["validation"][loss],
        "final_validation_loss": records[-1]["validation"][loss], "minimum_validation_loss": low["validation"][loss],
        "final_train_loss": records[-1]["train"][loss],
        "final_validation_minus_train_loss": records[-1]["validation"][loss] - records[-1]["train"][loss],
        "per_epoch_lr": NOT_RECORDED, "task_gradients": NOT_RECORDED, "fusion_gate_trace": NOT_RECORDED,
        "counterfactual_checkpoint_evaluated": False}
    if not baseline:
        lo, la = source["identity"].get("lambda_ordinal"), source["identity"].get("lambda_attention")
        lo, la = number(lo), number(la)
        for record in records:
            for split in ("train", "validation"):
                m = record[split]
                weighted = {"weighted_ordinal_loss": lo * m["ordinal_loss"], "weighted_attention_loss": la * m["attention_loss"]}
                require(math.isclose(m["total_loss"], m["diagnostic_loss"] + sum(weighted.values()), rel_tol=1e-5, abs_tol=1e-6),
                        "Auxiliary loss composition differs")
                for key, value in weighted.items():
                    rows.append({"experiment_id": source["experiment_id"], "adapter": source["adapter"], "epoch": record["epoch"],
                                 "split": split, "metric": key, "original_key": "derived_from_bound_lambda_and_stored_component", "value": value})
    return rows, selection


def metadata_rows(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    names = reader.fieldnames or []
    require(len(names) == len(set(names)) and {"split", "patient_id", "nodule_id", "series_instance_uid", "label"} <= set(names),
            "Missing/duplicate metadata columns")
    rows, seen, patients = [], set(), {"train": set(), "val": set()}
    for line, row in enumerate(reader, 2):
        # Deliberately inspect only split before touching labels/readers/identities.
        split = row.get("split")
        if split == "test":
            continue
        require(split in ("train", "val"), "Unexpected metadata split")
        require(None not in row and all(row.get(k) is not None for k in names), "Malformed CSV row")
        keep = {k: row[k] for k in ("patient_id", "nodule_id", "series_instance_uid", "split", "label")}
        require(all(v and v == v.strip() for v in keep.values()) and keep["label"] in CLASSES, "Invalid train/val identity or label")
        require(keep["nodule_id"] not in seen, "Duplicate nodule identity")
        seen.add(keep["nodule_id"]); patients[split].add(keep["patient_id"])
        keep["source_csv_line"] = line
        for name in ("n_valid_malignancy_ratings", "q_benign", "q_indeterminate", "q_malignant", "disagreement_entropy"):
            if name in row:
                keep[name] = row[name]
        rows.append(keep)
    require(not patients["train"] & patients["val"], "Train/validation patient overlap")
    require(all(patients.values()), "Missing train or validation rows")
    return rows


def summarize_metadata(rows):
    result = {}
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        counts = {c: sum(r["label"] == c for r in subset) for c in CLASSES}
        readers = {}
        for r in subset:
            value = r.get("n_valid_malignancy_ratings")
            if value is None:
                key = NOT_RECORDED
            else:
                require(value in ("2", "3", "4"), "Invalid frozen reader-count eligibility")
                key = value
            readers[key] = readers.get(key, 0) + 1
        result[split] = {"nodules": len(subset), "patients": len({r["patient_id"] for r in subset}),
                         "class_counts": counts, "reader_counts": readers}
    counts = result["train"]["class_counts"]
    require(all(counts.values()), "Training class absent")
    result["balanced_weights"] = [result["train"]["nodules"] / (3 * counts[c]) for c in CLASSES]
    result["train_validation_patient_overlap"] = 0
    return result


def conflict_rows(rows):
    result = []
    for row in rows:
        if row["split"] != "train":
            continue
        required = ["q_" + c for c in CLASSES] + ["disagreement_entropy"]
        require(all(k in row for k in required), "Missing training supervision columns: " + ",".join(required))
        try:
            q = [float(row["q_" + c]) for c in CLASSES]; u = float(row["disagreement_entropy"])
        except (TypeError, ValueError) as exc:
            raise AuditError("Malformed training q/u") from exc
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in [*q, u]) and abs(sum(q) - 1) <= 1e-6,
                "Invalid training q/u; no renormalization")
        qy = q[CLASSES.index(row["label"])]; maximum = max(q)
        tied = sum(abs(x - maximum) <= 1e-8 for x in q)
        category = "hard_not_maximal" if qy < maximum - 1e-8 else ("maximal_tie" if tied > 1 else "unique_agreement")
        bin_index = min(int(u * 4), 3)
        result.append({**row, "q_y": qy, "one_minus_q_y": 1 - qy, "discordance_proxy": u * (1 - qy),
                       "category": category, "u_bin": ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]")[bin_index]})
    return result


def summarize_conflicts(rows, experiment_id):
    summaries = []
    for label in ("ALL", *CLASSES):
        for group in ("ALL", "[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]"):
            selected = [r for r in rows if (label == "ALL" or r["label"] == label) and (group == "ALL" or r["u_bin"] == group)]
            record = {"experiment_id": experiment_id, "label": label, "u_bin": group, "n": len(selected),
                      "status": "ok" if selected else "empty"}
            for field in ("q_y", "one_minus_q_y", "discordance_proxy"):
                record["mean_" + field] = sum(r[field] for r in selected) / len(selected) if selected else None
            for field in ("hard_not_maximal", "maximal_tie", "unique_agreement"):
                record[field] = sum(r["category"] == field for r in selected)
            summaries.append(record)
    return summaries


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def extract(repo, output_parent, *, argv=None):
    repo, output_parent = safe_path(repo), safe_path(output_parent)
    require(repo.is_dir() and output_parent.is_dir(), "Repository/output parent must already exist")
    require(not output_parent.is_relative_to(repo) and not repo.is_relative_to(output_parent), "Output/source directory overlap")
    before = git_state(repo)
    reader = Reader(); issues = []; inventory = []; epoch_rows = []; selections = []; metadata = {}; conflicts = []; traces = []
    identities = {}; slots = {}; cohorts = {}; configs = {}; saved_val = {}; supervision = {}; output = None
    try:
        candidates = discover(repo)
        for directory in candidates:
            entry = {"source_dir": str(directory), "status": "unverified"}
            inventory.append(entry)
            if (directory / "FAILED.json").exists() or not (directory / "COMPLETED.json").exists():
                entry["status"] = "failed" if (directory / "FAILED.json").exists() else ("synthetic" if (directory / "DRY_RUN_COMPLETED.json").exists() else "incomplete")
                continue
            try:
                marker = json_load(reader.read(directory, "COMPLETED.json"))
                if marker.get("run_mode") == "tuning" or marker.get("synthetic_dry_run") is True:
                    entry.update(status="tuning" if marker.get("run_mode") == "tuning" else "synthetic",
                                 experiment_id=marker.get("experiment_id"), eligibility="excluded_from_diagnostics_marker_only_not_validated")
                    continue
                source = load_run(reader, directory)
                entry.update({k: source[k] for k in ("experiment_id", "variant", "seed", "execution_commit", "adapter", "manifest_sha256")})
                identities[source["experiment_id"]] = source["identity"]
                configs[source["experiment_id"]] = source["config_records"]
                if source["variant"] == "B0" or source["seed"] not in SEEDS:
                    entry["status"] = "historical_context_not_primary"
                    continue
                slot = (source["variant"], source["seed"])
                slots.setdefault(slot, []).append(source["experiment_id"])
                rows, selection = histories(source)
                saved_val[source["experiment_id"]] = source["val"]
                samples = metadata_rows(source["metadata"])
                summary = summarize_metadata(samples)
                for split, n, patients in (("train", 807, 437), ("val", 143, 84)):
                    if (summary[split]["nodules"], summary[split]["patients"]) != (n, patients):
                        issues.append({"source": str(directory), "issue": f"{split} cohort counts differ from expected {n}/{patients}; investigate, do not filter"})
                common = [{k: r[k] for k in ("patient_id", "nodule_id", "series_instance_uid", "split", "label")} for r in samples]
                cohorts[source["experiment_id"]] = sorted(common, key=lambda r: r["nodule_id"])
                weights = source["run"].get("class_weights_from_train_hard_labels", source["run"].get("class_weights"))
                summary["recorded_class_weights"] = weights if weights is not None else NOT_RECORDED
                if weights is not None:
                    require(isinstance(weights, list) and len(weights) == 3 and all(math.isclose(number(a), b, rel_tol=1e-6, abs_tol=1e-8)
                            for a, b in zip(weights, summary["balanced_weights"])), "Stored train class weights differ")
                summary["train_val_identity_sha256"] = digest(canonical(cohorts[source["experiment_id"]]))
                metadata[source["experiment_id"]] = summary
                epoch_rows.extend(rows); selections.append(selection)
                if source["variant"] != "B1":
                    proxies = conflict_rows(samples)
                    supervision[source["experiment_id"]] = sorted(
                        [{k: r.get(k) for k in ("nodule_id", "q_benign", "q_indeterminate", "q_malignant", "disagreement_entropy", "n_valid_malignancy_ratings")}
                         for r in samples], key=lambda r: r["nodule_id"])
                    conflicts.extend(summarize_conflicts(proxies, source["experiment_id"]))
                else:
                    proxies = []
                by_line = {r["source_csv_line"]: r for r in proxies}
                for row in samples:
                    traces.append({"experiment_id": source["experiment_id"], "source_file": source["metadata_path"],
                                   **by_line.get(row["source_csv_line"], row)})
                entry["status"] = "subset_validated"
                if source["config"] is None:
                    issues.append({"source": str(directory), "issue": "B1 YAML-only hyperparameters pending desktop parsing; ordered class metrics bound via run prediction_policy"})
            except (AuditError, KeyError, TypeError, ValueError, OSError) as error:
                entry["status"] = "blocked"; entry["reason"] = str(error)
                issues.append({"source": str(directory), "issue": str(error)})
        for variant in VARIANTS:
            for seed in SEEDS:
                matches = slots.get((variant, seed), [])
                if len(matches) != 1:
                    issues.append({"slot": [variant, seed], "issue": "missing_or_multiple_candidates_no_score_selection", "candidates": matches})
        cohort_values = list(cohorts.values())
        if cohort_values and any(c != cohort_values[0] for c in cohort_values[1:]):
            issues.append({"issue": "exact train/validation identity or label mismatch across sources"})
        supervision_values = list(supervision.values())
        if supervision_values and any(s != supervision_values[0] for s in supervision_values[1:]):
            issues.append({"issue": "train/validation supervision text differs across full/OFF sources; investigate exact binding, no reconciliation by averaging"})
        reader.recheck()
        after = git_state(repo)
        require(before == after, "Git state changed during extraction")
        snapshot = list(reader.records.values())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = safe_path(output_parent / f"audit-{stamp}-{digest(canonical(snapshot))[:12]}")
        destination.mkdir(exist_ok=False)
        output = destination  # Exception handling may write only into this newly owned directory.
        write_csv(output / "run_inventory.csv", inventory)
        write_csv(output / "epoch_history_long.csv", epoch_rows)
        write_csv(output / "checkpoint_selection_diagnostics.csv", selections)
        write_csv(output / "train_target_conflict_summary.csv", conflicts)
        write_csv(output / "train_val_source_rows.csv", traces)
        write_json(output / "train_val_metadata_summary.json", metadata)
        write_json(output / "saved_validation_metrics.json", saved_val)
        write_json(output / "source_manifest.json", {"files": snapshot, "recorded_identities": identities,
                   "configuration_snapshots": configs,
                   "scope": "allowlisted payloads only; recorded checkpoint/other hashes not freshly verified"})
        write_json(output / "missing_or_unverified.json", {"issues": issues,
            "always_unverified": ["checkpoint tensors", "full original bundle", "live execution claims", "per-epoch LR, gradients and gate traces"]})
        # Recheck after writing, before success/partial status and final marker.
        reader.recheck(); require(git_state(repo) == before, "Git changed during export")
        audit = {"schema": VERSION, "created_utc": stamp, "status": "partial" if issues else "completed",
            "script_sha256": digest(Path(__file__).read_bytes()), "argv": argv or [], "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "git_before": before, "git_after": after,
            "model_forwards": 0, "training_performed": False, "checkpoint_tensors_loaded": False,
            "test_performance_computed": False, "development_diagnostics_computed": True,
            "full_source_bundle_verified": False, "mixed_containers_hashed_test_rows_discarded": True,
            "pilot_execution_authorized": False, "source_changes_detected": False}
        write_json(output / "audit.json", audit)
        (output / "limitations.md").write_text(
            "# Diagnostic limits\nTrain/validation only. Mixed CSV containers were read/hashed; test rows discarded before label interpretation. "
            "No test predictions/metrics, checkpoint tensors, images or array archives opened. Subset integrity is not full-bundle verification. "
            "History counterfactuals do not identify available alternative checkpoints. Augmented online train versus deterministic validation gaps are not definitive overfitting. "
            "Target discordance is not measured gradient conflict. LR/gradient/gate traces NOT_RECORDED. No pilot release.\n", encoding="utf-8")
        (output / "README.md").write_text("# Performance-v2 development diagnostics\nInspect audit.json status and missing_or_unverified.json first. "
            "Partial does not mean all diagnoses passed. Retain all candidates; do not select by scores. Return this report for desktop review before Gate B/C.\n", encoding="utf-8")
        files = {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir())}
        write_json(output / "artifact_manifest.json", {"schema": VERSION, "files": files, "bundle_sha256": digest(canonical(files))})
        write_json(output / "AUDIT_COMPLETED.json", {"schema": VERSION, "audit_completed": True,
            "status": audit["status"], "research_certification": False, "pilot_execution_authorized": False,
            "artifact_manifest_sha256": digest((output / "artifact_manifest.json").read_bytes())})
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.iterdir()):
                z.write(path, output.name + "/" + path.name)
        return output, audit["status"]
    except BaseException as error:
        if output is not None and not (output / "AUDIT_COMPLETED.json").exists():
            write_json(output / "AUDIT_FAILED.json", {"schema": VERSION, "audit_completed": False,
                       "error_type": type(error).__name__, "error": str(error), "pilot_execution_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    try:
        output, status = extract(args.repo, args.output_parent, argv=sys.argv[1:])
    except (AuditError, OSError, ValueError, subprocess.SubprocessError) as error:
        print("AUDIT FAILED; no completion claim: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "zip": str(output.with_suffix('.zip')), "status": status}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
