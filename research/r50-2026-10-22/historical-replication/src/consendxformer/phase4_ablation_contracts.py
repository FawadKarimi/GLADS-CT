"""Separate execution identities, checkpoints and predictions for component ablations.

Historical final/tuning contracts and non-executing specifications stay unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER, evaluate_predictions
from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_bytes
from .phase3_predictions import stable_softmax
from .phase4_ablations import (
    FINAL_SEEDS, FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256,
    FROZEN_PHASE4_INPUT_LOCK_SHA256, FROZEN_SOURCE_FINAL_MODEL_COMMIT,
    phase4_ablation_specifications, phase4_ablation_protocol_record,
    validate_phase4_ablation_specification, _values_are_exactly_equal,
)
from .phase4_checkpoints import select_phase4_best_epoch
from .phase4_evaluation_metrics import calibration_metrics

RUN_KIND = "phase4_component_ablation"
NAMESPACE = "artifacts/phase4/ablations"
ID_FIELDS = ("patient_id", "nodule_id", "series_instance_uid", "split", "true_class")
LOGITS = tuple(f"logit_{label}" for label in CLASS_ORDER)
PROBS = tuple(f"prob_{label}" for label in CLASS_ORDER)
PREDICTION_COLUMNS = (*ID_FIELDS, "predicted_class", *LOGITS, *PROBS,
                      "run_kind", "variant", "seed", "lambda_ordinal", "lambda_attention", "experiment_id")


@dataclass(frozen=True)
class AblationRunSpec:
    variant: str
    seed: int

    def specification(self):
        if type(self.seed) is not int or self.seed not in FINAL_SEEDS:
            raise ValueError("Ablation seed must be exactly 42, 43 or 44")
        matches = [s for s in phase4_ablation_specifications() if s.name == self.variant]
        if len(matches) != 1:
            raise ValueError("Unknown single-component ablation variant")
        return validate_phase4_ablation_specification(matches[0])

    def to_dict(self):
        spec = self.specification()
        return {"run_kind": RUN_KIND, "variant": spec.name, "seed": self.seed,
                "toggles": spec.toggles, "lambda_ordinal": spec.lambda_ordinal,
                "lambda_attention": spec.lambda_attention, "off_semantics": spec.off_semantics}


def execution_protocol():
    return {"schema_version": 1, "protocol": "phase4_component_execution_v1",
            "historical_specifications": phase4_ablation_protocol_record(),
            "chronology": "component evaluations occur after complete-model test and supplementary attention results were seen",
            "preregistered": False, "test_driven_redesign_or_tuning_permitted": False,
            "runtime_authorization_required": True, "initialization": "fresh_seeded_weights_not_finetuning",
            "loss_composition_check": {"atol": 1e-6, "rtol": 1e-5},
            "selection": "minimum within-run validation total loss under remaining authorized terms; never compare total loss across variants",
            "first_group": {"variant": "attention_alignment", "seeds": list(FINAL_SEEDS)},
            "supplementary_reference_id": "phase4-evaluation-22cb656eabd040c2",
            "class_order": list(CLASS_ORDER), "prior_recomputation_permitted": False}


def effective_ablation_config(bundle, spec: AblationRunSpec):
    """No arbitrary overrides: change only the declared controls and run metadata."""
    if bundle.combined_config_sha256 != FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256:
        raise ValueError("Complete-model base configuration differs from frozen hash")
    definition = spec.specification()
    config = deepcopy(bundle.phase4_config)
    if not all(config["ablation"].values()) or config["model"]["use_spatial_attention"] is not True:
        raise ValueError("Frozen complete model must enable all components and DSA")
    config["ablation"] = definition.toggles
    config["model"]["use_hierarchical_fusion"] = definition.use_hierarchical_fusion
    config["experiment"]["seed"] = spec.seed
    config["experiment"]["output_root"] = f"{NAMESPACE}/runs"
    config["execution"] = {**spec.to_dict(), "input_lock_sha256": FROZEN_PHASE4_INPUT_LOCK_SHA256,
                           "fresh_initialization": True, "evaluate_test_after_best_reload": True}
    return config


def make_identity(spec, config, commit, readiness, reference):
    fields = ("manifest_bundle_sha256", "supervision_sha256", "ordered_supervision_sha256",
              "phase3_input_fingerprint", "cohort_view", "expected_resolution", "sample_count", "split_counts")
    payload = {"schema_version": 1, **spec.to_dict(), "ablation_code_commit": commit,
               "full_model_training_commit": FROZEN_SOURCE_FINAL_MODEL_COMMIT,
               "base_combined_config_sha256": FROZEN_PHASE4_BASE_COMBINED_CONFIG_SHA256,
               "input_lock_sha256": FROZEN_PHASE4_INPUT_LOCK_SHA256,
               "configuration_sha256": sha256_json(config), "protocol_sha256": sha256_json(execution_protocol()),
               "inputs": {field: readiness[field] for field in fields}, "full_model_reference": reference}
    digest = sha256_json(payload)
    return {**payload, "digest": digest, "experiment_id": f"phase4-ablation-{spec.variant}-s{spec.seed}-{digest[:16]}"}


def validate_identity(identity, config, bundle):
    spec = AblationRunSpec(identity["variant"], identity["seed"])
    if not _values_are_exactly_equal(config, effective_ablation_config(bundle, spec)):
        raise ValueError("Ablation effective configuration contains unauthorized overrides")
    commit = identity["ablation_code_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Ablation code commit is not an exact SHA")
    expected = make_identity(spec, config, commit, identity["inputs"], identity["full_model_reference"])
    if not _values_are_exactly_equal(identity, expected):
        raise ValueError("Ablation identity differs from its content address")
    return spec


def identity_rows(frame, *, label="label"):
    fields = ["patient_id", "nodule_id", "series_instance_uid", "split", label]
    if frame.empty or not frame.columns.is_unique or not set(fields).issubset(frame):
        raise ValueError("Missing exact sample identities")
    values = frame[fields].rename(columns={label: "true_class"})
    for field in ID_FIELDS:
        if not values[field].map(lambda v: isinstance(v, str) and bool(v.strip()) and v == v.strip()).all():
            raise ValueError("Sample identities must be canonical nonempty text")
    if values["nodule_id"].duplicated().any() or not values["true_class"].isin(CLASS_ORDER).all():
        raise ValueError("Duplicate sample or noncanonical hard label")
    return values.to_dict("records")


def build_predictions(frame, logits, identity):
    values = np.asarray(logits, dtype=np.float64)
    probabilities = stable_softmax(values)
    rows = identity_rows(frame)
    if values.shape != (len(rows), 3):
        raise ValueError("Prediction row count differs from exact cohort")
    for index, row in enumerate(rows):
        row.update(predicted_class=CLASS_ORDER[int(probabilities[index].argmax())])
        row.update({name: float(values[index, i]) for i, name in enumerate(LOGITS)})
        row.update({name: float(probabilities[index, i]) for i, name in enumerate(PROBS)})
        row.update({field: identity[field] for field in PREDICTION_COLUMNS[-6:]})
    return validate_predictions(pd.DataFrame(rows, columns=PREDICTION_COLUMNS), frame, identity)


def validate_predictions(predictions, frame, identity):
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("Separate ablation prediction schema required")
    if identity_rows(predictions, label="true_class") != identity_rows(frame):
        raise ValueError("Predictions differ from exact ordered identities/labels")
    if predictions["split"].nunique() != 1 or not predictions["split"].isin(("val", "test")).all():
        raise ValueError("Prediction export must contain exactly one val/test split")
    for field in PREDICTION_COLUMNS[-6:]:
        if not predictions[field].eq(identity[field]).all():
            raise ValueError(f"Prediction provenance differs: {field}")
    values = predictions[list(LOGITS)].to_numpy(dtype=float)
    expected = stable_softmax(values)
    actual = predictions[list(PROBS)].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-12, rtol=1e-12):
        raise ValueError("Probabilities disagree with exported logits")
    if predictions["predicted_class"].tolist() != [CLASS_ORDER[int(i)] for i in expected.argmax(axis=1)]:
        raise ValueError("Predicted classes disagree with probabilities")
    return predictions.copy(deep=True)


def prediction_metrics(frame):
    return {"classification": evaluate_predictions(frame, identifier_column="nodule_id",
            true_label_column="true_class", predicted_class_column="predicted_class", probability_columns=PROBS),
            "calibration": calibration_metrics(frame["true_class"].tolist(), frame[list(PROBS)].to_numpy(dtype=float))}


def validate_loss_metrics(metrics, identity):
    values = {}
    for field in ("loss", "total_loss", "diagnostic_loss", "ordinal_loss", "attention_loss"):
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid ablation loss metric: {field}")
        values[field] = float(value)
    if values["loss"] != values["total_loss"]:
        raise ValueError("Ablation validation-loss alias differs from total loss")
    for toggle, field in (("use_ordinal_head", "ordinal_loss"), ("use_attention_alignment", "attention_loss")):
        if not identity["toggles"][toggle] and values[field] != 0.:
            raise ValueError("Disabled auxiliary contributed a nonzero loss")
    expected = values["diagnostic_loss"] + identity["lambda_ordinal"] * values["ordinal_loss"] + identity["lambda_attention"] * values["attention_loss"]
    if not np.isclose(values["total_loss"], expected, **execution_protocol()["loss_composition_check"]):
        raise ValueError("Reported ablation loss composition differs from authorized terms")


def checkpoint_payload(identity, *, kind, epoch, best_epoch, best_loss, validation, model, optimizer, scheduler):
    return {"schema": "phase4_ablation_checkpoint_v1", "run_kind": RUN_KIND, "identity": deepcopy(identity),
            "checkpoint_kind": kind, "epoch": epoch, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "validation_metrics": deepcopy(validation),
            "selection": "within_run_minimum_validation_total_loss", "class_order": list(CLASS_ORDER),
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()}


def save_checkpoint(payload, path):
    import torch
    stream = io.BytesIO()
    torch.save(payload, stream)
    return atomic_write_bytes(path, stream.getvalue(), overwrite=True)


def load_checkpoint_pair(directory, identity, history, *, device="cpu"):
    """Disk snapshots only; never retain aliased model/optimizer state dictionaries."""
    import torch
    selection = select_phase4_best_epoch(history)
    by_epoch = {record["epoch"]: record["validation"] for record in history}
    records = []
    for kind, epoch in (("best", selection["epoch"]), ("last", history[-1]["epoch"])):
        record = torch.load(Path(directory) / f"{kind}.pt", map_location=device, weights_only=False)
        expected = {"schema": "phase4_ablation_checkpoint_v1", "run_kind": RUN_KIND, "identity": identity,
                    "checkpoint_kind": kind, "epoch": epoch, "best_epoch": selection["epoch"],
                    "best_validation_loss": selection["value"], "validation_metrics": by_epoch[epoch],
                    "selection": "within_run_minimum_validation_total_loss", "class_order": list(CLASS_ORDER)}
        if not isinstance(record, dict) or set(record) != {*expected, "model_state", "optimizer_state", "scheduler_state"}:
            raise ValueError("Invalid ablation checkpoint schema")
        for key, value in expected.items():
            if not _values_are_exactly_equal(record[key], value):
                raise ValueError(f"Persisted {kind} checkpoint differs: {key}")
        for key in ("model_state", "optimizer_state", "scheduler_state"):
            if not isinstance(record[key], Mapping) or not record[key]:
                raise ValueError(f"Persisted checkpoint lacks complete {key}")
        records.append(record)
    return tuple(records)


def checkpoint_metadata(record, path):
    return {key: value for key, value in record.items() if not key.endswith("_state")} | {
        "checkpoint_sha256": sha256_file(path), "checkpoint_file": Path(path).name}
