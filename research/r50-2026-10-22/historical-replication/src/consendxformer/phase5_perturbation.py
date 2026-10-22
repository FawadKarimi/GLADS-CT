"""Fixed-order, image-only insertion/deletion; no model loading or hypothesis tests."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .hashing import sha256_json
from .phase4_attention_evidence import IDENTITY, array_record
from .phase4_attention_runtime import _saved_replay
from .phase5_descriptive import PROPOSAL_SHA256, identities, unresolved_hypotheses

SCOPE = "insertion_deletion_only"
MODELS = ("ConsenDxFormer", "B1")
SEEDS = (42, 43, 44)
ORDERS = ("dsa", "prior", "hash_order")
DIRECTIONS = ("insertion", "deletion")
KEYS = tuple((model, seed) for model in MODELS for seed in SEEDS)
STEPS = 64
WARNING = (
    "Exploratory post-hoc insertion_deletion_only. Zero-baseline black tiles can be "
    "out of distribution; zero is a normalized input bound, not physical air. "
    "AUC describes fixed-target model response, not clinical causality, attention-loss "
    "contribution, diagnostic performance or superiority. Cross-model targets can differ. "
    "Comparisons are descriptive within-model only; seeds reuse the same patients. "
    "Macro-F1 remains primary. All 15 formal hypotheses remain unresolved; "
    "the full supplement and Stage A are incomplete."
)


def scheduled_budget(n_nodules=183):
    if type(n_nodules) is not int or n_nodules < 1:
        raise ValueError("Positive nodule count required")
    units = n_nodules * len(KEYS)
    return {"units": units, "unperturbed": units, "baseline": units,
            "interior": units * 378, "total": units * 380,
            "new_nonoriginal": units * 379, "curves": units * 6,
            "curve_positions": units * 390, "per_seed": n_nodules * 2 * 380}


def perturbation_protocol():
    return {
        "schema_version": 1, "protocol": "phase5_insertion_deletion_v1", "scope": SCOPE,
        "proposal_sha256_lf": PROPOSAL_SHA256, "proposal_base_commit": "e3c1a24d87d5e21662984a9d1a4e77d6e5b0f03a",
        "analysis_status": "exploratory_post_hoc", "preregistered": False,
        "models": list(MODELS), "seeds": list(SEEDS), "class_order": list(CLASS_ORDER),
        "test_nodules": 183, "test_patients": 99, "training_prior_nodules": 807,
        "attention_reference_id": "phase4-evaluation-22cb656eabd040c2",
        "input": {"shape": [183, 1, 64, 64], "dtype": "float32", "range": [0, 1]},
        "native_grid": [8, 8], "tile_shape": [8, 8], "steps": 64, "curve_positions": 65,
        "baseline": "float32_all_zero_in_existing_normalized_input_coordinates_both_directions",
        "orders": list(ORDERS), "directions": list(DIRECTIONS),
        "ranking": "descending_saved_value_then_ascending_flat_index_exact_ties_include_zeros",
        "hash_order": {"algorithm": "sha256", "prefix": "phase5-supplement-v1-order",
                       "fields": ["patient_id", "nodule_id", "series_instance_uid", "integer_flat_index"],
                       "json": "ensure_ascii=True,separators=(',',':'),UTF-8,no_newline",
                       "sort": "ascending_32_unsigned_digest_bytes_then_flat_index"},
        "target": "validated_saved_unperturbed_predicted_class_fixed_for_all_steps_orders_directions",
        "replay": {"atol": 1e-5, "rtol": 1e-4, "argmax_must_agree": True,
                   "global_barrier": "all_1098_pass_before_any_baseline_or_interior_forward"},
        "probability": {"dtype": "float32", "shape": "nonempty_N_by_3", "range": [0, 1],
                        "finite": True, "sum_dtype": "float64_private_copy", "atol": 1e-6,
                        "rtol": 0, "normalize": False, "clip": False, "finite_logits": True},
        "auc": {"rule": "trapezoid_on_k_over_64", "dtype": "float64", "normalization": "none",
                "readback_atol": 1e-12, "readback_rtol": 0},
        "summaries": {"nodule_weighted": True, "statistics": ["mean", "median", "q25", "q75"],
                      "quantile_method": "linear", "per_seed_summaries": 36,
                      "within_model_paired_summaries": 24,
                      "contrasts": ["dsa_minus_prior", "dsa_minus_hash_order"], "formal_tests": False},
        "execution": {"batch_size": 1, "eval": True, "frozen_parameters": True, "gradients": False,
                      "autocast": False, "deterministic_algorithms": True, "automatic_retries": 0,
                      "cache": "only_two_endpoints_within_each_unit_no_other_deduplication",
                      "unit_order": "model_then_seed_then_original_archive_row",
                      "interior_order": "unit_then_order_then_direction_then_step_1_through_63"},
        "successful_run_budget": scheduled_budget(), "synthetic_timing_evidence": None,
        "full_supplemental_completion": False, "stage_a_complete": False,
        "formal_tests_computed": False, "hypotheses": unresolved_hypotheses(),
        "excluded": ["pointing", "accuracy_intervals", "boundary", "uncertainty", "formal_testing",
                     "multiplicity", "patient_weighted_correctness", "calibration_campaign", "training"],
    }


def validate_probabilities(probabilities):
    """Check a private float64 sum; return unchanged float32 bytes and diagnostics."""
    raw = np.asarray(probabilities)
    if raw.dtype != np.dtype("float32") or raw.ndim != 2 or raw.shape[1:] != (3,) or not len(raw):
        raise ValueError("Probabilities must be nonempty float32 [N,3]")
    check = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(check).all() or (check < 0).any() or (check > 1).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    mass = check.sum(axis=1, dtype=np.float64)
    deviation = mass - 1.0
    if not np.allclose(mass, 1, rtol=0, atol=1e-6):
        raise ValueError("Raw probability mass differs by more than 1e-6")
    return raw.copy(), mass, deviation


def validate_outputs(output):
    if not isinstance(output, dict) or not {"logits", "probs"}.issubset(output):
        raise ValueError("Image predictor must return logits and probs")
    probs, mass, deviation = validate_probabilities(output["probs"])
    logits = np.asarray(output["logits"])
    if logits.dtype != np.dtype("float32") or logits.shape != probs.shape or not np.isfinite(logits).all():
        raise ValueError("Logits must be finite float32 with the same [N,3] shape")
    if len(probs) != 1:
        raise ValueError("Execution requires batch size one")
    return logits.copy(), probs, mass, deviation


def validate_images(images):
    value = np.asarray(images)
    if value.dtype != np.dtype("float32") or value.ndim != 4 or value.shape[1:] != (1, 64, 64) or not len(value):
        raise ValueError("Expected nonempty float32 [N,1,64,64] images")
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("Images must be finite in [0,1]")
    return value


def rank_native(value):
    value = np.asarray(value)
    if value.shape != (8, 8) or value.dtype.kind not in "fiu" or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Finite nonnegative native 8x8 map required")
    if not np.isclose(value.sum(dtype=np.float64), 1, atol=1e-6, rtol=0):
        raise ValueError("Native map must already have unit spatial mass")
    return np.lexsort((np.arange(64), -value.reshape(-1).astype(np.float64))).astype(np.int64)


def hash_order(identity):
    fields = [identity[k] for k in IDENTITY[:3]]
    if any(not isinstance(v, str) or not v or v.strip() != v for v in fields):
        raise ValueError("Exact nonempty textual ordering identities required")
    digests = [hashlib.sha256(json.dumps(["phase5-supplement-v1-order", *fields, j],
                ensure_ascii=True, separators=(",", ":")).encode("utf-8")).digest() for j in range(64)]
    order = np.array(sorted(range(64), key=lambda j: (digests[j], j)), dtype=np.int64)
    return order, np.stack([np.frombuffer(digest, dtype=np.uint8) for digest in digests])


def validate_permutation(order):
    value = np.asarray(order)
    if value.shape != (64,) or value.dtype.kind not in "iu" or not np.array_equal(np.sort(value), np.arange(64)):
        raise ValueError("A full permutation of all 64 native cells is required")
    return value


def tile_input(image, order, direction, step):
    """Public pure mapping oracle; no target mask is accepted by this API."""
    image = validate_images(np.asarray(image)[None])[0]
    order = validate_permutation(order)
    if direction not in DIRECTIONS or type(step) is not int or not 0 <= step <= 64:
        raise ValueError("Invalid direction/step")
    result = np.zeros_like(image) if direction == "insertion" else image.copy()
    for j in order[:step]:
        r, c = divmod(int(j), 8)
        result[:, 8*r:8*r+8, 8*c:8*c+8] = image[:, 8*r:8*r+8, 8*c:8*c+8] if direction == "insertion" else 0
    return result


def curve_auc(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1:] != (65,) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Every curve needs all 65 finite probabilities in [0,1]")
    return ((values[..., :-1] + values[..., 1:]) * (0.5 / 64)).sum(axis=-1, dtype=np.float64)


@dataclass
class CallLedger:
    attempted: dict = field(default_factory=lambda: dict(unperturbed=0, baseline=0, interior=0))
    accepted: dict = field(default_factory=lambda: dict(unperturbed=0, baseline=0, interior=0))
    replays: list = field(default_factory=list)
    barrier_passed: bool = False
    last_attempt: dict | None = None
    failure: dict | None = None

    def record(self):
        return {"attempted": dict(self.attempted), "accepted": dict(self.accepted),
                "total_attempted": sum(self.attempted.values()), "total_accepted": sum(self.accepted.values()),
                "global_replay_barrier_passed": self.barrier_passed, "automatic_retries": 0,
                "last_attempt": self.last_attempt, "failure": self.failure}


@dataclass(frozen=True)
class PerturbationResult:
    units: pd.DataFrame
    orders: dict
    curves: dict
    replay: dict
    baseline: dict
    ledger: CallLedger


def execute_curves(ids, images, maps, prior, predictions, predictors, *, source_binding,
                   source_records, ledger=None, on_barrier=None):
    """All-six-source replay barrier, then exact endpoint-only caching.

    Small cohorts are permitted only by this pure scheduling boundary for CPU
    tests. The production source loader/runner independently require 183/99.
    Each predictor accepts just a private [1,1,64,64] image array.
    """
    rows = identities(ids)
    images = validate_images(images)
    if len(images) != len(rows) or any(set(v) != set(KEYS) for v in (maps, predictions, predictors, source_records)):
        raise ValueError("Exact six-source maps/predictions/predictors and image identities required")
    ledger = ledger if ledger is not None else CallLedger()
    if sum(ledger.attempted.values()) or ledger.replays or ledger.barrier_passed:
        raise ValueError("Automatic retries/resume are forbidden; a new ledger is required")
    prior_order = rank_native(prior)
    def memory_snapshot():
        return {"images": array_record(images), "prior": array_record(prior),
                "maps": {str(k): array_record(v) for k, v in maps.items()}}
    before = memory_snapshot()
    units, permutations, digests, frames = [], [], [], {}
    controls = [hash_order(row) for row in rows]
    for model, seed in KEYS:
        key = (model, seed)
        frame = predictions[key]
        identities(frame)
        frame = frame.set_index("nodule_id", verify_integrity=True).loc[ids.nodule_id].reset_index()
        if len(predictions[key]) != len(ids) or identities(frame) != rows or not frame.predicted_class.isin(CLASS_ORDER).all():
            raise ValueError("Saved targets/identities differ from exact archive membership")
        frames[key] = frame
        if np.asarray(maps[key]).shape != (len(rows), 1, 8, 8):
            raise ValueError("Saved DSA grid must be [N,1,8,8]")
        for i, row in enumerate(rows):
            unit = len(units)
            target = CLASS_ORDER.index(frame.iloc[i].predicted_class)
            units.append({"unit_id": f"unit_{unit:04d}", **row, "model": model, "seed": seed,
                "archive_row": i, "target_index": target, "target_class": CLASS_ORDER[target],
                "input_sha256": array_record(images[i])["array_sha256"],
                "map_sha256": array_record(maps[key][i, 0])["array_sha256"],
                "prior_sha256": array_record(prior)["array_sha256"],
                "source_record_sha256": sha256_json(source_records[key]),
                "best_checkpoint_sha256": source_records[key]["best_checkpoint_sha256"],
                "constant_input": bool(np.all(images[i] == images[i, 0, 0, 0])),
                "baseline_identical_input": bool(np.all(images[i] == 0))})
            permutations.append(np.stack((rank_native(maps[key][i, 0]), prior_order, controls[i][0])))
            digests.append(controls[i][1])
    units = pd.DataFrame(units)
    orders = {"unit_id": units.unit_id.to_numpy(dtype=np.str_), "permutations": np.stack(permutations),
              "hash_digests_by_flat_index": np.stack(digests)}
    order_hash = sha256_json({k: array_record(v) for k, v in orders.items()})
    count = len(units)
    raw_logits = np.empty((count, 3, 2, 65, 3), dtype=np.float32)
    raw_probs = np.empty_like(raw_logits)
    calls = np.empty((count, 3, 2, 65), dtype=np.int64)
    input_hashes = np.empty(calls.shape, dtype="S64")
    zero = np.zeros((1, 64, 64), dtype=np.float32)
    baseline_hash = array_record(zero)["array_sha256"]
    units["baseline_sha256"] = baseline_hash
    replay_logits = np.empty((count, 3), dtype=np.float32)
    replay_probs = np.empty_like(replay_logits)
    baseline_logits, baseline_probs = np.empty_like(replay_logits), np.empty_like(replay_probs)

    def fresh(unit, image, stage, **position):
        meta = units.iloc[unit]
        ledger.attempted[stage] += 1
        call = sum(ledger.attempted.values())
        digest = array_record(image)["array_sha256"]
        ledger.last_attempt = {"call_index": call, "stage": stage, "unit_id": meta.unit_id,
                               "model": meta.model, "seed": int(meta.seed), "input_sha256": digest, **position}
        try:
            output = predictors[(meta.model, int(meta.seed))](image[None].copy())
            logits, probs, _, _ = validate_outputs(output)
        except BaseException as exc:
            # Include KeyboardInterrupt/SystemExit/GeneratorExit without accepting
            # an incomplete output or swallowing the original interruption.
            try:
                ledger.failure = {**ledger.last_attempt, "error_type": type(exc).__name__, "message": str(exc)}
            except BaseException:
                pass  # Even failure-detail construction must not replace the original.
            raise
        ledger.accepted[stage] += 1
        return logits[0], probs[0], call, digest

    for u, meta in units.iterrows():
        i, key = int(meta.archive_row), (meta.model, int(meta.seed))
        logits, probs, call, digest = fresh(u, images[i], "unperturbed")
        replay_logits[u], replay_probs[u] = logits, probs
        report = _saved_replay(frames[key].iloc[[i]], ids.iloc[[i]], logits[None], probs[None])
        predicted = int(probs.argmax())
        report.update(unit_id=meta.unit_id, model=meta.model, seed=int(meta.seed), nodule_id=meta.nodule_id,
                      target_class=meta.target_class, replayed_class=CLASS_ORDER[predicted], call_index=call,
                      raw_row_sum=float(probs.astype(np.float64).sum()),
                      raw_mass_deviation=float(probs.astype(np.float64).sum()-1))
        report["consistent"] = report["consistent"] and predicted == int(meta.target_index)
        ledger.replays.append(report)
        if not report["consistent"]:
            ledger.failure = {**ledger.last_attempt, "error_type": "ReplayMismatch", "report": report}
            raise ValueError("Global unperturbed replay/argmax barrier failed; no perturbation permitted")
        for d, step in ((0, 64), (1, 0)):
            raw_logits[u, :, d, step], raw_probs[u, :, d, step] = logits, probs
            calls[u, :, d, step], input_hashes[u, :, d, step] = call, digest.encode("ascii")
    replay = {"unit_id": orders["unit_id"].copy(), "logits": replay_logits, "probabilities": replay_probs,
              "target_index": units.target_index.to_numpy(dtype=np.int64, copy=True)}
    binding = {"source_binding": source_binding, "orders_sha256": order_hash,
               "units_sha256": sha256_json(units.to_dict("records")),
               "arrays": {k: array_record(v) for k, v in replay.items()}}
    for value in replay.values():
        value.setflags(write=False)
    if memory_snapshot() != before:
        raise ValueError("Original inputs/maps/prior/rankings changed before perturbation barrier")
    if on_barrier is not None:
        on_barrier(replay, {**binding, "cache_binding_sha256": sha256_json(binding)}, ledger)
    if {k: array_record(v) for k, v in replay.items()} != binding["arrays"]:
        raise ValueError("Sealed replay cache changed before perturbation")
    ledger.barrier_passed = True

    for u, meta in units.iterrows():
        image = images[int(meta.archive_row)]
        logits, probs, call, digest = fresh(u, zero, "baseline")
        baseline_logits[u], baseline_probs[u] = logits, probs
        for d, step in ((0, 0), (1, 64)):
            raw_logits[u, :, d, step], raw_probs[u, :, d, step] = logits, probs
            calls[u, :, d, step], input_hashes[u, :, d, step] = call, digest.encode("ascii")
        for o, order in enumerate(orders["permutations"][u]):
            for d, direction in enumerate(DIRECTIONS):
                canvas = zero.copy() if d == 0 else image.copy()
                for step, flat in enumerate(order[:-1], start=1):
                    r, c = divmod(int(flat), 8)
                    canvas[:, r*8:r*8+8, c*8:c*8+8] = image[:, r*8:r*8+8, c*8:c*8+8] if d == 0 else 0
                    logits, probs, call, digest = fresh(u, canvas, "interior", order=ORDERS[o], direction=direction, step=step)
                    raw_logits[u, o, d, step], raw_probs[u, o, d, step] = logits, probs
                    calls[u, o, d, step], input_hashes[u, o, d, step] = call, digest.encode("ascii")
    after = memory_snapshot()
    if (before != after or sha256_json({k: array_record(v) for k, v in orders.items()}) != order_hash
            or {k: array_record(v) for k, v in replay.items()} != binding["arrays"]):
        raise ValueError("Original inputs/maps/prior/rankings changed during execution")
    budget = scheduled_budget(len(rows))
    if ledger.attempted != {k: budget[k] for k in ledger.attempted} or ledger.accepted != ledger.attempted:
        raise ValueError("Successful-run call accounting differs")
    curves = {"logits": raw_logits, "probabilities": raw_probs, "call_index": calls,
              "input_sha256": input_hashes}
    baseline = {"unit_id": orders["unit_id"].copy(), "logits": baseline_logits, "probabilities": baseline_probs}
    return PerturbationResult(units, orders, curves, replay, baseline, ledger)


def response_arrays(result):
    probs = result.curves["probabilities"]
    validate_probabilities(probs.reshape(-1, 3))
    targets = result.units.target_index.to_numpy(dtype=np.int64)
    return np.take_along_axis(probs, targets[:, None, None, None, None], axis=-1)[..., 0].astype(np.float64)


def summarize_curves(result):
    values = response_arrays(result)
    auc = curve_auc(values)
    raw, differences = [], []
    for u, meta in result.units.iterrows():
        identity = {k: meta[k] for k in ("unit_id", *IDENTITY, "model", "seed", "target_class")}
        for o, order in enumerate(ORDERS):
            for d, direction in enumerate(DIRECTIONS):
                curve = values[u, o, d]
                raw.append({**identity, "order": order, "direction": direction, "auc": float(auc[u, o, d]),
                            "flat_curve": bool(np.all(curve == curve[0])),
                            "nonmonotonic_curve": bool((np.diff(curve) > 0).any() and (np.diff(curve) < 0).any()),
                            "equal_all_order_curves": bool(np.all(values[u, :, d] == values[u, 0, d]))})
        for o in (1, 2):
            for d, direction in enumerate(DIRECTIONS):
                differences.append({**identity, "contrast": f"dsa_minus_{ORDERS[o]}", "direction": direction,
                                    "auc_difference": float(auc[u, 0, d]-auc[u, o, d])})
    raw, differences = pd.DataFrame(raw), pd.DataFrame(differences)
    def summary(frame, groups, column):
        records = []
        for key, group in frame.groupby(groups, sort=False):
            v = group[column].to_numpy(dtype=np.float64)
            records.append({**dict(zip(groups, key)), "n_nodules": len(group), "n_patients": group.patient_id.nunique(),
                            "mean": float(v.mean()), "median": float(np.quantile(v, .5, method="linear")),
                            "q25": float(np.quantile(v, .25, method="linear")),
                            "q75": float(np.quantile(v, .75, method="linear"))})
        return pd.DataFrame(records)
    mean_curves = []
    for model, seed in KEYS:
        selected = (result.units.model == model) & (result.units.seed == seed)
        means = values[selected].mean(axis=0)
        for o, order in enumerate(ORDERS):
            for d, direction in enumerate(DIRECTIONS):
                mean_curves.extend({"model": model, "seed": seed, "order": order, "direction": direction,
                                    "step": k, "fraction": k/64, "mean_response": float(v),
                                    "n_nodules": int(selected.sum()),
                                    "n_patients": result.units.loc[selected, "patient_id"].nunique()}
                                   for k, v in enumerate(means[o, d]))
    return {"perturbation_auc_per_nodule.csv": raw,
            "perturbation_paired_per_nodule.csv": differences,
            "perturbation_per_seed.csv": summary(raw, ["model", "seed", "order", "direction"], "auc"),
            "perturbation_paired_descriptive.csv": summary(differences, ["model", "seed", "contrast", "direction"], "auc_difference"),
            "perturbation_mean_curves.csv": pd.DataFrame(mean_curves)}
