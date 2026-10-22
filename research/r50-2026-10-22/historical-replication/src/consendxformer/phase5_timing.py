"""Pure synthetic schedule, call accounting and descriptive timing arithmetic."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from .hashing import sha256_json
from .phase4_attention_evidence import array_record
from .phase5_perturbation import KEYS, tile_input, validate_outputs

SCOPE = "synthetic_timing_only"
TOTAL = 2496
DURATION_FIELDS = ("input_construction", "adapter", "validation_hashing")


def timing_specification():
    return {"schema": "phase5_synthetic_timing_v1", "evidence_revision": 2, "scope": SCOPE,
        "models": ["ConsenDxFormer", "B1"], "seeds": [42, 43, 44],
        "input_shape": [1, 1, 64, 64], "dtype": "float32", "input_seed": 20260917,
        "warmup_per_unit": 32, "blocks_per_unit": 3, "calls_per_block": 128,
        "calls_per_unit": 416, "warmup_calls": 192, "measured_calls": 2304, "total_calls": TOTAL,
        "resident_models": 6, "probe_calls": 0, "automatic_retries": 0,
        "sequence": {"index": "j=0..415 independently repeated for each model/seed",
                     "image": "[zero,ramp,PCG64_random][j%3]",
                     "direction": "[insertion,deletion][(j//3)%2]", "step": "(j//6)%65",
                     "order": "row_major_arange_64", "tile_constructor": "phase5_perturbation.tile_input",
                     "ramp": "np.linspace(0,1,4096,dtype=float32).reshape(1,64,64)",
                     "random": "Generator(PCG64(20260917)).random((1,64,64),dtype=float32)"},
        "execution": {"eval": True, "frozen": True, "gradients": False, "autocast": False,
                      "deterministic_algorithms": True, "warn_only": False, "cudnn_benchmark": False},
        "cuda": {"selection": "full_physical_UUID_before_torch_import", "logical_device": "cuda:0",
                 "mapping_verification": "cuda_driver_uuid_pci_v2",
                 "memory_totals": "separate_nvidia_smi_and_torch_diagnostics_no_cross_api_equality",
                 "cublas_workspace_config": ":4096:8", "max_startup_used_mib": 256,
                 "launch_blocking": "required_raw_pre_import_value_null_for_unset_no_default_or_mutation_exact_runtime_binding",
                 "active_compute_processes_allowed": 0, "automatic_switching": False},
        "measurement": {"clock": "time.perf_counter_ns", "adapter": "pre_sync_then_start_adapter_post_sync_end",
            "adapter_includes": "input guards/private copy/H2D/model/D2H/output copies/post-sync",
            "disjoint_components": ["input_construction", "adapter", "validation_hashing"],
            "loop_envelope": {"raw_unit": "integer_nanoseconds", "includes": "all_2496_calls_including_warmup",
                "seconds": "canonical_nanoseconds_divided_by_1e9",
                "integer_guard": "loop_nanoseconds_at_least_exact_sum_of_all_disjoint_nanoseconds",
                "float_guard": "loop_seconds_at_least_math_fsum_of_disjoint_seconds_minus_4_ulp_of_sum",
                "equality_required": False},
            "envelopes_not_added": ["preparation", "loop_wall"],
            "bounded_export": "one deterministic NPZ of 2496 raw logit/probability rows; write/hash/read-back"},
        "extrapolation": {"research_calls_per_model_seed": 69540, "research_calls_total": 417240,
                          "point": "mean of three block adapter seconds/call times 69540, sum six",
                          "range": "sum six per-unit block minima/maxima times 69540; not a CI"},
        "research_execution_authorized": False, "research_forwards": 0,
        "full_supplemental_completion": False, "stage_a_complete": False, "formal_tests_computed": False}


def make_plan():
    images = {"zero": np.zeros((1, 64, 64), np.float32),
              "ramp": np.linspace(0, 1, 4096, dtype=np.float32).reshape(1, 64, 64),
              "random": np.random.Generator(np.random.PCG64(20260917)).random((1, 64, 64), dtype=np.float32)}
    order = np.arange(64, dtype=np.int64)
    sequence = []
    for j in range(416):
        name, direction, step = ("zero", "ramp", "random")[j % 3], ("insertion", "deletion")[(j//3) % 2], (j//6) % 65
        image = tile_input(images[name], order, direction, step)
        sequence.append({"sequence_index": j, "input": name, "direction": direction, "step": step,
                         "phase": "warmup" if j < 32 else "measured", "block": -1 if j < 32 else (j-32)//128,
                         "input_sha256": array_record(image)["array_sha256"]})
    plan = {"schema": "phase5_synthetic_sequence_v1", "images": {k: array_record(v) for k, v in images.items()},
            "order": order.tolist(), "sequence": sequence}
    return images, plan


class PerfClock:
    name = "time.perf_counter_ns"
    def __call__(self):
        return time.perf_counter_ns()


def elapsed_ns(clock, start):
    end = clock()
    if type(start) is not int or type(end) is not int or end < start:
        raise ValueError("Invalid/nonmonotonic clock")
    return end-start


def elapsed(clock, start):
    return elapsed_ns(clock, start)/1e9


def validate_loop_envelope(measured):
    """Stable all-call sum plus exact ns containment; floating allowance is roundoff only."""
    calls = measured["calls"]
    if len(calls) != TOTAL:
        raise ValueError("Loop envelope requires all 2496 calls including warm-up")
    intervals = [(row, field) for row in calls for field in DURATION_FIELDS]
    intervals.append((measured, "loop_wall"))
    for record, field in intervals:
        ns, seconds = record.get(field + "_nanoseconds"), record.get(field + "_seconds")
        if (type(ns) is not int or ns < 0 or type(seconds) not in (float, int)
                or not math.isfinite(seconds) or seconds != ns/1e9):
            raise ValueError("Missing/invalid nanosecond duration or noncanonical seconds")
    contained_ns = sum(row[field + "_nanoseconds"] for row in calls for field in DURATION_FIELDS)
    contained_seconds = math.fsum(row[field + "_seconds"] for row in calls for field in DURATION_FIELDS)
    # Binary64 conversion, fsum rounding (including a possible last-bit double
    # rounding) and envelope conversion only. No absolute/proportional time slack;
    # the exact integer check still rejects even a one-nanosecond deficit.
    allowance = 4 * math.ulp(contained_seconds)
    if (measured["loop_wall_nanoseconds"] < contained_ns
            or contained_seconds - measured["loop_wall_seconds"] > allowance):
        raise ValueError("Loop envelope is smaller than its all-call disjoint durations")


@dataclass
class TimingLedger:
    attempted: dict = field(default_factory=lambda: {"warmup": 0, "measured": 0})
    accepted: dict = field(default_factory=lambda: {"warmup": 0, "measured": 0})
    last_attempt: dict | None = None
    failure: dict | None = None

    def record(self):
        return {"attempted": dict(self.attempted), "accepted": dict(self.accepted),
                "last_attempt": self.last_attempt, "failure": self.failure, "automatic_retries": 0}


def measure_schedule(predictors, images, plan, *, clock, synchronize, ledger, memory):
    """Injectable pure test boundary. Production runner supplies actual six constructors."""
    if set(predictors) != set(KEYS) or sum(ledger.attempted.values()):
        raise ValueError("All six resident models and a fresh ledger are required")
    original = sha256_json({k: array_record(v) for k, v in images.items()})
    original_plan = sha256_json(plan)
    rows, blocks, outputs = [], [], []
    loop_start = clock()
    for model, seed in KEYS:
        for position in plan["sequence"]:
            ledger.last_attempt = {"model": model, "seed": seed, **position, "stage": "input_construction"}
            try:
                start = clock()
                image = tile_input(images[position["input"]], np.asarray(plan["order"]), position["direction"], position["step"])
                construction_ns = elapsed_ns(clock, start)
                start = clock()
                if array_record(image)["array_sha256"] != position["input_sha256"]:
                    raise ValueError("Synthetic input sequence changed")
                before_validation_ns = elapsed_ns(clock, start)
                synchronize()
                ledger.last_attempt["stage"] = "adapter"
                ledger.attempted[position["phase"]] += 1
                start = clock()
                output = predictors[(model, seed)](image[None])
                synchronize()
                adapter_ns = elapsed_ns(clock, start)
                ledger.last_attempt["stage"] = "output_validation"
                start = clock()
                logits, probabilities, mass, deviation = validate_outputs(output)
                diagnostics = {"logits": array_record(logits), "probabilities": array_record(probabilities),
                               "raw_row_sum": float(mass[0]), "raw_mass_deviation": float(deviation[0])}
                validation_ns = before_validation_ns + elapsed_ns(clock, start)
                ledger.accepted[position["phase"]] += 1
                rows.append({"call_index": len(rows)+1, "model": model, "seed": seed, **position,
                             "input_construction_nanoseconds": construction_ns, "adapter_nanoseconds": adapter_ns,
                             "validation_hashing_nanoseconds": validation_ns,
                             "input_construction_seconds": construction_ns/1e9, "adapter_seconds": adapter_ns/1e9,
                             "validation_hashing_seconds": validation_ns/1e9, **diagnostics})
                outputs.append((logits[0], probabilities[0]))
                if position["phase"] == "measured" and (position["sequence_index"]-31) % 128 == 0:
                    group = rows[-128:]
                    blocks.append({"model": model, "seed": seed, "block": position["block"], "calls": 128,
                        **{k: math.fsum(r[k] for r in group) for k in ("input_construction_seconds", "adapter_seconds", "validation_hashing_seconds")},
                        "memory": memory()})
            except BaseException as exc:
                try: ledger.failure = {**ledger.last_attempt, "error_type": type(exc).__name__, "message": str(exc)}
                except BaseException: pass
                raise
    loop_wall_ns = elapsed_ns(clock, loop_start)
    if original != sha256_json({k: array_record(v) for k, v in images.items()}) or original_plan != sha256_json(plan):
        raise ValueError("Synthetic inputs/plan mutated during execution")
    if ledger.attempted != {"warmup": 192, "measured": 2304} or ledger.accepted != ledger.attempted:
        raise ValueError("Timing call budget differs")
    result = {"calls": rows, "blocks": blocks, "loop_wall_seconds": loop_wall_ns/1e9,
              "loop_wall_nanoseconds": loop_wall_ns, "clock": clock.name,
              "raw": {"logits": np.stack([o[0] for o in outputs]), "probabilities": np.stack([o[1] for o in outputs])}}
    validate_loop_envelope(result)
    return result


def extrapolate(blocks):
    if len(blocks) != 18:
        raise ValueError("All 18 measured blocks required")
    units = []
    for model, seed in KEYS:
        selected = [b for b in blocks if (b["model"], b["seed"]) == (model, seed)]
        if sorted(b["block"] for b in selected) != [0, 1, 2] or any(b["calls"] != 128 for b in selected):
            raise ValueError("Missing/duplicate timing blocks")
        times = [b["adapter_seconds"]/128 for b in selected]
        if any(not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError("Invalid measured adapter duration")
        units.append({"model": model, "seed": seed, "research_calls": 69540,
                      "seconds_per_call": math.fsum(times)/3,
                      "estimated_adapter_seconds": math.fsum(times)/3*69540,
                      "block_min_estimate_seconds": min(times)*69540, "block_max_estimate_seconds": max(times)*69540})
    return {"units": units, "estimated_adapter_seconds": math.fsum(u["estimated_adapter_seconds"] for u in units),
            "block_min_estimate_seconds": math.fsum(u["block_min_estimate_seconds"] for u in units),
            "block_max_estimate_seconds": math.fsum(u["block_max_estimate_seconds"] for u in units),
            "actual_research_runtime": False, "range_is_confidence_interval": False,
            "excludes": ["research loading/startup", "input construction", "external validation/hashing", "research export/I/O", "contention changes"]}
