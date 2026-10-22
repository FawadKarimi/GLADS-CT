"""Fail-closed comparison of supplied run bindings; never launches or reads data."""
import hashlib
import json
import re

MODEL_ID = "performance_v2_r50_hybrid_v2"
BINDINGS = (
    "effective_scientific_and_execution_config_sha256", "source_revision_and_source_files_sha256",
    "topology_and_module_initialization_spec_sha256", "source_weight_file_and_loaded_state_sha256",
    "common_encoder_and_diagnostic_head_initial_state_sha256", "train_validation_cohort_identity_label_order_sha256",
    "source_image_and_supervision_input_bindings_sha256", "transform_contract_and_implementation_sha256",
    "optimizer_groups_and_rates_schedule_sha256", "objective_coefficients_and_reductions_sha256",
    "rng_streams_seeds_shuffle_augmentation_sha256", "runtime_hardware_precision_determinism_sha256",
    "stopping_checkpoint_selection_metric_definition_sha256",
)
ADMIN = {"screen_or_repeat_phase_label", "review_timestamp", "analysis_role_alias"}
KEYS = {"run_id", "model_id", "protocol_version", "initialization", "objective", "seed", "recipe", "bindings"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json(raw):
    if type(raw) is not bytes or len(raw) > 1048576:
        raise ValueError("bounded raw JSON bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def scientific_contract(record):
    if type(record) is not dict or set(record) not in (KEYS, KEYS | {"administrative"}):
        raise ValueError("missing or unclassified execution fields")
    if (record["model_id"] != MODEL_ID or type(record["protocol_version"]) is not int
            or record["protocol_version"] != 2 or record["recipe"] != "A"
            or type(record["seed"]) is not int or record["seed"] not in (42, 43, 44)
            or record["initialization"] not in ("RANDOM", "IMAGENET", "RADIMAGENET")
            or record["objective"] not in ("HARD", "LEGACY", "SHARED-010", "SEPARATE-010")
            or not isinstance(record["run_id"], str) or not record["run_id"]):
        raise ValueError("unreviewed run identity")
    bindings = record["bindings"]
    if type(bindings) is not dict or set(bindings) != set(BINDINGS) or not all(_hash(x) for x in bindings.values()):
        raise ValueError("complete exact execution/input digests required")
    admin = record.get("administrative", {})
    if type(admin) is not dict or not set(admin) <= ADMIN or not all(isinstance(x, str) for x in admin.values()):
        raise ValueError("unreviewed administrative exclusion")
    return {k: record[k] for k in sorted(KEYS)}


def require_exact_reuse(previous, requested, *, completion_bytes, completion_sha256,
                       manifest_bytes, verified_output_hashes):
    """Validate a supplied receipt and freshly verified artifact digest map.

    The caller must obtain completion_sha256 from its trusted run ledger and
    rehash actual output bytes before supplying verified_output_hashes. This pure
    validator does not claim those reads occurred and never grants execution.
    """
    old, new = scientific_contract(previous), scientific_contract(requested)
    if old != new:
        raise ValueError("run reuse rejected: execution or input binding mismatch")
    if not _hash(completion_sha256) or digest(completion_bytes) != completion_sha256:
        raise ValueError("completion receipt hash mismatch")
    receipt = _json(completion_bytes)
    if (type(receipt) is not dict or receipt.get("status") != "complete"
            or receipt.get("run_id") != old["run_id"]
            or receipt.get("contract_sha256") != digest(canonical(old))
            or receipt.get("output_manifest_sha256") != digest(manifest_bytes)):
        raise ValueError("incomplete or unbound run receipt")
    manifest = _json(manifest_bytes)
    outputs = manifest.get("files") if type(manifest) is dict else None
    if (type(outputs) is not dict or not outputs
            or not all(isinstance(k, str) and k and _hash(v) for k, v in outputs.items())
            or type(verified_output_hashes) is not dict or outputs != verified_output_hashes):
        raise ValueError("fresh output-manifest verification required")
    return {"run_id": old["run_id"], "reuse_compatible": True,
            "counts_as_additional_run": False, "pilot_execution_authorized": False}
