"""Bounded non-pickle CPU state transport and exact resume checks.

No pretrained loader, torch.load, optimizer step, or production data access.
The caller supplies trusted hashes and owns original-source authentication.
"""
import base64
import json
import math

from .metrics import metric_definition_sha256
from .optimization import epoch_multiplier, optimizer_groups
from .reuse import canonical, digest, _hash, scientific_contract
from .selection import ValidationEpoch, ValidationPatience, select_checkpoint
from .streams import validate_cpu_rng

MAX_BYTES = 512 * 1024 * 1024
GROUPS = ("encoder_weights", "encoder_bias_normalization", "head_weights", "head_bias_normalization")
SELECTION = {"primary": "validation_macro_f1", "tolerance": 1e-12,
             "tie": ["unweighted_validation_diagnostic_ce", "earliest_epoch"],
             "min_epochs": 20, "max_epochs": 100, "patience": 15, "strict_delta": 1e-4}
DTYPES = {"float32": (4, "float32"), "float64": (8, "float64"),
          "int64": (8, "int64"), "int32": (4, "int32"), "uint8": (1, "uint8"), "bool": (1, "bool")}


def encode_state(value):
    """Strict typed JSON tree; contiguous little-endian CPU tensors only."""
    import sys
    import torch
    if sys.byteorder != "little":
        raise ValueError("state codec requires an explicitly reviewed little-endian host")
    counter = [0]
    def encode(x, depth=0):
        counter[0] += 1
        if depth > 32 or counter[0] > 100000:
            raise ValueError("state tree bound exceeded")
        if isinstance(x, torch.Tensor):
            name = str(x.dtype).removeprefix("torch.")
            if (x.device.type != "cpu" or x.layout != torch.strided or name not in DTYPES
                    or x.numel() * x.element_size() > MAX_BYTES or not torch.isfinite(x).all()):
                raise ValueError("bounded finite CPU state tensors required")
            # Explicit byte copy: no tensor pickle/storage reconstruction hooks.
            raw = x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return ["tensor", name, list(x.shape), base64.b64encode(raw).decode("ascii")]
        if x is None or type(x) in (str, bool, int):
            return ["scalar", x]
        if type(x) is float and math.isfinite(x):
            return ["scalar", x]
        if type(x) in (list, tuple):
            return ["tuple" if type(x) is tuple else "list", [encode(v, depth + 1) for v in x]]
        if isinstance(x, dict):
            if any(type(k) not in (str, int) for k in x):
                raise ValueError("only string/integer state keys allowed")
            return ["dict", [[encode(k, depth + 1), encode(x[k], depth + 1)]
                             for k in sorted(x, key=lambda k: (type(k).__name__, str(k)))]]
        raise ValueError("unclassified state type")
    raw = canonical(encode(value))
    if len(raw) > MAX_BYTES:
        raise ValueError("checkpoint byte cap exceeded")
    return raw


def decode_state(raw, *, expected_sha256):
    import sys
    import torch
    if (sys.byteorder != "little" or type(raw) is not bytes or len(raw) > MAX_BYTES
            or not _hash(expected_sha256) or digest(raw) != expected_sha256):
        raise ValueError("checkpoint size, byte order or digest mismatch before parsing")
    def constant(_):
        raise ValueError("nonfinite JSON")
    try:
        tree = json.loads(raw, parse_constant=constant)
    except (ValueError, RecursionError) as error:
        raise ValueError("malformed checkpoint") from error
    counter = [0]
    def decode(x, depth=0):
        counter[0] += 1
        if depth > 32 or counter[0] > 100000 or type(x) is not list or not x:
            raise ValueError("invalid bounded state tree")
        tag = x[0]
        if tag == "scalar" and len(x) == 2:
            v = x[1]
            if v is None or type(v) in (str, int, bool) or type(v) is float and math.isfinite(v):
                return v
        elif tag in ("tuple", "list") and len(x) == 2 and type(x[1]) is list:
            values = [decode(v, depth + 1) for v in x[1]]
            return tuple(values) if tag == "tuple" else values
        elif tag == "dict" and len(x) == 2 and type(x[1]) is list:
            result = {}
            for pair in x[1]:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError("invalid state dictionary pair")
                key, value = decode(pair[0], depth + 1), decode(pair[1], depth + 1)
                if type(key) not in (str, int) or key in result:
                    raise ValueError("duplicate or unsafe state key")
                result[key] = value
            return result
        elif tag == "tensor" and len(x) == 4:
            name, shape, text = x[1:]
            if (not isinstance(name, str) or name not in DTYPES or type(shape) is not list
                    or len(shape) > 16 or any(type(v) is not int or v < 0 for v in shape)
                    or type(text) is not str):
                raise ValueError("invalid tensor descriptor")
            size = math.prod(shape) * DTYPES[name][0]
            if size > MAX_BYTES:
                raise ValueError("tensor allocation cap exceeded")
            data = base64.b64decode(text, validate=True)
            if len(data) != size:
                raise ValueError("tensor length mismatch")
            dtype = getattr(torch, DTYPES[name][1])
            tensor = (torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)
                      if size else torch.empty(shape, dtype=dtype))
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite checkpoint tensor")
            return tensor
        raise ValueError("unknown state encoding")
    try:
        result = decode(tree)
    except (TypeError, OverflowError, RuntimeError) as error:
        raise ValueError("invalid checkpoint state") from error
    if encode_state(result) != raw:
        raise ValueError("noncanonical state encoding")
    return result


def state_sha256(value):
    return digest(encode_state(value))


def tensor_layout(state):
    import torch
    if not isinstance(state, dict) or not state or any(not isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError("complete named model tensor state required")
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()}


def _model_inventory(model):
    """Observe actual objects, not caller-declared state-dict parameter names."""
    from torch import nn
    import torch
    if not isinstance(model, nn.Module):
        raise ValueError("actual model module required for complete parameter inventory")
    # This is the authoritative reviewed rule implementation, including module
    # ownership, MHA projections, positional exception and frozen/aliased rejection.
    _, expected_ledger = optimizer_groups(model)
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    objects, storage = set(), set()
    for name, tensor in parameters + buffers:
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("checkpoint inventory requires ordinary CPU tensors")
        if id(tensor) in objects:
            raise ValueError("aliased parameter/buffer object: " + name)
        objects.add(id(tensor))
        if tensor.numel():
            key = tensor.untyped_storage().data_ptr()
            if key in storage:
                raise ValueError("shared parameter/buffer storage: " + name)
            storage.add(key)
    layout = tensor_layout(model.state_dict())
    actual = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
              for name, tensor in parameters + buffers}
    if len(actual) != len(parameters) + len(buffers) or layout != actual:
        raise ValueError("model state must contain exactly all parameters and persistent buffers")
    assignments = {row["name"]: row["group"] for row in expected_ledger}
    trainable = [dict(name=name, **actual[name], optimizer_group=assignments[name])
                 for name, _ in parameters]
    buffer_inventory = [dict(name=name, **actual[name]) for name, _ in buffers]
    return layout, trainable, buffer_inventory, expected_ledger


def _validate_execution_inventory(spec):
    """Check the serialized partition against the model-bound requested spec.

    Group assignments are obtained from optimizer_groups(actual_model) when the
    spec is built. On resume the complete saved spec must equal that freshly
    built requested spec; self-declared inventories are not an authenticity proof.
    """
    if (type(spec) is not dict or spec.get("schema") != "performance_v2_execution_interface_v2"
            or type(spec.get("trainable_parameter_inventory")) is not list
            or not spec["trainable_parameter_inventory"]
            or type(spec.get("buffer_inventory")) is not list):
        raise ValueError("complete model-bound execution inventory required; no v1 fallback")
    from .retention import enabled
    enabled(spec)  # Validate explicit opt-in; absence preserves historical behavior.
    mapping = spec.get('mapping_evidence')
    if mapping is not None:
        if not isinstance(mapping, dict) or not _hash(spec.get('mapping_evidence_sha256')) or spec['mapping_evidence_sha256'] != digest(canonical(mapping)):
            raise ValueError('complete mapping evidence digest required')
        wanted = digest(canonical({'input_lock': spec['input_lock_sha256'],
                                   'paths': mapping.get('path_manifest_sha256'), 'mapping': mapping}))
        if spec['contract']['bindings']['source_image_and_supervision_input_bindings_sha256'] != wanted:
            raise ValueError('execution mapping evidence differs from scientific contract')
        if mapping.get('input_lock_sha256') != spec['input_lock_sha256']:
            raise ValueError('mapping input lock drift')
    if spec.get('scope') == 'production':
        if (not isinstance(mapping, dict) or mapping.get('scope') != 'production'
                or mapping.get('mapping_review_approved') is not True
                or mapping.get('pilot_execution_authorized') is not False
                or any(not _hash(mapping.get(k)) for k in ('bundle_sha256', 'lineage_sha256', 'approval_sha256'))
                or any(not isinstance(mapping.get(k), str) or not mapping[k] for k in ('source_root', 'sidecar_root'))):
            raise ValueError('production requires approved lineage, roots and complete mapping bundle')
    parameters, buffers = spec["trainable_parameter_inventory"], spec["buffer_inventory"]
    inventory = {"trainable_parameters": parameters, "buffers": buffers}
    if spec.get("model_inventory_sha256") != digest(canonical(inventory)):
        raise ValueError("model inventory digest mismatch")
    names, layout, expected_rows = set(), {}, []
    for records, trainable in ((parameters, True), (buffers, False)):
        fields = {"name", "shape", "dtype"} | ({"optimizer_group"} if trainable else set())
        for record in records:
            if (type(record) is not dict or set(record) != fields
                    or not isinstance(record["name"], str) or not record["name"] or record["name"] in names
                    or type(record["shape"]) is not list
                    or any(type(n) is not int or n < 0 for n in record["shape"])
                    or record["dtype"] not in {"torch." + name for name in DTYPES}):
                raise ValueError("invalid or overlapping parameter/buffer inventory")
            name = record["name"]
            names.add(name)
            layout[name] = {"shape": record["shape"], "dtype": record["dtype"]}
            if trainable:
                group = record["optimizer_group"]
                if group not in GROUPS or record["dtype"] != "torch.float32":
                    raise ValueError("unreviewed trainable-parameter group or dtype")
                expected_rows.append({"name": name, "shape": record["shape"], "group": group,
                                      "lr": 1e-4 if group.startswith("encoder") else 1e-3,
                                      "weight_decay": 1e-4 if group.endswith("weights") else 0.0})
    if (spec.get("model_layout") != layout or spec.get("optimizer_ledger") != expected_rows
            or {row["group"] for row in expected_rows} != set(GROUPS)):
        raise ValueError("optimizer ledger must cover every actual trainable parameter exactly in reviewed order/groups")


def execution_spec(contract, inputs, *, model, optimizer_ledger, mapping_evidence=None):
    """Bind the actual complete model inventory and the reviewed optimizer rules."""
    layout, parameters, buffers, expected = _model_inventory(model)
    if list(optimizer_ledger) != expected:
        raise ValueError("optimizer ledger differs from complete reviewed model grouping")
    spec = {"schema": "performance_v2_execution_interface_v2", "contract": scientific_contract(contract),
            "input_lock_sha256": inputs.sha256, "scope": inputs.scope, "model_layout": layout,
            "trainable_parameter_inventory": parameters, "buffer_inventory": buffers,
            "model_inventory_sha256": digest(canonical({"trainable_parameters": parameters, "buffers": buffers})),
            "optimizer_ledger": expected, "scheduler": "recipe_A_warmup_cosine_v1",
            "selection": dict(SELECTION), "metric_definition_sha256": metric_definition_sha256(),
            "batch_size": 16, "drop_last": False, "workers": 0,
            "resume_boundary": "completed_validation_epoch_only", "pilot_execution_authorized": False}
    if mapping_evidence is not None:
        import copy
        spec['mapping_evidence'] = copy.deepcopy(mapping_evidence)
        spec['mapping_evidence_sha256'] = digest(canonical(mapping_evidence))
    _validate_execution_inventory(spec)
    return spec


def scheduler_state(epoch):
    multiplier = epoch_multiplier(epoch)
    return {"schema": "recipe_A_warmup_cosine_v1", "last_epoch": epoch,
            "next_epoch": epoch + 1, "multiplier": multiplier}


def selection_state(history):
    records = [ValidationEpoch(**row) for row in history]
    patience = ValidationPatience()
    for record in records:
        patience.observe(record)
    best = select_checkpoint(records)
    return {"history": [dict(r.__dict__) for r in records], "best_epoch": best.epoch,
            "last_epoch": patience.last_epoch, "raw_best": patience.best,
            "without_improvement": patience.without_improvement, "stopped": patience.stopped,
            "rules": dict(SELECTION)}


def validate_components(spec, components, *, epoch, updates):
    import torch
    _validate_execution_inventory(spec)
    if type(components) is not dict or set(components) != {"model", "optimizer", "scheduler", "rng"}:
        raise ValueError("all resume components required")
    if tensor_layout(components["model"]) != spec["model_layout"]:
        raise ValueError("model keys/shapes/dtypes changed")
    if components["scheduler"] != scheduler_state(epoch):
        raise ValueError("scheduler state drift")
    opt = components["optimizer"]
    if type(opt) is not dict or set(opt) != {"state", "param_groups"} or type(opt["state"]) is not dict:
        raise ValueError("exact AdamW state required")
    groups = opt["param_groups"]
    if type(groups) is not list or len(groups) != 4:
        raise ValueError("four optimizer groups required")
    expected_ids = []
    next_id = 0
    for group_name, group in zip(GROUPS, groups):
        rows = [r for r in spec["optimizer_ledger"] if r["group"] == group_name]
        ids = list(range(next_id, next_id + len(rows)))
        next_id += len(rows)
        required = {"name": group_name, "params": ids,
                    "lr": rows[0]["lr"] * epoch_multiplier(epoch),
                    "weight_decay": rows[0]["weight_decay"], "betas": (0.9, 0.999), "eps": 1e-8,
                    "amsgrad": False, "maximize": False}
        # Torch backend flags are runtime-bound but cannot enable alternate math.
        if (not isinstance(group, dict) or not set(required) <= set(group)
                or any(group[k] != v for k, v in required.items())
                or not set(group) <= set(required) | {"foreach", "capturable", "differentiable", "fused", "decoupled_weight_decay"}
                or any(group.get(k) not in (None, False) for k in ("foreach", "capturable", "differentiable", "fused"))
                or group.get("decoupled_weight_decay", True) is not True):
            raise ValueError("optimizer grouping, rates or algorithm drift")
        for param_id, row in zip(ids, rows):
            expected_ids.append(param_id)
            value = opt["state"].get(param_id)
            if type(value) is not dict or set(value) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("missing exact optimizer moments/step")
            step = value["step"]
            if isinstance(step, torch.Tensor):
                if step.device.type != "cpu" or step.numel() != 1:
                    raise ValueError("invalid optimizer step tensor")
                step = step.item()
            if type(step) not in (int, float) or step != updates:
                raise ValueError("optimizer update count drift")
            for name in ("exp_avg", "exp_avg_sq"):
                moment = value[name]
                if (not isinstance(moment, torch.Tensor) or list(moment.shape) != row["shape"]
                        or moment.dtype != torch.float32 or moment.device.type != "cpu"
                        or not torch.isfinite(moment).all()):
                    raise ValueError("invalid optimizer moment")
    if set(opt["state"]) != set(expected_ids):
        raise ValueError("unclassified optimizer state")
    rng = components["rng"]
    if (type(rng) is not dict or set(rng) != {"schema", "torch_cpu", "python", "numpy", "data_streams", "cuda"}
            or rng["data_streams"].get("last_epoch") != epoch
            or rng["data_streams"].get("seed") != spec["contract"]["seed"]):
        raise ValueError("RNG bundle or epoch/seed mismatch")
    try:
        if 'device_binding' in spec:
            from .training_runner import validate_device_rng
            validate_device_rng(rng, binding=spec['device_binding'], seed=spec['contract']['seed'])
        else:
            validate_cpu_rng(rng, seed=spec["contract"]["seed"])
    except (RuntimeError, TypeError, KeyError) as error:
        raise ValueError("invalid resumable RNG state") from error
    # Full serialization also rejects nonfinite/foreign objects before disk write.
    encode_state(components)


def validate_checkpoint(raw, *, expected_sha256, requested_spec):
    payload = decode_state(raw, expected_sha256=expected_sha256)
    if (type(payload) is not dict or set(payload) != {"schema", "spec", "components", "progress", "selection"}
            or payload["schema"] != "performance_v2_checkpoint_v1" or payload["spec"] != requested_spec):
        raise ValueError("resume contract drift")
    progress = payload["progress"]
    if (type(progress) is not dict or set(progress) != {"epoch", "updates", "elapsed_seconds"}
            or type(progress["epoch"]) is not int or not 1 <= progress["epoch"] <= 100
            or type(progress["updates"]) is not int or progress["updates"] < 1
            or type(progress["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(progress["elapsed_seconds"]) or progress["elapsed_seconds"] < 0):
        raise ValueError("invalid checkpoint progress")
    selection = payload["selection"]
    if (selection_state(selection["history"]) != selection
            or selection["last_epoch"] != progress["epoch"]):
        raise ValueError("checkpoint selection or patience drift")
    validate_components(requested_spec, payload["components"], epoch=progress["epoch"], updates=progress["updates"])
    return payload
