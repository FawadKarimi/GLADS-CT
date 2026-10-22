"""Hash-authenticated non-pickle ImageNet import, separate from resume.

Preparation does not authorize calling this on real tensors. Constructor remains
the frozen RANDOM constructor; only the approved backbone state is copied.
"""
import copy
import json
from pathlib import Path

import torch

from .checkpoint import decode_state, state_sha256
from .imagenet_mapping import r50_spec, semantic_inventory
from .imagenet_legacy_receipt import verify_receipt_metadata, MISSING
from .reuse import digest
from .streams import capture_cpu_rng

EXPORT_ROOT = Path('/home/chaidir/.performance-v2-imagenet-receiver-20261012/artifacts')
EXPORT_BYTES = 125682928
EXPORT_SHA = '95da3910a39f06fe76d20c6551f709f00669e5e8d3f733e89ec9e103e0be079e'
RECEIPT_SHA = '96fbe4d4ced3d11f84aa213bd52c6e04f01e21b4f9e618baeceb49b28273c8f2'
PLAN_SHA = 'a75c41e6eebaff38031019361144fe4ed08a1e6f4fc43be94635def9ce7c5161'
ACCEPTANCE_SHA = '2319d35740fa5e9bb85d321d0e1ddc16722665f8094b800d280c6ee644331c5e'
ARTIFACT_SHA = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'


def verified_export(raw, receipt_bytes):
    """Authenticate receipt before JSON; export before typed decoding."""
    if type(receipt_bytes) is not bytes or len(receipt_bytes) > 1024**2 or digest(receipt_bytes) != RECEIPT_SHA:
        raise ValueError('receipt authentication before parse')
    receipt = json.loads(receipt_bytes)
    if (receipt['operation_plan_sha256'] != PLAN_SHA or receipt['artifact']['sha256'] != ARTIFACT_SHA
            or receipt['artifact']['bytes'] != 102530333):
        raise ValueError('original artifact / operation plan binding')
    _, spec = r50_spec()
    verify_receipt_metadata(receipt, spec)
    if (type(raw) is not bytes or len(raw) != EXPORT_BYTES or digest(raw) != EXPORT_SHA
            or receipt['export_sha256'] != EXPORT_SHA or receipt['export_bytes'] != EXPORT_BYTES):
        raise ValueError('export identity before decode')
    state = decode_state(raw, expected_sha256=EXPORT_SHA)
    expected = set(spec) - {'fc.weight', 'fc.bias'}
    if type(state) is not dict or set(state) != expected or len(state) != 318:
        raise ValueError('exact 318 supplied/bookkeeping entries')
    seen = set()
    for name, tensor in state.items():
        shape, dtype, _ = spec[name]
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                or tensor.layout != torch.strided or tuple(tensor.shape) != tuple(shape)
                or str(tensor.dtype) != dtype or tensor.requires_grad or tensor.is_conj() or tensor.is_neg()
                or not tensor.is_contiguous() or tensor.storage_offset() != 0
                or tensor.untyped_storage().nbytes() != tensor.numel()*tensor.element_size()
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError('invalid exported tensor: ' + name)
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in seen: raise ValueError('export storage alias')
        seen.add(pointer)
        if state_sha256({name: tensor}) != receipt['adapted_inventory'][name]['value_sha256']:
            raise ValueError('export value / origin drift: ' + name)
        if name in MISSING and not torch.equal(tensor, torch.zeros((), dtype=torch.int64)):
            raise ValueError('bookkeeping counter not zero')
        if name.endswith('running_var') and bool((tensor < 0).any()):
            raise ValueError('negative normalization variance')
    return state, receipt


def verify_target(model, *, admitted_objectives=('HARD',)):
    """The frozen hybrid only, without constructing another ResNet or forwarding."""
    from torchvision.models.resnet import ResNet, Bottleneck
    from .model import R50Hybrid, MODEL_ID, _check_backbone
    leaves, spec = r50_spec()
    if (type(admitted_objectives) is not tuple or not admitted_objectives
            or any(n not in ('HARD','LEGACY','SHARED-010','SEPARATE-010') for n in admitted_objectives)):
        raise ValueError('bounded target objective scope')
    if (type(model) is not R50Hybrid or model.model_id != MODEL_ID or model.objective not in admitted_objectives
            or model.initialization != 'RANDOM' or type(model.backbone) is not ResNet
            or model.backbone.groups != 1 or model.backbone.base_width != 64 or model.backbone.dilation != 1):
        raise ValueError('frozen RANDOM-constructor HARD R50 hybrid required')
    _check_backbone(model.backbone)
    if semantic_inventory(model.backbone) != dict(leaves, fc=('Identity',), avgpool=('Identity',)):
        raise ValueError('frozen backbone semantics')
    for stage, depth in enumerate((3, 4, 6, 3), 1):
        blocks = getattr(model.backbone, 'layer' + str(stage))
        if type(blocks) is not torch.nn.Sequential or len(blocks) != depth:
            raise ValueError('stage container semantics')
        for number, block in enumerate(blocks):
            if type(block) is not Bottleneck or block.expansion != 4 or block.stride != (2 if stage > 1 and number == 0 else 1):
                raise ValueError('bottleneck semantics')
    if any('forward' in m.__dict__ or m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('unreviewed forward override/hooks')
    parameters = dict(model.backbone.named_parameters(remove_duplicate=False))
    buffers = dict(model.backbone.named_buffers(remove_duplicate=False))
    expected = set(spec) - {'fc.weight', 'fc.bias'}
    if (set(parameters) & set(buffers) or set(parameters) | set(buffers) != expected
            or len({id(t) for t in [*parameters.values(), *buffers.values()]}) != 318
            or any(('parameter' if k in parameters else 'buffer') != spec[k][2] for k in expected)):
        raise ValueError('complete distinct backbone parameter/buffer inventory')


def copy_verified_backbone(model, state, receipt, streams):
    """Internal copy gate. Caller authenticates bytes first; no paths or pickle here.

    Any copy/postcondition failure poisons this disposable candidate. No rollback
    or reuse is permitted. Detailed rows cover all 318 mappings after copying.
    """
    if getattr(model, '_pv2_pretrained_applied', False) or getattr(model, '_pv2_initialization_failed', False):
        raise ValueError('one fresh initialization only; discard failed candidate')
    verify_target(model)
    target = model.state_dict()
    mapping = receipt['mapping']
    _, spec = r50_spec()
    expected = {k: 'backbone.' + k for k in spec if k not in ('fc.weight', 'fc.bias')}
    if mapping != expected or len(mapping) != 318 or set(state) != set(mapping):
        raise ValueError('exact approved source-to-backbone mapping')
    if {k for k in target if k.startswith('backbone.')} != set(mapping.values()):
        raise ValueError('complete target coverage')
    if len({t.untyped_storage().data_ptr() for t in target.values()}) != len(target):
        raise ValueError('target storage alias')
    for name, mapped in mapping.items():
        t, v = target[mapped], state[name]
        if (t.device.type != 'cpu' or tuple(t.shape) != tuple(spec[name][0]) or str(t.dtype) != spec[name][1]
                or type(v) is not torch.Tensor or v.device.type != 'cpu' or v.shape != t.shape or v.dtype != t.dtype
                or not bool(torch.isfinite(v).all())
                or state_sha256({name: v}) != receipt['adapted_inventory'][name]['value_sha256']):
            raise ValueError('shape/dtype/value pre-copy gate: ' + mapped)
        if (receipt['export_origins'][name] != ('initialized_bookkeeping' if name in MISSING else 'supplied')
                or name in MISSING and not torch.equal(v, torch.zeros((), dtype=torch.int64))):
            raise ValueError('bookkeeping/origin pre-copy gate')
    # Owned copies, including non-backbone buffers, independently check preservation.
    new = {k: v.detach().clone() for k, v in target.items() if not k.startswith('backbone.')}
    modes = {k: m.training for k, m in model.named_modules()}
    grad_flags = {k: v.requires_grad for k, v in model.named_parameters()}
    rng_before = capture_cpu_rng(streams)
    try:
        with torch.no_grad():
            for name, mapped in mapping.items(): target[mapped].copy_(state[name])
        rows = []
        after = model.state_dict()
        for name, mapped in sorted(mapping.items()):
            if not torch.equal(after[mapped], state[name]): raise ValueError('post-import value mismatch: ' + mapped)
            if state_sha256({name: after[mapped]}) != receipt['adapted_inventory'][name]['value_sha256']:
                raise ValueError('post-import value hash mismatch: ' + mapped)
            origin = receipt['export_origins'][name]
            if origin != ('initialized_bookkeeping' if name in MISSING else 'supplied'):
                raise ValueError('origin substitution')
            rows.append({'source': name, 'target': mapped, 'shape': list(after[mapped].shape),
                         'dtype': str(after[mapped].dtype), 'value_sha256': receipt['adapted_inventory'][name]['value_sha256'],
                         'origin': origin, 'equal': True})
        if (any(not torch.equal(after[k], v) for k, v in new.items())
                or set(after) != set(target) or {k: m.training for k, m in model.named_modules()} != modes
                or {k: p.requires_grad for k, p in model.named_parameters()} != grad_flags):
            raise ValueError('non-backbone/mode/gradient flag drift')
        if state_sha256(capture_cpu_rng(streams)) != state_sha256(rng_before):
            raise ValueError('initialization RNG stream drift')
        from .training_runner import initial_receipt
        initialization = initial_receipt(model)
        initialization.update(kind='IMAGENET', artifact_sha256=ARTIFACT_SHA, transfer_sha256=EXPORT_SHA,
                              loading_review_sha256=ACCEPTANCE_SHA)
        report = {'schema': 'pv2_imagenet_model_import_v1', 'entries': rows, 'verified_count': len(rows),
                  'supplied_count': sum(r['origin'] == 'supplied' for r in rows),
                  'bookkeeping_count': sum(r['origin'] == 'initialized_bookkeeping' for r in rows),
                  'non_backbone_state_sha256': state_sha256(new), 'non_backbone_unchanged': True,
                  'rng_state_sha256': state_sha256(rng_before), 'rng_unchanged': True,
                  'receipt_sha256': RECEIPT_SHA, 'export_sha256': EXPORT_SHA,
                  'operation_plan_sha256': PLAN_SHA, 'initialization': copy.deepcopy(initialization),
                  'training_authorized': False, 'pilot_execution_authorized': False}
        model._pv2_pretrained_applied = True
        return initialization, report
    except BaseException:
        model._pv2_initialization_failed = True
        raise


def initialize_fresh(model, *, kind, streams, read):
    from .training_runner import initial_receipt
    if kind == 'RANDOM':
        return initial_receipt(model), {'kind': 'RANDOM', 'pretrained_read': False, 'training_authorized': False}
    if kind == 'RADIMAGENET':
        from .radimagenet_initialization import initialize
        return initialize(model,streams=streams,read=read)
    if kind != 'IMAGENET': raise PermissionError('only reviewed RANDOM/ImageNet route')
    receipt = read(EXPORT_ROOT/'export-receipt.json', cap=1024**2)
    # Authenticate before opening the large payload, not just before parsing it.
    if digest(receipt) != RECEIPT_SHA: raise ValueError('receipt authentication before payload read')
    raw = read(EXPORT_ROOT/'backbone.typed-json', cap=EXPORT_BYTES)
    state, metadata = verified_export(raw, receipt)
    return copy_verified_backbone(model, state, metadata, streams)


def prepare_model(spec, adapter, *, device_binding, resume_parent, read):
    """Fresh = optional verified copy; resume = no pretrained file access/copy."""
    from .model import R50Hybrid
    from .streams import TrainingStreams
    from .training_runner import training_spec, resume_structure_spec
    from .supervision_admission import admit_contract
    objective = admit_contract(spec['contract'])
    model = R50Hybrid(seed=spec['contract']['seed'], objective=objective)
    streams = TrainingStreams(spec['contract']['seed'])
    if resume_parent is not None:
        # Shape/inventory/source/input/runtime gates, before checkpoint restoration.
        fresh = resume_structure_spec(spec, adapter, model, device_binding=device_binding)
        report = {'kind': spec['contract']['initialization'], 'start': 'resume', 'pretrained_read': False,
                  'pretrained_reapplied': False, 'training_authorized': False}
    else:
        if objective != 'HARD' or (spec['contract']['initialization']=='RADIMAGENET' and spec['contract']['seed'] in (43,44)):
            from .radimagenet_initialization import initialize
            receipt, report = initialize(model, streams=streams, read=read,
                                         expected_initialization=spec['initialization_receipt'])
        else:
            receipt, report = initialize_fresh(model, kind=spec['contract']['initialization'], streams=streams, read=read)
        fresh = training_spec(spec['contract'], adapter, model, initialization=receipt, device_binding=device_binding,checkpoint_retention=spec.get('checkpoint_retention'))
    if fresh != spec: raise ValueError('source/runtime/initialization/input drift')
    return model, streams, report
