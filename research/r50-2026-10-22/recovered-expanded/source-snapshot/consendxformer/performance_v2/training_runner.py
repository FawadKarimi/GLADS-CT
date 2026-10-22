"""Real-update integration with an exact locked UUID adapter; no pretrained reader or TEST interface.

Production admission requires a separately approved, exact single-attempt record.
Invented CPU fixtures can exercise updates but receive synthetic receipts only.
The historical PreparedRunner remains a mock-only entry point.
"""
import copy
from datetime import datetime, timezone
import json
import torch

from .checkpoint import state_sha256, scheduler_state, validate_components
from .model import R50Hybrid
from .objectives import objective_loss
from .reuse import canonical, digest, _hash, _json
from .runner import CpuOperations, PreparedRunner, effective_spec
from .streams import capture_cpu_rng, restore_cpu_rng, validate_cpu_rng


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def validate_device_rng(state, *, binding, seed):
    """Validate saved CUDA bytes without querying/initializing any GPU."""
    if binding['device'] == 'cpu':
        return validate_cpu_rng(state, seed=seed)
    if (set(state) != {'schema', 'torch_cpu', 'python', 'numpy', 'data_streams', 'cuda'} or state['schema'] != 'pv2_single_cuda_rng_v1'
            or type(state['cuda']) is not dict
            or set(state['cuda']) != {'binding', 'state'} or state['cuda']['binding'] != binding):
        raise ValueError('exact device and RNG binding required')
    raw = state['cuda']['state']
    if (not isinstance(raw, torch.Tensor) or raw.device.type != 'cpu'
            or raw.dtype != torch.uint8 or raw.ndim != 1 or not 1 <= raw.numel() <= 65536):
        raise ValueError('bounded CUDA RNG byte vector required')
    normalized = dict(state, schema='performance_v2_cpu_rng_v1', cuda='not_used_cpu_only')
    return validate_cpu_rng(normalized, seed=seed)


def validate_device_binding(binding):
    if type(binding) is not dict or binding.get('device') not in ('cpu', 'cuda'):
        raise ValueError('explicit single-device binding required')
    fields = {'device', 'runtime_qualification_sha256'}
    if binding['device'] == 'cuda':
        fields |= {'index', 'uuid', 'name', 'total_memory', 'cuda_runtime', 'cudnn', 'driver'}
        if (type(binding.get('index')) is not int or binding['index'] < 0
                or type(binding.get('total_memory')) is not int or binding['total_memory'] <= 0
                or any(not isinstance(binding.get(k), str) or not binding[k]
                       for k in ('uuid', 'name', 'cuda_runtime', 'driver'))
                or type(binding.get('cudnn')) is not int):
            raise ValueError('complete observed GPU/runtime identity required')
    if set(binding) != fields or not _hash(binding['runtime_qualification_sha256']):
        raise ValueError('unclassified device fields or missing qualified runtime')


def training_spec(contract, adapter, model, *, initialization, device_binding, checkpoint_retention=None):
    """Build inventory on a CPU copy before moving the actual model to GPU.

    A supplied receipt is a binding, not evidence of provenance clearance by itself.
    No pretrained transfer happens here. The non-pickle initializer separately
    verifies the actual backbone AND preserved newly initialized modules.
    """
    validate_device_binding(device_binding)
    if adapter.inputs.scope == 'production' and type(model) is not R50Hybrid:
        raise ValueError('production requires the existing exact R50 hybrid')
    if (type(initialization) is not dict or initialization.get('kind') != contract['initialization']
            or not _hash(initialization.get('backbone_state_sha256'))
            or not _hash(initialization.get('new_modules_state_sha256'))):
        raise ValueError('initialization receipt required')
    observed = cpu_copy(dict(model.state_dict()))
    if (state_sha256({k: v for k, v in observed.items() if k.startswith('backbone.')})
            != initialization['backbone_state_sha256']
            or state_sha256({k: v for k, v in observed.items() if not k.startswith('backbone.')})
            != initialization['new_modules_state_sha256']):
        raise ValueError('loaded backbone/new-module identity mismatch')
    common = {'kind', 'backbone_state_sha256', 'new_modules_state_sha256'}
    if initialization['kind'] == 'RANDOM':
        if set(initialization) != common:
            raise ValueError('random initialization has no weight artifact')
    elif (set(initialization) != common | {'artifact_sha256', 'transfer_sha256', 'loading_review_sha256'}
          or any(not _hash(initialization[k]) for k in ('artifact_sha256', 'transfer_sha256', 'loading_review_sha256'))):
        raise ValueError('verified artifact/transfer/loading disposition required')
    # Constructor initialization remains RANDOM: the verified transfer
    # is represented by its exact receipt, never by pretending a loader exists.
    placeholder = copy.deepcopy(contract)
    placeholder['initialization'] = 'RANDOM'
    spec = effective_spec(placeholder, adapter, copy.deepcopy(model).cpu())
    spec['contract']['initialization'] = contract['initialization']
    spec['contract']['bindings']['source_weight_file_and_loaded_state_sha256'] = digest(canonical(initialization))
    spec['device_binding'] = copy.deepcopy(device_binding)
    spec['training_integration'] = 'pv2_real_updates_v1'
    spec['initialization_receipt'] = copy.deepcopy(initialization)
    spec['contract']['bindings']['effective_scientific_and_execution_config_sha256'] = digest(canonical({
        'prepared': spec['contract']['bindings']['effective_scientific_and_execution_config_sha256'],
        'initialization': initialization, 'device': device_binding, 'integration': spec['training_integration']}))
    spec['contract']['bindings']['runtime_hardware_precision_determinism_sha256'] = digest(canonical({
        'prepared': spec['contract']['bindings']['runtime_hardware_precision_determinism_sha256'],
        'device': device_binding}))
    if checkpoint_retention is not None:
        from .retention import enabled
        spec['checkpoint_retention']=copy.deepcopy(checkpoint_retention);enabled(spec)
    return spec


def initial_receipt(model):
    """RANDOM-only helper; never creates a pretrained clearance record."""
    state = cpu_copy(dict(model.state_dict()))
    return {'kind': 'RANDOM',
            'backbone_state_sha256': state_sha256({k: v for k, v in state.items() if k.startswith('backbone.')}),
            'new_modules_state_sha256': state_sha256({k: v for k, v in state.items() if not k.startswith('backbone.')})}


def resume_structure_spec(spec, adapter, model, *, device_binding):
    """Recheck executable/input/inventory bindings without applying initial weights.

    Only the three original initialization-dependent bindings are restored from
    the authenticated requested spec. Checkpoint bytes/moments/RNG still undergo
    existing review_resume + validate_components + native restore/recapture.
    """
    fresh = training_spec(dict(spec['contract'], initialization='RANDOM'), adapter, model,
                          initialization=initial_receipt(model), device_binding=device_binding)
    fresh['initialization_receipt'] = copy.deepcopy(spec['initialization_receipt'])
    fresh['contract']['initialization'] = spec['contract']['initialization']
    for k in ('source_weight_file_and_loaded_state_sha256',
              'common_encoder_and_diagnostic_head_initial_state_sha256',
              'effective_scientific_and_execution_config_sha256'):
        fresh['contract']['bindings'][k] = spec['contract']['bindings'][k]
    if 'checkpoint_retention' in spec:
        from .retention import enabled
        fresh['checkpoint_retention']=copy.deepcopy(spec['checkpoint_retention']);enabled(fresh)
    return fresh


class TrainingOperations(CpuOperations):
    def __init__(self, model, optimizer, streams, *, device_binding):
        validate_device_binding(device_binding)
        self.model, self.optimizer, self.streams = model, optimizer, streams
        self.binding = copy.deepcopy(device_binding)
        self.device = torch.device('cpu' if self.binding['device'] == 'cpu' else 'cuda:' + str(self.binding['index']))
        self.epoch, self.updates = 0, 0
        self._admitted = False
        self.check_optimizer()
        if any(p.device != self.device or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('all actual parameters must use the bound FP32 device')

    def check_device(self):
        if self.device.type == 'cpu':
            return
        props = torch.cuda.get_device_properties(self.device)
        observed = {'uuid': str(getattr(props, 'uuid', '')), 'name': props.name,
                    'total_memory': props.total_memory, 'cuda_runtime': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version()}
        if not observed['uuid'] or any(observed[k] != self.binding[k] for k in observed):
            raise ValueError('fresh actual GPU/runtime identity mismatch')
        # Driver and live job/ownership/free-memory gates remain the bound outer
        # host preflight's responsibility; never infer exclusivity from CUDA.

    def capture(self):
        rng = capture_cpu_rng(self.streams)
        if self.device.type == 'cuda':
            rng = dict(rng, schema='pv2_single_cuda_rng_v1',
                       cuda={'binding': copy.deepcopy(self.binding),
                             'state': torch.cuda.get_rng_state(self.device).cpu().clone()})
        return {'model': cpu_copy(dict(self.model.state_dict())),
                'optimizer': cpu_copy(self.optimizer.state_dict()),
                'scheduler': scheduler_state(self.epoch), 'rng': rng}

    def restore(self, components, *, spec, epoch, updates):
        # The caller first obtains review_resume on a disposable candidate.
        if spec['device_binding'] != self.binding:
            raise ValueError('resume device/runtime drift')
        validate_components(spec, components, epoch=epoch, updates=updates)
        validate_device_rng(components['rng'], binding=self.binding, seed=self.streams.seed)
        if self.device.type == 'cuda':
            # Native-format validation on a private generator before mutating
            # the model or process-global RNG. Never exercised by CPU fixtures.
            torch.Generator(device=self.device).set_state(components['rng']['cuda']['state'])
        self.model.load_state_dict(components['model'], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(components['optimizer']))
        cpu_rng = dict(components['rng'], schema='performance_v2_cpu_rng_v1', cuda='not_used_cpu_only')
        restore_cpu_rng(cpu_rng, self.streams)
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state(components['rng']['cuda']['state'], self.device)
        self.epoch, self.updates = epoch, updates
        self.check_optimizer()
        if state_sha256(self.capture()) != state_sha256(components):
            raise ValueError('resume recapture mismatch; discard candidate, no rollback')

    def train_batch(self, batch, *, objective, weights, synthetic_updates=None):
        if not self._admitted or synthetic_updates is not None:
            raise PermissionError('real updates require admitted integration scope')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch['images'])
        losses = objective_loss(objective, output['logits'], batch['labels'], weights.to(self.device),
            reader_targets=batch['reader_targets'], disagreement=batch['disagreement'],
            reader_logits=output['reader_logits'])
        losses['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return output['logits'].detach(), {k: float(v.detach()) for k, v in losses.items()}, float(norm)


class TrainingRunner(PreparedRunner):
    """One real-update attempt; synthetic receipts cannot qualify for reuse."""
    def __init__(self, *, authorization_bytes=None, authorization_sha256=None, **kwargs):
        super().__init__(**kwargs)
        self.authorization_bytes, self.authorization_sha256 = authorization_bytes, authorization_sha256

    def _admission(self):
        allowed = (TrainingOperations,)
        if self.spec['device_binding']['device'] == 'cuda':
            from .cuda_identity import UuidBoundOperations
            allowed = (UuidBoundOperations,)
        if type(self.operations) not in allowed:
            raise PermissionError('exact reviewed training operations required')
        if self.spec['device_binding'] != self.operations.binding:
            raise ValueError('device binding drift')
        self.operations.check_device()
        if self.spec['scope'] != self.adapter.inputs.scope or self.spec['pilot_execution_authorized'] is not False:
            raise ValueError('historical flags/scope must be preserved')
        if self.spec['scope'] == 'synthetic':
            if self.operations.device.type != 'cpu' or self.authorization_bytes is not None:
                raise PermissionError('synthetic preparation is CPU-only, never production admission')
        else:
            if (type(self.authorization_bytes) is not bytes or not _hash(self.authorization_sha256)
                    or digest(self.authorization_bytes) != self.authorization_sha256):
                raise PermissionError('separate authenticated execution authorization required')
            a = _json(self.authorization_bytes)
            required = {'schema', 'authorized', 'spec_sha256', 'decision_sha256', 'scope', 'not_before', 'expires'}
            if (set(a) != required or a['schema'] != 'pv2_training_single_attempt_v1'
                    or a['authorized'] is not True or a['scope'] != 'TRAIN_VAL'
                    or a['spec_sha256'] != digest(canonical(self.spec)) or a['decision_sha256'] != self.decision):
                raise PermissionError('inactive or mismatched single-attempt authorization')
            import re
            if any(not isinstance(a[k], str) or re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', a[k]) is None
                   for k in ('not_before', 'expires')):
                raise PermissionError('exact UTC authorization timestamps required')
            start, end = [datetime.strptime(a[k], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                          for k in ('not_before', 'expires')]
            if not 0 < (end-start).total_seconds() <= 900 or not start <= datetime.now(timezone.utc) < end:
                raise PermissionError('authorization window rejected; no automatic extension')
        # No optimizer work until exclusive durable admission succeeds.
        # _updates is called only after DurableRun constructor has reserved.

    def _fresh_spec(self):
        # For resume validate actual components separately; preserve original
        # initialization receipts rather than pretending updated tensors are initial.
        if self.operations.epoch:
            model = copy.deepcopy(self.operations.model).cpu()
            return resume_structure_spec(self.spec, self.adapter, model, device_binding=self.operations.binding)
        return training_spec(self.spec['contract'], self.adapter, self.operations.model,
                             initialization=self.spec['initialization_receipt'], device_binding=self.operations.binding,checkpoint_retention=self.spec.get('checkpoint_retention'))

    def _updates(self):
        self.operations._admitted = True
        return None

    def _batch(self, batch):
        return {k: v.to(self.operations.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _mocked(self):
        return False

    def run(self, **kwargs):
        try:
            return super().run(**kwargs)
        finally:
            self.operations._admitted = False
