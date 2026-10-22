"""CPU runner integration, inactive for production; no CLI or job launcher.

Only synthetic fixtures with parameter-preserving mock updates can enter run().
The numerical step implementation is prepared for later review, not released.
No real/pretrained state importer, TEST loop, resume discovery or retry exists.
"""
import copy
import importlib.metadata
from pathlib import Path
import platform
import sys
import time

import torch

from .checkpoint import (execution_spec, scheduler_state, state_sha256, validate_components)
from .durable_lifecycle import DurableRun
from .metrics import DiagnosticAccumulator, metric_definition_sha256
from .objectives import objective_loss, class_weights_from_training_counts
from .optimization import optimizer_groups, set_epoch_learning_rates, epoch_multiplier
from .reuse import canonical, digest, scientific_contract
from .streams import TrainingStreams, capture_cpu_rng, restore_cpu_rng
from .materialization import execution_binding

PRODUCTION_EXECUTION_AUTHORIZED = False


def effective_spec(contract, adapter, model):
    """Measure effective bindings rather than accepting caller labels for them.

    Source-weight/provenance review still requires independent evidence.
    RANDOM is the only implemented initializer. Synthetic
    fixture layouts do not certify a different model as the reviewed R50 hybrid.
    """
    contract = copy.deepcopy(scientific_contract(contract))
    if contract['initialization'] != 'RANDOM':
        raise ValueError('pretrained integration is not implemented/released')
    if (getattr(model, 'objective', None) != contract['objective']
            or getattr(model, 'seed', None) != contract['seed']
            or getattr(model, 'initialization', None) != 'RANDOM'):
        raise ValueError('actual model identity differs from contract')
    if (not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cudnn.benchmark):
        raise ValueError('strict deterministic algorithms required')
    package = Path(__file__).resolve().parent
    protocol = package.parents[2] / 'configs/performance_v2/pilot_protocol_v2.json'
    source = {p.name: digest(p.read_bytes()) for p in sorted(package.glob('*.py'))}
    source['pilot_protocol_v2.json'] = digest(protocol.read_bytes())
    _, ledger = optimizer_groups(model)
    initial = state_sha256(dict(model.state_dict()))
    mapping = execution_binding(adapter.source)
    runtime = {'python': sys.version, 'platform': sys.platform, 'machine': platform.machine(),
               'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'opencv-python')},
               'torch_threads': torch.get_num_threads(), 'precision': 'float32', 'device': 'cpu',
               'deterministic_algorithms': True, 'cudnn_benchmark': False}
    actual = {
        'effective_scientific_and_execution_config_sha256': digest(canonical({
            'protocol': source['pilot_protocol_v2.json'], 'objective': contract['objective'],
            'seed': contract['seed'], 'initializer': 'RANDOM', 'device': 'cpu', 'scope': adapter.inputs.scope})),
        'source_revision_and_source_files_sha256': digest(canonical(source)),
        'topology_and_module_initialization_spec_sha256': digest(canonical({
            'actual_modules': repr(model), 'implementation': source['model.py'],
            'protocol': source['pilot_protocol_v2.json']})),
        'common_encoder_and_diagnostic_head_initial_state_sha256': state_sha256({
            k: v for k, v in model.state_dict().items() if not k.startswith('reader_head.')}),
        'source_weight_file_and_loaded_state_sha256': digest(canonical({'source': 'RANDOM-no-file', 'initial_state': initial})),
        'train_validation_cohort_identity_label_order_sha256': adapter.inputs.sha256,
        'source_image_and_supervision_input_bindings_sha256': digest(canonical({
            'input_lock': adapter.inputs.sha256, 'paths': adapter.source.sha256, 'mapping': mapping})),
        'transform_contract_and_implementation_sha256': digest(canonical({
            'contract': adapter.inputs.transform_sha256, 'implementation': source['input_adapter.py']})),
        'optimizer_groups_and_rates_schedule_sha256': digest(canonical({'ledger': ledger, 'implementation': source['optimization.py']})),
        'objective_coefficients_and_reductions_sha256': digest(canonical({'objective': contract['objective'], 'implementation': source['objectives.py']})),
        'rng_streams_seeds_shuffle_augmentation_sha256': digest(canonical({'seed': contract['seed'], 'implementation': [source['rng.py'], source['streams.py']]})),
        'runtime_hardware_precision_determinism_sha256': digest(canonical(runtime)),
        'stopping_checkpoint_selection_metric_definition_sha256': digest(canonical({'metric': metric_definition_sha256(), 'selection_source': source['selection.py']})),
    }
    contract['bindings'].update(actual)
    return execution_spec(contract, adapter.inputs, model=model, optimizer_ledger=ledger, mapping_evidence=mapping)


class SyntheticUpdates:
    """Mock AdamW moments/counters only. Never changes parameter values."""
    def __init__(self):
        self.calls = 0

    def step(self, optimizer):
        self.calls += 1
        for group in optimizer.param_groups:
            for parameter in group['params']:
                optimizer.state[parameter] = {'step': torch.tensor(float(self.calls)),
                    'exp_avg': torch.zeros_like(parameter), 'exp_avg_sq': torch.zeros_like(parameter)}


class CpuOperations:
    """Actual forward/backward/schedule/state bridge; caller controls step mode."""
    def __init__(self, model, optimizer, streams):
        self.model, self.optimizer, self.streams = model, optimizer, streams
        self.epoch, self.updates = 0, 0
        self.check_optimizer()
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('CPU float32 only; GPU adapter is not implemented')

    def check_optimizer(self):
        expected, _ = optimizer_groups(self.model)
        optimizer = self.optimizer
        if len(optimizer.param_groups) != 4:
            raise ValueError('four actual optimizer groups required')
        for actual, wanted in zip(optimizer.param_groups, expected):
            if (actual['name'] != wanted['name'] or actual['weight_decay'] != wanted['weight_decay']
                    or actual['lr'] != wanted['lr'] * (epoch_multiplier(self.epoch) if self.epoch else 1)
                    or actual['betas'] != (.9, .999)
                    or actual['eps'] != 1e-8 or actual['amsgrad'] or actual['maximize']
                    or any(actual.get(k) not in (None, False) for k in ('foreach', 'fused', 'capturable', 'differentiable'))
                    or [id(p) for p in actual['params']] != [id(p) for p in wanted['params']]):
                raise ValueError('actual optimizer parameter identities/order differ')
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError('reviewed AdamW implementation required')

    def capture(self):
        return {'model': copy.deepcopy(dict(self.model.state_dict())),
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'scheduler': scheduler_state(self.epoch), 'rng': capture_cpu_rng(self.streams)}

    def restore(self, components, *, spec, epoch, updates):
        # Caller must target a disposable candidate through review_resume.
        validate_components(spec, components, epoch=epoch, updates=updates)
        self.model.load_state_dict(components['model'], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(components['optimizer']))
        restore_cpu_rng(components['rng'], self.streams)
        self.epoch, self.updates = epoch, updates
        if state_sha256(self.capture()) != state_sha256(components):
            raise ValueError('resume recapture mismatch; discard disposable candidate')

    def train_batch(self, batch, *, objective, weights, synthetic_updates=None):
        if synthetic_updates is None:
            if not PRODUCTION_EXECUTION_AUTHORIZED:
                raise PermissionError('real optimizer updates are not released')
        elif type(synthetic_updates) is not SyntheticUpdates:
            raise PermissionError('exact synthetic updater required')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch['images'])
        losses = objective_loss(objective, output['logits'], batch['labels'], weights,
            reader_targets=batch['reader_targets'], disagreement=batch['disagreement'],
            reader_logits=output['reader_logits'])
        losses['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        if synthetic_updates is None:
            self.optimizer.step()  # unreachable until a separately reviewed release
        else:
            synthetic_updates.step(self.optimizer)
        self.updates += 1
        if synthetic_updates is not None and synthetic_updates.calls != self.updates:
            raise ValueError('mock update counter drift')
        return output['logits'].detach(), {k: float(v.detach()) for k, v in losses.items()}, float(norm)


class PreparedRunner:
    """One admitted synthetic execution, no budget initialization or retries.

    The owner supplies an existing externally anchored budget and explicit
    decision hash. Real input use and real updates remain unavailable here.
    """
    pilot_execution_authorized = False

    def __init__(self, *, spec, adapter, operations, budget, decision_sha256, clock=time.monotonic):
        self.spec, self.adapter, self.operations = copy.deepcopy(spec), adapter, operations
        self.budget, self.decision, self.clock = budget, decision_sha256, clock
        self.used = False

    def _admission(self):
        if (self.spec['scope'] != 'synthetic' or self.adapter.inputs.scope != 'synthetic'
                or self.spec['pilot_execution_authorized'] is not False):
            raise PermissionError('production execution remains unauthorized')

    def _fresh_spec(self):
        return effective_spec(self.spec['contract'], self.adapter, self.operations.model)

    def _updates(self):
        updates = SyntheticUpdates()
        updates.calls = self.operations.updates
        return updates

    def _batch(self, batch):
        return batch

    def _mocked(self):
        return True

    def run(self, *, resume=None, parent_attempt=None, reserved_session=None):
        if self.used:
            raise ValueError('runner attempt already used; no retry')
        self.used = True
        self._admission()
        op = self.operations
        op.check_optimizer()
        # Rebuild inventory and all immutable bindings before reservation. For a
        # resume, initial-state digests remain those in the requested frozen spec;
        # review_resume separately requires exact complete live recapture.
        fresh = self._fresh_spec()
        if resume is not None:
            for key in ('common_encoder_and_diagnostic_head_initial_state_sha256', 'source_weight_file_and_loaded_state_sha256'):
                fresh['contract']['bindings'][key] = self.spec['contract']['bindings'][key]
            if state_sha256(op.capture()) != state_sha256(resume.payload['components']):
                raise ValueError('restored live state differs from reviewed resume')
        elif op.epoch != 0 or op.updates != 0 or op.optimizer.state or op.streams.last_epoch != 0:
            raise ValueError('fresh run must start from exact initial state')
        if fresh != self.spec or op.streams.seed != self.spec['contract']['seed']:
            raise ValueError('effective runner state/configuration drift')
        if reserved_session is None:
            session = DurableRun(spec=self.spec, inputs=self.adapter.inputs, budget=self.budget,
                                 decision_sha256=self.decision, resume=resume, parent_attempt=parent_attempt)
        else:
            session = reserved_session
            if (type(session) is not DurableRun or session.spec != self.spec
                    or session.inputs.sha256 != self.adapter.inputs.sha256
                    or session.ticket.budget is not self.budget
                    or session.ticket.decision != self.decision
                    or session.ticket.parent != parent_attempt
                    or session.status != 'running'
                    or (resume is not None and session.progress != resume.payload['progress'])):
                raise ValueError('exact active pre-reserved lifecycle required')
            session.ticket.check()
        updates = self._updates()
        started, prior = self.clock(), session.progress['elapsed_seconds']
        def elapsed():
            value = prior + self.clock() - started
            if value < session.progress['elapsed_seconds']:
                raise ValueError('monotone run clock required')
            session.progress['elapsed_seconds'] = value
            return value
        with session.guard():
            try:
                weights = class_weights_from_training_counts([
                    sum(r.label == c for r in self.adapter.inputs.records('train')) for c in range(3)])
                for epoch in range(session.progress['epoch'] + 1, 101):
                    plan = op.streams.epoch_plan(self.adapter.inputs.ids('train'), epoch=epoch)
                    session.begin_epoch(plan)
                    op.epoch = epoch
                    set_epoch_learning_rates(op.optimizer, epoch)
                    training = DiagnosticAccumulator(self.adapter.inputs, split='train')
                    components, offset = [], 0
                    for ids in plan['batches']:
                        batch = self.adapter.batch(ids, split='train', traces=plan['augmentation'][offset:offset + len(ids)])
                        logits, losses, norm = op.train_batch(self._batch(batch), objective=self.spec['contract']['objective'],
                                                           weights=weights, synthetic_updates=updates)
                        elapsed()
                        session.record_batch(ids, observed_optimizer_updates=op.updates)
                        training.add(ids, logits.detach().cpu(), batch['labels'])
                        components.append(dict(losses, n=len(ids), gradient_norm_before_clip=norm))
                        offset += len(ids)
                    validation = DiagnosticAccumulator(self.adapter.inputs, split='val')
                    op.model.eval()
                    with torch.no_grad():
                        ids = self.adapter.inputs.ids('val')
                        for start in range(0, len(ids), 16):
                            batch = self.adapter.batch(ids[start:start + 16], split='val')
                            validation.add(batch['ids'], op.model(self._batch(batch)['images'])['logits'].detach().cpu(), batch['labels'])
                    # Ratios use epoch sums, not equal means of partial batches.
                    n = sum(x['n'] for x in components)
                    totals = {'hard': sum(x['hard_numerator'] for x in components) / sum(x['hard_denominator'] for x in components),
                              'reader': sum(x['reader'] * x['n'] for x in components) / n}
                    totals['legacy'] = (sum(x['legacy_numerator'] for x in components) / sum(x['legacy_denominator'] for x in components)
                                        if self.spec['contract']['objective'] == 'LEGACY' else 0.0)
                    session.store.put('epoch-%03d.runner.json' % epoch, canonical({
                        'train': training.summary(), 'batches': components, 'loss_totals': totals,
                        'learning_rates': {g['name']: g['lr'] for g in op.optimizer.param_groups},
                        'elapsed_seconds': elapsed(), 'optimizer_updates_mocked': self._mocked(),
                        'pilot_execution_authorized': False}))
                    selection = session.save_epoch(validation, op.capture(), elapsed_seconds=elapsed())
                    if selection['stopped']:
                        return session.finalize()
                raise ValueError('maximum epoch stop invariant violated')
            except BaseException:
                # Retain failed work timing; no retry/rollback or budget refund.
                # A disk/commit failure may leave only prior evidence; guard and
                # durable reconciliation remain authoritative in that case.
                if session.status == 'running':
                    total = prior + self.clock() - started
                    # Keep an already saved epoch's progress byte-identical for
                    # resume; separately retain later failed/administrative time.
                    try:
                        session.store.put('runner-stop.json', canonical({
                            'elapsed_seconds_including_failed_work': total,
                            'checkpoint_boundary': session._boundary,
                            'pilot_execution_authorized': False}))
                    except BaseException:
                        pass  # original error is authoritative; guard attempts terminal evidence
                    if not session._boundary:
                        try:
                            elapsed()
                        except BaseException:
                            pass
                raise
