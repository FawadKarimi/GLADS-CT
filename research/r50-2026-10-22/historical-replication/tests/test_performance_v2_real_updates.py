"""Invented CPU fixtures only; no research/checkpoint reads or CUDA execution."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

from test_performance_v2_runner import RunnerTests as FixtureSetup, Tiny
from consendxformer.performance_v2.training_runner import (
    TrainingOperations, TrainingRunner, training_spec, initial_receipt, validate_device_rng)
from consendxformer.performance_v2.optimization import make_adamw, set_epoch_learning_rates
from consendxformer.performance_v2.streams import TrainingStreams, capture_cpu_rng
from consendxformer.performance_v2.checkpoint import validate_components, state_sha256, decode_state
from consendxformer.performance_v2.durable_budget import PersistentBudget, AdmissionBlocked
from consendxformer.performance_v2.durable_lifecycle import DurableRun
from consendxformer.performance_v2.lifecycle import review_resume, require_completed_reuse
from consendxformer.performance_v2.reuse import canonical, digest
from consendxformer.performance_v2.timing_probe import TimingProbe

# Avoid unittest collecting the imported historical test class.
fixture_setup = FixtureSetup.setUp
del FixtureSetup


class RealUpdateTests(unittest.TestCase):
    def setUp(self):
        fixture_setup(self)
        self.binding = {'device': 'cpu', 'runtime_qualification_sha256': digest(b'invented-runtime')}
        self.op = TrainingOperations(self.model, self.optimizer, TrainingStreams(42), device_binding=self.binding)
        self.spec = training_spec(self.contract, self.adapter, self.model,
                                  initialization=initial_receipt(self.model), device_binding=self.binding)
        plan = canonical({'schema': 'performance_v2_durable_plan_v1', 'scope': 'synthetic', 'limit': 2,
                          'runs': {self.contract['run_id']: digest(canonical(self.spec))},
                          'pilot_execution_authorized': False})
        self.budget = PersistentBudget.create(self.root/'real-update-budget', plan_bytes=plan,
                         expected_plan_sha256=digest(plan), synthetic_directory_sync=lambda _: None)

    def runner(self, op=None, decision=b'invented-start'):
        return TrainingRunner(spec=self.spec, adapter=self.adapter, operations=op or self.op,
                              budget=self.budget, decision_sha256=digest(decision))

    def test_unadmitted_step_is_rejected_before_forward(self):
        with patch.object(self.model, 'forward', side_effect=AssertionError('forward')):
            with self.assertRaises(PermissionError): self.op.train_batch({}, objective='HARD', weights=None)

    def test_actual_partial_batch_update_and_complete_moments(self):
        initial = state_sha256(dict(self.model.state_dict()))
        plan = self.op.streams.epoch_plan(self.inputs.ids('train'), epoch=1)
        batch = self.adapter.batch(plan['batches'][-1], split='train', traces=plan['augmentation'][-1:])
        self.op.epoch = 1
        set_epoch_learning_rates(self.optimizer, 1)
        self.op._admitted = True  # invented unit fixture, not production authorization
        logits, loss, norm = self.op.train_batch(batch, objective='HARD', weights=torch.ones(3))
        self.assertEqual(logits.shape, (1, 3))
        self.assertGreater(loss['total'], 0)
        self.assertGreater(norm, 0)
        self.assertNotEqual(initial, state_sha256(dict(self.model.state_dict())))
        validate_components(self.spec, self.op.capture(), epoch=1, updates=1)
        self.assertEqual(len(self.optimizer.state), len(list(self.model.parameters())))

    def test_full_real_update_loop_validation_selection_and_no_production_reuse(self):
        initial = state_sha256(dict(self.model.state_dict()))
        runner = self.runner()
        receipt, manifest = runner.run()
        self.assertEqual(receipt['status'], 'synthetic_complete')
        self.assertTrue(manifest['selection']['stopped'])
        self.assertGreaterEqual(manifest['progress']['epoch'], 20)
        self.assertLessEqual(manifest['progress']['epoch'], 100)
        self.assertEqual(manifest['progress']['updates'], 2*manifest['progress']['epoch'])
        self.assertNotEqual(initial, state_sha256(dict(self.model.state_dict())))
        self.assertFalse(self.op._admitted)
        log = json.loads((self.budget.run_path(1)/'epoch-001.runner.json').read_bytes())
        self.assertFalse(log['optimizer_updates_mocked'])
        self.assertEqual([r['n'] for r in log['batches']], [16, 1])
        self.assertEqual(log['train']['n'], 17)
        with self.assertRaises(ValueError): runner.run()
        with self.assertRaises(ValueError): require_completed_reuse(self.spec,
            completion_bytes=canonical(receipt), completion_sha256=digest(canonical(receipt)),
            manifest_bytes=canonical(manifest), verified_output_hashes=manifest['files'])

    def test_boundary_resume_preserves_real_moments_and_consumes_second_slot(self):
        original = DurableRun.save_epoch
        def interrupt(session, *args, **kwargs):
            original(session, *args, **kwargs)
            raise KeyboardInterrupt('invented boundary')
        with patch.object(DurableRun, 'save_epoch', interrupt):
            with self.assertRaises(KeyboardInterrupt): self.runner().run()
        raw = self.budget.receipt_path(1).read_bytes()
        terminal = json.loads(raw)
        hashes = terminal['checkpoint_hashes']
        checkpoints = {k: (self.budget.run_path(1)/('epoch-%03d.state.json'%int(k))).read_bytes() for k in hashes}
        model = Tiny(); optimizer, _ = make_adamw(model)
        op = TrainingOperations(model, optimizer, TrainingStreams(42), device_binding=self.binding)
        admission = review_resume(requested_spec=self.spec, terminal_bytes=raw, terminal_sha256=digest(raw),
            checkpoint_bytes=checkpoints, checkpoint_hashes=hashes,
            restore_components=lambda c: op.restore(c, spec=self.spec, epoch=1, updates=2), capture_components=op.capture)
        receipt, _ = self.runner(op, b'explicit-resume').run(resume=admission, parent_attempt=1)
        self.assertEqual(receipt['status'], 'synthetic_complete')
        self.assertEqual(self.budget.inspect()['consumed'], 2)

    def test_inactive_production_record_rejected_before_io_and_reservation(self):
        runner = self.runner()
        runner.spec['scope'] = 'production'
        runner.adapter = SimpleNamespace(inputs=SimpleNamespace(scope='production'))
        record = {'schema': 'pv2_training_single_attempt_v1', 'authorized': False,
                  'spec_sha256': digest(canonical(runner.spec)), 'decision_sha256': runner.decision,
                  'scope': 'TRAIN_VAL', 'not_before': '2026-10-11T00:00:00Z', 'expires': '2026-10-11T00:15:00Z'}
        runner.authorization_bytes = canonical(record)
        runner.authorization_sha256 = digest(runner.authorization_bytes)
        with self.assertRaises(PermissionError): runner.run()
        self.assertEqual(self.budget.inspect()['consumed'], 0)

    def test_failed_read_consumes_slot_and_cannot_retry(self):
        runner = self.runner()
        with patch.object(self.adapter, 'batch', side_effect=ValueError('invented failure')):
            with self.assertRaisesRegex(ValueError, 'invented failure'): runner.run()
        self.assertEqual(json.loads(self.budget.receipt_path(1).read_bytes())['status'], 'failed')
        self.assertEqual(self.budget.inspect()['consumed'], 1)
        self.assertFalse(self.op._admitted)
        with self.assertRaises(ValueError): runner.run()

    def test_original_error_survives_evidence_write_and_terminal_commit_failure(self):
        from consendxformer.performance_v2.durable_lifecycle import _SynchronizedStore
        original = _SynchronizedStore.put
        def write(store, name, raw):
            if name == 'runner-stop.json': raise OSError('invented full disk')
            return original(store, name, raw)
        with (patch.object(self.adapter, 'batch', side_effect=ValueError('original failure')),
             patch.object(_SynchronizedStore, 'put', write),
             patch('consendxformer.performance_v2.durable_budget.Admission.terminate', side_effect=OSError('terminal disk'))):
            with self.assertRaisesRegex(ValueError, 'original failure'): self.runner().run()
        self.assertEqual(self.budget.inspect()['consumed'], 1)
        self.assertFalse((self.budget.run_path(1)/'completion.json').exists())

    def test_initialization_and_runtime_drift_rejected(self):
        with torch.no_grad(): self.model.diagnostic_head.bias.add_(.1)
        with self.assertRaisesRegex(ValueError, 'identity'): self.runner().run()
        self.assertEqual(self.budget.inspect()['consumed'], 0)
        op = self.op
        op.binding['runtime_qualification_sha256'] = digest(b'different')
        with self.assertRaisesRegex(ValueError, 'device'): self.runner().run()

    def test_checkpoint_write_error_survives_terminal_failure(self):
        from consendxformer.performance_v2.durable_lifecycle import _SynchronizedStore
        original = _SynchronizedStore.put
        def write(store, name, raw):
            if name == 'epoch-001.state.json': raise OSError('original checkpoint failure')
            return original(store, name, raw)
        with (patch.object(_SynchronizedStore, 'put', write),
              patch('consendxformer.performance_v2.durable_budget.Admission.terminate', side_effect=ValueError('terminal failure'))):
            with self.assertRaisesRegex(OSError, 'original checkpoint failure'): self.runner().run()
        self.assertEqual(self.budget.inspect()['consumed'], 1)
        self.assertFalse((self.budget.run_path(1)/'completion.json').exists())

    def test_cuda_rng_envelope_checked_offline_without_gpu_calls(self):
        binding = dict(self.binding, device='cuda', index=0, uuid='invented-GPU', name='invented',
                       total_memory=1, cuda_runtime='invented', cudnn=1, driver='invented')
        state = dict(capture_cpu_rng(TrainingStreams(42)), schema='pv2_single_cuda_rng_v1',
                     cuda={'binding': binding, 'state': torch.arange(12, dtype=torch.uint8)})
        with patch.object(torch.cuda, 'get_rng_state', side_effect=AssertionError('CUDA queried')):
            validate_device_rng(state, binding=binding, seed=42)
            changed = copy.deepcopy(state); changed['cuda']['binding']['uuid'] = 'other'
            with self.assertRaises(ValueError): validate_device_rng(changed, binding=binding, seed=42)
            changed = copy.deepcopy(state); changed['cuda']['state'] = torch.zeros(2, dtype=torch.float32)
            with self.assertRaises(ValueError): validate_device_rng(changed, binding=binding, seed=42)

    def test_timing_probe_reads_training_only_and_never_completes_or_reuses(self):
        original = self.adapter.batch
        def train_only(ids, *, split, **kwargs):
            self.assertEqual(split, 'train')
            return original(ids, split=split, **kwargs)
        probe = TimingProbe(spec=self.spec, adapter=self.adapter, operations=self.op,
                            budget=self.budget, decision_sha256=digest(b'invented-timing'))
        with patch.object(self.adapter, 'batch', side_effect=train_only):
            report = probe.run(warmup=1, measured=1)
        self.assertFalse(report['training_complete'])
        self.assertFalse(report['validation_evaluated'])
        self.assertEqual(self.op.updates, 2)
        self.assertEqual(json.loads(self.budget.receipt_path(1).read_bytes())['status'], 'interrupted')
        self.assertEqual(self.budget.inspect()['consumed'], 1)
        self.assertFalse((self.budget.run_path(1)/'completion.json').exists())
        self.assertFalse(self.op._admitted)

    def test_gpu_identity_preflight_positive_and_uuid_drift_with_mocks_only(self):
        self.op.binding = dict(self.binding, device='cuda', index=0, uuid='invented-GPU', name='invented',
                               total_memory=10, cuda_runtime='invented', cudnn=1, driver='invented')
        self.op.device = torch.device('cuda:0')
        props = SimpleNamespace(uuid='invented-GPU', name='invented', total_memory=10)
        with (patch.object(torch.cuda, 'get_device_properties', return_value=props),
              patch.object(torch.version, 'cuda', 'invented'),
              patch.object(torch.backends.cudnn, 'version', return_value=1)):
            self.op.check_device()
            props.uuid = 'other'
            with self.assertRaisesRegex(ValueError, 'identity'): self.op.check_device()

    def test_timing_probe_does_not_recycle_small_fixture_or_expand_bound(self):
        probe = TimingProbe(spec=self.spec, adapter=self.adapter, operations=self.op,
                            budget=self.budget, decision_sha256=digest(b'invented-timing'))
        with self.assertRaisesRegex(ValueError, 'recycle'): probe.run()
        self.assertEqual(self.budget.inspect()['consumed'], 0)

    def test_separate_objective_real_update_reaches_reader_and_preserves_stream_traces(self):
        model = Tiny('SEPARATE-010'); optimizer, _ = make_adamw(model)
        op = TrainingOperations(model, optimizer, TrainingStreams(42), device_binding=self.binding)
        hard_plan = TrainingStreams(42).epoch_plan(self.inputs.ids('train'), epoch=1)
        plan = op.streams.epoch_plan(self.inputs.ids('train'), epoch=1)
        self.assertEqual(plan, hard_plan)
        hard = Tiny()
        for name, value in hard.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        batch = self.adapter.batch(plan['batches'][-1], split='train', traces=plan['augmentation'][-1:])
        before = model.reader_head.weight.detach().clone()
        op._admitted = True; op.epoch = 1
        set_epoch_learning_rates(optimizer, 1)
        _, losses, _ = op.train_batch(batch, objective='SEPARATE-010', weights=torch.ones(3))
        self.assertGreater(losses['reader'], 0)
        self.assertIsNotNone(model.backbone.weight.grad)
        self.assertIsNotNone(model.diagnostic_head.weight.grad)
        self.assertIsNotNone(model.reader_head.weight.grad)
        self.assertFalse(torch.equal(before, model.reader_head.weight))


if __name__ == '__main__': unittest.main()
