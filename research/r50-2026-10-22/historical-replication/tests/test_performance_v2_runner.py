"""Invented NPY bytes and tiny modules; no research files or optimizer steps."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from consendxformer.performance_v2.bindings import FrozenInputs, InputRecord
from consendxformer.performance_v2.input_adapter import (BoundInputAdapter, FrozenFileSource,
    decode_grayscale, supervision_bytes, transform)
from consendxformer.performance_v2.runner import (CpuOperations, PreparedRunner, SyntheticUpdates, effective_spec)
from consendxformer.performance_v2.optimization import make_adamw
from consendxformer.performance_v2.streams import TrainingStreams
from consendxformer.performance_v2.durable_budget import PersistentBudget, AdmissionBlocked
from consendxformer.performance_v2.lifecycle import review_resume, require_completed_reuse
from consendxformer.performance_v2.checkpoint import state_sha256
from consendxformer.performance_v2.reuse import canonical, digest
from test_performance_v2_lifecycle import lock_object, Fixture


class Tiny(torch.nn.Module):
    def __init__(self, objective='HARD'):
        super().__init__()
        self.seed, self.initialization, self.objective = 42, 'RANDOM', objective
        with torch.random.fork_rng():
            torch.manual_seed(53)
            self.backbone = torch.nn.Linear(3, 4)
            self.diagnostic_head = torch.nn.Linear(4, 3)
            self.reader_head = torch.nn.Linear(4, 3) if objective == 'SEPARATE-010' else None

    def forward(self, x):
        z = self.backbone(x.mean(dim=(2, 3)))
        return {'logits': self.diagnostic_head(z),
                'reader_logits': None if self.reader_head is None else self.reader_head(z)}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pv2-runner-invented-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old_determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms, self.old_determinism)
        rows = lock_object()
        paths = {}
        for row in rows['records']:
            identity = row['nodule_id']
            stream = io.BytesIO()
            np.save(stream, np.full((64, 64), .25 + .1 * row['label'], dtype=np.float32), allow_pickle=False)
            raw = stream.getvalue()
            row['image_sha256'] = digest(raw)
            sidecar = supervision_bytes(InputRecord.parse(row))
            row['supervision_sha256'] = digest(sidecar)
            # Non-research suffixes deliberately distinguish fixture files.
            image, supervision = identity + '.image.bin', identity + '.supervision.json'
            (self.root/image).write_bytes(raw)
            (self.root/supervision).write_bytes(sidecar)
            paths[identity] = {'image': image, 'supervision': supervision}
        raw = canonical(rows)
        self.inputs = FrozenInputs(raw, expected_sha256=digest(raw))
        self.manifest = {'schema': 'performance_v2_source_paths_v1', 'scope': 'synthetic',
                         'input_lock_sha256': self.inputs.sha256, 'records': paths}
        raw = canonical(self.manifest)
        self.source = FrozenFileSource(self.root, self.inputs, raw, expected_sha256=digest(raw))
        self.adapter = BoundInputAdapter(self.inputs, self.source, expected_transform_sha256=self.inputs.transform_sha256)
        self.contract = Fixture().contract
        self.contract['objective'] = 'HARD'
        self.model = Tiny()
        self.optimizer, _ = make_adamw(self.model)
        self.op = CpuOperations(self.model, self.optimizer, TrainingStreams(42))
        self.spec = effective_spec(self.contract, self.adapter, self.model)
        plan = canonical({'schema': 'performance_v2_durable_plan_v1', 'scope': 'synthetic', 'limit': 2,
                          'runs': {self.contract['run_id']: digest(canonical(self.spec))},
                          'pilot_execution_authorized': False})
        self.plan_sha = digest(plan)
        self.budget = PersistentBudget.create(self.root/'budget', plan_bytes=plan, expected_plan_sha256=self.plan_sha,
                                             synthetic_directory_sync=lambda _: None)

    def runner(self, decision=b'fixture-start', operations=None):
        return PreparedRunner(spec=self.spec, adapter=self.adapter, operations=operations or self.op,
                              budget=self.budget, decision_sha256=digest(decision))

    def test_validation_tensor_scaling_order_and_no_random_stream_consumption(self):
        rng = torch.get_rng_state().clone()
        batch = self.adapter.batch(self.inputs.ids('val'), split='val')
        self.assertEqual(batch['images'].shape, (5, 3, 224, 224))
        expected = (torch.tensor(.25) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(batch['images'][0, :, 80, 80], expected)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_digest_before_decoder_and_supervision_drift(self):
        identity = self.inputs.ids('val')[0]
        p = self.root / self.manifest['records'][identity]['image']
        p.write_bytes(b'not an array')
        with patch('consendxformer.performance_v2.input_adapter.decode_grayscale', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.adapter.batch([identity], split='val')
        p = self.root / self.manifest['records'][identity]['supervision']
        p.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.adapter.batch([identity], split='val')

    def test_unknown_test_or_cross_split_rejected_before_io(self):
        with patch.object(self.source, 'read', side_effect=AssertionError('source read')):
            for split, ids in [('test', ['invented-test']), ('val', self.inputs.ids('train'))]:
                with self.assertRaises(ValueError):
                    self.adapter.batch(ids, split=split)

    def test_manifest_hash_membership_traversal_alias_and_symlink_rejection(self):
        for name in ('../outside', '/absolute', 'a\\b', 'a:b', 'a//b', './a'):
            obj = copy.deepcopy(self.manifest)
            obj['records'][self.inputs.ids('val')[0]]['image'] = name
            raw = canonical(obj)
            with self.assertRaises(ValueError):
                FrozenFileSource(self.root, self.inputs, raw, expected_sha256=digest(raw))
        with self.assertRaisesRegex(ValueError, 'digest'):
            FrozenFileSource(self.root, self.inputs, b'bad json', expected_sha256=digest(b'other'))
        identity = self.inputs.ids('val')[0]
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.source.read(identity, 'image')

    def test_npy_shape_dtype_range_trailing_data_and_allocation_bomb_rejected(self):
        for array in (np.ones((65, 64)), np.ones((64, 64), dtype=bool),
                      np.full((64, 64), np.nan), np.full((64, 64), -1)):
            stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
            with self.assertRaises(ValueError): decode_grayscale(stream.getvalue())
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': '<f4', 'fortran_order': False, 'shape': (10**9, 10**9)})
        with patch.object(np, 'load', side_effect=AssertionError('allocated')):
            with self.assertRaises(ValueError): decode_grayscale(header.getvalue())
        raw = self.source.read(self.inputs.ids('val')[0], 'image')
        with self.assertRaises(ValueError): decode_grayscale(raw + b'extra')

    def test_geometry_trace_and_partial_train_batch(self):
        plan = TrainingStreams(42).epoch_plan(self.inputs.ids('train'), epoch=1)
        batch = self.adapter.batch(plan['batches'][-1], split='train', traces=plan['augmentation'][-1:])
        self.assertEqual(batch['images'].shape, (1, 3, 224, 224))
        trace = copy.deepcopy(plan['augmentation'][0]); trace['translate_pixels'][0] += 1
        with self.assertRaises(ValueError): transform(torch.zeros(1, 64, 64), trace=trace)
        with self.assertRaises(ValueError): self.adapter.batch(self.inputs.ids('val'), split='val', traces=[])

    def test_full_mock_lifecycle_durable_complete_validation_only_and_no_parameter_updates(self):
        initial = state_sha256(dict(self.model.state_dict()))
        runner = self.runner()
        receipt, manifest = runner.run()
        self.assertEqual(receipt['status'], 'synthetic_complete')
        self.assertEqual(manifest['progress']['epoch'], 20)
        self.assertEqual(manifest['progress']['updates'], 40)
        self.assertEqual(manifest['selection']['best_epoch'], 1)
        self.assertEqual(initial, state_sha256(dict(self.model.state_dict())))
        self.assertEqual(json.loads(self.budget.committed_receipt(1))['status'], 'synthetic_complete')
        log = json.loads((self.budget.run_path(1)/'epoch-001.runner.json').read_bytes())
        self.assertEqual([x['n'] for x in log['batches']], [16, 1])
        self.assertEqual(log['train']['n'], 17)
        with self.assertRaises(ValueError): runner.run()
        with self.assertRaises(ValueError):
            require_completed_reuse(self.spec, completion_bytes=canonical(receipt), completion_sha256=digest(canonical(receipt)),
                                    manifest_bytes=canonical(manifest), verified_output_hashes=manifest['files'])

    def test_failed_source_read_consumes_slot_preserves_timing_and_no_completion(self):
        with patch.object(self.adapter, 'batch', side_effect=ValueError('invented corruption')):
            with self.assertRaises(ValueError): self.runner().run()
        receipt = json.loads(self.budget.receipt_path(1).read_bytes())
        self.assertEqual(receipt['status'], 'failed')
        self.assertGreater(receipt['progress']['elapsed_seconds'], 0)
        self.assertFalse((self.budget.run_path(1)/'completion.json').exists())
        fresh_model = Tiny(); fresh_optimizer, _ = make_adamw(fresh_model)
        fresh = CpuOperations(fresh_model, fresh_optimizer, TrainingStreams(42))
        with self.assertRaises(AdmissionBlocked): self.runner(b'duplicate', operations=fresh).run()

    def test_real_updates_and_production_dispatch_rejected_before_io_or_admission(self):
        with self.assertRaises(PermissionError):
            self.op.train_batch({}, objective='HARD', weights=None)
        runner = self.runner(); runner.spec['scope'] = 'production'
        with patch.object(self.adapter, 'batch', side_effect=AssertionError('read')):
            with self.assertRaises(PermissionError): runner.run()
        self.assertEqual(self.budget.inspect()['attempts'], [])

    def test_model_objective_initial_weights_and_optimizer_drift_reject(self):
        self.model.objective = 'LEGACY'
        with self.assertRaises(ValueError): self.runner().run()
        self.model.objective = 'HARD'
        with torch.no_grad(): self.model.diagnostic_head.bias.add_(.1)
        with self.assertRaises(ValueError): self.runner().run()
        self.optimizer.param_groups[0]['eps'] = .1
        with self.assertRaises(ValueError): CpuOperations(self.model, self.optimizer, TrainingStreams(42))

    def test_optimizer_mutation_after_construction_rejected_before_reservation(self):
        self.optimizer.param_groups[0]['betas'] = (.8, .99)
        with self.assertRaises(ValueError): self.runner().run()
        self.assertEqual(self.budget.inspect()['consumed'], 0)

    def test_transform_horizontal_flip_and_integer_fill_oracle(self):
        gray = torch.linspace(0, 1, 64).expand(64, 64).unsqueeze(0)
        trace = {'id': 'invented', 'horizontal_flip': True, 'vertical_flip': False,
                 'clockwise_angle': 0., 'translate_fraction': [2/224, 0.], 'translate_pixels': [2, 0]}
        actual = transform(gray, trace=trace)
        resized = torch.nn.functional.interpolate(gray[None], size=(224, 224), mode='bilinear',
                                                  align_corners=False, antialias=True)[0].flip(-1)
        expected = torch.zeros_like(resized); expected[:, :, 2:] = resized[:, :, :-2]
        # Compare in grayscale units with the preserved transform probe's
        # documented 5e-5 tolerance (normalization magnifies roundoff).
        grayscale = actual * torch.tensor([.229, .224, .225])[:, None, None] + torch.tensor([.485, .456, .406])[:, None, None]
        torch.testing.assert_close(grayscale, expected.repeat(3, 1, 1), atol=5e-5, rtol=0)

    def test_source_mapping_mutation_rejected(self):
        identity = self.inputs.ids('val')[0]
        self.source.paths[identity]['image'] = 'different.bin'
        with self.assertRaisesRegex(ValueError, 'mapping changed'):
            self.source.read(identity, 'image')

    def test_reader_objective_uses_same_bridge_without_parameter_updates(self):
        model = Tiny('SEPARATE-010'); optimizer, _ = make_adamw(model)
        op = CpuOperations(model, optimizer, TrainingStreams(42))
        plan = op.streams.epoch_plan(self.inputs.ids('train'), epoch=1)
        batch = self.adapter.batch(plan['batches'][-1], split='train', traces=plan['augmentation'][-1:])
        initial = state_sha256(dict(model.state_dict()))
        _, losses, _ = op.train_batch(batch, objective='SEPARATE-010', weights=torch.ones(3), synthetic_updates=SyntheticUpdates())
        self.assertGreater(losses['reader'], 0)
        self.assertIsNotNone(model.reader_head.weight.grad)
        self.assertEqual(initial, state_sha256(dict(model.state_dict())))

    def test_mid_epoch_interruption_rejects_checkpoint_rollback(self):
        original = self.adapter.batch
        calls = 0
        def stop_next_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4: raise KeyboardInterrupt('invented mid-epoch interruption')
            return original(*args, **kwargs)
        with patch.object(self.adapter, 'batch', side_effect=stop_next_epoch):
            with self.assertRaises(KeyboardInterrupt): self.runner().run()
        receipt = json.loads(self.budget.receipt_path(1).read_bytes())
        self.assertEqual(receipt['status'], 'interrupted')
        self.assertFalse(receipt['epoch_boundary'])
        # Mid-epoch event cannot silently roll back to the retained epoch state.
        with self.assertRaises(ValueError):
            review_resume(requested_spec=self.spec, terminal_bytes=canonical(receipt), terminal_sha256=digest(canonical(receipt)),
                          checkpoint_bytes={}, checkpoint_hashes={}, restore_components=lambda _: None, capture_components=lambda: {})

    def test_boundary_resume_recaptures_all_live_state_and_consumes_second_slot(self):
        from consendxformer.performance_v2.durable_lifecycle import DurableRun
        original = DurableRun.save_epoch
        def interrupt_after_saved(session, *args, **kwargs):
            original(session, *args, **kwargs)
            raise KeyboardInterrupt('invented saved boundary')
        with patch.object(DurableRun, 'save_epoch', interrupt_after_saved):
            with self.assertRaises(KeyboardInterrupt): self.runner().run()
        raw = self.budget.receipt_path(1).read_bytes(); terminal = json.loads(raw)
        self.assertTrue(terminal['epoch_boundary'])
        hashes = terminal['checkpoint_hashes']
        checkpoints = {key: (self.budget.run_path(1)/('epoch-%03d.state.json' % int(key))).read_bytes() for key in hashes}
        model = Tiny(); optimizer, _ = make_adamw(model)
        op = CpuOperations(model, optimizer, TrainingStreams(42))
        admission = review_resume(requested_spec=self.spec, terminal_bytes=raw, terminal_sha256=digest(raw),
            checkpoint_bytes=checkpoints, checkpoint_hashes=hashes,
            restore_components=lambda c: op.restore(c, spec=self.spec, epoch=1, updates=2),
            capture_components=op.capture)
        receipt, _ = self.runner(b'explicit-resume', operations=op).run(resume=admission, parent_attempt=1)
        self.assertEqual(receipt['status'], 'synthetic_complete')
        self.assertEqual(self.budget.inspect()['consumed'], 2)


if __name__ == '__main__':
    unittest.main()
