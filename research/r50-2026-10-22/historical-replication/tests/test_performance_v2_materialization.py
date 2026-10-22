"""Invented mappings/tensors only; Linux filesystem tests use private temp dirs."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from consendxformer.performance_v2 import materialization as m
from consendxformer.performance_v2 import checkpoint as ck, lifecycle as life
from consendxformer.performance_v2.reuse import canonical, digest, _json, require_exact_reuse
from test_performance_v2_evidence_mapping import fixture
from consendxformer.performance_v2.evidence_mapping import prepare_mapping


def bundle(root='/invented'):
    b, cohort, supervision, images = fixture()
    b['source_root'] = root
    for c, s in zip(cohort, supervision):
        c['path'] = root + '/rois/' + c['relative_image_path']; s['image_path'] = c['path']
    b['cohort_sha256'] = digest(canonical(cohort)); b['projection_sha256'] = digest(canonical(supervision))
    raw = canonical(b)
    return prepare_mapping(binding_bytes=raw, expected_binding_sha256=digest(raw),
            cohort_bytes=canonical(cohort), supervision_projection_bytes=canonical(supervision), image_bytes=images)


def approval(data, sidecar_root='/private/candidate'):
    ledger = m.bundle_ledger(data)
    obj = {'schema': 'pv2_mapping_approval_v1', 'scope': 'synthetic',
           'bundle_sha256': digest(canonical(ledger)), 'lineage_sha256': ledger['lineage.json'],
           'input_lock_sha256': ledger['input-lock.json'], 'path_manifest_sha256': ledger['source-paths.json'],
           'source_root': _json(data['lineage.json'])['source_root'], 'sidecar_root': sidecar_root,
           'mapping_review_approved': True, 'pilot_execution_authorized': False}
    return canonical(obj)


class BundleTests(unittest.TestCase):
    def setUp(self): self.data = bundle(); self.approval = approval(self.data)

    def check(self, data=None, raw=None):
        return m.mapping_identity(data or self.data, approval_bytes=raw or self.approval,
                                  expected_approval_sha256=digest(raw or self.approval))

    def test_complete_bundle_identity(self):
        result = self.check()
        self.assertEqual(result['bundle_sha256'], digest(canonical(m.bundle_ledger(self.data))))
        self.assertEqual(result['source_root'], '/invented')
        self.assertFalse(result['pilot_execution_authorized'])

    def test_lineage_substitution_even_with_same_lock_and_paths(self):
        data = dict(self.data); row = _json(data['lineage.json']); row['lineage_review_sha256'] = digest(b'other')
        data['lineage.json'] = canonical(row)
        with self.assertRaises(ValueError): self.check(data)

    def test_partial_extra_and_sidecar_drift(self):
        sidecar = next(k for k in self.data if k.startswith('sidecars/'))
        for change in ('missing', 'extra', 'drift'):
            data = dict(self.data)
            if change == 'missing': del data[sidecar]
            elif change == 'extra': data['rogue.json'] = b'{}'
            else: data[sidecar] += b' '
            with self.subTest(change=change), self.assertRaises(ValueError): self.check(data)

    def test_inactive_approval_rejected(self):
        obj = _json(self.approval); obj['mapping_review_approved'] = False
        with self.assertRaises(ValueError): self.check(raw=canonical(obj))

    def test_approval_digest_before_parsing(self):
        with self.assertRaisesRegex(ValueError, 'approval hash'):
            m.mapping_identity(self.data, approval_bytes=b'not json', expected_approval_sha256=digest(b'other'))

    def test_unsafe_reserved_collision_and_bounds(self):
        for name, raw in [('../x', b'x'), ('READY.json', b'x'), ('sidecars', b'x'), ('oversized', b'x' * (m.MAX_FILE + 1))]:
            data = dict(self.data); data[name] = raw
            with self.subTest(name=name), self.assertRaises(ValueError): m.bundle_ledger(data)

    def test_root_substitution_rejected_before_read(self):
        with patch.object(m, 'read_materialized', side_effect=AssertionError('read')):
            with self.assertRaisesRegex(ValueError, 'roots differ'):
                m.open_mapping_source('/other', '/private/candidate', approval_bytes=self.approval,
                                      expected_approval_sha256=digest(self.approval))

    def test_no_nonlinux_permission_fallback(self):
        with patch.object(sys, 'platform', 'win32'), patch.object(os, 'mkdir') as mkdir:
            with self.assertRaisesRegex(RuntimeError, 'native Linux'):
                m.materialize('/unused', 'candidate', self.data,
                              expected_bundle_sha256=digest(canonical(m.bundle_ledger(self.data))))
            mkdir.assert_not_called()


class ExecutionBindingTests(unittest.TestCase):
    def setUp(self):
        from test_performance_v2_runner import RunnerTests
        self.f = RunnerTests('test_validation_tensor_scaling_order_and_no_random_stream_consumption')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.mapping = {'schema': 'pv2_mapping_approval_v1', 'scope': 'synthetic',
                        'source_root': str(self.f.root), 'sidecar_root': str(self.f.root),
                        'bundle_sha256': digest(b'bundle'), 'lineage_sha256': digest(b'lineage'),
                        'approval_sha256': digest(b'approval'), 'input_lock_sha256': self.f.inputs.sha256,
                        'path_manifest_sha256': self.f.source.sha256,
                        'mapping_review_approved': True, 'pilot_execution_authorized': False}
        self.f.source.mapping_execution_binding = lambda: copy.deepcopy(self.mapping)

    def spec(self):
        from consendxformer.performance_v2.runner import effective_spec
        return effective_spec(self.f.contract, self.f.adapter, self.f.model)

    def test_all_mapping_fields_change_execution_contract(self):
        initial = self.spec()
        for key in ('lineage_sha256', 'bundle_sha256', 'approval_sha256', 'source_root', 'sidecar_root'):
            original = self.mapping[key]; self.mapping[key] = digest(b'substitution')
            with self.subTest(key=key):
                changed = self.spec()
                self.assertNotEqual(initial['contract'], changed['contract'])
                self.assertNotEqual(initial['mapping_evidence_sha256'], changed['mapping_evidence_sha256'])
            self.mapping[key] = original

    def test_checkpoint_resume_and_reuse_reject_drift_before_restore(self):
        before = self.spec(); self.mapping['lineage_sha256'] = digest(b'new lineage'); after = self.spec()
        # Checkpoint comparison is the first gate, before component parsing/loading.
        raw = ck.encode_state({'schema': 'performance_v2_checkpoint_v1', 'spec': before})
        with self.assertRaises(ValueError): ck.validate_checkpoint(raw, expected_sha256=digest(raw), requested_spec=after)
        terminal = canonical({'schema': 'performance_v2_terminal_v1', 'status': 'interrupted',
                  'epoch_boundary': True, 'run_id': before['contract']['run_id'],
                  'spec_sha256': digest(canonical(before)), 'pilot_execution_authorized': False})
        restore = Mock()
        with self.assertRaises(ValueError):
            life.review_resume(requested_spec=after, terminal_bytes=terminal, terminal_sha256=digest(terminal),
                checkpoint_bytes={}, checkpoint_hashes={}, restore_components=restore, capture_components=Mock())
        restore.assert_not_called()
        with self.assertRaises(ValueError):
            require_exact_reuse(before['contract'], after['contract'], completion_bytes=b'{}',
                completion_sha256=digest(b'{}'), manifest_bytes=b'{}', verified_output_hashes={})

    def test_mutated_mapping_cannot_keep_old_contract(self):
        spec = self.spec(); spec['mapping_evidence']['bundle_sha256'] = digest(b'new')
        spec['mapping_evidence_sha256'] = digest(canonical(spec['mapping_evidence']))
        with self.assertRaisesRegex(ValueError, 'scientific contract'): ck._validate_execution_inventory(spec)

    def test_production_rejects_unbound_synthetic_mapping(self):
        spec = self.f.spec; spec = copy.deepcopy(spec); spec['scope'] = 'production'
        with self.assertRaisesRegex(ValueError, 'production requires'): ck._validate_execution_inventory(spec)
        with self.assertRaisesRegex(ValueError, 'production requires'):
            life.require_completed_reuse(spec, completion_bytes=b'{}', completion_sha256=digest(b'{}'),
                                         manifest_bytes=b'{}', verified_output_hashes={})

    def test_source_root_mutation_rejected_before_read(self):
        self.f.source.supervision_root = self.f.root / 'other'
        with self.assertRaisesRegex(ValueError, 'roots changed'):
            self.f.source.read(self.f.inputs.ids('val')[0], 'supervision')


@unittest.skipUnless(sys.platform.startswith('linux'), 'native Linux descriptor/ACL/fsync verification required')
class NativeMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pv2-materialization-synthetic-')
        self.addCleanup(self.tmp.cleanup); self.parent = Path(self.tmp.name)
        self.images = self.parent / 'invented-images'; self.images.mkdir(mode=0o700)
        self.data = bundle(str(self.images)); self.sha = digest(canonical(m.bundle_ledger(self.data)))

    def write(self, boundary=lambda _: None):
        return m.materialize(self.parent, 'candidate', self.data, expected_bundle_sha256=self.sha, boundary=boundary)

    def test_complete_roundtrip_and_bound_separate_roots(self):
        dest = self.write()
        self.assertEqual(m.read_materialized(dest, expected_bundle_sha256=self.sha), self.data)
        raw = approval(self.data, str(dest))
        source = m.open_mapping_source(self.images, dest, approval_bytes=raw, expected_approval_sha256=digest(raw))
        self.assertEqual(source.mapping_execution_binding()['bundle_sha256'], self.sha)
        # Only synthetic canonical sidecars read; no image files are created/read.
        identity = source.inputs.ids('train')[0]
        self.assertEqual(digest(source.read(identity, 'supervision')), source.inputs.record(identity).supervision_sha256)

    def test_existing_destination_unchanged(self):
        dest = self.parent / 'candidate'; dest.mkdir(mode=0o700); (dest/'original').write_bytes(b'keep')
        with self.assertRaises(FileExistsError): self.write()
        self.assertEqual((dest/'original').read_bytes(), b'keep')

    def test_failure_at_each_write_and_commit_boundary_retained(self):
        for point in ('opened:REQUEST.json', 'written:input-lock.json', 'synced:lineage.json', 'before_ready', 'after_ready'):
            name = 'failed-' + str(len(list(self.parent.iterdir())))
            def fail(event):
                if event == point: raise OSError('invented disk failure')
            with self.subTest(point=point), self.assertRaises(OSError):
                m.materialize(self.parent, name, self.data, expected_bundle_sha256=self.sha, boundary=fail)
            dest = self.parent/name
            self.assertTrue(dest.is_dir()); self.assertTrue((dest/'FAILURE.json').is_file())
            with self.assertRaises((OSError, ValueError)):
                m.read_materialized(dest, expected_bundle_sha256=self.sha)

    def test_interruption_retains_consumed_destination(self):
        def fail(event):
            if event == 'before_ready': raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.write(fail)
        with self.assertRaises(FileExistsError): self.write()

    def test_symlink_parent_and_destination_rejected(self):
        link = self.parent/'link'; link.symlink_to(self.parent, target_is_directory=True)
        with self.assertRaises(OSError): m.materialize(link, 'candidate', self.data, expected_bundle_sha256=self.sha)
        (self.parent/'candidate').symlink_to(self.images, target_is_directory=True)
        with self.assertRaises(FileExistsError): self.write()

    def test_group_writable_parent_rejected_unchanged(self):
        self.parent.chmod(0o770)
        with self.assertRaises(ValueError): self.write()
        self.assertEqual(self.parent.stat().st_mode & 0o777, 0o770)

    def test_fifo_and_symlink_member_rejected_before_open(self):
        dest = self.write(); member = dest/'lineage.json'; member.unlink(); os.mkfifo(member, 0o600)
        with self.assertRaises(ValueError): m.read_materialized(dest, expected_bundle_sha256=self.sha)
        member.unlink(); member.symlink_to(dest/'input-lock.json')
        with self.assertRaises(ValueError): m.read_materialized(dest, expected_bundle_sha256=self.sha)

    def test_extra_content_and_tampered_bytes_rejected(self):
        dest = self.write(); (dest/'rogue').write_bytes(b'x')
        with self.assertRaises(ValueError): m.read_materialized(dest, expected_bundle_sha256=self.sha)
        (dest/'rogue').unlink(); (dest/'lineage.json').write_bytes(b'{}')
        with self.assertRaises(ValueError): m.read_materialized(dest, expected_bundle_sha256=self.sha)

    def test_disk_write_failure_preserves_partial_evidence(self):
        with patch.object(os, 'write', side_effect=OSError('synthetic ENOSPC')):
            with self.assertRaises(OSError): self.write()
        dest = self.parent/'candidate'; self.assertTrue((dest/'REQUEST.json').exists())
        self.assertFalse((dest/'READY.json').exists())

    def test_extended_acl_and_public_file_modes_rejected(self):
        with patch.object(os, 'getxattr', return_value=b'synthetic extended ACL'):
            with self.assertRaisesRegex(ValueError, 'ACL'): self.write()
        dest = self.write(); (dest/'lineage.json').chmod(0o640)
        with self.assertRaises(ValueError): m.read_materialized(dest, expected_bundle_sha256=self.sha)

    def test_regular_to_fifo_race_nonblocking_descriptor_check(self):
        import signal
        dest = self.write(); original_open = os.open
        def raced(name, flags, *args, **kwargs):
            if name == 'lineage.json' and flags & os.O_NONBLOCK:
                os.unlink(name, dir_fd=kwargs['dir_fd']); os.mkfifo(name, 0o600, dir_fd=kwargs['dir_fd'])
            return original_open(name, flags, *args, **kwargs)
        previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(AssertionError('FIFO blocked')))
        signal.alarm(5)
        try:
            with patch.object(os, 'open', side_effect=raced), self.assertRaisesRegex(ValueError, 'regular member'):
                m.read_materialized(dest, expected_bundle_sha256=self.sha)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous)
