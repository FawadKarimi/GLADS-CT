"""Fake Linux observations; directory-sync injection is explicit on Windows."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_linux_lifecycle import FakeHost
from test_supervisor_v2 import FakeEvidence
from linux_lifecycle import LinuxLifecycle
from linux_host import annotate
from supervisor_evidence import FileEvidence, sync_directory
from linux_integration import recover_records


class TerminationTests(unittest.TestCase):
    def make(self):
        h = FakeHost(); l = LinuxLifecycle(h, FakeEvidence(), 'a' * 32)
        return l, h

    def assert_denied(self, l, h, result):
        h.calls.clear()
        for call in (lambda: l.detach_volume('data', {'loop': '/dev/loop10'}, result),
                     lambda: l.partial_cleanup(result),
                     lambda: l.export_and_ack(Mock(), {}, {}, result)):
            with self.assertRaises(ValueError):
                call()
        self.assertFalse(any(c[0] == 'command' and c[1][0] in ('umount', 'losetup') for c in h.calls))

    def test_truthy_failed_dictionary_never_tears_down(self):
        l, h = self.make()
        self.assert_denied(l, h, {'termination_confirmed': False, 'errors': ['failed']})

    def test_malformed_results(self):
        for result in (True, 1, 'yes', {}, {'termination_confirmed': True}, None):
            l, h = self.make(); self.assert_denied(l, h, result)

    def test_failed_bound_result(self):
        l, h = self.make(); h.busy = True
        result = l.terminate(l.cgroup_identity())
        h.busy = False
        self.assert_denied(l, h, result)

    def test_foreign_tampered_superseded_and_stale(self):
        for mutate in ('foreign', 'timestamp', 'superseded', 'stale', 'numeric_true', 'errors'):
            l, h = self.make(); result = l.terminate(l.cgroup_identity())
            if mutate == 'foreign': result['token'] = 'b' * 32
            elif mutate == 'timestamp': result['observed_at'] = float('nan')
            elif mutate == 'superseded': l.terminate(l.cgroup_identity())
            elif mutate == 'stale': h.time = 11
            elif mutate == 'numeric_true': result['termination_confirmed'] = 1
            else: result['errors'] = ['TERM failed']
            self.assert_denied(l, h, result)

    def test_repopulation_after_success_refused(self):
        l, h = self.make(); result = l.terminate(l.cgroup_identity()); h.busy = True
        self.assert_denied(l, h, result)

    def test_changed_boot_cgroup_and_socket_refused(self):
        for changed in ('inode', 'boot', 'listener'):
            l, h = self.make(); result = l.terminate(l.cgroup_identity())
            if changed == 'inode': h.inode += 1
            elif changed == 'boot': h.files['/proc/sys/kernel/random/boot_id'] = 'new'
            else: h.files['/proc/net/unix'] = l.runtime + '/docker.sock'
            self.assert_denied(l, h, result)

    def test_fresh_observation_after_volume_inspection(self):
        l, h = self.make(); result = l.terminate(l.cgroup_identity())
        v = {'loop': '/dev/loop10'}
        def changed(*args):
            h.busy = True
            return v
        with patch.object(l, 'volume_identity', side_effect=changed), self.assertRaises(ValueError):
            l.detach_volume('data', v, result)
        self.assertFalse(any(c[0] == 'command' and c[1][0] == 'umount' for c in h.calls))


class DurabilityTests(unittest.TestCase):
    def test_directory_sync_order(self):
        with tempfile.TemporaryDirectory() as t:
            calls = []
            e = FileEvidence(Path(t) / 'e', directory_sync=lambda p: calls.append(('dir', str(p))))
            self.assertEqual(calls, [('dir', t), ('dir', str(e.root))])
            calls.clear()
            e.append('ownership', {'id': 1})
            self.assertEqual(calls, [('dir', str(e.root))])
            calls.clear()
            import os
            original = os.replace
            def replace(a, b):
                calls.append(('replace', str(b))); original(a, b)
            with patch('supervisor_evidence.os.replace', side_effect=replace): e.seal()
            self.assertEqual([x[0] for x in calls], ['dir', 'replace', 'dir'])

    def test_sync_failure_blocks_ownership_effect(self):
        with tempfile.TemporaryDirectory() as t:
            sync = Mock(); e = FileEvidence(Path(t) / 'e', directory_sync=sync)
            l = LinuxLifecycle(FakeHost(), e, 'a' * 32)
            sync.side_effect = OSError('directory sync failed'); effect = Mock()
            with self.assertRaises(OSError): l.acquire('x', '/a', effect, lambda: {})
            effect.assert_not_called()
            with self.assertRaises(OSError): e.complete('VERIFIED_FAKE')

    def test_post_replace_sync_failure_poisoned(self):
        with tempfile.TemporaryDirectory() as t:
            e = FileEvidence(Path(t) / 'e', directory_sync=lambda p: None)
            e.append('ownership', {})
            e.directory_sync = Mock(side_effect=[None, OSError('post replace')])
            with self.assertRaises(OSError): e.seal()
            self.assertTrue((e.root / 'report-hashes.json').exists())
            with self.assertRaises(OSError): e.complete('VERIFIED_FAKE')

    def test_native_sync_closes_fd_on_failure(self):
        with patch('supervisor_evidence.os.name', 'posix'), patch('supervisor_evidence.os.O_DIRECTORY', 65536, create=True), patch('supervisor_evidence.os.O_NOFOLLOW', 131072, create=True), patch('supervisor_evidence.os.open', return_value=9), patch('supervisor_evidence.os.fsync', side_effect=OSError('sync')), patch('supervisor_evidence.os.close') as close:
            with self.assertRaises(OSError): sync_directory('/invented')
            close.assert_called_once_with(9)

    def test_recovery_bound_records_and_unresolved_intent(self):
        with tempfile.TemporaryDirectory() as t:
            e = FileEvidence(Path(t) / 'e', directory_sync=lambda p: None)
            l = LinuxLifecycle(FakeHost(), e, 'a' * 32)
            l.acquire('directory', l.root, lambda: None, lambda: {'path': l.root, 'inode': 5})
            recovered = recover_records(e.root, l.token, l.boot)
            self.assertTrue(recovered['admission_allowed'])
            e.append('provision-intent', {'kind': 'x', 'requested': l.root + '/x', 'boot_id': l.boot, 'phase': 'intent'})
            e.seal()
            self.assertFalse(recover_records(e.root, l.token, l.boot)['admission_allowed'])
            (e.root / '0001-provision-intent.json').write_bytes(b'{}')
            with self.assertRaises(ValueError): recover_records(e.root, l.token, l.boot)

    def test_python310_exception_fallback(self):
        class OldError(Exception): add_note = None
        error = OldError('primary'); annotate(error, 'cleanup failed')
        self.assertEqual(str(error), 'primary')
        self.assertEqual(error.__notes__, ['cleanup failed'])

    def test_controller_sources_python310_syntax(self):
        import ast
        for name in ('linux_host.py', 'linux_lifecycle.py', 'linux_integration.py', 'watchdog_worker.py', 'supervisor_evidence.py'):
            ast.parse((Path(__file__).parent / name).read_text(), feature_version=(3, 10))
