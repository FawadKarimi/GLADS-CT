"""Invented Linux files, fake monotonic clock and mocked subprocesses ONLY."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import json

from linux_host import LinuxHost, mount_records, resolve_cgroup
from linux_lifecycle import LinuxLifecycle, IndependentWatchdog
from test_supervisor_v2 import FakeEvidence

MOUNTS = '1 0 8:1 / / rw - ext4 /dev/root rw\n29 1 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n'

def slice_path(name):
    parts = name.removesuffix('.slice').split('-')
    return '/' + '/'.join('-'.join(parts[:i]) + '.slice' for i in range(1, len(parts) + 1))


class FakeHost:
    is_fake = True
    def __init__(self):
        self.time = 0
        self.files = {'/proc/self/mountinfo': MOUNTS, '/proc/self/cgroup': '0::/supervisor\n',
                      '/proc/sys/kernel/random/boot_id': 'boot', '/proc/net/unix': '',
                      '/sys/fs/cgroup/cgroup.controllers': 'cpu memory pids',
                      '/sys/fs/cgroup/cgroup.subtree_control': 'cpu memory pids'}
        self.calls = []; self.inode = 111; self.fail_term = False; self.busy = False
    def text(self, path):
        if path.endswith('/cgroup.events'):
            return 'populated ' + ('1' if self.busy else '0')
        if path.endswith('/cgroup.procs'):
            return ''
        return self.files[path]
    def now(self):
        return self.time
    def pause(self, sec):
        self.time += sec
    def stat(self, path):
        return SimpleNamespace(st_ino=self.inode)
    def put(self, path, value):
        self.calls.append(('put', path, value)); self.files[path] = value
    def mkdir(self, path):
        self.calls.append(('mkdir', path))
    def children(self, path):
        return []
    def digest(self, path):
        return 'a' * 64
    def command(self, argv, timeout):
        self.calls.append(('command', argv, timeout))
        if self.fail_term and '--signal=TERM' in argv:
            raise TimeoutError('TERM deadline')
        if argv[0] == 'busctl':
            unit = argv[7]; props = dict((argv[i], argv[i + 2]) for i in range(10, len(argv) - 1, 3))
            root = '/sys/fs/cgroup' + slice_path(unit)
            self.files.update({root + '/memory.max': props['MemoryMax'], root + '/memory.swap.max': '0',
                               root + '/pids.max': props['TasksMax'], root + '/cpu.max': str(int(props['CPUQuotaPerSecUSec']) // 10) + ' 100000'})
        if argv[:2] == ['systemctl', 'list-units']:
            return json.dumps([{'unit': 'fixture.service'}]).encode()
        if argv[:2] == ['systemctl', 'show']:
            unit = argv[2]
            path = slice_path(unit) if unit.endswith('.slice') else '/pv2work' + 'a' * 32 + '.slice/fixture.service'
            return f'Id={unit}\nActiveState=active\nControlGroup={path}\n'.encode()
        return b''


class LifecycleTests(unittest.TestCase):
    def make(self):
        h, e = FakeHost(), FakeEvidence()
        return LinuxLifecycle(h, e, 'a' * 32), h, e

    def test_live_disabled(self):
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            LinuxHost()
        h = FakeHost(); h.is_fake = False
        with self.assertRaisesRegex(ValueError, 'disabled'):
            LinuxLifecycle(h, FakeEvidence(), 'a' * 32)

    def test_mount_parser_and_namespace_refusal(self):
        self.assertEqual(resolve_cgroup(MOUNTS, '0::/outside'), ('/sys/fs/cgroup', '/outside'))
        with self.assertRaises(ValueError):
            resolve_cgroup(MOUNTS.replace(' / /sys', ' /restricted /sys'), '0::/')
        with self.assertRaises(ValueError):
            resolve_cgroup(MOUNTS + MOUNTS, '0::/')

    def test_intent_precedes_mutation_and_observation(self):
        l, h, e = self.make()
        def mutate():
            self.assertEqual(e.ledger[-1][1]['phase'], 'intent')
        l.acquire('invented', '/owned', mutate, lambda: {'inode': 1})
        self.assertEqual(e.ledger[-1][1]['phase'], 'observed')

    def test_intent_write_failure_prevents_effect(self):
        l, h, e = self.make(); e.fail = 'provision-intent'; effect = Mock()
        with self.assertRaises(OSError):
            l.acquire('invented', '/owned', effect, lambda: {})
        effect.assert_not_called()

    def test_observed_write_failure_retains_cleanup_identity(self):
        l, h, e = self.make(); e.fail = 'provision-observed'
        with self.assertRaises(OSError):
            l.acquire('invented', '/owned', lambda: None, lambda: {'inode': 12})
        self.assertEqual(l.resources[0]['identity']['inode'], 12)

    def test_resource_controls_written_and_read_back(self):
        l, h, e = self.make(); identity = l.provision_cgroup()
        self.assertEqual(identity['inode'], 111)
        self.assertEqual(h.files[l.cg + '/memory.max'], str(6 * 1024**3))
        self.assertEqual(h.files[l.cg + '/memory.swap.max'], '0')
        self.assertEqual(h.files[l.cg + '/pids.max'], '256')

    def test_missing_controller_refused_before_creation(self):
        l, h, e = self.make(); h.files['/sys/fs/cgroup/cgroup.controllers'] = 'cpu'
        with self.assertRaises(ValueError):
            l.provision_cgroup()
        self.assertEqual(h.calls, [])

    def test_failed_term_does_not_skip_subtree_kill(self):
        l, h, e = self.make(); h.fail_term = True
        result = l.terminate(l.cgroup_identity())
        self.assertTrue(result['termination_confirmed'])
        self.assertTrue(result['errors'])
        self.assertTrue(any(c[0] == 'command' and '--signal=KILL' in c[1] for c in h.calls))

    def test_changed_identity_never_killed(self):
        l, h, e = self.make(); identity = l.cgroup_identity(); h.inode += 1
        result = l.terminate(identity)
        self.assertFalse(result['termination_confirmed'])
        self.assertFalse(any(c[0] == 'command' and c[1][:2] == ['systemctl', 'kill'] for c in h.calls))

    def test_nonempty_times_out(self):
        l, h, e = self.make(); h.busy = True
        result = l.terminate(l.cgroup_identity())
        self.assertFalse(result['termination_confirmed'])
        self.assertLess(h.time, 10.1)

    def test_private_listener_blocks_success(self):
        l, h, e = self.make(); h.files['/proc/net/unix'] = l.runtime + '/docker.sock'
        self.assertFalse(l.terminate(l.cgroup_identity())['termination_confirmed'])

    def test_watchdog_owner_loss_and_stale_ack_do_readback(self):
        for alive, ack in ((False, False), (True, True)):
            l, h, e = self.make(); h.busy = True
            result = IndependentWatchdog(l, l.cgroup_identity(), 1200, e).supervise(lambda: alive, lambda: ack)
            self.assertFalse(result['termination_confirmed'])
            self.assertTrue(e.ledger)

    def test_watchdog_expiry(self):
        l, h, e = self.make()
        result = IndependentWatchdog(l, l.cgroup_identity(), .2, e).supervise(lambda: True, lambda: False)
        self.assertTrue(result['termination_confirmed'])
        self.assertGreaterEqual(h.time, .2)

    def test_export_wrong_ack_preserves_volume(self):
        l, h, e = self.make(); exporter = Mock()
        exporter.export.return_value = {'manifest_sha256': 'wrong', 'readback_verified': True}
        with patch.object(l, 'detach_volume') as detach, self.assertRaises(ValueError):
            l.export_and_ack(exporter, {'files': []}, {}, l.terminate(l.cgroup_identity()))
        detach.assert_not_called()

    def test_export_ack_precedes_teardown(self):
        import hashlib
        import json
        l, h, e = self.make(); exporter = Mock(); manifest = {'files': []}
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        exporter.export.return_value = {'manifest_sha256': digest, 'readback_verified': True}
        def detach(*args):
            self.assertEqual(e.ledger[-1][0], 'export-ack')
        with patch.object(l, 'detach_volume', side_effect=detach):
            self.assertTrue(l.export_and_ack(exporter, manifest, {}, l.terminate(l.cgroup_identity()))['backing_retained'])

    def test_daemon_arguments_private_and_inventory_bound(self):
        l, h, e = self.make()
        with self.assertRaises(ValueError):
            l.daemon_arguments({})
        args = l.daemon_arguments({p: 'a' * 64 for p in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc')})
        self.assertIn('--bridge=none', args['dockerd'])
        self.assertIn('--containerd=' + l.runtime + '/containerd.sock', args['dockerd'])

    def test_native_command_timeout_preserves_primary_when_kill_fails(self):
        host = object.__new__(LinuxHost)
        process = Mock(); process.poll.return_value = None; process.kill.side_effect = OSError('cleanup')
        with patch.object(host, 'start', return_value=process), patch.object(host, 'now', side_effect=[0, 11]):
            with self.assertRaises(TimeoutError) as caught:
                host.command(['invented'], 10)
        self.assertIn('cleanup', str(caught.exception.__notes__))

    def test_native_output_overflow(self):
        host = object.__new__(LinuxHost); process = Mock(); process.poll.return_value = 0
        with patch.object(host, 'start', return_value=process), patch.object(host, 'now', return_value=0), patch.object(host, 'read', return_value=b'x' * 65536):
            with self.assertRaisesRegex(ValueError, 'overflow'):
                host.command(['invented'], 10)
