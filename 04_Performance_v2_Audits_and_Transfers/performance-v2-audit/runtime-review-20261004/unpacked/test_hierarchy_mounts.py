"""Regression reproductions use fake systemd expansion and mount namespaces."""
import hashlib
from unittest import TestCase
from unittest.mock import Mock, patch

from test_linux_integration import Kernel
from test_linux_lifecycle import slice_path
from test_supervisor_v2 import FakeEvidence
from linux_integration import Coordinator
from supervisor_v2 import Supervisor, within


class HierarchyMountTests(TestCase):
    def setup_run(self):
        h = Kernel(); c = Coordinator(h, FakeEvidence(), 'a' * 32)
        inventory = {p: 'a' * 64 for p in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc')}
        backend, record = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
        return h, c, backend, record

    def exporter(self):
        exporter = Mock()
        exporter.export.side_effect = lambda raw, manifest, **kw: {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'readback_verified': True}
        return exporter

    def test_documented_expansion_reproduces_old_nested_control(self):
        token = 'a' * 32
        work = slice_path('pv2-' + token + '.slice')
        control = slice_path('pv2-' + token + '-control.slice')
        self.assertEqual(work, '/pv2.slice/pv2-' + token + '.slice')
        self.assertTrue(within(control, work))
        self.assertEqual(slice_path('foo-bar-baz.slice'), '/foo.slice/foo-bar.slice/foo-bar-baz.slice')

    def test_workload_kill_leaves_watchdog_alive_and_bounded(self):
        h, c, b, r = self.setup_run()
        h.processes[77] = {'pid': 77, 'start_ticks': 770, 'executable_sha256': 'c' * 64, 'cgroup': h.controller_path}
        h.argv[77] = ['controller', c.l.token]
        self.assertFalse(within(c.l.guard_path, c.l.work_path))
        self.assertFalse(within(c.l.parent, c.l.work_path))
        wd = r['watchdog']
        result = c.l.terminate(c.l.cgroup_identity())
        self.assertTrue(result['termination_confirmed'], result)
        self.assertEqual(h.process(wd['pid'])['cgroup'], wd['cgroup'])
        self.assertEqual(h.text('/sys/fs/cgroup' + wd['cgroup'] + '/pids.max'), '16')
        self.assertFalse(any(within(p['cgroup'], c.l.work_path) for p in h.processes.values()))
        self.assertFalse(any(x[0] == 'put' for x in h.calls))
        self.assertTrue(c.l.require_termination(result))
        self.assertEqual(h.process(77)['cgroup'], h.controller_path)

    def test_reported_path_change_denies_signalling(self):
        h, c, b, r = self.setup_run(); identity = c.l.cgroup_identity()
        h.slices[c.l.manager.work_unit]['path'] = '/unexpected.slice'
        before = len(h.calls)
        result = c.l.terminate(identity)
        self.assertFalse(result['termination_confirmed'])
        self.assertFalse(any(x[0] == 'command' and x[1][:2] == ['systemctl', 'kill'] for x in h.calls[before:]))

    def test_container_uses_systemd_parent_not_manual_fs_path(self):
        h, c, b, r = self.setup_run()
        argv = c.services.arguments['dockerd']
        self.assertIn('--exec-opt=native.cgroupdriver=systemd', argv)
        self.assertIn('--cgroup-parent=' + c.l.manager.work_unit, argv)
        self.assertEqual(r['cgroup'], h.slices[c.l.manager.work_unit]['path'])
        self.assertEqual(r['systemd_owner']['controller_path'], h.controller_path)

    def test_fake_rejects_missing_nonempty_and_exposes_unmount(self):
        h, c, b, r = self.setup_run()
        path = c.l.root + '/data/downloads'
        self.assertTrue(h.stat(path).st_ino)
        with self.assertRaises(FileNotFoundError): h.remove_owned(c.l.root + '/missing', 1, directory=True)
        with self.assertRaisesRegex(OSError, 'not empty'): h.remove_owned(c.l.root, h.stat(c.l.root).st_ino, directory=True)
        self.assertEqual(Supervisor(b, FakeEvidence()).run(r)['state'], 'VERIFIED_FAKE')
        with self.assertRaises(FileNotFoundError): h.stat(path)

    def test_finalization_never_stats_hidden_mounted_children(self):
        h, c, b, r = self.setup_run()
        internals = [x['identity'] for x in c.l.resources if x['kind'] == 'directory' and x['identity']['disposition'] == 'filesystem-internal']
        self.assertEqual({x['path'].rsplit('/', 1)[-1] for x in internals}, {'tmp', 'downloads'})
        Supervisor(b, FakeEvidence()).run(r)
        old_stat = h.stat
        def guarded(path):
            if path in {v['path'] for v in internals}:
                raise AssertionError('old mounted identity resolved against uncovered host')
            return old_stat(path)
        with patch.object(h, 'stat', side_effect=guarded):
            report = c.finalize(b, self.exporter(), {'files': {}}, FakeEvidence())
        self.assertTrue(report['complete'], report)
        self.assertEqual(len(report['teardown']['filesystem_internal_entries_retired']), 2)
        self.assertFalse(any(v['path'] in h.removed for v in internals))
        self.assertFalse(h.mounts or h.loops)

    def test_interruption_before_evidence_unmount_preserves_backings(self):
        h, c, b, r = self.setup_run(); Supervisor(b, FakeEvidence()).run(r)
        original = h.command
        def interrupted(argv, timeout):
            if argv == ['umount', c.l.root + '/evidence']:
                raise InterruptedError('invented interruption')
            return original(argv, timeout)
        with patch.object(h, 'command', side_effect=interrupted):
            result = c.finalize(b, self.exporter(), {'files': {}}, FakeEvidence())
        self.assertFalse(result['complete'])
        self.assertTrue(any('interruption' in e for e in result['teardown']['errors']))
        self.assertTrue(h.stat(c.l.root + '/data.ext4'))
        self.assertTrue(h.stat(c.l.root + '/evidence.ext4'))
        self.assertFalse(any(p.endswith('.ext4') for p in h.removed))

    def test_interruption_after_unmount_does_not_resolve_internal_paths(self):
        h, c, b, r = self.setup_run(); Supervisor(b, FakeEvidence()).run(r)
        original = h.remove_owned
        def interrupted(path, inode, directory=False):
            if path.endswith('/data.ext4'): raise InterruptedError('before backing disposal')
            return original(path, inode, directory)
        with patch.object(h, 'remove_owned', side_effect=interrupted):
            result = c.finalize(b, self.exporter(), {'files': {}}, FakeEvidence())
        self.assertFalse(result['complete'])
        self.assertFalse(h.mounts)
        self.assertTrue(h.stat(c.l.root + '/data.ext4'))
        with self.assertRaises(FileNotFoundError): h.stat(c.l.root + '/data/tmp')

    def test_host_mountpoint_replacement_is_not_removed(self):
        h, c, b, r = self.setup_run(); Supervisor(b, FakeEvidence()).run(r)
        h.stats[c.l.root + '/data'].st_ino += 1000
        result = c.finalize(b, self.exporter(), {'files': {}}, FakeEvidence())
        self.assertFalse(result['complete'])
        self.assertNotIn(c.l.root + '/data', h.removed)

    def test_owned_slice_inode_replacement_blocks_kill(self):
        h, c, b, r = self.setup_run()
        identity = c.l.cgroup_identity()
        h.stats[c.l.cg].st_ino += 1
        before = len(h.calls)
        result = c.l.terminate(identity)
        self.assertFalse(result['termination_confirmed'])
        self.assertFalse(any(x[0] == 'command' and x[1][:2] == ['systemctl', 'kill'] for x in h.calls[before:]))

    def test_unknown_host_child_prevents_successful_root_removal(self):
        h, c, b, r = self.setup_run(); Supervisor(b, FakeEvidence()).run(r)
        h.mkdir(c.l.root + '/unrelated')
        result = c.finalize(b, self.exporter(), {'files': {}}, FakeEvidence())
        self.assertFalse(result['complete'])
        self.assertTrue(h.stat(c.l.root + '/unrelated'))
        self.assertNotIn(c.l.root, h.removed)
