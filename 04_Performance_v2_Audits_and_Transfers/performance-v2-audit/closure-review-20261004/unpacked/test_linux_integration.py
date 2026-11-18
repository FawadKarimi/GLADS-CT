"""Integrated coordinator through an invented in-memory Linux OS boundary.

No Docker command, native process, mount, cgroup or watchdog service is executed.
The fake interprets explicit argv and records requests for assertions.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from linux_integration import Coordinator
from test_supervisor_v2 import FakeEvidence
from test_linux_lifecycle import FakeHost, MOUNTS, slice_path
from supervisor_v2 import within


class Node:
    def __init__(self, path, directory=False):
        self.path, self.directory = path, directory
        self.name = path.rsplit('/', 1)[-1]
    def __str__(self): return self.path
    def is_dir(self): return self.directory
    def is_symlink(self): return False


class Kernel(FakeHost):
    def __init__(self):
        super().__init__()
        self.stats = {}; self.mounts = []; self.loops = {}; self.processes = {}; self.units = {}; self.argv = {}
        self.slices = {}; self.covered = {}; self.bindings = {}; self.hidden = {}; self.removed = []
        self.stats.update({p: SimpleNamespace(st_ino=i) for i, p in enumerate(('/', '/var', '/var/tmp', '/run', '/sys/fs/cgroup'), 1)})
        self.next_pid = 100; self.next_inode = 500; self.next_loop = 10
        self.unarmed = False; self.corrupt = False; self.escape = False; self.spill = False
        self.blocks = [b'invented output', b'']; self.worker_seen_before_daemon = False
        self.controller_path = '/system.slice/pv2-' + 'a' * 32 + '-controller.service'
        self.files['/proc/self/cgroup'] = '0::' + self.controller_path + '\n'
        control = '/sys/fs/cgroup' + self.controller_path
        self.files.update({control + '/memory.max': str(256 * 1024**2), control + '/memory.swap.max': '0',
                           control + '/pids.max': '16', control + '/cpu.max': '25000 100000'})

    def mkdir(self, path):
        if path in self.stats: raise FileExistsError(path)
        self.stat(path.rsplit('/', 1)[0])
        self.stats[path] = SimpleNamespace(st_ino=self.next_inode); self.next_inode += 1
        self.bind_entry(path)
        self.calls.append(('mkdir', path))

    def stat(self, path):
        if path not in self.stats: raise FileNotFoundError(path)
        return self.stats[path]

    def bind_entry(self, path):
        owners = [m for m in self.mounts if within(path, m['mount'])]
        self.bindings[path] = max(owners, key=lambda m: len(m['mount']))['id'] if owners else 1

    def allocate_exclusive(self, path, capacity):
        if path in self.stats: raise FileExistsError(path)
        self.stats[path] = SimpleNamespace(st_ino=self.next_inode, st_size=capacity, st_blocks=capacity // 512)
        self.next_inode += 1
        self.bind_entry(path)

    def free_bytes(self, path): return 20 * 1024**3

    def digest(self, path):
        if path in self.files:
            return hashlib.sha256(self.files[path].encode()).hexdigest()
        return 'a' * 64

    def write_exclusive(self, path, raw):
        if path in self.files: raise FileExistsError(path)
        self.files[path] = raw.decode()
        self.stats[path] = SimpleNamespace(st_ino=self.next_inode); self.next_inode += 1
        self.bind_entry(path)

    def assert_no_symlinks(self, path): pass

    def writable_paths(self, pid): return ['/unbounded/bad'] if self.spill else []

    def socket_owned(self, pid, path): return pid in self.processes

    def filesystem(self, path):
        return {'inode_capacity': 65536, 'free_bytes': 100000, 'free_inodes': 1000,
                'capacity_bytes': 64 * 1024**2 if path.startswith('/run/') else 8 * 1024**3}

    def process(self, pid):
        if pid not in self.processes: raise FileNotFoundError(str(pid))
        p = deepcopy(self.processes[pid])
        if self.escape and 'dockerd' in p['cgroup']: p['cgroup'] = '/escaped'
        return p

    def children(self, path):
        if path == '/proc': return [Node('/proc/' + str(pid)) for pid in self.processes]
        return []

    def put(self, path, value):
        if path.startswith('/sys/fs/cgroup'):
            raise AssertionError('manual writes into manager-owned cgroups forbidden')
        super().put(path, value)

    def text(self, path):
        if path == '/proc/self/mountinfo':
            return MOUNTS + ''.join(f"{m['id']} 1 {m['device']} / {m['mount']} rw - {m['filesystem']} {m['source']} rw\n" for m in self.mounts)
        if path == '/proc/net/unix':
            if any('dockerd' in p['cgroup'] for p in self.processes.values()):
                return self.runtime + '/docker.sock\n' + self.runtime + '/containerd.sock'
            return ''
        if path.endswith('/cgroup.events'):
            cg = path.removeprefix('/sys/fs/cgroup').removesuffix('/cgroup.events')
            return 'populated ' + str(int(any(within(p['cgroup'], cg) for p in self.processes.values())))
        if path.endswith('/cgroup.procs'):
            cg = path.removeprefix('/sys/fs/cgroup').removesuffix('/cgroup.procs')
            return '\n'.join(str(pid) for pid, p in self.processes.items() if within(p['cgroup'], cg))
        if path.startswith('/proc/') and path.endswith('/cmdline'):
            return '\0'.join(self.argv[int(path.split('/')[2])]) + '\0'
        if 'watchdog-armed.json' in path and self.unarmed: raise FileNotFoundError(path)
        if path not in self.files: raise FileNotFoundError(path)
        return self.files[path]

    def command(self, argv, timeout):
        self.calls.append(('command', list(argv), timeout))
        if argv[0] == 'busctl':
            unit = argv[7]
            if unit in self.slices: raise FileExistsError(unit)
            path = slice_path(unit)
            self.slices[unit] = {'active': True, 'path': path}
            self.stats['/sys/fs/cgroup' + path] = SimpleNamespace(st_ino=self.next_inode); self.next_inode += 1
            props = dict((argv[i], argv[i + 2]) for i in range(10, len(argv) - 1, 3))
            root = '/sys/fs/cgroup' + path
            self.files.update({root + '/memory.max': props['MemoryMax'], root + '/memory.swap.max': '0',
                               root + '/pids.max': props['TasksMax'], root + '/cpu.max': str(int(props['CPUQuotaPerSecUSec']) // 10) + ' 100000'})
            return b'o "/invented/job"'
        if argv[:2] == ['systemctl', 'list-units']:
            return json.dumps([{'unit': u} for u in self.units]).encode()
        if argv[0] == 'losetup':
            if '--find' in argv:
                loop = '/dev/loop' + str(self.next_loop); self.next_loop += 1
                self.loops[loop] = argv[-1]; return loop.encode()
            if '--detach' in argv:
                self.loops.pop(argv[-1]); return b''
            if '--associated' in argv:
                backing = argv[argv.index('--associated') + 1]
                rows = [{'name': k, 'back-file': v} for k, v in self.loops.items() if v == backing]
            else:
                loop = argv[-1]; rows = [{'name': loop, 'back-file': self.loops[loop]}]
            return json.dumps({'loopdevices': rows}).encode()
        if argv[0] == 'blkid': return b'invented-uuid'
        if argv[0] == 'mount':
            fs = argv[argv.index('-t') + 1]; source, target = argv[-2:]
            self.covered[target] = self.stat(target)
            self.mounts.append({'id': 100 + len(self.mounts), 'device': '7:10', 'root': '/', 'mount': target,
                                'filesystem': fs, 'source': source})
            self.stats[target] = SimpleNamespace(st_ino=self.next_inode); self.next_inode += 1
            self.bind_entry(target)
            if fs == 'tmpfs': self.runtime = target
            return b''
        if argv[0] == 'umount':
            target = argv[1]
            if target not in self.covered: raise FileNotFoundError(target)
            mount_id = next(m['id'] for m in self.mounts if m['mount'] == target)
            for p in list(self.stats):
                if self.bindings.get(p) == mount_id:
                    self.hidden[(mount_id, p)] = self.stats.pop(p)
                    self.bindings.pop(p)
            for p in list(self.files):
                if within(p, target): del self.files[p]
            self.mounts = [m for m in self.mounts if m['mount'] != target]
            self.stats[target] = self.covered.pop(target); self.bind_entry(target)
            return b''
        if argv[0] == 'systemd-run':
            unit = next(a.split('=', 1)[1] for a in argv if a.startswith('--unit='))
            slice_name = next(a.split('=', 1)[1] for a in argv if a.startswith('--slice='))
            role = unit.rsplit('-', 1)[-1].split('.')[0]
            pid = self.next_pid; self.next_pid += 1
            p = {'pid': pid, 'start_ticks': pid * 10, 'executable_sha256': 'a' * 64,
                 'cgroup': slice_path(slice_name) + '/' + unit}
            self.processes[pid] = p; self.units[unit] = pid; self.argv[pid] = argv[argv.index('--') + 1:]
            if role == 'watchdog':
                control = '/sys/fs/cgroup' + p['cgroup']
                self.files.update({control + '/memory.max': str(256 * 1024**2), control + '/memory.swap.max': '0',
                                   control + '/pids.max': '16', control + '/cpu.max': '25000 100000'})
                wd = dict(p, outside_workload=True, lifetime_seconds=1200)
                token = unit.split('-')[1]
                path = '/var/tmp/pv2-' + token + '/evidence/watchdog-armed.json'
                self.files[path] = json.dumps({'token': token, 'boot_id': 'boot', 'watchdog': wd})
            elif role in ('dockerd', 'containerd'):
                self.worker_seen_before_daemon = any('watchdog' in x for x in self.units)
            return b''
        if argv[:2] == ['systemctl', 'show']:
            unit = argv[2]
            if unit.endswith('-controller.service'):
                return f'Id={unit}\nActiveState=active\nControlGroup={self.controller_path}\n'.encode()
            if unit.endswith('.slice'):
                row = self.slices[unit]; active = 'active' if row['active'] else 'inactive'; path = row['path'] if row['active'] else ''
                return f'Id={unit}\nActiveState={active}\nControlGroup={path}\n'.encode()
            pid = self.units[unit]
            if pid not in self.processes: return f'Id={unit}\nActiveState=inactive\nControlGroup=\n'.encode()
            p = self.process(pid)
            return f"Id={unit}\nActiveState=active\nMainPID={pid}\nControlGroup={p['cgroup']}\n".encode()
        if argv[:2] == ['systemctl', 'kill']:
            if '--signal=KILL' in argv: self.processes.pop(self.units[argv[-1]], None)
            return b''
        if argv[:2] == ['systemctl', 'stop']:
            if argv[2].endswith('.slice'):
                row = self.slices[argv[2]]
                if any(within(p['cgroup'], row['path']) for p in self.processes.values()): raise OSError('slice busy')
                row['active'] = False; self.stats.pop('/sys/fs/cgroup' + row['path'])
                return b''
            self.processes.pop(self.units[argv[2]], None); return b''
        return b''

    def start(self, argv):
        self.command(argv, 10)
        return SimpleNamespace(poll=lambda: 0)

    def read(self, process, maximum, timeout): return self.blocks.pop(0)

    def remove_owned(self, path, inode, directory=False):
        if path.startswith('/sys/fs/cgroup'): raise AssertionError('manual cgroup removal forbidden')
        if self.stat(path).st_ino != inode: raise ValueError('inode changed')
        if directory and any(p != path and within(p, path) for p in set(self.stats) | set(self.files)):
            raise OSError('directory not empty: ' + path)
        self.stats.pop(path); self.bindings.pop(path, None); self.removed.append(path)


class IntegrationTests(unittest.TestCase):
    def make(self):
        h, journal, evidence = Kernel(), FakeEvidence(), FakeEvidence()
        c = Coordinator(h, journal, 'a' * 32)
        inventory = {p: 'a' * 64 for p in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc')}
        return h, c, inventory, evidence

    def test_integrated_provision_operation_cleanup(self):
        h, c, inventory, evidence = self.make()
        result = c.execute(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1', evidence)
        self.assertEqual(result['state'], 'VERIFIED_FAKE', result)
        self.assertTrue(h.worker_seen_before_daemon)
        self.assertFalse(any(m['mount'].endswith('/data') for m in h.mounts))
        self.assertEqual(evidence.stdout['state'], 'EVIDENCE_PENDING')
        self.assertFalse(result['execution_approved'])

    def test_unarmed_watchdog_blocks_daemons_and_operation(self):
        h, c, inventory, evidence = self.make(); h.unarmed = True
        result = c.execute(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1', evidence)
        self.assertEqual(result['state'], 'FAILED')
        self.assertFalse(h.worker_seen_before_daemon)
        self.assertFalse(any('-operation.service' in str(call) for call in h.calls))

    def test_snapshot_rejects_escape_spill_config_and_limits(self):
        for mutation in ('escape', 'spill', 'config', 'limit'):
            h, c, inventory, evidence = self.make()
            backend, record = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
            if mutation == 'escape': h.escape = True
            elif mutation == 'spill': h.spill = True
            elif mutation == 'config': h.files[c.l.runtime + '/dockerd.json'] = '{"bad":true}'
            else: h.files[c.l.cg + '/memory.max'] = 'max'
            from supervisor_v2 import verify_snapshot
            with self.assertRaises((ValueError, KeyError)):
                verify_snapshot(record, backend.snapshot(record), True)

    def test_failure_after_provision_keeps_primary_and_cleanup(self):
        h, c, inventory, evidence = self.make(); h.blocks = [b'x' * 65536] * 17
        result = c.execute(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1', evidence)
        self.assertEqual(result['state'], 'FAILED')
        self.assertIn('overflow', result['primary_error'])
        self.assertTrue(result['daemon_io_termination_verified'])

    def test_final_export_and_teardown(self):
        h, c, inventory, evidence = self.make()
        backend, record = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
        from supervisor_v2 import Supervisor
        self.assertEqual(Supervisor(backend, evidence).run(record)['state'], 'VERIFIED_FAKE')
        exporter = Mock()
        def export(raw, manifest, **kwargs):
            return {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'readback_verified': True}
        exporter.export.side_effect = export
        final_sink = FakeEvidence()
        result = c.finalize(backend, exporter, {'files': {}}, final_sink)
        self.assertTrue(result['complete'], result)
        self.assertFalse(h.loops)
        self.assertFalse(h.mounts)
        self.assertFalse(h.processes)
        self.assertTrue(final_sink.ledger)

    def test_bad_export_ack_preserves_evidence(self):
        h, c, inventory, evidence = self.make()
        backend, record = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
        from supervisor_v2 import Supervisor
        Supervisor(backend, evidence).run(record)
        exporter = Mock(); exporter.export.return_value = {'manifest_sha256': 'bad', 'readback_verified': True}
        result = c.finalize(backend, exporter, {'files': {}}, FakeEvidence())
        self.assertFalse(result['complete'])
        self.assertTrue(any(m['mount'].endswith('/evidence') for m in h.mounts))

    def test_worker_algorithm_arms_and_rechecks_owner(self):
        h, c, inventory, evidence = self.make()
        backend, record = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
        wd = record['watchdog']; owner = {'pid': 9876, 'start_ticks': 1, 'executable_sha256': 'b' * 64, 'cgroup': '/owner'}
        path = c.l.root + '/evidence/watchdog-armed.json'; del h.files[path]
        from watchdog_worker import worker
        from unittest.mock import patch
        with patch('watchdog_worker.os.getpid', return_value=wd['pid']):
            result = worker(c.l, c.l.cgroup_identity(), owner, FakeEvidence(), lambda: False)
        self.assertTrue(result['termination_confirmed'])
        self.assertEqual(json.loads(h.files[path])['watchdog'], wd)

    def test_unbounded_controller_refused_before_creation(self):
        h = Kernel(); h.files['/sys/fs/cgroup' + h.controller_path + '/memory.max'] = 'max'
        with self.assertRaises(ValueError): Coordinator(h, FakeEvidence(), 'a' * 32)
        self.assertFalse(h.calls)

    def test_final_persistence_failure_never_saved_success(self):
        h, c, inventory, evidence = self.make()
        backend, record = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
        from supervisor_v2 import Supervisor
        Supervisor(backend, evidence).run(record)
        exporter = Mock()
        exporter.export.side_effect = lambda raw, manifest, **kw: {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'readback_verified': True}
        sink = FakeEvidence('seal')
        result = c.finalize(backend, exporter, {'files': {}}, sink)
        self.assertFalse(result['complete'])
        self.assertEqual(result['state'], 'FAILED')
        self.assertEqual(sink.stdout['state'], 'FAILED')
        self.assertTrue(all(row.get('state') not in ('VERIFIED_FAKE', 'CLEANUP_VERIFIED_FAKE') for _, row in sink.rows))
