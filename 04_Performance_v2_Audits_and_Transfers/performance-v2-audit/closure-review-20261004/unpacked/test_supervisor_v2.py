"""Invented ownership observations and FAKE clock/backend only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import supervisor_v2 as s
from supervisor_evidence import FileEvidence

def record():
    token = 'a' * 32
    base, runtime, cg = '/var/tmp/pv2-' + token, '/run/pv2-' + token, '/pv2-' + token + '.slice'
    r = {'schema': 'pv2_ownership_v2', 'token': token, 'run_id': '123', 'attempt': '1',
         'root': base, 'runtime_root': runtime, 'cgroup': cg, 'cgroup_inode': 1234, 'boot_id': 'invented-boot',
         'policy': deepcopy(s.POLICY), 'service': 'pv2-' + token + '.service', 'runtime_tmpfs_bytes': 64 * 1024**2,
         'network': {'bridge': False, 'iptables': False, 'ip_forward': False}, 'container_cgroup_parent': cg,
         'supervisor_cgroup': '/supervisor', 'watchdog': {'outside_workload': True, 'lifetime_seconds': 1200,
             'pid': 999, 'start_ticks': 888, 'executable_sha256': 'c' * 64, 'cgroup': '/watchdog'},
         'paths': {'docker_socket': runtime + '/docker.sock', 'containerd_socket': runtime + '/containerd.sock',
                   'pidfile': runtime + '/dockerd.pid', 'exec_root': runtime + '/exec', 'data_root': base + '/data/docker',
                   'containerd_root': base + '/data/containerd', 'containerd_state': runtime + '/containerd',
                   'download_root': base + '/data/downloads', 'scratch_root': base + '/data/tmp', 'daemon_log': base + '/evidence/daemon.log'},
         'volumes': {}, 'processes': []}
    for number, (role, capacity) in enumerate((('data', 8 * s.GIB), ('evidence', 512 * 1024**2))):
        r['volumes'][role] = {'mount': base + '/' + role, 'backing': base + '/' + role + '.ext4',
                              'loop': '/dev/loop' + str(number + 10), 'capacity_bytes': capacity, 'allocated_bytes': capacity,
                              'filesystem': 'ext4', 'uuid': 'invented-' + role, 'device': '7:' + str(number + 10),
                              'mount_id': number + 1, 'inode_capacity': 65536, 'auto_expand': False}
    for number, role in enumerate(('dockerd', 'containerd')):
        r['processes'].append({'role': role, 'pid': 100 + number, 'start_ticks': 1000 + number,
                               'executable_sha256': 'b' * 64, 'cgroup': cg + '/daemons'})
    return r

class FakeEvidence:
    def __init__(self, fail=None):
        self.fail = fail; self.rows = []; self.ledger = None; self.stdout = None
    def append(self, name, value):
        if name == self.fail:
            raise OSError('invented evidence disk full')
        self.rows.append((name, deepcopy(value)))
    def seal(self):
        if self.fail == 'seal':
            raise OSError('invented ledger failure')
        self.ledger = deepcopy(self.rows)
    def fallback(self, value):
        self.stdout = deepcopy(value)
    def complete(self, outcome):
        self.completion = {'outcome': outcome, 'ledger': deepcopy(self.ledger)}

class FakeBackend:
    is_fake = True
    def __init__(self, r=None):
        self.r = deepcopy(r or record()); self.time = 0; self.calls = []; self.fail = set()
        self.alive = True; self.mounted = True; self.attached = True; self.busy = False; self.unKillable = False
        self.edits = {}; self.blocks = [b'invented output', b'']; self.streaming_timeout = False
    def now(self):
        return self.time
    def snapshot(self):
        x = {'identity': {'boot_id': self.r['boot_id'], 'cgroup_inode': self.r['cgroup_inode'], 'token': self.r['token']},
             'paths': deepcopy(self.r['paths']), 'volumes': deepcopy(self.r['volumes']), 'policy': deepcopy(self.r['policy']),
             'all_storage_confined': True, 'external_owned_processes': [], 'processes': deepcopy(self.r['processes']) if self.alive else [],
             'watchdog_armed': True, 'watchdog_identity': deepcopy(self.r['watchdog']), 'free_bytes': 1024, 'free_inodes': 10, 'populated': self.alive,
             'listeners': [self.r['paths']['docker_socket']] if self.alive else []}
        x.update(deepcopy(self.edits)); return x
    def request(self, action, r, timeout):
        self.calls.append((action, timeout)); self.time += .01
        if action in self.fail:
            raise TimeoutError('invented ' + action + ' timeout')
        if action in ('snapshot', 'wait_empty'):
            return self.snapshot()
        if action == 'kill_subtree' and not self.unKillable:
            self.alive = False
        if action == 'unmount_data' and not self.busy:
            self.mounted = False
        if action == 'detach_data_loop':
            self.attached = False
        if action == 'read_mount':
            return {'record_identity': deepcopy(r['volumes']['data']), 'mounted': self.mounted}
        if action == 'read_loop':
            return {'record_identity': deepcopy(r['volumes']['data']), 'attached': self.attached}
        if action == 'begin_operation':
            return {'handle': 'invented-stream', 'deadline_seconds': 60}
        return {'exit_code': 0}
    def read(self, handle, maximum, timeout):
        self.time += timeout if self.streaming_timeout else .01
        if self.streaming_timeout:
            return None
        return self.blocks.pop(0)

class SupervisorTests(unittest.TestCase):
    def run_case(self, backend=None, evidence=None, r=None):
        b, e = backend or FakeBackend(), evidence or FakeEvidence()
        report = s.Supervisor(b, e).run(r or b.r)
        return report, b, e

    def test_success_seals_cleanup_and_false_approval(self):
        result, b, e = self.run_case()
        self.assertEqual(result['state'], 'VERIFIED_FAKE')
        self.assertTrue(result['daemon_io_termination_verified'])
        self.assertTrue(result['work_storage_detached'])
        self.assertFalse(result['execution_approved'])
        self.assertEqual(e.ledger[-1][0], 'supervisor-report')
        self.assertTrue(all(t <= 10 for _, t in b.calls))

    def test_no_live_backend(self):
        b = FakeBackend(); b.is_fake = False
        with self.assertRaisesRegex(s.Refusal, 'live backend'):
            s.Supervisor(b, FakeEvidence())

    def test_watchdog_not_armed_or_wrong_identity(self):
        for key, value in (('watchdog_armed', False), ('watchdog_identity', {})):
            b = FakeBackend(); b.edits[key] = value
            self.assertIn('watchdog', self.run_case(b)[0]['primary_error'])

    def test_shared_socket_and_containerd_refused(self):
        for key in ('docker_socket', 'containerd_socket', 'containerd_root'):
            r = record(); r['paths'][key] = '/var/run/docker.sock'
            report, b, _ = self.run_case(r=r)
            self.assertEqual(report['state'], 'FAILED'); self.assertFalse(b.calls)

    def test_cancellation_before_record_no_host_mutation(self):
        report, b, e = self.run_case(evidence=FakeEvidence('ownership'))
        self.assertFalse(b.calls)
        self.assertIn('disk full', report['primary_error'])
        self.assertFalse(report['daemon_io_termination_verified'])

    def test_pid_reuse(self):
        b = FakeBackend(); ps = deepcopy(b.r['processes']); ps[0]['start_ticks'] += 1; b.edits['processes'] = ps
        report, b, _ = self.run_case(b)
        self.assertIn('PID reuse', report['primary_error'])
        self.assertNotIn('kill_subtree', [a for a, _ in b.calls])

    def test_subtree_escape_census(self):
        b = FakeBackend(); b.edits['external_owned_processes'] = [{'pid': 999, 'cgroup': '/system.slice'}]
        report, _, _ = self.run_case(b)
        self.assertIn('subtree escape', report['primary_error'])

    def test_record_subtree_escape(self):
        r = record(); r['processes'][0]['cgroup'] = '/unowned'
        report, b, _ = self.run_case(r=r)
        self.assertEqual(report['state'], 'FAILED'); self.assertFalse(b.calls)

    def test_cli_timeout_live_daemon_killed(self):
        b = FakeBackend(); b.streaming_timeout = True
        report, b, _ = self.run_case(b)
        self.assertIn('timeout', report['primary_error']); self.assertFalse(b.alive)
        self.assertTrue(report['daemon_io_termination_verified'])

    def test_failed_term_does_not_skip_kill_readback(self):
        b = FakeBackend(); b.fail.add('term_service')
        report, b, _ = self.run_case(b)
        self.assertTrue(report['daemon_io_termination_verified']); self.assertEqual(report['state'], 'FAILED')
        self.assertIn('kill_subtree', [a for a, _ in b.calls])

    def test_failed_kill_still_reads_state(self):
        b = FakeBackend(); b.fail.add('kill_subtree')
        report, b, _ = self.run_case(b)
        self.assertIn('wait_empty', [a for a, _ in b.calls]); self.assertTrue(b.mounted)
        self.assertFalse(report['daemon_io_termination_verified'])

    def test_nonempty_after_successful_kill_not_pass(self):
        b = FakeBackend(); b.unKillable = True
        report, _, _ = self.run_case(b)
        self.assertFalse(report['daemon_io_termination_verified']); self.assertFalse(report['work_storage_detached'])

    def test_independent_cleanup_after_initial_failure(self):
        b = FakeBackend(); b.fail.add('kill_subtree'); self.run_case(b)
        b.fail.clear()
        report = s.Supervisor(b, FakeEvidence()).independent_cleanup(b.r)
        self.assertTrue(report['work_storage_detached'])

    def test_stale_cleanup_claim_ignored(self):
        b = FakeBackend(); b.unKillable = True
        report = s.Supervisor(b, FakeEvidence()).independent_cleanup(b.r)
        self.assertFalse(report['daemon_io_termination_verified'])

    def test_volume_mount_and_loop_mismatch(self):
        for key in ('loop', 'mount_id', 'uuid'):
            b = FakeBackend(); v = deepcopy(b.r['volumes']); v['data'][key] = 'different'; b.edits['volumes'] = v
            report, b, _ = self.run_case(b)
            self.assertIn('mismatch', report['primary_error']); self.assertTrue(b.attached)

    def test_sparse_and_spill_rejected(self):
        r = record(); r['volumes']['data']['allocated_bytes'] = 10
        self.assertIn('sparse', self.run_case(r=r)[0]['primary_error'])
        b = FakeBackend(); b.edits['all_storage_confined'] = False
        self.assertIn('spill', self.run_case(b)[0]['primary_error'])

    def test_bytes_and_inode_exhaustion(self):
        for key in ('free_bytes', 'free_inodes'):
            b = FakeBackend(); b.edits[key] = 0
            report, _, _ = self.run_case(b)
            self.assertIn('ENOSPC', report['primary_error'])

    def test_busy_mount_not_detached(self):
        b = FakeBackend(); b.busy = True
        report, b, _ = self.run_case(b)
        self.assertTrue(report['daemon_io_termination_verified']); self.assertTrue(b.attached)
        self.assertFalse(report['work_storage_detached'])

    def test_stream_aggregate_overflow(self):
        b = FakeBackend(); b.blocks = [b'x' * s.CHUNK] * 17
        report, b, _ = self.run_case(b)
        self.assertIn('overflow', report['primary_error']); self.assertFalse(b.alive)

    def test_stream_oversize_chunk_rejected(self):
        b = FakeBackend(); b.blocks = [b'x' * (s.CHUNK + 1)]
        self.assertIn('bounded read', self.run_case(b)[0]['primary_error'])

    def test_primary_error_preserved_when_disk_full(self):
        b = FakeBackend(); b.streaming_timeout = True
        report, _, e = self.run_case(b, FakeEvidence('supervisor-report'))
        self.assertIn('timeout', report['primary_error']); self.assertTrue(report['persistence_errors'])
        self.assertIn('timeout', e.stdout['primary_error'])

    def test_ledger_failure_is_failure(self):
        report, _, _ = self.run_case(evidence=FakeEvidence('seal'))
        self.assertEqual(report['state'], 'FAILED')

    def test_late_cancellation_keeps_diagnostics(self):
        b = FakeBackend()
        with patch.object(b, 'read', side_effect=InterruptedError('invented cancellation')):
            report, _, _ = self.run_case(b)
        self.assertIn('cancellation', report['primary_error']); self.assertTrue(report['daemon_io_termination_verified'])

    def test_backend_deadline_violation(self):
        b = FakeBackend(); original = b.request
        def delayed(action, r, timeout):
            value = original(action, r, timeout)
            if action == 'begin_operation':
                b.time += 11
            return value
        b.request = delayed
        self.assertIn('deadline', self.run_case(b)[0]['primary_error'])

    def test_file_evidence_no_overwrite_and_final_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'evidence'; sink = FileEvidence(root, directory_sync=lambda p: None)
            with patch('builtins.print'):
                self.run_case(evidence=sink)
            ledger = json.loads((root / 'report-hashes.json').read_text())
            for name, entry in ledger.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), entry['sha256'])
            with self.assertRaises(FileExistsError):
                FileEvidence(root)

if __name__ == '__main__':
    unittest.main()
