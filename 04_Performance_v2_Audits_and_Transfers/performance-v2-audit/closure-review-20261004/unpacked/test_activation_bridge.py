"""Mocked boundaries only: invented Linux state and in-memory trusted filesystem.

Separate reconstruction is a new object from serialized JSON, not a real Linux
interpreter. No native launch, mount, daemon, package or network call is made.
"""
from copy import deepcopy
import json
import stat
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import activation_bridge as bridge
from activation_trust import (AUTH, STATE, Reservations, TrustedIO, validate_trust)
from launcher_binding import ROOT, SOURCES, canonical, command, sha
from test_launcher_binding import fixture
from test_linux_integration import Kernel
from test_supervisor_v2 import FakeEvidence
from linux_integration import Coordinator


class IO:
    """Invented ownership/hash/OS observations; not a real trust check."""
    def __init__(self): self.files = {}; self.dirs = {STATE}; self.digests = {}; self.denied = set(); self.calls = []
    def check_path(self, path, directory=False):
        self.calls.append(('check', path))
        if path in self.denied: raise ValueError('untrusted ownership/mode/link')
    def read(self, path):
        self.check_path(path)
        if path not in self.files: raise FileNotFoundError(path)
        return self.files[path]
    def json(self, path): return json.loads(self.read(path))
    def digest(self, path):
        self.check_path(path)
        return self.digests[path] if path in self.digests else sha(self.read(path))
    def verify_environment(self, inventory): self.check_path('invented-environment')
    def verify_storage(self, args, auth): return {'fixture': 'storage OS boundary mocked; policy tested separately'}
    def mkdir(self, path):
        if path in self.dirs: raise FileExistsError(path)
        self.dirs.add(path)
    def exclusive(self, path, value):
        if path in self.files: raise FileExistsError(path)
        self.files[path] = canonical(value)


def setup(role='bootstrap', scenario='success'):
    manifest, inventory, auth = fixture()
    auth['scenario'] = scenario
    inventory['runtime_files'] = {'/usr/lib/invented-stdlib.py': 'd'*64}
    inventory['runtime_closure_reviewed'] = True
    auth['inventory_sha256'] = sha(canonical(inventory))
    args = SimpleNamespace(role=role, token='a'*32, scenario=scenario,
                           source_manifest=PurePosixPath(ROOT+'/launcher-sources.json'),
                           inventory=PurePosixPath(AUTH+'/inventory.json'), authorization=PurePosixPath(AUTH+'/authorization.json'))
    io = IO()
    for name in SOURCES: io.files[ROOT+'/'+name] = (Path(__file__).parent/name).read_bytes()
    io.files[str(args.source_manifest)] = canonical(manifest)
    io.files[str(args.inventory)] = canonical(inventory)
    io.files[str(args.authorization)] = canonical(auth)
    io.digests.update({r['resolved_path']: r['sha256'] for r in inventory['executables'].values()})
    io.digests.update(inventory['runtime_files'])
    io.digests[AUTH+'/security_review.json'] = inventory['security_review_sha256']
    io.digests[AUTH+'/host_evidence.json'] = inventory['host_evidence_sha256']
    return args, io


class TrustTests(unittest.TestCase):
    def test_native_path_ownership_mode_and_symlink_checks(self):
        good = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o600)
        parent = SimpleNamespace(st_uid=0, st_mode=stat.S_IFDIR | 0o755)
        # Mock lstat, including every ancestor. No real protected path opened.
        for bad in (SimpleNamespace(st_uid=1000, st_mode=stat.S_IFREG | 0o600),
                    SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o622),
                    SimpleNamespace(st_uid=0, st_mode=stat.S_IFLNK | 0o777)):
            with patch('activation_trust.Path', PurePosixPath), patch('activation_trust.os.lstat', return_value=bad):
                with self.assertRaisesRegex(ValueError, 'untrusted ownership'):
                    TrustedIO().check_path('/invented/file')

    def test_grant_does_not_retain_mutable_authorization(self):
        args, io = setup(); grant = validate_trust(args, io, 150)
        args.role = 'helper'; args.scenario = 'timeout'
        self.assertEqual(grant.args.role, 'bootstrap'); self.assertEqual(grant.args.scenario, 'success')

    def test_verified_mock_grant_is_required_by_native_host(self):
        from linux_host import LinuxHost
        for invalid in (None, SimpleNamespace(is_fake=True)):
            with self.assertRaises(RuntimeError): LinuxHost(invalid)
        args, io = setup()
        grant = validate_trust(args, io, 150)
        # Constructor does no host effects; no method on this native object runs.
        host = LinuxHost(grant)
        self.assertFalse(host.is_fake)

    def test_scenario_substitution(self):
        for other in ('output_overflow', 'timeout'):
            args, io = setup(); args.scenario = other
            with self.assertRaisesRegex(ValueError, 'scenario substitution'): validate_trust(args, io, 150)

    def test_bad_source_inventory_and_closure(self):
        for target in (ROOT+'/activation_bridge.py', '/usr/bin/runc', '/usr/lib/invented-stdlib.py', AUTH+'/security_review.json'):
            args, io = setup()
            if target in io.digests: io.digests[target] = '0'*64
            else: io.files[target] += b'altered'
            with self.subTest(target=target), self.assertRaises(ValueError): validate_trust(args, io, 150)

    def test_owner_modes_links_and_environment_fail_closed(self):
        for target in (ROOT, ROOT+'/native_launcher.py', AUTH+'/authorization.json', 'invented-environment'):
            args, io = setup(); io.denied.add(target)
            with self.subTest(target=target), self.assertRaises(ValueError): validate_trust(args, io, 150)

    def test_single_use_chain_and_controller_replay(self):
        args, io = setup()
        for role in ('bootstrap', 'controller', 'watchdog', 'helper'):
            args.role = role
            grant = validate_trust(args, io, 150)
            reservation = Reservations(io, grant)
            reservation.claim(role, {'pid': 100})
            with self.assertRaises(FileExistsError): reservation.claim(role, {'pid': 101})
        args.role = 'bootstrap'
        with self.assertRaises(FileExistsError): Reservations(io, validate_trust(args, io, 150)).claim('bootstrap', {})

    def test_helper_before_watchdog_rejected(self):
        args, io = setup()
        for role in ('bootstrap', 'controller'):
            args.role = role; Reservations(io, validate_trust(args, io, 150)).claim(role, {})
        args.role = 'helper'
        with self.assertRaises(FileNotFoundError): Reservations(io, validate_trust(args, io, 150)).claim('helper', {})

    def test_reservation_changed_auth_fails(self):
        args, io = setup(); Reservations(io, validate_trust(args, io, 150)).claim('bootstrap', {})
        a = io.json(str(args.authorization)); a['expires'] = 199; io.files[str(args.authorization)] = canonical(a)
        args.role = 'controller'
        with self.assertRaisesRegex(ValueError, 'reservation binding'):
            Reservations(io, validate_trust(args, io, 150)).claim('controller', {})

    def test_durable_reservation_failure_is_consumed(self):
        args, io = setup(); r = Reservations(io, validate_trust(args, io, 150))
        with patch.object(io, 'exclusive', side_effect=OSError('directory fsync failure')):
            with self.assertRaises(OSError): r.claim('bootstrap', {})
        with self.assertRaises(FileExistsError): r.claim('bootstrap', {})


class RoleRT:
    """Handlers connect to real coordinator, with fake kernel and fake evidence."""
    def __init__(self):
        self.host = Kernel(); self.token = 'a'*32; self.saved = {}; self.sinks = {}
        self.g = SimpleNamespace(args=SimpleNamespace(scenario='success'), inventory={'executables': {
            p: {'sha256': 'a'*64} for p in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc')}})
        self.saved['bootstrap.claim.json'] = {'process': {'pid': 900, 'start_ticks': 1, 'executable_sha256': 'a'*64, 'cgroup': '/system.slice/bootstrap'}}
        self.owner = {'pid': 901, 'start_ticks': 2, 'executable_sha256': 'a'*64, 'cgroup': self.host.controller_path}
        self.parked = False; self.output_blocks = []
    def evidence(self, name):
        self.sinks[name] = FakeEvidence(); return self.sinks[name]
    def read(self, name):
        if name not in self.saved: raise FileNotFoundError(name)
        return deepcopy(self.saved[name])
    def write(self, name, value):
        if name in self.saved: raise FileExistsError(name)
        self.saved[name] = deepcopy(value)
    def me(self): return deepcopy(self.owner)
    def argv(self, role): return command(role, self.token, self.g.args.scenario)
    def controller_done(self): return 'controller-result.json' in self.saved
    def wait_watchdog(self):
        # Invented separate-interpreter result at this OS boundary; worker
        # reconstruction/arming/owner-loss algorithm is independently tested below.
        return {'token': self.token, 'termination_confirmed': True, 'errors': [],
                'ownership': self.saved['full-state.json']['identity']}
    def park_watchdog(self, seconds): self.parked = seconds
    def synthetic_wait(self, seconds): self.parked = seconds
    def output(self, block): self.output_blocks.append(block)


class HandlerTests(unittest.TestCase):
    def test_actual_role_placement_and_resource_rejection(self):
        args, io = setup(); grant = validate_trust(args, io, 150)
        host = Kernel()
        group = '/system.slice/pv2-'+args.token+'-bootstrap.service'
        unit = 'pv2-'+args.token+'-bootstrap.service'
        host.files['/proc/self/cgroup'] = '0::'+group+'\n'
        for k, v in {'memory.max': '268435456', 'memory.swap.max': '0', 'pids.max': '16', 'cpu.max': '25000 100000'}.items():
            host.files['/sys/fs/cgroup'+group+'/'+k] = v
        rt = bridge.Runtime(grant, io, host)
        answer = f'Id={unit}\nActiveState=active\nMainPID=42\nControlGroup={group}\n'.encode()
        with patch('activation_bridge.os.getpid', return_value=42), patch.object(host, 'command', return_value=answer):
            rt.verify_role()
            host.files['/sys/fs/cgroup'+group+'/memory.max'] = 'max'
            with self.assertRaises(ValueError): rt.verify_role()

    def test_offline_receipt_validates_all_bytes_and_rejects_incomplete(self):
        from supervisor_evidence import FileEvidence
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for directory, outcome in (('operation', 'VERIFIED_NATIVE'), ('final', 'CLEANUP_VERIFIED_NATIVE')):
                sink = FileEvidence(root/directory, directory_sync=lambda p: None)  # explicitly fake directory fsync
                sink.append('supervisor-report', {'state': 'EVIDENCE_PENDING', 'operation_cleanup_outcome': outcome,
                                                  'persistence_errors': [], 'fixture': 'invented native result; not executed'})
                sink.seal(); sink.complete(outcome)
            (root/'activation-result.json').write_bytes(canonical({'complete': False, 'state': 'EVIDENCE_PENDING', 'operation_teardown_complete': True}))
            entries = {p.relative_to(root).as_posix(): {'sha256': sha(p.read_bytes()), 'size_bytes': p.stat().st_size}
                       for p in root.rglob('*') if p.is_file()}
            raw = canonical({'files': entries}); (root/'activation-manifest.json').write_bytes(raw)
            receipt = {'manifest_sha256': sha(raw), 'complete': True, 'scope': 'A_native_synthetic_only'}
            (root/'activation-receipt.json').write_bytes(canonical(receipt))
            self.assertTrue(bridge.validate_activation(root)['complete'])
            (root/'activation-result.json').write_bytes(b'altered')
            with self.assertRaises(ValueError): bridge.validate_activation(root)
            receipt['complete'] = False; (root/'activation-receipt.json').write_bytes(canonical(receipt))
            with self.assertRaisesRegex(ValueError, 'incomplete'): bridge.validate_activation(root)

    def test_finalization_persistence_never_saves_premature_success(self):
        for fail in ('activation-manifest.json', 'activation-receipt.json', 'stdout'):
            rt = Mock(root='/invented')
            saved, stdout = {}, []
            def write(name, value):
                if name == fail: raise OSError('invented persistence')
                saved[name] = deepcopy(value)
            rt.write.side_effect = write
            rt.collect_manifest.return_value = {}
            rt.host.digest.return_value = 'a'*64
            def emit(value):
                stdout.append(deepcopy(value))
                if fail == 'stdout': raise OSError('invented pipe')
            rt.emit.side_effect = emit
            result = bridge.Runtime.finish(rt, {'complete': True})
            self.assertFalse(result['complete'])
            self.assertFalse(saved['activation-result.json']['complete'])
            self.assertFalse(stdout[-1]['complete'])
            self.assertIn('activation-persistence-failure.json', saved)
            self.assertNotIn('activation-receipt.json', saved)

    def test_finalization_receipt_last(self):
        rt = Mock(root='/invented'); sequence = []
        rt.write.side_effect = lambda name, value: sequence.append(name)
        rt.emit.side_effect = lambda value: sequence.append('stdout-provisional')
        rt.collect_manifest.return_value = {}; rt.host.digest.return_value = 'a'*64
        self.assertTrue(bridge.Runtime.finish(rt, {'complete': True})['complete'])
        self.assertEqual(sequence[-2:], ['stdout-provisional', 'activation-receipt.json'])

    def test_collector_connects_export_and_final_teardown(self):
        rt = RoleRT(); bridge.controller(rt)
        rt.root = STATE+'/'+rt.token
        # The coordinator result is fake, so cleanup can finish but native
        # acceptance must remain false. File listing/export are OS-boundary fakes.
        capsule = rt.saved['full-state.json']
        exporter = Mock()
        exporter.export.side_effect = lambda raw, manifest, **kw: {'manifest_sha256': sha(raw), 'readback_verified': True}
        member = Mock(); member.name = 'watchdog-armed.json'
        member.is_file.return_value = True; member.is_symlink.return_value = False
        member.stat.return_value = SimpleNamespace(st_size=10)
        with patch('activation_bridge.recover_records', return_value={'unresolved_intents': [], 'observed': capsule['resources']}), \
             patch('activation_bridge.Path') as paths, patch('evidence_export.LocalEvidenceExport', return_value=exporter):
            paths.return_value.iterdir.return_value = [member]
            result = bridge.Runtime.collect_and_finalize(rt)
        self.assertTrue(result['teardown']['complete'], result)
        self.assertFalse(result['complete'])
        self.assertFalse(rt.host.loops); self.assertFalse(rt.host.mounts)
        exporter.export.assert_called_once()

    def test_collector_loss_not_reported_as_operation_success(self):
        rt = RoleRT(); bridge.controller(rt); rt.root = STATE+'/'+rt.token
        rt.saved.pop('controller-result.json')
        capsule = rt.saved['full-state.json']
        exporter = Mock()
        exporter.export.side_effect = lambda raw, manifest, **kw: {'manifest_sha256': sha(raw), 'readback_verified': True}
        with patch('activation_bridge.recover_records', return_value={'unresolved_intents': [], 'observed': capsule['resources']}), \
             patch('activation_bridge.Path') as paths, patch('evidence_export.LocalEvidenceExport', return_value=exporter):
            paths.return_value.iterdir.return_value = []
            result = bridge.Runtime.collect_and_finalize(rt)
        self.assertFalse(result['complete']); self.assertEqual(result['operation']['primary_error'], 'controller loss')
        self.assertTrue(result['teardown']['complete'])

    def test_collector_partial_ownership_retains_resources(self):
        rt = RoleRT(); bridge.controller(rt); rt.root = STATE+'/'+rt.token
        rt.saved.pop('full-state.json')
        result = bridge.Runtime.collect_and_finalize(rt)
        self.assertFalse(result['complete']); self.assertTrue(result['resources_retained'])
        self.assertTrue(rt.host.loops)

    def test_dispatch_calls_real_role_only_after_admission(self):
        rt = Mock(); rt.g.args.role = 'helper'; rt.g.args.scenario = 'success'
        rt.me.return_value = {'pid': 12}
        rt.output = Mock()
        result = bridge.dispatch(rt)
        self.assertTrue(result['complete']); rt.verify_role.assert_called_once()
        rt.reservations.claim.assert_called_once_with('helper', {'pid': 12})
        rt.verify_role.side_effect = ValueError('placement denied')
        rt.reservations.reset_mock(); rt.output.reset_mock()
        with self.assertRaises(ValueError): bridge.dispatch(rt)
        rt.reservations.claim.assert_not_called(); rt.output.assert_not_called()

    def test_controller_connected_and_capsules_saved(self):
        rt = RoleRT(); result = bridge.controller(rt)
        self.assertEqual(result['state'], 'VERIFIED_FAKE')
        self.assertTrue(rt.host.worker_seen_before_daemon)
        self.assertEqual(rt.saved['full-state.json']['record']['token'], rt.token)
        self.assertTrue(rt.saved['watchdog-bootstrap.json']['resources'])
        self.assertFalse(any('configuration' == x['kind'] for x in rt.saved['watchdog-bootstrap.json']['resources']))

    def test_new_watchdog_reconstructs_and_arms(self):
        rt = RoleRT(); bridge.controller(rt)
        capsule = json.loads(json.dumps(rt.saved['watchdog-bootstrap.json']))
        restored = bridge.restore(rt.host, FakeEvidence(), capsule, rt.token)
        self.assertIsNot(restored.resources, capsule['resources'])
        self.assertEqual(restored.parent, rt.host.controller_path)
        wd = rt.saved['full-state.json']['record']['watchdog']
        del rt.host.files[restored.root+'/evidence/watchdog-armed.json']
        from watchdog_worker import arm
        with patch('watchdog_worker.os.getpid', return_value=wd['pid']): arm(restored)
        self.assertEqual(json.loads(rt.host.files[restored.root+'/evidence/watchdog-armed.json'])['watchdog'], wd)

    def test_watchdog_handler_controller_loss_collects_and_parks(self):
        rt = RoleRT(); bridge.controller(rt)
        rt.saved.pop('controller-result.json')
        wd = rt.saved['full-state.json']['record']['watchdog']
        del rt.host.files['/var/tmp/pv2-'+rt.token+'/evidence/watchdog-armed.json']
        with patch('watchdog_worker.os.getpid', return_value=wd['pid']):
            result = bridge.watchdog(rt)
        self.assertTrue(result['termination_confirmed'])
        self.assertIn('watchdog-result.json', rt.saved)
        self.assertEqual(rt.parked, 1200)
        self.assertIn(wd['pid'], rt.host.processes)

    def test_capsule_rejects_boot_and_identity_change(self):
        rt = RoleRT(); bridge.controller(rt)
        for field, value in (('boot_id', 'other'), ('token', 'b'*32), ('identity', {})):
            capsule = deepcopy(rt.saved['watchdog-bootstrap.json']); capsule[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): bridge.restore(rt.host, FakeEvidence(), capsule, rt.token)

    def test_helper_dispatch_no_model_or_shell(self):
        rt = RoleRT(); rt.saved['full-state.json'] = {}
        bridge.helper(rt); self.assertEqual(rt.output_blocks, [b'pv2-native-synthetic-v1\n'])
        rt.g.args.scenario = 'output_overflow'; rt.output_blocks = []; bridge.helper(rt)
        self.assertEqual(sum(map(len, rt.output_blocks)), 17*65536)
        rt.g.args.scenario = 'timeout'; bridge.helper(rt); self.assertEqual(rt.parked, 65)

    def test_bootstrap_failure_collects_preserves_primary(self):
        rt = Mock(token='a'*32)
        rt.argv.return_value = command('controller', rt.token)
        rt.host.command.side_effect = OSError('invented start failure')
        rt.collect_and_finalize.return_value = {'complete': True}
        rt.finish.side_effect = lambda x: x
        result = bridge.bootstrap(rt)
        self.assertFalse(result['complete'])
        self.assertIn('start failure', result['bootstrap_error'])
        rt.stop_controller.assert_called_once(); rt.collect_and_finalize.assert_called_once()

    def test_bootstrap_cleanup_error_preserves_primary(self):
        rt = Mock(token='a'*32); rt.argv.return_value = command('controller', rt.token)
        rt.host.command.side_effect = OSError('primary')
        rt.stop_controller.side_effect = OSError('cleanup')
        rt.collect_and_finalize.return_value = {'complete': False}
        rt.finish.side_effect = lambda x: x
        result = bridge.bootstrap(rt)
        self.assertIn('primary', result['bootstrap_error']); self.assertIn('cleanup', result['bootstrap_cleanup_errors'][0])
        rt.collect_and_finalize.assert_called_once()
        rt.finish.assert_called_once()

    def test_bootstrap_timeout_requests_independent_stop(self):
        rt = Mock(token='a'*32); rt.argv.return_value = command('controller', rt.token)
        rt.wait_controller.side_effect = TimeoutError('invented timeout')
        rt.collect_and_finalize.return_value = {'complete': False, 'resources_retained': True}
        rt.finish.side_effect = lambda x: x
        self.assertFalse(bridge.bootstrap(rt)['complete']); rt.stop_controller.assert_called_once()


if __name__ == '__main__': unittest.main()
