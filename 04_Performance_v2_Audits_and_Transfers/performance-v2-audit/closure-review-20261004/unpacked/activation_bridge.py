"""Connected native role handlers. All effects use explicit injectable boundaries.

There is no fake CLI switch. The CLI creates Runtime only from a verified Grant.
Mock tests replace Runtime/host/TrustedIO, not authorization with an is_fake flag.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from activation_trust import Reservations, STATE, valid_grant
from launcher_binding import command, need, canonical
from linux_integration import Coordinator, LinuxBackend, Services, recover_records
from linux_lifecycle import LinuxLifecycle
from supervisor_v2 import Supervisor, within


def restore(host, journal, capsule, token):
    """Reconstruct from serialized independent state, never inherited Python objects."""
    need(capsule['token'] == token, 'capsule token')
    life = LinuxLifecycle(host, journal, token)
    need(life.boot == capsule['boot_id'], 'capsule boot')
    life.parent = capsule['controller']['cgroup']
    life.resources = deepcopy(capsule['resources'])
    life.control_processes = deepcopy(capsule.get('control_processes', []))
    life.owned_slices = {x['requested']: x['identity'] for x in life.resources if x['kind'] == 'systemd-slice'}
    need(set(life.owned_slices) == {life.manager.work_unit, life.manager.guard_unit}, 'capsule slice coverage')
    identity = life.cgroup_identity()
    need(identity == capsule['identity'], 'capsule fresh identity')
    life.guard_path = life.manager.show(life.manager.guard_unit)['ControlGroup']
    need(not within(life.guard_path, life.work_path) and not within(life.work_path, life.guard_path), 'capsule branches')
    need(life.owned_slices[life.manager.guard_unit]['inode'] == host.stat(life.cgmount+life.guard_path).st_ino,
         'guard replaced')
    return life


def controller_unit(token, argv):
    return ['systemd-run', '--unit=pv2-'+token+'-controller.service', '--service-type=exec',
            '--property=Restart=no', '--property=KillMode=control-group',
            '--property=RuntimeMaxSec=1080', '--property=TimeoutStopSec=10',
            '--property=MemoryMax=268435456', '--property=MemorySwapMax=0',
            '--property=CPUQuota=25%', '--property=TasksMax=16',
            '--property=StandardOutput=null', '--property=StandardError=null', '--', *argv]


def dispatch(runtime):
    """Role capability, service placement and single-use reservation precede effects."""
    role = runtime.g.args.role
    runtime.verify_role()
    runtime.reservations.claim(role, runtime.me())
    if role == 'bootstrap':
        runtime.write('storage-initial.json', runtime.g.storage)
    else:
        runtime.write(role+'-storage-admission.json', runtime.g.storage)
    handlers = {'bootstrap': bootstrap, 'controller': controller, 'watchdog': watchdog, 'helper': helper}
    return handlers[role](runtime)


def bootstrap(rt):
    primary = None
    result = None
    cleanup_errors = []
    try:
        rt.write('controller-start-intent.json', {'argv': controller_unit(rt.token, rt.argv('controller'))})
        rt.host.command(controller_unit(rt.token, rt.argv('controller')), 10)
        # Poll the independently managed service, not a long-running Docker CLI.
        rt.wait_controller(1100)
    except BaseException as exc:
        primary = repr(exc)
    finally:
        # Independent recovery even on start failure, timeout, or controller loss.
        try:
            rt.stop_controller()
        except BaseException as exc:
            cleanup_errors.append(repr(exc))
        try:
            result = rt.collect_and_finalize()
        except BaseException as exc:
            result = {'complete': False, 'collector_error': repr(exc), 'resources_retained': True}
        result['bootstrap_error'] = primary
        result['bootstrap_cleanup_errors'] = cleanup_errors
        result['complete'] = bool(result.get('complete') and primary is None and not cleanup_errors)
    return rt.finish(result)


def controller(rt):
    journal = rt.evidence('journal')
    coordinator = Coordinator(rt.host, journal, rt.token)
    coordinator.l.control_processes = [rt.read('bootstrap.claim.json')['process']]
    operation = rt.evidence('operation')
    def save_capsule(identity, name, record=None):
        capsule = {'token': rt.token, 'boot_id': coordinator.l.boot, 'identity': identity,
                   'controller': rt.me(), 'resources': deepcopy(coordinator.l.resources), 'record': record,
                   'control_processes': deepcopy(coordinator.l.control_processes)}
        rt.write(name, capsule)
    coordinator.before_watchdog = lambda identity: save_capsule(identity, 'watchdog-bootstrap.json')
    coordinator.after_record = lambda record: save_capsule(coordinator.l.cgroup_identity(), 'full-state.json', record)
    inventory = {p: r['sha256'] for p, r in rt.g.inventory['executables'].items()}
    result = coordinator.execute(inventory, rt.argv('watchdog'), rt.argv('helper'), '1', '1', operation)
    rt.write('controller-result.json', result)
    # Bootstrap owns final disposal; the watchdog stays independently alive.
    return result


def watchdog(rt):
    from watchdog_worker import worker
    capsule = rt.read('watchdog-bootstrap.json')
    report = rt.evidence('watchdog')
    life = restore(rt.host, report, capsule, rt.token)
    result = worker(life, capsule['identity'], capsule['controller'], report, lambda: rt.controller_done())
    rt.write('watchdog-result.json', result)
    # Remain observable until independently stopped by finalization. No busy loop.
    rt.park_watchdog(1200)
    return result


def helper(rt):
    from synthetic_helper import blocks
    # Snapshot admission binds fixed scenario/role chain. No images/containers.
    rt.read('full-state.json')
    if rt.g.args.scenario == 'timeout':
        rt.synthetic_wait(65)  # exceeds operation budget; systemd/parent terminate
    else:
        for block in blocks(rt.g.args.scenario): rt.output(block)
    return {'complete': True, 'scope': 'synthetic_helper_only'}


class Runtime:
    def __init__(self, grant, io, host):
        need(valid_grant(grant), 'runtime grant')
        self.g, self.io, self.host = grant, io, host
        self.token, self.root = grant.args.token, STATE+'/'+grant.args.token
        self.reservations = Reservations(io, grant)

    def argv(self, role): return command(role, self.token, self.g.args.scenario)
    def me(self): return self.host.process(os.getpid())
    def write(self, name, value): self.io.exclusive(self.root+'/'+name, value)
    def read(self, name): return self.io.json(self.root+'/'+name)

    def verify_role(self):
        from systemd_ownership import SystemdOwnership
        from linux_host import resolve_cgroup
        mount, group = resolve_cgroup(self.host.text('/proc/self/mountinfo'), self.host.text('/proc/self/cgroup'))
        role = self.g.args.role
        suffix = 'operation' if role == 'helper' else role
        unit = 'pv2-'+self.token+'-'+suffix+'.service'
        raw = self.host.command(['systemctl', 'show', unit, '--property=Id,ActiveState,MainPID,ControlGroup'], 10).decode()
        found = dict(x.split('=', 1) for x in raw.splitlines() if '=' in x)
        need(found.get('Id') == unit and found.get('ActiveState') == 'active' and
             found.get('MainPID') == str(os.getpid()) and found.get('ControlGroup') == group,
             'role service identity')
        # Bootstrap must itself be launched in a bounded independent unit by the
        # owner; no uncontained self-reexec or privilege escalation is performed.
        limits = {'memory.max': 256*1024**2, 'pids.max': 16}
        for key, maximum in limits.items():
            need(0 < int(self.host.text(mount+group+'/'+key)) <= maximum, 'role '+key)
        need(self.host.text(mount+group+'/memory.swap.max').strip() == '0', 'role swap')
        q, p = map(int, self.host.text(mount+group+'/cpu.max').split())
        need(0 < q*4 <= p, 'role cpu')
        if role != 'bootstrap':
            parent = 'bootstrap' if role == 'controller' else 'controller'
            saved = self.read(parent+'.claim.json')['process']
            current = self.host.process(saved['pid'])
            need(all(current[k] == saved[k] for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup')), 'parent process changed')

    def evidence(self, name):
        from supervisor_evidence import FileEvidence
        return FileEvidence(self.root+'/'+name)

    def controller_done(self):
        try: self.read('controller-result.json'); return True
        except FileNotFoundError: return False

    def wait_controller(self, timeout):
        deadline = self.host.now()+timeout
        unit = 'pv2-'+self.token+'-controller.service'
        while self.host.now() < deadline:
            remaining = deadline-self.host.now()
            raw = self.host.command(['systemctl', 'show', unit, '--property=Id,ActiveState'], min(10, remaining)).decode()
            need(self.host.now() <= deadline, 'controller wait backend overrun')
            fields = dict(x.split('=', 1) for x in raw.splitlines() if '=' in x)
            need(fields.get('Id') == unit, 'controller wait identity')
            if fields.get('ActiveState') in ('inactive', 'failed'): return
            self.host.pause(min(.1, max(.001, deadline-self.host.now())))
        raise TimeoutError('controller aggregate wait')

    def stop_controller(self):
        unit = 'pv2-'+self.token+'-controller.service'
        need(self.read('controller-start-intent.json')['argv'] == controller_unit(self.token, self.argv('controller')), 'controller start ownership')
        raw = self.host.command(['systemctl', 'show', unit, '--property=Id,ActiveState,MainPID,ControlGroup'], 10).decode()
        fields = dict(x.split('=', 1) for x in raw.splitlines() if '=' in x)
        need(fields.get('Id') == unit, 'controller stop identity')
        if fields.get('ActiveState') in ('inactive', 'failed'): return
        pid = int(fields.get('MainPID', '0'))
        need(pid > 1, 'controller stop PID')
        argv = self.host.text('/proc/'+str(pid)+'/cmdline').rstrip('\0').split('\0')
        expected = self.argv('controller')
        expected[0] = self.g.inventory['executables'][expected[0]]['resolved_path']
        need(argv == expected and self.host.process(pid)['cgroup'] == fields.get('ControlGroup'), 'controller stop argv/cgroup')
        self.host.command(['systemctl', 'stop', unit], 10)
        try: self.host.process(pid)
        except FileNotFoundError: return
        raise ValueError('controller stop unconfirmed')

    def synthetic_wait(self, seconds):
        end = self.host.now()+seconds
        while self.host.now() < end: self.host.pause(min(1, end-self.host.now()))

    def park_watchdog(self, seconds): self.synthetic_wait(seconds)

    def wait_watchdog(self, timeout=35):
        deadline = self.host.now()+timeout
        while self.host.now() < deadline:
            try:
                result = self.read('watchdog-result.json')
                need(result.get('token') == self.token and result.get('termination_confirmed') is True
                     and result.get('errors') == [] and 'persistence_error' not in result, 'watchdog result failed')
                return result
            except FileNotFoundError:
                self.host.pause(min(.1, max(.001, deadline-self.host.now())))
        raise TimeoutError('independent watchdog result deadline')
    def output(self, block):
        import sys
        sys.stdout.buffer.write(block); sys.stdout.buffer.flush()

    def collect_and_finalize(self):
        # Start only after controller absence is independently confirmed.
        try:
            owner = self.read('controller.claim.json')['process']
            current = self.host.process(owner['pid'])
            need(not all(current.get(k) == owner.get(k) for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup')),
                 'controller still active; refuse concurrent recovery')
        except FileNotFoundError:
            pass
        try:
            capsule = self.read('full-state.json')
        except FileNotFoundError:
            # Partial provisioning is evidence, not ownership authorization.
            # The armed watchdog still terminates workload if it was started.
            partial = {'complete': False, 'resources_retained': True, 'reason': 'no complete ownership capsule'}
            try:
                early = self.read('watchdog-bootstrap.json')
                life = restore(self.host, self.evidence('partial-collector'), early, self.token)
                partial['termination'] = life.terminate(early['identity'])
            except BaseException as exc:
                partial['cleanup_error'] = repr(exc)
            return partial
        journal = self.evidence('collector')
        life = restore(self.host, journal, capsule, self.token)
        # Independently armed worker must also report its observation. Merely
        # killing it during successful controller cleanup is not watchdog evidence.
        watchdog_result = self.wait_watchdog()
        need(watchdog_result['ownership'] == capsule['identity'], 'watchdog ownership result')
        recovered = recover_records(self.root+'/journal', self.token, life.boot)
        need(not recovered['unresolved_intents'] and recovered['observed'] == capsule['resources'], 'journal/capsule mismatch')
        c = Coordinator.__new__(Coordinator)
        c.l, c.host, c.journal = life, self.host, journal
        c.services = Services(self.host, life)
        backend = LinuxBackend(life, c.services, self.argv('helper'))
        backend.record = capsule['record']
        # Loss may precede supervisor cleanup. Observe mount presence, then use
        # the same receipt/identity checks, never infer successful termination.
        termination = life.terminate(capsule['identity'])
        life.require_termination(termination)
        from linux_host import mount_records
        if any(m['mount'] == life.root+'/data' for m in mount_records(self.host.text('/proc/self/mountinfo'))):
            data = next(x['identity'] for x in life.resources if x['kind'] == 'mount' and x['identity']['mount']['mount'] == life.root+'/data')
            life.detach_volume('data', data, termination)
        from evidence_export import LocalEvidenceExport
        exporter = LocalEvidenceExport(life.root+'/evidence', self.root+'/export')
        def manifest():
            files = {}
            for path in Path(life.root+'/evidence').iterdir():
                need(path.is_file() and not path.is_symlink(), 'evidence member')
                size = path.stat().st_size
                need(size <= 512*1024**2, 'evidence size')
                files[path.name] = {'sha256': self.host.digest(str(path)), 'size_bytes': size}
            return {'files': files}
        teardown = c.finalize(backend, exporter, manifest, self.evidence('final'))
        try: outcome = self.read('controller-result.json')
        except FileNotFoundError: outcome = {'state': 'FAILED', 'primary_error': 'controller loss'}
        # Cleanup completion never erases a failed/missing operation result.
        return {'complete': teardown.get('complete', False) and outcome.get('state') == 'VERIFIED_NATIVE',
                'operation': outcome, 'watchdog': watchdog_result, 'teardown': teardown, 'scope': 'A_native_synthetic_only'}

    def finish(self, result):
        result = dict(result, pilot_execution_authorized=False, stage_b='paused', dependency_inspection_authorized=False)
        outcome = bool(result.get('complete'))
        provisional = dict(result, complete=False, operation_teardown_complete=outcome, state='EVIDENCE_PENDING')
        try:
            self.write('activation-result.json', provisional)
            self.write('activation-manifest.json', {'files': self.collect_manifest()})
            self.emit(provisional)  # stdout is provisional before the final durable receipt
            self.write('activation-receipt.json', {'manifest_sha256': self.host.digest(self.root+'/activation-manifest.json'),
                                                 'complete': outcome, 'scope': 'A_native_synthetic_only'})
        except BaseException as exc:
            result = dict(result, complete=False, state='FAILED', persistence_error=repr(exc))
            try: self.write('activation-persistence-failure.json', result)
            except BaseException as persistence: result['failure_record_error'] = repr(persistence)
            try: self.emit(result)
            except BaseException as fallback: result['fallback_error'] = repr(fallback)
        return result

    def emit(self, value):
        raw = json.dumps(value, sort_keys=True)
        need(len(raw.encode()) <= 128*1024, 'activation stdout bound')
        print(raw, flush=True)

    def collect_manifest(self):
        files, total = {}, 0
        for path in sorted(Path(self.root).rglob('*')):
            need(not path.is_symlink(), 'final symlink')
            if path.is_file():
                total += path.stat().st_size
                need(total <= 768*1024**2 and len(files) < 2048, 'final export bound')
                files[path.relative_to(self.root).as_posix()] = {'sha256': self.host.digest(str(path)), 'size_bytes': path.stat().st_size}
        return files


def validate_activation(root):
    """Read-only offline acceptance check. A provisional result alone is invalid."""
    from supervisor_evidence import validate_complete
    root = Path(root)
    need(not root.is_symlink(), 'activation root symlink')
    receipt = json.loads((root/'activation-receipt.json').read_bytes())
    raw = (root/'activation-manifest.json').read_bytes()
    need(receipt.get('manifest_sha256') == hashlib.sha256(raw).hexdigest() and receipt.get('complete') is True
         and receipt.get('scope') == 'A_native_synthetic_only', 'activation incomplete')
    entries = json.loads(raw)['files']
    need(0 < len(entries) <= 2048, 'activation member count')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual == set(entries) | {'activation-manifest.json', 'activation-receipt.json'}
         and 'activation-persistence-failure.json' not in actual, 'activation coverage/failure')
    total = 0
    for name, rec in entries.items():
        need(not name.startswith('/') and '\\' not in name and '..' not in name.split('/'), 'activation member path')
        path = root/name
        need(not any(p.is_symlink() for p in [path, *path.parents]) and path.is_file(), 'activation member symlink/type')
        total += path.stat().st_size
        need(total <= 768*1024**2 and path.stat().st_size == rec['size_bytes'], 'activation size')
        h = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(65536): h.update(block)
        need(h.hexdigest() == rec['sha256'], 'activation bytes')
    result = json.loads((root/'activation-result.json').read_bytes())
    need(result.get('state') == 'EVIDENCE_PENDING' and result.get('complete') is False and
         result.get('operation_teardown_complete') is True, 'activation provisional binding')
    need(validate_complete(root/'operation')['outcome'] == 'VERIFIED_NATIVE', 'native operation receipt')
    need(validate_complete(root/'final')['outcome'] == 'CLEANUP_VERIFIED_NATIVE', 'native teardown receipt')
    return receipt
