"""Internal orchestration source; verified native grant or injected test host.

All commands below are source for Linux acceptance, not commands run by tests.
The supplied host must implement the LinuxHost OS boundary, never Docker APIs
against a shared socket. Missing observations block admission.
"""
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from linux_lifecycle import LinuxLifecycle
from linux_host import mount_records, resolve_cgroup
from supervisor_v2 import POLICY, Supervisor, require, within, validate_record


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def recover_records(root, token, boot):
    """Read-only journal recovery: complete hash coverage, pairing, no guesses.

    A durable intent without observation is returned unresolved. It cannot be
    authorized for cleanup by path alone. A new host observation/reconciliation
    is required; the coordinator blocks re-admission while intents remain.
    """
    root = Path(root)
    require(not root.is_symlink(), 'journal symlink')
    ledger = json.loads((root / 'report-hashes.json').read_bytes())
    require(set(ledger) == {p.name for p in root.iterdir()} - {'report-hashes.json'}, 'journal coverage')
    pending, observed = [], []
    for name, item in sorted(ledger.items()):
        require(re.fullmatch(r'[0-9]{4,}-[a-z-]+\.json', name), 'journal name')
        p = root / name
        require(not p.is_symlink() and p.is_file(), 'journal member')
        raw = p.read_bytes()
        require(len(raw) == item['size_bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'journal hash')
        value = json.loads(raw)
        if 'provision-intent' in name:
            require(value.get('boot_id') == boot and token in str(value.get('requested')), 'journal ownership')
            pending.append(value)
        elif 'provision-observed' in name:
            key = {k: v for k, v in value.items() if k != 'identity'}
            key['phase'] = 'intent'
            require(key in pending, 'unpaired observation')
            pending.remove(key)
            observed.append(value)
    return {'observed': observed, 'unresolved_intents': pending, 'admission_allowed': not pending}


class Services:
    def __init__(self, host, lifecycle):
        self.host, self.l = host, lifecycle
        self.arguments = {}

    def inspect(self, unit, deadline=None):
        deadline = self.host.now() + 10 if deadline is None else deadline
        raw = self.l.manager.command_until(['systemctl', 'show', unit, '--property=Id,ActiveState,MainPID,ControlGroup'],
                                           deadline, 'service identity').decode()
        values = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        require(values.get('Id') == unit and values.get('ActiveState') == 'active', 'service not independently active')
        pid = int(values.get('MainPID', '0'))
        require(pid > 1, 'missing service PID')
        proc = self.host.process(pid)
        require(proc['cgroup'] == values.get('ControlGroup'), 'service/proc cgroup mismatch')
        self.l.manager.check_return(deadline, 'service identity read-back')
        return proc

    def start(self, role, argv, outside=False):
        require(role in ('containerd', 'dockerd', 'watchdog'), 'service role')
        unit = 'pv2-' + self.l.token + '-' + role + '.service'
        self.arguments[role] = list(argv)
        slice_name = self.l.manager.guard_unit if outside else self.l.manager.work_unit
        command = ['systemd-run', '--unit=' + unit, '--slice=' + slice_name, '--service-type=exec',
                   '--property=Restart=no', '--property=KillMode=control-group',
                   '--property=TimeoutStopSec=10', '--property=RuntimeMaxSec=1200',
                   '--property=MemoryMax=' + str(256 * 1024**2 if outside else POLICY['memory_max']),
                   '--property=MemorySwapMax=0', '--property=TasksMax=' + ('16' if outside else '256'),
                   '--property=CPUQuota=' + ('25%' if outside else '150%'),
                   '--property=StandardOutput=append:' + self.l.root + '/evidence/' + role + '.log',
                   '--property=StandardError=inherit', '--setenv=TMPDIR=' + self.l.root + '/data/tmp', '--', *argv]
        def observe():
            p = self.inspect(unit)
            expected = self.l.manager.show(slice_name)['ControlGroup']
            require(within(p['cgroup'], expected), 'service escaped reported slice')
            require(not outside or not within(p['cgroup'], self.l.work_path), 'watchdog in workload')
            return dict(p, role=role, unit=unit)
        return self.l.acquire('service', unit, lambda: self.host.command(command, 10), observe)


class LinuxBackend:
    """Supervisor Backend.request/read adapter; fresh native observations."""
    is_fake = True
    def __init__(self, lifecycle, services, operation):
        from activation_trust import host_allowed
        require(host_allowed(lifecycle.host), 'verified backend capability required')
        self.is_fake = lifecycle.host.is_fake
        self.grant = getattr(lifecycle.host, 'grant', None)
        self.l, self.host, self.services, self.operation = lifecycle, lifecycle.host, services, operation
        self.handle = None
        self.result = None
        self.record = None
        self.terminated = None
        self.cleanup_phase_errors = []

    def now(self):
        return self.host.now()

    def processes(self):
        # Scan /proc globally, not only children visible in the expected subtree.
        # Unreadable process metadata is an error, not silently omitted.
        owned, escaped = [], []
        for path in self.host.children('/proc'):
            if not path.name.isdigit():
                continue
            p = self.host.process(int(path.name))
            argv = self.host.text(str(path) + '/cmdline')
            if any(all(p.get(k) == saved.get(k) for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup'))
                   for saved in self.l.control_processes):
                continue
            baseline = getattr(self.host, 'baseline_processes', {})
            if any(name in argv for name in ('containerd-shim', '/runc', '/dockerd', '/containerd')) and not within(p['cgroup'], self.l.work_path):
                require(baseline.get(p['pid']) == p, 'new or changed external runtime process; ownership inconclusive')
            belongs = self.l.token in argv or within(p['cgroup'], self.l.work_path)
            if not belongs:
                continue
            if (self.l.guard_path and within(p['cgroup'], self.l.guard_path)) or within(p['cgroup'], self.l.parent):
                continue
            if not within(p['cgroup'], self.l.work_path):
                escaped.append(p)
            else:
                owned.append(p)
        return owned, escaped

    def snapshot(self, record, deadline=None):
        identity = self.l.cgroup_identity(deadline=deadline)
        owner = record['systemd_owner']
        require(self.l.manager.show(owner['guard_unit'], deadline=deadline)['ControlGroup'] == owner['guard_path'], 'guard slice path changed')
        require(self.l.manager.show(owner['controller_unit'], deadline=deadline)['ControlGroup'] == owner['controller_path'], 'controller unit path changed')
        _, controller_path = resolve_cgroup(self.host.text('/proc/self/mountinfo'), self.host.text('/proc/self/cgroup'))
        require(controller_path == owner['controller_path'], 'controller process moved')
        require(identity['inode'] == record['cgroup_inode'] and identity['boot_id'] == record['boot_id'], 'cgroup changed')
        policy = {'memory_max': int(self.host.text(self.l.cg + '/memory.max')),
                  'swap_max': int(self.host.text(self.l.cg + '/memory.swap.max')),
                  'cpu_max': [int(x) for x in self.host.text(self.l.cg + '/cpu.max').split()],
                  'pids_max': int(self.host.text(self.l.cg + '/pids.max')),
                  'workload_seconds': 1200, 'operation_seconds': 60}
        processes, escaped = self.processes()
        saved = {p['pid']: p for p in record['processes']}
        normalized = []
        for p in processes:
            require(all(within(path, self.l.root + '/data') or within(path, self.l.root + '/evidence') or within(path, self.l.runtime)
                        for path in self.host.writable_paths(p['pid'])), 'writable FD outside bounded storage')
            q = {k: p[k] for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup')}
            q['role'] = saved.get(p['pid'], {}).get('role', 'descendant')
            normalized.append(q)
        volumes = {}
        for role, volume in record['volumes'].items():
            current = self.l.volume_identity(role, volume['loop'], deadline=deadline)
            volumes[role] = volume_record(current, self.host)
        paths = record['paths']
        # Independently bind daemon invocation, private socket and all configured
        # storage paths. No config-file default or shared daemon acceptance.
        for role in ('dockerd', 'containerd'):
            found = [p for p in record['processes'] if p['role'] == role]
            require(len(found) == 1, 'daemon record')
            if any(p['pid'] == found[0]['pid'] for p in processes):
                argv = self.host.text('/proc/' + str(found[0]['pid']) + '/cmdline').split('\0')
                if argv and argv[-1] == '':
                    argv.pop()
                require(argv == self.services.arguments[role], 'daemon exact argv binding')
                require(self.host.socket_owned(found[0]['pid'], paths['docker_socket' if role == 'dockerd' else 'containerd_socket']),
                        'private socket inode owner mismatch')
        for name, path in paths.items():
            require(within(path, self.l.runtime) or within(path, self.l.root + '/data') or within(path, self.l.root + '/evidence'), 'storage escape')
            self.host.assert_no_symlinks(path)
        for resource in self.l.resources:
            if resource['kind'] == 'configuration':
                c = resource['identity']
                require(self.host.stat(c['path']).st_ino == c['inode'] and self.host.digest(c['path']) == c['sha256'], 'configuration changed')
        runtime = [m for m in mount_records(self.host.text('/proc/self/mountinfo')) if m['mount'] == self.l.runtime]
        require(len(runtime) == 1 and runtime[0]['filesystem'] == 'tmpfs', 'runtime mount')
        runtime_stats = self.host.filesystem(self.l.runtime)
        require(runtime_stats['capacity_bytes'] <= 64 * 1024**2, 'runtime capacity')
        watchdog = self.services.inspect('pv2-' + self.l.token + '-watchdog.service', deadline=deadline)
        control = self.l.cgmount + watchdog['cgroup']
        require(int(self.host.text(control + '/memory.max')) <= 256 * 1024**2 and
                self.host.text(control + '/memory.swap.max').strip() == '0' and
                0 < int(self.host.text(control + '/pids.max')) <= 16, 'watchdog enforced controls')
        quota, period = [int(x) for x in self.host.text(control + '/cpu.max').split()]
        require(0 < quota <= period // 4, 'watchdog CPU control')
        wd = {k: watchdog[k] for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup')}
        wd.update(outside_workload=True, lifetime_seconds=1200)
        # Worker must publish its own armed identity, not a controller boolean.
        armed = json.loads(self.host.text(self.l.root + '/evidence/watchdog-armed.json'))
        require(armed == {'token': self.l.token, 'boot_id': self.l.boot, 'watchdog': wd}, 'watchdog handshake')
        fs = self.host.filesystem(self.l.root + '/data')
        events = dict(line.split() for line in self.host.text(self.l.cg + '/cgroup.events').splitlines())
        listeners = [p for p in (paths['docker_socket'], paths['containerd_socket']) if p in self.host.text('/proc/net/unix')]
        if deadline is not None:
            self.l.manager.check_return(deadline, 'snapshot completion')
        return {'identity': {'boot_id': identity['boot_id'], 'cgroup_inode': identity['inode'], 'token': self.l.token},
                'paths': deepcopy(paths), 'volumes': volumes, 'policy': policy, 'processes': normalized,
                'external_owned_processes': escaped, 'all_storage_confined': True,
                'watchdog_armed': True, 'watchdog_identity': wd, 'free_bytes': fs['free_bytes'],
                'free_inodes': fs['free_inodes'], 'populated': events.get('populated') != '0', 'listeners': listeners}

    def request(self, action, record, timeout):
        require(0 < timeout <= 10, 'control timeout')
        require(record == self.record, 'backend record binding')
        deadline = self.host.now() + timeout
        if action == 'snapshot':
            return self.snapshot(record, deadline=deadline)
        if action == 'begin_operation':
            require(self.l.admission and self.handle is None, 'operation admission')
            # systemd-run service stays below the parent; killing its client is
            # not the cleanup mechanism. No shell expansion or shared socket.
            argv = ['systemd-run', '--pipe', '--wait', '--unit=pv2-' + self.l.token + '-operation.service',
                    '--slice=' + self.l.manager.work_unit, '--property=RuntimeMaxSec=60',
                    '--property=KillMode=control-group', '--property=MemoryMax=268435456',
                    '--property=MemorySwapMax=0', '--property=TasksMax=16',
                    '--property=CPUQuota=25%', '--', *self.operation]
            self.handle = self.host.start(argv)
            return {'handle': 'operation', 'deadline_seconds': 60}
        if action == 'operation_status':
            return {'exit_code': self.handle.poll()}
        if action == 'close_admission':
            self.l.admission = False
            return {}
        if action in ('term_service', 'kill_subtree'):
            identity = {'path': self.l.cg, 'inode': record['cgroup_inode'], 'boot_id': record['boot_id']}
            try:
                self.l.signal_phase(identity, 'TERM' if action == 'term_service' else 'KILL', timeout=timeout)
            except BaseException as exc:
                self.cleanup_phase_errors.append(action + ': ' + repr(exc))
                raise
            return {}
        if action == 'wait_empty':
            identity = {'path': self.l.cg, 'inode': record['cgroup_inode'], 'boot_id': record['boot_id']}
            self.terminated = self.l.read_termination(identity, self.cleanup_phase_errors, deadline=deadline)
            self.l.require_termination(self.terminated, deadline=deadline)
            return self.snapshot(record, deadline=deadline)
        if action == 'unmount_data':
            self.l.require_termination(self.terminated)
            self.l.detach_volume('data', self.l.volume_identity('data', record['volumes']['data']['loop']), self.terminated)
            return {}
        if action in ('read_mount', 'read_loop', 'detach_data_loop'):
            v = record['volumes']['data']
            mounts = mount_records(self.host.text('/proc/self/mountinfo'))
            loops = json.loads(self.host.command(['losetup', '--json', '--associated', v['backing'], '--output', 'NAME,BACK-FILE'], 10))['loopdevices']
            require(not loops, 'data loop detach unconfirmed')
            return {'record_identity': deepcopy(v), 'mounted': any(m['mount'] == v['mount'] for m in mounts), 'attached': False}
        raise ValueError('unknown backend action: ' + action)

    def read(self, handle, maximum, timeout):
        require(handle == 'operation' and self.handle is not None, 'stream identity')
        return self.host.read(self.handle, maximum, timeout)


def volume_record(observed, host):
    fs = host.filesystem(observed['mount']['mount'])
    return {'mount': observed['mount']['mount'], 'backing': observed['backing'], 'loop': observed['loop'],
            'capacity_bytes': observed['size'], 'allocated_bytes': observed['allocated'], 'filesystem': 'ext4',
            'uuid': observed['uuid'], 'device': observed['mount']['device'], 'mount_id': observed['mount']['id'],
            'inode_capacity': fs['inode_capacity'], 'auto_expand': False}


class Coordinator:
    def __init__(self, host, journal, token):
        self.l = LinuxLifecycle(host, journal, token)
        self.host, self.journal = host, journal
        control = self.l.cgmount + self.l.parent
        require(0 < int(host.text(control + '/memory.max')) <= 256 * 1024**2 and
                host.text(control + '/memory.swap.max').strip() == '0' and
                0 < int(host.text(control + '/pids.max')) <= 16, 'controller enforced bounds required')
        quota, period = [int(x) for x in host.text(control + '/cpu.max').split()]
        require(0 < quota <= period // 4, 'controller CPU bound')
        controller_unit = 'pv2-' + token + '-controller.service'
        require(self.l.manager.show(controller_unit)['ControlGroup'] == self.l.parent, 'controller manager/proc binding')
        self.controller_unit = controller_unit
        host.workload_slice = self.l.manager.work_unit
        host.baseline_processes = {int(p.name): host.process(int(p.name)) for p in host.children('/proc') if p.name.isdigit()}
        self.services = Services(host, self.l)

    def directory(self, path):
        return self.l.acquire('directory', path, lambda: self.host.mkdir(path),
                              lambda: self.l.entry_identity(path))

    def provision(self, inventory, watchdog_argv, operation, run_id, attempt):
        # No daemon before durable volume/cgroup records and armed watchdog.
        l = self.l
        self.directory(l.root)
        identity = l.provision_cgroup()
        volumes = {role: l.provision_volume(role) for role in ('data', 'evidence')}
        self.directory(l.runtime)
        l.acquire('runtime-mount', l.runtime,
                  lambda: self.host.command(['mount', '-t', 'tmpfs', '-o', 'size=67108864,nodev,nosuid,noexec', 'tmpfs', l.runtime], 10),
                  lambda: next(m for m in mount_records(self.host.text('/proc/self/mountinfo')) if m['mount'] == l.runtime))
        for path in (l.root + '/data/tmp', l.root + '/data/downloads'):
            self.directory(path)
        if hasattr(self, 'before_watchdog'):
            self.before_watchdog(identity)
        watchdog = self.services.start('watchdog', watchdog_argv, outside=True)
        wd = {k: watchdog[k] for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup')}
        wd.update(outside_workload=True, lifetime_seconds=1200)
        deadline = self.host.now() + 10
        while True:
            try:
                armed = json.loads(self.host.text(l.root + '/evidence/watchdog-armed.json'))
                break
            except FileNotFoundError:
                require(self.host.now() < deadline, 'watchdog arm timeout')
                self.host.pause(.05)
        require(armed == {'token': l.token, 'boot_id': l.boot, 'watchdog': wd}, 'watchdog did not arm')
        args = l.daemon_arguments(inventory)
        configs = {'dockerd': (l.runtime + '/dockerd.json', b'{}\n'),
                   'containerd': (l.runtime + '/containerd.toml', b'version = 2\ndisabled_plugins = ["io.containerd.grpc.v1.cri"]\n')}
        for role, (path, raw) in configs.items():
            l.acquire('configuration', path, lambda p=path, b=raw: self.host.write_exclusive(p, b),
                      lambda p=path, b=raw: dict(l.entry_identity(p), sha256=hashlib.sha256(b).hexdigest()))
            args[role].extend((['--config-file', path] if role == 'dockerd' else ['--config', path]))
        processes = []
        for role in ('containerd', 'dockerd'):
            p = self.services.start(role, args[role])
            processes.append({k: p[k] for k in ('role', 'pid', 'start_ticks', 'executable_sha256', 'cgroup')})
        r = {'schema': 'pv2_ownership_v3', 'token': l.token, 'run_id': run_id, 'attempt': attempt,
             'root': l.root, 'runtime_root': l.runtime, 'cgroup': l.work_path,
             'systemd_owner': {'workload_unit': l.manager.work_unit, 'workload_path': l.work_path,
                               'guard_unit': l.manager.guard_unit, 'guard_path': l.guard_path,
                               'controller_unit': self.controller_unit, 'controller_path': l.parent},
             'cgroup_inode': identity['inode'], 'boot_id': identity['boot_id'], 'policy': deepcopy(POLICY),
             'service': 'pv2-' + l.token + '.service', 'runtime_tmpfs_bytes': 64 * 1024**2,
             'network': {'bridge': False, 'iptables': False, 'ip_forward': False},
             'container_cgroup_parent': l.manager.work_unit, 'supervisor_cgroup': l.parent,
             'watchdog': wd, 'processes': processes,
             'volumes': {role: volume_record(v, self.host) for role, v in volumes.items()},
             'paths': {'docker_socket': l.runtime + '/docker.sock', 'containerd_socket': l.runtime + '/containerd.sock',
                       'pidfile': l.runtime + '/dockerd.pid', 'exec_root': l.runtime + '/exec',
                       'data_root': l.root + '/data/docker', 'containerd_root': l.root + '/data/containerd',
                       'containerd_state': l.runtime + '/containerd', 'download_root': l.root + '/data/downloads',
                       'scratch_root': l.root + '/data/tmp', 'daemon_log': l.root + '/evidence/daemon.log'}}
        validate_record(r)
        backend = LinuxBackend(l, self.services, operation)
        backend.record = deepcopy(r)
        backend.snapshot(r)
        if hasattr(self, 'after_record'):
            self.after_record(r)
        l.admission = True
        return backend, r

    def recover(self, root):
        recovered = recover_records(root, self.l.token, self.l.boot)
        self.l.resources = recovered['observed']
        require(not recovered['unresolved_intents'], 'recovery blocked: intent without observed identity')
        self.l.owned_slices = {x['requested']: x['identity'] for x in recovered['observed'] if x['kind'] == 'systemd-slice'}
        require(set(self.l.owned_slices) == {self.l.manager.work_unit, self.l.manager.guard_unit}, 'recovery slice ownership incomplete')
        identity = self.l.cgroup_identity()
        self.l.guard_path = self.l.manager.show(self.l.manager.guard_unit)['ControlGroup']
        require(not within(self.l.guard_path, self.l.work_path) and not within(self.l.work_path, self.l.guard_path), 'recovered branches nested')
        self.l.admission = False
        result = self.l.terminate(identity)
        return self.l.partial_cleanup(result)

    def finalize(self, backend, exporter, manifest, final_sink):
        """Local export acknowledgement then bounded teardown, no upload.

        The external final_sink must be outside all removed resources. Failures
        retain backing files; final journal completion is never fabricated.
        """
        l = self.l
        if hasattr(final_sink, 'root'):
            destination = str(final_sink.root.resolve())
            require(not within(destination, l.root) and not within(destination, l.runtime), 'final receipt must survive teardown')
        report = {'execution_approved': False, 'stage_b': 'paused', 'complete': False, 'errors': []}
        try:
            termination = l.terminate(l.cgroup_identity())
            l.require_termination(termination)
            evidence = next(x['identity'] for x in l.resources if x['kind'] == 'mount' and x['identity']['mount']['mount'] == l.root + '/evidence')
            # Stop the external writer before copying a stable evidence snapshot.
            wd = self.services.inspect('pv2-' + l.token + '-watchdog.service')
            require(within(wd['cgroup'], l.manager.show(l.manager.guard_unit)['ControlGroup']) and not within(wd['cgroup'], l.work_path), 'watchdog ownership')
            require(all(wd[k] == backend.record['watchdog'][k] for k in ('pid', 'start_ticks', 'cgroup', 'executable_sha256')), 'watchdog identity replaced')
            self.host.command(['systemctl', 'stop', 'pv2-' + l.token + '-watchdog.service'], 10)
            try:
                self.host.process(wd['pid'])
            except FileNotFoundError:
                pass
            else:
                raise ValueError('watchdog termination not confirmed')
            # Generate the final manifest only after the external writer stops.
            final_manifest = manifest() if callable(manifest) else manifest
            report['export'] = l.export_and_ack(exporter, final_manifest, evidence, termination)
            fresh = l.terminate(l.cgroup_identity())
            l.require_termination(fresh)
            runtime = next(x['identity'] for x in l.resources if x['kind'] == 'runtime-mount')
            current = [m for m in mount_records(self.host.text('/proc/self/mountinfo')) if m['mount'] == l.runtime]
            require(current == [runtime], 'runtime mount changed')
            self.host.command(['umount', l.runtime], 10)
            require(not any(m['mount'] == l.runtime for m in mount_records(self.host.text('/proc/self/mountinfo'))), 'runtime mount survives')
            # Remove only hash-journaled backing inodes after both mount and loop
            # absence read-back. Never delete retained evidence on failed export.
            for item in l.resources:
                if item['kind'] != 'backing':
                    continue
                v = item['identity']
                loops = json.loads(self.host.command(['losetup', '--json', '--associated', v['path'], '--output', 'NAME,BACK-FILE'], 10))['loopdevices']
                require(loops == [], 'backing still attached')
                l.require_termination(fresh)
                l.verify_entry(v)
                self.host.remove_owned(v['path'], v['inode'])
            l.require_termination(fresh)
            # systemd is the sole cgroup owner: stop units, never rmdir its tree.
            l.manager.stop(l.manager.work_unit, l.work_path, l.owned_slices[l.manager.work_unit]['inode'])
            l.manager.stop(l.manager.guard_unit, l.guard_path, l.owned_slices[l.manager.guard_unit]['inode'])
            report['filesystem_internal_entries_retired'] = []
            detached = {x['identity']['mount']['id'] for x in l.resources if x['kind'] == 'mount'} | {runtime['id']}
            for item in reversed(l.resources):
                if item['kind'] in ('directory', 'mount-directory'):
                    v = item['identity']
                    if v['disposition'] == 'filesystem-internal':
                        require(v['filesystem']['id'] in detached, 'unknown internal filesystem')
                        require(not any(m['id'] == v['filesystem']['id'] for m in mount_records(self.host.text('/proc/self/mountinfo'))), 'internal filesystem still mounted')
                        report['filesystem_internal_entries_retired'].append(v)
                        continue  # identities belong to disposed FS, never stat uncovered paths
                    l.verify_entry(v)
                    self.host.remove_owned(v['path'], v['inode'], directory=True)
            report['complete'] = True
        except BaseException as exc:
            report['errors'].append(repr(exc))
        # Reuse the accepted provisional-report/last-receipt finalizer rather
        # than introducing another saved-success-before-ledger failure mode.
        finalizer = Supervisor(backend, final_sink)
        finalizer.report.update({'teardown': deepcopy(report), 'complete': False,
                                 'state': ('CLEANUP_VERIFIED_FAKE' if backend.is_fake else 'CLEANUP_VERIFIED_NATIVE') if report['complete'] else 'FAILED'})
        finalizer.persist()
        finalizer.report['complete'] = report['complete'] and finalizer.report['evidence_finalized']
        return deepcopy(finalizer.report)

    def execute(self, inventory, watchdog_argv, operation, run_id, attempt, evidence):
        try:
            backend, record = self.provision(inventory, watchdog_argv, operation, run_id, attempt)
            return Supervisor(backend, evidence).run(record)
        except BaseException as exc:
            report = {'state': 'FAILED', 'primary_error': repr(exc), 'cleanup_errors': [], 'execution_approved': False}
            try:
                result = self.l.terminate(self.l.cgroup_identity())
                report['cleanup'] = self.l.partial_cleanup(result)
            except BaseException as cleanup:
                report['cleanup_errors'].append(repr(cleanup))
            try:
                evidence.append('coordinator-failure', report)
                evidence.seal()
            except BaseException as persistence:
                report['persistence_error'] = repr(persistence)
            return report
