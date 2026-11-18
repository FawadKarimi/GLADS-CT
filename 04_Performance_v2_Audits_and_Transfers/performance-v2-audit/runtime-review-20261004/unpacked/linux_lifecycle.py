"""Review-only Linux lifecycle implementation, all effects through injected host.

No activation on import; native hosts require a verified grant. Journals preserve intent before
mutation, then independent observed identities. Incomplete intent is not proof
of ownership and blocks destructive recovery rather than guessing.
"""
from copy import deepcopy
import hashlib
import json
import re
import math
import secrets

from supervisor_v2 import POLICY, require, within
from linux_host import mount_records, resolve_cgroup
from systemd_ownership import SystemdOwnership


class LinuxLifecycle:
    def __init__(self, host, journal, token):
        from activation_trust import host_allowed
        require(host_allowed(host), 'live lifecycle disabled without verified capability')
        require(re.fullmatch('[0-9a-f]{32}', token), 'token')
        self.host, self.journal, self.token = host, journal, token
        self.root = '/var/tmp/pv2-' + token
        self.runtime = '/run/pv2-' + token
        self.resources, self.errors = [], []
        self.admission = False
        self.reserve_checked = False
        self.latest_termination = None
        self.cgmount, self.parent = resolve_cgroup(host.text('/proc/self/mountinfo'), host.text('/proc/self/cgroup'))
        self.manager = SystemdOwnership(host, token, self.cgmount)
        self.cg = None
        self.work_path = None
        self.guard_path = None
        self.owned_slices = {}
        self.control_processes = []
        self.boot = host.text('/proc/sys/kernel/random/boot_id').strip()

    def acquire(self, kind, identity, mutate, observe):
        intent = {'kind': kind, 'requested': identity, 'boot_id': self.boot, 'phase': 'intent'}
        self.journal.append('provision-intent', intent)
        self.journal.seal()
        mutate()
        observed = observe()
        owned = dict(intent, phase='observed', identity=observed)
        # Retain in memory even if persistence fails, for independently checked cleanup.
        self.resources.append(owned)
        self.journal.append('provision-observed', owned)
        self.journal.seal()
        return observed

    def entry_identity(self, path):
        mounts = [m for m in mount_records(self.host.text('/proc/self/mountinfo')) if within(path, m['mount'])]
        require(mounts, 'containing filesystem unavailable')
        owner = max(mounts, key=lambda m: len(m['mount']))
        internal = owner['mount'] in (self.root + '/data', self.root + '/evidence', self.runtime)
        return {'path': path, 'inode': self.host.stat(path).st_ino, 'filesystem': owner,
                'disposition': 'filesystem-internal' if internal else 'host-entry'}

    def verify_entry(self, identity):
        require(self.entry_identity(identity['path']) == identity, 'filesystem-bound entry changed')

    def cgroup_identity(self, deadline=None):
        path = self.manager.show(self.manager.work_unit, deadline=deadline)['ControlGroup']
        require(self.work_path is None or self.work_path == path, 'reported workload path changed')
        self.work_path, self.cg = path, self.cgmount + path
        observed = {'path': self.cg, 'inode': self.host.stat(self.cg).st_ino,
                    'boot_id': self.host.text('/proc/sys/kernel/random/boot_id').strip()}
        expected = self.owned_slices.get(self.manager.work_unit)
        require(expected is None or expected == observed, 'owned workload slice replaced')
        if deadline is not None:
            self.manager.check_return(deadline, 'cgroup identity read-back')
        return observed

    def provision_cgroup(self):
        controllers = set(self.host.text(self.cgmount + '/cgroup.controllers').split())
        require({'memory', 'cpu', 'pids'} <= controllers, 'required controllers unavailable')
        identity = self.acquire('systemd-slice', self.manager.work_unit,
                                lambda: self.manager.create(self.manager.work_unit, POLICY['memory_max'], 1500000, 256),
                                self.cgroup_identity)
        self.owned_slices[self.manager.work_unit] = deepcopy(identity)
        for filename, value in (('memory.max', str(POLICY['memory_max'])), ('memory.swap.max', '0'),
                                ('cpu.max', '150000 100000'), ('pids.max', '256')):
            self.assert_cgroup(identity)
            actual = self.host.text(self.cg + '/' + filename).strip()
            if filename == 'cpu.max':
                quota, period = [int(x) for x in actual.split()]
                require(quota > 0 and quota * 100000 == 150000 * period, 'CPU quota read-back')
            else:
                require(actual == value, 'limit read-back ' + filename)
        def guard_identity():
            self.guard_path = self.manager.show(self.manager.guard_unit)['ControlGroup']
            require(not within(self.guard_path, self.work_path) and not within(self.work_path, self.guard_path), 'nested independent branches')
            require(not within(self.parent, self.work_path), 'controller in workload')
            path = self.cgmount + self.guard_path
            require(self.host.text(path + '/memory.max').strip() == str(256 * 1024**2) and
                    self.host.text(path + '/memory.swap.max').strip() == '0' and
                    self.host.text(path + '/pids.max').strip() == '16', 'guard slice controls')
            quota, period = [int(x) for x in self.host.text(path + '/cpu.max').split()]
            require(quota > 0 and quota * 4 == period, 'guard slice CPU control')
            return {'path': self.cgmount + self.guard_path, 'inode': self.host.stat(self.cgmount + self.guard_path).st_ino,
                    'boot_id': self.boot}
        guard = self.acquire('systemd-slice', self.manager.guard_unit,
                             lambda: self.manager.create(self.manager.guard_unit, 256 * 1024**2, 250000, 16), guard_identity)
        self.owned_slices[self.manager.guard_unit] = deepcopy(guard)
        return identity

    def assert_cgroup(self, identity, deadline=None):
        require(self.cgroup_identity(deadline=deadline) == identity, 'cgroup identity changed')

    def census(self, path=None):
        path = path or self.cg
        require(within(path, self.cg), 'census escape')
        result = []
        for pid in self.host.text(path + '/cgroup.procs').split():
            p = self.host.process(int(pid))
            require(within(p['cgroup'], self.work_path), 'process escape')
            result.append(p)
        for child in self.host.children(path):
            if child.is_dir() and not child.is_symlink():
                result.extend(self.census(str(child)))
        return result

    def signal_phase(self, identity, signal, timeout=10):
        deadline = self.host.now() + timeout
        self.admission = False
        self.assert_cgroup(identity, deadline=deadline)
        self.manager.kill_members(self.manager.work_unit, self.work_path, signal, timeout=timeout, deadline=deadline)

    def read_termination(self, identity, errors=None, timeout=10, deadline=None):
        errors = list(errors or [])
        deadline = self.host.now() + timeout if deadline is None else deadline
        empty = False
        try:
            while self.host.now() < deadline:
                self.assert_cgroup(identity, deadline=deadline)
                events = dict(x.split() for x in self.host.text(self.cg + '/cgroup.events').splitlines())
                if events.get('populated') == '0' and not self.census():
                    listeners = self.host.text('/proc/net/unix')
                    require(self.runtime + '/docker.sock' not in listeners and self.runtime + '/containerd.sock' not in listeners,
                            'private listener survives')
                    self.manager.check_return(deadline, 'termination read-back')
                    empty = True
                    break
                self.host.pause(min(.05, self.manager.remaining(deadline, 'termination poll')))
            require(empty, 'subtree termination unconfirmed')
        except BaseException as exc:
            errors.append('readback: ' + repr(exc))
        result = {'schema': 'pv2_termination_v1', 'termination_confirmed': empty,
                  'errors': errors, 'token': self.token, 'ownership': deepcopy(identity),
                  'observed_at': self.host.now(), 'nonce': secrets.token_hex(16)}
        self.latest_termination = deepcopy(result)
        return result

    def terminate(self, identity):
        # Independent safe cleanup phases: TERM <=10s, KILL <=10s, read-back
        # <=10s. Failure/timeout in one does not consume or skip the next phase.
        errors = []
        for signal in ('TERM', 'KILL'):
            try:
                self.signal_phase(identity, signal, 10)
            except BaseException as exc:
                errors.append(signal.lower() + ': ' + repr(exc))
        return self.read_termination(identity, errors, timeout=10)

    def require_termination(self, result, deadline=None):
        """A truthy object is never permission; reobserve at each boundary.

        Receipts are valid only in the issuing lifecycle instance, for <=10s.
        Recovery must issue its own receipt. No cached success can authorize
        teardown after repopulation, changed boot/cgroup, or a surviving socket.
        """
        fields = {'schema', 'termination_confirmed', 'errors', 'token', 'ownership', 'observed_at', 'nonce'}
        require(type(result) is dict and set(result) == fields, 'malformed termination result')
        require(result['schema'] == 'pv2_termination_v1' and result['termination_confirmed'] is True,
                'termination was not confirmed')
        require(result['errors'] == [] and type(result['errors']) is list, 'termination errors')
        require(result['token'] == self.token and result == self.latest_termination, 'unrelated or superseded termination')
        require(type(result['observed_at']) in (int, float) and math.isfinite(result['observed_at']), 'termination timestamp')
        require(0 <= self.host.now() - result['observed_at'] <= 10, 'stale termination result')
        self.assert_cgroup(result['ownership'], deadline=deadline)
        events = dict(x.split() for x in self.host.text(self.cg + '/cgroup.events').splitlines())
        require(events.get('populated') == '0' and not self.census(), 'termination no longer holds')
        listeners = self.host.text('/proc/net/unix')
        require(self.runtime + '/docker.sock' not in listeners and self.runtime + '/containerd.sock' not in listeners,
                'private listener survives at destructive boundary')
        # Whole-host check as well as cgroup-local emptiness. Known service PID
        # reuse or escaped token-bearing workers makes teardown inconclusive.
        for entry in self.host.children('/proc'):
            if not entry.name.isdigit():
                continue
            p = self.host.process(int(entry.name))
            argv = self.host.text(str(entry) + '/cmdline')
            if any(all(p.get(k) == saved.get(k) for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup'))
                   for saved in self.control_processes):
                continue  # exact independently observed bootstrap; no branch-wide exemption
            if self.token in argv or within(p['cgroup'], self.work_path):
                require((self.guard_path and within(p['cgroup'], self.guard_path)) or within(p['cgroup'], self.parent),
                        'owned process survives outside empty subtree')
        if deadline is not None:
            self.manager.check_return(deadline, 'termination authorization')
        return True

    def volume_identity(self, role, loop, deadline=None):
        backing, target = self.root + '/' + role + '.ext4', self.root + '/' + role
        mounts = [m for m in mount_records(self.host.text('/proc/self/mountinfo')) if m['mount'] == target]
        require(len(mounts) == 1 and mounts[0]['filesystem'] == 'ext4' and mounts[0]['source'] == loop, 'mount identity')
        def command(argv, stage):
            end = self.host.now() + 10 if deadline is None else deadline
            return self.manager.command_until(argv, end, stage)
        observed = json.loads(command(['losetup', '--json', '--output', 'NAME,BACK-FILE', loop], 'volume loop identity'))['loopdevices']
        require(len(observed) == 1 and observed[0]['name'] == loop and observed[0]['back-file'] == backing, 'loop backing mismatch')
        st = self.host.stat(backing)
        uuid = command(['blkid', '-s', 'UUID', '-o', 'value', loop], 'volume UUID').decode().strip()
        require(uuid, 'filesystem UUID missing')
        return {'mount': mounts[0], 'loop': loop, 'backing': backing, 'inode': st.st_ino,
                'size': st.st_size, 'allocated': st.st_blocks * 512, 'uuid': uuid}

    def provision_volume(self, role):
        require(role in ('data', 'evidence'), 'volume role')
        capacity = 8 * 1024**3 if role == 'data' else 512 * 1024**2
        backing, target = self.root + '/' + role + '.ext4', self.root + '/' + role
        # Parent creation/free-space checks are prerequisites, never reclaim disk.
        if not self.reserve_checked:
            require(self.host.free_bytes(self.root) >= 10 * 1024**3, 'host disk reserve')
            self.reserve_checked = True
        self.acquire('backing', backing, lambda: self.host.allocate_exclusive(backing, capacity),
                     lambda: self.entry_identity(backing))
        st = self.host.stat(backing)
        require(st.st_size == capacity and st.st_blocks * 512 >= capacity, 'sparse backing refused')
        loop = self.acquire('loop', backing,
                            lambda: self.host.command(['losetup', '--find', '--show', backing], 10),
                            lambda: self.find_loop(backing))['loop']
        self.host.command(['mkfs.ext4', '-F', '-m', '0', backing], 60)
        self.acquire('mount-directory', target, lambda: self.host.mkdir(target),
                     lambda: self.entry_identity(target))
        return self.acquire('mount', target,
                            lambda: self.host.command(['mount', '-t', 'ext4', '-o', 'nodev,nosuid', loop, target], 10),
                            lambda: self.volume_identity(role, loop))

    def find_loop(self, backing):
        rows = json.loads(self.host.command(['losetup', '--json', '--associated', backing,
                                            '--output', 'NAME,BACK-FILE'], 10))['loopdevices']
        require(len(rows) == 1 and rows[0]['back-file'] == backing, 'ambiguous loop association')
        require(re.fullmatch('/dev/loop[0-9]+', rows[0]['name']), 'loop identity')
        return {'loop': rows[0]['name'], 'backing': backing}

    def daemon_arguments(self, executable_hashes):
        """No shell, shared daemon or external containerd endpoint allowed."""
        for exe in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc'):
            require(executable_hashes.get(exe) == self.host.digest(exe), 'executable inventory mismatch: ' + exe)
        docker = ['/usr/bin/dockerd', '--host=unix://' + self.runtime + '/docker.sock',
                  '--pidfile=' + self.runtime + '/dockerd.pid', '--exec-root=' + self.runtime + '/exec',
                  '--data-root=' + self.root + '/data/docker', '--containerd=' + self.runtime + '/containerd.sock',
                  '--bridge=none', '--iptables=false', '--ip-forward=false', '--ip-masq=false',
                  '--exec-opt=native.cgroupdriver=systemd', '--cgroup-parent=' + self.manager.work_unit,
                  '--log-driver=none']
        containerd = ['/usr/bin/containerd', '--root', self.root + '/data/containerd',
                      '--state', self.runtime + '/containerd', '--address', self.runtime + '/containerd.sock']
        return {'dockerd': docker, 'containerd': containerd}

    def partial_cleanup(self, termination):
        """Recover only observed owned resources; unresolved intents stay flagged.

        Reverse-order independent attempts, no guessed paths or recursive delete.
        Backings/directories intentionally retained as diagnostic evidence.
        """
        results = []
        self.require_termination(termination)
        for item in reversed(self.resources):
            try:
                require(item['boot_id'] == self.host.text('/proc/sys/kernel/random/boot_id').strip(), 'boot changed')
                identity = item['identity']
                if item['kind'] == 'mount':
                    role = 'evidence' if identity['mount']['mount'].endswith('/evidence') else 'data'
                    if role == 'evidence':
                        results.append({'kind': 'mount', 'retained_for_export': True})
                        continue
                    self.detach_volume(role, identity, termination)
                elif item['kind'] == 'loop':
                    self.require_termination(termination)
                    mounts = mount_records(self.host.text('/proc/self/mountinfo'))
                    require(not any(m['source'] == identity['loop'] for m in mounts), 'loop still mounted')
                    # Absence is not identity mismatch, but must be observed.
                    rows = json.loads(self.host.command(['losetup', '--json', '--associated', identity['backing'],
                                                        '--output', 'NAME,BACK-FILE'], 10))['loopdevices']
                    if rows:
                        require(rows == [{'name': identity['loop'], 'back-file': identity['backing']}], 'partial loop mismatch')
                        self.require_termination(termination)
                        self.host.command(['losetup', '--detach', identity['loop']], 10)
                        rows = json.loads(self.host.command(['losetup', '--json', '--associated', identity['backing'],
                                                            '--output', 'NAME,BACK-FILE'], 10))['loopdevices']
                        require(not rows, 'partial loop detach unconfirmed')
                else:
                    results.append({'kind': item['kind'], 'retained': True})
                    continue
                results.append({'kind': item['kind'], 'detached': True})
            except BaseException as exc:
                results.append({'kind': item['kind'], 'error': repr(exc)})
        return results

    def detach_volume(self, role, identity, terminated):
        self.require_termination(terminated)
        require(self.volume_identity(role, identity['loop']) == identity, 'volume changed')
        self.require_termination(terminated)
        self.host.command(['umount', identity['mount']['mount']], 10)
        require(not any(m['mount'] == identity['mount']['mount'] for m in mount_records(self.host.text('/proc/self/mountinfo'))), 'mount survives')
        require(self.find_loop(identity['backing'])['loop'] == identity['loop'], 'loop changed')
        self.require_termination(terminated)
        self.host.command(['losetup', '--detach', identity['loop']], 10)
        rows = json.loads(self.host.command(['losetup', '--json', '--associated', identity['backing'], '--output', 'NAME,BACK-FILE'], 10))['loopdevices']
        require(rows == [], 'loop survives')
        return True

    def export_and_ack(self, exporter, manifest, evidence_identity, termination):
        # Receiver must independently read its exported bytes and return the
        # exact manifest digest. No archive deletion before acknowledgement.
        self.require_termination(termination)
        raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(raw).hexdigest()
        receipt = exporter.export(raw, manifest, maximum_bytes=512 * 1024**2, timeout=60)
        require(receipt.get('manifest_sha256') == digest and receipt.get('readback_verified') is True,
                'export acknowledgement absent or mismatched')
        self.journal.append('export-ack', receipt)
        self.journal.seal()
        # Export can take >10s. Obtain a new observed receipt, never extend an
        # old timestamp or trust exporter/owner success as termination evidence.
        fresh = self.terminate(termination['ownership'])
        self.require_termination(fresh)
        self.detach_volume('evidence', evidence_identity, fresh)
        # Retain backing bytes; deletion needs final exported teardown receipt.
        return {'export_acknowledged': True, 'evidence_detached': True, 'backing_retained': True,
                'manifest_sha256': digest, 'receipt': receipt}


class IndependentWatchdog:
    """Separate worker algorithm; no CLI or live spawn is enabled."""
    def __init__(self, lifecycle, identity, deadline, report):
        self.lifecycle, self.identity, self.deadline, self.report = lifecycle, deepcopy(identity), deadline, report

    def supervise(self, owner_alive, acknowledged):
        host = self.lifecycle.host
        try:
            while host.now() < self.deadline and owner_alive():
                if acknowledged():
                    # Never trust owner's success: independently observe cleanup.
                    break
                host.pause(.1)
            result = self.lifecycle.terminate(self.identity)
            # An empty read-back does not erase a preceding cleanup timeout.
            if result.get('errors'):
                result['termination_confirmed'] = False
        except BaseException as exc:
            result = {'termination_confirmed': False, 'errors': [repr(exc)]}
        try:
            self.report.append('watchdog-result', result)
            self.report.seal()
        except BaseException as exc:
            result['persistence_error'] = repr(exc)
            try:
                self.report.fallback(result)
            except BaseException:
                pass
        return result
