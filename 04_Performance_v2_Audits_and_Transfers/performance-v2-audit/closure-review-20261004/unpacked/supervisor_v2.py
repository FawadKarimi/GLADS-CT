"""Preparation-only supervisor state machine. No executable Linux backend exists.

Backend methods MUST honor their deadlines and bounded-read contracts. The fake
implementation is in tests only. This module contains no subprocess, socket,
mount, cgroup-write or daemon-start implementation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Protocol

GIB = 1024**3
CONTROL_LIMIT = 1024**2
CHUNK = 65536
POLICY = {'memory_max': 6 * GIB, 'swap_max': 0, 'cpu_max': [150000, 100000],
          'pids_max': 256, 'workload_seconds': 1200, 'operation_seconds': 60}

class Refusal(ValueError):
    pass

def require(value, message):
    if not value:
        raise Refusal(message)

def within(path, parent):
    if not isinstance(path, str) or not path.startswith('/') or '\\' in path or '\x00' in path:
        return False
    if any(p in ('.', '..') for p in path.split('/')):
        return False
    p, root = PurePosixPath(path), PurePosixPath(parent)
    return p == root or root in p.parents

class Backend(Protocol):
    is_fake: bool
    def now(self) -> float: ...
    def request(self, action: str, record: dict, timeout: float) -> dict: ...
    def read(self, handle: str, maximum: int, timeout: float):
        """Return <=maximum bytes, None for no data yet, or b'' for EOF.
        Merge stdout/stderr upstream; no unbounded buffers. Deadline MUST be
        enforced by a future backend, not merely checked after a blocking call.
        """
        ...

class Evidence(Protocol):
    def append(self, name: str, value: dict): ...
    def seal(self): ...
    def fallback(self, value: dict): ...

def validate_record(r):
    require(len(json.dumps(r).encode('utf-8')) <= 65536, 'ownership record bound')
    require(r.get('schema') in ('pv2_ownership_v2', 'pv2_ownership_v3'), 'ownership schema')
    token = r.get('token', '')
    require(re.fullmatch('[0-9a-f]{32}', token) is not None, 'ownership token')
    require(r.get('run_id', '').isdigit() and r.get('attempt', '').isdigit(), 'run/attempt')
    base, runtime = '/var/tmp/pv2-' + token, '/run/pv2-' + token
    require(r.get('root') == base and r.get('runtime_root') == runtime, 'private roots')
    if r['schema'] == 'pv2_ownership_v3':
        owner = r.get('systemd_owner', {})
        require(owner.get('workload_unit') == 'pv2work' + token + '.slice' and
                owner.get('guard_unit') == 'pv2guard' + token + '.slice', 'private systemd units')
        require(owner.get('workload_path') == r.get('cgroup') and within(r.get('cgroup'), '/'), 'reported workload path')
        require(owner.get('guard_path') and not within(owner['guard_path'], r['cgroup']) and
                not within(r['cgroup'], owner['guard_path']), 'independent systemd branches')
        require(owner.get('controller_path') == r.get('supervisor_cgroup'), 'reported controller path')
        require(r.get('container_cgroup_parent') == owner['workload_unit'], 'systemd container parent')
    else:
        require(r.get('cgroup') == '/pv2-' + token + '.slice', 'private cgroup')
    require(r.get('cgroup_inode', 0) > 0 and r.get('boot_id'), 'cgroup identity/boot')
    require(r.get('policy') == POLICY, 'policy mismatch')
    required_paths = {'docker_socket': runtime + '/docker.sock', 'containerd_socket': runtime + '/containerd.sock',
                      'pidfile': runtime + '/dockerd.pid', 'exec_root': runtime + '/exec',
                      'data_root': base + '/data/docker', 'containerd_root': base + '/data/containerd',
                      'containerd_state': runtime + '/containerd', 'download_root': base + '/data/downloads',
                      'scratch_root': base + '/data/tmp', 'daemon_log': base + '/evidence/daemon.log'}
    require(r.get('paths') == required_paths, 'shared or escaping daemon paths')
    require(r.get('runtime_tmpfs_bytes') == 64 * 1024**2, 'runtime tmpfs bound')
    require(r.get('service') == 'pv2-' + token + '.service', 'private service')
    require(r.get('network') == {'bridge': False, 'iptables': False, 'ip_forward': False}, 'daemon network effects')
    if r['schema'] == 'pv2_ownership_v2':
        require(r.get('container_cgroup_parent') == r['cgroup'], 'container parent')
    for role, capacity in (('data', 8 * GIB), ('evidence', 512 * 1024**2)):
        v = r.get('volumes', {}).get(role, {})
        require(v.get('mount') == base + '/' + role and v.get('backing') == base + '/' + role + '.ext4', 'volume path')
        require(re.fullmatch(r'/dev/loop[0-9]+', v.get('loop', '')) is not None, 'loop name')
        require(v.get('capacity_bytes') == capacity and v.get('allocated_bytes') >= capacity, 'sparse or expandable backing')
        require(v.get('filesystem') == 'ext4' and v.get('uuid') and v.get('device') and v.get('mount_id', 0) > 0, 'filesystem identity')
        require(v.get('inode_capacity', 0) > 0 and v.get('auto_expand') is False, 'inode/expansion policy')
    require(r['volumes']['data']['loop'] != r['volumes']['evidence']['loop'], 'shared loop')
    require(r.get('supervisor_cgroup') and not within(r['supervisor_cgroup'], r['cgroup']), 'supervisor inside workload')
    require(r.get('watchdog', {}).get('outside_workload') is True and r['watchdog'].get('lifetime_seconds') == 1200, 'independent watchdog contract')
    watchdog = r['watchdog']
    require(type(watchdog.get('pid')) is int and watchdog['pid'] > 1 and watchdog.get('start_ticks', 0) > 0,
            'watchdog process identity')
    require(watchdog.get('cgroup') and not within(watchdog['cgroup'], r['cgroup']), 'watchdog inside workload')
    require(re.fullmatch('[0-9a-f]{64}', watchdog.get('executable_sha256', '')) is not None, 'watchdog executable identity')
    identities = r.get('processes', [])
    require({p.get('role') for p in identities} >= {'dockerd', 'containerd'}, 'daemon identities missing')
    require(len({p.get('pid') for p in identities}) == len(identities), 'duplicate PID')
    for p in identities:
        require(type(p.get('pid')) is int and p['pid'] > 1 and type(p.get('start_ticks')) is int and p['start_ticks'] > 0, 'PID identity')
        require(re.fullmatch('[0-9a-f]{64}', p.get('executable_sha256', '')) is not None, 'executable identity')
        require(within(p.get('cgroup'), r['cgroup']), 'recorded daemon subtree escape')
    return deepcopy(r)

def verify_snapshot(r, s, active):
    # A future backend must derive this from proc/mountinfo/cgroup/filesystem
    # read-back, NOT echo requested configuration. Exact source provenance required.
    require(s.get('identity') == {'boot_id': r['boot_id'], 'cgroup_inode': r['cgroup_inode'], 'token': r['token']}, 'ownership changed')
    require(s.get('paths') == r['paths'] and s.get('volumes') == r['volumes'], 'socket/mount/loop mismatch')
    require(s.get('policy') == r['policy'] and s.get('all_storage_confined') is True, 'resource or spill-path mismatch')
    require(s.get('external_owned_processes') == [], 'daemon subtree escape')
    processes = s.get('processes')
    require(isinstance(processes, list), 'process census missing')
    saved = {p['pid']: p for p in r['processes']}
    for p in processes:
        require(within(p.get('cgroup'), r['cgroup']), 'process escaped')
        require(p.get('pid', 0) > 1 and p.get('start_ticks', 0) > 0, 'bad process census')
        if p['pid'] in saved:
            require(p == saved[p['pid']], 'PID reuse or executable change')
    if active:
        require(all(p in processes for p in r['processes']), 'daemon vanished before admission')
        require(s.get('watchdog_armed') is True, 'watchdog not armed')
        require(s.get('watchdog_identity') == r['watchdog'], 'watchdog identity changed')
        require(s.get('free_bytes', 0) > 0 and s.get('free_inodes', 0) > 0, 'ENOSPC/inode exhaustion')
    return s

def bounded_output(backend, handle, timeout):
    """One aggregate stdout/stderr allocation bound, including empty polls."""
    deadline = backend.now() + timeout
    result = bytearray()
    observed = 0
    while True:
        remaining = deadline - backend.now()
        require(remaining > 0, 'control stream timeout')
        block = backend.read(handle, CHUNK, min(remaining, 1.0))
        if block is None:
            continue
        require(isinstance(block, bytes) and len(block) <= CHUNK, 'backend violated bounded read')
        observed += len(block)
        if len(result) + len(block) > CONTROL_LIMIT:
            raise Refusal('control stream overflow at ' + str(observed) + ' observed bytes; cap=' + str(CONTROL_LIMIT))
        if not block:
            return bytes(result)
        result.extend(block)

class Supervisor:
    def __init__(self, backend, evidence):
        # Native activation requires a verified grant; no override flag exists.
        from activation_trust import valid_grant
        require(getattr(backend, 'is_fake', False) is True or valid_grant(getattr(backend, 'grant', None)), 'live backend requires verified capability')
        self.backend, self.evidence = backend, evidence
        self.report = {'schema': 'pv2_supervisor_report_v2', 'state': 'NEW', 'events': [],
                       'primary_error': None, 'cleanup_errors': [], 'persistence_errors': [],
                       'daemon_io_termination_verified': False, 'work_storage_detached': False,
                       'evidence_volume_retained': True, 'execution_approved': False,
                       'pilot_execution_authorized': False, 'stage_b': 'paused',
                       'backend': 'fake' if backend.is_fake else 'native',
                       'native_synthetic_scope': not backend.is_fake}
        self.record = None
        self.persisted = False

    def event(self, state):
        self.report['state'] = state
        self.report['events'].append(state)

    def call(self, action, timeout=10):
        require(self.record is not None and self.persisted, 'no durable ownership')
        start = self.backend.now()
        value = self.backend.request(action, deepcopy(self.record), timeout)
        require(self.backend.now() - start <= timeout, 'backend deadline violation: ' + action)
        return value

    def persist(self):
        # A report cannot certify its own subsequent write or ledger commit.
        # Saved reports are provisional. Only a hash-bound completion receipt
        # can establish evidence success; stale reports never authorize it.
        outcome = self.report['state']
        self.report['operation_cleanup_outcome'] = outcome
        self.report['evidence_finalized'] = False
        self.report['state'] = 'EVIDENCE_PENDING' if outcome != 'FAILED' else 'FAILED'

        def failed(channel, exc):
            self.report['persistence_errors'].append(channel + ': ' + repr(exc))
            self.report['state'] = 'FAILED'
            self.report['evidence_finalized'] = False

        try:
            self.evidence.append('supervisor-report', deepcopy(self.report))
            self.evidence.seal()
            # Stdout is provisional too; never print finalized success before
            # the last durable write. A broken stdout prevents completion.
            self.evidence.fallback(deepcopy(self.report))
            self.evidence.complete(outcome)
            self.report['evidence_finalized'] = True
            self.report['state'] = outcome
        except BaseException as exc:
            failed('evidence-finalization', exc)
            try:
                self.evidence.append('persistence-failure', deepcopy(self.report))
                self.evidence.seal()
            except BaseException as recovery:
                failed('failure-record', recovery)
            try:
                self.evidence.fallback(deepcopy(self.report))
            except BaseException as fallback_error:
                failed('fallback', fallback_error)

    def cleanup(self):
        self.event('CLEANUP')
        if not self.persisted:
            self.report['cleanup_errors'].append('no durable ownership record; no host mutation attempted')
            return
        # Every destructive attempt requires a fresh identity snapshot. Failure of
        # TERM does not skip KILL/read-back, but identity mismatch denies mutation.
        for action in ('close_admission', 'term_service', 'kill_subtree'):
            try:
                verify_snapshot(self.record, self.call('snapshot'), active=False)
                self.call(action)
            except BaseException as exc:
                self.report['cleanup_errors'].append(action + ': ' + repr(exc))
        try:
            s = verify_snapshot(self.record, self.call('wait_empty', 10), active=False)
            require(s.get('populated') is False and s['processes'] == [] and s.get('listeners') == [], 'workload/daemon still present')
            self.report['daemon_io_termination_verified'] = True
        except BaseException as exc:
            self.report['cleanup_errors'].append('termination_readback: ' + repr(exc))
        if self.report['daemon_io_termination_verified']:
            try:
                verify_snapshot(self.record, self.call('snapshot'), active=False)
                self.call('unmount_data', 10)
            except BaseException as exc:
                self.report['cleanup_errors'].append('unmount: ' + repr(exc))
            absent = False
            try:
                s = self.call('read_mount')
                require(s.get('record_identity') == self.record['volumes']['data'], 'mount identity read-back missing')
                require(s.get('mounted') is False, 'mount still busy/present')
                absent = True
            except BaseException as exc:
                self.report['cleanup_errors'].append('mount_readback: ' + repr(exc))
            if absent:
                try:
                    s = self.call('read_loop')
                    require(s.get('record_identity') == self.record['volumes']['data'], 'loop identity changed')
                    self.call('detach_data_loop', 10)
                except BaseException as exc:
                    self.report['cleanup_errors'].append('detach: ' + repr(exc))
                try:
                    s = self.call('read_loop')
                    require(s.get('record_identity') == self.record['volumes']['data'] and s.get('attached') is False, 'loop absence not verified')
                    self.report['work_storage_detached'] = True
                except BaseException as exc:
                    self.report['cleanup_errors'].append('loop_readback: ' + repr(exc))

    def run(self, record):
        try:
            self.record = validate_record(record)
            self.evidence.append('ownership', self.record)
            self.persisted = True
            self.event('RECORDED')
            verify_snapshot(self.record, self.call('snapshot'), active=True)
            self.event('ACTIVE')
            operation = self.call('begin_operation', 10)
            require(operation.get('handle') and operation.get('deadline_seconds') == 60, 'operation identity/deadline')
            output = bounded_output(self.backend, operation['handle'], 60)
            require(self.call('operation_status').get('exit_code') == 0, 'operation failed')
            self.report['control_output_sha256'] = hashlib.sha256(output).hexdigest()
            self.report['control_output_bytes'] = len(output)
        except BaseException as exc:
            self.report['primary_error'] = repr(exc)
        finally:
            self.cleanup()
            good = (not self.report['primary_error'] and not self.report['cleanup_errors']
                    and self.report['daemon_io_termination_verified'] and self.report['work_storage_detached'])
            self.event(('VERIFIED_FAKE' if self.backend.is_fake else 'VERIFIED_NATIVE') if good else 'FAILED')
            self.persist()
        if self.report['persistence_errors']:
            self.report['state'] = 'FAILED'
        return deepcopy(self.report)

    def independent_cleanup(self, record):
        """Fresh backend observations; does not consume previous success booleans."""
        try:
            self.record = validate_record(record)
            self.evidence.append('cleanup-ownership', self.record)
            self.persisted = True
            self.cleanup()
        except BaseException as exc:
            self.report['primary_error'] = repr(exc)
        finally:
            self.event('FAILED' if self.report['cleanup_errors'] or self.report['primary_error'] else ('CLEANUP_VERIFIED_FAKE' if self.backend.is_fake else 'CLEANUP_VERIFIED_NATIVE'))
            self.persist()
        if self.report['persistence_errors']:
            self.report['state'] = 'FAILED'
        return deepcopy(self.report)
