"""Manager-owned transient slices. Never mkdir/write/rmdir systemd cgroups."""
import json
import math
from supervisor_v2 import require, within


class SystemdOwnership:
    def __init__(self, host, token, mount):
        self.host, self.token, self.mount = host, token, mount
        # No hyphens: distinct top-level branches, not encoded descendants.
        self.work_unit = 'pv2work' + token + '.slice'
        self.guard_unit = 'pv2guard' + token + '.slice'

    def remaining(self, deadline, stage):
        remaining = deadline - self.host.now()
        if remaining <= 0:
            raise TimeoutError(f'{stage}: aggregate deadline exhausted (deadline={deadline}, now={self.host.now()})')
        return remaining

    def check_return(self, deadline, stage):
        if self.host.now() > deadline:
            raise TimeoutError(f'{stage}: backend deadline overrun (deadline={deadline}, now={self.host.now()})')

    def command_until(self, argv, deadline, stage):
        budget = self.remaining(deadline, stage)
        try:
            result = self.host.command(argv, budget)
        except BaseException as exc:
            if self.host.now() > deadline:
                raise TimeoutError(f'{stage}: backend deadline overrun; original error={exc!r}') from exc
            raise
        self.check_return(deadline, stage)
        return result

    def show(self, unit, active=True, deadline=None):
        deadline = self.host.now() + 10 if deadline is None else deadline
        text = self.command_until(['systemctl', 'show', unit,
                                   '--property=Id,ActiveState,ControlGroup'], deadline, 'show ' + unit).decode()
        row = dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
        require(row.get('Id') == unit, 'systemd unit identity')
        if active:
            require(row.get('ActiveState') == 'active', 'systemd unit inactive')
            path = row.get('ControlGroup', '')
            require(path.startswith('/') and path != '/' and '..' not in path.split('/'), 'invalid reported ControlGroup')
        return row

    def create(self, unit, memory, quota, pids):
        require(unit in (self.work_unit, self.guard_unit), 'unowned slice')
        # fail mode does not replace an existing unit. UUID token must be new.
        self.host.command(['busctl', 'call', 'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
                           'org.freedesktop.systemd1.Manager', 'StartTransientUnit', 'ssa(sv)a(sa(sv))',
                           unit, 'fail', '5', 'Description', 's', 'pv2 isolated ' + self.token,
                           'MemoryMax', 't', str(memory), 'MemorySwapMax', 't', '0',
                           'CPUQuotaPerSecUSec', 't', str(quota), 'TasksMax', 't', str(pids), '0'], 10)
        return self.show(unit)

    def kill_members(self, unit, expected_path, signal, timeout=10, deadline=None):
        require(type(timeout) in (float, int) and math.isfinite(timeout) and 0 < timeout <= 10, 'kill budget')
        deadline = self.host.now() + timeout if deadline is None else min(deadline, self.host.now() + timeout)
        require(self.show(unit, deadline=deadline)['ControlGroup'] == expected_path, 'slice path changed')
        require(signal in ('TERM', 'KILL'), 'signal')
        rows = json.loads(self.command_until(['systemctl', 'list-units', '--all', '--output=json',
                                             '--type=service,scope', '--no-pager'], deadline, 'unit inventory'))
        require(type(rows) is list and len(rows) <= 4096, 'unit inventory bound')
        for row in rows:
            name = row['unit']
            require(name.endswith(('.service', '.scope')) and '/' not in name, 'unit inventory name')
            observed = self.show(name, active=False, deadline=deadline)
            path = observed.get('ControlGroup', '')
            if path and within(path, expected_path):
                # Re-read immediately before signalling; never signal a name
                # that has moved outside the owned manager-reported branch.
                require(self.show(name, active=False, deadline=deadline).get('ControlGroup') == path, 'unit moved before signal')
                self.command_until(['systemctl', 'kill', '--kill-whom=all', '--signal=' + signal, name],
                                   deadline, 'signal ' + signal + ' ' + name)
        self.check_return(deadline, 'kill_members completion')

    def stop(self, unit, expected_path, inode):
        require(self.show(unit)['ControlGroup'] == expected_path, 'slice changed before stop')
        require(self.host.stat(self.mount + expected_path).st_ino == inode, 'slice inode changed before stop')
        self.host.command(['systemctl', 'stop', unit], 10)
        row = self.show(unit, active=False)
        require(row.get('ActiveState') == 'inactive' and not row.get('ControlGroup'), 'slice stop unconfirmed')
        try:
            self.host.stat(self.mount + expected_path)
        except FileNotFoundError:
            return
        raise ValueError('manager-owned cgroup still present after stop')
