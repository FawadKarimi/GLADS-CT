"""Internal worker algorithm; deliberately no executable main/CLI entry point."""
import os
from linux_lifecycle import IndependentWatchdog
from linux_integration import canonical
from supervisor_v2 import require, within


def arm(lifecycle):
    host = lifecycle.host
    from activation_trust import host_allowed
    require(host_allowed(host), 'verified watchdog capability required')
    me = host.process(os.getpid())
    work = lifecycle.manager.show(lifecycle.manager.work_unit)['ControlGroup']
    guard = lifecycle.manager.show(lifecycle.manager.guard_unit)['ControlGroup']
    require(within(me['cgroup'], guard) and not within(me['cgroup'], work), 'watchdog branch binding')
    # The service manager resource checks are required before worker admission;
    # identity is observed from /proc, not the spawning client's PID estimate.
    wd = {k: me[k] for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup')}
    wd.update(outside_workload=True, lifetime_seconds=1200)
    host.write_exclusive(lifecycle.root + '/evidence/watchdog-armed.json',
                         canonical({'token': lifecycle.token, 'boot_id': lifecycle.boot, 'watchdog': wd}))
    return wd


def worker(lifecycle, identity, owner, report, acknowledgment):
    host = lifecycle.host
    arm(lifecycle)
    def alive():
        try:
            current = host.process(owner['pid'])
        except FileNotFoundError:
            return False
        return all(current.get(k) == owner.get(k) for k in ('pid', 'start_ticks', 'executable_sha256', 'cgroup'))
    return IndependentWatchdog(lifecycle, identity, host.now() + 1200, report).supervise(alive, acknowledgment)
