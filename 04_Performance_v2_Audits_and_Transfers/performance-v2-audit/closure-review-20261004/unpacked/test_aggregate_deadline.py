"""Invented monotonic time and command results; no sleeps or subprocesses."""
import json
import unittest
from unittest.mock import patch

from systemd_ownership import SystemdOwnership
from linux_lifecycle import IndependentWatchdog
from test_linux_integration import Kernel
from linux_integration import Coordinator
from test_supervisor_v2 import FakeEvidence


class TimedHost:
    is_fake = True
    def __init__(self, names, costs=None, hostile=False, error=None):
        self.time = 0.; self.calls = []; self.names = names
        self.costs = iter(costs or []); self.hostile = hostile; self.error = error
    def now(self): return self.time
    def command(self, argv, timeout):
        started = self.time
        cost = next(self.costs, 1.)
        self.calls.append((list(argv), timeout, started))
        if not self.hostile and cost > timeout:
            self.time += timeout
            raise TimeoutError('invented cooperative command timeout')
        self.time += cost
        if self.error: raise self.error
        if argv[1] == 'list-units': return json.dumps([{'unit': n} for n in self.names]).encode()
        if argv[1] == 'show':
            unit = argv[2]
            path = '/owned.slice' if unit.endswith('.slice') else ('/owned.slice/work.service' if unit == 'work.service' else '/system.slice/' + unit)
            return f'Id={unit}\nActiveState=active\nControlGroup={path}\n'.encode()
        return b''


class AggregateTests(unittest.TestCase):
    def manager(self, h): return SystemdOwnership(h, 'a' * 32, '/sys/fs/cgroup')
    def kill(self, h, **kwargs):
        m = self.manager(h)
        return m.kill_members(m.work_unit, '/owned.slice', 'KILL', **kwargs)
    def no_signal(self, h): self.assertFalse(any(a[1] == 'kill' for a, _, _ in h.calls))

    def test_twenty_unrelated_services_cannot_return_after_22_seconds(self):
        h = TimedHost(['unrelated' + str(i) + '.service' for i in range(20)])
        with self.assertRaisesRegex(TimeoutError, 'aggregate deadline exhausted'): self.kill(h)
        self.assertEqual(h.time, 10)
        self.assertEqual(len(h.calls), 10)
        self.assertEqual([budget for _, budget, _ in h.calls], list(range(10, 0, -1)))
        self.no_signal(h)

    def test_exhaustion_before_workload_is_reached(self):
        h = TimedHost(['other' + str(i) + '.service' for i in range(8)] + ['work.service'])
        with self.assertRaises(TimeoutError): self.kill(h)
        self.assertFalse(any('work.service' in a for a, _, _ in h.calls))
        self.no_signal(h)

    def test_slow_inventory_stops_before_inspection(self):
        h = TimedHost(['work.service'], [2, 9])
        with self.assertRaises(TimeoutError): self.kill(h)
        self.assertEqual(h.time, 10)
        self.assertEqual([b for _, b, _ in h.calls], [10, 8])
        self.no_signal(h)

    def test_recheck_uses_remaining_not_fresh_timeout(self):
        h = TimedHost(['work.service'], [1, 1, 1, 8])
        with self.assertRaises(TimeoutError): self.kill(h)
        self.assertEqual([b for _, b, _ in h.calls], [10, 9, 8, 7])
        self.no_signal(h)

    def test_backend_overrun_at_each_boundary_is_failure(self):
        for costs in ([11], [1, 10], [1, 1, 9], [1, 1, 1, 8], [1, 1, 1, 1, 7]):
            h = TimedHost(['work.service'], costs, hostile=True)
            with self.subTest(costs=costs), self.assertRaisesRegex(TimeoutError, 'backend deadline overrun'):
                self.kill(h)
            self.assertEqual(len(h.calls), len(costs))

    def test_original_backend_exception_is_preserved_on_overrun(self):
        original = OSError('invented I/O error')
        h = TimedHost([], [11], hostile=True, error=original)
        with self.assertRaises(TimeoutError) as caught: self.kill(h)
        self.assertIs(caught.exception.__cause__, original)
        self.assertIn('I/O error', str(caught.exception))

    def test_success_and_exact_deadline_completion(self):
        h = TimedHost(['work.service'], [1, 1, 1, 1, 6])
        self.kill(h)
        self.assertEqual(h.time, 10)
        self.assertEqual([b for _, b, _ in h.calls], [10, 9, 8, 7, 6])

    def test_inherited_deadline_is_not_reset(self):
        h = TimedHost(['work.service'])
        with self.assertRaises(TimeoutError): self.kill(h, timeout=10, deadline=2)
        self.assertEqual([b for _, b, _ in h.calls], [2, 1])


class CallerTests(unittest.TestCase):
    def make(self):
        h = Kernel(); c = Coordinator(h, FakeEvidence(), 'a' * 32)
        inventory = {p: 'a' * 64 for p in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc')}
        b, r = c.provision(inventory, ['invented-watchdog'], ['invented-operation'], '123', '1')
        return h, c, b, r

    def test_backend_short_timeout_propagates_to_initial_identity(self):
        h, c, b, r = self.make(); original = h.command; calls = []
        def slow(argv, timeout):
            calls.append((list(argv), timeout)); h.time += min(1, timeout)
            return original(argv, timeout)
        with patch.object(h, 'command', side_effect=slow), self.assertRaises(TimeoutError):
            b.request('term_service', r, 2)
        self.assertEqual(h.time, 2)
        self.assertEqual([budget for _, budget in calls], [2, 1])
        self.assertTrue(b.cleanup_phase_errors)

    def test_backend_phases_do_not_nest_three_phase_cleanup(self):
        h, c, b, r = self.make()
        with patch.object(c.l, 'terminate', side_effect=AssertionError('nested cleanup')), patch.object(c.l.manager, 'kill_members', wraps=c.l.manager.kill_members) as signal:
            b.request('term_service', r, 10)
            b.request('kill_subtree', r, 10)
            b.request('wait_empty', r, 10)
        self.assertEqual([call.args[2] for call in signal.call_args_list], ['TERM', 'KILL'])

    def test_watchdog_keeps_timeout_even_when_independent_kill_succeeds(self):
        h, c, b, r = self.make(); e = FakeEvidence()
        original = c.l.manager.kill_members; seen = []
        def first_timeout(unit, path, signal, **kw):
            seen.append(signal)
            if signal == 'TERM':
                h.time = kw['deadline']
                raise TimeoutError('invented aggregate TERM timeout')
            return original(unit, path, signal, **kw)
        with patch.object(c.l.manager, 'kill_members', side_effect=first_timeout):
            result = IndependentWatchdog(c.l, c.l.cgroup_identity(), h.time, e).supervise(lambda: False, lambda: False)
        self.assertEqual(seen, ['TERM', 'KILL'])
        self.assertFalse(result['termination_confirmed'])
        self.assertIn('TERM timeout', str(result['errors']))
        self.assertFalse(e.ledger[-1][1]['termination_confirmed'])
        self.assertFalse(any(p['cgroup'].startswith(c.l.work_path + '/') for p in h.processes.values()))

    def test_wait_snapshot_commands_share_remaining_budget(self):
        h, c, b, r = self.make(); b.request('kill_subtree', r, 10)
        original = h.command; budgets = []
        def slow(argv, timeout):
            budgets.append(timeout); h.time += min(1, timeout)
            return original(argv, timeout)
        with patch.object(h, 'command', side_effect=slow), self.assertRaises(TimeoutError):
            b.request('wait_empty', r, 3)
        self.assertEqual(h.time, 3)
        self.assertEqual(budgets, [3, 2, 1])
