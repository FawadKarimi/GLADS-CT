"""Real child interpreters and OS kills; synthetic records and mock steps only."""
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from test_performance_v2_lifecycle import Fixture
from consendxformer.performance_v2.durable_budget import PersistentBudget, AdmissionBlocked, ReconciliationRequired
from consendxformer.performance_v2.durable_lifecycle import DurableRun
from consendxformer.performance_v2.reuse import canonical, digest


def directory_options():
    # Linux uses the implementation's real directory barrier, including children.
    return {} if os.name == 'posix' else {'synthetic_directory_sync': lambda p: None}


def emit(path, value):
    tmp = path.with_suffix('.pending')
    with tmp.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def wait_forever():
    while True:
        time.sleep(0.1)


def worker(args):
    root = Path(args.root)
    plan = (root/'budget/plan.json').read_bytes()
    budget = PersistentBudget(root/'budget', expected_plan_sha256=digest(plan), **directory_options())
    fx = Fixture()
    def announce(event, **extra):
        emit(root/(args.slot + '-' + event + '.json'), {'pid': os.getpid(), 'event': event,
            'boundary': args.boundary, 'python': sys.version, 'executable': sys.executable,
            'source_lock_sha256': args.expected_lock_sha256, 'pilot_execution_authorized': False, **extra})
    if args.mode == 'compete':
        announce('ready')
        deadline = time.monotonic() + 60
        while not (root/'gate').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('parent gate missing')
            time.sleep(.01)
        try:
            ticket = budget.admit(fx.spec, decision_sha256=digest(args.slot.encode()))
        except AdmissionBlocked:
            announce('result', admitted=False)
            return 0
        announce('result', admitted=True, attempt=ticket.number)
        wait_forever()
    session = DurableRun(spec=fx.spec, inputs=fx.inputs, budget=budget, decision_sha256=digest(b'child start'))
    if args.mode == 'reserved':
        announce('boundary', attempt=session.attempt)
        wait_forever()
    if args.boundary.startswith('completion.') or args.boundary.startswith('mirror.'):
        for epoch in range(1, 21):
            fx.epoch(session, epoch)  # mock counters and invented state; no optimizer updates
    def boundary(label):
        if label == args.boundary:
            announce('boundary', attempt=session.attempt)
            wait_forever()  # parent kills the process; no Python exception/finally
    budget._fault = boundary
    if args.boundary.startswith('completion.') or args.boundary.startswith('mirror.'):
        session.finalize()
    else:
        session.terminate('failed', reason='invented failure in separate process')
    raise AssertionError('requested kill boundary was never reached')


class NativeProcessTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('PV2_NATIVE_ENTRY') or not os.environ.get('PV2_LOCK_SHA256'):
            raise RuntimeError('use the hash-bound portable acceptance entry point')
        self.temp = tempfile.TemporaryDirectory(prefix='process-case-', dir=os.environ['PV2_ACCEPTANCE_SCRATCH'])
        self.root = Path(self.temp.name)
        self.children = []
        self.fx = Fixture()
        self.other = copy.deepcopy(self.fx.spec)
        self.other['contract']['run_id'] = 'invented-separate-followup'
        self.plan = canonical({'schema': 'performance_v2_durable_plan_v1', 'scope': 'synthetic', 'limit': 2,
            'runs': {s['contract']['run_id']: digest(canonical(s)) for s in (self.fx.spec, self.other)},
            'pilot_execution_authorized': False})
        self.budget = PersistentBudget.create(self.root/'budget', plan_bytes=self.plan,
                                              expected_plan_sha256=digest(self.plan), **directory_options())

    def tearDown(self):
        for process, stream in self.children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
            stream.close()
        evidence = Path(os.environ['PV2_SYNTHETIC_EVIDENCE_DIRECTORY']) / self.id().split('.')[-1]
        shutil.copytree(self.root, evidence)
        self.temp.cleanup()

    def spawn(self, mode, boundary='reserved', slot='a'):
        command = [sys.executable, '-I', '-B', os.environ['PV2_NATIVE_ENTRY'], 'worker',
                   '--root', str(self.root), '--mode', mode, '--boundary', boundary, '--slot', slot,
                   '--expected-lock-sha256', os.environ['PV2_LOCK_SHA256']]
        log = (self.root/(slot + '-worker.log')).open('xb')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        self.children.append((process, log))
        return process

    def wait_event(self, process, event, slot='a'):
        path = self.root/(slot + '-' + event + '.json')
        deadline = time.monotonic() + 60
        while not path.exists():
            if process.poll() is not None:
                self.fail('child exited before boundary: ' + (self.root/(slot+'-worker.log')).read_text(errors='replace'))
            if time.monotonic() > deadline:
                self.fail('bounded wait expired at ' + event)
            time.sleep(.01)
        result = json.loads(path.read_bytes())
        self.assertEqual(result['pid'], process.pid)
        self.assertNotEqual(result['pid'], os.getpid())
        return result

    def kill(self, process):
        self.assertIsNone(process.poll())
        process.kill()
        code = process.wait(timeout=15)
        self.assertEqual(code, -signal.SIGKILL if os.name == 'posix' else 1)
        emit(self.root/('kill-%d.json' % process.pid), {'parent_pid': os.getpid(), 'child_pid': process.pid,
            'exit_code': code, 'method': 'SIGKILL' if os.name == 'posix' else 'TerminateProcess',
            'directory_fsync_mocked': os.name != 'posix', 'power_loss_tested': False})

    def fresh(self):
        return PersistentBudget(self.root/'budget', expected_plan_sha256=digest(self.plan), **directory_options())

    def reconcile(self, action):
        budget = self.fresh()
        observed = budget.inspect()
        return budget.reconcile(1, action=action, expected_snapshot_sha256=digest(canonical(observed)),
                                decision_sha256=digest(('explicit ' + action).encode()))

    def assert_budget_still_consumed(self):
        fresh = self.fresh()
        self.assertEqual(fresh.inspect()['consumed'], 1)
        with self.assertRaises(AdmissionBlocked):
            fresh.admit(self.fx.spec, decision_sha256=digest(b'no retry'))
        ticket = fresh.admit(self.other, decision_sha256=digest(b'new planned attempt'))
        try:
            self.assertEqual(ticket.number, 2)
        finally:
            ticket.close()
        observed = fresh.inspect()
        fresh.reconcile(2, action='interrupt_reserved', expected_snapshot_sha256=digest(canonical(observed)),
                        decision_sha256=digest(b'explicit close second attempt'))
        self.assertEqual(fresh.inspect()['consumed'], 2)
        with self.assertRaises(AdmissionBlocked):
            fresh.admit(self.other, decision_sha256=digest(b'no expansion'))
        emit(self.root/'final-budget.json', fresh.inspect())

    def test_separate_process_competing_admissions(self):
        a, b = self.spawn('compete', slot='a'), self.spawn('compete', slot='b')
        self.wait_event(a, 'ready', 'a'); self.wait_event(b, 'ready', 'b')
        (self.root/'gate').write_bytes(b'compete once')
        ra, rb = self.wait_event(a, 'result', 'a'), self.wait_event(b, 'result', 'b')
        self.assertEqual(int(ra['admitted']) + int(rb['admitted']), 1)
        winner, loser = (a, b) if ra['admitted'] else (b, a)
        self.assertEqual(loser.wait(timeout=15), 0)
        with self.assertRaises(AdmissionBlocked):
            self.reconcile('interrupt_reserved')
        self.kill(winner)
        self.reconcile('interrupt_reserved')
        self.assert_budget_still_consumed()

    def test_process_death_reserved_blocks_until_explicit_reconciliation(self):
        child = self.spawn('reserved')
        self.wait_event(child, 'boundary')
        self.kill(child)
        self.assertEqual(self.fresh().inspect()['attempts'][0]['state'], 'reserved')
        with self.assertRaises(ReconciliationRequired):
            self.fresh().admit(self.other, decision_sha256=digest(b'blocked after death'))
        receipt = self.reconcile('interrupt_reserved')
        self.assertFalse(receipt['epoch_boundary'])
        self.assert_budget_still_consumed()

    def terminal_case(self, boundary, expected, complete=False):
        child = self.spawn('terminal', boundary)
        self.wait_event(child, 'boundary')
        self.kill(child)
        fresh = self.fresh()
        observed = fresh.inspect()  # genuine recovery failures are test failures, never assumed success
        self.assertEqual(observed['attempts'][0]['state'], expected)
        emit(self.root/'after-kill-budget.json', observed)
        receipt = self.reconcile('interrupt_reserved' if expected == 'reserved' else 'export_committed')
        if complete:
            self.assertEqual(json.loads(self.fresh().committed_receipt(1)), receipt)
            self.assertEqual(receipt['status'], 'synthetic_complete')
        else:
            with self.assertRaises(AdmissionBlocked):
                self.fresh().committed_receipt(1)
        self.assert_budget_still_consumed()

    def test_kill_before_failed_terminal_commit(self):
        self.terminal_case('terminal.before_commit', 'reserved')

    def test_kill_after_failed_terminal_commit(self):
        self.terminal_case('terminal.after_commit', 'failed')

    def test_kill_before_complete_commit(self):
        self.terminal_case('completion.before_commit', 'reserved')

    def test_kill_after_complete_commit(self):
        self.terminal_case('completion.after_commit', 'synthetic_complete', complete=True)

    def test_kill_during_receipt_export_preserves_partial_bytes(self):
        self.terminal_case('mirror.after_link', 'synthetic_complete', complete=True)
        retained = self.fresh().inspect()['retained_evidence_hashes']
        self.assertTrue(retained)
        self.assertIn(digest(self.fresh().receipt_path(1).read_bytes()), retained.values())
