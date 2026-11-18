"""Mocked tests: no Docker/network/install/weights; fake clocks labeled explicitly."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import acceptance_driver as driver
import build_lifecycle as build

class FakeDocker:
    def __init__(self, timeout=False, cleanup_error=False, commit_error=False):
        self.timeout, self.cleanup_error, self.commit_error = timeout, cleanup_error, commit_error
        self.killed = False
        self.removed = False
        self.calls = []
    def __call__(self, args, timeout=10):
        self.calls.append((args, timeout))
        if args[0] == 'inspect':
            if '.HostConfig' in args[2]:
                return json.dumps({'Memory': 4 * 1024**3, 'MemorySwap': 4 * 1024**3, 'NanoCpus': 1000000000, 'PidsLimit': 16, 'NetworkMode': 'none', 'Privileged': False, 'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges']})
            running = self.timeout and not self.killed
            return json.dumps({'Running': running, 'Status': 'running' if running else 'exited', 'ExitCode': 0})
        if args[0] == 'kill':
            if self.cleanup_error:
                raise subprocess.TimeoutExpired(args, timeout)
            self.killed = True
        if args[0] == 'commit':
            if self.commit_error:
                raise subprocess.TimeoutExpired(args, timeout)
            return 'sha256:' + 'a' * 64
        if args[0] == 'rm':
            if self.cleanup_error:
                raise subprocess.TimeoutExpired(args, timeout)
            self.removed = True
        if args[0] == 'ps':
            return '' if self.removed else 'pv2-build-present'
        if args[0] == 'logs':
            return 'invented installation log'
        return ''

class Tests(unittest.TestCase):
    def test_approval_rejected_before_docker(self):
        with patch.object(subprocess, 'run', side_effect=AssertionError('must not execute')):
            with self.assertRaises(ValueError):
                driver.validate_approval({})

    def test_false_approval(self):
        approval = {'package_sha256': driver.PACKAGE_SHA, 'orchestration_hashes': {n: driver.sha(Path(driver.__file__).parent / n) for n in ('acceptance_driver.py', 'build_lifecycle.py')}, 'synthetic_execution_approved': False}
        with self.assertRaisesRegex(ValueError, 'approval missing'):
            driver.validate_approval(approval)

    def run_build(self, fake, expected=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            if expected:
                with self.assertRaises(expected):
                    build.build(path, 'pinned-base', path, run=fake, clock=iter([0, 301]).__next__, sleep=lambda _: None)
            else:
                build.build(path, 'pinned-base', path, run=fake)
            return json.loads((path / 'build-report.json').read_text()), (path / 'build.log').read_text()

    def test_timeout_kills_container_not_only_cli(self):
        fake = FakeDocker(timeout=True)
        report, log = self.run_build(fake, TimeoutError)
        self.assertTrue(fake.killed)
        self.assertTrue(report['cleanup']['termination_confirmed'])
        self.assertIn('300-second', report['primary_error'])
        self.assertIn('invented', log)

    def test_cleanup_timeout_preserves_original_failure(self):
        report, _ = self.run_build(FakeDocker(timeout=True, cleanup_error=True), TimeoutError)
        self.assertIn('300-second', report['primary_error'])
        self.assertFalse(report['cleanup']['termination_confirmed'])
        self.assertTrue(report['cleanup']['errors'])

    def test_commit_timeout_uncertainty_preserved(self):
        report, _ = self.run_build(FakeDocker(commit_error=True), subprocess.TimeoutExpired)
        self.assertFalse(report['snapshot_termination_confirmed'])
        self.assertTrue(report['cleanup']['termination_confirmed'])

    def test_success_resource_flags_and_bounded_calls(self):
        fake = FakeDocker()
        report, _ = self.run_build(fake)
        self.assertEqual(report['status'], 'complete')
        for flag in ('--memory=4g', '--memory-swap=4g', '--pids-limit=16', '--cpus=1', '--network=none', '--cap-drop=ALL'):
            self.assertIn(flag, fake.calls[0][0])
        self.assertTrue(all(0 < timeout <= 60 for _, timeout in fake.calls))

    def test_start_failure_preserves_report(self):
        fake = FakeDocker()
        def start_error(args, timeout=10):
            if args[0] == 'start':
                raise OSError('invented start failure')
            return fake(args, timeout)
        report, _ = self.run_build(start_error, OSError)
        self.assertIn('invented start failure', report['primary_error'])

    def test_cancellation_preserves_report(self):
        fake = FakeDocker()
        def cancel(args, timeout=10):
            if args[0] == 'start':
                raise InterruptedError('invented cancellation')
            return fake(args, timeout)
        report, _ = self.run_build(cancel, InterruptedError)
        self.assertIn('invented cancellation', report['primary_error'])

    def test_resource_mismatch_rejected_before_start(self):
        fake = FakeDocker()
        def mismatch(args, timeout=10):
            if args[0] == 'inspect' and '.HostConfig' in args[2]:
                return '{}'
            return fake(args, timeout)
        report, _ = self.run_build(mismatch, RuntimeError)
        self.assertIn('configuration mismatch', report['primary_error'])
        self.assertFalse(any(args[0] == 'start' for args, _ in fake.calls))

    def test_failed_kill_still_attempts_inspection_and_removal(self):
        fake = FakeDocker(timeout=True, cleanup_error=True)
        state = build.cleanup('pv2-build-test', fake)
        ops = [args[0] for args, _ in fake.calls]
        self.assertEqual(ops, ['inspect', 'kill', 'inspect', 'rm', 'ps'])
        self.assertFalse(state['removal_confirmed'])
        self.assertTrue(state['errors'])

    def test_build_write_failure_keeps_primary_error_in_stdout(self):
        import contextlib, io
        fake = FakeDocker(timeout=True)
        original = build.write
        def fail_report(path, obj):
            if path.name == 'build-report.json':
                raise OSError('invented full disk')
            return original(path, obj)
        with tempfile.TemporaryDirectory() as f, patch.object(build, 'write', side_effect=fail_report), contextlib.redirect_stdout(io.StringIO()) as log:
            with self.assertRaises(TimeoutError):
                build.build(Path(f), 'pinned-base', Path(f), run=fake, clock=iter([0, 301]).__next__, sleep=lambda _: None)
            report = json.loads(log.getvalue())['build_report']
            self.assertIn('300-second', report['primary_error'])
            self.assertIn('full disk', report['persistence_error'])

    def test_fallback_cleanup_failure_nonzero_and_final_hash_bound(self):
        import hashlib
        with tempfile.TemporaryDirectory() as f:
            reports = Path(f) / 'reports'; reports.mkdir()
            build.write(reports/'build-resource.json', {'container':'pv2-build-'+'b'*32})
            code = build.cleanup_record(Path(f), run=FakeDocker(timeout=True, cleanup_error=True))
            self.assertEqual(code, 1)
            ledger = json.loads((reports/'report-hashes.json').read_text())
            self.assertEqual(ledger['final-cleanup.json']['sha256'], hashlib.sha256((reports/'final-cleanup.json').read_bytes()).hexdigest())

    def test_prior_cleanup_claim_gets_fresh_absence_check(self):
        with tempfile.TemporaryDirectory() as f:
            reports = Path(f)/'reports'; reports.mkdir()
            build.write(reports/'build-resource.json', {'container':'pv2-build-'+'c'*32})
            build.write(reports/'build-report.json', {'cleanup': {'removal_confirmed':True}})
            code = build.cleanup_record(Path(f), run=FakeDocker())
            self.assertEqual(code, 1)

    def test_orchestration_persistence_failure_preserves_primary(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as f, patch.object(driver, 'save', side_effect=OSError('invented disk failure')), contextlib.redirect_stdout(io.StringIO()) as log:
            r = {'status':'partial','failure_traceback':'original failure'}
            self.assertFalse(driver.persist_report(Path(f), r))
            self.assertEqual(json.loads(log.getvalue())['failure_traceback'], 'original failure')

    def test_verifier_rejects_portable_path_escapes(self):
        from validate_review import safe_name
        for name in ('../x', '/x', 'C:/x', 'a\\b'):
            with self.assertRaises(ValueError):
                safe_name(name)

if __name__ == '__main__':
    unittest.main()
