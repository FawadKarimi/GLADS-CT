"""Persistence matrix covers returned, saved, ledger and stdout channels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supervisor_v2 import Supervisor
from supervisor_evidence import FileEvidence, validate_complete
from test_supervisor_v2 import FakeBackend, FakeEvidence


class PersistenceTests(unittest.TestCase):
    def test_both_paths_and_both_review_failures(self):
        for method in ('run', 'independent_cleanup'):
            for fail in ('supervisor-report', 'seal'):
                with self.subTest(method=method, fail=fail):
                    e, b = FakeEvidence(fail), FakeBackend()
                    result = getattr(Supervisor(b, e), method)(b.r)
                    self.assertEqual(result['state'], 'FAILED')
                    self.assertEqual(e.stdout['state'], 'FAILED')
                    self.assertFalse(result['evidence_finalized'])
                    self.assertTrue(e.stdout['persistence_errors'])
                    self.assertFalse(hasattr(e, 'completion'))
                    for _, row in e.rows:
                        self.assertNotIn(row.get('state'), ('VERIFIED_FAKE', 'CLEANUP_VERIFIED_FAKE'))
                    if e.ledger:
                        self.assertEqual(e.ledger[-1][1]['state'], 'FAILED')

    def test_success_saved_provisional_receipt_final(self):
        for method in ('run', 'independent_cleanup'):
            b, e = FakeBackend(), FakeEvidence()
            result = getattr(Supervisor(b, e), method)(b.r)
            self.assertTrue(result['evidence_finalized'])
            self.assertEqual(e.rows[-1][1]['state'], 'EVIDENCE_PENDING')
            self.assertEqual(e.completion['outcome'], result['state'])
            self.assertEqual(e.stdout['state'], 'EVIDENCE_PENDING')
            self.assertFalse(e.stdout['evidence_finalized'])

    def test_original_error_survives_all_channels_failed(self):
        b = FakeBackend(); b.streaming_timeout = True
        e = FakeEvidence()
        with patch.object(e, 'append', side_effect=OSError('disk')), patch.object(e, 'fallback', side_effect=OSError('stdout')):
            result = Supervisor(b, e).run(b.r)
        self.assertIn('disk', result['primary_error'])
        self.assertEqual(result['state'], 'FAILED')
        self.assertTrue(result['persistence_errors'])
        self.assertIsNone(e.ledger)
        self.assertIsNone(e.stdout)

    def test_operational_error_not_replaced_by_ledger_error(self):
        b = FakeBackend(); b.streaming_timeout = True
        e = FakeEvidence('seal')
        result = Supervisor(b, e).run(b.r)
        self.assertIn('timeout', result['primary_error'])
        self.assertEqual(e.stdout['primary_error'], result['primary_error'])

    def test_disk_consumer_accepts_only_complete_binding(self):
        with tempfile.TemporaryDirectory() as t:
            e = FileEvidence(Path(t) / 'reports', directory_sync=lambda p: None); b = FakeBackend()
            with patch('builtins.print'):
                result = Supervisor(b, e).run(b.r)
            self.assertEqual(validate_complete(e.root)['outcome'], result['state'])
            (e.root / 'completion.json').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_complete(e.root)

    def test_disk_seal_failure_leaves_no_success_claim(self):
        for method in ('run', 'independent_cleanup'):
            with tempfile.TemporaryDirectory() as t:
                e = FileEvidence(Path(t) / 'reports', directory_sync=lambda p: None); b = FakeBackend()
                with patch.object(e, 'seal', side_effect=OSError('ledger')), patch('builtins.print') as stdout:
                    result = getattr(Supervisor(b, e), method)(b.r)
                self.assertEqual(json.loads(stdout.call_args.args[0])['state'], 'FAILED')
                self.assertEqual(result['state'], 'FAILED')
                with self.assertRaises(FileNotFoundError):
                    validate_complete(e.root)
                for p in e.root.glob('*supervisor-report.json'):
                    self.assertEqual(json.loads(p.read_text())['state'], 'EVIDENCE_PENDING')

    def test_fallback_failure_invalidates_old_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            e = FileEvidence(Path(t) / 'reports', directory_sync=lambda p: None); b = FakeBackend()
            with patch.object(e, 'fallback', side_effect=OSError('stdout')):
                result = Supervisor(b, e).run(b.r)
            self.assertEqual(result['state'], 'FAILED')
            with self.assertRaises((ValueError, FileNotFoundError)):
                validate_complete(e.root)

    def test_complete_failure_saved_failure_and_stdout(self):
        b, e = FakeBackend(), FakeEvidence()
        with patch.object(e, 'complete', side_effect=OSError('receipt')):
            result = Supervisor(b, e).run(b.r)
        self.assertEqual(result['state'], 'FAILED')
        self.assertEqual(e.ledger[-1][1]['state'], 'FAILED')
        self.assertEqual(e.stdout['state'], 'FAILED')
