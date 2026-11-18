import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from evidence_export import LocalEvidenceExport


class ExportTests(unittest.TestCase):
    def fixture(self, root):
        src = Path(root) / 'source'; src.mkdir(); (src / 'record.json').write_bytes(b'{}')
        m = {'files': {'record.json': {'size_bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}}}
        return src, m
    def test_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            src, m = self.fixture(t); dst = Path(t) / 'destination'
            raw = json.dumps(m, sort_keys=True, separators=(',', ':')).encode()
            e = LocalEvidenceExport(src, dst, directory_sync=lambda p: None)
            self.assertTrue(e.export(raw, m, 4096, 1)['readback_verified'])
            self.assertEqual((dst / 'record.json').read_bytes(), b'{}')
            with self.assertRaises(FileExistsError):
                e.export(raw, m, 4096, 1)
    def test_corruption_retains_partial_destination(self):
        with tempfile.TemporaryDirectory() as t:
            src, m = self.fixture(t); dst = Path(t) / 'destination'
            (src / 'record.json').write_bytes(b'XX')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                LocalEvidenceExport(src, dst, directory_sync=lambda p: None).export(json.dumps(m, sort_keys=True, separators=(',', ':')).encode(), m, 4096, 1)
            self.assertTrue(dst.exists())
            self.assertTrue(src.exists())
    def test_aggregate_bound_prevents_destination(self):
        with tempfile.TemporaryDirectory() as t:
            src, m = self.fixture(t); dst = Path(t) / 'destination'
            with self.assertRaises(ValueError):
                LocalEvidenceExport(src, dst, directory_sync=lambda p: None).export(json.dumps(m, sort_keys=True, separators=(',', ':')).encode(), m, 1, 1)
            self.assertFalse(dst.exists())
    def test_timeout_preserves_source(self):
        with tempfile.TemporaryDirectory() as t:
            src, m = self.fixture(t); dst = Path(t) / 'destination'
            ticks = iter([0, 2])
            with self.assertRaises(TimeoutError):
                LocalEvidenceExport(src, dst, lambda: next(ticks), directory_sync=lambda p: None).export(json.dumps(m, sort_keys=True, separators=(',', ':')).encode(), m, 4096, 1)
            self.assertEqual((src / 'record.json').read_bytes(), b'{}')
