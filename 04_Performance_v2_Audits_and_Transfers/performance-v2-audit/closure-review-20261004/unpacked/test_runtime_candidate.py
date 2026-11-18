"""Synthetic/static runtime-refresh tests; no network or native host operations."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import metadata_resolution as m
import native_launcher
from test_metadata_resolution import FakeTransport

ROOT = Path(__file__).resolve().parent


class RuntimeCandidateTests(unittest.TestCase):
    def test_new_minor_and_unapproved_state(self):
        data = json.loads((ROOT / 'runtime-candidate.json').read_text())
        self.assertEqual(m.CANDIDATES['python'], '3.12.15')
        self.assertEqual(data['python'], m.CANDIDATES['python'])
        self.assertEqual(data['wheel_interpreter_abi'], 'cp312-cp312')
        self.assertIsNone(data['native_inventory'])
        for key in ('artifact_bytes_verified', 'dependency_closure_verified', 'compatibility_verified',
                    'security_review_complete', 'native_supervisor_enabled', 'execution_approved',
                    'security_approved', 'dependencies_approved', 'pilot_execution_authorized'):
            self.assertIs(data[key], False)
    def test_cp310_wheel_cannot_be_reused(self):
        html = ('<a href="https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=' + 'a' * 64 + '">old</a>').encode()
        with self.assertRaisesRegex(ValueError, 'missing'):
            m.Resolver(FakeTransport([(html, {})])).cpu_index('torch', '2.13.0+cpu')
    def test_correct_abi_not_gpu(self):
        html = ('<a href="https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=' + 'a' * 64 + '">gpu</a>').encode()
        with self.assertRaisesRegex(ValueError, 'missing'):
            m.Resolver(FakeTransport([(html, {})])).cpu_index('torch', '2.13.0+cpu')
    def test_native_main_stops_before_bootstrap(self):
        with patch.object(native_launcher, 'trusted_bootstrap', side_effect=AssertionError('must not inspect host')) as bootstrap, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(native_launcher.main(), 2)
        bootstrap.assert_not_called()
        record = json.loads(out.getvalue())
        self.assertFalse(record['complete']); self.assertFalse(record['pilot_execution_authorized'])
        self.assertIn('disabled', record['error'])
    def test_no_package_download_endpoint_added(self):
        for url in ('https://files.pythonhosted.org/a.whl', 'https://registry-1.docker.io/v2/library/python/blobs/sha256:' + 'a' * 64):
            self.assertFalse(m.allowed(url))


if __name__ == '__main__': unittest.main()
