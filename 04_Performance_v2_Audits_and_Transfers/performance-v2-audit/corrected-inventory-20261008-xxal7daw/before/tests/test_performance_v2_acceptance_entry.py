"""Acceptance harness regressions; never loosen missing-dependency gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('acceptance_entry', Path(__file__).resolve().parents[1]/'scripts/performance_v2_native_acceptance.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class AcceptanceEntryTests(unittest.TestCase):
    def test_lock_digest_precedes_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'lock.json'
            path.write_bytes(b'not json')
            with patch.object(entry, 'LOCK', path), self.assertRaisesRegex(ValueError, 'digest'):
                entry.verify_lock('0'*64)

    def test_source_drift_and_added_test_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tests').mkdir()
            source = root/'tests/test_performance_v2_fake.py'
            source.write_bytes(b'# synthetic\n')
            lock = root/'lock.json'
            lock.write_text(json.dumps({'pilot_execution_authorized': False,
                                       'files': {source.relative_to(root).as_posix(): entry.sha(source)}}))
            expected = entry.sha(lock)
            with patch.object(entry, 'ROOT', root), patch.object(entry, 'LOCK', lock):
                entry.verify_lock(expected)
                source.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'drift'):
                    entry.verify_lock(expected)
                source.write_bytes(b'# synthetic\n')
                (root/'tests/test_performance_v2_extra.py').write_bytes(b'# unbound')
                with self.assertRaisesRegex(ValueError, 'unfrozen'):
                    entry.verify_lock(expected)

    def test_no_shell_or_arbitrary_python_subprocess(self):
        with self.assertRaisesRegex(RuntimeError, 'subprocess denied'):
            subprocess.run([sys.executable, '-c', 'raise AssertionError("must not execute")'], check=True)
        with self.assertRaisesRegex(RuntimeError, 'denied'):
            os.system('must-not-execute')

    def test_network_socket_rejected_before_contact(self):
        with self.assertRaisesRegex(RuntimeError, 'denied'):
            socket.socket()

    def test_inventory_is_metadata_only_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = entry.inventory(tmp)
            self.assertTrue(result['read_only'])
            self.assertFalse(result['weights_devices_and_research_data_inspected'])
            self.assertFalse(result['pilot_execution_authorized'])
            self.assertEqual(list(Path(tmp).iterdir()), [])
            path = Path(tmp)/'invented.json'
            entry.write_json(path, result)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                entry.write_json(path, {})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
