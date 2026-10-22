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
from contextlib import ExitStack
from io import StringIO
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('acceptance_entry', Path(__file__).resolve().parents[1]/'scripts/performance_v2_native_acceptance.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class MetadataBindingTests(unittest.TestCase):
    """Invented metadata only; no scientific imports or actual environment scan."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.base, self.venv = self.root/'base', self.root/'venv'
        for p in (self.base/'site', self.venv/'site', self.venv/'bin'):
            p.mkdir(parents=True)
        self.config = self.venv/'pyvenv.cfg'
        self.config.write_text('home = synthetic-base\ninclude-system-site-packages = false\n')
        self.stack.enter_context(patch.multiple(entry.sys, executable=str(self.venv/'bin/python'),
            prefix=str(self.base), base_prefix=str(self.base), base_exec_prefix=str(self.base)))
        self.stack.enter_context(patch.object(entry.site, 'getsitepackages',
            side_effect=lambda prefixes: [str(Path(p)/'site') for p in prefixes]))

    def distribution(self, root, descriptor, raw=b'Name: torch\r\nVersion: 2.3.0\r\n\r\n', member='METADATA'):
        path = root/'site'/descriptor
        path.mkdir()
        if raw is not None:
            (path/member).write_bytes(raw)
        return path

    def test_python310_no_site_binds_venv_and_base_separately(self):
        self.distribution(self.venv, 'torch-2.3.dist-info')
        self.distribution(self.base, 'torch-9.dist-info', b'Name: torch\nVersion: 9\n')
        observed = entry.dependency_metadata()
        self.assertTrue(observed['valid'])
        self.assertEqual(observed['active_distributions']['torch']['version'], '2.3.0')
        self.assertEqual(observed['scopes']['base_interpreter']['distributions'][0]['version'], '9')
        self.assertEqual(len(observed['duplicates']['torch']), 2)
        self.assertEqual(observed['active_scopes'], ['virtual_environment'])
        self.assertEqual(observed['pyvenv_config']['sha256'], entry.sha(self.config))

    def test_missing_environment_package_not_filled_from_base(self):
        self.distribution(self.base, 'torch-2.dist-info')
        observed = entry.dependency_metadata()
        self.assertNotIn('torch', observed['active_distributions'])
        self.assertIn('torch', observed['scopes']['virtual_environment']['missing_selected_names'])

    def test_same_scope_duplicate_is_rejected_without_overwriting(self):
        self.distribution(self.venv, 'torch-a.dist-info')
        self.distribution(self.venv, 'torch-b.dist-info', b'Name: Torch\nVersion: 9\n')
        observed = entry.dependency_metadata()
        self.assertFalse(observed['valid'])
        self.assertEqual(len(observed['duplicates']['torch']), 2)
        self.assertTrue(any('ambiguous active' in e['reason'] for e in observed['errors']))

    def test_normalized_names_identify_duplicates(self):
        for name, version in [('opencv_python', '1'), ('OpenCV.Python', '2')]:
            self.distribution(self.venv, name+'.dist-info', ('Name: '+name+'\nVersion: '+version+'\n').encode())
        observed = entry.dependency_metadata()
        self.assertFalse(observed['valid'])
        self.assertIn('opencv-python', observed['duplicates'])

    def test_missing_empty_and_malformed_metadata_rejected(self):
        for number, raw in enumerate((None, b'', b'Name: torch\n',
                                     b'Name: torch\nName: numpy\nVersion: 1\n')):
            self.distribution(self.venv, 'invalid%d.dist-info' % number, raw)
        observed = entry.dependency_metadata()
        self.assertFalse(observed['valid'])
        self.assertEqual(len(observed['errors']), 4)
        self.assertEqual(observed['active_distributions'], {})
        self.assertNotIn(hashlib.sha256(b'').hexdigest(), json.dumps(observed))

    def test_exact_bytes_and_standard_egg_metadata(self):
        raw = b'Name: packaging\r\nVersion: 20.3\r\n\r\n'
        self.distribution(self.base, 'packaging.egg-info', raw, 'PKG-INFO')
        observed = entry.dependency_metadata()
        record = observed['scopes']['base_interpreter']['distributions'][0]
        self.assertEqual(record['metadata_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertTrue(record['metadata_path'].endswith('PKG-INFO'))
        self.assertTrue(observed['valid'])

    def test_base_missing_metadata_is_not_an_empty_success(self):
        self.distribution(self.base, 'packaging.egg-info', None)
        observed = entry.dependency_metadata()
        self.assertFalse(observed['valid'])
        self.assertEqual(observed['errors'][0]['scope'], 'base_interpreter')

    def test_nonvirtual_interpreter_uses_base(self):
        self.config.unlink()
        self.distribution(self.base, 'torch.dist-info')
        observed = entry.dependency_metadata()
        self.assertTrue(observed['valid'])
        self.assertEqual(observed['active_scopes'], ['base_interpreter'])
        self.assertEqual(observed['active_distributions']['torch']['scope'], 'base_interpreter')

    def test_system_site_enabled_is_explicit_and_conflicts_rejected(self):
        self.config.write_text('include-system-site-packages = true\n')
        self.distribution(self.base, 'torch.dist-info')
        observed = entry.dependency_metadata()
        self.assertEqual(observed['active_scopes'], ['virtual_environment', 'base_interpreter'])
        self.assertTrue(observed['valid'])
        self.distribution(self.venv, 'torch.dist-info')
        self.assertFalse(entry.dependency_metadata()['valid'])

    def test_unknown_configuration_and_unbound_prefix_rejected(self):
        self.config.write_text('home = synthetic\n')
        self.assertFalse(entry.dependency_metadata()['valid'])
        self.config.unlink()
        with patch.object(entry.sys, 'prefix', str(self.venv)):
            self.assertFalse(entry.dependency_metadata()['valid'])

    def test_no_site_hooks_or_package_imports(self):
        (self.venv/'site/evil.pth').write_text('import forbidden_inventory_probe\n')
        (self.venv/'site/sitecustomize.py').write_text('raise AssertionError("executed")\n')
        self.distribution(self.venv, 'torch.dist-info')
        before = set(sys.modules)
        observed = entry.dependency_metadata()
        self.assertTrue(observed['valid'])
        self.assertFalse(observed['site_initialization_executed'])
        self.assertFalse(observed['import_resolution_verified'])
        self.assertFalse({'torch', 'torchvision', 'numpy', 'cv2', 'forbidden_inventory_probe'} & (set(sys.modules)-before))

    def test_invalid_inventory_prints_evidence_and_returns_two(self):
        self.distribution(self.venv, 'torch.dist-info', None)
        observed = {'dependency_metadata': entry.dependency_metadata()}
        with patch.object(entry, 'verify_lock'), patch.object(entry, 'inventory', return_value=observed), \
                patch.object(entry.sys, 'argv', ['entry', 'inventory', '--expected-lock-sha256', '0'*64]), \
                patch('sys.stdout', new_callable=StringIO) as output:
            self.assertEqual(entry.main(), 2)
            self.assertFalse(json.loads(output.getvalue())['dependency_metadata']['valid'])

    def test_invalid_metadata_blocks_before_scientific_imports(self):
        observed = {'dependency_metadata': {'valid': False, 'errors': [{'reason': 'invented missing metadata'}]}}
        destination = self.root/'retained-failure'
        with patch.object(entry, 'verify_lock'), patch.object(entry, 'inventory', return_value=observed), \
                patch.object(entry, 'torch_guards', side_effect=AssertionError('must not import')) as imports, \
                patch.object(entry.sys, 'argv', ['entry', 'test', '--scratch-parent', str(self.root),
                    '--output', str(destination), '--expected-lock-sha256', '0'*64]), \
                patch('sys.stdout', new_callable=StringIO):
            self.assertEqual(entry.main(), 2)
            imports.assert_not_called()
        result = json.loads((destination/'result.json').read_bytes())
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['pilot_execution_authorized'])
        self.assertFalse(result['native_acceptance'])
        self.assertTrue((destination/'inventory.json').is_file())


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
