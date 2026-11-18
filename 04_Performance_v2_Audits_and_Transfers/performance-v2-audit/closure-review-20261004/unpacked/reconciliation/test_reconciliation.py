"""Offline integrity regressions; no network, installation or native operations."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import reconcile_runtime_metadata as check

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('resolver', ROOT.parent/'metadata_resolution.py')
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.files = {p.name:p.read_bytes() for p in ROOT.glob('*.metadata')}
        self.files['metadata-resolution.json'] = (ROOT/'metadata-resolution.json').read_bytes()

    def mutate_result(self, fn):
        d = json.loads(self.files['metadata-resolution.json']); fn(d)
        self.files['metadata-resolution.json'] = json.dumps(d).encode()

    def test_offline_reproduction_and_limits(self):
        r = check.reconcile(self.files, resolver)
        self.assertEqual(r['receipt_count'], 15)
        self.assertEqual(r['known_size_subtotal'], 48568708)
        self.assertEqual(sum(x['size'] is None for x in r['artifacts']), 2)
        self.assertFalse(r['execution_approved'])
        self.assertFalse(r['full_dependency_closure_verified'])

    def test_metadata_corruption_rejected(self):
        name = next(n for n in self.files if n.endswith('.metadata'))
        self.files[name] += b' '
        with self.assertRaises(ValueError): check.reconcile(self.files, resolver)

    def test_wrong_publisher_hash_rejected(self):
        self.mutate_result(lambda d:d['resolved']['torch_cpu'].update(publisher_reported_sha256='0'*64))
        with self.assertRaises(ValueError): check.reconcile(self.files, resolver)

    def test_oci_descriptor_substitution_rejected(self):
        self.mutate_result(lambda d:d['resolved']['base']['config_publisher_descriptor'].update(size=1))
        with self.assertRaises(ValueError): check.reconcile(self.files, resolver)

    def test_approval_substitution_rejected(self):
        self.mutate_result(lambda d:d.update(execution_approved=True))
        with self.assertRaises(ValueError): check.reconcile(self.files, resolver)

    def test_missing_receipt_bytes_rejected(self):
        del self.files[next(n for n in self.files if n.endswith('.metadata'))]
        with self.assertRaises(KeyError): check.reconcile(self.files, resolver)

    def test_deterministic(self):
        self.assertEqual(check.reconcile(self.files, resolver), check.reconcile(copy.deepcopy(self.files), resolver))

if __name__ == '__main__': unittest.main()
