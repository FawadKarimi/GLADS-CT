"""Invented metadata only; no HTTP requests or dependency downloads."""
import hashlib
import json
import unittest
from metadata_resolution import Resolver, allowed, MAX_METADATA

class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses); self.requests = []
    def get(self, url, headers=None):
        self.requests.append(url)
        value = self.responses.pop(0)
        return value if isinstance(value, tuple) else (json.dumps(value).encode(), {})

def pypi():
    return {'info': {'name': 'pip', 'version': '26.2.1', 'requires_python': '>=3.10', 'requires_dist': []},
            'urls': [{'packagetype': 'bdist_wheel', 'filename': 'pip-26.2.1-py3-none-any.whl', 'size': 100,
                      'url': 'https://files.pythonhosted.org/invented.whl', 'digests': {'sha256': 'a' * 64}, 'yanked': False}]}

class MetadataTests(unittest.TestCase):
    def test_binary_and_secret_urls_rejected(self):
        for url in ('https://files.pythonhosted.org/a.whl', 'https://registry-1.docker.io/v2/library/python/blobs/sha256:abc',
                    'http://pypi.org/pypi/pip/26.2.1/json', 'https://user:password@pypi.org/pypi/pip/26.2.1/json',
                    'https://download.pytorch.org/whl/cpu/torch/a.whl'):
            self.assertFalse(allowed(url))
            t = FakeTransport([])
            with self.assertRaises(ValueError):
                Resolver(t).get(url)
            self.assertFalse(t.requests)

    def test_publisher_hash_not_byte_verification(self):
        r = Resolver(FakeTransport([pypi()])); out = r.pypi('pip', '26.2.1')
        self.assertFalse(out['wheels'][0]['artifact_bytes_verified'])
        self.assertEqual(out['wheels'][0]['publisher_reported_sha256'], 'a' * 64)
        self.assertTrue(r.receipts[0]['metadata_bytes_independently_hashed'])
        self.assertEqual(len(r.documents), 1)

    def test_version_project_and_hash_mismatch(self):
        for mutation in ('version', 'name', 'sha'):
            data = pypi()
            if mutation == 'sha':
                data['urls'][0]['digests']['sha256'] = 'not-a-hash'
            else:
                data['info'][mutation] = 'wrong'
            with self.assertRaises(ValueError):
                Resolver(FakeTransport([data])).pypi('pip', '26.2.1')

    def test_cpu_index_exact_cp312_linux_selection(self):
        url = 'https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=' + 'b' * 64
        html = ('<a href="' + url + '">fixture</a>').encode()
        out = Resolver(FakeTransport([(html, {})])).cpu_index('torch', '2.13.0+cpu')
        self.assertFalse(out['artifact_bytes_verified'])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            Resolver(FakeTransport([(html + html, {})])).cpu_index('torch', '2.13.0+cpu')

    def test_oci_hashes_metadata_not_layers_and_hides_token(self):
        body = json.dumps({'config': {'digest': 'sha256:' + 'c' * 64}, 'layers': [{'digest': 'sha256:' + 'd' * 64}]}).encode()
        digest = 'sha256:' + hashlib.sha256(body).hexdigest()
        index = {'manifests': [{'platform': {'os': 'linux', 'architecture': 'amd64'}, 'digest': digest}]}
        r = Resolver(FakeTransport([{'token': 'secret-fixture'}, index, (body, {})])); out = r.oci()
        self.assertTrue(out['manifest_bytes_verified']); self.assertFalse(out['config_or_layer_bytes_verified'])
        self.assertNotIn('secret-fixture', str(r.documents) + str(r.receipts))
        self.assertTrue(all('/blobs/' not in url for url in r.transport.requests))

    def test_oci_corrupt_manifest_rejected(self):
        index = {'manifests': [{'platform': {'os': 'linux', 'architecture': 'amd64'}, 'digest': 'sha256:' + 'a' * 64}]}
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            Resolver(FakeTransport([{'token': 'fake'}, index, ({'wrong': 1})])).oci()

    def test_advisory_pages_and_empty_not_clearance(self):
        t = FakeTransport([(b'[]', {'Link': '<next>; rel="next"'}), (b'[]', {})])
        out = Resolver(t).advisory('pip', 'pip', '26.2.1')
        self.assertTrue(out['pagination_complete']); self.assertFalse(out['security_approved']); self.assertEqual(len(t.requests), 2)

    def test_pagination_bound_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pagination'):
            Resolver(FakeTransport([(b'[]', {'Link': 'rel="next"'})] * 10)).advisory('pip', 'pip', '26.2.1')

    def test_actions_lock_not_bundled_dist_proof(self):
        lock = {'packages': {'': {}, 'node_modules/example': {'version': '1.0.0', 'integrity': 'sha512-invented'}}}
        out = Resolver(FakeTransport([lock])).actions_inventory('checkout', 'a' * 40)
        self.assertEqual(out['lockfile_candidates'][0]['name'], 'example')
        self.assertFalse(out['bundled_dist_equivalence_verified'])

    def test_metadata_size_limit(self):
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            Resolver(FakeTransport([(b'x' * (MAX_METADATA + 1), {})])).get('https://pypi.org/pypi/pip/26.2.1/json')

    def test_annotated_tag_resolves_commit(self):
        r = Resolver(FakeTransport([{'object': {'type': 'tag', 'sha': 'a' * 40}}, {'object': {'type': 'commit', 'sha': 'b' * 40}}]))
        self.assertEqual(r.vision_commit()['commit'], 'b' * 40)

if __name__ == '__main__':
    unittest.main()
