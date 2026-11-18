"""Metadata-only candidate resolution. Never fetches wheel, OCI blob or layer bytes.

HTTP is opt-in; unit tests inject a fake transport. Runtime approvals are not
produced here. Publisher hashes are never labeled independently byte-verified.
"""
import argparse
import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

MAX_METADATA = 8 * 1024**2
CANDIDATES = {'python': '3.12.15', 'torch': '2.13.0+cpu', 'torchvision': '0.28.0+cpu',
              'pip': '26.2.1', 'wheel': '0.46.2', 'setuptools': '84.0.0'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def allowed(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or p.username or p.password or p.port not in (None, 443) or p.fragment:
        return False
    if p.hostname == 'pypi.org':
        return bool(re.fullmatch(r'/pypi/[A-Za-z0-9_.-]+/[A-Za-z0-9+_.-]+/json', p.path)) and not p.query
    if p.hostname == 'download.pytorch.org':
        return p.path in ('/whl/cpu/torch/', '/whl/cpu/torchvision/') and not p.query
    if p.hostname == 'registry-1.docker.io':
        return bool(re.fullmatch(r'/v2/library/python/manifests/[A-Za-z0-9_:.-]+', p.path)) and not p.query
    if p.hostname == 'auth.docker.io':
        return p.path == '/token' and urllib.parse.parse_qs(p.query) == {'service': ['registry.docker.io'], 'scope': ['repository:library/python:pull']}
    if p.hostname == 'api.github.com':
        if p.path == '/advisories':
            q = urllib.parse.parse_qs(p.query)
            return set(q) <= {'ecosystem', 'affects', 'per_page', 'page'} and q.get('ecosystem', [''])[0] in ('pip', 'actions', 'npm')
        return bool(re.fullmatch(r'/repos/pytorch/vision/git/(ref/tags/v0\.28\.0|tags/[0-9a-f]{40})', p.path)) and not p.query
    if p.hostname == 'raw.githubusercontent.com':
        return bool(re.fullmatch(r'/actions/(checkout|upload-artifact)/[0-9a-f]{40}/(package-lock\.json|action\.yml)', p.path)) and not p.query
    return False

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('metadata redirect requires explicit review')

class HttpMetadata:
    def get(self, url, headers=None):
        require(allowed(url), 'non-metadata URL forbidden')
        request = urllib.request.Request(url, headers={'User-Agent': 'pv2-metadata-review', **(headers or {})})
        started = time.monotonic()
        with urllib.request.build_opener(NoRedirect).open(request, timeout=15) as response:
            raw = bytearray()
            while True:
                require(time.monotonic() - started < 30, 'metadata total deadline')
                b = response.read(min(65536, MAX_METADATA + 1 - len(raw)))
                if not b:
                    break
                raw.extend(b)
                require(len(raw) <= MAX_METADATA, 'metadata byte limit')
            return bytes(raw), dict(response.headers)

class Resolver:
    def __init__(self, transport):
        self.transport = transport
        self.receipts = []
        self.documents = {}

    def get(self, url, headers=None, private=False):
        require(allowed(url), 'binary/unapproved endpoint refused')
        data, response_headers = self.transport.get(url, headers)
        require(len(data) <= MAX_METADATA, 'metadata byte limit')
        data.decode('utf-8')
        if not private:
            digest = hashlib.sha256(data).hexdigest()
            self.documents[digest + '.metadata'] = data
            self.receipts.append({'url': url, 'metadata_sha256': digest,
                                  'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  'evidence_file': digest + '.metadata',
                                  'metadata_bytes_independently_hashed': True, 'size_bytes': len(data),
                                  'artifact_bytes_downloaded': False})
        return data, response_headers

    def pypi(self, name, version):
        raw, _ = self.get(f'https://pypi.org/pypi/{name}/{version}/json')
        data = json.loads(raw)
        require(data['info']['version'] == version, 'publisher version mismatch')
        normalize = lambda n: re.sub('[-_.]+', '-', n).lower()
        require(normalize(data['info']['name']) == normalize(name), 'publisher project mismatch')
        artifacts = []
        for item in data['urls']:
            if item['packagetype'] != 'bdist_wheel':
                continue
            require(item['filename'].endswith('.whl'), 'wheel filename invalid')
            require(urllib.parse.urlsplit(item['url']).hostname == 'files.pythonhosted.org', 'wheel publisher host')
            digest = item['digests']['sha256']
            require(re.fullmatch('[0-9a-f]{64}', digest), 'invalid publisher SHA')
            artifacts.append({'filename': item['filename'], 'url': item['url'], 'size': item['size'],
                              'publisher_reported_sha256': digest, 'artifact_bytes_verified': False,
                              'yanked': item.get('yanked'), 'requires_python': item.get('requires_python')})
        require(artifacts, 'no wheel metadata')
        return {'name': name, 'version': version, 'requires_dist': data['info'].get('requires_dist'),
                'requires_python': data['info'].get('requires_python'), 'wheels': artifacts,
                'dependency_closure_verified': False, 'installation_approved': False}

    def cpu_index(self, name, version):
        raw, _ = self.get('https://download.pytorch.org/whl/cpu/' + name + '/')
        class Links(HTMLParser):
            def __init__(self):
                super().__init__(); self.urls = []
            def handle_starttag(self, tag, attrs):
                if tag == 'a':
                    self.urls.extend(v for k, v in attrs if k == 'href')
        parser = Links(); parser.feed(raw.decode())
        matches = []
        for url in parser.urls:
            url = urllib.parse.urljoin('https://download.pytorch.org/whl/cpu/' + name + '/', url)
            p = urllib.parse.urlsplit(url)
            filename = urllib.parse.unquote(p.path.rsplit('/', 1)[-1])
            if filename.startswith(name + '-' + version + '-cp312-cp312-') and filename.endswith('x86_64.whl') and 'manylinux' in filename:
                require(p.scheme == 'https' and p.hostname in ('download.pytorch.org', 'download-r2.pytorch.org'), 'CPU publisher host')
                digest = urllib.parse.parse_qs(p.fragment).get('sha256', [None])[0]
                require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'CPU index hash missing')
                matches.append({'filename': filename, 'url': urllib.parse.urlunsplit(p._replace(fragment='')),
                                'publisher_reported_sha256': digest, 'artifact_bytes_verified': False})
        require(len(matches) == 1, 'CPU wheel missing or ambiguous for CPython312 Linux x64')
        return matches[0]

    def oci(self, tag='3.12.15-slim-bookworm'):
        raw, _ = self.get('https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull', private=True)
        token = json.loads(raw)['token']
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json'}
        raw, _ = self.get('https://registry-1.docker.io/v2/library/python/manifests/' + tag, headers)
        index_digest = hashlib.sha256(raw).hexdigest()
        manifests = json.loads(raw)['manifests']
        match = [m for m in manifests if m.get('platform', {}).get('os') == 'linux' and m['platform'].get('architecture') == 'amd64' and m['platform'].get('variant') in (None, '')]
        require(len(match) == 1, 'OCI amd64 identity missing/ambiguous')
        digest = match[0]['digest']
        require(re.fullmatch('sha256:[0-9a-f]{64}', digest), 'OCI digest invalid')
        raw, _ = self.get('https://registry-1.docker.io/v2/library/python/manifests/' + digest, headers)
        require('sha256:' + hashlib.sha256(raw).hexdigest() == digest, 'OCI manifest byte hash mismatch')
        m = json.loads(raw)
        return {'tag_at_resolution': tag, 'index_metadata_sha256': index_digest,
                'platform_manifest_digest': digest, 'manifest_bytes_verified': True,
                'config_publisher_descriptor': m['config'], 'layer_publisher_descriptors': m['layers'],
                'config_or_layer_bytes_verified': False, 'native_inventory_verified': False}

    def vision_commit(self):
        raw, _ = self.get('https://api.github.com/repos/pytorch/vision/git/ref/tags/v0.28.0')
        obj = json.loads(raw)['object']
        if obj['type'] == 'tag':
            raw, _ = self.get('https://api.github.com/repos/pytorch/vision/git/tags/' + obj['sha'])
            obj = json.loads(raw)['object']
        require(obj['type'] == 'commit' and re.fullmatch('[0-9a-f]{40}', obj['sha']), 'vision ref not immutable commit')
        return {'commit': obj['sha'], 'source_bytes_verified': False, 'topology_verified': False}

    def advisory(self, ecosystem, name, version):
        entries = []
        for page in range(1, 11):
            query = urllib.parse.urlencode({'ecosystem': ecosystem, 'affects': name + '@' + version, 'per_page': 100, 'page': page})
            raw, headers = self.get('https://api.github.com/advisories?' + query)
            data = json.loads(raw); require(isinstance(data, list), 'advisory error payload')
            entries.extend(data)
            link = next((v for k, v in headers.items() if k.lower() == 'link'), '')
            if 'rel="next"' not in link:
                return {'entries': entries, 'pagination_complete': True, 'security_approved': False}
        raise ValueError('advisory pagination bound; coverage incomplete')

    def actions_inventory(self, action, commit):
        require(action in ('checkout', 'upload-artifact') and re.fullmatch('[0-9a-f]{40}', commit), 'action identity')
        raw, _ = self.get(f'https://raw.githubusercontent.com/actions/{action}/{commit}/package-lock.json')
        lock = json.loads(raw)
        require(isinstance(lock.get('packages'), dict), 'unsupported npm lock schema')
        shipped_candidates = [{'path': p, 'name': p.rsplit('node_modules/', 1)[-1], 'version': v.get('version'), 'integrity_publisher_reported': v.get('integrity')}
                              for p, v in lock['packages'].items() if p and not v.get('dev', False)]
        return {'action': action, 'commit': commit, 'lockfile_candidates': shipped_candidates,
                'bundled_dist_equivalence_verified': False, 'runner_node_verified': False, 'security_approved': False}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fetch-metadata', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(a.fetch_metadata, 'metadata network access needs explicit flag')
    a.output.mkdir(exist_ok=False)
    r = Resolver(HttpMetadata())
    result = {'candidate_versions': CANDIDATES, 'resolved': {}, 'errors': {}, 'execution_approved': False}
    tasks = {'pip': lambda: r.pypi('pip', '26.2.1'), 'wheel': lambda: r.pypi('wheel', '0.46.2'),
             'setuptools': lambda: r.pypi('setuptools', '84.0.0'), 'base': r.oci, 'vision_commit': r.vision_commit,
             'torch_cpu': lambda: r.cpu_index('torch', '2.13.0+cpu'), 'vision_cpu': lambda: r.cpu_index('torchvision', '0.28.0+cpu'),
             'checkout': lambda: r.actions_inventory('checkout', '11bd71901bbe5b1630ceea73d27597364c9af683'),
             'upload-artifact': lambda: r.actions_inventory('upload-artifact', 'ea165f8d65b6e75b540449e92b4886f43607fa02')}
    for name, version in CANDIDATES.items():
        if name != 'python':
            tasks['advisory-' + name] = lambda n=name, v=version: r.advisory('pip', n, v.split('+')[0])
    for key, action in tasks.items():
        try:
            result['resolved'][key] = action()
        except Exception as exc:
            result['errors'][key] = repr(exc)
    result['receipts'] = r.receipts
    result['unresolved'] = ['wheel/image bytes not acquired or verified', 'native source mapping and backport coverage',
                            'Python/base bootstrap provenance', 'CPU wheel dependency metadata and closure',
                            'bundled Actions dist and Node runtime review', 'topology/seeded-head compatibility',
                            'all execution approvals remain false']
    result['status'] = 'partial' if result['errors'] else 'metadata_only_not_approved'
    for name, data in r.documents.items():
        with (a.output / name).open('xb') as f:
            f.write(data)
    with (a.output / 'metadata-resolution.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return int(bool(result['errors']))

if __name__ == '__main__':
    raise SystemExit(main())
