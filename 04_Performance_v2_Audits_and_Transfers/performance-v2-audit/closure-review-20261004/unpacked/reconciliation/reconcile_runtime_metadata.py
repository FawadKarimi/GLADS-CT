"""Offline archive reconciliation. No network, subprocess, imports from wheels or installation."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import zipfile

BASE_SHA = '808c6df3c515802f3c4b237010cd9f67abf1b21ad54a97539dac7659a3b0aadc'
META_SHA = 'b9c3397f798464b85560d0f8faf2763c2b428144e3d677304985b115720a8edb'

def ident(b):
    return {'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}

def archive(path, expected):
    raw = Path(path).read_bytes()
    if ident(raw)['sha256'] != expected:
        raise ValueError('archive hash mismatch')
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive members')
        out = {}
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n or ':' in n:
                raise ValueError('unsafe archive path')
            out[n] = z.read(n)
        return out

def verify_receipts(files, result):
    urls = {}
    for r in result['receipts']:
        name = r['evidence_file']
        raw = files[name]
        if ident(raw) != {'sha256': r['metadata_sha256'], 'size_bytes': r['size_bytes']}:
            raise ValueError('receipt hash/size mismatch')
        if name != r['metadata_sha256'] + '.metadata' or r['url'] in urls:
            raise ValueError('receipt identity mismatch')
        if r['artifact_bytes_downloaded'] is not False:
            raise ValueError('unexpected artifact acquisition')
        urls[r['url']] = raw
    return urls

def reconcile(files, module):
    result = json.loads(files['metadata-resolution.json'])
    if result['candidate_versions'] != module.CANDIDATES or result['errors'] or result['execution_approved'] is not False:
        raise ValueError('candidate/status mismatch')
    urls = verify_receipts(files, result)
    class Offline:
        def get(self, url, headers=None):
            return urls[url], {}  # Header absence is NOT proof of archived pagination headers.
    resolver = module.Resolver(Offline())
    r = result['resolved']
    derived = {n: resolver.pypi(n, module.CANDIDATES[n]) for n in ('pip', 'wheel', 'setuptools')}
    derived.update(torch_cpu=resolver.cpu_index('torch', '2.13.0+cpu'), vision_cpu=resolver.cpu_index('torchvision', '0.28.0+cpu'), vision_commit=resolver.vision_commit())
    for n in ('checkout', 'upload-artifact'):
        derived[n] = resolver.actions_inventory(n, r[n]['commit'])
    for n, value in derived.items():
        if value != r[n]:
            raise ValueError('derived metadata mismatch: ' + n)
    base = r['base']
    index = json.loads(files[base['index_metadata_sha256'] + '.metadata'])
    choices = [m for m in index['manifests'] if m.get('platform', {}).get('os') == 'linux' and m['platform'].get('architecture') == 'amd64' and m['platform'].get('variant') in (None, '')]
    if len(choices) != 1 or choices[0]['digest'] != base['platform_manifest_digest']:
        raise ValueError('OCI platform binding')
    raw = files[base['platform_manifest_digest'].split(':')[1] + '.metadata']
    if 'sha256:' + ident(raw)['sha256'] != choices[0]['digest'] or len(raw) != choices[0]['size']:
        raise ValueError('OCI manifest digest/size')
    manifest = json.loads(raw)
    if manifest['config'] != base['config_publisher_descriptor'] or manifest['layers'] != base['layer_publisher_descriptors']:
        raise ValueError('OCI descriptor mismatch')
    advisories = [x for x in result['receipts'] if '/advisories?' in x['url']]
    if len(advisories) != 5 or any(json.loads(files[x['evidence_file']]) != [] for x in advisories):
        raise ValueError('advisory response mismatch')
    for n in ('torch', 'torchvision', 'pip', 'wheel', 'setuptools'):
        if r['advisory-' + n] != {'entries': [], 'pagination_complete': True, 'security_approved': False}:
            raise ValueError('advisory derived mismatch')
    artifacts = []
    for n in ('pip', 'wheel', 'setuptools'):
        artifacts.extend(dict(x, kind='wheel') for x in r[n]['wheels'])
    for n in ('torch_cpu', 'vision_cpu'):
        artifacts.append(dict(r[n], size=None, kind='wheel', size_status='not supplied; acquisition blocked until exact bounded size is established'))
    for kind, desc in [('oci_config', manifest['config']), *[('oci_layer', x) for x in manifest['layers']]]:
        artifacts.append(dict(kind=kind, filename=desc['digest'].replace(':', '_'), size=desc['size'], publisher_reported_sha256=desc['digest'].split(':')[1], url='https://registry-1.docker.io/v2/library/python/blobs/'+desc['digest'], artifact_bytes_verified=False))
    return {'archive_sha256': META_SHA, 'base_archive_sha256': BASE_SHA, 'receipt_count': len(urls), 'unique_metadata_files': len({x['evidence_file'] for x in result['receipts']}), 'derived_metadata_reproduced': list(derived), 'oci': base, 'vision_commit': r['vision_commit'], 'actions': {n: {'commit': r[n]['commit'], 'candidate_count': len(r[n]['lockfile_candidates']), 'bundled_dist_equivalence_verified': False, 'runner_node_verified': False} for n in ('checkout','upload-artifact')}, 'advisory_receipts': advisories, 'advisory_limit': 'Five archived empty bodies verified; HTTP response headers absent, pagination_complete attributed to resolver, not independently header-verified. Not security clearance.', 'artifacts': artifacts, 'known_size_subtotal': sum(x['size'] for x in artifacts if x['size'] is not None), 'full_dependency_closure_verified': False, 'execution_approved': False, 'security_approved': False, 'pilot_execution_authorized': False}

def main():
    parent = Path(__file__).parent
    base = archive(parent/'performance-v2-supported-runtime-review-20261004-final.zip', BASE_SHA)
    ledger = json.loads(base['file-hashes.json'])
    for name, item in ledger.items():
        if ident(base[name]) != item:
            raise ValueError('base ledger mismatch: '+name)
    target = parent/'performance-v2-metadata-reconciliation-20261004'
    target.mkdir(exist_ok=False)
    for name, raw in base.items():
        dest = target/name
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.write_bytes(raw)
    meta = archive('D:/Downloads/metadata-run-01.zip', META_SHA)
    files = {n.removeprefix('metadata-run-01/'): raw for n, raw in meta.items()}
    spec = importlib.util.spec_from_file_location('reviewed_metadata', target/'metadata_resolution.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = reconcile(files, module)
    output = target/'reconciliation'; output.mkdir()
    for name, raw in files.items():
        (output/name).write_bytes(raw)
    (output/'reconciled.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    (output/'reconcile_runtime_metadata.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifacts','advisory_receipts','oci')}, indent=2))

if __name__ == '__main__':
    main()
