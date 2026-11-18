"""Explicit artifact writer; separate from the read-only verifier. No network/Docker."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'baseline', 'package', 'inventory', 'lock', 'archive'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    r = a.root
    (r / 'reviewed').mkdir(exist_ok=True)
    shutil.copyfile(a.package, r / 'reviewed' / 'inspector.zip')
    shutil.copyfile(a.inventory, r / 'reviewed' / 'base-inventory.json')
    shutil.copyfile(a.lock, r / 'reviewed' / 'environment-lock.json')
    receipt = json.loads((r / 'security-sources-live' / 'receipt.json').read_bytes())
    inventory = json.loads(a.inventory.read_bytes())
    coverage = {'date': '2026-09-28', 'decision': 'not_cleared', 'native': [], 'python': [], 'actions': []}
    for item in inventory['os_packages']:
        row = dict(item)
        row['disposition'] = 'unresolved source mapping and complete scoped advisory/backport review'
        if row['name'] in ('openssl', 'libssl3'):
            row['known_affected_example'] = {'id': 'CVE-2025-9230', 'bookworm_fixed': '3.0.17-1~deb12u3'}
        if row['name'] == 'libc6':
            row['known_affected_example'] = {'id': 'CVE-2025-4802', 'bookworm_fixed': '2.36-9+deb12u11'}
        coverage['native'].append(row)
    for index, (name, version) in enumerate(receipt['python_queries']):
        f = r / 'security-sources-live' / f'pip-{index}.json'
        findings = json.loads(f.read_bytes()) if f.exists() else None
        coverage['python'].append({'name': name, 'version': version, 'evidence': str(f.relative_to(r)), 'advisories': None if findings is None else [{'id': x['ghsa_id'], 'summary': x['summary'], 'vulnerabilities': x['vulnerabilities']} for x in findings], 'disposition': 'bounded applicability assessment in SECURITY_REVIEW.md; not blanket approval'})
    for name in ('checkout', 'upload-artifact'):
        f = r / 'security-sources-live' / (name + '.json')
        coverage['actions'].append({'name': 'actions/' + name, 'advisories': json.loads(f.read_bytes()), 'bundled_npm_and_node_review': 'unresolved'})
    write(r / 'security-coverage.json', coverage)
    write(r / 'synthetic-approval-TEMPLATE.json', {'package_sha256': sha(a.package), 'orchestration_hashes': {n: sha(r / n) for n in ('acceptance_driver.py', 'build_lifecycle.py')}, 'synthetic_execution_approved': False, 'native_build_execution_reviewed': False, 'production_loading_approved': False, 'review_reference': None})
    # Focused authored-source diff only; full ledger covers retrieved/generated evidence.
    diff = []
    names = {x.name for x in a.baseline.iterdir() if x.suffix in ('.py', '.md', '.yml')} | {x.name for x in r.iterdir() if x.suffix in ('.py', '.md', '.yml')}
    for name in sorted(names):
        old, new = a.baseline / name, r / name
        diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [], new.read_text().splitlines(True) if new.exists() else [], fromfile='a/' + name if old.exists() else '/dev/null', tofile='b/' + name))
    (r / 'focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
    excluded = {'file-hashes.json'}
    ledger = {str(f.relative_to(r)).replace('\\', '/'): {'sha256': sha(f), 'size_bytes': f.stat().st_size} for f in sorted(r.rglob('*')) if f.is_file() and f.name not in excluded}
    write(r / 'file-hashes.json', ledger)
    with zipfile.ZipFile(a.archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(r.rglob('*')):
            if f.is_file():
                z.write(f, str(f.relative_to(r)))
    print(json.dumps({'archive': str(a.archive.resolve()), 'sha256': sha(a.archive), 'files': len(ledger) + 1}, indent=2))

if __name__ == '__main__':
    main()
