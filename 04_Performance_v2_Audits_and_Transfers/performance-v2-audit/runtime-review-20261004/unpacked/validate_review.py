"""Portable read-only verification. Required input paths; stdout only; no network."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import zipfile
import yaml

def check(value, message):
    if not value:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_name(name):
    p = PurePosixPath(name)
    check(bool(name) and not p.is_absolute() and '..' not in p.parts
          and '\\' not in name and not PureWindowsPath(name).drive,
          'unsafe ledger path')
    return p

def zip_members(z):
    names = z.namelist()
    check(len(names) == len(set(names)), 'duplicate archive member')
    for item in z.infolist():
        safe_name(item.filename)
        check((item.external_attr >> 16) & 0o170000 != 0o120000, 'archive symlink')

def verify(root, evidence_path, package, ledger_path):
    check(digest(evidence_path.read_bytes()) == '041f038ccb61a51ca8af5aca03e648e7b8bbda73a0dc0726647620b8a46dd86b', 'evidence archive identity')
    with zipfile.ZipFile(evidence_path) as z:
        zip_members(z)
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            expected, name = line.split(None, 1)
            check(digest(z.read(name.strip())) == expected, 'evidence hash')
        e = json.loads(z.read('evidence.json'))
        r = json.loads(z.read('host-readiness.json'))
        check(e['report'] == r and e['decoded_job_log'] == z.read('job.log').decode(), 'evidence copies')
        check(e['run']['id'] == int(r['run_id']) == 36412911139, 'run ID')
        check(e['run']['head_sha'] == r['workflow_commit'] == '12bb66279168c73b65a9b06f3112c6fa7a70950b', 'commit')
        check(e['run']['run_attempt'] == int(r['run_attempt']) == 1, 'attempt')
        check(e['run']['conclusion'] == e['jobs']['jobs'][0]['conclusion'] == 'success', 'conclusion')
        check(r['linux_acceptance_passed'] is r['security_clearance_passed'] is r['container_enforcement_verified'] is False, 'scope')
    check(digest(package.read_bytes()) == '2f97fdaf4dd7e3dd5a04255aa62a7dd7b603fcda0fd82ffcb026b3957407d45e', 'package identity')
    with zipfile.ZipFile(package) as z:
        zip_members(z)
        for name, record in json.loads(z.read('file-hashes.json')).items():
            data = z.read(name)
            check(len(data) == record['size_bytes'] and digest(data) == record['sha256'], 'package member')
    workflow = yaml.safe_load((root / 'linux-synthetic-acceptance.yml').read_text())
    check(set(workflow.get('on', workflow.get(True))) == {'workflow_dispatch'}, 'trigger')
    check(workflow['permissions'] == {'contents': 'read'}, 'permissions')
    check(workflow['jobs']['synthetic']['runs-on'] == 'ubuntu-24.04', 'runner')
    for p in root.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    ledger = json.loads(ledger_path.read_bytes())
    files = set()
    for p in root.rglob('*'):
        check(not p.is_symlink(), 'candidate symlink')
        if p.is_file() and p.resolve() != ledger_path.resolve():
            files.add(p.relative_to(root).as_posix())
    check(set(ledger) == files, 'candidate ledger coverage')
    for name, record in ledger.items():
        path = safe_name(name)
        data = (root / path).read_bytes()
        check(digest(data) == record['sha256'] and len(data) == record['size_bytes'], 'candidate hash: ' + name)
    approval = json.loads((root / 'synthetic-approval-TEMPLATE.json').read_bytes())
    for key in ('synthetic_execution_approved', 'native_build_execution_reviewed', 'production_loading_approved'):
        check(approval.get(key) is False, 'proposal must remain unapproved: ' + key)
    check(approval.get('review_reference') is None, 'proposal must not invent an approval')
    check(approval.get('package_sha256') == digest(package.read_bytes()), 'approval package binding')
    for name in ('acceptance_driver.py', 'build_lifecycle.py'):
        check(approval.get('orchestration_hashes', {}).get(name) == digest((root / name).read_bytes()), 'approval implementation binding')
    for p in root.iterdir():
        if p.suffix in ('.py', '.md', '.yml', '.json'):
            check(all(line.rstrip() == line for line in p.read_text().splitlines()), 'whitespace: ' + p.name)
    return {'status': 'passed', 'scope': 'inert_read_only_validation', 'evidence_sha256': digest(evidence_path.read_bytes()), 'docker_executed': False}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'evidence', 'package', 'ledger'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.evidence, a.package, a.ledger), indent=2))
