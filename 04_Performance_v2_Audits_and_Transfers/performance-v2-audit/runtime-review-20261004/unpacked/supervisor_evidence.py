"""Local report-only sink. No host/backend implementation and no network."""
import hashlib
import json
import os
from pathlib import Path
import re

def sync_directory(path):
    """Linux directory entry durability; unsupported platforms fail closed."""
    if os.name != 'posix' or not hasattr(os, 'O_DIRECTORY'):
        raise OSError('Linux directory fsync required; use explicit fake in non-Linux tests')
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

class FileEvidence:
    def __init__(self, output, directory_sync=None):
        self.directory_sync = directory_sync or sync_directory
        self.durability_failed = False
        self.root = Path(output)
        self.root.mkdir(exist_ok=False)
        self.serial = 0
        self.sync(self.root.parent)
        self.sync(self.root)

    def sync(self, path):
        try:
            self.directory_sync(path)
        except BaseException:
            self.durability_failed = True
            raise

    def append(self, name, value):
        if re.fullmatch('[a-z-]+', name) is None:
            raise ValueError('invalid evidence name')
        raw = json.dumps(value, sort_keys=True, indent=2).encode('utf-8') + b'\n'
        if len(raw) > 128 * 1024:
            raise ValueError('report size limit')
        self.serial += 1
        target = self.root / f'{self.serial:04d}-{name}.json'
        with target.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        self.sync(self.root)

    def seal(self):
        entries = {}
        for p in sorted(self.root.iterdir()):
            if p.is_symlink():
                raise ValueError('evidence symlink')
            if p.name in ('report-hashes.json', '.ledger.tmp'):
                continue
            if not p.is_file():
                raise ValueError('unexpected report directory')
            data = p.read_bytes()
            entries[p.name] = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
        tmp = self.root / '.ledger.tmp'
        with tmp.open('xb') as f:
            f.write(json.dumps(entries, indent=2, sort_keys=True).encode() + b'\n')
            f.flush()
            os.fsync(f.fileno())
        self.sync(self.root)
        os.replace(tmp, self.root / 'report-hashes.json')
        self.sync(self.root)

    def fallback(self, report):
        raw = json.dumps(report, sort_keys=True)
        if len(raw.encode()) > 128 * 1024:
            raw = json.dumps({'state': 'FAILED', 'fallback_truncated': True, 'primary_error': str(report.get('primary_error'))[:4096]})
        print(raw, flush=True)

    def complete(self, outcome):
        if self.durability_failed:
            raise OSError('prior directory durability failure; completion refused')
        raw = (self.root / 'report-hashes.json').read_bytes()
        receipt = {'schema': 'pv2_evidence_completion_v1', 'outcome': outcome,
                   'ledger_sha256': hashlib.sha256(raw).hexdigest()}
        with (self.root / 'completion.json').open('xb') as f:
            f.write(json.dumps(receipt, sort_keys=True).encode() + b'\n')
            f.flush()
            os.fsync(f.fileno())
        self.sync(self.root)


def validate_complete(root):
    """Read-only, fail closed: provisional report alone is never completion."""
    root = Path(root)
    receipt = json.loads((root / 'completion.json').read_bytes())
    raw = (root / 'report-hashes.json').read_bytes()
    if receipt.get('schema') != 'pv2_evidence_completion_v1' or receipt.get('ledger_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('incomplete evidence: completion/ledger mismatch')
    ledger = json.loads(raw)
    actual = {p.name for p in root.iterdir()}
    if actual != set(ledger) | {'completion.json', 'report-hashes.json'}:
        raise ValueError('incomplete evidence: unbound records')
    reports = []
    for name, entry in ledger.items():
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('unsafe ledger path')
        p = root / name
        if p.is_symlink() or not p.is_file():
            raise ValueError('unsafe evidence member')
        data = p.read_bytes()
        if len(data) != entry['size_bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('evidence hash mismatch')
        if 'persistence-failure' in name:
            raise ValueError('persistence failed')
        if name.endswith('-supervisor-report.json'):
            reports.append(json.loads(data))
    if len(reports) != 1 or reports[0].get('persistence_errors'):
        raise ValueError('missing/ambiguous report')
    if receipt.get('outcome') not in ('VERIFIED_FAKE', 'CLEANUP_VERIFIED_FAKE', 'VERIFIED_NATIVE', 'CLEANUP_VERIFIED_NATIVE'):
        raise ValueError('operation or cleanup failed')
    if reports[0].get('operation_cleanup_outcome') != receipt['outcome'] or reports[0].get('state') != 'EVIDENCE_PENDING':
        raise ValueError('outcome mismatch')
    return receipt
