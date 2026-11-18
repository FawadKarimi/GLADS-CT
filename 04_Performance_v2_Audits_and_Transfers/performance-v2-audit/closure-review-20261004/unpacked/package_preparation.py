"""Explicit local review artifact generation; verification is separate and read-only."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

BASE_SHA = 'c374d9c61c29f2ee08fbdb95ab8ca1277429112b67f8dfe78046a9eef4b95caf'
PROPOSAL_SHA = 'd850279dbf6e13e5f774d174e6a3a6455e4073877cd3c6ed899fbef66b995d74'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'base', 'proposal', 'archive'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    if digest(a.base.read_bytes()) != BASE_SHA or digest(a.proposal.read_bytes()) != PROPOSAL_SHA:
        raise ValueError('base/proposal identity mismatch')
    root = a.root.resolve()
    base = zipfile.ZipFile(a.base)
    # Every original member except the regenerated root ledger must remain exact.
    for n in base.namelist():
        if n != 'file-hashes.json' and (root / n).read_bytes() != base.read(n):
            raise ValueError('original revised file changed: ' + n)
    (root / 'history' / 'revised-base-file-hashes.json').write_bytes(base.read('file-hashes.json'))
    (root / 'accepted-proposal.md').write_bytes(a.proposal.read_bytes())
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v'], capture_output=True, text=True)
    (root / 'preparation-tests.log').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    if tests.returncode:
        raise RuntimeError('tests failed; see preparation-tests.log; no package generated')
    for f in root.glob('*.py'):
        compile(f.read_text(encoding='utf-8'), str(f), 'exec')
    for f in root.iterdir():
        if f.suffix in ('.py', '.md', '.json', '.yml') and any(line.rstrip() != line for line in f.read_text(encoding='utf-8').splitlines()):
            raise ValueError('trailing whitespace: ' + f.name)
    import re
    summary = re.search(r'Ran \d+ tests in [^\n]+', tests.stderr)
    write(root / 'PREPARATION_VALIDATION.json', {'status': 'mocked_tests_passed', 'test_summary': summary.group() if summary else 'see log',
        'command': tests.args, 'python': sys.version, 'platform': platform.platform(), 'base_sha256': BASE_SHA,
        'proposal_sha256': PROPOSAL_SHA, 'base_implementation_members_unchanged': True, 'docker_executed': False,
        'metadata_network_executed': False, 'dependency_binaries_downloaded': False, 'approvals': False,
        'limitations': 'fake backend only; no Linux acceptance or security clearance'})
    # Review-focused delta excludes bulky unchanged historical advisory records.
    diff = []
    for f in sorted(root.iterdir()):
        if f.suffix not in ('.py', '.md') or f.name == 'accepted-proposal.md':
            continue
        before = base.read(f.name).decode('utf-8').splitlines(True) if f.name in base.namelist() else []
        after = f.read_text(encoding='utf-8').splitlines(True)
        if before != after:
            diff.extend(difflib.unified_diff(before, after, fromfile='a/' + f.name if before else '/dev/null', tofile='b/' + f.name))
    (root / 'supervisor-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
    ledger = {}
    for f in sorted(root.rglob('*')):
        if f.is_symlink():
            raise ValueError('candidate symlink')
        if f.is_file() and f != root / 'file-hashes.json':
            ledger[f.relative_to(root).as_posix()] = {'sha256': digest(f.read_bytes()), 'size_bytes': f.stat().st_size}
    write(root / 'file-hashes.json', ledger)
    with zipfile.ZipFile(a.archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in [*sorted(ledger), 'file-hashes.json']:
            z.write(root / name, name)
    print(json.dumps({'archive': str(a.archive.resolve()), 'sha256': digest(a.archive.read_bytes()), 'members': len(ledger) + 1,
                      'tests': summary.group() if summary else 'see log'}, indent=2))

if __name__ == '__main__':
    main()
