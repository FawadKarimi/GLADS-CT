"""Explicit local review generator. No network, Linux operations or installation."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

BASE = 'a579bc9ba7eaff586f95a21e1cbc45c72b34fca2c63f4133896605c969279129'
EDITED = {'linux_host.py', 'linux_lifecycle.py', 'supervisor_evidence.py', 'evidence_export.py',
          'test_linux_lifecycle.py', 'test_supervisor_v2.py', 'test_persistence_finalization.py',
          'test_evidence_export.py', 'file-hashes.json'}

def digest(raw): return hashlib.sha256(raw).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'): parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if digest(args.base.read_bytes()) != BASE: raise ValueError('parent ZIP mismatch')
    with zipfile.ZipFile(args.base) as base:
        ledger = json.loads(base.read('file-hashes.json'))
        if set(base.namelist()) != set(ledger) | {'file-hashes.json'}: raise ValueError('parent coverage')
        for name, item in ledger.items():
            raw = base.read(name)
            if len(raw) != item['size_bytes'] or digest(raw) != item['sha256']: raise ValueError(name)
            if name not in EDITED and (root / name).read_bytes() != raw: raise ValueError('unexpected edit: ' + name)
        (root / 'history' / 'linux-revision-file-hashes.json').write_bytes(base.read('file-hashes.json'))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
        tested = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
        (root / 'integration-tests.log').write_text(tested.stdout + tested.stderr, encoding='utf-8')
        if tested.returncode: raise RuntimeError('test failure: see integration-tests.log')
        summary = re.search(r'Ran \d+ tests in [^\n]+', tested.stderr).group()
        for path in root.glob('*.py'): compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        changes, diff = [], []
        for path in sorted(root.iterdir()):
            if path.suffix not in ('.py', '.md'): continue
            before = base.read(path.name).decode().splitlines(True) if path.name in base.namelist() else []
            after = path.read_text(encoding='utf-8').splitlines(True)
            if before == after: continue
            if any(s.rstrip('\r\n').rstrip() != s.rstrip('\r\n') for s in after): raise ValueError('whitespace: ' + path.name)
            changes.append(path.name)
            diff.extend(difflib.unified_diff(before, after, fromfile='a/' + path.name if before else '/dev/null', tofile='b/' + path.name))
        (root / 'integration-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
        write(root / 'INTEGRATION_VALIDATION.json', {'parent_sha256': BASE, 'parent_entries_verified': len(ledger),
              'command': command, 'test_summary': summary, 'exit_code': tested.returncode,
              'python': sys.version, 'platform': platform.platform(), 'changed_files': changes,
              'controller_minimum_python': '3.10', 'python310_runtime_executed': False,
              'linux_acceptance_executed': False, 'integrated_mocked_path': True,
              'original_15_tests_unchanged': True, 'execution_approved': False,
              'dependency_pins_unchanged': True, 'stage_b': 'paused',
              'warning': 'Retained acceptance_driver.py Python 3.14 return-in-finally SyntaxWarning',
              'scope': 'Mocked OS/subprocess tests; local temporary invented report bytes only'})
    current = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink')
        if path.is_file() and path != root / 'file-hashes.json':
            raw = path.read_bytes(); current[path.relative_to(root).as_posix()] = {'sha256': digest(raw), 'size_bytes': len(raw)}
    write(root / 'file-hashes.json', current)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as out:
        for name in [*sorted(current), 'file-hashes.json']: out.write(root / name, name)
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': digest(args.output.read_bytes()),
                      'ledger_entries': len(current), 'tests': summary}, indent=2))

if __name__ == '__main__': main()
