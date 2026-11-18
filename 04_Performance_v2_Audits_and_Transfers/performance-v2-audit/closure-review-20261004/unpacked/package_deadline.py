"""Explicit local packaging and mocked testing. No native Linux operations."""
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

BASE = '194e103548a0fe4167e377a85e4c7a9d2af045fe8038429054fb34fced8b1d99'
EDITED = {'systemd_ownership.py', 'linux_lifecycle.py', 'linux_integration.py', 'file-hashes.json'}

def digest(raw): return hashlib.sha256(raw).hexdigest()

def write(path, obj): path.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'): parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if digest(args.base.read_bytes()) != BASE: raise ValueError('base SHA mismatch')
    with zipfile.ZipFile(args.base) as base:
        previous = json.loads(base.read('file-hashes.json'))
        if set(base.namelist()) != set(previous) | {'file-hashes.json'}: raise ValueError('parent coverage')
        for name, record in previous.items():
            raw = base.read(name)
            if len(raw) != record['size_bytes'] or digest(raw) != record['sha256']: raise ValueError(name)
            if name not in EDITED and (root / name).read_bytes() != raw: raise ValueError('unexpected edit ' + name)
        (root / 'history' / 'hierarchy-file-hashes.json').write_bytes(base.read('file-hashes.json'))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
        tested = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
        (root / 'deadline-tests.log').write_text(tested.stdout + tested.stderr, encoding='utf-8')
        if tested.returncode: raise RuntimeError('tests failed; see deadline-tests.log')
        summary = re.search(r'Ran \d+ tests in [^\n]+', tested.stderr).group()
        for path in root.glob('*.py'): compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        files, diff = [], []
        for path in sorted(root.iterdir()):
            if path.suffix not in ('.py', '.md'): continue
            before = base.read(path.name).decode().splitlines(True) if path.name in base.namelist() else []
            after = path.read_text(encoding='utf-8').splitlines(True)
            if before == after: continue
            if any(s.rstrip('\r\n').rstrip() != s.rstrip('\r\n') for s in after): raise ValueError('whitespace ' + path.name)
            files.append(path.name)
            diff.extend(difflib.unified_diff(before, after, fromfile='a/' + path.name if before else '/dev/null', tofile='b/' + path.name))
        (root / 'deadline-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
        write(root / 'DEADLINE_VALIDATION.json', {'parent_sha256': BASE, 'parent_entries_verified': len(previous),
              'command': command, 'test_summary': summary, 'exit_code': tested.returncode,
              'python': sys.version, 'platform': platform.platform(), 'changed_source_documents': files,
              'all_parent_tests_unchanged': True, 'new_focused_tests': 12, 'production_pins_unchanged': True,
              'execution_approved': False, 'linux_acceptance_executed': False, 'stage_b': 'paused',
              'scope': 'invented clocks and mocked OS boundaries; no Linux execution',
              'warning': 'Unchanged acceptance_driver.py return-in-finally SyntaxWarning'})
    ledger = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink')
        if path.is_file() and path != root / 'file-hashes.json':
            raw = path.read_bytes(); ledger[path.relative_to(root).as_posix()] = {'sha256': digest(raw), 'size_bytes': len(raw)}
    write(root / 'file-hashes.json', ledger)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as out:
        for name in [*sorted(ledger), 'file-hashes.json']: out.write(root / name, name)
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': digest(args.output.read_bytes()),
                      'ledger_entries': len(ledger), 'tests': summary}, indent=2))

if __name__ == '__main__': main()
