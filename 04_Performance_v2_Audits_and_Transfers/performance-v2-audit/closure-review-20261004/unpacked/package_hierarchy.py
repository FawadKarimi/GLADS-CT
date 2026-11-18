"""Explicit local review generation; no native Linux actions or network."""
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

BASE = 'c56f961ce6103877607f5608ddb6535bba800dd34d972b3e20c885b298ef0610'
EDITED = {'linux_lifecycle.py', 'linux_integration.py', 'supervisor_v2.py', 'watchdog_worker.py',
          'test_linux_integration.py', 'test_linux_lifecycle.py', 'file-hashes.json'}

def digest(raw): return hashlib.sha256(raw).hexdigest()

def write(path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    if digest(a.base.read_bytes()) != BASE: raise ValueError('base mismatch')
    with zipfile.ZipFile(a.base) as base:
        old = json.loads(base.read('file-hashes.json'))
        if set(base.namelist()) != set(old) | {'file-hashes.json'}: raise ValueError('base coverage')
        for name, entry in old.items():
            raw = base.read(name)
            if digest(raw) != entry['sha256'] or len(raw) != entry['size_bytes']: raise ValueError(name)
            if name not in EDITED and (root / name).read_bytes() != raw: raise ValueError('unexpected edit ' + name)
        (root / 'history' / 'integrated-file-hashes.json').write_bytes(base.read('file-hashes.json'))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
        tested = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=120)
        (root / 'hierarchy-tests.log').write_text(tested.stdout + tested.stderr, encoding='utf-8')
        if tested.returncode: raise RuntimeError('tests failed; see hierarchy-tests.log')
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
        (root / 'hierarchy-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
        write(root / 'HIERARCHY_VALIDATION.json', {'parent_sha256': BASE, 'parent_entries_verified': len(old),
              'command': command, 'summary': summary, 'exit_code': tested.returncode, 'changed_source_documents': files,
              'python': sys.version, 'platform': platform.platform(), 'linux_acceptance': False,
              'execution_approved': False, 'stage_b': 'paused', 'production_pins_unchanged': True,
              'original_15_regressions_unchanged': True, 'scope': 'fake systemd and mount namespace only',
              'warning': 'Unchanged acceptance_driver.py return-in-finally SyntaxWarning under Python 3.14'})
    ledger = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink')
        if path.is_file() and path != root / 'file-hashes.json':
            raw = path.read_bytes(); ledger[path.relative_to(root).as_posix()] = {'sha256': digest(raw), 'size_bytes': len(raw)}
    write(root / 'file-hashes.json', ledger)
    with zipfile.ZipFile(a.output, 'x', zipfile.ZIP_DEFLATED) as out:
        for name in [*sorted(ledger), 'file-hashes.json']: out.write(root / name, name)
    print(json.dumps({'archive': str(a.output.resolve()), 'sha256': digest(a.output.read_bytes()),
                      'ledger_entries': len(ledger), 'tests': summary}, indent=2))

if __name__ == '__main__': main()
