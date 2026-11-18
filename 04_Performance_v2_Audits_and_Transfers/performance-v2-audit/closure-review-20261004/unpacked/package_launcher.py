"""Local review artifact generation; no native host operations."""
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
from launcher_binding import source_manifest

BASE = '737f7a8c814592378c27f7240aea5be2ab20affb9b4310dfe52ebe615f9c0df7'


def sha(raw): return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    if sha(a.base.read_bytes()) != BASE: raise ValueError('base hash')
    with zipfile.ZipFile(a.base) as z:
        old = json.loads(z.read('file-hashes.json'))
        if set(z.namelist()) != set(old) | {'file-hashes.json'}: raise ValueError('base coverage')
        for name, rec in old.items():
            raw = z.read(name)
            if sha(raw) != rec['sha256'] or len(raw) != rec['size_bytes']: raise ValueError(name)
            if (root/name).read_bytes() != raw: raise ValueError('parent changed: '+name)
        (root/'history'/'deadline-file-hashes.json').write_bytes(z.read('file-hashes.json'))
        write(root/'launcher-sources.json', source_manifest(root))
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
        (root/'launcher-tests.log').write_text(run.stdout+run.stderr, encoding='utf-8')
        if run.returncode: raise RuntimeError('tests failed')
        summary = re.search(r'Ran \d+ tests in [^\n]+', run.stderr).group()
        diff = []; changed = []
        for path in sorted(root.glob('*')):
            if path.suffix not in ('.py', '.md') or path.name in old: continue
            after = path.read_text(encoding='utf-8').splitlines(True)
            if any(x.rstrip('\r\n') != x.rstrip() for x in after): raise ValueError('whitespace')
            changed.append(path.name)
            diff.extend(difflib.unified_diff([], after, fromfile='/dev/null', tofile='b/'+path.name))
        for path in root.glob('*.py'): compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        (root/'launcher-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
        write(root/'LAUNCHER_VALIDATION.json', {
            'base_sha256': BASE, 'base_entries_verified': len(old), 'all_base_files_unchanged': True,
            'command': command, 'result': summary, 'exit_code': run.returncode,
            'runtime': sys.version, 'platform': platform.platform(), 'new_tests': 8,
            'changed_source_documents': changed, 'execution_approved': False,
            'linux_acceptance_executed': False, 'live_bridge_complete': False,
            'scope': 'mocked OS only; local temporary synthetic evidence',
            'initial_test_failure': 'New test imported IntegratedTests instead of IntegrationTests; corrected before final suite.',
            'warning': 'Retained acceptance_driver.py return-in-finally SyntaxWarning under Python 3.14',
            'stage_b': 'paused', 'pilot_execution_authorized': False})
    ledger = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink')
        if path.is_file() and path.name != 'file-hashes.json':
            raw = path.read_bytes(); ledger[path.relative_to(root).as_posix()] = {'sha256': sha(raw), 'size_bytes': len(raw)}
    write(root/'file-hashes.json', ledger)
    with zipfile.ZipFile(a.output, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in [*sorted(ledger), 'file-hashes.json']: z.write(root/name, name)
    with zipfile.ZipFile(a.output) as z:
        if set(z.namelist()) != set(ledger) | {'file-hashes.json'}: raise ValueError('output coverage')
        for name, rec in ledger.items():
            raw = z.read(name)
            if sha(raw) != rec['sha256'] or len(raw) != rec['size_bytes']: raise ValueError('output hash')
    print(json.dumps({'zip': str(a.output.resolve()), 'sha256': sha(a.output.read_bytes()),
                      'ledger_entries': len(ledger), 'tests': summary}, indent=2))


if __name__ == '__main__': main()
