"""Review packaging and mocked tests only; no activation or host operations."""
import argparse
import ast
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

BASE = 'a1f9f1568132414ab4cbbc6c97afcf9bcc0050d85f63cb90071b644c5f18ad8e'
EDITED = {'activation_trust.py', 'activation_bridge.py', 'test_activation_bridge.py', 'launcher-sources.json'}


def sha(raw): return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); root = a.root.resolve()
    if sha(a.base.read_bytes()) != BASE: raise ValueError('base SHA')
    with zipfile.ZipFile(a.base) as z:
        old = json.loads(z.read('file-hashes.json'))
        if set(z.namelist()) != set(old) | {'file-hashes.json'}: raise ValueError('parent coverage')
        for name, rec in old.items():
            raw = z.read(name)
            if len(raw) != rec['size_bytes'] or sha(raw) != rec['sha256']: raise ValueError('parent hash '+name)
            if name not in EDITED and (root/name).read_bytes() != raw: raise ValueError('unintended edit '+name)
        (root/'history'/'activation-file-hashes.json').write_bytes(z.read('file-hashes.json'))
        (root/'history'/'launcher-sources-before-storage.json').write_bytes(z.read('launcher-sources.json'))
        write(root/'launcher-sources.json', source_manifest(root))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        tests = []
        commands = [('targeted', [sys.executable, '-B', '-m', 'unittest', 'test_storage_admission', '-v']),
                    ('full', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v'])]
        for label, cmd in commands:
            run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=120)
            (root/('storage-'+label+'-tests.log')).write_text(run.stdout+run.stderr, encoding='utf-8')
            if run.returncode: raise RuntimeError(label+' tests failed')
            tests.append({'command': cmd, 'suite': label, 'exit_code': run.returncode,
                          'result': re.search(r'Ran \d+ tests in [^\n]+', run.stderr).group()})
        changed, patch = [], []
        for path in sorted(root.iterdir()):
            if path.suffix not in ('.py', '.md') and path.name != 'launcher-sources.json': continue
            before = z.read(path.name) if path.name in old else b''
            after = path.read_bytes()
            if before == after: continue
            text = after.decode()
            if any(s.rstrip() != s for s in text.splitlines()): raise ValueError('whitespace '+path.name)
            if path.suffix == '.py': ast.parse(text, feature_version=(3, 10)); compile(text, path.name, 'exec')
            changed.append(path.name)
            patch.extend(difflib.unified_diff(before.decode().splitlines(True), text.splitlines(True),
                                              fromfile='a/'+path.name if before else '/dev/null', tofile='b/'+path.name))
        (root/'storage-focused.patch').write_text(''.join(patch), encoding='utf-8', newline='\n')
        write(root/'STORAGE_VALIDATION.json', {'base_sha256': BASE, 'base_ledger_entries_verified': len(old),
              'tests': tests, 'python': sys.version, 'platform': platform.platform(), 'changed_files': changed,
              'python310_syntax': 'passed', 'execution_approved': False, 'security_approved': False,
              'dependencies_approved': False, 'pilot_execution_authorized': False, 'stage_b': 'paused',
              'native_execution': False, 'production_pins_unchanged': True,
              'boundaries': 'invented filesystem/allocations and mocked OS only; no Docker or privileged operations',
              'development_failure': 'Windows lacks os.statvfs; Linux-only mocked attribute now explicitly created',
              'historical_warning': 'unchanged acceptance_driver.py return-in-finally SyntaxWarning'})
    ledger = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink')
        if path.is_file() and path != root/'file-hashes.json':
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
                      'ledger_entries': len(ledger), 'tests': tests}, indent=2))


if __name__ == '__main__': main()
