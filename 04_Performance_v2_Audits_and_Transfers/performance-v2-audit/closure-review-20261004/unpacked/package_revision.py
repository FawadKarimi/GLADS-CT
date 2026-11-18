"""Explicit review generation, no host operations, network or dependency install."""
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

BASE = '28872953c4abbadaeaef4c6c40ebcc713817c266d735136a27da725ab59b591b'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if sha(args.base.read_bytes()) != BASE:
        raise ValueError('parent archive mismatch')
    root = args.root.resolve()
    with zipfile.ZipFile(args.base) as base:
        old = json.loads(base.read('file-hashes.json'))
        if set(base.namelist()) != set(old) | {'file-hashes.json'}:
            raise ValueError('parent member coverage')
        for name, item in old.items():
            raw = base.read(name)
            if sha(raw) != item['sha256'] or len(raw) != item['size_bytes']:
                raise ValueError('parent ledger mismatch: ' + name)
        changed = {'supervisor_v2.py', 'supervisor_evidence.py', 'test_supervisor_v2.py', 'file-hashes.json'}
        for name in base.namelist():
            if name not in changed and (root / name).read_bytes() != base.read(name):
                raise ValueError('unexpected historical change: ' + name)
        (root / 'history' / 'supervisor-parent-file-hashes.json').write_bytes(base.read('file-hashes.json'))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
        tested = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=120)
        (root / 'revision-tests.log').write_text(tested.stdout + tested.stderr, encoding='utf-8')
        if tested.returncode:
            raise RuntimeError('test failure; no package')
        summary = re.search(r'Ran \d+ tests in [^\n]+', tested.stderr).group()
        for p in root.glob('*.py'):
            compile(p.read_text(encoding='utf-8'), str(p), 'exec')
        diff, files = [], []
        for p in sorted(root.iterdir()):
            if p.suffix not in ('.py', '.md'):
                continue
            before = base.read(p.name).decode().splitlines(True) if p.name in base.namelist() else []
            after = p.read_text(encoding='utf-8').splitlines(True)
            if before != after:
                if any(x.rstrip('\r\n').rstrip() != x.rstrip('\r\n') for x in after):
                    raise ValueError('whitespace: ' + p.name)
                files.append(p.name)
                diff.extend(difflib.unified_diff(before, after, fromfile='a/' + p.name if before else '/dev/null', tofile='b/' + p.name))
        (root / 'revision-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
        write(root / 'REVISION_VALIDATION.json', {'status': 'mocked_tests_passed_partial_implementation',
              'summary': summary, 'command': command, 'python': sys.version, 'platform': platform.platform(),
              'parent_sha256': BASE, 'parent_ledger_entries_verified': len(old), 'changed_source_documents': files,
              'original_15_tests_unchanged': True, 'production_pins_unchanged': True,
              'linux_acceptance': False, 'complete_linux_integration': False,
              'execution_approved': False, 'stage_b': 'paused', 'no_host_operations': True})
    ledger = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('symlink')
        if p.is_file() and p.name != 'file-hashes.json':
            raw = p.read_bytes()
            ledger[p.relative_to(root).as_posix()] = {'sha256': sha(raw), 'size_bytes': len(raw)}
    write(root / 'file-hashes.json', ledger)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as out:
        for name in [*sorted(ledger), 'file-hashes.json']:
            out.write(root / name, name)
    print(json.dumps({'archive': str(args.output.resolve()), 'sha256': sha(args.output.read_bytes()),
                      'ledger_entries': len(ledger), 'tests': summary}, indent=2))

if __name__ == '__main__':
    main()
