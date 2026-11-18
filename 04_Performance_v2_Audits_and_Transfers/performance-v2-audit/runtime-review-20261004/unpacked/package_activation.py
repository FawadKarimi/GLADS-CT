"""Explicit local packaging and mocked validation; never invokes native roles."""
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

BASE = '8a0365021c70f5fe0110fee2ef5ef8e209e52c232e81cd5b06aa78b1e8b68988'
EDITED = {'launcher_binding.py', 'native_launcher.py', 'linux_host.py', 'linux_lifecycle.py',
          'linux_integration.py', 'watchdog_worker.py', 'supervisor_v2.py',
          'supervisor_evidence.py', 'test_launcher_binding.py', 'launcher-sources.json'}


def sha(raw): return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'base', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    if sha(a.base.read_bytes()) != BASE: raise ValueError('base SHA')
    with zipfile.ZipFile(a.base) as z:
        previous = json.loads(z.read('file-hashes.json'))
        if set(z.namelist()) != set(previous) | {'file-hashes.json'}: raise ValueError('base coverage')
        for name, rec in previous.items():
            raw = z.read(name)
            if sha(raw) != rec['sha256'] or len(raw) != rec['size_bytes']: raise ValueError('parent '+name)
            if name not in EDITED and (root/name).read_bytes() != raw: raise ValueError('unintended edit '+name)
        (root/'history'/'launcher-file-hashes.json').write_bytes(z.read('file-hashes.json'))
        (root/'history'/'launcher-sources-before-activation.json').write_bytes(z.read('launcher-sources.json'))
        write(root/'launcher-sources.json', source_manifest(root))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        targeted = [sys.executable, '-B', '-m', 'unittest', 'test_activation_bridge', 'test_launcher_binding', '-v']
        full = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(root), '-p', 'test_*.py', '-v']
        results = []
        for label, cmd in (('targeted', targeted), ('full', full)):
            run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=120)
            (root/('activation-'+label+'-tests.log')).write_text(run.stdout+run.stderr, encoding='utf-8')
            if run.returncode: raise RuntimeError(label+' tests failed')
            results.append({'suite': label, 'command': cmd, 'exit_code': run.returncode,
                            'summary': re.search(r'Ran \d+ tests in [^\n]+', run.stderr).group()})
        changed, diff = [], []
        for path in sorted(root.iterdir()):
            if path.suffix not in ('.py', '.md') and path.name != 'launcher-sources.json': continue
            raw = path.read_bytes()
            before = z.read(path.name) if path.name in previous else b''
            if raw == before: continue
            text = raw.decode()
            if any(line.rstrip() != line for line in text.splitlines()): raise ValueError('whitespace '+path.name)
            if path.suffix == '.py':
                ast.parse(text, feature_version=(3, 10)); compile(text, path.name, 'exec')
            changed.append(path.name)
            diff.extend(difflib.unified_diff(before.decode().splitlines(True), text.splitlines(True),
                                             fromfile='a/'+path.name if before else '/dev/null', tofile='b/'+path.name))
        (root/'activation-focused.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')
        write(root/'ACTIVATION_VALIDATION.json', {
            'base_sha256': BASE, 'base_entries_verified': len(previous), 'tests': results,
            'runtime': sys.version, 'platform': platform.platform(), 'changed_source_documents': changed,
            'python_310_syntax': 'passed for changed source; execution used recorded Windows runtime',
            'test_boundaries': 'invented kernel/processes, filesystem, clocks and mocked subprocess boundaries; synthetic temporary files',
            'real_separate_interpreter_tested': False, 'native_linux_acceptance_executed': False,
            'implementation': 'connected role handlers, trust/admission, independent reconstruction, loss collection and teardown source',
            'unresolved': ['native inventory/runtime closure/security evidence', 'owner execution authorization', 'native Linux acceptance'],
            'execution_approved': False, 'security_approved': False, 'dependencies_approved': False,
            'pilot_execution_authorized': False, 'stage_b': 'paused', 'production_pins_unchanged': True,
            'preserved_warning': 'historical acceptance_driver.py return-in-finally SyntaxWarning under Python 3.14',
            'development_failures': ['old rejection error-message assertions restored', 'invented Linux path fixtures corrected to PurePosixPath on Windows'],
            'original_15_regressions_unchanged': True})
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
            if len(raw) != rec['size_bytes'] or sha(raw) != rec['sha256']: raise ValueError('output '+name)
    print(json.dumps({'zip': str(a.output.resolve()), 'sha256': sha(a.output.read_bytes()),
                      'ledger_entries': len(ledger), 'tests': results}, indent=2))


if __name__ == '__main__': main()
