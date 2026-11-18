"""Assemble review artifacts without tests, publication or checkout staging."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

R = Path(__file__).resolve().parent
S = R.parent/'ConsenDxFormer-performance-v2'
digest = lambda raw: hashlib.sha256(raw).hexdigest()
sha = lambda path: digest(path.read_bytes())
def git(*args):
    return subprocess.check_output(['git', *args], cwd=S)
def dump(name, value):
    (R/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')

before = json.loads((R/'before.json').read_bytes())
changed = ['scripts/performance_v2_native_acceptance.py', 'tests/test_performance_v2_acceptance_entry.py',
           'docs/performance_v2_native_acceptance_checklist.md', 'configs/performance_v2/native_acceptance_source_lock.json']
names = sorted(filter(None, git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')))
assert set(names) == set(before['hashes'])
assert git('rev-parse', 'HEAD').decode().strip() == before['head']
assert git('branch', '--show-current').decode().strip() == before['branch']
assert digest(git('diff', '--cached', '--binary')) == before['staged_diff']
assert digest(git('diff', '--binary')) == before['tracked_diff']
actual_changed = sorted(n for n in names if sha(S/n) != before['hashes'][n])
assert actual_changed == sorted(changed), actual_changed
for name in changed:
    if name.endswith('.py'):
        ast.parse((S/name).read_text(), feature_version=(3, 10))
lock_name = 'configs/performance_v2/native_acceptance_source_lock.json'
lock_hash = sha(S/lock_name)
lock = json.loads((S/lock_name).read_bytes())
for name, expected in lock['files'].items():
    assert sha(S/name) == sha(R/'candidate'/name) == expected
assert sha(R/'candidate'/lock_name) == lock_hash
assert lock['pilot_execution_authorized'] is False

def diff(names, prior):
    result = []
    for name in names:
        old = (R/'before'/name).read_bytes().decode('utf-8').splitlines(keepends=True) if prior else []
        new = (S/name).read_bytes().decode('utf-8').splitlines(keepends=True)
        result.append('diff --git a/'+name+' b/'+name+'\n')
        if not prior:
            result.append('new file mode 100644\n')
        for line in difflib.unified_diff(old, new, fromfile='a/'+name if prior else '/dev/null', tofile='b/'+name):
            result.append(line)
            if not line.endswith('\n'):
                result.append('\n\\ No newline at end of file\n')
    return ''.join(result)
untracked = sorted(filter(None, git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')))
for output, paths, prior in [('FOCUSED.diff', changed, True), ('COMPLETE-WORKTREE.diff', untracked, False)]:
    (R/output).write_text(diff(paths, prior), encoding='utf-8', newline='\n')
    with tempfile.TemporaryDirectory(prefix='pv2-inventory-patch-') as tmp:
        target = Path(tmp)
        if prior:
            shutil.copytree(R/'before', target, dirs_exist_ok=True)
        for flags in (['--check'], []):
            check = subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', '--whitespace=nowarn',
                                    *flags, str(R/output)], cwd=target, capture_output=True, text=True)
            assert check.returncode == 0, check.stderr
        for name in paths:
            assert (target/name).read_bytes() == (S/name).read_bytes(), name

references = R/'references'
references.mkdir(exist_ok=True)
old = R.parent/'performance-v2-native-acceptance-20261007/performance-v2-consolidated-native-acceptance-REVIEW-20261007.zip'
assert sha(old) == '92d59995665782e0aa7ca1912a2bddc29d5c60b9842cc597627d08a1be25e7aa'
shutil.copyfile(old, references/old.name)
prior = R.parent/'performance-v2-university-preflight-20261008'
for name in ('STDOUT-STDERR.txt', 'REVIEW.md', 'SHA256-LEDGER.json'):
    shutil.copyfile(prior/name, references/('prior-preflight-'+name))
commands = (R.parent/'performance-v2-native-acceptance-20261007/ACCEPTANCE-COMMANDS.md').read_text()
commands = commands.replace('4a8b34884d0760f93336fc14b80f634b0d7eaa9be113762e04f09a907465edba', lock_hash)
commands = commands.replace('130', '143')
commands = commands.replace('umask 077\n', 'umask 077\nexport CUDA_VISIBLE_DEVICES=""\n')
commands = commands[:commands.index('## Explicit partial check')]
commands += ('\n## Inventory correction\n\nSee REVIEW.md. Invalid/missing metadata yields JSON with errors and exit 2. '
             'Virtual/base records and duplicates are separate. No full acceptance command was run for this revision. '
             'The directory proposal is not creation or execution authorization.\n')
(R/'ACCEPTANCE-COMMANDS.md').write_text(commands, encoding='utf-8', newline='\n')
tests = json.loads((R/'FOCUSED-TESTS.json').read_bytes())
assert tests['passed'] and tests['tests'] == 13 and not tests['acceptance_suite_executed']
dump('VALIDATION.json', {'focused_tests': tests, 'source_lock_sha256': lock_hash,
    'frozen_files': len(lock['files']), 'changed_files': changed, 'unchanged_files': len(names)-len(changed),
    'HEAD': before['head'], 'branch': before['branch'], 'HEAD_index_tracked_diff_unchanged': True,
    'both_diffs_reconstructed_exact_bytes': True, 'Python_3_10_AST_passed': True,
    'acceptance_suite_executed': False, 'pilot_execution_authorized': False})
dump('SHA256-LEDGER.json', {'source_lock_sha256': lock_hash,
    'changed_files': {n: sha(S/n) for n in changed},
    'diffs': {n: sha(R/n) for n in ('FOCUSED.diff', 'COMPLETE-WORKTREE.diff')},
    'pilot_execution_authorized': False})
zip_name = 'performance-v2-native-inventory-correction-REVIEW-20261008.zip'
excluded = {zip_name, zip_name+'.sha256', 'PACKAGE-LEDGER.json', 'ZIP-VERIFICATION.json'}
members = sorted(p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.relative_to(R).as_posix() not in excluded)
ledger = {p.relative_to(R).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in members}
dump('PACKAGE-LEDGER.json', {'files': ledger, 'self_excluded': True, 'pilot_execution_authorized': False})
members.append(R/'PACKAGE-LEDGER.json')
with zipfile.ZipFile(R/zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(members):
        z.write(p, p.relative_to(R).as_posix())
with zipfile.ZipFile(R/zip_name) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == len(members)
    for name, record in ledger.items():
        raw = z.read(name)
        assert digest(raw) == record['sha256'] and len(raw) == record['bytes']
verification = {'zip_sha256': sha(R/zip_name), 'source_lock_sha256': lock_hash,
                'zip_members_verified': len(members), 'pilot_execution_authorized': False}
dump('ZIP-VERIFICATION.json', verification)
(R/(zip_name+'.sha256')).write_text(verification['zip_sha256']+'  '+zip_name+'\n', encoding='utf-8')
print(json.dumps(verification, indent=2))
