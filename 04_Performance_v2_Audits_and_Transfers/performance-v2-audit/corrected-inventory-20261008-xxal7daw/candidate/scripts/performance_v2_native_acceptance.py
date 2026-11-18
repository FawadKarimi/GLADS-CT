"""Portable, offline, synthetic acceptance only. No runner or installation path."""
import argparse
import contextvars
from contextlib import ExitStack
import hashlib
from email.parser import BytesParser
from email.policy import default as email_policy
import json
import os
from pathlib import Path
import platform
import re
import shutil
import site
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import warnings
import _sqlite3

ROOT = Path(__file__).resolve().parents[1]
ENTRY = Path(__file__).resolve()
LOCK = ROOT / 'configs/performance_v2/native_acceptance_source_lock.json'
CORE = ['test_performance_v2_contracts', 'test_performance_v2_implementation',
        'test_performance_v2_training_mode', 'test_performance_v2_lifecycle',
        'test_performance_v2_checkpoint_inventory', 'test_performance_v2_durable_lifecycle',
        'test_performance_v2_acceptance_entry', 'test_performance_v2_native_processes']
LEGACY = ['test_performance_v2_diagnostics', 'test_performance_v2_transform_probe', 'test_performance_v2_input_audit']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_lock(expected):
    if sha(LOCK) != expected:
        raise ValueError('source lock digest mismatch')
    lock = json.loads(LOCK.read_bytes())
    if lock['pilot_execution_authorized'] is not False:
        raise ValueError('inactive source lock required')
    for name, expected_hash in lock['files'].items():
        path = ROOT / name
        if path.resolve() != path.absolute() or not path.is_file() or sha(path) != expected_hash:
            raise ValueError('source/configuration drift: ' + name)
    # Added test/module files must not escape the frozen suite inventory.
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('test_performance_v2_*.py')}
    actual |= {p.relative_to(ROOT).as_posix() for p in (ROOT/'src/consendxformer/performance_v2').glob('*.py')}
    if actual - set(lock['files']):
        raise ValueError('unfrozen module or synthetic test')
    return lock


def mount_record(path):
    if not sys.platform.startswith('linux'):
        return {'verified': False, 'reason': 'not Linux'}
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, extra = left.split(), right.split()
        mount = fields[4].replace('\\040', ' ').replace('\\134', '\\')
        try:
            Path(path).resolve().relative_to(mount)
        except ValueError:
            continue
        matches.append((len(mount), {'mount_point': mount, 'mount_options': fields[5],
                                    'filesystem': extra[0], 'source': extra[1], 'super_options': extra[2]}))
    return max(matches, key=lambda x: x[0])[1] if matches else {'verified': False}


INVENTORY_DEPENDENCIES = {'torch', 'torchvision', 'numpy', 'pytest',
                        'opencv-python', 'opencv-python-headless', 'packaging'}


def dependency_metadata():
    """Inspect explicit prefixes, never execute site initialization or .pth files.

    Python 3.10 -S leaves sys.prefix pointing at the base interpreter. Preserve
    the invocation path (not its executable symlink target) to find pyvenv.cfg.
    Scope records are evidence, not an emulation of arbitrary import hooks.
    """
    invocation = Path(sys.executable).absolute()
    config_candidates = [invocation.parent/'pyvenv.cfg', invocation.parent.parent/'pyvenv.cfg']
    config = next((p for p in config_candidates if p.is_file()), None)
    errors = []
    venv_root, include_base, config_record = None, False, None
    if config:
        raw = config.read_bytes()
        settings = {}
        for line in raw.decode('utf-8').splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                if key in settings:
                    errors.append({'path': str(config), 'reason': 'duplicate pyvenv.cfg key: ' + key})
                settings[key] = value.strip()
        venv_root = config.parent
        setting = settings.get('include-system-site-packages', '').lower()
        if setting not in ('true', 'false'):
            errors.append({'path': str(config), 'reason': 'missing/invalid include-system-site-packages'})
        include_base = setting == 'true'
        config_record = {'path': str(config), 'sha256': hashlib.sha256(raw).hexdigest(),
                         'include_system_site_packages': setting, 'home': settings.get('home')}
    elif sys.prefix != sys.base_prefix:
        errors.append({'path': str(invocation), 'reason': 'virtual prefix without pyvenv.cfg binding'})

    def paths(prefixes):
        return list(dict.fromkeys(str(Path(p).absolute()) for p in site.getsitepackages(prefixes)))

    roots = {'virtual_environment': paths([str(venv_root)]) if venv_root else [],
             'base_interpreter': paths(list(dict.fromkeys([sys.base_prefix, sys.base_exec_prefix])))}
    scopes = {}
    for scope, search_paths in roots.items():
        records, seen = [], set()
        for search_path in search_paths:
            parent = Path(search_path)
            if not parent.is_dir():
                continue
            for descriptor in sorted(parent.iterdir()):
                if descriptor.suffix.lower() not in ('.dist-info', '.egg-info'):
                    continue
                identity = str(descriptor.resolve())
                if identity in seen:
                    continue  # aliases of the same search root are not duplicate installs
                seen.add(identity)
                filename = 'METADATA' if descriptor.suffix.lower() == '.dist-info' else 'PKG-INFO'
                member = descriptor/filename if descriptor.is_dir() else descriptor
                if descriptor.suffix.lower() == '.dist-info' and not descriptor.is_dir():
                    errors.append({'scope': scope, 'path': str(descriptor), 'reason': 'invalid dist-info descriptor'})
                    continue
                try:
                    raw = member.read_bytes()
                    parsed = BytesParser(policy=email_policy).parsebytes(raw)
                    names, versions = parsed.get_all('Name', []), parsed.get_all('Version', [])
                    if (not raw.strip() or parsed.defects or len(names) != 1 or len(versions) != 1
                            or not str(names[0]).strip() or not str(versions[0]).strip()):
                        raise ValueError('empty/malformed metadata or missing/duplicate Name/Version')
                    name = re.sub(r'[-_.]+', '-', str(names[0]).strip()).lower()
                    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name):
                        raise ValueError('invalid distribution Name')
                except (OSError, ValueError) as error:
                    errors.append({'scope': scope, 'path': str(member), 'reason': str(error)})
                    continue
                if name in INVENTORY_DEPENDENCIES:
                    records.append({'name': name, 'version': str(versions[0]).strip(),
                                    'descriptor': str(descriptor), 'metadata_path': str(member),
                                    'metadata_sha256': hashlib.sha256(raw).hexdigest(),
                                    'metadata_bytes': len(raw), 'search_path': search_path})
        scopes[scope] = {'prefix': str(venv_root) if scope == 'virtual_environment' and venv_root
                        else sys.base_prefix if scope == 'base_interpreter' else None,
                        'search_paths': search_paths, 'distributions': records,
                        'missing_selected_names': sorted(INVENTORY_DEPENDENCIES - {r['name'] for r in records})}
    by_name = {}
    for scope, data in scopes.items():
        for record in data['distributions']:
            by_name.setdefault(record['name'], []).append({'scope': scope, **record})
    duplicates = {name: records for name, records in by_name.items() if len(records) > 1}
    active_scopes = ['virtual_environment'] if venv_root else ['base_interpreter']
    if venv_root and include_base:
        active_scopes.append('base_interpreter')
    active = {}
    for scope in active_scopes:
        for record in scopes[scope]['distributions']:
            if record['name'] in active:
                errors.append({'scope': scope, 'path': record['descriptor'],
                               'reason': 'ambiguous active distribution: ' + record['name']})
            else:
                active[record['name']] = {'scope': scope, **record}
    return {'schema': 'performance_v2_dependency_metadata_v2', 'invocation': str(invocation),
            'resolved_executable': str(invocation.resolve()), 'pyvenv_config': config_record,
            'prefix_under_no_site': sys.prefix, 'base_prefix': sys.base_prefix,
            'base_exec_prefix': sys.base_exec_prefix, 'scopes': scopes, 'duplicates': duplicates,
            'active_scopes': active_scopes, 'active_distributions': active,
            'errors': errors, 'valid': not errors,
            'import_resolution_verified': False, 'site_initialization_executed': False}


def inventory(scratch=None):
    # Metadata inspection only. No import of Torch/cv2, device probe, package
    # installation, subprocess or persistent SQLite database is needed.
    dependencies = dependency_metadata()
    git = shutil.which('git')
    result = {'python': sys.version, 'executable': sys.executable, 'executable_sha256': sha(sys.executable),
              'os': sys.platform, 'machine': platform.machine(), 'sqlite': sqlite3.sqlite_version,
              'sqlite_extension_sha256': sha(_sqlite3.__file__), 'target_image_digest': None,
              'distributions': dependencies['active_distributions'], 'dependency_metadata': dependencies,
              'git': {'path': git, 'sha256': sha(git)} if git else None,
              'pilot_execution_authorized': False, 'read_only': True,
              'weights_devices_and_research_data_inspected': False}
    connection = sqlite3.connect(':memory:')
    try:
        result['sqlite_compile_options'] = [row[0] for row in connection.execute('PRAGMA compile_options')]
    finally:
        connection.close()
    if sys.platform.startswith('linux'):
        result.update(uid=os.getuid(), gid=os.getgid(), uname=list(os.uname()))
        for name in ('/etc/os-release', '/proc/self/status', '/proc/self/cgroup'):
            result[name] = Path(name).read_text()
        result['libc'] = os.confstr('CS_GNU_LIBC_VERSION') if 'CS_GNU_LIBC_VERSION' in os.confstr_names else 'unresolved'
    if scratch:
        path = Path(scratch).absolute()
        info = path.stat()
        result['scratch'] = {'path': str(path), 'mode': oct(stat.S_IMODE(info.st_mode)),
                             'uid': info.st_uid, 'device': info.st_dev, 'mount': mount_record(path)}
    return result


def inside(path, root):
    try:
        Path(path).absolute().resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, TypeError, OSError):
        return False


def guards(scratch, *, worker=False, legacy=False):
    scratch = Path(scratch).resolve()
    git = shutil.which('git')
    authorized = contextvars.ContextVar('bounded_child_spawn', default=None)
    original_popen = subprocess.Popen

    def permitted(argv):
        if worker or not isinstance(argv, (list, tuple)) or not all(type(x) is str for x in argv):
            return False
        prefix = [sys.executable, '-I', '-B', str(ENTRY), 'worker']
        if list(argv[:5]) == prefix and len(argv) == 15:
            options = dict(zip(argv[5::2], argv[6::2]))
            return (set(options) == {'--root', '--mode', '--boundary', '--slot', '--expected-lock-sha256'}
                and inside(options['--root'], scratch) and options['--mode'] in ('compete', 'reserved', 'terminal')
                and options['--slot'] in ('a', 'b') and options['--expected-lock-sha256'] == os.environ['PV2_LOCK_SHA256'])
        if legacy and git and argv and argv[0] in ('git', git):
            a = list(argv[1:])
            if len(a) == 2 and a[0] == 'init' and inside(a[1], scratch):
                return True
            if len(a) == 10 and a[:1] == ['-C'] and inside(a[1], scratch):
                return a[2:] == ['-c', 'user.name=Synthetic Fixture', '-c', 'user.email=fixture@example.invalid',
                                 'commit', '--allow-empty', '-m', 'invented fixture']
            if len(a) > 5 and a[:4] == ['--no-optional-locks', '-c', 'core.fsmonitor=false', '-C'] and inside(a[4], scratch):
                return a[5:] in (['branch', '--show-current'], ['rev-parse', 'HEAD'],
                                  ['status', '--porcelain=v1', '--untracked-files=all'])
        return False

    class BoundedPopen(original_popen):
        def __init__(self, args, **kwargs):
            if not permitted(args) or kwargs.get('shell') or kwargs.get('executable') not in (None, args[0]):
                raise RuntimeError('unapproved subprocess denied')
            token = authorized.set(tuple(args))
            try:
                super().__init__(args, **kwargs)
            finally:
                authorized.reset(token)

    def audit(event, args):
        if event.startswith('socket.') or event == 'os.system':
            raise RuntimeError('acceptance network/shell denied: ' + event)
        if event in ('os.posix_spawn', 'os.posix_spawnp'):
            bound = authorized.get()
            if bound is None or tuple(args[1]) != bound or os.fsdecode(args[0]) != bound[0]:
                raise RuntimeError('unapproved subprocess denied')
        if event == 'subprocess.Popen':
            executable, argv, cwd, env = args
            bound = authorized.get()
            # Windows exposes the flattened CRT command line at this audit
            # event. Compare it to the *prevalidated list*, never parse a shell.
            same = bound is not None and (argv == subprocess.list2cmdline(bound) if isinstance(argv, str) else tuple(argv) == bound)
            if not same or executable is not None and os.fsdecode(executable) != bound[0]:
                raise RuntimeError('unapproved subprocess denied')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0]).replace('\\', '/').lower()
            sensitive = name.endswith(('.pt', '.pth', '.ckpt', '.safetensors', '.npy', '.npz', '.dcm'))
            if (sensitive or '/data/processed/' in name or '/data/raw/' in name) and not inside(args[0], scratch):
                raise RuntimeError('existing checkpoint/research file access denied')
    sys.addaudithook(audit)
    subprocess.Popen = BoundedPopen


def torch_guards(stack):
    import torch
    import torchvision
    import numpy
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise RuntimeError('weights, compilation, GPU and optimizer updates forbidden')
    for obj, name in ((torch, 'load'), (torch.jit, 'load'), (torch, 'compile'),
                      (torch.hub, 'load_state_dict_from_url'), (torch.hub, 'download_url_to_file'),
                      (torch.optim.AdamW, 'step'), (torch.optim.SGD, 'step'), (torch.cuda, 'init')):
        stack.enter_context(patch.object(obj, name, forbidden))
    return {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': numpy.__version__,
            'cuda_build': torch.version.cuda, 'GPU_exercised': False,
            'module_hashes': {str(Path(m.__file__).resolve()): sha(m.__file__) for m in (torch, torchvision, numpy)}}


def worker_main(args):
    verify_lock(args.expected_lock_sha256)
    scratch = os.environ['PV2_ACCEPTANCE_SCRATCH']
    if not inside(args.root, scratch):
        raise ValueError('worker outside synthetic scratch')
    platform.uname(); platform.processor()
    guards(scratch, worker=True)
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
    with ExitStack() as stack:
        torch_guards(stack)
        from test_performance_v2_native_processes import worker
        return worker(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    inv = subs.add_parser('inventory')
    inv.add_argument('--scratch-parent')
    inv.add_argument('--expected-lock-sha256', required=True)
    run = subs.add_parser('test')
    run.add_argument('--expected-lock-sha256', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--scratch-parent', required=True)
    run.add_argument('--suite', choices=('full', 'implementation'), default='full')
    run.add_argument('--require-linux', action='store_true')
    child = subs.add_parser('worker')
    for key in ('root', 'mode', 'boundary', 'slot', 'expected-lock-sha256'):
        child.add_argument('--' + key, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', TORCHDYNAMO_DISABLE='1',
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='')
    os.environ.pop('PYTEST_PLUGINS', None)
    if args.command == 'worker':
        return worker_main(args)
    lock = verify_lock(args.expected_lock_sha256)
    observed = inventory(args.scratch_parent)
    observed['source_lock_sha256'] = args.expected_lock_sha256
    if args.command == 'inventory':
        print(json.dumps(observed, indent=2, sort_keys=True))
        return 0 if observed['dependency_metadata']['valid'] else 2
    output = Path(args.output).absolute()
    if inside(output, ROOT) or inside(ROOT, output):
        raise ValueError('review output must not overlap source')
    output.mkdir(mode=0o700, exist_ok=False)
    result = {'schema': 'performance_v2_native_acceptance_result_v1', 'inventory': observed,
              'suite': args.suite, 'pilot_execution_authorized': False, 'native_acceptance': False,
              'power_loss_tested': False, 'status': 'blocked', 'required_tests': CORE + LEGACY}
    write_json(output/'inventory.json', observed)
    try:
        if not observed['dependency_metadata']['valid']:
            raise RuntimeError('invalid/ambiguous installed dependency metadata; see retained inventory errors')
        missing = [x for x in ('torch', 'torchvision', 'numpy') if x not in observed['distributions']]
        if args.suite == 'full':
            missing += [x for x in ('pytest',) if x not in observed['distributions']]
            if not any(x in observed['distributions'] for x in ('opencv-python', 'opencv-python-headless')):
                missing.append('opencv-python or opencv-python-headless')
            if not observed['git']:
                missing.append('git executable')
        if missing:
            raise RuntimeError('missing installed dependencies; no substitution: ' + ', '.join(missing))
        parent = Path(args.scratch_parent).absolute()
        if any(p.is_symlink() for p in (parent, *parent.parents)) or not parent.is_dir():
            raise ValueError('existing private nonsymlink scratch parent required')
        if args.require_linux:
            if not sys.platform.startswith('linux'):
                raise RuntimeError('native Linux required; this host is ' + sys.platform)
            info = parent.stat()
            if os.getuid() == 0 or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise RuntimeError('non-root owner and mode 0700 required')
            status = observed['/proc/self/status']
            mask = next((line.split()[1] for line in status.splitlines() if line.startswith('Umask:')), None)
            if mask is None or int(mask, 8) & 0o077 != 0o077:
                raise RuntimeError('umask 077 required; inventory cannot establish it')
            if observed['scratch']['mount'].get('filesystem') not in ('ext4', 'xfs'):
                raise RuntimeError('filesystem not in proposed ext4/xfs acceptance scope')
        os.environ['PV2_LOCK_SHA256'] = args.expected_lock_sha256
        os.environ['PV2_NATIVE_ENTRY'] = str(ENTRY)
        evidence = output/'synthetic-evidence'
        evidence.mkdir()
        os.environ['PV2_SYNTHETIC_EVIDENCE_DIRECTORY'] = str(evidence)
        with tempfile.TemporaryDirectory(prefix='pv2-native-synthetic-', dir=parent) as tmp:
            os.environ['PV2_ACCEPTANCE_SCRATCH'] = tmp
            os.environ['TORCH_HOME'] = str(Path(tmp)/'disabled-weights')
            tempfile.tempdir = tmp
            templates = Path(tmp)/'empty-git-template'
            templates.mkdir()
            for key in list(os.environ):
                if key.startswith('GIT_'):
                    del os.environ[key]
            os.environ.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_COUNT='2',
                GIT_CONFIG_KEY_0='core.hooksPath', GIT_CONFIG_VALUE_0=str(templates),
                GIT_CONFIG_KEY_1='init.templateDir', GIT_CONFIG_VALUE_1=str(templates))
            # Populate Windows metadata cache before denying unrelated subprocesses.
            platform.uname(); platform.processor()
            guards(tmp, legacy=args.suite == 'full')
            sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
            with warnings.catch_warnings(record=True) as caught, ExitStack() as stack:
                result['runtime'] = torch_guards(stack)
                suite = unittest.defaultTestLoader.loadTestsFromNames(CORE)
                with (output/'unittest.log').open('x', encoding='utf-8') as log:
                    outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
                result['unittest'] = {'tests': outcome.testsRun, 'failures': len(outcome.failures),
                    'errors': len(outcome.errors), 'skipped': len(outcome.skipped), 'successful': outcome.wasSuccessful()}
                legacy_ok = False
                if args.suite == 'full':
                    import pytest
                    class Counts:
                        def __init__(self):
                            self.passed, self.failed, self.skipped = 0, 0, 0
                        def pytest_runtest_logreport(self, report):
                            if report.skipped or hasattr(report, 'wasxfail'):
                                self.skipped += 1
                            elif report.failed:
                                self.failed += 1
                            elif report.when == 'call' and report.passed:
                                self.passed += 1
                    counts = Counts()
                    config = Path(tmp)/'pytest.ini'
                    config.write_text('[pytest]\n', encoding='utf-8')
                    code = pytest.main(['-q', '-c', str(config), '--noconftest', '-p', 'no:cacheprovider',
                                        '--basetemp', str(Path(tmp)/'pytest'),
                                        '--junitxml', str(output/'pytest.xml'),
                                        *[str(ROOT/'tests'/(n+'.py')) for n in LEGACY]], plugins=[counts])
                    result['pytest_exit_code'] = int(code)
                    result['pytest'] = vars(counts)
                    legacy_ok = code == 0 and counts.passed > 0 and not counts.skipped and not counts.failed
                result['warnings'] = [str(w.message) for w in caught]
            verify_lock(args.expected_lock_sha256)
            kills = [json.loads(p.read_bytes()) for p in evidence.glob('test_*/kill-*.json')]
            result['process_kill_evidence'] = kills
            killed_all = len(kills) == 7 and all(k['child_pid'] != k['parent_pid'] for k in kills)
            success = outcome.wasSuccessful() and not outcome.skipped and killed_all and (legacy_ok or args.suite == 'implementation')
            result['status'] = 'passed' if success else 'failed'
            result['complete_suite_passed'] = success and args.suite == 'full'
            result['native_acceptance'] = False  # a report is evidence, never owner acceptance
            result['native_acceptance_evidence_passed'] = success and args.suite == 'full' and args.require_linux
            result['separate_process_kill_tests_executed'] = killed_all
            result['kill_method'] = 'SIGKILL' if os.name == 'posix' else 'TerminateProcess'
            result['process_suite_directory_fsync_mocked'] = os.name != 'posix'
        return 0 if result['status'] == 'passed' else 1
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        return 2
    finally:
        result['evidence_hashes'] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'result.json', result)
        print(json.dumps({'status': result['status'], 'output': str(output), 'error': result.get('error'),
                          'pilot_execution_authorized': False}, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
