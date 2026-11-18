"""Proposed orchestration only; never changes the reviewed inspection package."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
import signal
import build_lifecycle

PACKAGE_SHA = '2f97fdaf4dd7e3dd5a04255aa62a7dd7b603fcda0fd82ffcb026b3957407d45e'
SCENARIOS = ('pipeline', 'deny-network', 'deny-fork', 'deny-root-write', 'forced-failure', 'overflow', 'timeout')
REQUIRED = ('inspector.py', 'cgroup_limits.py', 'container_entry.py', 'linux_synthetic.py', 'launch_synthetic.py', 'launch.py', 'environment-lock.json', 'seccomp.json', 'Dockerfile', 'requirements.lock', 'sources/frozen-resnet.py')

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)

def persist_report(reports, report):
    """Preserve failure evidence independently; failure to persist is not success."""
    errors = []
    try:
        save(reports / 'orchestration-report.json', report)
    except BaseException:
        errors.append(traceback.format_exc())
    try:
        # The always-cleanup step refreshes this ledger after its own evidence.
        build_lifecycle.finalize_report_hashes(reports)
    except BaseException:
        errors.append(traceback.format_exc())
    if errors:
        report['status'] = 'partial'
        report['persistence_errors'] = errors
    print(json.dumps(report, sort_keys=True), flush=True)
    return not errors

def validate_approval(approval):
    require(approval.get('package_sha256') == PACKAGE_SHA, 'approval package binding')
    root = Path(__file__).resolve().parent
    for name in ('acceptance_driver.py', 'build_lifecycle.py'):
        require(approval.get('orchestration_hashes', {}).get(name) == sha(root / name), 'approval implementation binding: ' + name)
    require(approval.get('synthetic_execution_approved') is True and approval.get('native_build_execution_reviewed') is True, 'review approval missing')
    require(approval.get('production_loading_approved') is False, 'production authorization forbidden')
    require(bool(approval.get('review_reference')), 'missing security review reference')

def metadata(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
    require(len(result.stdout) + len(result.stderr) < 262144, 'oversize metadata')
    return result.stdout

def unpack(package, root):
    require(sha(package) == PACKAGE_SHA, 'reviewed package hash mismatch')
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate ZIP member')
        require(sum(i.file_size for i in archive.infolist()) < 16 * 1024**2, 'package too large')
        ledger = json.loads(archive.read('file-hashes.json'))
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            require(not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename, 'unsafe ZIP path')
            require((item.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink')
            data = archive.read(item)
            if item.filename in ledger:
                expected = ledger[item.filename]
                require(len(data) == expected['size_bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], 'package member mismatch')
        require(all(name in ledger for name in REQUIRED), 'required member not hash-bound')
        # Only actual build/launcher inputs are materialized, never package helper scripts.
        for name in REQUIRED:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(archive.read(name))

class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(allowed_url(newurl), 'nonofficial download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def allowed_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == 'https' and parsed.hostname in {'download.pytorch.org', 'download-r2.pytorch.org', 'files.pythonhosted.org'} and not parsed.username and not parsed.password

def download(package, destination):
    require(allowed_url(package['url']), 'unapproved dependency host')
    started = time.monotonic()
    size = 0
    with urllib.request.build_opener(OfficialRedirect).open(package['url'], timeout=20) as response, destination.open('xb') as stream:
        require(allowed_url(response.url), 'nonofficial final URL')
        resolved = response.url
        while True:
            require(time.monotonic() - started < 180, 'download deadline')
            block = response.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            require(size <= package['size_bytes'], 'oversize dependency')
            stream.write(block)
    require(size == package['size_bytes'] and sha(destination) == package['sha256'], 'dependency hash/size mismatch')
    return {'url': package['url'], 'resolved_url': resolved, 'size_bytes': size, 'sha256': sha(destination)}

def main():
    def interrupted(signum, frame):
        raise InterruptedError('orchestration interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    reports = args.output / 'reports'
    reports.mkdir()
    report = {'status': 'partial', 'scope': 'linux_synthetic_acceptance_only', 'pilot_execution_authorized': False, 'stage_b': 'paused', 'security_clearance_passed': False, 'linux_acceptance_passed': False, 'package_sha256': PACKAGE_SHA, 'run_id': os.environ.get('GITHUB_RUN_ID'), 'workflow_commit': os.environ.get('GITHUB_SHA'), 'scenarios': []}
    try:
        approval = json.loads(args.approval.read_bytes())
        validate_approval(approval)
        report['approval_sha256'] = sha(args.approval)
        require(platform.machine() == 'x86_64' and os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted', 'host platform')
        require('VERSION_ID="24.04"' in Path('/etc/os-release').read_text(), 'host OS')
        require(os.environ.get('DOCKER_HOST') == 'unix:///var/run/docker.sock' and not os.environ.get('DOCKER_CONTEXT'), 'local Docker only')
        info = json.loads(metadata(['docker', 'info', '--format', '{{json .}}']))
        fields = ('ServerVersion', 'OSType', 'Architecture', 'KernelVersion', 'CgroupVersion', 'CgroupDriver', 'SecurityOptions', 'MemoryLimit', 'SwapLimit', 'CpuCfsQuota', 'CpuCfsPeriod', 'PidsLimit', 'Warnings')
        report['host'] = {k: info.get(k) for k in fields}
        require(info.get('OSType') == 'linux' and info.get('Architecture') == 'x86_64', 'Docker platform')
        require(all(info.get(k) is True for k in ('MemoryLimit', 'SwapLimit', 'CpuCfsQuota', 'CpuCfsPeriod', 'PidsLimit')), 'Docker controls unavailable')
        require(info.get('CgroupVersion') in ('1', '2') and not info.get('Warnings'), 'Docker version/warnings')
        require(any('seccomp' in s for s in info.get('SecurityOptions', [])), 'seccomp unavailable')
        report['meminfo'] = Path('/proc/meminfo').read_text()
        report['swaps'] = Path('/proc/swaps').read_text()
        require(int(next(line.split()[1] for line in report['meminfo'].splitlines() if line.startswith('MemAvailable:'))) >= 5 * 1024**2, 'need 5 GiB available host RAM')
        report['disk'] = {p: shutil.disk_usage(p)._asdict() for p in (str(args.output), '/var/lib/docker')}
        require(all(d['free'] >= 10 * 1024**3 for d in report['disk'].values()), 'need 10 GiB free; no automatic cleanup')
        root = args.output / 'package'
        root.mkdir()
        unpack(args.package, root)
        lock = json.loads((root / 'environment-lock.json').read_bytes())
        report['downloads'] = []
        (root / 'wheels').mkdir()
        for package in lock['packages']:
            target = root / 'wheels' / Path(package['file']).name
            report['downloads'].append(download(package, target))
        # Reuse reviewed bounded host lifecycle wrapper for synthetic containers.
        sys.path.insert(0, str(root))
        import launch
        import launch_synthetic
        # Build/pull logs have a 16 MiB file bound and explicit deadlines. No retries.
        def build_command(command, label, seconds):
            import resource
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
            with (reports / (label + '.log')).open('xb') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=seconds, check=True, preexec_fn=limits)
        build_command(['docker', 'pull', '--platform=linux/amd64', lock['base_image']], 'pull', 180)
        base = json.loads(metadata(['docker', 'image', 'inspect', lock['base_image']]))[0]
        require(base['Id'] == lock['image_config_digest'] and base['Architecture'] == 'amd64', 'base config identity')
        image = build_lifecycle.build(root, lock['base_image'], reports)
        receipt = {'image_id': image, 'synthetic_execution_approved': True, 'security_approved': False, 'implementation_hashes': {name: sha(root / name) for name in REQUIRED}, 'approval_sha256': sha(args.approval)}
        save(reports / 'synthetic-build-receipt.json', receipt)
        # Execute the unchanged CLI in a subprocess, sequentially, once per scenario.
        for scenario in SCENARIOS:
            out = reports / scenario
            code = subprocess.call([sys.executable, '-B', str(root / 'launch_synthetic.py'), '--authorize', 'AUTHORIZE_LINUX_SYNTHETIC_ACCEPTANCE_ONLY', '--scenario', scenario, '--output', str(out), '--build-receipt', str(reports / 'synthetic-build-receipt.json')])
            state = json.loads((out / 'launch-report.json').read_bytes())
            require(state.get('container_termination_confirmed') is True and not state.get('cleanup_errors') and not state.get('persistence_errors'), 'unconfirmed cleanup: ' + scenario)
            if scenario in SCENARIOS[:4]:
                require(code == 0 and state['status'] == 'complete', 'positive scenario failed: ' + scenario)
            else:
                require(code != 0 and state['status'] == 'partial', 'expected failure not observed: ' + scenario)
                error = json.dumps(state.get('primary_error', {}))
                if scenario == 'timeout':
                    require('120-second loading timeout' in error, 'wrong timeout failure')
                elif scenario == 'overflow':
                    require(state.get('output_overflow') is True, 'wrong overflow failure')
                else:
                    require('invented Linux acceptance failure' in (out / 'container-output.log').read_text(), 'wrong forced failure')
            report['scenarios'].append({'scenario': scenario, 'returncode': code, 'expected_observation_matched': True})
        report['status'] = 'observations_complete_pending_independent_review'
    except BaseException:
        report['failure_traceback'] = traceback.format_exc()
        return 1
    finally:
        persisted = persist_report(reports, report)
        if not persisted:
            return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
