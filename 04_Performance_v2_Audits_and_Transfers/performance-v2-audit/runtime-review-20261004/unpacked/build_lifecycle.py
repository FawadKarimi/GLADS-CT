"""Named offline installation container, not an unbounded Docker build job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import uuid
import hashlib

LIMIT = 16 * 1024**2

def control(args, timeout=10):
    result = subprocess.run(['docker', '--host', 'unix:///var/run/docker.sock', *args], capture_output=True, timeout=timeout)
    if len(result.stdout) + len(result.stderr) > LIMIT:
        raise RuntimeError('oversize Docker response')
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[:4096])
    return result.stdout.decode()

def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)

def cleanup(name, run=control):
    """Never confuse a killed CLI or a missing response with stopped execution."""
    result = {'termination_confirmed': False, 'removal_confirmed': False, 'errors': []}
    try:
        state = json.loads(run(['inspect', '--format', '{{json .State}}', name]))
        if state.get('Running'):
            run(['kill', '--signal=KILL', name])
    except BaseException:
        result['errors'].append(traceback.format_exc())
    # Inspection/removal are independent attempts even if the kill CLI failed.
    try:
        state = json.loads(run(['inspect', '--format', '{{json .State}}', name]))
        result['state'] = state
        result['termination_confirmed'] = state.get('Running') is False and state.get('Status') in ('exited', 'dead', 'created')
        if not result['termination_confirmed']:
            raise RuntimeError('container termination not confirmed')
    except BaseException:
        result['errors'].append(traceback.format_exc())
    try:
        run(['rm', '-f', name])
    except BaseException:
        result['errors'].append(traceback.format_exc())
    try:
        remaining = run(['ps', '-a', '--filter', 'name=^/' + name + '$', '--format', '{{.Names}}']).strip()
        result['removal_confirmed'] = not remaining
        if remaining:
            raise RuntimeError('container removal not confirmed')
        # A successful exact-name absence query confirms removal; earlier
        # errors remain visible and still prevent a clean acceptance result.
        result['termination_confirmed'] = True
    except BaseException:
        result['errors'].append(traceback.format_exc())
    return result

def build(root, base, reports, run=control, clock=time.monotonic, sleep=time.sleep):
    """Bounded compute installation and stopped-state snapshot; production untested."""
    name = 'pv2-build-' + uuid.uuid4().hex
    tag = name + ':synthetic'
    report = {'status': 'partial', 'container': name, 'image_tag': tag, 'method': 'offline_named_container_install_then_commit', 'cleanup': {}, 'snapshot_termination_confirmed': False}
    write(reports / 'build-resource.json', {'container': name, 'image_tag': tag})
    primary = None
    image = None
    try:
        # No bind mounts, daemon socket or credentials. Writable image layer is
        # necessary to install the locked wheels; runtime read-only guard unchanged.
        install = ('import subprocess; '
                   'subprocess.run(["python","-m","pip","install","--no-index","--find-links=/app/wheels","--require-hashes","--no-deps","-r","/app/requirements.lock"],check=True,timeout=240); '
                   'subprocess.run(["python","-m","pip","check"],check=True,timeout=30)')
        run(['create', '--name', name, '--label', 'performance-v2.build=' + name,
             '--network=none', '--cap-drop=ALL', '--security-opt=no-new-privileges',
             '--cpus=1', '--memory=4g', '--memory-swap=4g', '--pids-limit=16',
             '--ulimit=core=0', '--ulimit=fsize=536870912',
             '--log-driver=local', '--log-opt=max-size=8m', '--log-opt=max-file=1',
             '--log-opt=compress=false', '-e', 'PYTHONDONTWRITEBYTECODE=1',
             '-e', 'CUDA_VISIBLE_DEVICES=', '--workdir=/app', base,
             'python', '-B', '-c', install])
        applied = json.loads(run(['inspect', '--format', '{{json .HostConfig}}', name]))
        report['applied_host_config'] = applied
        if not (applied.get('Memory') == 4 * 1024**3 and applied.get('MemorySwap') == 4 * 1024**3
                and applied.get('NanoCpus') == 1000000000 and applied.get('PidsLimit') == 16
                and applied.get('NetworkMode') == 'none' and applied.get('Privileged') is False
                and 'ALL' in applied.get('CapDrop', [])
                and any(v.startswith('no-new-privileges') for v in applied.get('SecurityOpt', []))):
            raise RuntimeError('build resource/isolation configuration mismatch')
        # Copy while stopped; only preverified files enter the container.
        run(['cp', str(root) + '/.', name + ':/app'], timeout=60)
        run(['start', name])
        deadline = clock() + 300
        while True:
            state = json.loads(run(['inspect', '--format', '{{json .State}}', name]))
            report['last_state'] = state
            if not state.get('Running'):
                if state.get('ExitCode') != 0 or state.get('OOMKilled'):
                    raise RuntimeError('offline installation failed')
                break
            if clock() >= deadline:
                raise TimeoutError('300-second build installation timeout')
            sleep(1)
        # This contains no executing build process: snapshot a confirmed-stopped container.
        # A timed-out commit may still be doing daemon I/O; record uncertainty, never
        # claim cancellation of that operation merely because its CLI ended.
        image = run(['commit', '--change', 'USER 65534:65534', '--change', 'WORKDIR /app',
                     '--change', 'ENV CPU_INSPECTION_CONTAINER=1', name, tag], timeout=60).strip()
        report['snapshot_termination_confirmed'] = True
        if not image.startswith('sha256:') or len(image) != 71:
            raise RuntimeError('invalid immutable image identity')
        report['image_id'] = image
    except BaseException as exc:
        primary = exc
        report['primary_error'] = traceback.format_exc()
    finally:
        try:
            raw = run(['logs', '--tail', '20000', name])
            with (reports / 'build.log').open('x', encoding='utf-8') as stream:
                stream.write(raw[-LIMIT:])
            report['log_tail_only'] = True
        except BaseException:
            report['log_error'] = traceback.format_exc()
        report['cleanup'] = cleanup(name, run)
        if primary is None and report['cleanup']['termination_confirmed'] and report['cleanup']['removal_confirmed'] and not report['cleanup']['errors'] and not report.get('log_error'):
            report['status'] = 'complete'
        try:
            write(reports / 'build-report.json', report)
        except BaseException:
            report['status'] = 'partial'
            report['persistence_error'] = traceback.format_exc()
            if primary is None:
                primary = RuntimeError('build report persistence failed')
        # Preserve primary failure evidence in job logs even if disk writing fails.
        print(json.dumps({'build_report': report}, sort_keys=True), flush=True)
    if primary is not None:
        raise primary
    if report['status'] != 'complete':
        raise RuntimeError('build cleanup/log preservation failed')
    return image

def finalize_report_hashes(reports):
    """Seal all available reports after the independent cleanup step."""
    records = {}
    for path in sorted(reports.rglob('*')):
        if path.is_symlink():
            raise ValueError('report symlink forbidden')
        if path.is_file() and path.name not in ('report-hashes.json', '.report-hashes.tmp'):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    h.update(block)
            records[str(path.relative_to(reports))] = {'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size}
    temp = reports / '.report-hashes.tmp'
    write(temp, records)
    os.replace(temp, reports / 'report-hashes.json')

def cleanup_record(output, run=control):
    """Independent always-step fallback, scoped to the exact recorded resource."""
    reports = Path(output) / 'reports'
    record = reports / 'build-resource.json'
    result = {'status': 'partial', 'scope': 'installation_container_only',
              'daemon_io_termination_verified': False, 'errors': []}
    code = 1
    try:
        reports.mkdir(parents=True, exist_ok=True)
        if not record.exists():
            result['status'] = 'no_build_resource_record'
            # No resource record is not affirmative proof of cleanup.
            result['termination_confirmed'] = False
            code = 0
        else:
            name = json.loads(record.read_text())['container']
            import re
            if not re.fullmatch(r'pv2-build-[0-9a-f]{32}', name):
                raise ValueError('unsafe build resource name')
            previous = reports / 'build-report.json'
            old = json.loads(previous.read_text()) if previous.exists() else {}
            if old.get('cleanup', {}).get('removal_confirmed') is True:
                # Fresh absence check, not just trusting the prior report.
                remaining = run(['ps', '-a', '--filter', 'name=^/' + name + '$', '--format', '{{.Names}}']).strip()
                if remaining:
                    raise RuntimeError('previously removed container still present')
                result.update(status='already_removed', termination_confirmed=True,
                              removal_confirmed=True, evidence='build-report.json')
                code = 0
            else:
                result.update(cleanup(name, run))
                if result['termination_confirmed'] and result['removal_confirmed'] and not result['errors']:
                    result['status'] = 'complete'
                    code = 0
    except BaseException:
        result['errors'].append(traceback.format_exc())
    finally:
        try:
            write(reports / 'final-cleanup.json', result)
        except BaseException:
            result['errors'].append(traceback.format_exc())
            result['status'] = 'partial'
            code = 1
        try:
            finalize_report_hashes(reports)
        except BaseException:
            result['errors'].append(traceback.format_exc())
            code = 1
        print(json.dumps({'final_cleanup': result}, sort_keys=True), flush=True)
    return code

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup-output', type=Path, required=True)
    import sys
    sys.exit(cleanup_record(parser.parse_args().cleanup_output))
