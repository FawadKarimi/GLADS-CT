"""Closed, hash-bound launcher contract; no host activation on import."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = '/opt/pv2-native-acceptance'
PYTHON = '/usr/bin/python3'
ROLES = ('bootstrap', 'controller', 'watchdog', 'helper')
SOURCES = ('launcher_binding.py', 'native_launcher.py', 'synthetic_helper.py',
           'watchdog_worker.py', 'linux_host.py', 'linux_lifecycle.py',
           'linux_integration.py', 'systemd_ownership.py', 'supervisor_v2.py',
           'supervisor_evidence.py', 'evidence_export.py', 'activation_bridge.py',
           'activation_trust.py')
EXECUTABLES = ('/usr/bin/python3', '/usr/bin/systemctl', '/usr/bin/systemd-run',
               '/usr/bin/busctl', '/usr/bin/dockerd', '/usr/bin/containerd',
               '/usr/bin/runc', '/usr/bin/fallocate', '/usr/sbin/mkfs.ext4',
               '/usr/sbin/losetup', '/usr/bin/mount', '/usr/bin/umount',
               '/usr/sbin/blkid')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def source_manifest(root):
    root = Path(root)
    records = {}
    for name in SOURCES:
        path = root / name
        need(not path.is_symlink() and path.is_file(), 'source not regular: ' + name)
        raw = path.read_bytes()
        records[ROOT + '/' + name] = {'sha256': sha(raw), 'size_bytes': len(raw)}
    return {'schema': 'pv2_launcher_sources_v1', 'root': ROOT, 'files': records}


def verify_sources(root, manifest):
    need(source_manifest(root) == manifest, 'launcher source mismatch')


def command(role, token, scenario='success'):
    need(role in ROLES, 'unknown role')
    need(re.fullmatch('[a-f0-9]{32}', token), 'token')
    need(scenario in ('success', 'output_overflow', 'timeout'), 'scenario')
    return [PYTHON, '-I', '-S', '-B', ROOT + '/native_launcher.py', '--role', role,
            '--token', token, '--scenario', scenario,
            '--source-manifest', ROOT + '/launcher-sources.json',
            '--inventory', '/run/pv2-authorization/inventory.json',
            '--authorization', '/run/pv2-authorization/authorization.json']


def validate_inventory(inventory):
    need(inventory.get('schema') == 'pv2_native_inventory_v1', 'inventory schema')
    need(inventory.get('architecture') == 'x86_64' and inventory.get('cgroup_version') == 2,
         'native acceptance requires reviewed unified v2 host')
    for field in ('boot_id', 'kernel', 'os_release', 'python_version', 'systemd_version',
                  'security_review_sha256', 'host_evidence_sha256'):
        need(isinstance(inventory.get(field), str) and inventory[field], 'missing ' + field)
    for field in ('security_review_sha256', 'host_evidence_sha256'):
        need(re.fullmatch('[0-9a-f]{64}', inventory[field]), 'invalid ' + field)
    need(set(inventory.get('executables', {})) == set(EXECUTABLES), 'executable coverage')
    for path, entry in inventory['executables'].items():
        need(set(entry) == {'sha256', 'resolved_path', 'version'}, 'executable fields')
        need(re.fullmatch('[0-9a-f]{64}', entry['sha256']), 'executable hash')
        need(PurePosixPath(entry['resolved_path']).is_absolute() and '..' not in PurePosixPath(entry['resolved_path']).parts,
             'executable resolved path')
        need(bool(entry['version']), 'executable version')


def authorize(auth, manifest, inventory, token, now, scenario=None, role=None):
    need(isinstance(token, str) and re.fullmatch('[a-f0-9]{32}', token), 'authorization token syntax')
    validate_inventory(inventory)
    need(manifest.get('schema') == 'pv2_launcher_sources_v1' and manifest.get('root') == ROOT, 'manifest schema/root')
    need(auth.get('schema') == 'pv2_native_authorization_v1', 'authorization schema')
    need(auth.get('scope') == 'A_native_synthetic_only', 'scope')
    need(auth.get('execution_authorized') is True, 'execution not authorized')
    need(auth.get('security_approved') is True, 'native security not approved')
    need(auth.get('token') == token and auth.get('boot_id') == inventory['boot_id'], 'run/boot binding')
    need(auth.get('sources_sha256') == sha(canonical(manifest)), 'sources binding')
    need(auth.get('inventory_sha256') == sha(canonical(inventory)), 'inventory binding')
    start, end = auth.get('not_before'), auth.get('expires')
    need(type(start) is int and type(end) is int and start <= now < end and 0 < end-start <= 3600,
         'authorization validity window')
    need(auth.get('dependency_inspection_authorized') is False and auth.get('pilot_execution_authorized') is False,
         'scope escalation')
    # Mandatory even for legacy call sites: old scenario-agnostic grants fail.
    need(auth.get('scenario') in ('success', 'output_overflow', 'timeout'), 'scenario binding missing')
    if scenario is not None:
        need(auth['scenario'] == scenario, 'scenario substitution')
    need(auth.get('role_chain') == ['bootstrap', 'controller', 'watchdog', 'helper'], 'role chain')
    if role is not None:
        need(role in auth['role_chain'], 'role substitution')


def parser():
    p = argparse.ArgumentParser(description='Native acceptance source; explicit verified authorization required')
    p.add_argument('--role', choices=ROLES, required=True)
    p.add_argument('--token', required=True)
    p.add_argument('--scenario', choices=('success', 'output_overflow', 'timeout'), default='success')
    for name in ('source-manifest', 'inventory', 'authorization'):
        p.add_argument('--' + name, type=Path, required=True)
    return p
