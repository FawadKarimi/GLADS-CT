"""Concrete role interfaces; native bridge requires explicit verified authorization.

The injectable legacy dispatcher is solely a mocked integration seam, never
selected by CLI arguments. CLI handlers use validated native capabilities.
"""
import json
import hashlib
import os
import stat
from pathlib import Path
import sys
import time

def trusted_bootstrap():
    """Stdlib-only trust check BEFORE importing any adjacent source.

    Trust anchors are owner-delivered, root-owned authorization and reviewed
    interpreter/runtime. Root compromise is outside the threat model.
    """
    root = Path('/opt/pv2-native-acceptance')
    if Path(__file__).absolute() != root/'native_launcher.py':
        raise ValueError('fixed launcher path required')
    if sys.platform != 'linux' or os.geteuid() != 0 or not sys.flags.isolated or not sys.flags.no_site:
        raise ValueError('root Linux python -I -S required')
    def read(path):
        for p in [*reversed(path.parents), path]:
            s = os.lstat(p)
            if s.st_uid != 0 or s.st_mode & 0o022 or stat.S_ISLNK(s.st_mode):
                raise ValueError('untrusted bootstrap path')
        if not stat.S_ISREG(s.st_mode) or s.st_size > 2*1024*1024:
            raise ValueError('bootstrap file type/size')
        return path.read_bytes()
    def pairs(rows):
        out = {}
        for k, v in rows:
            if k in out: raise ValueError('duplicate bootstrap key')
            out[k] = v
        return out
    auth = json.loads(read(Path('/run/pv2-authorization/authorization.json')), object_pairs_hook=pairs)
    manifest = json.loads(read(root/'launcher-sources.json'), object_pairs_hook=pairs)
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    if auth.get('execution_authorized') is not True or auth.get('sources_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('bootstrap source authorization')
    # Closed bootstrap list is deliberately duplicated here: importing the list
    # from an adjacent module before authenticating it would defeat this gate.
    expected = {'launcher_binding.py', 'native_launcher.py', 'synthetic_helper.py',
                'watchdog_worker.py', 'linux_host.py', 'linux_lifecycle.py',
                'linux_integration.py', 'systemd_ownership.py', 'supervisor_v2.py',
                'supervisor_evidence.py', 'evidence_export.py', 'activation_bridge.py',
                'activation_trust.py'}
    if set(manifest.get('files', {})) != {str(root/name) for name in expected}:
        raise ValueError('closed bootstrap source coverage')
    for name, entry in manifest['files'].items():
        path = Path(name)
        if path.parent != root or path.suffix != '.py': raise ValueError('bootstrap source path')
        data = read(path)
        if len(data) != entry['size_bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('bootstrap source bytes')
    sys.path.insert(0, str(root))


def dispatch_mocked(args, manifest, inventory, auth, now, backend):
    from launcher_binding import authorize, command
    authorize(auth, manifest, inventory, args.token, now, args.scenario, args.role)
    if getattr(backend, 'is_fake', False) is not True:
        raise RuntimeError('live launcher binding disabled')
    argv = command(args.role, args.token, args.scenario)
    return backend.launch_role(args.role, argv, inventory, args.scenario)


def main():
    try:
        # This review candidate intentionally cannot activate any native role,
        # even if an older authorization document is present on a host.
        raise RuntimeError('native supervisor disabled: supported-runtime candidate lacks artifact, native inventory, security and lifecycle acceptance')
        trusted_bootstrap()
        from launcher_binding import parser
        from activation_trust import TrustedIO, validate_trust
        from activation_bridge import Runtime, dispatch
        from linux_host import LinuxHost
        args = parser().parse_args()
        io = TrustedIO()
        grant = validate_trust(args, io, int(time.time()))
        result = dispatch(Runtime(grant, io, LinuxHost(grant)))
        if args.role == 'controller':
            return 0 if result.get('state') == 'VERIFIED_NATIVE' and result.get('evidence_finalized') is True else 1
        return 0 if result.get('complete') else 1
    except Exception as exc:
        print(json.dumps({'schema': 'pv2_launcher_rejection_v1', 'complete': False,
                          'error': str(exc), 'host_operations': 'consult journal; not inferred from failure',
                          'stage_b': 'paused', 'pilot_execution_authorized': False}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
