"""Root-owned trust boundary and durable, single-use role admission.

No native activation on import. Tests inject a filesystem/OS boundary. A native
grant is minted only after all checks; is_fake does not mint a native grant.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from copy import deepcopy

from launcher_binding import ROOT, PYTHON, authorize, canonical, need, sha, verify_sources

STATE = '/var/lib/pv2-acceptance'
AUTH = '/run/pv2-authorization'
_SEAL = object()
INITIAL_STORAGE = 12 * 1024**3
STORAGE_METADATA_ALLOWANCE = 16 * 1024**2
VOLUME_BYTES = {'data': 8 * 1024**3, 'evidence': 512 * 1024**2}


class Grant:
    def __init__(self, seal, args, inventory, auth, storage=None):
        need(seal is _SEAL, 'grant issuer')
        self._seal, self.args, self.inventory, self.auth = seal, deepcopy(args), deepcopy(inventory), deepcopy(auth)
        self.storage = deepcopy(storage)


def valid_grant(grant):
    return type(grant) is Grant and grant._seal is _SEAL


def host_allowed(host):
    # Historical injected fake tests remain allowed; native hosts never identify
    # as fake. The public CLI has no host/fake factory argument.
    return (getattr(host, 'is_fake', False) is True and type(host).__module__.startswith('test_')) or valid_grant(getattr(host, 'grant', None))


def no_duplicates(items):
    out = {}
    for key, value in items:
        need(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


class TrustedIO:
    """Actual Linux filesystem boundary; never constructed by mocked tests."""
    def check_path(self, path, directory=False):
        p = Path(path)
        need(p.is_absolute() and '..' not in p.parts, 'absolute trusted path required')
        for component in [*reversed(p.parents), p]:
            s = os.lstat(component)
            need(s.st_uid == 0 and not s.st_mode & 0o022 and not stat.S_ISLNK(s.st_mode),
                 'untrusted ownership/mode/link: ' + str(component))
        need(stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode), 'trusted type')
        return s

    def read(self, path):
        before = self.check_path(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as f:
            current = os.fstat(f.fileno())
            need((before.st_dev, before.st_ino) == (current.st_dev, current.st_ino), 'file replaced')
            raw = f.read(2*1024*1024+1)
            need(len(raw) <= 2*1024*1024, 'metadata size')
        return raw

    def digest(self, path):
        before = self.check_path(path)
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            current = os.fstat(f.fileno())
            need((before.st_dev, before.st_ino) == (current.st_dev, current.st_ino), 'artifact replaced')
            while block := f.read(65536): h.update(block)
            after = os.fstat(f.fileno())
        need((current.st_size, current.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'artifact mutated')
        return h.hexdigest()

    def json(self, path):
        return json.loads(self.read(path), object_pairs_hook=no_duplicates)

    def exclusive(self, path, value):
        from supervisor_evidence import sync_directory
        self.check_path(str(Path(path).parent), directory=True)
        raw = canonical(value)
        need(len(raw) <= 128*1024, 'record bound')
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        sync_directory(Path(path).parent)

    def mkdir(self, path):
        from supervisor_evidence import sync_directory
        self.check_path(str(Path(path).parent), directory=True)
        os.mkdir(path, 0o700)
        sync_directory(Path(path).parent); sync_directory(path)

    def identity(self):
        return {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'python_version': '.'.join(map(str, sys.version_info[:3])),
                'kernel': os.uname().release, 'architecture': os.uname().machine}

    def verify_environment(self, inventory):
        need(sys.platform == 'linux' and os.geteuid() == 0 and sys.version_info >= (3, 10), 'Linux root controller >=3.10 required')
        for k, v in self.identity().items(): need(inventory.get(k) == v, 'host inventory drift: '+k)
        need(inventory.get('kernel_version') == os.uname().version, 'kernel build drift')
        need(inventory.get('os_release_sha256') == self.digest(os.path.realpath('/etc/os-release')), 'OS release bytes')
        need(self.digest(os.path.realpath(sys.executable)) == inventory['executables'][PYTHON]['sha256'], 'running interpreter bytes')
        for path, entry in inventory['executables'].items():
            need(os.path.realpath(path) == entry['resolved_path'], 'executable alias drift: '+path)
        for path in ('/usr/bin/dockerd', '/usr/bin/containerd', '/usr/bin/runc'):
            need(inventory['executables'][path]['resolved_path'] == path, 'daemon canonical-path amendment required')
        # A trusted dedicated filesystem bounds control/journal storage; it is
        # provisioned separately, never mounted or expanded by this launcher.
        self.check_path(STATE, directory=True)
        s, parent = os.stat(STATE), os.stat(str(Path(STATE).parent))
        fs = os.statvfs(STATE)
        need(s.st_dev != parent.st_dev and fs.f_blocks*fs.f_frsize <= 1024**3 and
             fs.f_bavail*fs.f_frsize >= 768*1024**2, 'dedicated <=1GiB state filesystem with >=768MiB free required')

    def mounts(self):
        from linux_host import mount_records
        return mount_records(Path('/proc/self/mountinfo').read_text())

    def storage_observation(self):
        mounts = [m for m in self.mounts() if '/var/tmp' == m['mount'] or '/var/tmp'.startswith(m['mount'].rstrip('/')+'/')]
        need(mounts, 'storage filesystem unavailable')
        fs = os.statvfs('/var/tmp')
        return {'filesystem': max(mounts, key=lambda m: len(m['mount'])),
                'device': os.stat('/var/tmp').st_dev, 'available_bytes': fs.f_bavail*fs.f_frsize}

    def backing_observation(self, path):
        # /var/tmp is intentionally a sticky shared ancestor, not a trusted
        # document directory. Only the fixed run directory/backing are trusted.
        p = Path(path)
        for component in [*reversed(p.parents), p]:
            s = os.lstat(component)
            need(not stat.S_ISLNK(s.st_mode), 'storage symlink')
        parent = os.lstat(p.parent)
        need(parent.st_uid == 0 and stat.S_ISDIR(parent.st_mode) and not parent.st_mode & 0o022,
             'backing parent ownership')
        need(s.st_uid == 0 and stat.S_ISREG(s.st_mode) and not s.st_mode & 0o022, 'backing ownership/type')
        return {'inode': s.st_ino, 'device': s.st_dev, 'size': s.st_size, 'allocated': s.st_blocks*512}

    def loop_backing(self, loop):
        import re
        need(re.fullmatch('/dev/loop[0-9]+', loop), 'storage loop name')
        path = Path('/sys/class/block') / loop.rsplit('/', 1)[-1] / 'loop/backing_file'
        return '/'+path.read_text().strip().lstrip('/')

    def verify_storage(self, args, auth):
        return verify_storage_admission(self, args, auth)


def verify_storage_admission(io, args, auth):
    """Pure policy with explicit read-only observations; no disk-space repair."""
    root = STATE+'/'+args.token
    binding = {'token': args.token, 'boot_id': auth['boot_id'], 'authorization_sha256': sha(canonical(auth)),
               'scenario': args.scenario}
    current = io.storage_observation()
    need(type(current['available_bytes']) is int and current['available_bytes'] >= 0, 'storage free bytes')
    if args.role == 'bootstrap':
        need(current['available_bytes'] >= INITIAL_STORAGE, 'insufficient initial storage capacity')
        return {'schema': 'pv2_storage_admission_v1', 'binding': binding, 'initial': current}
    initial_record = io.json(root+'/storage-initial.json')
    need(initial_record.get('schema') == 'pv2_storage_admission_v1' and initial_record.get('binding') == binding,
         'initial storage binding')
    initial = initial_record['initial']
    need(initial['available_bytes'] >= INITIAL_STORAGE, 'invalid initial storage capacity')
    need(all(current[k] == initial[k] for k in ('filesystem', 'device')), 'storage filesystem changed')
    if args.role == 'controller':
        need(current['available_bytes'] >= INITIAL_STORAGE and
             current['available_bytes'] >= initial['available_bytes']-STORAGE_METADATA_ALLOWANCE,
             'pre-provision storage consumption')
        return initial_record
    need(args.role in ('watchdog', 'helper'), 'storage role')
    capsule = io.json(root+('/watchdog-bootstrap.json' if args.role == 'watchdog' else '/full-state.json'))
    need(capsule['token'] == args.token and capsule['boot_id'] == binding['boot_id'], 'allocation capsule binding')
    # Verify durable observed entries against their ledger bytes and matching
    # intent. A pending service-start intent is allowed: it is not a volume.
    ledger_path = root+'/journal/report-hashes.json'
    ledger_raw = io.read(ledger_path)
    ledger = json.loads(ledger_raw, object_pairs_hook=no_duplicates)
    pending, observed = [], []
    import re
    need(len(ledger) <= 1024, 'storage journal bound')
    for name, rec in sorted(ledger.items()):
        need(re.fullmatch(r'[0-9]{4,}-[a-z-]+\.json', name), 'storage journal name')
        raw = io.read(root+'/journal/'+name)
        need(len(raw) == rec['size_bytes'] and sha(raw) == rec['sha256'], 'storage journal hash')
        value = json.loads(raw, object_pairs_hook=no_duplicates)
        if name.endswith('-provision-intent.json'): pending.append(value)
        elif name.endswith('-provision-observed.json'):
            intent = {k: v for k, v in value.items() if k != 'identity'}; intent['phase'] = 'intent'
            need(intent in pending, 'allocation observation without intent')
            pending.remove(intent); observed.append(value)
    need(io.read(ledger_path) == ledger_raw, 'storage journal changed during validation')
    allocated = 0
    mounts = io.mounts()
    for role, size in VOLUME_BYTES.items():
        backing = '/var/tmp/pv2-'+args.token+'/'+role+'.ext4'
        target = '/var/tmp/pv2-'+args.token+'/'+role
        records = [r for r in capsule['resources'] if r['kind'] == 'backing' and r['requested'] == backing]
        volumes = [r for r in capsule['resources'] if r['kind'] == 'mount' and r['requested'] == target]
        need(len(records) == len(volumes) == 1, 'allocation record coverage')
        for record in records+volumes:
            need(record in observed and observed.count(record) == 1 and record['boot_id'] == binding['boot_id'],
                 'allocation not durably owned')
        recorded, volume = records[0]['identity'], volumes[0]['identity']
        actual = io.backing_observation(backing)
        need(recorded['path'] == backing and recorded['filesystem'] == initial['filesystem'] and
             actual['device'] == initial['device'] and actual['inode'] == recorded['inode'], 'allocation identity mismatch')
        need(volume['backing'] == backing and volume['inode'] == actual['inode'] and
             volume['size'] == actual['size'] == size and volume['allocated'] == actual['allocated'] >= size,
             'allocation size/sparse/mismatch')
        need(volume['mount']['mount'] == target and volume['mount']['source'] == volume['loop'] and
             volume['mount']['filesystem'] == 'ext4' and mounts.count(volume['mount']) == 1 and
             io.loop_backing(volume['loop']) == backing, 'allocation mount/loop mismatch')
        allocated += actual['allocated']
    expected = sum(VOLUME_BYTES.values())
    need(expected <= allocated <= expected+STORAGE_METADATA_ALLOWANCE, 'unexpected backing allocation extent')
    remaining = max(INITIAL_STORAGE-expected-STORAGE_METADATA_ALLOWANCE,
                    initial['available_bytes']-allocated-STORAGE_METADATA_ALLOWANCE)
    # Reobserve after validating allocations; do not use a stale pre-check count.
    after = io.storage_observation()
    need(all(after[k] == initial[k] for k in ('filesystem', 'device')), 'storage filesystem changed after validation')
    need(after['available_bytes'] >= remaining, 'unexpected post-provision storage consumption')
    return {'schema': 'pv2_storage_child_check_v1', 'binding': binding, 'initial': initial,
            'allocated_bytes': allocated, 'remaining_required_bytes': remaining, 'observed': after}


def validate_trust(args, io, now):
    need(str(args.source_manifest) == ROOT+'/launcher-sources.json' and
         str(args.inventory) == AUTH+'/inventory.json' and
         str(args.authorization) == AUTH+'/authorization.json', 'fixed trust paths')
    manifest, inventory, auth = [io.json(str(p)) for p in (args.source_manifest, args.inventory, args.authorization)]
    authorize(auth, manifest, inventory, args.token, now, args.scenario, args.role)
    io.check_path(ROOT, directory=True)
    from launcher_binding import SOURCES
    need(set(manifest.get('files', {})) == {ROOT+'/'+s for s in SOURCES}, 'closed source coverage')
    for path, rec in manifest['files'].items():
        raw = io.read(path)
        need(sha(raw) == rec['sha256'] and len(raw) == rec['size_bytes'], 'source bytes: '+path)
    for path, rec in inventory['executables'].items():
        # Aliases must resolve exactly as independently inventoried; canonical
        # target and all parent directories still pass ownership/mode checks.
        need(io.digest(rec['resolved_path']) == rec['sha256'], 'executable bytes: '+path)
    closure = inventory.get('runtime_files')
    need(isinstance(closure, dict) and closure and inventory.get('runtime_closure_reviewed') is True,
         'reviewed interpreter/stdlib/native runtime closure absent')
    for path, digest in closure.items(): need(io.digest(path) == digest, 'runtime bytes: '+path)
    for field in ('security_review', 'host_evidence'):
        need(io.digest(AUTH+'/'+field+'.json') == inventory[field+'_sha256'], field+' evidence bytes')
    io.verify_environment(inventory)
    storage = io.verify_storage(args, auth)
    return Grant(_SEAL, args, inventory, auth, storage)


class Reservations:
    def __init__(self, io, grant):
        need(valid_grant(grant), 'reservation grant')
        self.io, self.g = io, grant
        self.root = STATE+'/'+grant.args.token
        self.binding = {'token': grant.args.token, 'authorization_sha256': sha(canonical(grant.auth)),
                        'scenario': grant.args.scenario, 'boot_id': grant.inventory['boot_id']}

    def claim(self, role, process):
        need(role == self.g.args.role, 'claim role')
        if role == 'bootstrap':
            self.io.mkdir(self.root)  # EEXIST permanently rejects reused token.
            self.io.exclusive(self.root+'/reservation.json', self.binding)
        else:
            need(self.io.json(self.root+'/reservation.json') == self.binding, 'reservation binding')
            parent = 'bootstrap' if role == 'controller' else 'controller'
            record = self.io.json(self.root+'/'+parent+'.claim.json')
            need(record['binding'] == self.binding, 'parent binding')
            if role == 'helper':
                need(self.io.json(self.root+'/watchdog.claim.json')['binding'] == self.binding, 'watchdog must precede helper')
        self.io.exclusive(self.root+'/'+role+'.claim.json', {'binding': self.binding, 'process': process})
