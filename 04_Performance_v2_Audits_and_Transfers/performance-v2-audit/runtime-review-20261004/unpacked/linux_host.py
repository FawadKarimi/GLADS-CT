"""Linux host primitives, inert on import; verified grant required to construct.

Source for review only. Tests patch OS/subprocess boundaries. Execution approval,
executable inventory and Linux acceptance remain separate gates.
"""
import hashlib
import os
from pathlib import Path
import selectors
import subprocess
import time
import shutil

def annotate(primary, message):
    """Controller supports Python >=3.10; add_note appeared only in 3.11."""
    method = getattr(primary, 'add_note', None)
    if callable(method):
        method(message)
    else:
        notes = list(getattr(primary, '__notes__', []))
        notes.append(message)
        primary.__notes__ = notes


class LinuxHost:
    is_fake = False

    def __init__(self, grant=None):
        from activation_trust import valid_grant
        if not valid_grant(grant):
            raise RuntimeError('live Linux backend disabled without verified authorization')
        self.grant = grant

    def now(self):
        return time.monotonic()

    def text(self, path):
        with open(path, 'rb') as f:
            raw = f.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('observation too large')
        return raw.decode()

    def put(self, path, value):
        # cgroup pseudo-files only; caller must verify parent identity first.
        with open(path, 'w', encoding='ascii') as f:
            f.write(value)

    def stat(self, path):
        return os.stat(path, follow_symlinks=False)

    def children(self, path):
        return sorted(Path(path).iterdir())

    def mkdir(self, path):
        Path(path).mkdir(mode=0o700, exist_ok=False)

    def assert_no_symlinks(self, path):
        current = Path(path)
        for parent in [current, *current.parents]:
            if parent.is_symlink():
                raise ValueError('symlink path component')

    def filesystem(self, path):
        self.assert_no_symlinks(path)
        st = os.statvfs(path)
        return {'capacity_bytes': st.f_blocks * st.f_frsize,
                'free_bytes': st.f_bavail * st.f_frsize, 'inode_capacity': st.f_files, 'free_inodes': st.f_favail}

    def write_exclusive(self, path, raw):
        from supervisor_evidence import sync_directory
        self.assert_no_symlinks(str(Path(path).parent))
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        sync_directory(Path(path).parent)

    def remove_owned(self, path, inode, directory=False):
        from supervisor_evidence import sync_directory
        self.assert_no_symlinks(path)
        if self.stat(path).st_ino != inode:
            raise ValueError('removal identity changed')
        if directory:
            os.rmdir(path)  # never recursive; busy/nonempty fails
        else:
            os.unlink(path)
        sync_directory(Path(path).parent)
        if os.path.lexists(path):
            raise ValueError('removal unconfirmed')

    def pause(self, seconds):
        if not 0 < seconds <= 1:
            raise ValueError('bounded pause required')
        time.sleep(seconds)

    def free_bytes(self, path):
        return shutil.disk_usage(path).free

    def allocate_exclusive(self, path, capacity):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            # Allocation is a bounded helper in the private workload slice,
            # not an uninterruptible call in the supervising Python process.
            if not getattr(self, 'workload_slice', None):
                raise ValueError('allocation requires workload containment')
            self.command(['fallocate', '--length', str(capacity), path], 60)
            os.fsync(fd)
        finally:
            os.close(fd)

    def digest(self, path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while block := f.read(65536):
                h.update(block)
        return h.hexdigest()

    def process(self, pid):
        prefix = '/proc/' + str(int(pid))
        stat = self.text(prefix + '/stat')
        fields = stat[stat.rindex(')') + 2:].split()
        groups = self.text(prefix + '/cgroup').splitlines()
        group = [x[3:] for x in groups if x.startswith('0::')]
        if len(group) != 1:
            raise ValueError('requires unambiguous unified cgroup')
        exe = os.readlink(prefix + '/exe')
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'executable_sha256': self.digest(prefix + '/exe'),
                'cgroup': group[0], 'executable': exe}

    def writable_paths(self, pid):
        paths = []
        for fd in self.children('/proc/' + str(pid) + '/fd'):
            info = dict(line.split(':', 1) for line in self.text('/proc/' + str(pid) + '/fdinfo/' + fd.name).splitlines() if ':' in line)
            flags = int(info['flags'].strip(), 8)
            if flags & 3 not in (1, 2):
                continue
            target = os.readlink(str(fd))
            if target.startswith(('socket:[', 'pipe:[', 'anon_inode:')) or target in ('/dev/null', '/dev/urandom'):
                continue
            paths.append(target)
        return paths

    def socket_owned(self, pid, path):
        rows = [line.split() for line in self.text('/proc/net/unix').splitlines()[1:]]
        rows = [row for row in rows if len(row) == 8 and row[-1] == path]
        if len(rows) != 1:
            return False
        inode = rows[0][6]
        return any(os.readlink(str(fd)) == 'socket:[' + inode + ']'
                   for fd in self.children('/proc/' + str(pid) + '/fd'))

    def start(self, argv):
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError('explicit argv required')
        if argv[0] in ('fallocate', 'mkfs.ext4'):
            slice_name = getattr(self, 'workload_slice', None)
            if not slice_name:
                raise ValueError('provisioning helper containment unavailable')
            argv = ['systemd-run', '--pipe', '--wait', '--collect', '--slice=' + slice_name,
                    '--property=RuntimeMaxSec=60', '--property=KillMode=control-group', '--', *argv]
        if hasattr(self, 'grant'):
            # No PATH lookup in activated native dispatch, including nested helpers.
            records = self.grant.inventory['executables']
            by_name = {Path(p).name: r['resolved_path'] for p, r in records.items()}
            argv = [records[x]['resolved_path'] if x in records else by_name.get(x, x) for x in argv]
            if argv[0] not in {r['resolved_path'] for r in records.values()}:
                raise ValueError('uninventoried executable')
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
                             env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C'})
        os.set_blocking(p.stdout.fileno(), False)
        return p

    def read(self, process, maximum, timeout):
        if not 0 < maximum <= 65536 or not 0 < timeout <= 1:
            raise ValueError('bounded read required')
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                return None
            return os.read(process.stdout.fileno(), maximum)

    def command(self, argv, timeout=10):
        if not 0 < timeout <= 60:
            raise ValueError('command deadline')
        p = self.start(argv)
        deadline, output = self.now() + timeout, bytearray()
        primary = None
        try:
            while True:
                remaining = deadline - self.now()
                if remaining <= 0:
                    raise TimeoutError('command deadline')
                block = self.read(p, 65536, min(1, remaining))
                if block is None:
                    continue
                if len(output) + len(block) > 1024 * 1024:
                    raise ValueError('command output overflow')
                if not block:
                    code = p.wait(timeout=max(.001, deadline - self.now()))
                    if code:
                        raise RuntimeError('command exit ' + str(code))
                    return bytes(output)
                output.extend(block)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            # Reaping this helper is NOT daemon/subtree termination evidence.
            if p.poll() is None:
                try:
                    p.kill()
                except BaseException as exc:
                    if primary is not None:
                        annotate(primary, 'helper kill failed: ' + repr(exc))
                    else:
                        raise
                try:
                    p.wait(timeout=1)
                except BaseException as exc:
                    if primary is not None:
                        annotate(primary, 'helper reap failed: ' + repr(exc))
                    else:
                        raise


def mount_records(text):
    """Decode mountinfo, including namespace root and octal escapes."""
    def unescape(s):
        for old, new in ((r'\040', ' '), (r'\011', '\t'), (r'\012', '\n'), (r'\134', '\\')):
            s = s.replace(old, new)
        return s
    rows = []
    for line in text.splitlines():
        left, right = line.split(' - ', 1)
        a, b = left.split(), right.split()
        rows.append({'id': int(a[0]), 'device': a[2], 'root': unescape(a[3]),
                     'mount': unescape(a[4]), 'filesystem': b[0], 'source': unescape(b[1])})
    return rows


def resolve_cgroup(mountinfo, membership):
    groups = [x[3:] for x in membership.splitlines() if x.startswith('0::')]
    mounts = [x for x in mount_records(mountinfo) if x['filesystem'] == 'cgroup2']
    if len(groups) != 1 or len(mounts) != 1:
        raise ValueError('ambiguous/missing cgroup v2')
    m, group = mounts[0], groups[0]
    if m['root'] != '/':
        raise ValueError('host root cgroup view required; namespace-relative view refused')
    if not group.startswith('/') or '..' in group.split('/'):
        raise ValueError('invalid membership')
    return m['mount'], group
