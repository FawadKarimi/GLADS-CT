"""Exclusive Linux materialization of reviewed supplied bytes, never images.

Trusted owner/code, no intentional same-UID interference. POSIX descriptor and
ACL checks are not a sandbox or an external anti-rollback record.
"""
import errno
import os
from pathlib import Path
import stat
import sys

from .bindings import FrozenInputs
from .evidence_mapping import relative
from .reuse import canonical, digest, _hash, _json

RESERVED = {'REQUEST.json', 'READY.json', 'FAILURE.json'}
MAX_FILE = 1048576
MAX_TOTAL = 16 * MAX_FILE


def bundle_ledger(bundle):
    if type(bundle) is not dict or not 3 <= len(bundle) <= 953:
        raise ValueError('bounded complete mapping bundle required')
    seen, total, ledger = set(), 0, {}
    for name, raw in sorted(bundle.items()):
        relative(name)
        if (name in RESERVED or any(part.endswith('.pending') for part in name.split('/'))
                or name.casefold() in seen or type(raw) is not bytes or len(raw) > MAX_FILE):
            raise ValueError('invalid mapping member')
        seen.add(name.casefold()); total += len(raw)
        ledger[name] = digest(raw)
    if total > MAX_TOTAL or not {'input-lock.json', 'source-paths.json', 'lineage.json'} <= set(bundle):
        raise ValueError('mapping byte cap or required member violation')
    # File/directory collisions must fail before reserving a destination.
    if any('/'.join(n.split('/')[:i]).casefold() in seen for n in bundle for i in range(1, len(n.split('/')))):
        raise ValueError('mapping file/directory collision')
    return ledger


def mapping_identity(bundle, *, approval_bytes, expected_approval_sha256):
    if not _hash(expected_approval_sha256) or digest(approval_bytes) != expected_approval_sha256:
        raise ValueError('independently reviewed mapping approval hash required')
    ledger = bundle_ledger(bundle)
    inputs = FrozenInputs(bundle['input-lock.json'], expected_sha256=ledger['input-lock.json'])
    manifest, lineage = _json(bundle['source-paths.json']), _json(bundle['lineage.json'])
    if (manifest.get('schema') != 'performance_v2_source_paths_v1'
            or manifest.get('input_lock_sha256') != inputs.sha256 or manifest.get('scope') != inputs.scope
            or lineage.get('schema') != 'pv2_mapping_lineage_v1'
            or lineage.get('input_lock_sha256') != inputs.sha256
            or lineage.get('path_manifest_sha256') != ledger['source-paths.json']
            or lineage.get('production_execution_authorized') is not False):
        raise ValueError('lineage/manifest/input binding mismatch')
    records = manifest.get('records')
    if type(records) is not dict or set(records) != set(inputs.ids('train') + inputs.ids('val')):
        raise ValueError('mapping membership mismatch')
    sidecars, all_paths = set(), set()
    from .input_adapter import supervision_bytes
    for identity, paths in records.items():
        if type(paths) is not dict or set(paths) != {'image', 'supervision'}:
            raise ValueError('exact mapping roles required')
        for name in paths.values():
            relative(name)
            if name.casefold() in all_paths: raise ValueError('aliased mapping path')
            all_paths.add(name.casefold())
        name = paths['supervision']; sidecars.add(name)
        raw = bundle.get(name)
        if raw != supervision_bytes(inputs.record(identity)) or digest(raw) != inputs.record(identity).supervision_sha256:
            raise ValueError('exact canonical sidecar required')
    if set(bundle) != sidecars | {'input-lock.json', 'source-paths.json', 'lineage.json'}:
        raise ValueError('missing or extra bundle member; images must not be materialized')
    approval = _json(approval_bytes)
    expected = {'schema': 'pv2_mapping_approval_v1', 'scope': inputs.scope,
                'bundle_sha256': digest(canonical(ledger)), 'lineage_sha256': ledger['lineage.json'],
                'input_lock_sha256': inputs.sha256, 'path_manifest_sha256': ledger['source-paths.json'],
                'source_root': lineage.get('source_root'), 'sidecar_root': approval.get('sidecar_root'),
                'mapping_review_approved': True, 'pilot_execution_authorized': False}
    if approval != expected or any(type(expected[k]) is not str or not expected[k] for k in ('source_root', 'sidecar_root')):
        raise ValueError('exact approved bundle, roots and lineage required')
    return dict(expected, approval_sha256=expected_approval_sha256)


def _acl_free(fd):
    for name in ('system.posix_acl_access', 'system.posix_acl_default'):
        try:
            value = os.getxattr(fd, name)
        except OSError as exc:
            if exc.errno not in (errno.ENODATA, errno.ENOTSUP): raise
        else:
            if value: raise ValueError('extended/default ACL requires separate review')


def _directory(fd, private=True):
    s = os.fstat(fd)
    if (not stat.S_ISDIR(s.st_mode) or (private and (s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o700))):
        raise ValueError('private owner-only directory required')
    _acl_free(fd)


def _open_root(path):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('native Linux descriptor/ACL materialization required')
    path = Path(path).absolute()
    if str(path) != os.path.normpath(str(path)):
        raise ValueError('normalized root required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for i, part in enumerate(path.parts[1:]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd); fd = child
            _directory(fd, private=i == len(path.parts) - 2)
            s = os.fstat(fd)
            if s.st_uid not in (0, os.getuid()): raise ValueError('untrusted ancestor owner')
            if s.st_mode & 0o022 and not (s.st_uid == 0 and s.st_mode & stat.S_ISVTX):
                raise ValueError('writable ancestor; only root-owned sticky temp ancestor allowed')
        _directory(fd)
        return fd
    except BaseException:
        os.close(fd); raise


def _parent(fd, name, create=False):
    relative(name)
    current = os.dup(fd)
    try:
        for part in name.split('/')[:-1]:
            if create:
                try: os.mkdir(part, 0o700, dir_fd=current); os.fsync(current)
                except FileExistsError: pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current); current = child; _directory(current)
        return current, name.split('/')[-1]
    except BaseException:
        os.close(current); raise


def _read(fd, name):
    parent, leaf = _parent(fd, name)
    opened = None
    try:
        before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE:
            raise ValueError('bounded regular member required before open')
        opened = os.open(leaf, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        s = os.fstat(opened)
        if (not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o600
                or (s.st_dev, s.st_ino) != (before.st_dev, before.st_ino) or s.st_size > MAX_FILE or s.st_nlink != 1):
            raise ValueError('private stable unaliased regular member required')
        _acl_free(opened)
        with os.fdopen(opened, 'rb', closefd=False) as stream: raw = stream.read(MAX_FILE + 1)
        after = os.fstat(opened)
        if len(raw) != s.st_size or (s.st_size, s.st_mtime_ns, s.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('member changed while reading')
        return raw
    finally:
        if opened is not None: os.close(opened)
        os.close(parent)


def _write(fd, name, raw, boundary):
    parent, leaf = _parent(fd, name, create=True)
    opened = None
    try:
        opened = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        s = os.fstat(opened)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o600:
            raise ValueError('exclusive private output required')
        _acl_free(opened); boundary('opened:' + name)
        view = memoryview(raw)
        while view:
            count = os.write(opened, view)
            if count <= 0: raise OSError('short write')
            view = view[count:]
        boundary('written:' + name); os.fsync(opened); boundary('synced:' + name)
        os.fsync(parent)
    finally:
        if opened is not None: os.close(opened)
        os.close(parent)


def materialize(parent, name, bundle, *, expected_bundle_sha256, boundary=lambda _: None):
    """Reserve exclusively, write/request fsync, retain failures; never retry.

    boundary is fault injection/observation, never an approval override. A READY
    marker alone is insufficient: admission rechecks the complete member set.
    """
    relative(name)
    if '/' in name: raise ValueError('one exclusive destination component required')
    ledger = bundle_ledger(bundle)
    if digest(canonical(ledger)) != expected_bundle_sha256: raise ValueError('reviewed bundle digest mismatch')
    parent_fd = _open_root(parent); fd = None
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)  # any existing object stops; never chmod/reuse
        os.fsync(parent_fd)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        _directory(fd)
        try:
            request = canonical({'schema': 'pv2_materialization_v1', 'files': ledger,
                                 'bundle_sha256': expected_bundle_sha256, 'production_execution_authorized': False})
            _write(fd, 'REQUEST.json', request, boundary)
            for member, raw in sorted(bundle.items()): _write(fd, member, raw, boundary)
            for member, wanted in ledger.items():
                if digest(_read(fd, member)) != wanted: raise ValueError('post-write hash mismatch')
            boundary('before_ready')
            _write(fd, 'READY.json', canonical({'request_sha256': digest(request), 'bundle_sha256': expected_bundle_sha256}), boundary)
            os.fsync(fd); boundary('after_ready')
        except BaseException as exc:
            # Best effort only: a full disk or SIGKILL may prevent this record.
            # A partial directory is never removed or retried automatically.
            try: _write(fd, 'FAILURE.json', canonical({'error_type': type(exc).__name__, 'retry': False}), lambda _: None)
            except BaseException: pass
            raise
    finally:
        if fd is not None: os.close(fd)
        os.close(parent_fd)
    return Path(parent) / name


def read_materialized(path, *, expected_bundle_sha256):
    fd = _open_root(path)
    try:
        request_raw = _read(fd, 'REQUEST.json'); request = _json(request_raw)
        ledger = request.get('files')
        if (request.get('schema') != 'pv2_materialization_v1' or request.get('production_execution_authorized') is not False
                or type(ledger) is not dict or len(ledger) > 953
                or digest(canonical(ledger)) != expected_bundle_sha256
                or request.get('bundle_sha256') != expected_bundle_sha256):
            raise ValueError('materialization binding mismatch')
        if _json(_read(fd, 'READY.json')) != {'request_sha256': digest(request_raw), 'bundle_sha256': expected_bundle_sha256}:
            raise ValueError('complete materialization marker required')
        expected_files = set(ledger) | {'REQUEST.json', 'READY.json'}
        expected_dirs = {'/'.join(n.split('/')[:i]) for n in ledger for i in range(1, len(n.split('/')))}
        def walk(current, prefix=''):
            for name in os.listdir(current):
                rel = prefix + name; s = os.stat(name, dir_fd=current, follow_symlinks=False)
                if stat.S_ISDIR(s.st_mode) and rel in expected_dirs:
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
                    try: _directory(child); walk(child, rel + '/')
                    finally: os.close(child)
                elif not stat.S_ISREG(s.st_mode) or rel not in expected_files:
                    raise ValueError('partial, failed or unexpected materialization content')
        walk(fd)
        bundle = {n: _read(fd, n) for n in ledger}
        if bundle_ledger(bundle) != ledger: raise ValueError('materialized member hash mismatch')
        return bundle
    finally: os.close(fd)


def execution_binding(source):
    if hasattr(source, 'mapping_execution_binding'):
        return source.mapping_execution_binding()
    if source.inputs.scope != 'synthetic': raise ValueError('production requires approved complete mapping materialization')
    return {'scope': 'legacy_synthetic_fixture_only', 'source_root': str(source.root),
            'sidecar_root': str(source.supervision_root), 'input_lock_sha256': source.inputs.sha256,
            'path_manifest_sha256': source.sha256}


def open_mapping_source(image_root, sidecar_root, *, approval_bytes, expected_approval_sha256):
    from .input_adapter import FrozenFileSource
    # Verify the approval before using any of its paths/digests.
    if digest(approval_bytes) != expected_approval_sha256: raise ValueError('approval hash mismatch')
    approval = _json(approval_bytes)
    if (approval.get('source_root') != str(Path(image_root).absolute())
            or approval.get('sidecar_root') != str(Path(sidecar_root).absolute())):
        raise ValueError('approved roots differ before any materialized reads')
    class MappingSource(FrozenFileSource):
        def mapping_execution_binding(self):
            if str(self.root) != approval['source_root'] or str(self.supervision_root) != approval['sidecar_root']:
                raise ValueError('approved root substitution before reads')
            current = read_materialized(self.supervision_root, expected_bundle_sha256=approval['bundle_sha256'])
            identity = mapping_identity(current, approval_bytes=approval_bytes, expected_approval_sha256=expected_approval_sha256)
            if identity['source_root'] != str(self.root) or identity['sidecar_root'] != str(self.supervision_root):
                raise ValueError('approved root substitution')
            if identity['input_lock_sha256'] != self.inputs.sha256 or identity['path_manifest_sha256'] != self.sha256:
                raise ValueError('materialized input binding drift')
            return identity
    bundle = read_materialized(sidecar_root, expected_bundle_sha256=approval['bundle_sha256'])
    identity = mapping_identity(bundle, approval_bytes=approval_bytes, expected_approval_sha256=expected_approval_sha256)
    inputs = FrozenInputs(bundle['input-lock.json'], expected_sha256=identity['input_lock_sha256'])
    source = MappingSource(image_root, inputs, bundle['source-paths.json'],
                           expected_sha256=identity['path_manifest_sha256'], supervision_root=sidecar_root)
    source.mapping_execution_binding()
    return source
