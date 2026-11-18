"""Bounded local evidence copy/read-back; no upload or automatic deletion.

Destination is a newly created directory outside the retained evidence mount.
An acknowledgement proves byte identity locally, not GitHub artifact delivery.
"""
import hashlib
import json
import os
from pathlib import Path
import time


class LocalEvidenceExport:
    def __init__(self, source, destination, clock=time.monotonic, directory_sync=None):
        from supervisor_evidence import sync_directory
        self.source, self.destination = Path(source), Path(destination)
        self.clock = clock
        self.directory_sync = directory_sync or sync_directory

    def export(self, raw_manifest, manifest, maximum_bytes, timeout):
        if raw_manifest != json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode():
            raise ValueError('noncanonical manifest')
        if not 0 < maximum_bytes <= 512 * 1024**2 or not 0 < timeout <= 60:
            raise ValueError('export bounds')
        source = self.source.resolve(strict=True)
        target = self.destination.resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError('export must be independent of evidence source')
        entries = manifest.get('files')
        if not isinstance(entries, dict) or not entries or len(entries) > 1024:
            raise ValueError('export manifest entries')
        if set(entries) != {p.name for p in source.iterdir()}:
            raise ValueError('incomplete export manifest')
        total = len(raw_manifest)
        for name, entry in entries.items():
            if Path(name).name != name or name in ('manifest.json', '', '.', '..') or '/' in name or '\\' in name:
                raise ValueError('unsafe export name')
            if type(entry.get('size_bytes')) is not int or entry['size_bytes'] < 0:
                raise ValueError('export size')
            total += entry['size_bytes']
        if total > maximum_bytes:
            raise ValueError('export aggregate size')
        self.destination.mkdir(exist_ok=False)
        self.directory_sync(self.destination.parent)
        deadline = self.clock() + timeout

        def digest_copy(src, expected_size, dst=None):
            if src.is_symlink() or not src.is_file():
                raise ValueError('unsafe export member')
            h, count = hashlib.sha256(), 0
            with src.open('rb') as inp:
                before = os.fstat(inp.fileno())
                output = dst.open('xb') if dst else None
                try:
                    while True:
                        if self.clock() >= deadline:
                            raise TimeoutError('evidence export deadline')
                        block = inp.read(65536)
                        if not block:
                            break
                        count += len(block)
                        if count > expected_size:
                            raise ValueError('source grew beyond export limit')
                        h.update(block)
                        if output:
                            output.write(block)
                    after = os.fstat(inp.fileno())
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                        raise ValueError('source changed during export')
                    if output:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    if output:
                        output.close()
            return {'size_bytes': count, 'sha256': h.hexdigest()}

        for name, expected in sorted(entries.items()):
            # Separate copy and destination read-back, not just buffered checksum.
            if digest_copy(source / name, expected['size_bytes'], self.destination / name) != expected:
                raise ValueError('source/manifest mismatch; partial destination retained')
            if digest_copy(self.destination / name, expected['size_bytes']) != expected:
                raise ValueError('exported bytes mismatch; partial destination retained')
        with (self.destination / 'manifest.json').open('xb') as f:
            f.write(raw_manifest)
            f.flush()
            os.fsync(f.fileno())
        digest = hashlib.sha256(raw_manifest).hexdigest()
        self.directory_sync(self.destination)
        if hashlib.sha256((self.destination / 'manifest.json').read_bytes()).hexdigest() != digest:
            raise ValueError('manifest read-back mismatch')
        return {'manifest_sha256': digest, 'readback_verified': True,
                'destination': str(self.destination.resolve()), 'bytes': total,
                'scope': 'local_copy_not_remote_upload', 'execution_approved': False}
