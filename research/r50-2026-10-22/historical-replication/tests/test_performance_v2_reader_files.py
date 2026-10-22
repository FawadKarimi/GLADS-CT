"""Invented files/FIFOs only. Native Linux cases run under an alarm as well."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_performance_v2_runner as fixtures
from consendxformer.performance_v2 import input_adapter as reader


@contextmanager
def alarm():
    import signal
    def expired(*_):
        raise AssertionError('synthetic FIFO operation exceeded five seconds')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class ReaderFileTests(unittest.TestCase):
    def setUp(self):
        fixtures.RunnerTests.setUp(self)
        self.identity = self.inputs.ids('val')[0]
        self.path = self.source.root / self.source.paths[self.identity]['image']

    def read(self):
        return self.source.read(self.identity, 'image')

    def test_all_nonregular_modes_reject_before_open(self):
        original = Path.lstat
        for mode in (stat.S_IFIFO, stat.S_IFDIR, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK):
            def fake(path, *a, **kw):
                if path == self.path:
                    return SimpleNamespace(st_mode=mode | 0o600, st_size=0)
                return original(path, *a, **kw)
            with self.subTest(mode=mode), patch.object(Path, 'lstat', fake), \
                 patch.object(reader.os, 'open', side_effect=AssertionError('must not open')):
                with self.assertRaisesRegex(ValueError, 'before opening'):
                    self.read()

    def test_real_directory_rejects_before_open(self):
        self.path.unlink(); self.path.mkdir()
        with patch.object(reader.os, 'open', side_effect=AssertionError('directory opened')):
            with self.assertRaisesRegex(ValueError, 'before opening'): self.read()

    def test_oversized_regular_file_rejects_before_open(self):
        with self.path.open('wb') as f: f.truncate(reader.MAX_SOURCE_BYTES + 1)
        with patch.object(reader.os, 'open', side_effect=AssertionError('oversize opened')):
            with self.assertRaisesRegex(ValueError, 'before opening'): self.read()

    def test_descriptor_nonregular_or_identity_change_rejects_before_fdopen_and_closes(self):
        original_open = reader.os.open
        for wrong_type in (True, False):
            opened = []
            def capture(*a, **kw):
                fd = original_open(*a, **kw); opened.append(fd); return fd
            original = self.path.stat()
            fake = SimpleNamespace(st_mode=stat.S_IFIFO if wrong_type else original.st_mode,
                                   st_size=original.st_size, st_dev=original.st_dev,
                                   st_ino=original.st_ino if wrong_type else original.st_ino + 1)
            with self.subTest(wrong_type=wrong_type), patch.object(reader.os, 'open', capture), \
                 patch.object(reader.os, 'fstat', return_value=fake), \
                 patch.object(reader.os, 'fdopen', side_effect=AssertionError('must not wrap/read')):
                with self.assertRaisesRegex(ValueError, 'stable regular'): self.read()
            with self.assertRaises(OSError): os.fstat(opened[0])

    def test_fstat_failure_closes_descriptor(self):
        original_open = reader.os.open; opened = []
        def capture(*a, **kw):
            fd = original_open(*a, **kw); opened.append(fd); return fd
        with patch.object(reader.os, 'open', capture), patch.object(reader.os, 'fstat', side_effect=OSError('invented fstat failure')):
            with self.assertRaises(OSError): self.read()
        with self.assertRaises(OSError): os.fstat(opened[0])

    def test_regular_file_preserves_exact_bytes_and_digest_rejection(self):
        expected = self.path.read_bytes()
        self.assertEqual(self.read(), expected)
        self.path.write_bytes(expected[:-1] + bytes([expected[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'digest mismatch'): self.read()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'actual FIFO requires native Linux')
    def test_linux_fifo_without_writer_rejected_before_open(self):
        self.path.unlink(); os.mkfifo(self.path, 0o600)
        with alarm(), patch.object(reader.os, 'open', side_effect=AssertionError('FIFO opened')):
            with self.assertRaisesRegex(ValueError, 'before opening'): self.read()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'actual FIFO substitution requires native Linux')
    def test_linux_regular_to_fifo_race_uses_nonblock_and_rejects_descriptor(self):
        original_open = reader.os.open; opened = []
        def substitute(path, flags, *a, **kw):
            self.assertTrue(flags & os.O_NONBLOCK)
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertTrue(flags & os.O_CLOEXEC)
            self.path.unlink(); os.mkfifo(self.path, 0o600)
            fd = original_open(path, flags, *a, **kw); opened.append(fd); return fd
        with alarm(), patch.object(reader.os, 'open', substitute), \
             patch.object(reader.os, 'fdopen', side_effect=AssertionError('FIFO read attempted')):
            with self.assertRaisesRegex(ValueError, 'stable regular'): self.read()
        with self.assertRaises(OSError): os.fstat(opened[0])


if __name__ == '__main__': unittest.main()
