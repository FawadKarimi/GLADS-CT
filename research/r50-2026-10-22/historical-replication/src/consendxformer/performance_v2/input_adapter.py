"""Bound TRAIN/VALIDATION byte adapters. No discovery, TEST reader or launcher.

The reviewed mapping must be derived from archived evidence, not from a new
directory scan. Canonical supervision sidecars are a NEW interface; their hashes
must not be confused with original CSV/ledger hashes. No production mapping is
supplied by this module. Filesystem controls assume trusted code and owner UID.
"""
import io
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys

import numpy as np
import torch

from .reuse import _json, canonical, digest

MAX_SOURCE_BYTES = 1024 * 1024


def supervision_bytes(record):
    row = record.as_dict()
    for field in ('image_sha256', 'supervision_sha256'):
        del row[field]
    return canonical({'schema': 'performance_v2_supervision_sidecar_v1', 'record': row})


class FrozenFileSource:
    """Explicit allowlist, hash-before-decode, bounded no-follow regular reads.

    Construction does not open image/supervision files. It authenticates the
    supplied mapping only; original-evidence mapping review remains required.
    Symlink/identity checks are accidental substitution controls, not a sandbox.
    """
    def __init__(self, root, inputs, manifest_bytes, *, expected_sha256, supervision_root=None):
        if digest(manifest_bytes) != expected_sha256:
            raise ValueError('source manifest digest mismatch before parsing')
        obj = _json(manifest_bytes)
        if (set(obj) != {'schema', 'input_lock_sha256', 'scope', 'records'}
                or obj['schema'] != 'performance_v2_source_paths_v1'
                or obj['input_lock_sha256'] != inputs.sha256 or obj['scope'] != inputs.scope
                or type(obj['records']) is not dict
                or set(obj['records']) != set(inputs.ids('train') + inputs.ids('val'))):
            raise ValueError('exact frozen TRAIN/VALIDATION mapping required')
        self.root = Path(root).absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)) or not self.root.is_dir():
            raise ValueError('explicit nonsymlink source root required')
        # Normalize Windows short-name spelling after checking original parents.
        self.root = self.root.resolve()
        self.supervision_root = Path(supervision_root).absolute() if supervision_root is not None else self.root
        if (any(p.is_symlink() for p in (self.supervision_root, *self.supervision_root.parents))
                or not self.supervision_root.is_dir()):
            raise ValueError('explicit nonsymlink supervision root required')
        self.supervision_root = self.supervision_root.resolve()
        self._roots = (self.root, self.supervision_root)
        self.inputs, self.sha256 = inputs, expected_sha256
        self.paths = {}
        used = set()
        for identity, fields in obj['records'].items():
            if type(fields) is not dict or set(fields) != {'image', 'supervision'}:
                raise ValueError('exact source roles required')
            self.paths[identity] = {}
            for role, name in fields.items():
                if (type(name) is not str or not name or '\\' in name or ':' in name
                        or name != PurePosixPath(name).as_posix()
                        or PurePosixPath(name).is_absolute()
                        or any(x in ('', '.', '..') for x in name.split('/'))
                        or name.casefold() in used):
                    raise ValueError('unsafe or duplicate source path')
                used.add(name.casefold())
                self.paths[identity][role] = name
        self._mapping_sha256 = digest(canonical(self.paths))

    def read(self, identity, role):
        if (self.root, self.supervision_root) != self._roots:
            raise ValueError('source roots changed after construction')
        if digest(canonical(self.paths)) != self._mapping_sha256:
            raise ValueError('source mapping changed after review')
        if identity not in self.paths or role not in ('image', 'supervision'):
            raise ValueError('unbound source read')
        path = (self.root if role == 'image' else self.supervision_root) / self.paths[identity][role]
        if any(p.is_symlink() for p in (path, *path.parents)) or path.resolve() != path:
            raise ValueError('source symlink or path substitution')
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
            raise ValueError('bounded regular source required before opening')
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
        if sys.platform.startswith('linux'):
            # A regular path can be replaced by a FIFO after lstat. Never wait
            # for a writer; reject the opened descriptor before any read.
            flags |= os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_SOURCE_BYTES
                    or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)):
                raise ValueError('bounded stable regular source required')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(MAX_SOURCE_BYTES + 1)
                after = os.fstat(fd)
        finally:
            os.close(fd)
        if (len(raw) > MAX_SOURCE_BYTES or len(raw) != opened.st_size
                or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError('source changed during read')
        wanted = getattr(self.inputs.record(identity), role + '_sha256')
        if digest(raw) != wanted:
            raise ValueError('source digest mismatch before decoding')
        return raw


def decode_grayscale(raw):
    """Reviewed NPY branch, header-bounded before allocation; no pickle/NPZ."""
    if type(raw) is not bytes or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('bounded NPY bytes required')
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError('unreviewed NPY version')
    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = header(stream, max_header_size=4096)
    if (shape != (64, 64) or dtype.kind not in 'uif' or dtype.itemsize not in (1, 2, 4, 8)
            or dtype.hasobject or len(raw) != stream.tell() + 4096 * dtype.itemsize):
        raise ValueError('exact bounded numeric grayscale array required')
    image = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError('finite frozen grayscale range required')
    # Preserve the archived decoder operation even though it is a 64->64 resize.
    import cv2
    image = cv2.resize(np.asarray(image, dtype=np.float32), (64, 64), interpolation=cv2.INTER_LINEAR)
    if image.max() > 1:
        image = image / 255.0
    return torch.from_numpy(image.astype(np.float32, copy=True)).unsqueeze(0)


def transform(image, *, trace=None):
    from torchvision.transforms import functional as tf, InterpolationMode
    if (image.device.type != 'cpu' or image.dtype != torch.float32 or image.shape != (1, 64, 64)
            or image.requires_grad or not torch.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError('decoded CPU grayscale required')
    x = torch.nn.functional.interpolate(image[None], size=(224, 224), mode='bilinear',
                                        align_corners=False, antialias=True)[0]
    if trace is not None:
        if (set(trace) != {'id', 'horizontal_flip', 'vertical_flip', 'clockwise_angle',
                          'translate_fraction', 'translate_pixels'}
                or type(trace['horizontal_flip']) is not bool or type(trace['vertical_flip']) is not bool
                or type(trace['clockwise_angle']) not in (float, int)
                or not math.isfinite(trace['clockwise_angle']) or abs(trace['clockwise_angle']) > 10
                or len(trace['translate_fraction']) != 2
                or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > .08
                       for v in trace['translate_fraction'])
                or trace['translate_pixels'] != [int(round(v * 224)) for v in trace['translate_fraction']]):
            raise ValueError('unreviewed augmentation trace')
        if trace['horizontal_flip']:
            x = tf.hflip(x)
        if trace['vertical_flip']:
            x = tf.vflip(x)
        x = tf.affine(x, angle=trace['clockwise_angle'], translate=trace['translate_pixels'],
                      scale=1, shear=[0, 0], interpolation=InterpolationMode.BILINEAR,
                      fill=0, center=[112, 112]).clamp(0, 1)
    x = x.repeat(3, 1, 1)
    return tf.normalize(x, mean=[.485, .456, .406], std=[.229, .224, .225])


class BoundInputAdapter:
    def __init__(self, inputs, source, *, expected_transform_sha256):
        if (source.inputs.sha256 != inputs.sha256 or inputs.transform_sha256 != expected_transform_sha256):
            raise ValueError('input/transform binding mismatch')
        self.inputs, self.source = inputs, source

    def batch(self, ids, *, split, traces=None):
        ids = tuple(ids)
        supervision = self.inputs.supervision(ids, split=split)  # reject TEST before any reads
        if len(ids) > 16 or ((split == 'train') != (traces is not None)):
            raise ValueError('batch16, train trace, and unaugmented validation required')
        if traces is not None and (len(traces) != len(ids) or [t['id'] for t in traces] != list(ids)):
            raise ValueError('trace identities mismatch')
        tensors, hashes = [], {}
        for index, identity in enumerate(ids):
            row = self.inputs.record(identity)
            sidecar = self.source.read(identity, 'supervision')
            if digest(sidecar) != row.supervision_sha256 or sidecar != supervision_bytes(row):
                raise ValueError('authenticated supervision differs from frozen record')
            raw = self.source.read(identity, 'image')
            hashes[identity] = self.inputs.verify_image_bytes(identity, raw)
            tensors.append(transform(decode_grayscale(raw), trace=None if traces is None else traces[index]))
        self.inputs.validate_batch(ids, split=split, image_hashes=hashes, **supervision)
        return dict(supervision, ids=ids, images=torch.stack(tensors), image_hashes=hashes)
