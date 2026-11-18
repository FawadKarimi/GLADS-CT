"""Invented allocations and filesystem observations only; no native operations."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from activation_trust import (INITIAL_STORAGE, STORAGE_METADATA_ALLOWANCE, VOLUME_BYTES,
                              STATE, TrustedIO, verify_storage_admission)
from launcher_binding import canonical, sha
from test_activation_bridge import IO, setup


class StorageIO(IO):
    def __init__(self):
        super().__init__()
        self.free = INITIAL_STORAGE
        self.fs = {'id': 1, 'device': '8:1', 'root': '/', 'mount': '/', 'filesystem': 'ext4', 'source': '/dev/invented'}
        self.backings = {}; self.mounted = []; self.loops = {}
    def storage_observation(self): return {'filesystem': deepcopy(self.fs), 'device': 2049, 'available_bytes': self.free}
    def backing_observation(self, path): return deepcopy(self.backings[path])
    def mounts(self): return deepcopy(self.mounted)
    def loop_backing(self, loop): return self.loops[loop]


def prepared(initial=INITIAL_STORAGE):
    args, _ = setup()
    auth = {'boot_id': 'invented-boot', 'scenario': 'success'}  # policy fixture, not an execution authorization
    io = StorageIO(); io.free = initial
    root = STATE+'/'+args.token
    first = verify_storage_admission(io, args, auth)
    io.files[root+'/storage-initial.json'] = canonical(first)
    args.role = 'controller'
    verify_storage_admission(io, args, auth)
    resources = []
    for index, (role, size) in enumerate(VOLUME_BYTES.items()):
        backing = '/var/tmp/pv2-'+args.token+'/'+role+'.ext4'
        target = '/var/tmp/pv2-'+args.token+'/'+role
        inode, loop = 50+index, '/dev/loop'+str(10+index)
        mount = {'id': 20+index, 'device': '7:'+str(10+index), 'root': '/', 'mount': target, 'filesystem': 'ext4', 'source': loop}
        io.backings[backing] = {'inode': inode, 'device': 2049, 'size': size, 'allocated': size}
        io.mounted.append(mount); io.loops[loop] = backing
        for kind, requested, identity in (
            ('backing', backing, {'path': backing, 'inode': inode, 'filesystem': deepcopy(io.fs), 'disposition': 'host-entry'}),
            ('mount', target, {'backing': backing, 'inode': inode, 'size': size, 'allocated': size, 'loop': loop, 'mount': mount})):
            resources.append({'kind': kind, 'requested': requested, 'boot_id': auth['boot_id'], 'phase': 'observed', 'identity': identity})
    ledger = {}; serial = 0
    for resource in resources:
        intent = {k: v for k, v in resource.items() if k != 'identity'}; intent['phase'] = 'intent'
        for suffix, value in (('intent', intent), ('observed', resource)):
            serial += 1; name = f'{serial:04d}-provision-{suffix}.json'; raw = canonical(value)
            io.files[root+'/journal/'+name] = raw
            ledger[name] = {'sha256': sha(raw), 'size_bytes': len(raw)}
    io.files[root+'/journal/report-hashes.json'] = canonical(ledger)
    capsule = {'token': args.token, 'boot_id': auth['boot_id'], 'resources': resources}
    for name in ('watchdog-bootstrap.json', 'full-state.json'): io.files[root+'/'+name] = canonical(capsule)
    io.free -= sum(VOLUME_BYTES.values())
    args.role = 'watchdog'
    return args, auth, io


class StorageAdmissionTests(unittest.TestCase):
    def test_initial_controller_and_both_children_after_allocations(self):
        args, auth, io = prepared()
        self.assertEqual(io.free, 7*1024**3//2)
        for role in ('watchdog', 'helper'):
            args.role = role
            check = verify_storage_admission(io, args, auth)
            self.assertEqual(check['allocated_bytes'], sum(VOLUME_BYTES.values()))
            self.assertEqual(check['remaining_required_bytes'], io.free-STORAGE_METADATA_ALLOWANCE)
            self.assertLess(check['observed']['available_bytes'], INITIAL_STORAGE)

    def test_insufficient_initial_capacity(self):
        args, _ = setup(); io = StorageIO(); io.free = INITIAL_STORAGE-1
        with self.assertRaisesRegex(ValueError, 'insufficient initial'): verify_storage_admission(io, args, {'boot_id': 'invented'}, )

    def test_unexpected_consumption_even_with_large_initial_disk(self):
        for initial in (INITIAL_STORAGE, 100*1024**3):
            args, auth, io = prepared(initial)
            io.free -= STORAGE_METADATA_ALLOWANCE+1
            with self.assertRaisesRegex(ValueError, 'unexpected post-provision'): verify_storage_admission(io, args, auth)

    def test_metadata_allowance_boundary(self):
        args, auth, io = prepared(); io.free -= STORAGE_METADATA_ALLOWANCE
        verify_storage_admission(io, args, auth)
        io.free -= 1
        with self.assertRaises(ValueError): verify_storage_admission(io, args, auth)

    def test_allocation_inode_sparse_size_device_and_loop_mismatch(self):
        for field, value in (('inode', 999), ('allocated', 1), ('size', 1), ('device', 999)):
            args, auth, io = prepared()
            io.backings[next(iter(io.backings))][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): verify_storage_admission(io, args, auth)
        args, auth, io = prepared(); io.loops['/dev/loop10'] = '/var/tmp/unrelated'
        with self.assertRaisesRegex(ValueError, 'mount/loop'): verify_storage_admission(io, args, auth)

    def test_missing_duplicate_and_unjournaled_allocations(self):
        for mutation in ('missing', 'duplicate', 'unjournaled'):
            args, auth, io = prepared(); name = STATE+'/'+args.token+'/watchdog-bootstrap.json'
            capsule = io.json(name)
            if mutation == 'missing': capsule['resources'].pop()
            elif mutation == 'duplicate': capsule['resources'].append(deepcopy(capsule['resources'][0]))
            else: capsule['resources'][0]['identity']['inode'] += 1
            io.files[name] = canonical(capsule)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): verify_storage_admission(io, args, auth)

    def test_journal_corruption(self):
        args, auth, io = prepared()
        io.files[STATE+'/'+args.token+'/journal/0002-provision-observed.json'] += b' '
        with self.assertRaisesRegex(ValueError, 'journal hash'): verify_storage_admission(io, args, auth)

    def test_initial_binding_or_filesystem_change(self):
        args, auth, io = prepared(); auth['scenario'] = 'other'
        with self.assertRaisesRegex(ValueError, 'initial storage binding'): verify_storage_admission(io, args, auth)
        args, auth, io = prepared(); io.fs['id'] = 999
        with self.assertRaisesRegex(ValueError, 'filesystem changed'): verify_storage_admission(io, args, auth)

    def test_native_observation_reads_one_statvfs_snapshot(self):
        io = TrustedIO()
        filesystem = StorageIO().fs
        with patch.object(io, 'mounts', return_value=[filesystem]), \
             patch('activation_trust.os.stat', return_value=SimpleNamespace(st_dev=2049)), \
             patch('activation_trust.os.statvfs', create=True, return_value=SimpleNamespace(f_bavail=123, f_frsize=4096)) as statvfs:
            self.assertEqual(io.storage_observation()['available_bytes'], 123*4096)
            statvfs.assert_called_once_with('/var/tmp')

    def test_initial_record_persisted_after_reservation(self):
        from activation_bridge import dispatch
        from unittest.mock import Mock
        rt = Mock(); rt.g.args.role = 'bootstrap'; rt.g.storage = {'initial': 'invented'}
        with patch('activation_bridge.bootstrap', return_value={'complete': False}): dispatch(rt)
        rt.write.assert_called_once_with('storage-initial.json', rt.g.storage)
        names = [call[0] for call in rt.mock_calls]
        self.assertLess(names.index('reservations.claim'), names.index('write'))


if __name__ == '__main__': unittest.main()
