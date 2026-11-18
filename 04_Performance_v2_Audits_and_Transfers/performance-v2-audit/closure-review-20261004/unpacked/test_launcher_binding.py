"""Invented inventory and fake OS only, not Linux acceptance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from launcher_binding import (EXECUTABLES, ROOT, authorize, canonical, command,
                              sha, source_manifest, verify_sources)
from native_launcher import dispatch_mocked
from synthetic_helper import blocks
import test_linux_integration as integration


def fixture():
    manifest = source_manifest(Path(__file__).parent)
    inventory = {'schema': 'pv2_native_inventory_v1', 'architecture': 'x86_64',
                 'cgroup_version': 2, 'boot_id': 'invented-boot', 'kernel': 'invented',
                 'os_release': 'invented', 'python_version': '3.10 invented',
                 'systemd_version': 'invented', 'security_review_sha256': 'b'*64,
                 'host_evidence_sha256': 'c'*64,
                 'executables': {p: {'sha256': 'a'*64, 'resolved_path': p, 'version': 'invented'} for p in EXECUTABLES}}
    auth = {'schema': 'pv2_native_authorization_v1', 'scope': 'A_native_synthetic_only',
            'execution_authorized': True, 'security_approved': True, 'token': 'a'*32,
            'boot_id': 'invented-boot', 'sources_sha256': sha(canonical(manifest)),
            'inventory_sha256': sha(canonical(inventory)), 'not_before': 100, 'expires': 200,
            'dependency_inspection_authorized': False, 'pilot_execution_authorized': False,
            'scenario': 'success', 'role_chain': ['bootstrap', 'controller', 'watchdog', 'helper']}
    return manifest, inventory, auth


class BindingTests(unittest.TestCase):
    def test_exact_sources(self):
        manifest, _, _ = fixture()
        verify_sources(Path(__file__).parent, manifest)
        manifest['files'][ROOT+'/watchdog_worker.py']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            verify_sources(Path(__file__).parent, manifest)

    def test_authorization(self):
        m, i, a = fixture()
        authorize(a, m, i, 'a'*32, 150)
        for field, value in [('execution_authorized', False), ('security_approved', False),
                             ('sources_sha256', '0'*64), ('inventory_sha256', '0'*64),
                             ('boot_id', 'other'), ('scope', 'B'), ('expires', 150),
                             ('pilot_execution_authorized', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                bad = dict(a, **{field: value}); authorize(bad, m, i, 'a'*32, 150)

    def test_missing_inventory(self):
        m, i, a = fixture()
        del i['executables']['/usr/bin/runc']
        with self.assertRaisesRegex(ValueError, 'coverage'):
            authorize(a, m, i, 'a'*32, 150)

    def test_no_live_backend(self):
        m, i, a = fixture()
        args = SimpleNamespace(role='controller', token='a'*32, scenario='success')
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            dispatch_mocked(args, m, i, a, 150, SimpleNamespace(is_fake=False))

    def test_rejected_authorization_never_dispatches(self):
        m, i, a = fixture(); a['execution_authorized'] = False
        backend = Mock(is_fake=True)
        with self.assertRaises(ValueError):
            dispatch_mocked(SimpleNamespace(role='controller', token='a'*32, scenario='success'), m, i, a, 150, backend)
        backend.launch_role.assert_not_called()

    def test_concrete_roles(self):
        m, i, a = fixture()
        backend = Mock(is_fake=True)
        for role in ('controller', 'watchdog', 'helper'):
            args = SimpleNamespace(role=role, token='a'*32, scenario='success')
            dispatch_mocked(args, m, i, a, 150, backend)
            backend.launch_role.assert_called_with(role, command(role, 'a'*32), i, 'success')
        self.assertEqual(backend.launch_role.call_count, 3)

    def test_payload_boundaries(self):
        self.assertEqual(b''.join(blocks('success')), b'pv2-native-synthetic-v1\n')
        self.assertEqual(sum(map(len, blocks('output_overflow'))), 17*65536)
        with self.assertRaises(TimeoutError): list(blocks('timeout'))

    def test_integrated_concrete_argv(self):
        # Actual coordinator/state machine with fake services, mounts and clock.
        h, c, inventory, evidence = integration.IntegrationTests().make()
        h.blocks = list(blocks('success')) + [b'']
        result = c.execute(inventory, command('watchdog', 'a'*32),
                           command('helper', 'a'*32), '123', '1', evidence)
        self.assertEqual(result['state'], 'VERIFIED_FAKE')


if __name__ == '__main__': unittest.main()
