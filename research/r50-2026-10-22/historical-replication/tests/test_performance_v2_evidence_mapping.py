"""Invented bytes only; no filesystem, original supervision or research reads."""
import copy
import unittest

from consendxformer.performance_v2 import evidence_mapping as m
from consendxformer.performance_v2.bindings import FrozenInputs, CLASS_ORDER
from consendxformer.performance_v2.reuse import canonical, digest, _json


def fixture(counts=(3, 2)):
    cohort, supervision, images = [], [], {}
    for split, count in zip(('train', 'val'), counts):
        for i in range(count):
            nid = f'invented-{split}-{i:04d}'
            raw = ('not decoded: ' + nid).encode(); images[nid] = raw
            row = dict(patient_id='p-' + nid, nodule_id=nid, series_instance_uid='s-' + nid,
                       split=split, label=CLASS_ORDER[i % 3], n_valid_malignancy_ratings='3',
                       nodule_sha256=digest(nid.encode()), relative_image_path=f'resolution_64/{nid}.npy',
                       path=f'/invented/rois/resolution_64/{nid}.npy', image_sha256=digest(raw))
            cohort.append(row)
            supervision.append({**{k: row[k] for k in m.SUPERVISION_FIELDS & m.COHORT_FIELDS},
                                'image_path': row['path'], 'resolution': '64', 'q_benign': '0.2',
                                'q_indeterminate': '0.3', 'q_malignant': '0.5', 'disagreement_entropy': '0.4'})
    b = {'schema': 'pv2_mapping_binding_v1', 'scope': 'synthetic', 'class_order': list(CLASS_ORDER),
         'cohort_sha256': digest(canonical(cohort)), 'projection_sha256': digest(canonical(supervision)),
         'transform_sha256': digest(b'invented transform'), 'source_root': '/invented',
         'image_prefix': 'rois', 'sidecar_prefix': 'sidecars',
         'original_sources': {role: {'reference': role + '.csv', 'sha256': digest(role.encode())} for role in m.ROLES},
         'lineage_review_sha256': digest(b'invented external review'), 'pilot_execution_authorized': False}
    return b, cohort, supervision, images


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.b, self.c, self.s, self.images = fixture()

    def run_mapping(self, rebind=False):
        if rebind:
            self.b['cohort_sha256'] = digest(canonical(self.c))
            self.b['projection_sha256'] = digest(canonical(self.s))
        raw = canonical(self.b)
        return m.prepare_mapping(binding_bytes=raw, expected_binding_sha256=digest(raw),
                                 cohort_bytes=canonical(self.c), supervision_projection_bytes=canonical(self.s),
                                 image_bytes=self.images)

    def test_exact_roundtrip_and_canonical_sidecars(self):
        from consendxformer.performance_v2.input_adapter import supervision_bytes
        out = self.run_mapping(); lock = out['input-lock.json']
        inputs = FrozenInputs(lock, expected_sha256=digest(lock)); manifest = _json(out['source-paths.json'])
        for split in ('train', 'val'):
            for r in inputs.records(split):
                sidecar = out[manifest['records'][r.nodule_id]['supervision']]
                self.assertEqual(sidecar, supervision_bytes(r))
                self.assertEqual(digest(sidecar), r.supervision_sha256)
        self.assertFalse(_json(out['lineage.json'])['production_execution_authorized'])
        self.assertEqual(out, self.run_mapping())

    def test_807_143_invented_records_scale(self):
        self.b, self.c, self.s, self.images = fixture((807, 143))
        out = self.run_mapping(); rows = _json(out['input-lock.json'])['records']
        self.assertEqual([sum(r['split'] == s for r in rows) for s in ('train', 'val')], [807, 143])
        self.assertEqual(len(out), 953)

    def test_hash_before_parse(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            m.checked(b'not JSON', digest(b'different'))

    def test_projection_hash_drift(self):
        self.s[0]['q_benign'] = '0.7'
        with self.assertRaises(ValueError): self.run_mapping()

    def test_labels_not_recomputed(self):
        self.s[0]['label'] = 'malignant'
        with self.assertRaisesRegex(ValueError, 'label'): self.run_mapping(True)

    def test_reader_values_not_normalized(self):
        for value in ('NaN', 'inf', '-0.1', '0.9'):
            with self.subTest(value=value):
                self.s[0]['q_benign'] = value
                with self.assertRaises(ValueError): self.run_mapping(True)

    def test_test_rows_rejected_not_filtered(self):
        self.s[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'TEST forbidden'): self.run_mapping(True)

    def test_missing_extra_duplicate_members(self):
        for change in ('missing', 'extra', 'duplicate'):
            with self.subTest(change=change):
                self.setUp()
                if change == 'missing': self.s.pop()
                elif change == 'extra': self.images['unapproved'] = b'x'
                else: self.s.append(copy.deepcopy(self.s[0]))
                with self.assertRaises(ValueError): self.run_mapping(True)

    def test_class_order_frozen(self):
        self.b['class_order'].reverse()
        with self.assertRaises(ValueError): self.run_mapping()

    def test_corrupted_image_rejected_before_any_decode(self):
        self.images[next(iter(self.images))] = b'changed'
        with self.assertRaisesRegex(ValueError, 'image hash'): self.run_mapping()

    def test_image_path_substitution(self):
        self.s[0]['image_path'] = '/invented/TEST/x.npy'
        with self.assertRaisesRegex(ValueError, 'path mismatch'): self.run_mapping(True)

    def test_traversal_and_alias_rejected(self):
        for prefix in ('../bad', '/absolute', 'a\\b', 'a//b', 'a:stream'):
            with self.subTest(prefix=prefix):
                self.b['sidecar_prefix'] = prefix
                with self.assertRaises(ValueError): self.run_mapping()

    def test_original_lineage_mandatory(self):
        del self.b['original_sources']['original_supervision']
        with self.assertRaises(ValueError): self.run_mapping()

    def test_same_count_production_substitution_fails_exact_digest(self):
        self.b, self.c, self.s, self.images = fixture((807, 143))
        self.b['scope'] = 'production'
        with self.assertRaisesRegex(ValueError, 'exact cohort digest'): self.run_mapping()

    def test_order_preserved_not_silently_resorted(self):
        self.c.reverse()
        with self.assertRaisesRegex(ValueError, 'ordering'): self.run_mapping(True)

    def test_patient_overlap_rejected(self):
        self.c[-1]['patient_id'] = self.c[0]['patient_id']
        self.s[-1]['patient_id'] = self.c[0]['patient_id']
        with self.assertRaisesRegex(ValueError, 'disjoint'): self.run_mapping(True)

    def test_original_row_lineage_and_manifest_bound(self):
        out = self.run_mapping(); lineage = _json(out['lineage.json'])
        self.assertEqual(lineage['projection_sha256'], self.b['projection_sha256'])
        self.assertEqual(lineage['input_lock_sha256'], digest(out['input-lock.json']))
        self.assertEqual(lineage['path_manifest_sha256'], digest(out['source-paths.json']))
        self.assertEqual(lineage['records'][0]['original_supervision_row_sha256'], digest(canonical(self.s[0])))
