"""Pure byte-to-byte TRAIN/VAL mapping proposal. No filesystem or execution API.

A separately reviewed projection binding is a trust input: a digest alone does
not prove that a custodian faithfully extracted rows from an original source.
Original mixed-split files must NOT be supplied to this train/val-only interface.
"""
from pathlib import PurePosixPath

from .bindings import CLASS_ORDER, FrozenInputs, InputRecord
from .reuse import _hash, _json, canonical, digest

COHORT_SHA = '74ade0365dc6b7ea5593b4736692956b863b4245e122829e2921b0937663b5b1'
COHORT_FIELDS = {'patient_id', 'nodule_id', 'series_instance_uid', 'split', 'label',
                 'n_valid_malignancy_ratings', 'nodule_sha256', 'path',
                 'relative_image_path', 'image_sha256'}
SUPERVISION_FIELDS = {'patient_id', 'nodule_id', 'series_instance_uid', 'split',
                      'label', 'n_valid_malignancy_ratings', 'nodule_sha256',
                      'image_path', 'resolution', 'q_benign', 'q_indeterminate',
                      'q_malignant', 'disagreement_entropy'}
ROLES = {'original_supervision', 'original_input_lock', 'original_run_identity',
         'original_artifact_manifest', 'original_completion', 'archived_input_ledger',
         'archived_source_hashes', 'rendered_checksum_ledger'}


def checked(raw, wanted):
    if type(raw) is not bytes or not _hash(wanted) or digest(raw) != wanted:
        raise ValueError('evidence hash mismatch before parsing')
    return _json(raw)


def relative(name):
    if (type(name) is not str or not name or '\\' in name or ':' in name
            or name != PurePosixPath(name).as_posix()
            or PurePosixPath(name).is_absolute()
            or any(x in ('', '.', '..') for x in name.split('/'))):
        raise ValueError('unsafe relative evidence path')
    return name


def index(rows, fields):
    if type(rows) is not list or not rows:
        raise ValueError('nonempty reviewed rows required')
    result = {}
    for row in rows:
        if type(row) is not dict or set(row) != fields or row['split'] not in ('train', 'val'):
            raise ValueError('exact TRAIN/VAL fields required; TEST forbidden')
        identity = row['nodule_id']
        if type(identity) is not str or not identity or identity in result:
            raise ValueError('duplicate or invalid nodule identity')
        result[identity] = row
    return result


def prepare_mapping(*, binding_bytes, expected_binding_sha256, cohort_bytes,
                    supervision_projection_bytes, image_bytes):
    """Return proposed lock, path manifest, lineage and canonical sidecar bytes.

    No writes, decoding, discovery, tensor operations, TEST filtering or inferred
    labels. image_bytes must contain exactly the 950 approved images in production.
    The caller must separately review/exclusively materialize these proposed bytes.
    """
    b = checked(binding_bytes, expected_binding_sha256)
    required = {'schema', 'scope', 'class_order', 'cohort_sha256', 'projection_sha256',
                'transform_sha256', 'source_root', 'image_prefix', 'sidecar_prefix',
                'original_sources', 'lineage_review_sha256', 'pilot_execution_authorized'}
    if (type(b) is not dict or set(b) != required or b['schema'] != 'pv2_mapping_binding_v1'
            or b['scope'] not in ('synthetic', 'production')
            or b['class_order'] != list(CLASS_ORDER)
            or b['pilot_execution_authorized'] is not False
            or any(not _hash(b[k]) for k in ('cohort_sha256', 'projection_sha256',
                                            'transform_sha256', 'lineage_review_sha256'))):
        raise ValueError('exact inactive reviewed binding required')
    if b['scope'] == 'production' and b['cohort_sha256'] != COHORT_SHA:
        raise ValueError('production requires the accepted exact cohort digest')
    root = b['source_root']
    if (type(root) is not str or not root.startswith('/') or root == '/'
            or str(PurePosixPath(root)) != root or '\\' in root or ':' in root
            or '..' in root.split('/')):
        raise ValueError('explicit normalized Linux source root required')
    for k in ('image_prefix', 'sidecar_prefix'):
        relative(b[k])
    sources = b['original_sources']
    if type(sources) is not dict or set(sources) != ROLES:
        raise ValueError('complete original source lineage required')
    for source in sources.values():
        if (type(source) is not dict or set(source) != {'reference', 'sha256'}
                or not _hash(source['sha256'])):
            raise ValueError('original source reference/hash required')
        relative(source['reference'])
    cohort = checked(cohort_bytes, b['cohort_sha256'])
    c = index(cohort, COHORT_FIELDS)
    if canonical(sorted(cohort, key=lambda r: (r['split'], r['nodule_id']))) != cohort_bytes:
        raise ValueError('exact archived canonical cohort ordering required')
    projection = checked(supervision_projection_bytes, b['projection_sha256'])
    s = index(projection, SUPERVISION_FIELDS)
    if set(c) != set(s) or type(image_bytes) is not dict or set(image_bytes) != set(c):
        raise ValueError('exact cohort, supervision and image membership required')
    records, paths, sidecars, lineage, used = [], {}, {}, [], set()
    for row in cohort:
        identity = row['nodule_id']; sup = s[identity]
        for key in ('patient_id', 'nodule_id', 'series_instance_uid', 'split', 'label',
                    'n_valid_malignancy_ratings', 'nodule_sha256'):
            if row[key] != sup[key]:
                raise ValueError('original supervision/cohort mismatch: ' + key)
        if (row['label'] not in CLASS_ORDER or sup['resolution'] != '64'
                or sup['n_valid_malignancy_ratings'] not in ('2', '3', '4')
                or not _hash(row['nodule_sha256'])):
            raise ValueError('frozen label/eligibility mismatch')
        image_path = relative(b['image_prefix'] + '/' + relative(row['relative_image_path']))
        if (row['relative_image_path'] != 'resolution_64/' + identity + '.npy'
                or row['path'] != root + '/' + image_path or sup['image_path'] != row['path']):
            raise ValueError('frozen image path mismatch')
        raw = image_bytes[identity]
        if (type(raw) is not bytes or not 0 < len(raw) <= 1048576
                or not _hash(row['image_sha256']) or digest(raw) != row['image_sha256']):
            raise ValueError('bounded original image hash mismatch; no decoding')
        record = {'patient_id': row['patient_id'], 'nodule_id': identity,
                  'series_instance_uid': row['series_instance_uid'], 'split': row['split'],
                  'label': CLASS_ORDER.index(row['label']),
                  'reader_target': [float(sup['q_' + x]) for x in CLASS_ORDER],
                  'disagreement': float(sup['disagreement_entropy']),
                  'valid_readers': int(sup['n_valid_malignancy_ratings']),
                  'image_sha256': row['image_sha256'], 'supervision_sha256': '0' * 64}
        # InputRecord rejects nonfinite/out-of-range targets; never renormalize.
        parsed = InputRecord.parse(record).as_dict()
        body = {k: v for k, v in parsed.items() if k not in ('image_sha256', 'supervision_sha256')}
        sidecar = canonical({'schema': 'performance_v2_supervision_sidecar_v1', 'record': body})
        record['supervision_sha256'] = digest(sidecar)
        sidecar_path = relative(b['sidecar_prefix'] + '/' + digest(identity.encode()) + '.json')
        for name in (image_path, sidecar_path):
            if name.casefold() in used:
                raise ValueError('duplicate or aliased path')
            used.add(name.casefold())
        paths[identity] = {'image': image_path, 'supervision': sidecar_path}
        sidecars[sidecar_path] = sidecar
        records.append(record)
        lineage.append({'nodule_id': identity, 'original_supervision_row_sha256': digest(canonical(sup)),
                        'cohort_row_sha256': digest(canonical(row)),
                        'image_sha256': row['image_sha256'], 'sidecar_sha256': digest(sidecar)})
    lock = canonical({'schema': 'performance_v2_input_interface_v1', 'scope': b['scope'],
                      'class_order': list(CLASS_ORDER), 'archived_cohort_sha256': b['cohort_sha256'],
                      'transform_sha256': b['transform_sha256'], 'records': records})
    inputs = FrozenInputs(lock, expected_sha256=digest(lock))  # production counts and disjointness
    manifest = canonical({'schema': 'performance_v2_source_paths_v1', 'scope': inputs.scope,
                          'input_lock_sha256': inputs.sha256, 'records': paths})
    provenance = canonical({'schema': 'pv2_mapping_lineage_v1', 'binding_sha256': expected_binding_sha256,
                            'original_sources': sources, 'projection_sha256': b['projection_sha256'],
                            'source_root': root, 'lineage_review_sha256': b['lineage_review_sha256'],
                            'records': lineage, 'input_lock_sha256': digest(lock),
                            'path_manifest_sha256': digest(manifest),
                            'production_execution_authorized': False})
    return {'input-lock.json': lock, 'source-paths.json': manifest,
            'lineage.json': provenance, **sidecars}
