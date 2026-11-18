import hashlib, json, os
from pathlib import Path

p = Path('/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit/MULTISLICE-CROSSWALK-READONLY-20261019-001.py')
b = p.read_bytes()
assert hashlib.sha256(b).hexdigest() == 'e254a5c2e0f929fe35f7f3a57ce7a0b3a7a3d8d87792c30f1b5f8dda2a6a2039'
s = b.decode()
exec(compile(s[:s.index('result=dict(original')], str(p), 'exec'))
records = []
rep_outside = 0
for row in rows:
    if row['readers'] < 2:
        continue
    rep = row['representative']
    planes = row.get('at_least_two_reader_planes', [])
    others = sorted(set(planes) - {rep})
    m = min(6, len(others))
    rep_outside += rep not in planes
    ix = ([len(others)//2] if m == 1 else [j*(len(others)-1)//(m-1) for j in range(m)]) if m else []
    chosen = sorted([rep] + [others[i] for i in ix])
    assert len(chosen) == len(set(chosen)) <= 7
    records.append(dict(nodule_id=row['nodule_id'], patient_id=row['patient_id'], series=row['series'], label=row['label'], readers=row['readers'], representative=rep, slice_indices=chosen, per_view_weight=1/len(chosen), crosswalk_status='metadata_candidate_not_released' if planes else 'existing_representative_only_unresolved_crosswalk'))
N = sum(len(x['slice_indices']) for x in records)
unique = {(x['series'], k) for x in records for k in x['slice_indices']}
result = dict(TRAIN_patients=len({x['patient_id'] for x in records}), TRAIN_nodules=len(records), TRAIN_nodule_view_candidates=N, distinct_TRAIN_series_axial_planes=len(unique), view_classes=[sum(len(x['slice_indices']) for x in records if x['label']==i) for i in range(3)], nodule_classes=[sum(x['label']==i for x in records) for i in range(3)], VAL_nodules_unchanged=143, TEST_nodules_unchanged=183, total_example_candidates_including_frozen_VAL_TEST=N+326, existing_representatives_outside_new_two_reader_plane_rule=rep_outside, existing_only_unresolved=[x['nodule_id'] for x in records if 'unresolved' in x['crosswalk_status']], view_count_histogram={str(k):sum(len(x['slice_indices'])==k for x in records) for k in range(1,8)}, actual_new_crop_count=0, execution_authorized=False, full_annotation_signature_SOP_geometry_consensus_checks_pending=True, selection='existing representative plus up to six endpoint-spanning rank-quantile planes with >=2 inclusion readers; no repeated slice', records=records)
out = Path('/home/chaidir/.performance-v2-reports/multislice-feasibility-20261019-001/PROPOSED-POOL.NOT-RELEASED.json')
raw = (json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n').encode()
with out.open('xb') as f:
    f.write(raw)
print(json.dumps({k:v for k,v in result.items() if k!='records'}, sort_keys=True))
print('POOL_SHA256='+hashlib.sha256(raw).hexdigest())
