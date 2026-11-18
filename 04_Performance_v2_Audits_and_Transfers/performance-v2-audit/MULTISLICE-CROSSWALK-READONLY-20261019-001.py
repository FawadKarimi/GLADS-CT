import os,json,sqlite3,math,itertools,hashlib
from pathlib import Path
from collections import Counter
os.umask(0o077)
R=Path('/home/chaidir/.performance-v2-reports/multislice-feasibility-20261019-001')
rows=json.loads((R/'NODULE-SLICE-CANDIDATES.json').read_bytes());counts=json.loads((R/'COUNTS.json').read_bytes());failed={x['nodule_id'] for x in counts['failures']}
c=sqlite3.connect('file:/experiment-storage/chaidir/env/env-310-fawad/lib/python3.10/site-packages/pylidc/pylidc.sqlite?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.row_factory=sqlite3.Row
changes=[]
for row in rows:
 if row['nodule_id'] not in failed:continue
 scan=c.execute('SELECT id FROM scans WHERE patient_id=? AND series_instance_uid=?',(row['patient_id'],row['series'])).fetchall();assert len(scan)==1;sid=scan[0]['id'];zs=[x[0] for x in c.execute('SELECT val FROM zvals WHERE scan_id=? ORDER BY val',(sid,))];zm={z:i for i,z in enumerate(zs)};cand=[];box=row['bbox']
 for a in c.execute('SELECT id,malignancy FROM annotations WHERE scan_id=?',(sid,)).fetchall():
  pts=[];planes=set()
  for co in c.execute('SELECT inclusion,image_z_position,coords FROM contours WHERE annotation_id=?',(a['id'],)):
   k=zm[co['image_z_position']];xy=[tuple(map(int,x.split(','))) for x in co['coords'].splitlines()];pts.extend((p[1],p[0],k) for p in xy)
   area=abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[(i+1)%len(xy)][0]*xy[i][1] for i in range(len(xy)))) if xy else 0
   if co['inclusion'] and len(set(xy))>=3 and area>0:planes.add(k)
  if not pts:continue
  ab=[[min(p[i] for p in pts),max(p[i] for p in pts)+1] for i in range(3)]
  if all(box[i][0]<=ab[i][0] and ab[i][1]<=box[i][1] for i in range(3)):cand.append(dict(id=a['id'],rating=a['malignancy'],bbox=ab,planes=planes))
 assert math.comb(len(cand),row['readers'])<=10000
 matches=[cs for cs in itertools.combinations(cand,row['readers']) if sorted(a['rating'] for a in cs)==row['ratings'] and [[min(a['bbox'][i][0] for a in cs),max(a['bbox'][i][1] for a in cs)] for i in range(3)]==box]
 change=dict(nodule_id=row['nodule_id'],contained_annotations=len(cand),exact_bbox_rating_subsets=len(matches),status='UNRESOLVED')
 if len(matches)==1:
  cs=matches[0];ct=Counter(k for a in cs for k in a['planes']);row['annotation_ids_metadata_crosswalk']=[a['id'] for a in cs];row['any_reader_planes']=sorted(ct);row['at_least_two_reader_planes']=[k for k,v in sorted(ct.items()) if v>=2];row['all_reader_planes']=[k for k,v in sorted(ct.items()) if v==row['readers']];change['status']='UNIQUE_METADATA_SUBSET_NOT_FULL_SIGNATURE_CLEARANCE'
 changes.append(change)
c.close()
def stats(pool):
 matched=[x for x in pool if 'any_reader_planes' in x]
 result=dict(nodules=len(pool),patients=len({x['patient_id'] for x in pool}),classes=[sum(x['label']==i for x in pool) for i in range(3)],readers=dict(Counter(x['readers'] for x in pool)),extent_planes=sum(x['span'] for x in pool),extent_cap5=sum(min(5,x['span']) for x in pool),metadata_crosswalk_matched=len(matched))
 for field in ['any_reader_planes','at_least_two_reader_planes','all_reader_planes']:
  result[field]={str(cap):sum(min(cap,len(x[field])) for x in matched) for cap in [1,3,5,7,9999]}
 return result
result=dict(original_multireader807=stats([x for x in rows if x['readers']>=2]),TRAIN1166=stats(rows),crosswalk_dispositions=changes,original_counts_preserved=True,full_signature_and_SOP_and_nonempty_consensus_still_pending=True)
raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n').encode()
with (R/'CROSSWALK-RECONCILIATION.json').open('xb') as f:f.write(raw)
print(json.dumps(result,sort_keys=True));print('RECONCILIATION_SHA256='+hashlib.sha256(raw).hexdigest())
