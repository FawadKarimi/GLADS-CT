import os,sys,csv,json,hashlib,stat,io,zipfile,ast,sqlite3,math,time
from pathlib import Path
from collections import Counter,defaultdict
started=time.monotonic();os.umask(0o077)
ROOT=Path('/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/03_ConsenDxFormer_and_Datasets/ConsenDxFormer');BASE=ROOT/'data/processed/consendxformer_phase2'
INPUT=Path('/home/chaidir/.performance-v2-train1166-timing-20261019-001/input-lock.PROPOSED.json')
OUT=Path('/home/chaidir/.performance-v2-reports/multislice-feasibility-20261019-001');OUT.mkdir(mode=0o700)
def raw(p,cap):
 a=p.lstat();assert stat.S_ISREG(a.st_mode) and not p.is_symlink() and a.st_size<=cap
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  b=os.fstat(fd);assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
  with os.fdopen(fd,'rb',closefd=False) as f:v=f.read(cap+1)
  assert len(v)<=cap;return v
 finally:os.close(fd)
def sha(b):return hashlib.sha256(b).hexdigest()
def output(n,j):
 b=(json.dumps(j,sort_keys=True,separators=(',',':'))+'\n').encode()
 with (OUT/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return sha(b)
cl=raw(BASE/'canonical_nodule_ledger.csv',4*1024**2);assert sha(cl)=='903ab5bf94875dde44d19fbb3f170a8f1ecab8da700e3107f4a7651b2c469bd6'
il=raw(INPUT,2*1024**2);assert sha(il)=='234cb64587ad2b02965f6d2efc5b2d5870b89128b97f9c55be723ff7d35b58f1'
inputs=json.loads(il);train={r['nodule_id']:r for r in inputs['records'] if r['split']=='train'};assert len(train)==1166
led={r['nodule_id']:r for r in csv.DictReader(io.StringIO(cl.decode()))};rows=[];fail=[]
for nid,t in sorted(train.items()):
 row=led[nid];assert row['split']=='train' and row['patient_id']==t['patient_id'] and row['series_instance_uid']==t['series_instance_uid'] and row['source_validity_status']=='valid'
 p=BASE/row['mask_path'];assert p.parent==BASE/'masks';b=raw(p,4*1024**2);assert sha(b)==row['mask_sha256']
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  info=z.getinfo('metadata_json_utf8.npy');assert info.file_size<=65536;v=z.read(info)
  h=10 if v[6]==1 else 12;l=int.from_bytes(v[8:h],'little');head=ast.literal_eval(v[h:h+l].decode().strip());assert head['descr']=='|u1';meta=json.loads(v[h+l:].decode())
 assert meta['annotation_sha256s']==json.loads(row['annotation_sha256s_json']) and meta['representative_slice_index']==int(row['representative_slice_index'])
 box=meta['full_image_bbox'];assert len(box)==3 and all(len(x)==2 and x[1]>x[0] for x in box)
 rows.append(dict(nodule_id=nid,patient_id=t['patient_id'],series=t['series_instance_uid'],label=t['label'],readers=t['valid_readers'],bbox=box,representative=int(row['representative_slice_index']),representative_z=float(row['representative_z']),ratings=sorted(json.loads(row['malignancy_ratings_json'])),mask_sha256=row['mask_sha256'],entropy=t['disagreement'],span=box[2][1]-box[2][0]))
DB=Path('/experiment-storage/chaidir/env/env-310-fawad/lib/python3.10/site-packages/pylidc/pylidc.sqlite');dbstat=DB.stat();c=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('BEGIN');c.row_factory=sqlite3.Row
scans=defaultdict(list)
for row in rows:scans[(row['patient_id'],row['series'])].append(row)
for (patient,series),group in scans.items():
 assert time.monotonic()-started<160
 ss=c.execute('SELECT id FROM scans WHERE patient_id=? AND series_instance_uid=?',(patient,series)).fetchall()
 if len(ss)!=1:
  fail.extend(dict(nodule_id=x['nodule_id'],reason='scan_not_unique') for x in group);continue
 sid=ss[0]['id'];zs=[r[0] for r in c.execute('SELECT val FROM zvals WHERE scan_id=? ORDER BY val',(sid,))];zmap={z:i for i,z in enumerate(zs)}
 aa=c.execute('SELECT id,malignancy FROM annotations WHERE scan_id=?',(sid,)).fetchall();annotations=[]
 for a in aa:
  cs=c.execute('SELECT inclusion,image_z_position,dicom_file_name,coords FROM contours WHERE annotation_id=? ORDER BY id',(a['id'],)).fetchall();points=[];planes=set();bad=False
  for co in cs:
   if co['image_z_position'] not in zmap:bad=True;break
   k=zmap[co['image_z_position']]
   try:xy=[tuple(int(v) for v in line.split(',')) for line in co['coords'].splitlines()];assert all(len(p)==2 for p in xy)
   except Exception:bad=True;break
   points.extend((p[1],p[0],k) for p in xy)
   area=abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[(i+1)%len(xy)][0]*xy[i][1] for i in range(len(xy)))) if xy else 0
   if co['inclusion'] and len(set(xy))>=3 and area>0:planes.add(k)
  if bad or not points:continue
  ab=[[min(p[i] for p in points),max(p[i] for p in points)+1] for i in range(3)]
  annotations.append(dict(id=a['id'],rating=a['malignancy'],bbox=ab,planes=planes))
 for row in group:
  box=row['bbox'];cand=[a for a in annotations if all(box[i][0]<=a['bbox'][i][0] and a['bbox'][i][1]<=box[i][1] for i in range(3))]
  if len(zs)!=len(zmap) or row['representative']>=len(zs) or zs[row['representative']]!=row['representative_z']:reason='slice_z_binding_mismatch'
  elif len(cand)!=row['readers'] or sorted(a['rating'] for a in cand)!=row['ratings']:reason='bbox_reader_crosswalk_ambiguous'
  elif [[min(a['bbox'][i][0] for a in cand),max(a['bbox'][i][1] for a in cand)] for i in range(3)]!=box:reason='union_bbox_mismatch'
  else:
   count=Counter(k for a in cand for k in a['planes']);row['annotation_ids_metadata_crosswalk']=[a['id'] for a in cand]
   row['any_reader_planes']=sorted(count);row['at_least_two_reader_planes']=[k for k,v in sorted(count.items()) if v>=2];row['all_reader_planes']=[k for k,v in sorted(count.items()) if v==row['readers']];continue
  fail.append(dict(nodule_id=row['nodule_id'],reason=reason))
c.rollback();c.close();assert DB.stat().st_size==dbstat.st_size and DB.stat().st_mtime_ns==dbstat.st_mtime_ns
# Crosswalk is a feasibility association, NOT replacement for full annotation/SOP signature validation.
def stats(pool):
 matched=[x for x in pool if 'any_reader_planes' in x]
 result=dict(nodules=len(pool),patients=len({x['patient_id'] for x in pool}),classes=[sum(x['label']==i for x in pool) for i in range(3)],readers=dict(Counter(x['readers'] for x in pool)),extent_planes=sum(x['span'] for x in pool),extent_cap5=sum(min(5,x['span']) for x in pool),metadata_crosswalk_matched=len(matched))
 for field in ['any_reader_planes','at_least_two_reader_planes','all_reader_planes']:
  result[field]={str(cap):sum(min(cap,len(x[field])) for x in matched) for cap in [1,3,5,7,9999]}
 return result
result=dict(schema='glads_multislice_readonly_feasibility_v1',status='candidate_counts_not_released_crops',canonical_sha256=sha(cl),input_lock_sha256=sha(il),database_path=str(DB),database_bytes=dbstat.st_size,database_mtime_ns=dbstat.st_mtime_ns,TRAIN1166=stats(rows),original_multireader807=stats([x for x in rows if x['readers']>=2]),single_reader359=stats([x for x in rows if x['readers']==1]),failures=fail,geometry_release_pending=['full_annotation_signature_to_contour_crosswalk','per_slice_SOP_identity_and_geometry','nonempty_consensus_after_exclusion_contours','eligible_crop_bytes_do_not_exist_yet'],read_scope='TRAIN annotation masks/metadata only; no CT image bytes, VAL/TEST masks, models or crops',runtime_seconds=time.monotonic()-started)
rh=output('NODULE-SLICE-CANDIDATES.json',rows);result['records_sha256']=rh;h=output('COUNTS.json',result)
print(json.dumps(result,sort_keys=True));print('COUNTS_SHA256='+h+' RECORDS_SHA256='+rh+' REPORT='+str(OUT))
