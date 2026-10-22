"""Opt-in new-supervision best/latest retention; never scans/deletes other runs."""
import copy,os,re,stat
from .reuse import canonical,digest

POLICY={'schema':'pv2_best_latest_retention_v1','scope':'new_RADIMAGENET_seed42_supervision_only',
        'keep':['validation_selected_best','latest_complete_resumable'],
        'epoch_metrics_and_selection_history':'all','checkpoint_body_history':'sparse',
        'validation_granularity_proof':'three_present_classes_N_le_143_v1'}
REPLICATION_POLICY=dict(POLICY,scope='new_RADIMAGENET_HARD_LEGACY_seeds43_44_replication_v1')
REPLICATION_RUNS={(s,o):f'R50-RADIMAGENET-{s}-{o}-REPLICATION-001-RETENTION-V1'
                   for s in (43,44) for o in ('HARD','LEGACY')}

def enabled(spec):
 if 'checkpoint_retention' not in spec:return False
 c=spec['contract']
 if (spec['checkpoint_retention']==REPLICATION_POLICY and c['initialization']=='RADIMAGENET'
     and REPLICATION_RUNS.get((c['seed'],c['objective']))==c['run_id']):return True
 if (spec['checkpoint_retention']!=POLICY or c['initialization']!='RADIMAGENET'
     or c['objective'] not in ('LEGACY','SHARED-010','SEPARATE-010') or c['seed']!=42
     or not c['run_id'].endswith('-RETENTION-V1')):raise ValueError('retention amendment exact new-run scope')
 return True
def metadata_name(n):return bool(re.fullmatch(r'(epoch-\d{3}\.(metrics|retention)\.json|prune-at-\d{3}-epoch-\d{3}\.(intent|done)\.json)',n))
def sync(session):
 # Only DurableRun's already reviewed native or explicit synthetic barrier.
 if not hasattr(session,'ticket'):raise ValueError('sparse retention requires durable lifecycle')
 session.ticket.budget._sync(session.store.path)
def fault(session,n):session.ticket.budget._fault('retention.'+n)
def write(session,n,obj):
 raw=canonical(obj);session.store.put(n,raw);sync(session)
 if (session.store.path/n).read_bytes()!=raw:raise ValueError('retention record readback mismatch')
 return digest(raw)
def file_map(store):return store._store.files if hasattr(store,'_store') else store.files
def selection_safe(inputs):
 rows=inputs.records('val');n=len(rows)
 if not 3<=n<=143 or any(not any(r.label==c for r in rows) for c in range(3)):
  raise ValueError('sparse selection requires bounded three-present-class validation')
 # d1+d2+d3=2N, D=lcm(d1,d2,d3) is even; distinct exact Macro-F1
 # values differ by at least4/(3*Dmax^2). 64 binary64 ulps allowance.
 q,r=divmod(2*n,3);maximum=q**(3-r)*(q+1)**r
 gap=4/(3*maximum**2)-64*2**-52
 if gap<=1e-12:raise ValueError('selection tolerance/granularity not proven')
 return {'N':n,'denominator_product_bound':maximum,'distinct_score_gap_minus_roundoff':gap}
class SparseRetention:
 def __init__(self,session,resume=None):
  self.session=session;self.proof=selection_safe(session.inputs)
  self.all_hashes=copy.deepcopy(resume.evidence['retention']['historical_checkpoint_hashes']) if resume else {}
  self.root_identity=None;self._own_root()
 def _own_root(self):
  p=self.session.store.path;s=p.lstat()
  if p.is_symlink() or not stat.S_ISDIR(s.st_mode):raise ValueError('retention root type')
  observed=(s.st_dev,s.st_ino)
  if self.root_identity is None:self.root_identity=observed
  if observed!=self.root_identity:raise ValueError('retention root identity drift')
  if os.name=='posix' and (s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o700):raise ValueError('private retention root')
  if os.name!='posix' and self.session.spec['scope']!='synthetic':raise ValueError('production retention requires native POSIX')
  return p
 def snapshot(self):
  s=self.session;n=s.progress['epoch']
  for e,(_,h) in s.checkpoints.items():
   if e<=n:
    k=str(e)
    if k in self.all_hashes and self.all_hashes[k]!=h:raise ValueError('historical checkpoint digest changed')
    self.all_hashes[k]=h
  retained={str(e):h for e,(_,h) in s.checkpoints.items() if e<=n}
  from .checkpoint import selection_state
  selection=selection_state(s.history) if s.history else None
  return {'schema':'pv2_sparse_retention_evidence_v1','policy':copy.deepcopy(s.spec['checkpoint_retention']),
          'historical_checkpoint_hashes':copy.deepcopy(self.all_hashes),'retained_checkpoint_hashes':retained,
          'selection':selection,'latest_epoch':n,'best_epoch':selection['best_epoch'] if selection else None,
          'metadata_hashes':{k:v for k,v in file_map(s.store).items() if metadata_name(k)},
          'full_checkpoint_bodies_retained':False,'selection_granularity':self.proof}
 def complete_epoch(self):
  s=self.session;root=self._own_root();state=self.snapshot();n=s.progress['epoch'];best=state['best_epoch']
  keep={best,n}
  if not keep<=set(s.checkpoints):raise ValueError('selection would require a deleted checkpoint')
  # Publish and verify BOTH future recovery anchors before any removal.
  for e in keep:
   name,h=s.checkpoints[e];raw=(root/name).read_bytes()
   if digest(raw)!=h:raise ValueError('best/latest checkpoint readback')
  sync(s);fault(s,'before_commit_record')
  write(s,'epoch-%03d.retention.json'%n,{'retention':state,'committed_keep':sorted(keep),'progress':copy.deepcopy(s.progress)})
  fault(s,'after_commit_record')
  for e in sorted(set(s.checkpoints)-keep):self.remove(e,at=n,keep=keep)
 def remove(self,e,*,at,keep):
  s=self.session;root=self._own_root()
  if e in keep:raise ValueError('referenced best/latest cannot be pruned')
  name,h=s.checkpoints[e];files=file_map(s.store)
  if name!='epoch-%03d.state.json'%e or files.get(name)!=h:raise ValueError('only registered own-run checkpoint may be removed')
  path=root/name;before=path.lstat()
  if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>512*1024**2:raise ValueError('prune regular exclusive checkpoint required')
  # Current code runs under the retained trusted same-UID assumption. POSIX
  # unlink is relative to a held, checked OWN-RUN directory descriptor.
  dfd=None;fd=None
  try:
   if os.name=='posix':
    dfd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    if (os.fstat(dfd).st_dev,os.fstat(dfd).st_ino)!=self.root_identity:raise ValueError('prune directory descriptor drift')
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=dfd)
    opened=os.fstat(fd)
    if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('prune file descriptor drift')
    with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(512*1024**2+1)
   else:raw=path.read_bytes() # Explicit synthetic-only Windows path above.
   if digest(raw)!=h:raise ValueError('prune hash mismatch')
   intent={'epoch':e,'checkpoint':name,'sha256':h,'bytes':before.st_size,'best_epoch':min(keep),
           'keep_epochs':sorted(keep),'at_epoch':at,'own_run_directory':str(root),'root_device_inode':list(self.root_identity)}
   write(s,'prune-at-%03d-epoch-%03d.intent.json'%(at,e),intent);fault(s,'before_unlink')
   now=os.stat(name,dir_fd=dfd,follow_symlinks=False) if dfd is not None else path.lstat()
   if (now.st_dev,now.st_ino,now.st_size)!=(before.st_dev,before.st_ino,before.st_size):raise ValueError('prune identity changed')
   if dfd is not None:os.unlink(name,dir_fd=dfd)
   else:path.unlink()
   del files[name];del s.checkpoints[e] # Update before interruption injection.
   fault(s,'after_unlink');sync(s);fault(s,'after_unlink_sync')
   write(s,'prune-at-%03d-epoch-%03d.done.json'%(at,e),dict(intent,status='deleted_and_directory_synced'))
  finally:
   if fd is not None:os.close(fd)
   if dfd is not None:os.close(dfd)

def validate_snapshot(spec,snapshot,hashes,*,complete=False):
 from .checkpoint import selection_state
 if not enabled(spec) or snapshot.get('schema')!='pv2_sparse_retention_evidence_v1' or snapshot.get('policy')!=spec.get('checkpoint_retention'):raise ValueError('exact sparse retention evidence required')
 n=snapshot['latest_epoch'];history=snapshot['selection']['history'];selection=selection_state(history)
 if selection!=snapshot['selection'] or n!=len(history) or snapshot['best_epoch']!=selection['best_epoch']:raise ValueError('sparse selection history drift')
 historical=snapshot['historical_checkpoint_hashes'];wanted={str(e) for e in range(1,n+1)}
 if set(historical)!=wanted or any(not re.fullmatch('[0-9a-f]{64}',h) for h in historical.values()):raise ValueError('complete historical digest ledger required')
 if hashes!=snapshot['retained_checkpoint_hashes'] or not {str(n),str(selection['best_epoch'])}<=set(hashes) or any(historical.get(k)!=v for k,v in hashes.items()):raise ValueError('best/latest sparse identities missing')
 if complete and set(hashes)!={str(n),str(selection['best_epoch'])}:raise ValueError('completion requires exact best/latest body set')
 if len(hashes)>4:raise ValueError('unbounded recovery checkpoint pins')
 metadata=snapshot['metadata_hashes']
 if any(not metadata_name(k) or not re.fullmatch('[0-9a-f]{64}',v) for k,v in metadata.items()):raise ValueError('sparse metadata bindings')
 if any('epoch-%03d.metrics.json'%e not in metadata for e in range(1,n+1)):raise ValueError('every epoch metric retained')
 return selection

class MetadataReferences:
 """Bounded lazy authenticated epoch/audit metadata; never checkpoint bodies."""
 def __init__(self,source,hashes):
  if source is None or set(source)!=set(hashes):raise ValueError('all sparse epoch/audit metadata required')
  self.source=source;self.hashes=copy.deepcopy(hashes)
 def items(self):
  from .reuse import _json
  for name,h in sorted(self.hashes.items()):
   if not metadata_name(name):raise ValueError('flat sparse metadata scope')
   raw=self.source[name]
   if type(raw) is not bytes or len(raw)>1024**2 or digest(raw)!=h:raise ValueError('sparse metadata hash/size')
   if canonical(_json(raw))!=raw:raise ValueError('canonical sparse metadata required')
   yield name,raw
 def validate(self,selection=None):
  from .reuse import _json
  import math
  records={r['epoch']:r for r in selection['history']} if selection else {}
  for name,raw in self.items():
   if name.endswith('.metrics.json'):
    summary=_json(raw);epoch=int(name[6:9])
    if selection:
     row=records.get(epoch)
     matrix=summary.get('confusion_matrix');n=summary.get('n')
     if (row is None or summary.get('split')!='val' or type(matrix) is not list or len(matrix)!=3
         or any(type(r) is not list or len(r)!=3 or any(type(v) is not int or v<0 for v in r) for r in matrix)
         or type(n) is not int or not 3<=n<=143 or sum(map(sum,matrix))!=n or any(sum(r)==0 for r in matrix)):
      raise ValueError('bound complete three-class epoch metrics')
     f1=[]
     for c in range(3):
      tp=matrix[c][c];predicted=sum(r[c] for r in matrix);support=sum(matrix[c])
      precision=tp/predicted if predicted else 0.;recall=tp/support
      f1.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
     if (summary['macro_f1']!=math.fsum(f1)/3 or row['macro_f1']!=summary['macro_f1']
         or row['diagnostic_ce']!=summary['diagnostic_ce']):raise ValueError('metrics/selection history contradiction')
