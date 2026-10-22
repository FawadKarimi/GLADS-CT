"""Storage-only caps for four NEW replication IDs. Never prunes evidence."""
import hashlib,os,re,stat,zipfile
from pathlib import Path
MiB=1024**2
METADATA_TOTAL=128*MiB
METADATA_FILE=8*MiB
LOG_STREAM=32*MiB
# Supervisor and outer controller EACH have32MiB:64MiB combined per run.
CONTROL_TOTAL=32*MiB
ARCHIVE_FILE=512*MiB
ARCHIVE_INPUT=480*MiB
BUDGET_DATABASE=64*MiB
BUDGET_AUXILIARY=64*MiB
SNAPSHOT_FILE=8*MiB
CHECKPOINT_SLOTS=6
class AuxiliaryLimit(ValueError):pass
def active(spec):
 from .retention import REPLICATION_POLICY,REPLICATION_RUNS
 c=spec.get('contract',{})
 return (c.get('initialization')=='RADIMAGENET' and
         c.get('run_id')==REPLICATION_RUNS.get((c.get('seed'),c.get('objective'))) and
         spec.get('checkpoint_retention')==REPLICATION_POLICY)
def plan_active(plan):
 from .retention import REPLICATION_RUNS
 return plan.get('limit')==4 and set(plan.get('runs',{}))==set(REPLICATION_RUNS.values())
def rows(root):
 root=Path(root);s=root.lstat()
 if root.is_symlink() or not stat.S_ISDIR(s.st_mode):raise AuxiliaryLimit('private directory identity')
 result=[]
 for p in root.iterdir():
  t=p.lstat()
  if p.is_symlink() or not stat.S_ISREG(t.st_mode):raise AuxiliaryLimit('unclassified flat output')
  result.append((p.name,t.st_size))
 return result
def checkpoint_name(n):return bool(re.fullmatch(r'epoch-\d{3}\.state\.json',n))
def admit_store(root,name,raw):
 # Unknown pending bytes count against BOTH allowances. Failure never removes them.
 existing=rows(root)
 pending=sum(size for n,size in existing if n.endswith('.pending'))
 if checkpoint_name(name):
  slots=sum(checkpoint_name(n) or n.endswith('.pending') for n,_ in existing)
  # Publication temporarily creates a second link name. Count it conservatively.
  if slots+2>CHECKPOINT_SLOTS:raise AuxiliaryLimit('checkpoint transition slots including recovery/pending pins')
  return
 total=sum(size for n,size in existing if not checkpoint_name(n) and not n.endswith('.pending'))+pending
 if len(raw)>METADATA_FILE or total+len(raw)>METADATA_TOTAL:raise AuxiliaryLimit('complete metadata cap; retain failure, never omit metrics')
def apply_sqlite_limit(connection):
 page=connection.execute('PRAGMA page_size').fetchone()[0]
 count=connection.execute('PRAGMA page_count').fetchone()[0]
 if page<=0 or count*page>BUDGET_DATABASE:raise AuxiliaryLimit('budget database bound before mutation')
 maximum=BUDGET_DATABASE//page
 actual=connection.execute('PRAGMA max_page_count=%d'%maximum).fetchone()[0]
 if actual!=maximum:raise AuxiliaryLimit('SQLite page cap not configured')
def admit_budget_aux(root,raw):
 total=0
 for sub in ('receipts','retained'):
  total+=sum(size for _,size in rows(Path(root)/sub))
 if len(raw)>SNAPSHOT_FILE or total+len(raw)>BUDGET_AUXILIARY:raise AuxiliaryLimit('budget receipt/retention auxiliary cap')
class BoundedSink:
 def __init__(self,stream,cap):self.stream=stream;self.cap=cap;self.bytes=0;self.rejected_bytes=0
 def write(self,raw):
  if type(raw) is not bytes:raise TypeError('byte output only')
  remaining=self.cap-self.bytes
  keep=raw[:remaining]
  if keep:self.stream.write(keep);self.stream.flush();self.bytes+=len(keep)
  if len(raw)>len(keep):
   self.rejected_bytes+=len(raw)-len(keep)
   raise AuxiliaryLimit('output cap reached; retained prefix is partial')
def write_control(root,name,raw):
 # This flat report root may contain a separately observed empty/managed tmp directory.
 total=0
 for p in Path(root).iterdir():
  if p.name in ('stdout.log','stderr.log'):continue
  s=p.lstat()
  if stat.S_ISDIR(s.st_mode) and p.name=='tmp' and not p.is_symlink():continue
  if p.is_symlink() or not stat.S_ISREG(s.st_mode):raise AuxiliaryLimit('control output type')
  total+=s.st_size
 if total+len(raw)>CONTROL_TOTAL:raise AuxiliaryLimit('control finalization cap')
def package_metadata(destination,files):
 """One exclusive metadata-only archive; a failed partial remains in place.

 Call once only. All checkpoint/pending bodies stay at authoritative private paths.
 Exactly one package is sufficient; no duplicate per-run package is authorized.
 """
 destination=Path(destination)
 from .launch_security import private
 private(destination.parent,True)
 checked_parents=set()
 if destination.exists() or destination.is_symlink():raise AuxiliaryLimit('archive destination exists')
 names=set();total=0;frozen=[]
 for n,p,expected_hash in files:
  if type(expected_hash) is not str or re.fullmatch(r"[0-9a-f]{64}",expected_hash) is None:raise AuxiliaryLimit("authenticated member digest required")
  p=Path(p)
  if p.parent not in checked_parents:private(p.parent,True);checked_parents.add(p.parent)
  s=p.lstat()
  if s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o600:raise AuxiliaryLimit('private archive input permissions')
  if (n in names or not n or '/' in n or chr(92) in n or n in ('.','..') or
      n.endswith(('.state.json','.typed-json','.pt','.pth','.ckpt','.safetensors','.npz','.npy','.pending')) or
      p.is_symlink() or not stat.S_ISREG(s.st_mode)):
   raise AuxiliaryLimit('metadata-only archive membership')
  names.add(n);total+=s.st_size;frozen.append((n,p,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,expected_hash))
 if len(names)>20000 or total>ARCHIVE_INPUT:raise AuxiliaryLimit('archive input/count cap')
 os.umask(0o077)
 with destination.open('xb') as out:
  sink=BoundedSink(out,ARCHIVE_FILE)
  # ZipFile requires seek/tell; monitor before each bounded write instead of raw seek writes.
  class Limited:
   def write(self,b):sink.write(b);return len(b)
   def tell(self):return sink.bytes
   def flush(self):out.flush()
  with zipfile.ZipFile(Limited(),'w',compression=zipfile.ZIP_STORED,allowZip64=False) as z:
   for n,p,dev,ino,size,mtime,expected_hash in frozen:
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
     s=os.fstat(fd)
     if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(dev,ino,size,mtime):raise AuxiliaryLimit('archive descriptor identity drift')
     with os.fdopen(fd,'rb',closefd=False) as f,z.open(n,'w') as entry:
      remaining=size;h=hashlib.sha256()
      while remaining:
       b=f.read(min(65536,remaining))
       if not b:raise AuxiliaryLimit('short archive input')
       entry.write(b);h.update(b);remaining-=len(b)
      if f.read(1):raise AuxiliaryLimit('archive input grew')
      if h.hexdigest()!=expected_hash:raise AuxiliaryLimit('archive member digest mismatch')
     s=os.fstat(fd)
     if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(dev,ino,size,mtime):raise AuxiliaryLimit('archive input changed')
    finally:os.close(fd)
  out.flush();os.fsync(out.fileno())
 fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
 with zipfile.ZipFile(destination) as z:
  if z.testzip() is not None or set(z.namelist())!=names:raise AuxiliaryLimit('archive readback')
  for n,_,_,_,_,_,expected_hash in frozen:
   h=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(65536),b''):h.update(b)
   if h.hexdigest()!=expected_hash:raise AuxiliaryLimit('archive member readback digest')
 h=hashlib.sha256()
 with destination.open('rb') as f:
  for b in iter(lambda:f.read(65536),b''):h.update(b)
 return {'bytes':destination.stat().st_size,'sha256':h.hexdigest(),'members':len(names),'full_checkpoint_bodies_included':False}
