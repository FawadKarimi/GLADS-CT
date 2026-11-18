"""Exclusive deployment plus one CPU synthetic attempt; stdlib supervisor only."""
from pathlib import Path, PurePosixPath
import hashlib, io, json, os, resource, signal, stat, subprocess, sys, time, zipfile

upload=Path('/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit')
source=Path('/home/chaidir/.performance-v2-r50-preparation-20261011')
report=Path('/home/chaidir/.performance-v2-reports/r50-cpu-integration-001')
interpreter='/experiment-storage/chaidir/env/env-310-fawad/bin/python'
ZIP_SHA='1174d273d6fd5b9c44fffaecfa380470a50ce72574b7a8f41c9f35c111a08e9e'
LOCK_SHA='6ff5134ee531ff11f1755453063323f07579a9c04b97d4e6296e41f2bdc50529'
HARNESS_SHA='774b873751e3f0f86f168c26b11d37f796b3d66df6e5fc0919b5ac2bf0e6c0b1'
sha=lambda b:hashlib.sha256(b).hexdigest()
os.umask(0o077)
assert os.getuid()==1009
def read(path,cap):
    s=path.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=cap
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        t=os.fstat(fd);assert (s.st_dev,s.st_ino,s.st_size)==(t.st_dev,t.st_ino,t.st_size) and stat.S_ISREG(t.st_mode)
        with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(cap+1)
        assert len(b)==s.st_size and len(b)<=cap
        return b
    finally:os.close(fd)
def directory(path,private=False):
    s=path.lstat();assert stat.S_ISDIR(s.st_mode) and s.st_uid in (0,1009) and not s.st_mode&0o022
    if private:assert s.st_uid==1009 and stat.S_IMODE(s.st_mode)==0o700
    p=subprocess.run(['/usr/bin/getfacl','-cp',str(path)],capture_output=True,text=True,timeout=5)
    assert p.returncode==0 and len(p.stdout)<8192
    lines=[x for x in p.stdout.splitlines() if x and not x.startswith('#')]
    assert all(x.startswith(('user::','group::','other::')) for x in lines),('ACL rejected',str(path))
    return {'path':str(path),'uid':s.st_uid,'gid':s.st_gid,'mode':oct(stat.S_IMODE(s.st_mode)),'acl':lines,'device':s.st_dev}
checks=[]
for path in (Path('/'),Path('/home'),Path('/home/chaidir'),report.parent):checks.append(directory(path,path==report.parent))
assert not os.path.lexists(source) and not os.path.lexists(report)
v=os.statvfs(report.parent);assert v.f_bavail*v.f_frsize>=1024**3 and v.f_favail>=1000
archive=read(upload/'performance-v2-r50-production-preparation-REVIEW-20261011.zip',2*1024**2)
assert sha(archive)==ZIP_SHA
harness=read(upload/'native_cpu.py',65536);assert sha(harness)==HARNESS_SHA
stage_bytes=read(Path(__file__),65536)
assert sha(stage_bytes)==os.environ['REVIEWED_STAGE_SHA256']
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    info=z.infolist();names=[i.filename for i in info]
    assert len(names)==len(set(names))==302 and sum(i.file_size for i in info)<=16*1024**2
    assert z.testzip() is None
    for i in info:
        p=PurePosixPath(i.filename)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and not i.is_dir()
        assert not stat.S_ISLNK(i.external_attr>>16)
    ledger=json.loads(z.read('SHA256-LEDGER.json'));assert len(ledger)==301
    assert set(names)==set(ledger)|{'SHA256-LEDGER.json'}
    assert all(sha(z.read(n))==h for n,h in ledger.items())
    lock_bytes=z.read('source-lock.json');assert sha(lock_bytes)==LOCK_SHA
    lock=json.loads(lock_bytes);assert len(lock)==140
    source.mkdir(mode=0o700)
    for n in names:
        dst=source/n;dst.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(z.read(n))
with (source/'native_cpu.py').open('xb') as f:f.write(harness)
with (source/'stage_and_cpu.py').open('xb') as f:f.write(stage_bytes)
assert all(sha(read(source/'candidate'/n,4*1024**2))==h for n,h in lock.items())
for p in source.rglob('*'):
    if p.is_dir():directory(p,True)
    else:
        s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1009 and stat.S_IMODE(s.st_mode)==0o600
checks.append(directory(source,True))
assert Path(interpreter).exists()
report.mkdir(mode=0o700);(report/'tmp').mkdir(mode=0o700)
def write(name,value):
    raw=value if type(value) is bytes else json.dumps(value,indent=2).encode()
    with (report/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
write('staging.json',{'zip_sha256':ZIP_SHA,'source_lock_sha256':LOCK_SHA,'ledger_entries':301,
                     'candidate_bindings':140,'source':str(source),'checks':checks,
                     'harness_sha256':HARNESS_SHA,'stage_sha256':sha(stage_bytes),
                     'production_execution_authorized':False})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(report/'tmp'))
def limits():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(300,300))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
status=125;errors=[];child=None;timed_out=False;start=time.monotonic()
def processes():
    found=[]
    for p in list(Path('/proc').iterdir())[:4096]:
        if not p.name.isdigit() or not child:continue
        try:
            if p.stat().st_uid!=1009:continue
            with (p/'stat').open('rb') as f:raw=f.read(8193)
            assert len(raw)<=8192
            fields=raw.decode().split(') ',1)[1].split()
            if fields[2]==str(child.pid):
                found.append({'pid':int(p.name),'ppid':int(fields[1]),'pgid':int(fields[2]),
                              'state':fields[0],'starttime_ticks':int(fields[19])})
        except (FileNotFoundError,ProcessLookupError):continue
    return {'managed_pgid':child.pid if child else None,'remaining':found}
try:
    with (report/'stdout.log').open('xb') as out,(report/'stderr.log').open('xb') as err:
        child=subprocess.Popen([interpreter,'-I','-B',str(source/'native_cpu.py'),str(source),str(report)],
                  stdout=out,stderr=err,env=env,start_new_session=True,preexec_fn=limits)
        write('child.json',{'pid':child.pid,'pgid':child.pid,'argv':[interpreter,'-I','-B',str(source/'native_cpu.py'),str(source),str(report)]})
        try:status=child.wait(timeout=360)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
            status=124
except BaseException as error:
    errors.append({'stage':'supervisor','category':type(error).__name__})
    if isinstance(error,KeyboardInterrupt):status=130
    if child is not None and child.poll() is None:
        try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
finally:
    for name,fn in (
        ('exit.json',lambda:{'exit_status':status,'child_status':child.poll() if child else None,'timeout':timed_out,'seconds':time.monotonic()-start}),
        ('post-source.json',lambda:{'matching':all(sha(read(source/'candidate'/n,4*1024**2))==h for n,h in lock.items()),'bindings':len(lock)}),
        ('processes.json',processes)):
        try:write(name,fn())
        except BaseException as error:errors.append({'stage':name,'category':type(error).__name__})
    try:write('finalization-errors.json',errors)
    except BaseException:pass
    try:
        evidence={p.name:sha(p.read_bytes()) for p in report.iterdir() if p.is_file()}
        write('EVIDENCE-LEDGER.json',evidence)
    except BaseException:pass
print(json.dumps({'source':str(source),'report':str(report),'exit_status':status,'finalization_errors':errors,
 'result':json.loads((report/'results.json').read_bytes()) if (report/'results.json').is_file() else None},indent=2),flush=True)
sys.exit(status if status else (1 if errors else 0))
