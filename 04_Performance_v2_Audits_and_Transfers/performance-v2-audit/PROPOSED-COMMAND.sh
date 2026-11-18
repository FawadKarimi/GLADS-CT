#!/usr/bin/bash
# REVIEW ONLY: do not execute until the owner separately authorizes this run.
# First upload the exact ZIP named below to the existing Jupyter audit directory.
set -eu
umask 077
/experiment-storage/chaidir/env/env-310-fawad/bin/python -I -B -S - <<'PV2_PREPARE'
import hashlib,io,json,os,pathlib,stat,subprocess,zipfile
P=pathlib.Path
os.umask(0o077)
assert os.getuid()==1009
home=P('/home/chaidir')
for p in (P('/'),P('/home'),home):
    s=p.lstat()
    assert stat.S_ISDIR(s.st_mode) and not p.is_symlink() and p.resolve()==p
    assert stat.S_IMODE(s.st_mode)==0o755 and s.st_uid==(1009 if p==home else 0)
    acl=subprocess.check_output(['/usr/bin/getfacl','-cp','--',str(p)],text=True)
    assert [x for x in acl.splitlines() if x and not x.startswith('#')]==['user::rwx','group::r-x','other::r-x']
destination=home/'.performance-v2-reader-correction-20261008'
assert not os.path.lexists(destination),'private destination already exists; do not overwrite/retry'
uploaded=P('/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit/performance-v2-reader-correction-REVIEW-20261008.zip')
before=uploaded.lstat()
assert stat.S_ISREG(before.st_mode) and before.st_size<=128*1024**2 and not uploaded.is_symlink()
fd=os.open(uploaded,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
    opened=os.fstat(fd)
    assert stat.S_ISREG(opened.st_mode) and (opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino)
    with os.fdopen(fd,'rb',closefd=False) as f: raw=f.read(128*1024**2+1)
finally: os.close(fd)
assert len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()=='da87378dd395af6594f1d57335538ac83739261c83233d4ed3019017cc944967'
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    info=z.infolist()
    assert len(info)==78 and len({x.filename for x in info})==78
    assert sum(x.file_size for x in info)<=128*1024**2
    for x in info:
        name=x.filename
        assert '\\' not in name and ':' not in name and not name.startswith('/')
        assert all(part not in ('','.','..') for part in name.split('/'))
        assert not x.is_dir() and stat.S_IFMT(x.external_attr>>16) in (0,stat.S_IFREG)
        assert x.file_size<=16*1024**2
    ledger_raw=z.read('PACKAGE-LEDGER.json')
    assert hashlib.sha256(ledger_raw).hexdigest()=='64da3b906e20e80bed89aa407dfea0ae928faa218ab1fcc5e5077903138914be'
    ledger=json.loads(ledger_raw)
    assert set(z.namelist())==set(ledger['files'])|{'PACKAGE-LEDGER.json'}
    for name,want in ledger['files'].items(): assert hashlib.sha256(z.read(name)).hexdigest()==want,name
    destination.mkdir(mode=0o700)
    for x in info:
        p=destination/x.filename
        p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with p.open('xb') as f: f.write(z.read(x.filename))
for p in destination.rglob('*'):
    s=p.lstat(); assert not p.is_symlink() and p.resolve()==p and s.st_uid==1009
    assert stat.S_IMODE(s.st_mode)==(0o700 if p.is_dir() else 0o600)
source=destination/'ConsenDxFormer-performance-v2'
lock_path=source/'configs/performance_v2/runner_integration_source_lock_v1.json'
assert hashlib.sha256(lock_path.read_bytes()).hexdigest()=='aed4e1dd5b2f82accedcfdec9082f5ed8ce75fed47169d04310d6e44952fa378'
lock=json.loads(lock_path.read_bytes()); assert len(lock['files'])==65
for name,want in lock['files'].items(): assert hashlib.sha256((source/name).read_bytes()).hexdigest()==want,name
run=destination/'performance-v2-reader-correction-20261008/RUN-LINUX.py'
assert hashlib.sha256(run.read_bytes()).hexdigest()=='1fdd1c0b812ce604b868bf724b45776892f41b04569a4763dbc461f5b63d8452'
print('Verified new private candidate; 65 source bindings; historical copy untouched.')
PV2_PREPARE
PV2_INTEGRATION_SYNTHETIC_AUTH=reviewed-reader-integration-001 \
PV2_TRUST_REVIEW=trusted-code-no-same-uid-interference-v1 \
  /experiment-storage/chaidir/env/env-310-fawad/bin/python -I -B -S \
  /home/chaidir/.performance-v2-reader-correction-20261008/performance-v2-reader-correction-20261008/RUN-LINUX.py
