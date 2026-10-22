"""Inactive bounded wrapper reusing the accepted managed-group pattern.

No retry/repair. This controls only its new session/process group. Python/path
guards are accidental-access controls under trusted-code/shared-UID assumptions.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

PYTHON = '/experiment-storage/chaidir/env/env-310-fawad/bin/python'
ROOT = Path('/home/chaidir/.performance-v2-r50-preparation-20261011')
REPORTS = Path('/home/chaidir/.performance-v2-reports')
LOCK = '6ff5134ee531ff11f1755453063323f07579a9c04b97d4e6296e41f2bdc50529'


class ManagedInterruption(BaseException):
    def __init__(self, number): self.number=number


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(path, cap=4*1024**2):
    if any(p.is_symlink() for p in (path,*path.parents)): raise ValueError('symlink')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size > cap
            or before.st_uid!=os.getuid() or stat.S_IMODE(before.st_mode)!=0o600):
        raise ValueError('private bounded regular file')
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode) or (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino)):
            raise ValueError('descriptor identity')
        with os.fdopen(fd,'rb',closefd=False) as f: raw=f.read(cap+1)
        if len(raw)>cap: raise ValueError('size')
        return raw
    finally: os.close(fd)


def private(path, strict=False):
    observations=[]
    for p in reversed((path,*path.parents)):
        s=p.lstat()
        if (not stat.S_ISDIR(s.st_mode) or s.st_uid not in (0,os.getuid())
                or s.st_mode&0o022 or p.is_symlink()): raise ValueError('unsafe directory')
        acl=subprocess.run(['/usr/bin/getfacl','-cp',str(p)],capture_output=True,check=True,timeout=5).stdout.decode()
        if any(x and not x.startswith('#') and not x.startswith(('user::','group::','other::')) for x in acl.splitlines()):
            raise ValueError('nonbasic ACL')
        observations.append({'path':str(p),'uid':s.st_uid,'mode':oct(stat.S_IMODE(s.st_mode)),
                             'device':s.st_dev,'acl':acl.splitlines()})
    if strict and (path.stat().st_uid!=os.getuid() or stat.S_IMODE(path.stat().st_mode)!=0o700):
        raise ValueError('private 0700 owner required')
    return observations


def process_group(pgid):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            if path.stat().st_uid != os.getuid(): continue
            raw=(path/'stat').read_bytes()
            if len(raw)>8192: raise ValueError('proc record bound')
            f=raw.decode().split(') ',1)[1].split()
            if int(f[2])==pgid:
                result.append({'pid':int(path.name),'pgid':pgid,'starttime_ticks':int(f[19]),
                               'state':f[0],'rss_bytes':int(f[21])*os.sysconf('SC_PAGE_SIZE')})
        except (FileNotFoundError,ProcessLookupError): pass
    return result


RECORD_FIELDS={'schema','execution_authorized','interpreter','source_root','source_lock_sha256',
 'spec_path','spec_sha256','device_binding','image_root','mapping_root','mapping_approval_path',
 'mapping_approval_sha256','transform_sha256','authorization_path','authorization_sha256',
 'decision_sha256','budget_root','budget_plan_sha256','external_budget_record_path',
 'external_budget_record_sha256','runtime_inventory_path','runtime_inventory_sha256',
 'report_root','fresh_gpu_row','shared_gpu_permission_path','shared_gpu_permission_sha256',
 'resume_parent_attempt','minimum_free_mib','filesystem_description_sha256'}
def validate_record(r):
    from .training_runner import validate_device_binding
    if set(r)!=RECORD_FIELDS or r['schema']!='pv2_full_training_launch_v1': raise ValueError('exact launch schema')
    validate_device_binding(r['device_binding'])
    if (r['interpreter']!=PYTHON or r['device_binding']['index']!=0
        or r['device_binding']['uuid']!='GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90'
        or r['minimum_free_mib']!=10240): raise ValueError('reviewed device/runtime bounds')
    if r['resume_parent_attempt'] is not None and (type(r['resume_parent_attempt']) is not int or r['resume_parent_attempt']<1):
        raise ValueError('explicit resume parent required')
def verify_sources(r):
    root=Path(r['source_root']); lock=read(root/'source-lock.json')
    if sha(lock)!=r['source_lock_sha256']: raise ValueError('source lock drift')
    entries=json.loads(lock)
    for name,h in entries.items():
        if name.startswith('/') or '..' in Path(name).parts or '\\' in name: raise ValueError('source path rejected')
        if sha(read(root/'candidate'/name))!=h: raise ValueError('source drift')
    return {'matching':True,'bindings':len(entries)}
def authorization(raw,*,spec_sha256,decision):
    from datetime import datetime,timezone
    import re
    a=json.loads(raw)
    if (set(a)!={'schema','authorized','scope','spec_sha256','decision_sha256','not_before','expires'}
        or a['schema']!='pv2_training_single_attempt_v1' or a['authorized'] is not True
        or a['scope']!='TRAIN_VAL' or a['spec_sha256']!=spec_sha256 or a['decision_sha256']!=decision):
        raise PermissionError('exact active authorization required')
    if any(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',a[k] or '') is None for k in ('not_before','expires')):
        raise PermissionError('exact UTC Z required')
    start,end=[datetime.strptime(a[k],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc) for k in ('not_before','expires')]
    if not 0<(end-start).total_seconds()<=900 or not start<=datetime.now(timezone.utc)<end: raise PermissionError('expired window')
def verify_external(r,budget):
    raw=read(Path(r['external_budget_record_path']))
    if sha(raw)!=r['external_budget_record_sha256']: raise ValueError('external reconciliation digest')
    x=json.loads(raw)
    if (set(x)!={'schema','approved','retained_location','snapshot','plan_sha256','spec_sha256','decision_sha256'}
        or x['schema']!='pv2_independent_budget_reconciliation_v1' or x['approved'] is not True
        or not x['retained_location'] or x['snapshot']!=budget.inspect()
        or x['snapshot']['issues'] or x['plan_sha256']!=r['budget_plan_sha256']
        or x['spec_sha256']!=r['spec_sha256'] or x['decision_sha256']!=r['decision_sha256']):
        raise ValueError('fresh independent budget reconciliation required')
    # A local byte match is not external anti-rollback verification. The signed-off
    # retained location must be independently checked before each activation.
