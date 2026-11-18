"""One fresh seed-42 activation. Existing safeguards/source/specifications stay frozen."""
import os,sys,json,hashlib,base64,subprocess,datetime,copy
from pathlib import Path
ROOT=Path('/home/chaidir/.performance-v2-r50-matched-training-20261012')
PYTHON='/experiment-storage/chaidir/env/env-310-fawad/bin/python'
UUID='GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90'
SOURCE_LOCK='5d4a14af41afa97ba7cdb782381d9e76a01fe98680e8185bcce4a1fe838498a8'
EXTERNAL={'RANDOM':{'pin':'8319517252adfea8a2e9cac24f363e615c6fc8ddb7db918f11c6cd8bab03d837','base64':'eyJhdHRlbXB0cyI6W10sImF1ZGl0X3RhaWxfc2hhMjU2IjoiMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMCIsImNvbnN1bWVkIjowLCJldmVudF9jb3VudCI6MCwiaWRlbnRpdHkiOiI5ZDhlYThiOGZkOGM0NGQ0OGFkMjc2Yjk2M2VjZDdjYSIsImlzc3VlcyI6W10sImxpbWl0IjozLCJwZW5kaW5nX3JlY2VpcHRfaGFzaGVzIjp7fSwicGlsb3RfZXhlY3V0aW9uX2F1dGhvcml6ZWQiOmZhbHNlLCJwbGFuX3NoYTI1NiI6ImIyYTIxYzE2ODNkNTczNTM2OGNjZjBjMDhiMmUzZTZlOThjYWZjMGVjNjRjNTNmZTI5OTIwOTRjMjZiODU4MDgiLCJyZXRhaW5lZF9ldmlkZW5jZV9oYXNoZXMiOnt9LCJzY2hlbWEiOiJwZXJmb3JtYW5jZV92Ml9wZXJzaXN0ZW50X2J1ZGdldF9zbmFwc2hvdF92MSIsInN0b3JhZ2VfcHJvZmlsZSI6Im5hdGl2ZV9kaXJlY3RvcnlfZnN5bmMifQ=='},
          'IMAGENET':{'pin':'62200193a5d85477de14d3fa4fb2fa9e2a4ce3fa17735a0d06008d55652877d8','base64':'eyJhdHRlbXB0cyI6W10sImF1ZGl0X3RhaWxfc2hhMjU2IjoiMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMCIsImNvbnN1bWVkIjowLCJldmVudF9jb3VudCI6MCwiaWRlbnRpdHkiOiJlYjhiMDhiYWIwMjg0NjhmOTIxNjNhODI4ZWFlZjU0YiIsImlzc3VlcyI6W10sImxpbWl0IjozLCJwZW5kaW5nX3JlY2VpcHRfaGFzaGVzIjp7fSwicGlsb3RfZXhlY3V0aW9uX2F1dGhvcml6ZWQiOmZhbHNlLCJwbGFuX3NoYTI1NiI6Ijg4ZGE1OWUzN2NkZTQ3ZmY2YjRmMjg0Y2NlNWU5YjhmZDJlYWEyMjJlMjZiNzdkZWIxNzI0NjI0MmViODg2ZjQiLCJyZXRhaW5lZF9ldmlkZW5jZV9oYXNoZXMiOnt9LCJzY2hlbWEiOiJwZXJmb3JtYW5jZV92Ml9wZXJzaXN0ZW50X2J1ZGdldF9zbmFwc2hvdF92MSIsInN0b3JhZ2VfcHJvZmlsZSI6Im5hdGl2ZV9kaXJlY3RvcnlfZnN5bmMifQ=='}}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def records(base,decision,permission,snapshot,folder,location,row,now,evidence):
    r=copy.deepcopy(base);d=copy.deepcopy(decision);p=copy.deepcopy(permission)
    condition=d['condition'];assert condition in ('RANDOM','IMAGENET') and d['seed']==42 and d['objective']=='HARD'
    assert r['execution_authorized'] is False and r['resume_parent_attempt'] is None
    assert snapshot['consumed']==0 and snapshot['attempts']==[] and snapshot['issues']==[] and snapshot['event_count']==0 and snapshot['audit_tail_sha256']=='0'*64
    assert snapshot['plan_sha256']==r['budget_plan_sha256'] and snapshot['limit']==3
    d.update(budget_initialization_authorized=True,full_training_execution_authorized=True,
      scope='one-'+condition+'-seed42-HARD-TRAIN-VAL-fresh-attempt',approved_by='human user; direct recorded chat authorization',
      authorization_basis_sha256='05385decba2e8c29ea01277e254ba8b2aa5fecff6f34493e306eb2c3ec8f8e90',
      gate_evidence=evidence,gpu_resume_clearance_claimed=False)
    decision_sha=sha(canonical(d))
    p.update(user_attested_permission=True,basis='User explicitly confirmed existing Dr. Chaidir sharing permission covers prolonged full TRAIN/VAL jobs; this is a run-bound disposition, not a new administrator grant or independently authenticated signature')
    assert p['gpu_uuid']==UUID and p['spec_sha256']==r['spec_sha256'] and p['report_root']==r['report_root']
    assert p['scope']=='single-'+condition+'-HARD-TRAIN-VAL'
    x={'schema':'pv2_independent_budget_reconciliation_v1','approved':True,'retained_location':location,
       'snapshot':snapshot,'plan_sha256':r['budget_plan_sha256'],'spec_sha256':r['spec_sha256'],'decision_sha256':decision_sha}
    start=now.replace(microsecond=0)
    a={'schema':'pv2_training_single_attempt_v1','authorized':True,'scope':'TRAIN_VAL','spec_sha256':r['spec_sha256'],
      'decision_sha256':decision_sha,'not_before':start.strftime('%Y-%m-%dT%H:%M:%SZ'),
      'expires':(start+datetime.timedelta(seconds=900)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    r.update(execution_authorized=True,fresh_gpu_row=row,decision_sha256=decision_sha,
      authorization_path=str(folder/'authorization.ACTIVE.json'),authorization_sha256=sha(canonical(a)),
      external_budget_record_path=str(folder/'external-budget.ACTIVE.json'),external_budget_record_sha256=sha(canonical(x)),
      shared_gpu_permission_path=str(folder/'permission.ACTIVE.json'),shared_gpu_permission_sha256=sha(canonical(p)))
    return {'decision.ACTIVE.json':d,'permission.ACTIVE.json':p,'external-budget.ACTIVE.json':x,'authorization.ACTIVE.json':a,'launch.ACTIVE.json':r}
def main(condition):
    assert condition in EXTERNAL and sys.executable==PYTHON and os.getuid()==1009
    sys.path.insert(0,str(ROOT/'candidate/src'))
    from consendxformer.performance_v2.launch_security import private,read,verify_sources,validate_record,authorization,verify_external
    from consendxformer.performance_v2.durable_budget import PersistentBudget
    private(ROOT,True);private(Path('/home/chaidir/.performance-v2-reports'),True)
    r=json.loads(read(ROOT/'records/launches'/(condition+'-42.launch.INACTIVE.json')))
    verify_sources(r);assert r['source_lock_sha256']==SOURCE_LOCK
    spec_raw=read(Path(r['spec_path']));spec=json.loads(spec_raw);assert sha(canonical(spec))==r['spec_sha256']
    assert spec['contract']['seed']==42 and spec['contract']['objective']=='HARD' and spec['contract']['initialization']==condition
    assert spec['pilot_execution_authorized'] is False
    if condition=='IMAGENET':private(Path('/home/chaidir/.performance-v2-imagenet-receiver-20261012/artifacts'),True)
    inventory=subprocess.run([PYTHON,'-I','-B',r['runtime_inventory_path']],capture_output=True,check=True,timeout=15).stdout
    assert sha(inventory)==r['runtime_inventory_sha256']
    # Static library/source observations only. No CUDA context, tensor or model.
    meta="import torch,json,hashlib;from pathlib import Path;print(json.dumps({'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cuda_source_sha256':hashlib.sha256(Path(torch.cuda.__file__).read_bytes()).hexdigest()}))"
    m=subprocess.run([PYTHON,'-I','-B','-c',meta],capture_output=True,check=True,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES='')).stdout
    observed=json.loads(m);assert observed=={'cuda_runtime':'12.1','cudnn':8902,'cuda_source_sha256':'1183ef48a38265aec807bd4486ad75cbb0678374b02fdb5ef8b8a65abad4d788'}
    space=os.statvfs('/home/chaidir/.performance-v2-reports');assert space.f_bavail*space.f_frsize>=60*1024**3 and space.f_favail>=2000
    fs=subprocess.run(['findmnt','-n','-o','FSTYPE,TARGET','-T','/home/chaidir/.performance-v2-reports'],capture_output=True,check=True,timeout=5).stdout
    assert sha(fs)==r['filesystem_description_sha256']
    rows=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],capture_output=True,check=True,timeout=10).stdout.decode().splitlines()
    selected=[row for row in rows if row.split(',')[0].strip()==UUID];assert len(selected)==1
    row=selected[0];assert int(row.split(',')[3].strip())>=10240 and row.split(',')[4].strip()==r['device_binding']['driver']
    resident=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,check=True,timeout=10).stdout.decode()
    private(Path(r['budget_root']),True);budget=PersistentBudget(r['budget_root'],expected_plan_sha256=r['budget_plan_sha256'])
    external_raw=base64.b64decode(EXTERNAL[condition]['base64'],validate=True);assert sha(external_raw)==EXTERNAL[condition]['pin']
    snapshot=json.loads(external_raw);assert budget.inspect()==snapshot
    reports=Path('/home/chaidir/.performance-v2-reports');folder=reports/('r50-'+condition.lower()+'-42-full-activation-001-20261012')
    outer=reports/('r50-'+condition.lower()+'-42-full-outer-001-20261012')
    for path in (folder,outer,Path(r['report_root'])):assert not path.exists() and not path.is_symlink()
    cpu=reports/'r50-checkpoint-cpu-synthetic-001-20261012';entry=reports/'r50-entry-six-cpu-synthetic-001-20261012'
    result=json.loads(read(cpu/'results.json'));exits=json.loads(read(cpu/'exit.json'));final=json.loads(read(cpu/'finalization.json'))
    assert len(result['checks'])==5 and all(v['pass'] is True for v in result['checks']) and exits['supervisor_status']==exits['worker_status']==0
    assert final['complete'] and not final['errors'] and json.loads(read(cpu/'processes.json'))==[]
    entry_log=read(entry/'stdout.log').decode();entry_result=json.loads(next(s.split('=',1)[1] for s in entry_log.splitlines() if s.startswith('ENTRY_RESULT=')))
    assert entry_result['tests']==6 and all(entry_result[k]==0 for k in ('failures','errors','skips','forwards','optimizer_updates','gpu_calls','actual_budgets_created'))
    assert json.loads(read(entry/'exit.json'))['status']==0 and json.loads(read(entry/'processes.json'))==[]
    assert not json.loads(read(entry/'finalization.json'))['errors']
    evidence={'cpu_checkpoint_report':str(cpu),'cpu_checkpoint_results_sha256':sha(read(cpu/'results.json')),
      'cpu_checkpoint_ledger_sha256':sha(read(cpu/'EVIDENCE-LEDGER.json')),'native_entry_report':str(entry),
      'native_entry_stdout_sha256':sha(read(entry/'stdout.log')),'native_entry_exit_sha256':sha(read(entry/'exit.json')),
      'fresh_static_runtime':observed,'resident_rows':[s for s in resident.splitlines() if s.split(',')[0].strip()==UUID],
      'external_zero_snapshot_sha256':EXTERNAL[condition]['pin'],'no_gpu_resume_clearance':True}
    d=json.loads(read(ROOT/'records/decisions'/(condition+'-42.decision.INACTIVE.json')))
    p=json.loads(read(ROOT/'records/permissions'/(condition+'-42.permission.INACTIVE.json')))
    location='C:\\Users\\Lenovo y520\\Documents\\Playground\\performance-v2-readiness-execution-20261012\\external-budget-retention\\'+condition+'-zero-snapshot.json'
    values=records(r,d,p,snapshot,folder,location,row,datetime.datetime.now(datetime.timezone.utc),evidence)
    validate_record(values['launch.ACTIVE.json'])
    authorization(canonical(values['authorization.ACTIVE.json']),spec_sha256=r['spec_sha256'],decision=values['launch.ACTIVE.json']['decision_sha256'])
    os.umask(0o077);folder.mkdir(mode=0o700);private(folder,True)
    for name,value in values.items():
        with (folder/name).open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
    verify_external(values['launch.ACTIVE.json'],budget)
    for name,value in values.items():assert read(folder/name)==canonical(value)
    with (folder/'preflight-observations.json').open('xb') as f:f.write(canonical(evidence));f.flush();os.fsync(f.fileno())
    pin=sha(read(folder/'launch.ACTIVE.json'))
    command='REVIEWED_EXECUTION_BINDING_SHA256='+pin+' '+PYTHON+' -I -B '+str(ROOT/'candidate/src/consendxformer/performance_v2/supervisor_bootstrap.py')+' '+str(folder/'launch.ACTIVE.json')
    print('ACTIVATION',json.dumps({'condition':condition,'folder':str(folder),'record_sha256':pin,'authorization':values['authorization.ACTIVE.json'],'fresh_gpu_row':row,'budget_consumed':0,'launch_command':command}))
    print('ACTIVATION_VERIFIED_EXIT_0')
if __name__=='__main__':main(sys.argv[1])
