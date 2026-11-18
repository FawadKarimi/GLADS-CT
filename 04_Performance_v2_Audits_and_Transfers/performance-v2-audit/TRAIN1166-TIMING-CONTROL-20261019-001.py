"""One authorized timing budget and activation; no model import or training here."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
R=Path('/home/chaidir/.performance-v2-train1166-timing-20261019-001');A=R/'timing-addon';D=R/'records'
sys.path[:0]=[str(A),str(R/'candidate/src')]
from bounded_supervisor import read,private,sha,REPORTS,PYTHON,LOCK
from consendxformer.performance_v2.reuse import canonical,digest
from consendxformer.performance_v2.durable_budget import PersistentBudget
from timing_binding import validate_records
B=Path('/home/chaidir/.performance-v2-scratch/train1166-timing-budget-001-20261019')
IN=Path('/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/04_Performance_v2_Audits_and_Transfers/performance-v2-audit')
def put(path,obj):
 raw=obj if isinstance(obj,bytes) else canonical(obj)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return digest(raw)
def gate():
 private(R,True);private(A,True);private(REPORTS,True);private(B.parent,True)
 space=os.statvfs('/home/chaidir');assert space.f_bavail*space.f_frsize>=20*1024**3 and space.f_favail>=1000
 sl=read(R/'source-lock.json');assert sha(sl)==LOCK
 for n,h in json.loads(sl).items():assert sha(read(R/'candidate'/n))==h
 for n,h in json.loads(read(A/'SOURCE-LOCK.json')).items():assert sha(read(A/n))==h
 inv=subprocess.run([PYTHON,'-I','-B',str(A/'runtime_inventory.py')],capture_output=True,check=True,timeout=15).stdout
 assert inv==read(D/'runtime-inventory.json')
 rows=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],capture_output=True,check=True,timeout=10).stdout.decode().splitlines()
 binding=json.loads(read(D/'execution-spec.json'))['device_binding'];sel=[s for s in rows if s.split(',')[0].strip()==binding['uuid']];assert len(sel)==1
 row=sel[0];fields=[s.strip() for s in row.split(',')]
 observation=dict(utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),row=row,minimum_free_mib=10240,home_available_bytes=space.f_bavail*space.f_frsize,inodes=space.f_favail)
 # Retain exact admission row before evaluating assertions.
 put(D/('GPU-'+sys.argv[1]+'.json'),observation)
 assert fields[1]==binding['name'] and fields[4]==binding['driver'] and int(fields[3])>=10240
 return row
os.umask(0o077)
mode=sys.argv[1];row=gate();plan=read(D/'plan.json');psha=digest(plan)
if mode=='budget-zero':
 budget=PersistentBudget.create(B,plan_bytes=plan,expected_plan_sha256=psha);snapshot=budget.inspect()
 assert snapshot['consumed']==0 and not snapshot['attempts'] and not snapshot['issues'] and snapshot['event_count']==0
 zero=dict(schema='pv2_timing_actual_zero_retention_v1',snapshot=snapshot,budget_root=str(B),plan_sha256=psha)
 raw=canonical(zero);put(D/'ZERO-SNAPSHOT.json',raw);put(IN/'TRAIN1166-TIMING-ZERO-20261019-001.json',raw)
 print(json.dumps(dict(zero_snapshot_sha256=digest(raw),snapshot=snapshot,budget_root=str(B),reserved=False)),flush=True)
elif mode=='activate':
 expected=sys.argv[2];location=sys.argv[3];zero_raw=read(D/'ZERO-SNAPSHOT.json');assert digest(zero_raw)==expected
 zero=json.loads(zero_raw);budget=PersistentBudget(B,expected_plan_sha256=psha);snapshot=budget.inspect();assert snapshot==zero['snapshot'] and snapshot['consumed']==0 and not snapshot['attempts'] and not snapshot['issues']
 specraw=read(D/'execution-spec.json');specsha=digest(specraw);spec=json.loads(specraw);binding=spec['device_binding'];runtime_raw=read(D/'RUNTIME-ACCEPTANCE.json')
 external=dict(schema='pv2_timing_external_budget_v1',approved=True,external_record_location=location,operator='Codex technical retention under explicit user single-probe authorization; no invented independent reviewer',verified_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),budget_root=str(B),plan_sha256=psha,snapshot_sha256=digest(canonical(snapshot)),**{k:snapshot[k] for k in ('identity','audit_tail_sha256','event_count','consumed')})
 esha=put(D/'EXTERNAL-RECONCILIATION.json',external)
 report=REPORTS/'train1166-timing-attempt-001-20261019';assert not report.exists() and not report.is_symlink()
 decision=dict(schema='pv2_train1166_timing_owner_decision_v1',operation='ONE_PROBE_ONLY',source_lock_sha256=LOCK,spec_sha256=specsha,plan_sha256=psha,initialization='original_RADIMAGENET',objective='HARD',seed=42,warmup=5,measured=20,batch=16,distinct_TRAIN=400,worker_seconds=300,outer_seconds=420,full_training=False,TEST=False,VAL=False,no_retry=True,stage_b_paused=True,authorization_basis='User explicit single timing/memory probe instruction; continued via try now and try chrome')
 decsha=put(D/'DECISION.json',decision)
 f=lambda d:d.strftime('%Y-%m-%dT%H:%M:%SZ');now=datetime.now(timezone.utc)
 auth=dict(schema='pv2_training_only_timing_single_attempt_v1',authorized=True,scope='TRAIN_ONLY',spec_sha256=specsha,decision_sha256=decsha,not_before=f(now),expires=f(now+timedelta(minutes=10)),warmup=5,measured=20,batch_size=16,precision='FP32');asha=put(D/'AUTHORIZATION.ACTIVE.json',auth)
 permission=dict(schema='pv2_admin_shared_gpu_timing_v1',administrator='Dr. Chaidir',user_attested_permission=True,gpu_uuid=binding['uuid'],scope='RADIMAGENET-42-HARD-TRAIN1166-ONLY-TIMING',report_root=str(report),production_execution_authorized=False,basis='Existing user-attested permission covers full training on GPU1; this is the narrower authorized TRAIN-only timing operation; no new administrator grant claimed')
 permsha=put(D/'SHARED-GPU-PERMISSION.json',permission)
 path=D/'timing-binding.ACTIVE.json'
 record=dict(schema='pv2_train1166_timing_launch_v1',execution_authorized=True,interpreter=PYTHON,source_root=str(R),source_lock_sha256=LOCK,adapter_root=str(A),adapter_lock_sha256=sha(read(A/'SOURCE-LOCK.json')),spec_path=str(D/'execution-spec.json'),spec_sha256=specsha,device_binding=binding,image_root='/experiment-storage/chaidir/fawad-dsai/GLADS_CT Project/03_ConsenDxFormer_and_Datasets/ConsenDxFormer',mapping_root=str(R/'mapping'),mapping_approval_path=str(D/'MAPPING-APPROVAL.TIMING.json'),mapping_approval_sha256=sha(read(D/'MAPPING-APPROVAL.TIMING.json')),transform_sha256=json.loads(read(R/'input-lock.PROPOSED.json'))['transform_sha256'],authorization_path=str(D/'AUTHORIZATION.ACTIVE.json'),authorization_sha256=asha,decision_sha256=decsha,budget_root=str(B),budget_plan_sha256=psha,external_budget_record_path=str(D/'EXTERNAL-RECONCILIATION.json'),external_budget_record_sha256=esha,runtime_acceptance_path=str(D/'RUNTIME-ACCEPTANCE.json'),runtime_acceptance_sha256=digest(runtime_raw),runtime_inventory_sha256=sha(read(D/'runtime-inventory.json')),report_root=str(report),fresh_gpu_row=row,shared_gpu_permission_path=str(D/'SHARED-GPU-PERMISSION.json'),shared_gpu_permission_sha256=permsha,minimum_free_mib=10240,filesystem_description_sha256=sha(subprocess.run(['findmnt','-n','-o','FSTYPE,TARGET','-T',str(REPORTS)],capture_output=True,check=True,timeout=5).stdout),binding_path=str(path),supervisor_argv=[PYTHON,'-I','-B',str(A/'bounded_supervisor.py'),'timing',str(path)],worker_argv=[PYTHON,'-I','-B',str(A/'timing_worker.py'),str(R),str(path)],external_budget_anchor_reviewed=True,pilot_execution_authorized=False,production_execution_authorized=False,allowed_train_ids=json.loads(read(D/'TRAIN400.json'))['ids'],operation_id='TRAIN1166-RAD-HARD42-TIMING-001',contract_run_id_is_not_a_full_training_authorization=True)
 validate_records(record,runtime_raw,canonical(external),snapshot,plan)
 digest_binding=put(path,record)
 launch='\n'.join(['#!/bin/bash','set -u','umask 077','set -o noclobber','export REVIEWED_EXECUTION_BINDING_SHA256='+digest_binding,'/usr/bin/timeout --signal=TERM --kill-after=30s 390s '+PYTHON+' -I -B '+str(A/'bounded_supervisor.py')+' timing '+str(path)+' > '+str(D/'outer.stdout.log')+' 2> '+str(D/'outer.stderr.log'),'r=$?','printf "%s\\n" "$r" > '+str(D/'outer.exit.txt'),'exit "$r"'])+'\n'
 put(D/'LAUNCH-ONCE.sh',launch.encode());syntax=subprocess.run(['/bin/bash','-n',str(D/'LAUNCH-ONCE.sh')],capture_output=True,timeout=5);assert syntax.returncode==0
 print(json.dumps(dict(active_binding_sha256=digest_binding,launch_sha256=sha(launch.encode()),window=auth,launch_path=str(D/'LAUNCH-ONCE.sh'),not_launched=True)),flush=True)
else:raise ValueError('no other controller operation')
