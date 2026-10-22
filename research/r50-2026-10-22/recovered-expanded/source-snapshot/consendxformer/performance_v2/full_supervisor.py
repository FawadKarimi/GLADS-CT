"""Prepared 7200-second full-training supervisor; inactive until one-run authorization."""
from .launch_security import *
def main():
    os.umask(0o077)
    record_path=sys.argv[1]; raw=read(Path(record_path)); r=json.loads(raw)
    if sha(raw)!=os.environ.get('REVIEWED_EXECUTION_BINDING_SHA256'): raise ValueError('binding digest')
    validate_record(r)
    if r['execution_authorized'] is not True: raise PermissionError('inactive launch')
    root=Path(r['source_root']); report=Path(r['report_root'])
    if report.parent!=REPORTS or report.exists() or report.is_symlink(): raise ValueError('exclusive report')
    path_observations=[private(root,True),private(REPORTS,True),private(Path(r['budget_root']),True)]
    source_check=lambda:verify_sources(r)
    source_check()
    inventory_path=Path(r['runtime_inventory_path'])
    if inventory_path!=root/'candidate/src/consendxformer/performance_v2/runtime_inventory.py':raise ValueError('fixed reviewed inventory source required')
    inventory=subprocess.run([PYTHON,'-I','-B',str(inventory_path)],capture_output=True,check=True,timeout=15)
    if len(inventory.stdout)>16384 or sha(inventory.stdout)!=r['runtime_inventory_sha256']: raise ValueError('runtime drift')
    spec_for_scope=json.loads(read(Path(r['spec_path'])))
    if sha(canonical_spec := json.dumps(spec_for_scope,sort_keys=True,separators=(',',':'),allow_nan=False).encode())!=r['spec_sha256']:
        raise ValueError('permission specification digest')
    condition=spec_for_scope['contract']['initialization']
    space=os.statvfs(REPORTS)
    if condition=='RADIMAGENET':
        from .storage_admission import check
        storage_observation=check(spec_for_scope,REPORTS)
    elif space.f_bavail*space.f_frsize<60*1024**3 or space.f_favail<2000: raise ValueError('space/inode gate')
    filesystem=subprocess.run(['findmnt','-n','-o','FSTYPE,TARGET','-T',str(REPORTS)],capture_output=True,check=True,timeout=5).stdout
    if len(filesystem)>4096 or sha(filesystem)!=r['filesystem_description_sha256']: raise ValueError('filesystem drift')
    rows=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],capture_output=True,check=True,timeout=10).stdout.decode().splitlines()
    selected=[x for x in rows if x.split(',')[0].strip()==r['device_binding']['uuid']]
    if len(selected)!=1 or selected[0]!=r['fresh_gpu_row'] or int(selected[0].split(',')[3].strip())<10240 or selected[0].split(',')[4].strip()!=r['device_binding']['driver']: raise ValueError('fresh GPU/driver gate')
    permission=read(Path(r['shared_gpu_permission_path']))
    if sha(permission)!=r['shared_gpu_permission_sha256']: raise ValueError('permission digest')
    p=json.loads(permission)
    if condition not in ('RANDOM','IMAGENET','RADIMAGENET'):raise PermissionError('unreviewed initialization scope')
    if condition=='IMAGENET' and r['resume_parent_attempt'] is None:
        path_observations.append(private(Path('/home/chaidir/.performance-v2-imagenet-receiver-20261012/artifacts'),True))
    if condition=='RADIMAGENET' and r['resume_parent_attempt'] is None:
        path_observations.append(private(Path('/home/chaidir/.performance-v2-radimagenet-receiver-20261013/artifacts'),True))
    from .supervision_admission import shared_permission_scope
    if (p.get('schema')!='pv2_admin_shared_gpu_full_training_v1' or p.get('administrator')!='Dr. Chaidir'
        or p.get('user_attested_permission') is not True or p.get('gpu_uuid')!=r['device_binding']['uuid']
        or p.get('scope')!=shared_permission_scope(condition,spec_for_scope['contract']['objective']) or p.get('spec_sha256')!=r['spec_sha256']
        or p.get('report_root')!=r['report_root']): raise ValueError('exact full-training shared permission')
    jobs=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,check=True,timeout=10).stdout
    if len(jobs)>65536: raise ValueError('resident observation bound')
    occupancy={'policy':'exact-admin-shared-full','resident_memory_rows':[x for x in jobs.decode().splitlines() if x.split(',')[0].strip()==r['device_binding']['uuid']]}
    from .durable_budget import PersistentBudget
    budget=PersistentBudget(r['budget_root'],expected_plan_sha256=r['budget_plan_sha256'])
    verify_external(r,budget)
    authorization(read(Path(r['authorization_path'])),spec_sha256=r['spec_sha256'],decision=r['decision_sha256'])
    report.mkdir(mode=0o700);(report/'tmp').mkdir(mode=0o700)
    errors=[];child=None;status=125;reason=None
    def write(name,obj):
        with (report/name).open('xb') as f:
            f.write(json.dumps(obj,indent=2).encode());f.flush();os.fsync(f.fileno())
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_CPU,(28800,28800))
        resource.setrlimit(resource.RLIMIT_FSIZE,(512*1024**2,512*1024**2))
        # CUDA reserves virtual address space; bound aggregate observed RSS
        # below rather than mistaking address space for resident host memory.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=r['device_binding']['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_COMPILE_DISABLE='1',TMPDIR=str(report/'tmp'))
    argv=[PYTHON,'-I','-B','-m','consendxformer.performance_v2.full_worker',record_path]
    # -I excludes PYTHONPATH: use a fixed source bootstrap, not -m lookup.
    argv=[PYTHON,'-I','-B',str(Path(r['source_root'])/'candidate/src/consendxformer/performance_v2/full_bootstrap.py'),record_path]
    started=time.monotonic()
    handlers={}
    def interrupt(number, frame): raise ManagedInterruption(number)
    try:
        for number in (signal.SIGINT,signal.SIGTERM,getattr(signal,'SIGHUP',None)):
            if number is not None: handlers[number]=signal.signal(number,interrupt)
        write('preflight.json',{'source':source_check(),'gpu_row':selected[0],'binding_sha256':sha(raw),'argv':argv,
                                'private_paths':path_observations,'runtime':json.loads(inventory.stdout),'occupancy':occupancy})
        with (report/'stdout.log').open('xb') as out,(report/'stderr.log').open('xb') as err:
            child=subprocess.Popen(argv,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=limits)
            write('child.json',{'pid':child.pid,'pgid':child.pid})
            ceiling=7200
            while child.poll() is None:
                if time.monotonic()-started>=ceiling: reason='timeout';status=124;break
                if sum(x['rss_bytes'] for x in process_group(child.pid))>8*1024**3:
                    reason='host_RSS_limit';status=137;break
                time.sleep(.1)
            if reason is None: status=child.returncode
    except BaseException as e:
        errors.append({'stage':'supervisor','category':type(e).__name__})
        status=128+e.number if isinstance(e,ManagedInterruption) else (130 if isinstance(e,KeyboardInterrupt) else 125)
    finally:
        # Accidental repeated termination signals must not bypass independent
        # finalization attempts. The previous handlers are restored afterward.
        for number in handlers: signal.signal(number,signal.SIG_IGN)
        from .managed_cleanup import cleanup
        try:cleanup(child,observe=process_group,send=os.killpg)
        except BaseException as e:errors.append({'stage':'managed_cleanup','category':type(e).__name__})
        from .managed_finalization import finalize
        evidence_steps=[]
        for name,fn in (
            ('exit.json',lambda:{'status':status,'child_status':child.poll() if child else None,'reason':reason}),
            ('post-source.json',source_check),
            ('processes.json',lambda:process_group(child.pid) if child else [])):
            evidence_steps.append((name,lambda name=name,fn=fn:write(name,fn())))
        _,evidence_errors=finalize(status,evidence_steps)
        errors.extend(evidence_errors)
        try: write('finalization-errors.json',errors)
        except BaseException: errors.append({'stage':'error_write','category':'failed'})
        try: write('EVIDENCE-LEDGER.json',{p.name:sha(p.read_bytes()) for p in report.iterdir() if p.is_file()})
        except BaseException: errors.append({'stage':'ledger','category':'failed'})
        for number,previous in handlers.items(): signal.signal(number,previous)
    # Missing records or remaining children never become success.
    if not status and (errors or process_group(child.pid)): status=1
    print(json.dumps({'status':status,'errors':errors,'report':str(report)}),flush=True)
    return status if status>=0 else 128-status


if __name__=='__main__': sys.exit(main())
