"""Inactive RANDOM/ImageNet HARD TRAIN/VAL entry; no TEST or pickle interface."""
import copy, json, os, sys
from pathlib import Path

def main(record_path):
    from .launch_security import read, sha, verify_sources, validate_record, verify_external
    r=json.loads(read(Path(record_path)))
    from .transport_denial import install,deny
    install()  # Before validate_record's Torch import; preserve import-compatible classes.
    validate_record(r); verify_sources(r)
    if r['execution_authorized'] is not True: raise PermissionError('inactive full-training record')
    import signal
    def interrupted(number,frame):raise KeyboardInterrupt('managed termination')
    signal.signal(signal.SIGTERM,interrupted)
    import torch
    torch.load=deny
    from .model import R50Hybrid
    from .materialization import open_mapping_source
    from .input_adapter import BoundInputAdapter
    from .training_runner import TrainingRunner
    from .imagenet_initialization import prepare_model
    from .cuda_identity import UuidBoundOperations
    from .optimization import make_adamw
    from .streams import TrainingStreams
    from .durable_budget import PersistentBudget
    from .full_training import execute_reserved, interrupted_receipt
    from .reuse import canonical, digest
    from .lifecycle import review_resume
    from .checkpoint import validate_components
    from .training_runner import validate_device_rng, cpu_copy
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    spec=json.loads(read(Path(r['spec_path'])))
    if digest(canonical(spec))!=r['spec_sha256']: raise ValueError('spec digest drift')
    from .supervision_admission import admit_contract
    admit_contract(spec['contract'])
    if spec['device_binding']!=r['device_binding']:
        raise PermissionError('exact reviewed device required')
    source=open_mapping_source(r['image_root'],r['mapping_root'],
        approval_bytes=read(Path(r['mapping_approval_path'])),expected_approval_sha256=r['mapping_approval_sha256'])
    adapter=BoundInputAdapter(source.inputs,source,expected_transform_sha256=r['transform_sha256'])
    auth=read(Path(r['authorization_path']))
    if sha(auth)!=r['authorization_sha256']: raise ValueError('authorization drift')
    from .launch_security import authorization
    authorization(auth,spec_sha256=r['spec_sha256'],decision=r['decision_sha256'])
    model,streams,initialization_report=prepare_model(spec,adapter,device_binding=r['device_binding'],
        resume_parent=r['resume_parent_attempt'],read=read)
    report=Path(r['report_root'])
    with (report/'initialization.json').open('xb') as f:
        f.write(canonical(initialization_report));f.flush();os.fsync(f.fileno())
    budget=PersistentBudget(r['budget_root'],expected_plan_sha256=r['budget_plan_sha256'])
    verify_external(r,budget)
    resume=None; parent=r['resume_parent_attempt']
    if parent is not None:
        raw=interrupted_receipt(budget,parent,read=read)
        terminal=json.loads(raw)
        from collections.abc import Mapping
        class BoundedHistory(Mapping):
            # Freeze trusted manifest keys, but never retain checkpoint bodies.
            def __init__(self):self.keys=tuple(terminal['checkpoint_hashes'])
            def __iter__(self):return iter(self.keys)
            def __len__(self):return len(self.keys)
            def __getitem__(self,key):
                if key not in self.keys or key!=str(int(key)):raise KeyError(key)
                return read(budget.run_path(parent)/('epoch-%03d.state.json'%int(key)),cap=512*1024**2)
        checkpoints=BoundedHistory()
        retention_metadata=None
        if "retention" in terminal:
            class BoundedMetadata(Mapping):
                def __iter__(self):return iter(terminal["retention"]["metadata_hashes"])
                def __len__(self):return len(terminal["retention"]["metadata_hashes"])
                def __getitem__(self,key):
                    if key not in terminal["retention"]["metadata_hashes"]:raise KeyError(key)
                    return read(budget.run_path(parent)/key,cap=1024**2)
            retention_metadata=BoundedMetadata()
        captured={}
        def stage(components):
            validate_components(spec,components,epoch=terminal['progress']['epoch'],updates=terminal['progress']['updates'])
            validate_device_rng(components['rng'],binding=r['device_binding'],seed=spec['contract']['seed'])
            captured['components']=cpu_copy(components)
        # CPU structural recapture is not a claim of native CUDA restoration.
        # Actual device restore+recapture happens inside the reserved guard below.
        resume=review_resume(requested_spec=spec,terminal_bytes=raw,terminal_sha256=sha(raw),
            checkpoint_bytes=checkpoints,checkpoint_hashes=terminal['checkpoint_hashes'],
            restore_components=stage,capture_components=lambda:captured['components'],retention_metadata_bytes=retention_metadata)
    def setup():
        model.to('cuda:0')
        optimizer,_=make_adamw(model)
        op=UuidBoundOperations(model,optimizer,streams,device_binding=r['device_binding'])
        if resume is not None:
            op.restore(resume.payload['components'],spec=spec,epoch=resume.payload['progress']['epoch'],updates=resume.payload['progress']['updates'])
        return op
    def runner(op):
        return TrainingRunner(spec=spec,adapter=adapter,operations=op,budget=budget,
            decision_sha256=r['decision_sha256'],authorization_bytes=auth,authorization_sha256=sha(auth))
    receipt,manifest=execute_reserved(spec=spec,adapter=adapter,budget=budget,decision=r['decision_sha256'],
        setup=setup,runner_factory=runner,resume=resume,parent_attempt=parent)
    print(json.dumps({'receipt':receipt,'manifest_sha256':digest(canonical(manifest))}),flush=True)
    return 0

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    sys.exit(main(sys.argv[1]))
