"""One bounded CPU test entry; source must already be hash-verified privately.
Trusted code/packages; Python guards prevent accidental access, not OS isolation.
"""
from pathlib import Path
import ast, builtins, hashlib, importlib.metadata, io, json, os, sys, unittest
from unittest.mock import patch

root, report = (Path(x).resolve(strict=True) for x in sys.argv[1:3])
candidate=root/'candidate'
lock=json.loads((root/'source-lock.json').read_bytes())
assert hashlib.sha256((root/'source-lock.json').read_bytes()).hexdigest()=='6ff5134ee531ff11f1755453063323f07579a9c04b97d4e6296e41f2bdc50529'
assert len(lock)==140
for name,h in lock.items():
    assert hashlib.sha256((candidate/name).read_bytes()).hexdigest()==h
for p in candidate.rglob('*.py'): ast.parse(p.read_bytes(),filename=str(p))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['TORCH_COMPILE_DISABLE']='1'
sys.path[:0]=[str(candidate/'src'),str(candidate/'tests')]
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import socket,subprocess,tempfile
tempfile.tempdir=str(report/'tmp')

def reject():
    def denied(*a,**k): raise PermissionError('native synthetic scope denied operation')
    return denied

original_open, original_io_open, original_os_open = builtins.open, io.open, os.open
def permitted(file):
    if isinstance(file,int): return
    p=Path(os.fspath(file)).absolute()
    if p.suffix.lower() in ('.pt','.pth','.safetensors','.h5','.hdf5'):
        raise PermissionError('weights denied')
    allowed=[root,report,Path(sys.prefix),Path(sys.base_prefix)]
    if not any(p==a or a in p.parents for a in allowed):
        raise PermissionError('path outside private source/report and interpreter libraries')
def guarded(original):
    def wrapper(file,*a,**k):
        permitted(file)
        return original(file,*a,**k)
    return wrapper
def guarded_os_open(file, flags, mode=0o777, *, dir_fd=None):
    if dir_fd is not None:
        # Permit only descriptors freshly resolved into the same allowlisted
        # private fixture/source tree; Linux cleanup/readers use dir_fd.
        parent=Path(os.readlink('/proc/self/fd/'+str(dir_fd)))
        if not any(parent==a or a in parent.parents for a in (root,report)):
            raise PermissionError('unbound directory descriptor')
        permitted(parent/os.fspath(file))
    else:
        permitted(file)
    return original_os_open(file,flags,mode,dir_fd=dir_fd)

names=['test_performance_v2_real_updates',
 'test_performance_v2_runner.RunnerTests.test_real_updates_and_production_dispatch_rejected_before_io_or_admission',
 'test_performance_v2_runner.RunnerTests.test_full_mock_lifecycle_durable_complete_validation_only_and_no_parameter_updates',
 'test_performance_v2_runner.RunnerTests.test_boundary_resume_recaptures_all_live_state_and_consumes_second_slot',
 'test_performance_v2_checkpoint_inventory']
log=io.StringIO()
with (patch('socket.socket',reject()),patch('socket.create_connection',reject()),
      patch('subprocess.Popen',reject()),patch('torch.load',reject()),
      patch('torch.cuda.get_rng_state',reject()),patch('torch.cuda.set_rng_state',reject()),
      patch('torch.cuda.init',reject()),patch('builtins.open',guarded(original_open)),
      patch('io.open',guarded(original_io_open)),patch('os.open',guarded_os_open)):
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
text=log.getvalue()
with (report/'tests.log').open('x') as f:f.write(text)
record={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
        'success':result.testsRun==31 and result.wasSuccessful() and not result.skipped,
        'python':sys.version,'executable':sys.executable,
        'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','opencv-python')},
        'source_lock_sha256':'6ff5134ee531ff11f1755453063323f07579a9c04b97d4e6296e41f2bdc50529',
        'invented_fixtures_only':True,'gpu_execution':False,'research_access':False,
        'checkpoint_deserialization':False,'production_execution_authorized':False}
with (report/'results.json').open('x') as f:json.dump(record,f,indent=2)
print(text,flush=True);print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if record['success'] else 1)
