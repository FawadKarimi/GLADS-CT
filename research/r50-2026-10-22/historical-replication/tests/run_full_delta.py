"""Changed integration only. Invented CPU fixtures; no research or GPU access."""
from pathlib import Path
import hashlib,importlib.metadata as md,json,os,socket,subprocess,sys,unittest
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'tests')]
import torch
def deny(*a,**k):raise PermissionError('network/subprocess/CUDA forbidden in delta tests')
socket.socket=socket.create_connection=deny;subprocess.Popen=deny;torch.cuda.init=deny
torch.set_num_threads(2)
suite=unittest.defaultTestLoader.loadTestsFromName('test_full_training_delta')
if suite.countTestCases()!=16:raise RuntimeError('fixed delta test count mismatch')
result=unittest.TextTestRunner(verbosity=2).run(suite)
record={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
        'python':sys.version,'interpreter':sys.executable,'packages':{n:md.version(n) for n in ('torch','torchvision','numpy','opencv-python')},
        'cuda_initialized':torch.cuda.is_initialized(),'synthetic_only':True,'historical_suites_rerun':False}
print('DELTA_RESULT='+json.dumps(record,sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() and not result.skipped and not record['cuda_initialized'] else 1)
