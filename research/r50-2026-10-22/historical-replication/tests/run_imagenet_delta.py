"""Only the changed invented initialization/resume tests, CPU two threads."""
from pathlib import Path
import sys,unittest,json,importlib.metadata as md,tempfile
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'tests')]
import socket,subprocess
# Class-compatible guards before importing candidate packages.
class BlockedSocket(socket.socket):
    def connect(self,*a,**k):raise PermissionError('network denied')
    def connect_ex(self,*a,**k):raise PermissionError('network denied')
    def sendto(self,*a,**k):raise PermissionError('network denied')
class BlockedPopen(subprocess.Popen):
    def __init__(self,*a,**k):raise PermissionError('child execution denied')
socket.socket=BlockedSocket;subprocess.Popen=BlockedPopen
def deny(*a,**k):raise PermissionError('network/pickle/CUDA denied')
socket.create_connection=deny
import torch
torch.load=deny;torch.cuda.init=deny;torch.set_num_threads(2)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
suite=unittest.defaultTestLoader.loadTestsFromName('test_imagenet_initialization_delta')
# One existing resume regression is relevant to the changed shared resume helper.
suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_full_training_delta.FullDeltaTests.test_checkpoint_resume_under_same_pre_reserved_guard'))
if suite.countTestCases()!=23:raise ValueError('fixed delta count')
r=unittest.TextTestRunner(verbosity=2).run(suite)
print('RESULT='+json.dumps({'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skips':len(r.skipped),'python':sys.version,'executable':sys.executable,'packages':{n:md.version(n) for n in ('torch','torchvision','numpy')},'cuda_initialized':torch.cuda.is_initialized(),'real_export_opened':False,'model_forwards':'invented existing resume fixture only','historical_suites_repeated':False}),flush=True)
sys.exit(0 if r.wasSuccessful() and not r.skipped and not torch.cuda.is_initialized() else 1)
