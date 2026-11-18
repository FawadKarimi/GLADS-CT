"""Six invented control-flow checks of actual full_worker/execute_reserved.

All ML, data, device and budget adapters are mocked. No real package import,
weight reads, forwards, optimizer updates or persistent budget creation.
"""
import ast,builtins,copy,hashlib,json,os,sys,tempfile,types,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
digest=lambda b:hashlib.sha256(b).hexdigest()
SOURCE=None
RECORDS=None
def extract(path,names,scope):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope
class EntryChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.events=[];self.consumed=0;self.rejected=False;self.placement_error=False
        self.fixture_condition='RANDOM'
    def run_entry(self,condition='RANDOM',inactive=False,drift=False,uuid_drift=False):
        self.fixture_condition=condition
        record=json.loads((RECORDS/('records/launches/'+condition+'-42.launch.INACTIVE.json')).read_bytes())
        spec=json.loads((RECORDS/('specifications/'+condition+'-42.execution-spec.INACTIVE.json')).read_bytes())
        record['report_root']=self.tmp.name
        record['execution_authorized']=not inactive  # in-memory invented fixture only
        if drift:record['spec_sha256']='0'*64
        if uuid_drift:record['device_binding']=dict(record['device_binding'],uuid='GPU-unreviewed')
        auth=canonical({'invented_authorization_fixture':True})
        reads=[]
        def read(path,*a,**k):
            path=Path(path).as_posix();reads.append(path)
            if path=='/invented/launch.json':return canonical(record)
            if path==Path(record['spec_path']).as_posix():return canonical(spec)
            if path==Path(record['authorization_path']).as_posix():return auth
            raise AssertionError('unapproved invented read: '+path)
        self.reads=reads;me=self
        valid_scope={'_hash':lambda h:type(h)is str and len(h)==64 and all(c in '0123456789abcdef' for c in h)}
        extract(SOURCE/'training_runner.py',['validate_device_binding'],valid_scope)
        rfields=ast.parse((SOURCE/'launch_security.py').read_text())
        assignment=next(n for n in rfields.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RECORD_FIELDS' for t in n.targets))
        valid_record={'PYTHON':'/experiment-storage/chaidir/env/env-310-fawad/bin/python'}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<record-fields>','exec'),valid_record)
        extract(SOURCE/'launch_security.py',['validate_record'],valid_record)
        class Model:
            def to(self,device):
                me.assertEqual(me.consumed,1);me.events.append('device:'+device)
                if me.placement_error:raise RuntimeError('invented placement failure')
                return self
        def prepare_model(given,adapter,*,device_binding,resume_parent,read):
            me.assertEqual(given,spec);me.assertIsNone(resume_parent)
            me.assertEqual(device_binding,spec['device_binding'])
            me.events.append('initializer:'+given['contract']['initialization'])
            return Model(),object(),{'fixture_only':True,'condition':condition}
        class Budget:
            def __init__(self,*a,**k):me.events.append('budget_open_mock')
        class Session:
            def __init__(self,**kwargs):
                if me.rejected:raise PermissionError('invented admission rejected')
                me.consumed+=1;me.events.append('reserve')
            @contextmanager
            def guard(self):
                try:yield self
                except BaseException:me.events.append('failed_receipt_mock');raise
                finally:me.events.append('lease_cleanup_mock')
        coordination={'DurableRun':Session}
        extract(SOURCE/'full_training.py',['execute_reserved'],coordination)
        class Runner:
            def __init__(self,**kwargs):me.events.append('runner_construct_mock')
            def run(self,**kwargs):
                me.assertEqual(me.consumed,1);me.events.append('first_reader_mock')
                return {'status':'invented_only'}, {'fixture':True}
        def make_adamw(model):
            me.assertEqual(me.consumed,1);me.events.append('optimizer_construct_mock');return object(),[]
        torch=types.SimpleNamespace(load=None,set_num_threads=lambda n:me.events.append('threads:'+str(n)),
            use_deterministic_algorithms=lambda x:me.assertIs(x,True),
            backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True,allow_tf32=True),
               cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True))))
        modules={
         'launch_security':types.SimpleNamespace(read=read,sha=digest,verify_sources=lambda r:me.assertEqual(r['source_lock_sha256'],record['source_lock_sha256']),validate_record=valid_record['validate_record'],verify_external=lambda *a:me.events.append('external_mock'),authorization=lambda *a,**k:me.events.append('authorization_mock')),
         'transport_denial':types.SimpleNamespace(install=lambda:me.events.append('import_guards'),deny=lambda *a,**k:(_ for _ in ()).throw(PermissionError('denied'))),
         'torch':torch,'model':types.SimpleNamespace(R50Hybrid=object),
         'materialization':types.SimpleNamespace(open_mapping_source=lambda *a,**k:me.events.append('mapping_mock') or types.SimpleNamespace(inputs=object())),
         'input_adapter':types.SimpleNamespace(BoundInputAdapter=lambda *a,**k:types.SimpleNamespace(inputs=object())),
         'training_runner':types.SimpleNamespace(TrainingRunner=Runner,validate_device_rng=lambda *a,**k:None,cpu_copy=copy.deepcopy,validate_device_binding=valid_scope['validate_device_binding']),
         'imagenet_initialization':types.SimpleNamespace(prepare_model=prepare_model),
         'cuda_identity':types.SimpleNamespace(UuidBoundOperations=lambda *a,**k:object()),
         'optimization':types.SimpleNamespace(make_adamw=make_adamw),'streams':types.SimpleNamespace(TrainingStreams=object),
         'durable_budget':types.SimpleNamespace(PersistentBudget=Budget),
         'full_training':types.SimpleNamespace(execute_reserved=coordination['execute_reserved'],interrupted_receipt=lambda *a,**k:me.fail('unexpected resume')),
         'reuse':types.SimpleNamespace(canonical=canonical,digest=digest),
         'lifecycle':types.SimpleNamespace(review_resume=lambda *a,**k:me.fail('unexpected resume')),
         'checkpoint':types.SimpleNamespace(validate_components=lambda *a,**k:None)}
        # Approval metadata is an invented mapping fixture; no real sidecars/images.
        old_read=read
        def full_read(path,*a,**k):
            if Path(path).as_posix()==Path(record['mapping_approval_path']).as_posix():return b'invented mapping approval'
            if Path(path).as_posix()==Path(record['authorization_path']).as_posix():return auth
            return old_read(path,*a,**k)
        record['authorization_sha256']=digest(auth);modules['launch_security'].read=full_read
        original_import=builtins.__import__
        def imports(name,globals=None,locals=None,fromlist=(),level=0):
            if name in modules:return modules[name]
            return original_import(name,globals,locals,fromlist,level)
        worker_scope={'__name__':'consendxformer.performance_v2.full_worker','__package__':'consendxformer.performance_v2','Path':Path,'json':json,'os':os,'sys':sys}
        extract(SOURCE/'full_worker.py',['main'],worker_scope)
        # The actual worker installs signal handlers; mocks keep tests isolated.
        with patch('builtins.__import__',side_effect=imports),patch('signal.signal'):
            result=worker_scope['main']('/invented/launch.json')
        me.assertIs(torch.backends.cudnn.benchmark,False);me.assertIs(torch.backends.cuda.matmul.allow_tf32,False);me.assertIs(torch.backends.cudnn.allow_tf32,False)
        return result
    def test_ENT01_random_fresh_reserves_before_device_reader(self):
        self.assertEqual(self.run_entry('RANDOM'),0)
        self.assertLess(self.events.index('reserve'),self.events.index('device:cuda:0'))
        self.assertLess(self.events.index('reserve'),self.events.index('first_reader_mock'))
        self.assertIn('initializer:RANDOM',self.events)
    def test_ENT02_imagenet_fresh_same_reserved_entry(self):
        self.assertEqual(self.run_entry('IMAGENET'),0)
        self.assertLess(self.events.index('reserve'),self.events.index('device:cuda:0'))
        self.assertIn('initializer:IMAGENET',self.events);self.assertIn('lease_cleanup_mock',self.events)
    def test_ENT03_inactive_or_unreviewed_uuid_prevents_initialization(self):
        for kwargs in ({'inactive':True},{'uuid_drift':True}):
            self.events.clear()
            with self.assertRaises((PermissionError,ValueError)):self.run_entry(**kwargs)
            self.assertFalse(any(x.startswith('initializer:') for x in self.events));self.assertEqual(self.consumed,0)
    def test_ENT04_spec_digest_drift_prevents_mapping_and_device(self):
        with self.assertRaisesRegex(ValueError,'spec digest'):self.run_entry(drift=True)
        self.assertNotIn('mapping_mock',self.events);self.assertEqual(self.consumed,0)
    def test_ENT05_budget_rejection_prevents_device_reader(self):
        self.rejected=True
        with self.assertRaisesRegex(PermissionError,'admission'):self.run_entry()
        self.assertEqual(self.consumed,0);self.assertNotIn('device:cuda:0',self.events);self.assertNotIn('first_reader_mock',self.events)
    def test_ENT06_setup_failure_retains_consumption_cleanup(self):
        self.placement_error=True
        with self.assertRaisesRegex(RuntimeError,'placement'):self.run_entry()
        self.assertEqual(self.consumed,1);self.assertIn('failed_receipt_mock',self.events);self.assertIn('lease_cleanup_mock',self.events)
        self.assertNotIn('first_reader_mock',self.events)
def run(source_root,records):
    global SOURCE,RECORDS
    SOURCE=Path(source_root)/'candidate/src/consendxformer/performance_v2';RECORDS=Path(records)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EntryChecks)
    assert suite.countTestCases()==6
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return {'checks':6,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'forwards':0,'optimizer_updates':0,'gpu_calls':0,'actual_budgets_created':0,'scope':'actual entry/coordinator with invented adapters; not numerical training or permission enforcement'}
if __name__=='__main__':
    value=run(sys.argv[1],sys.argv[2]);print('ENTRY_RESULT='+json.dumps(value),flush=True)
    sys.exit(0 if value['tests']==6 and value['failures']==value['errors']==value['skips']==0 else 1)
