"""Invented tensors only. Reduced-shape fixtures mock topology, not production policy."""
import copy,json,random,unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from consendxformer.performance_v2 import imagenet_initialization as im
from consendxformer.performance_v2.checkpoint import state_sha256,encode_state
from consendxformer.performance_v2.streams import TrainingStreams,capture_cpu_rng
from consendxformer.performance_v2.imagenet_mapping import r50_spec
from consendxformer.performance_v2.reuse import digest

class Fixture(torch.nn.Module):
    def __init__(self, spec):
        super().__init__();self.diagnostic_head=torch.nn.Linear(2,3)
        self.register_buffer('new_buffer',torch.tensor([9.]))
        self.values={('backbone.'+k):torch.zeros(tuple(s),dtype=getattr(torch,d.removeprefix('torch.')))
                     for k,(s,d,_) in spec.items() if k not in ('fc.weight','fc.bias')}
        self.objective='HARD';self.seed=42;self.initialization='RANDOM'
    def state_dict(self,*a,**k):
        return dict(self.values,**{n:p.detach() for n,p in self.named_parameters()},new_buffer=self.new_buffer)

class InitializationDelta(unittest.TestCase):
    def test_worker_transport_preserves_importable_classes(self):
        import socket,subprocess
        from consendxformer.performance_v2 import transport_denial as guard
        before=(socket.socket,subprocess.Popen)
        with patch.object(guard.sys,'addaudithook') as add:guard.install()
        add.assert_called_once_with(guard.audit)
        self.assertEqual(before,(socket.socket,subprocess.Popen))
        class ImportedSocket(socket.socket):pass
        class ImportedChild(subprocess.Popen):pass
        self.assertTrue(issubclass(ImportedSocket,socket.socket));self.assertTrue(issubclass(ImportedChild,subprocess.Popen))
    def test_worker_transport_denies_native_network_and_child_events(self):
        from consendxformer.performance_v2.transport_denial import audit
        for event in ('socket.__new__','socket.connect','socket.sendto','subprocess.Popen','os.system','os.fork','_winapi.CreateProcess'):
            with self.assertRaises(PermissionError):audit(event,())
        audit('open',('invented source file',)) # Frozen adapters separately guard reads.
    def setUp(self):
        original=r50_spec()[1]
        self.spec={k:(() if k in im.MISSING else (1,),d,role) for k,(_,d,role) in original.items()}
        self.model=Fixture(self.spec);self.streams=TrainingStreams(42)
        self.state={k:(torch.tensor(0,dtype=torch.int64) if k in im.MISSING else torch.tensor([.25],dtype=torch.float32))
                    for k in self.spec if k not in ('fc.weight','fc.bias')}
        self.receipt={'mapping':{k:'backbone.'+k for k in self.state},
                      'adapted_inventory':{k:{'value_sha256':state_sha256({k:v})} for k,v in self.state.items()},
                      'export_origins':{k:'initialized_bookkeeping' if k in im.MISSING else 'supplied' for k in self.state}}
    def copy(self):
        with patch.object(im,'verify_target'),patch.object(im,'r50_spec',return_value=({},self.spec)):
            return im.copy_verified_backbone(self.model,self.state,self.receipt,self.streams)
    def test_all_318_exact_values_origins_and_nonbackbone_preserved(self):
        before={k:v.clone() for k,v in self.model.state_dict().items() if not k.startswith('backbone.')}
        receipt,report=self.copy()
        self.assertEqual((report['verified_count'],report['supplied_count'],report['bookkeeping_count']),(318,265,53))
        self.assertEqual(receipt['kind'],'IMAGENET')
        for k,v in before.items():self.assertTrue(torch.equal(v,self.model.state_dict()[k]))
        for row in report['entries']:self.assertTrue(row['equal']);self.assertEqual(row['target'],'backbone.'+row['source'])
    def test_rng_global_shuffle_and_augmentation_unchanged(self):
        before=state_sha256(capture_cpu_rng(self.streams));control=TrainingStreams(42)
        self.copy();self.assertEqual(before,state_sha256(capture_cpu_rng(self.streams)))
        self.assertEqual(self.streams.epoch_plan(['invented-A','invented-B'],epoch=1),control.epoch_plan(['invented-A','invented-B'],epoch=1))
    def test_random_no_weight_read_or_mutation(self):
        before=state_sha256(self.model.state_dict());rng=state_sha256(capture_cpu_rng(self.streams))
        receipt,report=im.initialize_fresh(self.model,kind='RANDOM',streams=self.streams,read=lambda *a,**k:self.fail('weight read'))
        self.assertEqual(receipt['kind'],'RANDOM');self.assertFalse(report['pretrained_read'])
        self.assertEqual(before,state_sha256(self.model.state_dict()));self.assertEqual(rng,state_sha256(capture_cpu_rng(self.streams)))
    def test_mapping_substitution_rejected_before_copy(self):
        self.receipt['mapping']['conv1.weight']='diagnostic_head.weight'
        before=state_sha256(self.model.state_dict())
        with self.assertRaisesRegex(ValueError,'mapping'):self.copy()
        self.assertEqual(before,state_sha256(self.model.state_dict()))
    def test_missing_and_extra_entries_rejected(self):
        for mutation in ('missing','extra'):
            original=dict(self.state)
            if mutation=='missing':self.state.pop('conv1.weight')
            else:self.state['fc.bias']=torch.zeros(1)
            with self.assertRaisesRegex(ValueError,'mapping'):self.copy()
            self.state=original
    def test_late_dtype_mismatch_rejected_before_any_mutation(self):
        before=state_sha256(self.model.state_dict());self.state['layer4.2.conv3.weight']=torch.zeros(1,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'pre-copy'):self.copy()
        self.assertEqual(before,state_sha256(self.model.state_dict()))
    def test_origin_and_nonzero_bookkeeping_rejected(self):
        k=sorted(im.MISSING)[0];self.receipt['export_origins'][k]='supplied'
        with self.assertRaisesRegex(ValueError,'origin'):self.copy()
        self.receipt['export_origins'][k]='initialized_bookkeeping';self.state[k]=torch.tensor(1)
        self.receipt['adapted_inventory'][k]['value_sha256']=state_sha256({k:self.state[k]})
        with self.assertRaisesRegex(ValueError,'bookkeeping'):self.copy()
    def test_second_pretrained_application_rejected(self):
        self.copy()
        with self.assertRaisesRegex(ValueError,'one fresh'):self.copy()
    def test_target_alias_rejected(self):
        a,b=tuple(self.model.values)[:2];self.model.values[b]=self.model.values[a]
        with self.assertRaisesRegex(ValueError,'alias'):self.copy()
    def test_copy_failure_poisoned_no_rollback_retry(self):
        with patch.object(torch.Tensor,'copy_',side_effect=RuntimeError('invented copy failure')):
            with self.assertRaisesRegex(RuntimeError,'copy failure'):self.copy()
        self.assertTrue(self.model._pv2_initialization_failed)
        with self.assertRaisesRegex(ValueError,'discard'):self.copy()
    def test_postcopy_nonbackbone_mutation_rejected(self):
        original=torch.Tensor.copy_;once=[False]
        def bad(t,v,*a,**k):
            if not once[0]:self.model.new_buffer.add_(1);once[0]=True
            return original(t,v,*a,**k)
        with patch.object(torch.Tensor,'copy_',bad):
            with self.assertRaisesRegex(ValueError,'non-backbone'):self.copy()
        self.assertTrue(self.model._pv2_initialization_failed)
    def test_rng_drift_rejected_without_hiding_change(self):
        original=torch.Tensor.copy_;once=[False]
        def bad(t,v,*a,**k):
            if not once[0]:random.random();once[0]=True
            return original(t,v,*a,**k)
        with patch.object(torch.Tensor,'copy_',bad):
            with self.assertRaisesRegex(ValueError,'RNG'):self.copy()
    def test_receipt_hash_before_json_and_payload_read(self):
        with patch.object(im.json,'loads',side_effect=AssertionError('parse reached')):
            with self.assertRaisesRegex(ValueError,'authentication'):im.verified_export(b'invented',b'{}')
        reads=[]
        def read(p,**k):reads.append(p.name);return b'{}'
        with self.assertRaisesRegex(ValueError,'authentication'):im.initialize_fresh(self.model,kind='IMAGENET',streams=self.streams,read=read)
        self.assertEqual(reads,['export-receipt.json'])
    def test_wrong_export_hash_before_decode(self):
        r=dict(self.receipt,artifact={'sha256':im.ARTIFACT_SHA,'bytes':102530333},operation_plan_sha256=im.PLAN_SHA,export_sha256=im.EXPORT_SHA,export_bytes=im.EXPORT_BYTES)
        rb=json.dumps(r).encode()
        with patch.object(im,'RECEIPT_SHA',digest(rb)),patch.object(im,'r50_spec',return_value=({},self.spec)),patch.object(im,'verify_receipt_metadata'),patch.object(im,'decode_state',side_effect=AssertionError('decode reached')):
            with self.assertRaisesRegex(ValueError,'before decode'):im.verified_export(b'wrong',rb)
    def test_typed_decode_then_value_tamper_rejected(self):
        r=dict(self.receipt,artifact={'sha256':im.ARTIFACT_SHA,'bytes':102530333},operation_plan_sha256=im.PLAN_SHA)
        raw=encode_state(self.state);r.update(export_sha256=digest(raw),export_bytes=len(raw));rb=json.dumps(r).encode()
        with patch.object(im,'RECEIPT_SHA',digest(rb)),patch.object(im,'EXPORT_SHA',digest(raw)),patch.object(im,'EXPORT_BYTES',len(raw)),patch.object(im,'r50_spec',return_value=({},self.spec)),patch.object(im,'verify_receipt_metadata'):
            state,_=im.verified_export(raw,rb);self.assertEqual(len(state),318)
            r['adapted_inventory']['conv1.weight']['value_sha256']='0'*64;rb=json.dumps(r).encode()
            with patch.object(im,'RECEIPT_SHA',digest(rb)):
                with self.assertRaisesRegex(ValueError,'value / origin'):im.verified_export(raw,rb)
    def test_actual_frozen_hybrid_semantics_and_shared_initialization(self):
        from consendxformer.performance_v2.model import R50Hybrid
        a=R50Hybrid(seed=42,objective='HARD');im.verify_target(a)
        self.assertEqual(len(a.backbone.state_dict()),318)
        before=state_sha256({k:v for k,v in a.state_dict().items() if not k.startswith('backbone.')})
        b=R50Hybrid(seed=42,objective='HARD')
        self.assertEqual(before,state_sha256({k:v for k,v in b.state_dict().items() if not k.startswith('backbone.')}))
    def test_resume_never_reads_or_reapplies_pretrained(self):
        spec={'contract':{'seed':42,'initialization':'IMAGENET'}}
        with patch('consendxformer.performance_v2.model.R50Hybrid',return_value=self.model),patch('consendxformer.performance_v2.training_runner.resume_structure_spec',return_value=spec) as check,patch.object(im,'initialize_fresh',side_effect=AssertionError('initializer reached')):
            model,streams,report=im.prepare_model(spec,None,device_binding={'device':'cpu'},resume_parent=1,read=lambda *a,**k:self.fail('pretrained read'))
        check.assert_called_once();self.assertFalse(report['pretrained_reapplied']);self.assertIs(model,self.model)
    def test_resume_spec_drift_rejected_without_pretrained_access(self):
        spec={'contract':{'seed':42,'initialization':'IMAGENET'}}
        with patch('consendxformer.performance_v2.model.R50Hybrid',return_value=self.model),patch('consendxformer.performance_v2.training_runner.resume_structure_spec',return_value={'drift':True}):
            with self.assertRaisesRegex(ValueError,'drift'):im.prepare_model(spec,None,device_binding={'device':'cpu'},resume_parent=1,read=lambda *a,**k:self.fail('read'))
    def test_imagenet_checkpoint_resume_keeps_original_initialization_binding(self):
        from test_full_training_delta import FullDeltaTests
        from consendxformer.performance_v2.training_runner import initial_receipt,training_spec
        case=FullDeltaTests('test_checkpoint_resume_under_same_pre_reserved_guard')
        try:
            case.setUp()
            receipt=initial_receipt(case.model)
            receipt.update(kind='IMAGENET',artifact_sha256=im.ARTIFACT_SHA,transfer_sha256=im.EXPORT_SHA,loading_review_sha256=im.ACCEPTANCE_SHA)
            case.spec=training_spec(dict(case.spec['contract'],initialization='IMAGENET'),case.adapter,case.model,initialization=receipt,device_binding=case.binding)
            # New invented immutable plan, never mutate/relabel the RANDOM budget.
            from consendxformer.performance_v2.durable_budget import PersistentBudget
            from consendxformer.performance_v2.reuse import canonical
            plan=canonical({'schema':'performance_v2_durable_plan_v1','scope':'synthetic','limit':2,
                            'runs':{case.contract['run_id']:digest(canonical(case.spec))},'pilot_execution_authorized':False})
            case.budget=PersistentBudget.create(case.root/'invented-imagenet-budget',plan_bytes=plan,
                expected_plan_sha256=digest(plan),synthetic_directory_sync=lambda _:None)
            with patch.object(im,'initialize_fresh',side_effect=AssertionError('pretrained reapplied')):
                case.test_checkpoint_resume_under_same_pre_reserved_guard()
            self.assertEqual(case.spec['initialization_receipt'],receipt)
        finally:case.doCleanups()
    def test_fresh_initializer_and_actual_spec_agree(self):
        spec={'contract':{'seed':42,'initialization':'IMAGENET'}}
        with patch('consendxformer.performance_v2.model.R50Hybrid',return_value=self.model),patch.object(im,'initialize_fresh',return_value=({'kind':'IMAGENET'},{'invented':True})) as initialize,patch('consendxformer.performance_v2.training_runner.training_spec',return_value=spec) as check:
            im.prepare_model(spec,None,device_binding={'device':'cpu'},resume_parent=None,read=lambda *a,**k:b'invented')
        initialize.assert_called_once();check.assert_called_once()

if __name__=='__main__':unittest.main()
