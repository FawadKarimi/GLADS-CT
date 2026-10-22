"""Changed integration only; invented CPU inputs, no historical suite rerun."""
import copy, json, sys, types, unittest
from unittest.mock import patch
from pathlib import Path
import torch
from test_performance_v2_real_updates import RealUpdateTests
fixture_setup=RealUpdateTests.setUp
del RealUpdateTests
from test_performance_v2_runner import Tiny
from consendxformer.performance_v2.training_runner import TrainingRunner, TrainingOperations
from consendxformer.performance_v2.cuda_identity import UuidBoundOperations
from consendxformer.performance_v2.full_training import execute_reserved
from consendxformer.performance_v2.durable_lifecycle import DurableRun
from consendxformer.performance_v2.lifecycle import review_resume, require_completed_reuse
from consendxformer.performance_v2.optimization import make_adamw
from consendxformer.performance_v2.streams import TrainingStreams
from consendxformer.performance_v2.reuse import canonical,digest

class FullDeltaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        fixture_setup(self)
    def runner(self, op=None, decision=None):
        return TrainingRunner(spec=self.spec,adapter=self.adapter,operations=op or self.op,
                              budget=self.budget,decision_sha256=decision or digest(b'invented-full'))
    def execute(self,setup=None,resume=None,parent=None):
        decision=digest(b'invented-full-resume' if resume is not None else b'invented-full')
        return execute_reserved(spec=self.spec,adapter=self.adapter,budget=self.budget,
            decision=decision,setup=setup or (lambda:self.op),
            runner_factory=lambda op:self.runner(op,decision),resume=resume,parent_attempt=parent)
    def test_reservation_precedes_device_setup_and_first_read(self):
        observed=[]; original=self.adapter.batch
        def setup():
            self.assertEqual(self.budget.inspect()['consumed'],1); observed.append('setup');return self.op
        def batch(*a,**k):
            self.assertEqual(observed[0],'setup'); self.assertEqual(self.budget.inspect()['consumed'],1)
            raise ValueError('first invented read')
        with patch.object(self.adapter,'batch',side_effect=batch):
            with self.assertRaisesRegex(ValueError,'first invented read'): self.execute(setup)
        self.assertEqual(json.loads(self.budget.receipt_path(1).read_bytes())['status'],'failed')
    def test_setup_failure_keeps_slot_and_original_error(self):
        with self.assertRaisesRegex(RuntimeError,'placement'):
            self.execute(lambda: (_ for _ in ()).throw(RuntimeError('placement')))
        self.assertEqual(self.budget.inspect()['consumed'],1)
        self.assertEqual(json.loads(self.budget.receipt_path(1).read_bytes())['status'],'failed')
    def test_rejected_budget_prevents_setup(self):
        with patch.object(self.budget,'admit',side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError): self.execute(lambda:self.fail('setup reached'))
        self.assertEqual(self.budget.inspect()['consumed'],0)
    def test_validation_selection_early_stop_and_durable_completion(self):
        original=self.model.forward
        def fixed_validation(x):
            if self.model.training:return original(x)
            return {'logits':torch.zeros((len(x),3)), 'reader_logits':None}
        with patch.object(self.model,'forward',side_effect=fixed_validation):receipt,manifest=self.execute()
        self.assertEqual(receipt['status'],'synthetic_complete')
        self.assertTrue(manifest['selection']['stopped'])
        self.assertEqual(manifest['progress']['epoch'],20)
        self.assertEqual(manifest['selection']['best_epoch'],1)
        self.assertEqual(manifest['best'],'epoch-001.state.json')
        self.assertEqual(self.budget.inspect()['consumed'],1)
        self.assertFalse(self.op._admitted)
        with self.assertRaises(ValueError): require_completed_reuse(self.spec,completion_bytes=canonical(receipt),
            completion_sha256=digest(canonical(receipt)),manifest_bytes=canonical(manifest),verified_output_hashes=manifest['files'])
    def test_checkpoint_resume_under_same_pre_reserved_guard(self):
        original=DurableRun.save_epoch
        def boundary(session,*a,**k): original(session,*a,**k);raise KeyboardInterrupt('invented boundary')
        with patch.object(DurableRun,'save_epoch',boundary):
            with self.assertRaises(KeyboardInterrupt): self.execute()
        from consendxformer.performance_v2.full_training import interrupted_receipt
        raw=interrupted_receipt(self.budget,1,read=lambda p:p.read_bytes());terminal=json.loads(raw)
        checkpoints={k:(self.budget.run_path(1)/('epoch-%03d.state.json'%int(k))).read_bytes() for k in terminal['checkpoint_hashes']}
        model=Tiny();optim,_=make_adamw(model)
        op=TrainingOperations(model,optim,TrainingStreams(42),device_binding=self.binding)
        resume=review_resume(requested_spec=self.spec,terminal_bytes=raw,terminal_sha256=digest(raw),checkpoint_bytes=checkpoints,
            checkpoint_hashes=terminal['checkpoint_hashes'],restore_components=lambda c:op.restore(c,spec=self.spec,epoch=1,updates=2),capture_components=op.capture)
        receipt,manifest=self.execute(lambda:op,resume,1)
        self.assertEqual(receipt['status'],'synthetic_complete');self.assertEqual(self.budget.inspect()['consumed'],2)
        self.assertGreater(manifest['progress']['updates'],2);self.assertFalse(op._admitted)
    def test_reserved_session_substitution_rejected(self):
        session=DurableRun(spec=self.spec,inputs=self.inputs,budget=self.budget,decision_sha256=digest(b'other'))
        with session.guard():
            with self.assertRaisesRegex(ValueError,'pre-reserved'):self.runner().run(reserved_session=session)
        self.assertEqual(self.budget.inspect()['consumed'],1)
    def test_complete_or_tampered_receipt_cannot_enter_resume(self):
        from consendxformer.performance_v2.full_training import interrupted_receipt
        from types import SimpleNamespace
        rows=[{'number':1,'state':'complete','exported':1,'receipt_sha':digest(b'original')}]
        budget=SimpleNamespace(inspect=lambda:{'issues':[],'attempts':rows},receipt_path=lambda n:'invented')
        with self.assertRaisesRegex(ValueError,'interrupted'):
            interrupted_receipt(budget,1,read=lambda p:self.fail('read complete receipt'))
        rows[0]['state']='interrupted'
        with self.assertRaisesRegex(ValueError,'hash'):interrupted_receipt(budget,1,read=lambda p:b'tampered')
    def test_exact_uuid_operations_admitted_with_cuda_calls_mocked(self):
        op=object.__new__(UuidBoundOperations)
        op.binding=dict(self.binding,device='cuda');op.device=torch.device('cpu')
        runner=self.runner(op);runner.spec['device_binding']=op.binding
        with patch.object(UuidBoundOperations,'check_device') as check:runner._admission();check.assert_called_once()
    def test_uuid_subclass_and_plain_operations_rejected_for_cuda(self):
        class Unreviewed(UuidBoundOperations):pass
        for cls in (Unreviewed,TrainingOperations):
            op=object.__new__(cls);op.binding=dict(self.binding,device='cuda');op.device=torch.device('cpu')
            runner=self.runner(op);runner.spec['device_binding']=op.binding
            with self.assertRaises(PermissionError):runner._admission()
    def test_read_failure_and_evidence_failure_preserve_first_error(self):
        with (patch.object(self.adapter,'batch',side_effect=ValueError('first error')),
              patch('consendxformer.performance_v2.durable_budget.Admission.terminate',side_effect=OSError('disk'))):
            with self.assertRaisesRegex(ValueError,'first error'): self.execute()
        self.assertEqual(self.budget.inspect()['consumed'],1);self.assertFalse(self.op._admitted)

class ManagedDeltaTests(unittest.TestCase):
    def test_cleanup_never_addresses_unrelated_group(self):
        from consendxformer.performance_v2.managed_cleanup import cleanup
        from types import SimpleNamespace
        calls=[];alive=[True]
        child=SimpleNamespace(pid=7123,poll=lambda:None if alive[0] else 0,wait=lambda **k:alive.__setitem__(0,False))
        cleanup(child,observe=lambda pg:[{'pid':pg}] if alive[0] else [],send=lambda pg,sig:calls.append(pg))
        self.assertEqual(calls,[7123])
    def test_cleanup_handles_surviving_group_after_parent_exit(self):
        from consendxformer.performance_v2.managed_cleanup import cleanup
        from types import SimpleNamespace
        import signal
        members=[{'pid':44}];sent=[];ticks=[0]
        def clock():ticks[0]+=8;return ticks[0]
        def send(pg,sig):
            sent.append((pg,sig))
            if sig==signal.SIGKILL:members.clear()
        child=SimpleNamespace(pid=44,poll=lambda:0,wait=lambda **k:None)
        # Linux-only termination symbol is mocked on this Windows CPU test host.
        with patch.object(signal,'SIGKILL',9,create=True):
            cleanup(child,observe=lambda pg:list(members),send=send,clock=clock,sleep=lambda _:None)
            self.assertEqual(sent,[(44,signal.SIGTERM),(44,signal.SIGKILL)])
    def test_finalization_retains_first_failure_and_attempts_every_write(self):
        from consendxformer.performance_v2.managed_finalization import finalize
        calls=[]
        def fail(): calls.append('source');raise OSError('invented disk')
        status,errors=finalize(124,[('source',fail),('exit',lambda:calls.append('exit')),('processes',lambda:calls.append('processes'))])
        self.assertEqual(status,124);self.assertEqual(calls,['source','exit','processes'])
        self.assertEqual(errors,[{'stage':'source','category':'OSError'}])
    def test_missing_evidence_never_becomes_success(self):
        from consendxformer.performance_v2.managed_finalization import finalize
        status,errors=finalize(0,[('exit',lambda:(_ for _ in ()).throw(OSError('disk')))])
        self.assertEqual(status,1);self.assertEqual(len(errors),1)
    def test_full_supervisor_preserves_accepted_complete_scan_and_cleanup(self):
        # Static equivalence checks plus mocked cleanup/finalization execution below.
        p=Path(__file__).parents[1]/'src/consendxformer/performance_v2'
        security=(p/'launch_security.py').read_text();s=(p/'full_supervisor.py').read_text()
        self.assertIn("for path in Path('/proc').iterdir():",security)
        self.assertNotIn('4096]',security);self.assertIn('cleanup(child,observe=process_group,send=os.killpg)',s)
        self.assertIn('ceiling=7200',s);self.assertIn("('exit.json'",s);self.assertIn("('post-source.json'",s)
    def test_source_security_module_is_linux_only_not_runtime_substitution(self):
        p=Path(__file__).parents[1]/'src/consendxformer/performance_v2'
        import ast
        for n in ('full_worker.py','full_supervisor.py','launch_security.py','full_bootstrap.py'):
            ast.parse((p/n).read_text())
        self.assertIn('verify_external(r,budget)',(p/'full_worker.py').read_text())

if __name__=='__main__':unittest.main()
