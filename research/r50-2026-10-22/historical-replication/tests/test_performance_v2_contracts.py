"""Invented validation records and receipts; no research data or model imports."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consendxformer.performance_v2.rng import stream_seed, private_random
from consendxformer.performance_v2.selection import (
    ValidationEpoch as E, SeedValidation as V, ValidationPatience,
    select_checkpoint, select_screen, select_repeated_family,
)
from consendxformer.performance_v2 import reuse as r


class SelectionTests(unittest.TestCase):
    def test_global_tolerance_not_transitive_pairwise_sort(self):
        records = [E(1, .8, .9), E(2, .8-5e-13, .8), E(3, .8-1.4e-12, .1)]
        self.assertEqual(select_checkpoint(records).epoch, 2)
        self.assertEqual(select_checkpoint(reversed(records)).epoch, 2)

    def test_macro_f1_precedes_ce_and_exact_ties_choose_earliest(self):
        self.assertEqual(select_checkpoint([E(1, .7, .1), E(3, .8, .5), E(2, .8, .5)]).epoch, 2)

    def test_test_records_nonfinite_and_invalid_values_rejected(self):
        for kwargs in [dict(split="test"), dict(macro_f1=float("nan")),
                       dict(diagnostic_ce=float("inf")), dict(epoch=0), dict(macro_f1=1.1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                E(**dict(dict(epoch=1, macro_f1=.5, diagnostic_ce=1), **kwargs))

    def test_unknown_metric_dictionary_and_duplicate_epoch_rejected(self):
        for records in [[], [{"test_macro_f1": .99}], [E(1,.5,1),E(1,.6,1)]]:
            with self.assertRaises(ValueError):select_checkpoint(records)

    def test_patience_respects_minimum_epoch_and_terminal_state(self):
        p = ValidationPatience()
        for epoch in range(1,20):self.assertFalse(p.observe(E(epoch,.5,1)))
        self.assertTrue(p.observe(E(20,.5,1)))
        with self.assertRaises(ValueError):p.observe(E(21,.6,1))

    def test_patience_strict_delta_separate_from_checkpoint_ranking(self):
        p=ValidationPatience();records=[E(1,0,1),E(2,1e-4,.9),E(3,1.0001e-4,.8)]
        p.observe(records[0]);p.observe(records[1])
        self.assertEqual(p.without_improvement,1)
        self.assertEqual(select_checkpoint(records[:2]).epoch,2)
        p.observe(records[2]);self.assertEqual(p.without_improvement,0)

    def test_maximum_epoch_and_nonsequential_rejection(self):
        p=ValidationPatience()
        with self.assertRaises(ValueError):p.observe(E(2,.1,1))
        for epoch in range(1,100):self.assertFalse(p.observe(E(epoch,epoch*.005,1)))
        self.assertTrue(p.observe(E(100,.5,1)))

    def test_screen_uses_seed42_ce_tie_order_and_returns_exact_max(self):
        candidates={"IMAGENET-A":V(42,E(1,.70005,.5),True),"RANDOM-A":V(42,E(1,.7,.5),True)}
        result=select_screen(candidates,kind="initialization",eligible_initializations=['RANDOM-A','IMAGENET-A'])
        self.assertEqual(result["selected"],"RANDOM-A")
        self.assertEqual(result["exact_maximum"],.70005)
        with self.assertRaises(ValueError):select_screen({"HARD":V(43,E(1,.5,1),True)},kind="objective")

    def test_failed_seed_cannot_be_ranked(self):
        with self.assertRaises(ValueError):V(42,E(1,.5,1),False)
        with self.assertRaises(ValueError):select_screen({"HARD":None},kind="objective")

    def test_screen_cannot_silently_drop_an_eligible_or_objective_arm(self):
        one={'RANDOM-A':V(42,E(1,.6,1),True)}
        with self.assertRaises(ValueError):select_screen(one,kind='initialization')
        with self.assertRaises(ValueError):
            select_screen(one,kind='initialization',eligible_initializations=['RANDOM-A','IMAGENET-A'])
        with self.assertRaises(ValueError):select_screen({'HARD':V(42,E(1,.6,1),True)},kind='objective')

    def test_full_objective_screen_uses_fixed_tie_order(self):
        candidates={name:V(42,E(1,.6,1),True) for name in ('HARD','LEGACY','SHARED-010','SEPARATE-010')}
        self.assertEqual(select_screen(candidates,kind='objective')['selected'],'HARD')

    def test_repeat_mean_all_seeds_not_best_seed(self):
        def family(scores):return [V(seed,E(2,f1,1),True) for seed,f1 in zip((42,43,44),scores)]
        families={"HARD":family((.9,.4,.4)),"SEPARATE-010":family((.6,.6,.6))}
        self.assertEqual(select_repeated_family(families)["selected"],"SEPARATE-010")
        families["HARD"]=families["HARD"][:2]
        with self.assertRaises(ValueError):select_repeated_family(families)

    def test_repeated_families_reject_duplicate_seeds_or_extra_candidates(self):
        records=[V(42,E(1,.5,1),True)]*3
        with self.assertRaises(ValueError):select_repeated_family({"HARD":records,"LEGACY":records})
        with self.assertRaises(ValueError):select_repeated_family({"HARD":records})


class PureRngTests(unittest.TestCase):
    def test_private_order_and_augmentation_streams_are_independent(self):
        expected=private_random(42,"shuffle").sample(range(20),20)
        augment=private_random(42,"augmentation")
        for _ in range(100):augment.uniform(-10,10)
        self.assertEqual(private_random(42,"shuffle").sample(range(20),20),expected)
        self.assertNotEqual(private_random(43,"shuffle").sample(range(20),20),expected)
        self.assertEqual(private_random(42,"augmentation").random(),private_random(42,"augmentation").random())

    def test_seed_derivation_and_unknown_stream_rejection(self):
        import hashlib
        expected=int.from_bytes(hashlib.sha256(b'performance-v2-rng-v1|42|shuffle').digest()[:8],"big")%(2**32)
        self.assertEqual(stream_seed(42,"shuffle"),expected)
        for seed,name in [(1,"shuffle"),(True,"shuffle"),(42,"HARD")]:
            with self.assertRaises(ValueError):stream_seed(seed,name)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.contract={"run_id":"synthetic-run-42","model_id":r.MODEL_ID,"protocol_version":2,
            "initialization":"RANDOM","objective":"HARD","seed":42,"recipe":"A",
            "bindings":{name:r.digest(name.encode()) for name in r.BINDINGS},
            "administrative":{"screen_or_repeat_phase_label":"phase1"}}
        self.outputs={"synthetic-output.json":r.digest(b'invented')}
        self.manifest=r.canonical({"files":self.outputs})
        self.receipt=r.canonical({"status":"complete","run_id":self.contract["run_id"],
            "contract_sha256":r.digest(r.canonical(r.scientific_contract(self.contract))),
            "output_manifest_sha256":r.digest(self.manifest)})

    def call(self,requested=None,**kwargs):
        args=dict(completion_bytes=self.receipt,completion_sha256=r.digest(self.receipt),
                  manifest_bytes=self.manifest,verified_output_hashes=self.outputs)
        args.update(kwargs)
        return r.require_exact_reuse(self.contract,requested or self.contract,**args)

    def test_exact_reuse_allows_only_administrative_alias(self):
        request=copy.deepcopy(self.contract);request['administrative']['screen_or_repeat_phase_label']='phase2'
        result=self.call(request)
        self.assertTrue(result['reuse_compatible']);self.assertFalse(result['pilot_execution_authorized'])
        self.assertFalse(result['counts_as_additional_run'])

    def test_each_execution_or_input_digest_mismatch_rejected(self):
        for name in r.BINDINGS:
            request=copy.deepcopy(self.contract);request['bindings'][name]='f'*64
            with self.subTest(binding=name),self.assertRaises(ValueError):self.call(request)

    def test_missing_hash_unknown_field_and_old_plain_cnn_rejected(self):
        cases=[]
        c=copy.deepcopy(self.contract);c['bindings'].pop(r.BINDINGS[0]);cases.append(c)
        c=copy.deepcopy(self.contract);c['learning_rate']=.5;cases.append(c)
        c=copy.deepcopy(self.contract);c['model_id']='plain_resnet50';cases.append(c)
        c=copy.deepcopy(self.contract);c['administrative']['seed']='43';cases.append(c)
        for c in cases:
            with self.assertRaises(ValueError):self.call(c)

    def test_objective_seed_recipe_and_run_identity_mismatch_rejected(self):
        for key,value in [('objective','SEPARATE-010'),('seed',43),('recipe','B'),('run_id','new-run')]:
            c=copy.deepcopy(self.contract);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.call(c)

    def test_incomplete_receipt_hash_drift_and_output_mismatch_rejected(self):
        receipt=json.loads(self.receipt);receipt['status']='partial';raw=r.canonical(receipt)
        cases=[dict(completion_bytes=raw,completion_sha256=r.digest(raw)),
               dict(completion_sha256='a'*64),dict(manifest_bytes=b'{}'),
               dict(verified_output_hashes={}),dict(verified_output_hashes={'extra':'a'*64})]
        for changes in cases:
            with self.assertRaises(ValueError):self.call(**changes)

    def test_changed_contract_cannot_rebind_old_completion(self):
        self.contract['bindings'][r.BINDINGS[1]]='a'*64
        with self.assertRaises(ValueError):self.call()

    def test_config_binding_names_and_inactive_flags_match(self):
        config=json.loads((ROOT/'configs/performance_v2/pilot_protocol_v2.json').read_bytes())
        self.assertEqual(list(r.BINDINGS),config['run_reuse']['equality_required'])
        self.assertFalse(config['pilot_execution_authorized'])
        self.assertFalse(config['baseline_retraining_required'])


if __name__ == '__main__':unittest.main()
