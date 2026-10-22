"""CPU synthetic tensors only; gradients are inspected, never optimizer steps."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from consendxformer.performance_v2.model import R50Hybrid
from consendxformer.performance_v2.objectives import objective_loss, class_weights_from_training_counts
from consendxformer.performance_v2.optimization import optimizer_groups, make_adamw, epoch_multiplier, set_epoch_learning_rates
from consendxformer.performance_v2.rng import cpu_generator


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hard=R50Hybrid(seed=42,objective='HARD').eval()
        cls.separate=R50Hybrid(seed=42,objective='SEPARATE-010').eval()
        cls.images=torch.randn(1,3,224,224,generator=cpu_generator(42,'augmentation'),device='cpu')

    @classmethod
    def tearDownClass(cls):
        del cls.hard,cls.separate,cls.images

    def test_exact_tensor_shapes_spatial_mass_and_probabilities(self):
        with torch.no_grad():out=self.separate(self.images)
        shapes={'c1_features':(1,512,28,28),'c2_features':(1,1024,14,14),
                'c3_features':(1,2048,7,7),'transformer_features':(1,128,7,7),
                'attention_map':(1,1,7,7),'deep_fused_features':(1,128,7,7),
                'fused_features':(1,128,28,28),'pooled_features':(1,128),
                'logits':(1,3),'reader_logits':(1,3)}
        for key,shape in shapes.items():self.assertEqual(tuple(out[key].shape),shape)
        torch.testing.assert_close(out['attention_map'].sum(),torch.tensor(1.0))
        torch.testing.assert_close(out['probs'].sum(1),torch.ones(1))
        self.assertNotIn('ordinal_logits',out)

    def test_shared_initialization_identical_across_hard_and_separate_arms(self):
        other=self.separate.state_dict()
        for name,state in self.hard.state_dict().items():
            self.assertTrue(torch.equal(state,other[name]),name)
        self.assertIsNone(self.hard.reader_head)

    def test_legacy_and_shared_arms_preserve_common_state(self):
        for objective in ('LEGACY','SHARED-010'):
            model=R50Hybrid(seed=42,objective=objective).eval()
            for name,state in self.hard.state_dict().items():
                self.assertTrue(torch.equal(state,model.state_dict()[name]),name)
            del model

    def test_private_streams_leave_global_rng_and_data_sequences_unchanged(self):
        before=torch.random.get_rng_state().clone()
        order=torch.randperm(13,generator=cpu_generator(42,'shuffle'))
        model=R50Hybrid(seed=43,objective='HARD')
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(order,torch.randperm(13,generator=cpu_generator(42,'shuffle'))))
        self.assertFalse(torch.equal(model.diagnostic_head.weight,self.hard.diagnostic_head.weight))

    def test_synthetic_backbone_change_does_not_change_new_shared_modules(self):
        model=R50Hybrid(seed=42,objective='HARD')
        # Invented in-memory mutation only, NOT pretrained loading or equivalence.
        with torch.no_grad():model.backbone.conv1.weight.fill_(.125)
        for name,value in self.hard.state_dict().items():
            if not name.startswith('backbone.'):
                self.assertTrue(torch.equal(value,model.state_dict()[name]),name)

    def test_diagnostic_inference_never_calls_reader_head(self):
        with patch.object(self.separate.reader_head,'forward',side_effect=AssertionError('reader in diagnosis')):
            with torch.no_grad():probs=self.separate.diagnostic_probabilities(self.images)
        with torch.no_grad():expected=self.hard.diagnostic_probabilities(self.images)
        torch.testing.assert_close(probs,expected,atol=0,rtol=0)

    def test_real_hybrid_reader_gradient_reaches_shared_encoder_not_diagnostic_head(self):
        out=self.separate(self.images)
        loss=objective_loss('SEPARATE-010',out['logits'],torch.tensor([1]),torch.ones(3),
            reader_targets=torch.tensor([[.2,.5,.3]]),reader_logits=out['reader_logits'])['reader']
        grads=torch.autograd.grad(loss,(self.separate.diagnostic_head.weight,
              self.separate.reader_head.weight,self.separate.backbone.conv1.weight),allow_unused=True)
        self.assertIsNone(grads[0])
        for grad in grads[1:]:
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)

    def test_wrong_shape_dtype_nan_and_annotation_forward_arguments_rejected(self):
        for images in (torch.zeros(1,1,224,224),torch.zeros(1,3,64,64),
                       torch.zeros(1,3,224,224,dtype=torch.float64),torch.full((1,3,224,224),float('nan'))):
            with self.assertRaises(ValueError):self.hard(images)
        with self.assertRaises(TypeError):self.hard(self.images,reader_targets=torch.ones(1,3))

    def test_pretrained_requests_and_unreleased_objectives_rejected(self):
        for initialization in ('IMAGENET','RADIMAGENET'):
            with self.assertRaisesRegex(ValueError,'pretrained integration'):
                R50Hybrid(initialization=initialization)
        for objective in ('SEPARATE-030','CAPACITY-010','B1'):
            with self.assertRaises(ValueError):R50Hybrid(objective=objective)

    def test_conditional_capacity_topology_matches_separate(self):
        capacity=R50Hybrid(objective='CAPACITY-010',capacity_control=True)
        for name,value in self.separate.state_dict().items():
            self.assertTrue(torch.equal(value,capacity.state_dict()[name]),name)
        self.assertFalse(capacity.pilot_execution_authorized)

    def test_exhaustive_groups_cover_all_parameters_and_special_cases(self):
        groups,ledger=optimizer_groups(self.separate)
        assigned=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(set(assigned),{id(p) for p in self.separate.parameters()})
        table={row['name']:row for row in ledger}
        self.assertEqual(table['position.position']['group'],'encoder_bias_normalization')
        self.assertEqual(table['transformer.0.attention.in_proj_weight']['weight_decay'],1e-4)
        self.assertEqual(table['transformer.0.attention.in_proj_bias']['weight_decay'],0)
        self.assertEqual(table['reader_head.weight']['lr'],1e-3)
        for row in ledger:
            if row['name'].endswith('bias'):self.assertEqual(row['weight_decay'],0)

    def test_unknown_frozen_or_aliased_parameter_rejected(self):
        model=self.hard
        model.register_parameter('unclassified',nn.Parameter(torch.zeros(1)))
        try:
            with self.assertRaises(ValueError):optimizer_groups(model)
        finally:del model.unclassified
        model.alias=model.diagnostic_head
        try:
            with self.assertRaisesRegex(ValueError,'aliased'):optimizer_groups(model)
        finally:del model.alias
        model.diagnostic_head.weight.requires_grad_(False)
        try:
            with self.assertRaises(ValueError):optimizer_groups(model)
        finally:model.diagnostic_head.weight.requires_grad_(True)

    def test_adamw_hyperparameters_and_schedule_without_steps(self):
        optimizer,ledger=make_adamw(self.hard)
        self.assertEqual(optimizer.defaults['betas'],(.9,.999));self.assertEqual(optimizer.defaults['eps'],1e-8)
        for epoch,expected in [(1,.1),(5,1),(6,1),(100,.01)]:
            self.assertAlmostEqual(epoch_multiplier(epoch),expected)
            set_epoch_learning_rates(optimizer,epoch)
            for group in optimizer.param_groups:
                base=1e-3 if group['name'].startswith('head_') else 1e-4
                self.assertAlmostEqual(group['lr'],base*expected)
        self.assertEqual(len(optimizer.state),0)
        optimizer.param_groups[0]['weight_decay']=.2
        with self.assertRaises(ValueError):set_epoch_learning_rates(optimizer,10)
        for epoch in (0,101,True):
            with self.assertRaises(ValueError):epoch_multiplier(epoch)


class LossTests(unittest.TestCase):
    def setUp(self):
        self.d=torch.tensor([[2.,-.5,1.],[-1.,1.,.3]],requires_grad=True)
        self.y=torch.tensor([0,2]);self.w=torch.tensor([1.,4.,2.])
        self.q=torch.tensor([[.1,.6,.3],[.6,.1,.3]])
        self.u=torch.tensor([.25,1.])
        self.reader=torch.tensor([[.1,1.,-.2],[.3,-.5,.8]],requires_grad=True)

    def expected_log(self,tensor):
        return [[x-math.log(sum(math.exp(v) for v in row)) for x in row]
                for row in tensor.detach().tolist()]

    def test_weighted_hard_ce_independent_reduction(self):
        values=self.expected_log(self.d)
        expected=(-values[0][0]-2*values[1][2])/3
        out=objective_loss('HARD',self.d,self.y,self.w)
        self.assertAlmostEqual(out['total'].item(),expected,places=6)
        self.assertEqual(out['hard_denominator'].item(),3)

    def test_legacy_is_batch_mass_not_mean_normalized_samples(self):
        logs=self.expected_log(self.d);masses=[];numerators=[]
        for i in range(2):
            target=[(1-self.u[i].item())*int(c==self.y[i].item())+self.u[i].item()*self.q[i,c].item() for c in range(3)]
            mass=[t*w for t,w in zip(target,self.w.tolist())]
            masses.append(sum(mass));numerators.append(-sum(m*l for m,l in zip(mass,logs[i])))
        out=objective_loss('LEGACY',self.d,self.y,self.w,reader_targets=self.q,disagreement=self.u)
        expected=sum(numerators)/sum(masses)
        self.assertAlmostEqual(out['total'].item(),expected,places=6)
        self.assertNotAlmostEqual(expected,sum(n/m for n,m in zip(numerators,masses))/2,places=4)

    def test_shared_and_separate_reader_ce_unweighted_mean(self):
        for name,logits in [('SHARED-010',self.d),('SEPARATE-010',self.reader)]:
            logs=self.expected_log(logits)
            expected=-sum(q*l for row,lp in zip(self.q.tolist(),logs) for q,l in zip(row,lp))/2
            out=objective_loss(name,self.d,self.y,self.w,reader_targets=self.q,
                reader_logits=self.reader if name=='SEPARATE-010' else None)
            self.assertAlmostEqual(out['reader'].item(),expected,places=6)
            self.assertAlmostEqual(out['total'].item(),out['hard'].item()+.1*expected,places=6)

    def test_leaf_gradient_routing_for_shared_and_separate(self):
        shared=objective_loss('SHARED-010',self.d,self.y,self.w,reader_targets=self.q)
        grad=torch.autograd.grad(shared['reader'],self.d)[0]
        self.assertGreater(grad.abs().sum().item(),0)
        separate=objective_loss('SEPARATE-010',self.d,self.y,self.w,reader_targets=self.q,reader_logits=self.reader)
        hard_to_reader=torch.autograd.grad(separate['hard'],self.reader,allow_unused=True,retain_graph=True)[0]
        reader_to_diagnostic=torch.autograd.grad(separate['reader'],self.d,allow_unused=True)[0]
        self.assertIsNone(hard_to_reader);self.assertIsNone(reader_to_diagnostic)

    def test_capacity_hard_reader_ce_has_no_reader_target_dependence(self):
        out=objective_loss('CAPACITY-010',self.d,self.y,self.w,reader_logits=self.reader,capacity_control=True)
        logs=self.expected_log(self.reader)
        self.assertAlmostEqual(out['reader'].item(),(-logs[0][0]-logs[1][2])/2,places=6)

    def test_invalid_targets_and_disagreement_fail_without_repair(self):
        for q in (self.q*2,torch.full_like(self.q,float('nan')),self.q.clone().requires_grad_()):
            with self.assertRaises(ValueError):objective_loss('SHARED-010',self.d,self.y,self.w,reader_targets=q)
        for u in (torch.tensor([-1.,.1]),torch.tensor([.1]),torch.tensor([float('inf'),0])):
            with self.assertRaises(ValueError):objective_loss('LEGACY',self.d,self.y,self.w,reader_targets=self.q,disagreement=u)

    def test_invalid_labels_weights_reader_heads_and_empty_batch_rejected(self):
        with self.assertRaises(ValueError):objective_loss('HARD',self.d,self.y.float(),self.w)
        for w in (torch.zeros(3),self.w.clone().requires_grad_(),torch.tensor([1.,float('inf'),1.])):
            with self.assertRaises(ValueError):objective_loss('HARD',self.d,self.y,w)
        with self.assertRaises(ValueError):objective_loss('SEPARATE-010',self.d,self.y,self.w,reader_targets=self.q)
        with self.assertRaises(ValueError):objective_loss('HARD',self.d,self.y,self.w,reader_logits=self.reader)
        with self.assertRaises(ValueError):objective_loss('HARD',self.d[:0],self.y[:0],self.w)

    def test_training_class_weights_formula_and_counts_rejection(self):
        torch.testing.assert_close(class_weights_from_training_counts([2,3,5]),torch.tensor([10/6,10/9,10/15]))
        for counts in ([0,1,2],[1,2],[True,2,3]):
            with self.assertRaises(ValueError):class_weights_from_training_counts(counts)


if __name__ == '__main__':unittest.main()
