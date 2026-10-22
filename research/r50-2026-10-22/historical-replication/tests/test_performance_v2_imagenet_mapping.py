"""Randomly initialized CPU tensors only; no real checkpoint bytes or forwards."""
import unittest
from unittest.mock import patch
import torch
from torchvision.models import resnet50
from consendxformer.performance_v2.model import R50Hybrid
from consendxformer.performance_v2 import imagenet_mapping as m


class ImageNetMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng():
            torch.manual_seed(17)
            cls.source = resnet50(weights=None)
            cls.target = R50Hybrid(seed=42)
        cls.state = cls.source.state_dict()

    def check(self, state=None):
        return m.verify_mapping(self.state if state is None else state, self.source, self.target)

    def test_exact_mapping_includes_all_buffers_and_excludes_only_fc(self):
        before = self.target.backbone.conv1.weight.detach().clone()
        result = self.check()
        self.assertEqual(len(result['mapping']), 318)
        self.assertEqual(result['excluded_source_keys'], ['fc.bias', 'fc.weight'])
        self.assertIn('bn1.num_batches_tracked', result['mapping'])
        self.assertIn('diagnostic_head.weight', result['new_hybrid_keys'])
        self.assertFalse(result['load_performed'])
        self.assertTrue(torch.equal(before, self.target.backbone.conv1.weight))

    def test_missing_source_head_is_rejected(self):
        state = dict(self.state); del state['fc.weight']
        with self.assertRaises(ValueError): self.check(state)

    def test_extra_source_key_is_rejected(self):
        state = dict(self.state); state['unreviewed'] = torch.zeros(1)
        with self.assertRaises(ValueError): self.check(state)

    def test_wrong_source_head_shape(self):
        state = dict(self.state); state['fc.weight'] = torch.zeros(3, 2048)
        with self.assertRaises(ValueError): self.check(state)

    def test_dtype_and_buffer_missing(self):
        for key, value in [('bn1.num_batches_tracked', torch.tensor(0.)),
                           ('conv1.weight', self.state['conv1.weight'].double())]:
            state = dict(self.state); state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(state)
        state = dict(self.state); del state['bn1.running_mean']
        with self.assertRaises(ValueError): self.check(state)

    def test_nonfinite_and_negative_variance(self):
        for key, value in [('bn1.weight', float('nan')), ('bn1.running_var', -1.)]:
            state = dict(self.state); state[key] = torch.full_like(state[key], value)
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(state)

    def test_aliases_and_views_rejected(self):
        state = dict(self.state); state['bn1.bias'] = state['bn1.weight']
        with self.assertRaisesRegex(ValueError, 'aliased'): self.check(state)
        state = dict(self.state); state['bn1.bias'] = torch.zeros(65)[1:]
        with self.assertRaises(ValueError): self.check(state)

    def test_stride_semantics_not_just_shapes(self):
        with patch.object(self.source.layer2[0].conv2, 'stride', (1, 1)):
            with self.assertRaisesRegex(ValueError, 'semantic'): self.check()

    def test_bn_and_pool_semantics(self):
        with patch.object(self.source.bn1, 'eps', 1e-3):
            with self.assertRaises(ValueError): self.check()
        with patch.object(self.source.maxpool, 'ceil_mode', True):
            with self.assertRaises(ValueError): self.check()

    def test_plain_control_cannot_be_renamed_hybrid(self):
        with self.assertRaises(ValueError): m.verify_mapping(self.state, self.source, self.source)

    def test_hybrid_backbone_semantics_required(self):
        with patch.object(self.target.backbone.conv1, 'padding_mode', 'reflect'):
            with self.assertRaises(ValueError): self.check()

    def test_container_stride_and_forward_hooks_rejected(self):
        with patch.object(self.source.layer2[0], 'stride', 1):
            with self.assertRaises(ValueError): self.check()
        hook = self.source.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(ValueError, 'hooks'): self.check()
        finally:
            hook.remove()

    def test_real_loader_not_available(self):
        self.assertFalse(hasattr(m, 'load_weights'))
        with self.assertRaisesRegex(ValueError, 'pretrained integration remains blocked'):
            R50Hybrid(initialization='IMAGENET')
