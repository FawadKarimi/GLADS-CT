"""Real CPU forward/backward checks with invented tensors, never optimizer steps."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consendxformer.performance_v2.model import R50Hybrid
from consendxformer.performance_v2.objectives import objective_loss
from consendxformer.performance_v2.rng import cpu_generator
from consendxformer.performance_v2.streams import TrainingStreams, trace_sha256


def images(n):
    return torch.randn(n, 3, 224, 224, generator=cpu_generator(42, "augmentation")) + .3


class TrainingModeTests(unittest.TestCase):
    def test_every_batch_norm_updates_in_train_and_is_frozen_in_eval(self):
        model = R50Hybrid(objective="SEPARATE-010").train()
        norms = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)}
        before = {name: (layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone())
                  for name, layer in norms.items()}
        with torch.no_grad():
            model(images(2))
        for name, layer in norms.items():
            mean, variance, count = before[name]
            self.assertEqual(layer.num_batches_tracked.item(), count.item() + 1, name)
            self.assertTrue(not torch.equal(mean, layer.running_mean) or not torch.equal(variance, layer.running_var), name)
            self.assertTrue(layer.weight.requires_grad and layer.bias.requires_grad, name)
        frozen = {name: (layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone())
                  for name, layer in norms.items()}
        model.eval()
        with torch.no_grad():
            model(images(1))
        for name, layer in norms.items():
            for left, right in zip(frozen[name], (layer.running_mean, layer.running_var, layer.num_batches_tracked)):
                self.assertTrue(torch.equal(left, right), name)

    def test_combined_loss_connects_every_parameter_and_all_intended_scopes(self):
        for objective in ("HARD", "LEGACY", "SHARED-010", "SEPARATE-010", "CAPACITY-010"):
            with self.subTest(objective=objective):
                model = R50Hybrid(objective=objective, capacity_control=objective == "CAPACITY-010").train()
                output = model(images(1))
                result = objective_loss(objective, output["logits"], torch.tensor([2]), torch.tensor([.7, 1.2, 1.5]),
                                        reader_targets=torch.tensor([[.2, .3, .5]]), disagreement=torch.tensor([.4]),
                                        reader_logits=output["reader_logits"], capacity_control=objective == "CAPACITY-010")
                names, parameters = zip(*model.named_parameters())
                gradients = torch.autograd.grad(result["total"], parameters, allow_unused=True)
                magnitudes = {}
                for name, gradient in zip(names, gradients):
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    root = name.split(".")[0]
                    magnitudes[root] = magnitudes.get(root, 0) + gradient.abs().sum().item()
                for root, magnitude in magnitudes.items():
                    self.assertGreater(magnitude, 0, root)
                self.assertIn("backbone", magnitudes)
                self.assertIn("diagnostic_head", magnitudes)
                self.assertEqual("reader_head" in magnitudes, objective in ("SEPARATE-010", "CAPACITY-010"))
                del model, output, result, gradients, parameters

    def test_separate_reader_has_no_direct_diagnostic_gradient_and_hard_gradient_is_unchanged(self):
        model = R50Hybrid(objective="SEPARATE-010").train()
        output = model(images(1))
        losses = objective_loss("SEPARATE-010", output["logits"], torch.tensor([1]), torch.ones(3),
                                reader_targets=torch.tensor([[.6, .1, .3]]), reader_logits=output["reader_logits"])
        diagnostic = tuple(model.diagnostic_head.parameters())
        reader = tuple(model.reader_head.parameters())
        reader_to_diagnostic = torch.autograd.grad(losses["reader"], diagnostic, allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None for g in reader_to_diagnostic))
        hard_to_reader = torch.autograd.grad(losses["hard"], reader, allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None for g in hard_to_reader))
        hard = torch.autograd.grad(losses["hard"], diagnostic, retain_graph=True)
        total = torch.autograd.grad(losses["total"], diagnostic, retain_graph=True)
        for left, right in zip(hard, total):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        shared = (model.backbone.conv1.weight, model.token_projection.weight, model.c1_fusion.refine[0].weight)
        for gradient in torch.autograd.grad(losses["reader"], shared):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_extra_head_preserves_initial_state_live_data_streams_order_and_augmentation(self):
        left, right = TrainingStreams(42), TrainingStreams(42)
        ids = tuple("invented-%02d" % i for i in range(33))
        self.assertEqual(left.epoch_plan(ids, epoch=1), right.epoch_plan(ids, epoch=1))
        before = torch.get_rng_state().clone()
        hard = R50Hybrid(objective="HARD")
        separate = R50Hybrid(objective="SEPARATE-010")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        separate_state = separate.state_dict()
        for name, value in hard.state_dict().items():
            self.assertTrue(torch.equal(value, separate_state[name]), name)
        for epoch in (2, 3):
            a, b = left.epoch_plan(ids, epoch=epoch), right.epoch_plan(ids, epoch=epoch)
            self.assertEqual(a["batches"], b["batches"])
            self.assertEqual(a["augmentation"], b["augmentation"])
            self.assertEqual(trace_sha256(a), trace_sha256(b))
            self.assertEqual([len(batch) for batch in a["batches"]], [16, 16, 1])

    def test_singleton_and_actual_seven_item_tail_are_valid_in_train_mode(self):
        model = R50Hybrid(objective="SEPARATE-010").train()
        self.assertEqual(807 % 16, 7)
        for count in (7, 1):
            with self.subTest(count=count), torch.no_grad():
                result = model(images(count))
                self.assertEqual(tuple(result["logits"].shape), (count, 3))
                self.assertEqual(tuple(result["reader_logits"].shape), (count, 3))
                losses = objective_loss("SEPARATE-010", result["logits"], torch.arange(count) % 3, torch.ones(3),
                                        reader_targets=torch.full((count, 3), 1 / 3), reader_logits=result["reader_logits"])
                self.assertTrue(torch.isfinite(losses["total"]))
                self.assertEqual(losses["hard_denominator"].item(), count)


if __name__ == "__main__":
    unittest.main()
