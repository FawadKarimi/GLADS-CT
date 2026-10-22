"""Actual synthetic modules and invented checkpoint states; no optimizer updates."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consendxformer.performance_v2 import checkpoint as ck
from consendxformer.performance_v2 import reuse as r
from consendxformer.performance_v2.model import R50Hybrid
from consendxformer.performance_v2.optimization import optimizer_groups
from consendxformer.performance_v2.rng import initialization_scope
from consendxformer.performance_v2.streams import TrainingStreams, capture_cpu_rng
from test_performance_v2_lifecycle import Fixture


class InventoryModel(nn.Module):
    """Multiple parameters in every group, real BN buffers and special rules."""
    def __init__(self):
        super().__init__()
        with initialization_scope(42, "encoder_initialization"):
            self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm2d(2), nn.Linear(2, 2))
            self.transformer = nn.MultiheadAttention(8, 2, batch_first=True)
            self.position = nn.Module()
            self.position.register_parameter("position", nn.Parameter(torch.zeros(1, 49, 128)))
            self.diagnostic_head = nn.Linear(2, 3)
            self.reader_head = nn.Linear(2, 3)


class CheckpointInventoryTests(unittest.TestCase):
    def setUp(self):
        fixture = Fixture()
        self.inputs, self.contract = fixture.inputs, fixture.contract
        self.model = InventoryModel()
        _, self.ledger = optimizer_groups(self.model)

    def spec(self, ledger=None, model=None):
        return ck.execution_spec(self.contract, self.inputs, model=self.model if model is None else model,
                                 optimizer_ledger=self.ledger if ledger is None else ledger)

    def components(self, spec):
        groups, states = [], {}
        next_id = 0
        for group in ck.GROUPS:
            rows = [row for row in spec["optimizer_ledger"] if row["group"] == group]
            ids = list(range(next_id, next_id + len(rows)))
            next_id += len(rows)
            groups.append({"name": group, "params": ids, "lr": rows[0]["lr"] * ck.epoch_multiplier(1),
                           "weight_decay": rows[0]["weight_decay"], "betas": (.9, .999), "eps": 1e-8,
                           "amsgrad": False, "maximize": False})
            for param_id, row in zip(ids, rows):
                # Invented state, not a step taken on this model or optimizer.
                states[param_id] = {"step": torch.tensor(1.), "exp_avg": torch.zeros(row["shape"]),
                                    "exp_avg_sq": torch.zeros(row["shape"])}
        streams = TrainingStreams(42)
        streams.epoch_plan(self.inputs.ids("train"), epoch=1)
        return {"model": copy.deepcopy(self.model.state_dict()), "optimizer": {"param_groups": groups, "state": states},
                "scheduler": ck.scheduler_state(1), "rng": capture_cpu_rng(streams)}

    def payload(self, spec):
        return {"schema": "performance_v2_checkpoint_v1", "spec": spec, "components": self.components(spec),
                "progress": {"epoch": 1, "updates": 1, "elapsed_seconds": 0.},
                "selection": ck.selection_state([{"epoch": 1, "macro_f1": .5, "diagnostic_ce": 1., "split": "validation"}])}

    def check_payload(self, payload, requested):
        raw = ck.encode_state(payload)
        return ck.validate_checkpoint(raw, expected_sha256=r.digest(raw), requested_spec=requested)

    def test_inventory_is_complete_and_buffers_are_separate(self):
        spec = self.spec()
        parameters = {row["name"] for row in spec["trainable_parameter_inventory"]}
        buffers = {row["name"] for row in spec["buffer_inventory"]}
        self.assertEqual(parameters, {name for name, p in self.model.named_parameters() if p.requires_grad})
        self.assertEqual(buffers, {name for name, _ in self.model.named_buffers()})
        self.assertFalse(parameters & buffers)
        self.assertEqual(parameters | buffers, set(spec["model_layout"]))
        self.assertEqual(parameters, {row["name"] for row in spec["optimizer_ledger"]})
        self.assertIn("backbone.1.running_mean", buffers)
        self.assertIn("backbone.1.num_batches_tracked", buffers)
        result = self.check_payload(self.payload(spec), spec)
        self.assertEqual(result["spec"], spec)

    def test_omission_rejected_even_when_all_four_groups_remain(self):
        for omitted in self.ledger:
            rows = [row for row in self.ledger if row["name"] != omitted["name"]]
            self.assertEqual({row["group"] for row in rows}, set(ck.GROUPS))
            with self.subTest(omitted=omitted["name"]), self.assertRaisesRegex(ValueError, "complete reviewed"):
                self.spec(rows)

    def test_buffer_substitution_with_same_shape_and_all_groups_rejected(self):
        rows = copy.deepcopy(self.ledger)
        row = next(row for row in rows if row["name"] == "backbone.1.bias")
        row["name"] = "backbone.1.running_mean"
        self.assertEqual(row["shape"], list(self.model.backbone[1].running_mean.shape))
        self.assertEqual({row["group"] for row in rows}, set(ck.GROUPS))
        with self.assertRaisesRegex(ValueError, "complete reviewed"):
            self.spec(rows)

    def test_extra_duplicate_and_reordered_ledger_entries_rejected(self):
        extra = dict(self.ledger[0], name="backbone.unregistered_weight")
        for rows in (self.ledger + [extra], self.ledger + [self.ledger[0]], list(reversed(self.ledger))):
            with self.subTest(names=[row["name"] for row in rows]), self.assertRaises(ValueError):
                self.spec(rows)

    def test_wrong_group_rejected_even_with_consistent_numeric_rates(self):
        cases = [("backbone.0.weight", "head_weights"),
                 ("backbone.1.weight", "encoder_weights"),
                 ("backbone.0.bias", "encoder_weights"),
                 ("transformer.in_proj_weight", "encoder_bias_normalization"),
                 ("transformer.in_proj_bias", "encoder_weights"),
                 ("position.position", "encoder_weights"),
                 ("reader_head.weight", "encoder_weights")]
        for name, wrong in cases:
            rows = copy.deepcopy(self.ledger)
            row = next(row for row in rows if row["name"] == name)
            row.update(group=wrong, lr=1e-4 if wrong.startswith("encoder") else 1e-3,
                       weight_decay=1e-4 if wrong.endswith("weights") else 0.)
            self.assertEqual({row["group"] for row in rows}, set(ck.GROUPS))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.spec(rows)

    def test_model_object_alias_rejected_even_with_deduplicated_ledger(self):
        self.model.backbone[2].weight = self.model.backbone[0].weight
        rows = [row for row in self.ledger if row["name"] != "backbone.2.weight"]
        with self.assertRaisesRegex(ValueError, "aliased"):
            self.spec(rows)

    def test_distinct_parameters_sharing_storage_are_rejected(self):
        self.model.backbone[2].weight = nn.Parameter(self.model.backbone[0].weight.detach())
        self.assertIsNot(self.model.backbone[2].weight, self.model.backbone[0].weight)
        _, rows = optimizer_groups(self.model)
        with self.assertRaisesRegex(ValueError, "shared.*storage"):
            self.spec(rows)

    def test_parameter_buffer_alias_and_uncheckpointed_buffer_are_rejected(self):
        self.model.register_buffer("parameter_alias", self.model.backbone[0].weight.detach())
        with self.assertRaisesRegex(ValueError, "shared.*storage"):
            self.spec()
        del self.model.parameter_alias
        self.model.register_buffer("uncheckpointed", torch.zeros(1), persistent=False)
        with self.assertRaisesRegex(ValueError, "persistent buffers"):
            self.spec()

    def test_frozen_parameter_cannot_silently_leave_reviewed_trainable_inventory(self):
        self.model.backbone[0].weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "all parameters trainable"):
            self.spec()

    def test_component_validation_rechecks_exact_ledger_inventory_coverage(self):
        spec = self.spec()
        components = self.components(spec)
        altered = copy.deepcopy(spec)
        altered["optimizer_ledger"] = altered["optimizer_ledger"][1:]
        self.assertEqual({row["group"] for row in altered["optimizer_ledger"]}, set(ck.GROUPS))
        with self.assertRaisesRegex(ValueError, "every actual trainable"):
            ck.validate_components(altered, components, epoch=1, updates=1)

    def test_missing_moments_rejected_for_every_parameter_not_just_each_group(self):
        spec = self.spec()
        components = self.components(spec)
        ck.validate_components(spec, components, epoch=1, updates=1)
        for param_id in components["optimizer"]["state"]:
            for member in (None, "exp_avg", "exp_avg_sq"):
                bad = copy.deepcopy(components)
                if member is None:
                    del bad["optimizer"]["state"][param_id]
                else:
                    del bad["optimizer"]["state"][param_id][member]
                with self.subTest(param_id=param_id, member=member), self.assertRaisesRegex(ValueError, "moments"):
                    ck.validate_components(spec, bad, epoch=1, updates=1)

    def test_checkpoint_resume_rejects_missing_moments_and_inventory_drift(self):
        spec = self.spec()
        payload = self.payload(spec)
        bad = copy.deepcopy(payload)
        del bad["components"]["optimizer"]["state"][0]
        with self.assertRaisesRegex(ValueError, "moments"):
            self.check_payload(bad, spec)
        bad = copy.deepcopy(payload)
        bad["spec"]["trainable_parameter_inventory"].pop(0)
        bad["spec"]["optimizer_ledger"].pop(0)
        bad["spec"]["model_inventory_sha256"] = r.digest(r.canonical({
            "trainable_parameters": bad["spec"]["trainable_parameter_inventory"], "buffers": bad["spec"]["buffer_inventory"]}))
        with self.assertRaisesRegex(ValueError, "resume contract drift"):
            self.check_payload(bad, spec)

    def test_legacy_inventory_free_spec_has_no_silent_fallback(self):
        spec = self.spec()
        components = self.components(spec)
        spec["schema"] = "performance_v2_execution_interface_v1"
        for field in ("trainable_parameter_inventory", "buffer_inventory", "model_inventory_sha256"):
            del spec[field]
        with self.assertRaisesRegex(ValueError, "no v1 fallback"):
            ck.validate_components(spec, components, epoch=1, updates=1)
        with self.assertRaisesRegex(ValueError, "actual model"):
            ck.execution_spec(self.contract, self.inputs, model=self.model.state_dict(), optimizer_ledger=self.ledger)

    def test_unmodified_reviewed_hybrid_inventory_passes_without_forward_or_steps(self):
        model = R50Hybrid(objective="SEPARATE-010")
        _, ledger = optimizer_groups(model)
        spec = self.spec(ledger, model)
        self.assertEqual(len(spec["trainable_parameter_inventory"]), len(list(model.parameters())))
        self.assertEqual(len(spec["buffer_inventory"]), len(list(model.buffers())))
        table = {row["name"]: row["optimizer_group"] for row in spec["trainable_parameter_inventory"]}
        self.assertEqual(table["position.position"], "encoder_bias_normalization")
        self.assertEqual(table["transformer.0.attention.in_proj_weight"], "encoder_weights")
        self.assertEqual(table["reader_head.weight"], "head_weights")


if __name__ == "__main__":
    unittest.main()
