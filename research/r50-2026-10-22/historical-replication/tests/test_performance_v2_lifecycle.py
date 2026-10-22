"""Invented locks, logits, tensor states and mocked steps; no training experiment."""
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consendxformer.performance_v2.bindings import FrozenInputs, CLASS_ORDER
from consendxformer.performance_v2.metrics import DiagnosticAccumulator
from consendxformer.performance_v2 import checkpoint as ck
from consendxformer.performance_v2 import lifecycle as life
from consendxformer.performance_v2 import reuse as r
from consendxformer.performance_v2.streams import TrainingStreams, capture_cpu_rng, restore_cpu_rng
from consendxformer.performance_v2.optimization import optimizer_groups
from consendxformer.performance_v2.rng import initialization_scope


def lock_object():
    rows = []
    for split, count in (("train", 17), ("val", 5)):
        for i in range(count):
            name = "invented-%s-%02d" % (split, i)
            rows.append({"patient_id": "patient-" + name, "series_instance_uid": "series-" + name,
                         "nodule_id": name, "split": split, "label": i % 3,
                         "reader_target": [.2, .3, .5], "disagreement": .4, "valid_readers": 3,
                         "image_sha256": r.digest(name.encode()), "supervision_sha256": r.digest((name + "-q").encode())})
    return {"schema": "performance_v2_input_interface_v1", "scope": "synthetic", "class_order": list(CLASS_ORDER),
            "archived_cohort_sha256": r.digest(b'invented cohort'), "transform_sha256": r.digest(b'invented transform'),
            "records": rows}


def inputs_from(obj=None):
    raw = r.canonical(lock_object() if obj is None else obj)
    return FrozenInputs(raw, expected_sha256=r.digest(raw))


class Fixture:
    def __init__(self):
        self.inputs = inputs_from()
        self.contract = {"run_id": "invented-hybrid-s42", "model_id": r.MODEL_ID, "protocol_version": 2,
                         "initialization": "RANDOM", "objective": "SEPARATE-010", "seed": 42, "recipe": "A",
                         "bindings": {name: r.digest(name.encode()) for name in r.BINDINGS}}
        # A real, tiny synthetic module is required to distinguish parameters
        # from buffers and apply the same grouping rules as the reviewed hybrid.
        with initialization_scope(42, "encoder_initialization"):
            self.module = torch.nn.Module()
            self.module.backbone = torch.nn.Linear(2, 2)
            self.module.diagnostic_head = torch.nn.Linear(2, 3)
        self.model = copy.deepcopy(self.module.state_dict())
        _, self.ledger = optimizer_groups(self.module)
        self.spec = ck.execution_spec(self.contract, self.inputs, model=self.module, optimizer_ledger=self.ledger)
        self.streams = TrainingStreams(42)
        self.updates = 0
        self.mock_step = Mock(name="mock_optimizer_step", side_effect=self._step)

    def _step(self):
        # Counts a synthetic event only. It never touches any model parameter.
        self.updates += 1
        return self.updates

    def components(self, epoch):
        groups, state = [], {}
        for i, row in enumerate(self.ledger):
            groups.append({"name": row["group"], "params": [i], "lr": row["lr"] * ck.epoch_multiplier(epoch),
                           "weight_decay": row["weight_decay"], "betas": (.9, .999), "eps": 1e-8,
                           "amsgrad": False, "maximize": False})
            state[i] = {"step": torch.tensor(float(self.updates)),
                        "exp_avg": torch.full(row["shape"], .01 * self.updates),
                        "exp_avg_sq": torch.full(row["shape"], .001 * self.updates)}
        return {"model": copy.deepcopy(self.model), "optimizer": {"param_groups": groups, "state": state},
                "scheduler": ck.scheduler_state(epoch), "rng": capture_cpu_rng(self.streams)}

    def metrics(self, split="val"):
        metrics = DiagnosticAccumulator(self.inputs, split=split)
        ids = self.inputs.ids(split)
        labels = self.inputs.supervision(ids, split=split)["labels"]
        logits = torch.nn.functional.one_hot(labels, 3).float() * 2
        metrics.add(ids, logits, labels)
        return metrics

    def epoch(self, session, epoch):
        plan = self.streams.epoch_plan(self.inputs.ids("train"), epoch=epoch)
        session.begin_epoch(plan)
        for batch in plan["batches"]:
            session.record_batch(batch, observed_optimizer_updates=self.mock_step())
        return session.save_epoch(self.metrics(), self.components(epoch), elapsed_seconds=float(epoch))


class InputMetricTests(unittest.TestCase):
    def test_lock_hash_verified_before_parsing(self):
        with patch("consendxformer.performance_v2.bindings._json", side_effect=AssertionError("parsed")):
            with self.assertRaisesRegex(ValueError, "digest"):
                FrozenInputs(b"malformed", expected_sha256="0" * 64)

    def test_no_test_duplicate_patient_overlap_or_unknown_record_fields(self):
        mutations = [lambda o: o["records"][0].update(split="test"),
                     lambda o: o["records"].append(copy.deepcopy(o["records"][0])),
                     lambda o: o["records"][-1].update(patient_id=o["records"][0]["patient_id"]),
                     lambda o: o["records"][0].update(unreviewed=True),
                     lambda o: o.update(class_order=list(reversed(CLASS_ORDER))),
                     lambda o: o["records"][0].update(reader_target=[.1, .1, .1]),
                     lambda o: o["records"][0].update(label=True),
                     lambda o: o["records"][0].update(disagreement=float("inf"))]
        for mutate in mutations:
            obj = lock_object()
            mutate(obj)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                inputs_from(obj)

    def test_synthetic_lock_cannot_masquerade_as_frozen_production(self):
        obj = lock_object()
        obj["scope"] = "production"
        with self.assertRaisesRegex(ValueError, "production"):
            inputs_from(obj)

    def test_source_bytes_and_supervision_are_bound_without_readers(self):
        inputs = inputs_from()
        ids = inputs.ids("train")[:2]
        supervision = inputs.supervision(ids, split="train")
        hashes = {i: inputs.verify_image_bytes(i, i.encode()) for i in ids}
        inputs.validate_batch(ids, split="train", image_hashes=hashes, **supervision)
        with self.assertRaises(ValueError):
            inputs.verify_image_bytes(ids[0], b"wrong")
        for key in supervision:
            altered = copy.deepcopy(supervision)
            altered[key].flatten()[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                inputs.validate_batch(ids, split="train", image_hashes=hashes, **altered)
        with self.assertRaises(ValueError):
            inputs.validate_batch(ids, split="val", image_hashes=hashes, **supervision)

    def test_fixed_order_ties_absent_classes_and_macro_include_all_three(self):
        obj = lock_object()
        obj["records"] = [x for x in obj["records"] if x["split"] == "train" or x["label"] == 0]
        inputs = inputs_from(obj)
        m = DiagnosticAccumulator(inputs, split="val")
        ids = inputs.ids("val")
        m.add(ids, torch.zeros(len(ids), 3), torch.zeros(len(ids), dtype=torch.int64))
        report = m.summary()
        self.assertEqual(report["confusion_matrix"], [[2, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(report["macro_f1"], 1 / 3)
        self.assertEqual(report["accuracy"], report["micro_f1"])
        self.assertEqual(report["accuracy"], 1)
        self.assertAlmostEqual(report["diagnostic_ce"], math.log(3), places=6)

    def test_confusion_per_class_and_nodule_weighted_ce_ignore_batch_partition(self):
        inputs = inputs_from()
        ids = inputs.ids("val")
        labels = inputs.supervision(ids, split="val")["labels"]  # 0,1,2,0,1
        logits = torch.tensor([[2., 0, 0], [3., 0, 0], [0, 0, 2.], [0, 1., 0], [0, 4., 0]])
        full, partial = (DiagnosticAccumulator(inputs, split="val") for _ in range(2))
        full.add(ids, logits, labels)
        for start, end in ((0, 2), (2, 4), (4, 5)):
            partial.add(ids[start:end], logits[start:end], labels[start:end])
        self.assertEqual(full.summary(), partial.summary())
        result = full.summary()
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["accuracy"], .6)
        per = -logits.log_softmax(1).gather(1, labels[:, None]).flatten()
        self.assertAlmostEqual(result["diagnostic_ce"], sum(per.tolist()) / 5)
        wrong = (per[:2].mean().item() + per[2:4].mean().item() + per[4:].mean().item()) / 3
        self.assertNotAlmostEqual(result["diagnostic_ce"], wrong, places=4)

    def test_partial_duplicate_mismatched_and_train_records_cannot_select(self):
        inputs = inputs_from()
        ids = inputs.ids("val")
        labels = inputs.supervision(ids[:1], split="val")["labels"]
        m = DiagnosticAccumulator(inputs, split="val")
        with self.assertRaises(ValueError):
            m.add(ids[:1], torch.zeros(1, 3), labels + 1)
        m.add(ids[:1], torch.zeros(1, 3), labels)
        with self.assertRaises(ValueError):
            m.add(ids[:1], torch.zeros(1, 3), labels)
        with self.assertRaises(ValueError):
            m.selection_record(1)
        with self.assertRaises(ValueError):
            Fixture().metrics("train").selection_record(1)
        with self.assertRaises(ValueError):
            DiagnosticAccumulator(inputs, split="test")


class StateTests(unittest.TestCase):
    def test_invalid_global_rng_rejected_without_mutating_live_streams(self):
        streams = TrainingStreams(42)
        original = capture_cpu_rng(streams)
        for mutate in (lambda s: s.update(torch_cpu=torch.zeros(4)),
                       lambda s: s.update(python=(3, (), None)),
                       lambda s: s.update(numpy=("MT19937", [0] * 624, 0, 0, float("nan"))),
                       lambda s: s.update(cuda="silently_changed_to_gpu")):
            candidate = copy.deepcopy(original)
            mutate(candidate)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, RuntimeError)):
                restore_cpu_rng(candidate, streams)
            self.assertEqual(ck.state_sha256(original), ck.state_sha256(capture_cpu_rng(streams)))

    def test_typed_tensor_codec_roundtrip_has_no_pickle_or_model_loader(self):
        state = {"x": torch.tensor([[1., 2.]]), "counter": torch.tensor(7),
                 "tuple": (None, True, 4.5), "moments": {0: torch.zeros(2)}, "empty": torch.empty(0)}
        raw = ck.encode_state(state)
        with patch.object(torch, "load", side_effect=AssertionError("loader")):
            decoded = ck.decode_state(raw, expected_sha256=r.digest(raw))
        self.assertEqual(raw, ck.encode_state(decoded))

    def test_codec_verifies_digest_before_parse_rejects_truncation_and_bad_descriptors(self):
        raw = ck.encode_state({"value": torch.ones(2)})
        with patch.object(ck.json, "loads", side_effect=AssertionError("parsed")):
            with self.assertRaisesRegex(ValueError, "digest"):
                ck.decode_state(raw[:-1], expected_sha256=r.digest(raw))
        for value in (torch.tensor(float("nan")), object(), {True: "bool key"}):
            with self.assertRaises(ValueError):
                ck.encode_state(value)
        for tree in (["tensor", "float32", [1000000000], ""], ["dict", [[["scalar", "x"], ["scalar", 1]],
                                                                            [["scalar", "x"], ["scalar", 2]]]]):
            bad = r.canonical(tree)
            with self.assertRaises(ValueError):
                ck.decode_state(bad, expected_sha256=r.digest(bad))

    def test_rng_resume_reproduces_all_next_draws_without_reseeding(self):
        streams = TrainingStreams(42)
        original = capture_cpu_rng(streams)
        try:
            streams.epoch_plan(["a", "b", "c"], epoch=1)
            saved = capture_cpu_rng(streams)
            expected = (torch.rand(4), random.random(), np.random.random(), streams.epoch_plan(["a", "b", "c"], epoch=2))
            restore_cpu_rng(saved, streams)
            self.assertTrue(torch.equal(expected[0], torch.rand(4)))
            self.assertEqual(expected[1], random.random())
            self.assertEqual(expected[2], np.random.random())
            self.assertEqual(expected[3], streams.epoch_plan(["a", "b", "c"], epoch=2))
            changed = copy.deepcopy(saved)
            changed["data_streams"]["stream_seeds"]["augmentation"] += 1
            before = ck.state_sha256(capture_cpu_rng(streams))
            with self.assertRaises(ValueError):
                restore_cpu_rng(changed, streams)
            self.assertEqual(before, ck.state_sha256(capture_cpu_rng(streams)))
        finally:
            restore_cpu_rng(original, streams)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pv2-synthetic-lifecycle-")
        self.root = Path(self.temp.name)
        self.fixture = Fixture()
        self.budget = life.BudgetLedger(limit=3, plan_sha256=r.digest(b"invented run plan"))
        self.session = self.new_session("first")

    def tearDown(self):
        self.temp.cleanup()

    def new_session(self, name, **kwargs):
        return life.RunLifecycle(self.root / name, spec=self.fixture.spec, inputs=self.fixture.inputs,
                                 budget=self.budget, decision_sha256=r.digest(b"invented decision, not authorization"), **kwargs)

    def retain(self, label):
        root = os.environ.get("PV2_SYNTHETIC_EVIDENCE_DIRECTORY")
        if root:
            target = Path(root) / label
            target.mkdir(parents=True, exist_ok=False)
            for file in self.session.store.path.iterdir():
                shutil.copyfile(file, target / file.name)
            (target / "test-observation.json").write_bytes(r.canonical({"synthetic": True, "status": self.session.status,
                "budget": self.budget.snapshot(), "completion_exists": (self.session.store.path / "completion.json").exists(),
                "mocked_optimizer_updates": self.fixture.mock_step.call_count, "pilot_execution_authorized": False}))

    def interrupted(self):
        self.fixture.epoch(self.session, 1)
        terminal = self.session.terminate("interrupted", reason="invented cancellation at boundary")
        raw = r.canonical(terminal)
        states = {str(e): (self.session.store.path / name).read_bytes() for e, (name, _) in self.session.checkpoints.items()}
        return dict(requested_spec=self.fixture.spec, terminal_bytes=raw, terminal_sha256=r.digest(raw),
                    checkpoint_bytes=states, checkpoint_hashes=terminal["checkpoint_hashes"])

    def test_batch_plan_no_drop_tail_update_accounting_and_partial_cannot_complete(self):
        plan = self.fixture.streams.epoch_plan(self.fixture.inputs.ids("train"), epoch=1)
        self.assertEqual([len(b) for b in plan["batches"]], [16, 1])
        self.session.begin_epoch(plan)
        with self.assertRaises(ValueError):
            self.session.record_batch(plan["batches"][1], observed_optimizer_updates=1)
        self.session.record_batch(plan["batches"][0], observed_optimizer_updates=self.fixture.mock_step())
        with self.assertRaises(ValueError):
            self.session.save_epoch(self.fixture.metrics(), self.fixture.components(1), elapsed_seconds=1)
        with self.assertRaises(ValueError):
            self.session.finalize()
        self.assertFalse((self.session.store.path / "completion.json").exists())

    def test_guard_marks_normal_early_exit_interrupted(self):
        with self.session.guard():
            self.fixture.epoch(self.session, 1)
        self.assertEqual(self.session.status, "interrupted")
        self.assertFalse((self.session.store.path / "completion.json").exists())
        self.retain("guard-early-exit")

    def test_guard_marks_uncaught_numeric_failure_and_retains_evidence(self):
        with self.assertRaises(ValueError):
            with self.session.guard():
                raise ValueError("invented nonfinite validation output")
        self.assertEqual(self.session.status, "failed")
        self.assertTrue((self.session.store.path / "terminal.json").exists())
        self.assertFalse((self.session.store.path / "completion.json").exists())
        self.retain("guard-numeric-failure")

    def test_admission_cannot_skip_review_and_mutate_a_verified_payload(self):
        with self.assertRaises(ValueError):
            life.ResumeAdmission(self.fixture.spec, {}, {}, {})
        args = self.interrupted()
        target = {}
        admission = life.review_resume(**args, restore_components=lambda state: target.update(state), capture_components=lambda: target)
        admission.payload["progress"]["updates"] += 1
        with self.assertRaises(ValueError):
            self.new_session("mutated", resume=admission)
        self.assertEqual(len(self.budget.snapshot()["entries"]), 1)

    def test_interruption_after_receipt_commit_does_not_create_failed_complete_contradiction(self):
        for epoch in range(1, 21):
            self.fixture.epoch(self.session, epoch)
        original = life.os.link
        def interrupt_after_link(source, target):
            original(source, target)
            if str(target).endswith("completion.json"):
                raise KeyboardInterrupt("invented interruption after atomic commit")
        with patch.object(life.os, "link", side_effect=interrupt_after_link):
            receipt, manifest = self.session.finalize()
        self.assertEqual(receipt["status"], "synthetic_complete")
        self.assertEqual(self.session.status, "synthetic_complete")
        self.assertEqual(self.session.post_commit_interruption, "KeyboardInterrupt")
        self.assertFalse((self.session.store.path / "terminal.json").exists())
        self.retain("post-commit-interruption")

    def test_changed_augmentation_plan_rejected_before_epoch(self):
        plan = self.fixture.streams.epoch_plan(self.fixture.inputs.ids("train"), epoch=1)
        plan["augmentation"][0]["clockwise_angle"] += 1
        with self.assertRaisesRegex(ValueError, "RNG"):
            self.session.begin_epoch(plan)

    def test_complete_mock_lifecycle_selects_earliest_validation_tie_and_is_never_reusable(self):
        for epoch in range(1, 21):
            selection = self.fixture.epoch(self.session, epoch)
        self.assertTrue(selection["stopped"])
        receipt, manifest = self.session.finalize()
        self.assertEqual(receipt["status"], "synthetic_complete")
        self.assertFalse(receipt["pilot_execution_authorized"])
        self.assertEqual(manifest["best"], "epoch-001.state.json")
        self.assertEqual(manifest["last"], "epoch-020.state.json")
        self.assertEqual(self.fixture.mock_step.call_count, 40)
        for name, sha in manifest["files"].items():
            self.assertEqual(r.digest((self.session.store.path / name).read_bytes()), sha)
        raw = r.canonical(receipt)
        args = dict(completion_bytes=raw, completion_sha256=r.digest(raw), manifest_bytes=r.canonical(manifest),
                    verified_output_hashes=manifest["files"])
        with self.assertRaises(ValueError):
            life.require_completed_reuse(self.fixture.spec, **args)
        with self.assertRaises(ValueError):
            r.require_exact_reuse(self.fixture.contract, self.fixture.contract, **args)
        with self.assertRaises(ValueError):
            self.session.finalize()
        self.retain("synthetic-completion")

    def test_valid_boundary_resume_recaptures_identical_state_and_consumes_new_attempt(self):
        args = self.interrupted()
        target = {}
        admission = life.review_resume(**args, restore_components=lambda state: target.update(state), capture_components=lambda: target)
        resumed = self.new_session("resumed", resume=admission)
        self.assertEqual(len(self.budget.snapshot()["entries"]), 2)
        self.assertEqual(resumed.progress["updates"], 2)
        self.fixture.epoch(resumed, 2)
        self.assertEqual(resumed.history[0], self.session.history[0])
        self.assertEqual(set(resumed.checkpoints), {1, 2})
        self.assertEqual(ck.selection_state(resumed.history)["best_epoch"], 1)
        self.assertFalse((self.session.store.path / "completion.json").exists())
        self.retain("interrupted-boundary")

    def test_every_resume_contract_binding_rejects_drift_before_restoration(self):
        args = self.interrupted()
        mutations = []
        for key in r.BINDINGS:
            mutations.append(lambda s, key=key: s["contract"]["bindings"].update({key: "f" * 64}))
        mutations.extend([lambda s: s["contract"].update(objective="HARD"),
                          lambda s: s["contract"].update(initialization="IMAGENET"),
                          lambda s: s["contract"].update(seed=43),
                          lambda s: s.update(input_lock_sha256="a" * 64),
                          lambda s: s.update(model_layout={}), lambda s: s.update(scheduler="changed"),
                          lambda s: s["selection"].update(primary="accuracy"),
                          lambda s: s["optimizer_ledger"][0].update(lr=.5)])
        for mutate in mutations:
            spec = copy.deepcopy(self.fixture.spec)
            mutate(spec)
            restore = Mock()
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                life.review_resume(**dict(args, requested_spec=spec), restore_components=restore, capture_components=Mock())
            restore.assert_not_called()

    def test_resume_recapture_rejects_lost_model_optimizer_scheduler_or_rng(self):
        args = self.interrupted()
        for key in ("model", "optimizer", "scheduler", "rng"):
            target = {}
            def restore(state):
                target.update(state)
                target.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recapture"):
                life.review_resume(**args, restore_components=restore, capture_components=lambda: target)
        self.assertEqual(len(self.budget.snapshot()["entries"]), 1)

    def test_resume_rejects_corruption_missing_state_and_stale_hash_before_callbacks(self):
        args = self.interrupted()
        for changes in ({"terminal_sha256": "0" * 64}, {"checkpoint_bytes": {}},
                        {"checkpoint_bytes": {"1": args["checkpoint_bytes"]["1"][:-1]}}):
            restore = Mock()
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                life.review_resume(**dict(args, **changes), restore_components=restore, capture_components=Mock())
            restore.assert_not_called()

    def test_resume_rejects_modified_patience_even_if_payload_digest_is_supplied(self):
        args = self.interrupted()
        original = ck.decode_state(args["checkpoint_bytes"]["1"], expected_sha256=args["checkpoint_hashes"]["1"])
        for mutate in (lambda p: p["selection"].update(best_epoch=3),
                       lambda p: p["selection"]["rules"].update(tolerance=.2),
                       lambda p: p["components"]["scheduler"].update(next_epoch=9),
                       lambda p: p["components"]["optimizer"]["param_groups"][0].update(lr=.8),
                       lambda p: p["components"]["rng"]["data_streams"].update(seed=43)):
            payload = copy.deepcopy(original)
            mutate(payload)
            raw = ck.encode_state(payload)
            with self.assertRaises(ValueError):
                ck.validate_checkpoint(raw, expected_sha256=r.digest(raw), requested_spec=self.fixture.spec)

    def test_mid_epoch_interruption_rejects_rollback_to_old_epoch_and_cannot_reuse(self):
        self.fixture.epoch(self.session, 1)
        plan = self.fixture.streams.epoch_plan(self.fixture.inputs.ids("train"), epoch=2)
        self.session.begin_epoch(plan)
        self.session.record_batch(plan["batches"][0], observed_optimizer_updates=self.fixture.mock_step())
        terminal = self.session.terminate("interrupted", reason="invented mid-epoch cancellation")
        raw = r.canonical(terminal)
        restore = Mock()
        with self.assertRaises(ValueError):
            life.review_resume(requested_spec=self.fixture.spec, terminal_bytes=raw, terminal_sha256=r.digest(raw),
                               checkpoint_bytes={}, checkpoint_hashes={}, restore_components=restore, capture_components=Mock())
        restore.assert_not_called()
        self.assertFalse((self.session.store.path / "completion.json").exists())
        with self.assertRaises(ValueError):
            self.session.finalize()
        self.retain("interrupted-mid-epoch")

    def test_failed_run_stays_terminal_and_has_no_complete_receipt(self):
        self.fixture.epoch(self.session, 1)
        terminal = self.session.terminate("failed", reason="invented nonfinite loss")
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(self.budget.snapshot()["entries"][0]["status"], "failed")
        with self.assertRaises(ValueError):
            self.session.finalize()
        with self.assertRaises(ValueError):
            self.session.begin_epoch({})
        self.assertFalse((self.session.store.path / "completion.json").exists())
        raw = r.canonical(terminal)
        with self.assertRaises(ValueError):
            r.require_exact_reuse(self.fixture.contract, self.fixture.contract, completion_bytes=raw,
                                 completion_sha256=r.digest(raw), manifest_bytes=b"{}", verified_output_hashes={})
        self.retain("failed-run")

    def test_checkpoint_publication_failure_retains_pending_bytes_and_charges_budget(self):
        plan = self.fixture.streams.epoch_plan(self.fixture.inputs.ids("train"), epoch=1)
        self.session.begin_epoch(plan)
        for batch in plan["batches"]:
            self.session.record_batch(batch, observed_optimizer_updates=self.fixture.mock_step())
        with patch.object(life.os, "link", side_effect=OSError("invented full filesystem")):
            with self.assertRaises(OSError):
                self.session.save_epoch(self.fixture.metrics(), self.fixture.components(1), elapsed_seconds=1)
        self.assertEqual(self.session.status, "failed")
        self.assertTrue(list(self.session.store.path.glob("*.pending")))
        self.assertFalse((self.session.store.path / "completion.json").exists())
        self.assertEqual(self.session.failure_retention_error, "OSError")
        self.retain("checkpoint-write-failure")

    def test_cancellation_during_checkpoint_retains_failure_record_without_completion(self):
        original = life.os.link
        def cancel(source, target):
            if str(target).endswith(".state.json"):
                raise KeyboardInterrupt("invented cancellation")
            return original(source, target)
        with patch.object(life.os, "link", side_effect=cancel), self.assertRaises(KeyboardInterrupt):
            self.fixture.epoch(self.session, 1)
        self.assertEqual(self.session.status, "interrupted")
        self.assertTrue((self.session.store.path / "terminal.json").exists())
        self.assertFalse((self.session.store.path / "completion.json").exists())
        self.retain("checkpoint-cancellation")

    def test_unknown_or_corrupt_artifact_blocks_completion_and_retains_failure(self):
        for epoch in range(1, 21):
            self.fixture.epoch(self.session, epoch)
        (self.session.store.path / "epoch-001.state.json").write_bytes(b"invented corruption")
        with self.assertRaises(ValueError):
            self.session.finalize()
        self.assertEqual(self.session.status, "failed")
        self.assertFalse((self.session.store.path / "completion.json").exists())
        self.retain("finalization-corruption")

    def test_missing_reader_rng_or_optimizer_moments_cannot_be_checkpointed(self):
        plan = self.fixture.streams.epoch_plan(self.fixture.inputs.ids("train"), epoch=1)
        self.session.begin_epoch(plan)
        for batch in plan["batches"]:
            self.session.record_batch(batch, observed_optimizer_updates=self.fixture.mock_step())
        for mutate in (lambda c: c.pop("rng"), lambda c: c["optimizer"]["state"].pop(0),
                       lambda c: c["model"].pop("diagnostic_head.bias"),
                       lambda c: c["rng"]["data_streams"].update(augmentation=TrainingStreams(43).augmentation.getstate())):
            components = self.fixture.components(1)
            mutate(components)
            with self.assertRaises(ValueError):
                self.session.save_epoch(self.fixture.metrics(), components, elapsed_seconds=1)
        self.assertFalse(self.session.checkpoints)


class BudgetArtifactTests(unittest.TestCase):
    def test_all_starts_failures_restarts_consume_slots_no_refunds_or_duplicate_finish(self):
        ledger = life.BudgetLedger(limit=2, plan_sha256="a" * 64)
        first = ledger.reserve("one", kind="start", decision_sha256="b" * 64)
        ledger.finish(first, "failed")
        second = ledger.reserve("one", kind="resume", decision_sha256="c" * 64)
        with self.assertRaises(ValueError):
            ledger.reserve("replacement", kind="start", decision_sha256="d" * 64)
        ledger.finish(second, "interrupted")
        with self.assertRaises(ValueError):
            ledger.finish(second, "complete")
        raw = r.canonical(ledger.snapshot())
        restored = life.BudgetLedger.restore(raw, expected_sha256=r.digest(raw))
        self.assertEqual(restored.snapshot(), ledger.snapshot())
        with self.assertRaises(ValueError):
            restored.reserve("third", kind="start", decision_sha256="d" * 64)
        with self.assertRaises(ValueError):
            life.BudgetLedger.restore(raw, expected_sha256="0" * 64)

    def test_exclusive_directory_no_overwrite_path_escape_or_silent_unknown_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invented"
            store = life.ArtifactStore(path)
            with self.assertRaises(FileExistsError):
                life.ArtifactStore(path)
            store.put("one.json", b"original")
            for name in ("one.json", "../escape", "nested/file", "C:\\escape", "x.pending"):
                with self.assertRaises(ValueError):
                    store.put(name, b"changed")
            self.assertEqual((path / "one.json").read_bytes(), b"original")
            self.assertEqual(store.verify(), {"one.json": r.digest(b"original")})
            (path / "unknown.json").write_bytes(b"unclassified")
            with self.assertRaises(ValueError):
                store.verify()

    def test_failed_directory_creation_consumes_attempt(self):
        fixture = Fixture()
        budget = life.BudgetLedger(limit=1, plan_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as existing:
            with self.assertRaises(FileExistsError):
                life.RunLifecycle(existing, spec=fixture.spec, inputs=fixture.inputs, budget=budget, decision_sha256="b" * 64)
        self.assertEqual(budget.snapshot()["entries"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
