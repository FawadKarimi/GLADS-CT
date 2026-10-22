"""Temporary private roots; invented evidence and mocked step counters only."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_performance_v2_lifecycle import Fixture
from consendxformer.performance_v2.durable_budget import PersistentBudget, AdmissionBlocked, ReconciliationRequired
from consendxformer.performance_v2.durable_lifecycle import DurableRun
from consendxformer.performance_v2.lifecycle import review_resume
from consendxformer.performance_v2.reuse import canonical, digest

TX = ("before_begin", "after_begin", "after_write", "before_commit", "after_commit", "after_sync")
MIRROR = ("before_write", "after_write", "after_file_sync", "after_link", "after_directory_sync", "after_cleanup")


class Crash(BaseException):
    """Synthetic boundary interruption, not an actual killed process."""


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pv2-durable-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fx = Fixture()
        self.other = copy.deepcopy(self.fx.spec)
        self.other["contract"]["run_id"] = "invented-other"
        self.plan = canonical({"schema": "performance_v2_durable_plan_v1", "scope": "synthetic", "limit": 3,
                               "runs": {s["contract"]["run_id"]: digest(canonical(s)) for s in (self.fx.spec, self.other)},
                               "pilot_execution_authorized": False})
        self.barriers = []

    def budget(self, name="budget"):
        return PersistentBudget.create(self.root / name, plan_bytes=self.plan, expected_plan_sha256=digest(self.plan),
                                       synthetic_directory_sync=lambda p: self.barriers.append(str(p)))

    def reopen(self, budget):
        return PersistentBudget(budget.root, expected_plan_sha256=digest(self.plan),
                                synthetic_directory_sync=lambda p: self.barriers.append(str(p)))

    def reserve(self, budget, spec=None, decision=b"start"):
        ticket = budget.admit(self.fx.spec if spec is None else spec, decision_sha256=digest(decision))
        self.addCleanup(ticket.close)
        return ticket

    def session(self, budget):
        session = DurableRun(spec=self.fx.spec, inputs=self.fx.inputs, budget=budget, decision_sha256=digest(b"start"))
        self.addCleanup(session.ticket.close)
        return session

    def stopped(self, budget):
        session = self.session(budget)
        for epoch in range(1, 21):
            self.fx.epoch(session, epoch)
        return session

    def inject(self, budget, label, error=Crash):
        def fault(observed):
            if observed == label:
                raise error(label)
        budget._fault = fault

    def reconcile(self, budget, action, number=1):
        snapshot = budget.inspect()
        return budget.reconcile(number, action=action, expected_snapshot_sha256=digest(canonical(snapshot)),
                                decision_sha256=digest((action + str(number)).encode()))

    def retain(self, label, budget):
        destination = os.environ.get("PV2_SYNTHETIC_EVIDENCE_DIRECTORY")
        if destination:
            path = Path(destination) / ("durable-" + label)
            shutil.copytree(budget.root, path)
            (path / "observed-snapshot.json").write_bytes(canonical(budget.inspect()))

    def test_competing_admissions_one_winner_and_restart_does_not_steal(self):
        budget = self.budget()
        barrier = threading.Barrier(2)
        def compete(i):
            candidate = self.reopen(budget)
            barrier.wait(timeout=10)
            try:
                return candidate.admit(self.fx.spec, decision_sha256=digest(str(i).encode()))
            except AdmissionBlocked:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(compete, (1, 2)))
        tickets = [t for t in results if t is not None]
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        try:
            self.assertEqual(budget.inspect()["consumed"], 1)
            with self.assertRaises(AdmissionBlocked):
                self.reserve(self.reopen(budget), self.other)
            with self.assertRaises(AdmissionBlocked):
                self.reconcile(self.reopen(budget), "interrupt_reserved")
        finally:
            ticket.close()
        self.assertEqual(self.reopen(budget).inspect()["attempts"][0]["state"], "reserved")

    def test_stale_lock_age_is_not_permission_and_explicit_recovery_consumes_slot(self):
        budget = self.budget()
        ticket = self.reserve(budget)
        ticket.close()
        os.utime(budget.root / "lease.lock", (1, 1))
        with self.assertRaises(ReconciliationRequired):
            self.reserve(self.reopen(budget), self.other)
        terminal = self.reconcile(self.reopen(budget), "interrupt_reserved")
        self.assertFalse(terminal["epoch_boundary"])
        self.assertIsNone(terminal["progress"])
        self.assertEqual(budget.inspect()["consumed"], 1)
        with self.assertRaises(AdmissionBlocked):
            self.reserve(budget, decision=b"no automatic retry")
        second = self.reserve(budget, self.other, b"separate planned run")
        self.assertEqual(second.number, 2)
        self.assertEqual(budget.inspect()["consumed"], 2)

    def test_admission_every_transaction_boundary(self):
        for boundary in TX:
            with self.subTest(boundary=boundary):
                budget = self.budget(boundary)
                self.inject(budget, "admission." + boundary)
                with self.assertRaises(Crash):
                    self.reserve(budget)
                fresh = self.reopen(budget)
                committed = boundary in ("after_commit", "after_sync")
                self.assertEqual(fresh.inspect()["consumed"], int(committed))
                if committed:
                    with self.assertRaises(ReconciliationRequired):
                        self.reserve(fresh, self.other)
                    self.reconcile(fresh, "interrupt_reserved")
                    self.assertEqual(fresh.inspect()["attempts"][0]["state"], "interrupted")
                else:
                    self.assertEqual(fresh.inspect()["issues"], [])
                self.retain("admission-" + boundary, fresh)

    def test_terminal_every_transaction_boundary(self):
        for boundary in TX:
            with self.subTest(boundary=boundary):
                self.fx = Fixture()
                budget = self.budget(boundary)
                session = self.session(budget)
                self.inject(budget, "terminal." + boundary)
                with self.assertRaises(Crash):
                    session.terminate("failed", reason="invented failure")
                fresh = self.reopen(budget)
                committed = boundary in ("after_commit", "after_sync")
                self.assertEqual(fresh.inspect()["attempts"][0]["state"], "failed" if committed else "reserved")
                self.reconcile(fresh, "export_committed" if committed else "interrupt_reserved")
                self.assertEqual(fresh.inspect()["consumed"], 1)
                with self.assertRaises(AdmissionBlocked):
                    fresh.committed_receipt(1)
                self.retain("terminal-" + boundary, fresh)

    def test_completion_every_transaction_boundary(self):
        for boundary in TX:
            with self.subTest(boundary=boundary):
                self.fx = Fixture()
                budget = self.budget(boundary)
                session = self.stopped(budget)
                self.inject(budget, "completion." + boundary)
                with self.assertRaises(Crash):
                    session.finalize()
                fresh = self.reopen(budget)
                committed = boundary in ("after_commit", "after_sync")
                self.assertEqual(fresh.inspect()["attempts"][0]["state"], "synthetic_complete" if committed else "reserved")
                self.assertFalse((session.store.path / "completion.json").exists())
                with self.assertRaises((ValueError, AdmissionBlocked)):
                    session.finalize()
                with self.assertRaises(ReconciliationRequired):
                    fresh.committed_receipt(1)
                self.reconcile(fresh, "export_committed" if committed else "interrupt_reserved")
                if committed:
                    self.assertEqual(json.loads(fresh.committed_receipt(1))["status"], "synthetic_complete")
                else:
                    with self.assertRaises(AdmissionBlocked):
                        fresh.committed_receipt(1)
                self.retain("completion-" + boundary, fresh)

    def test_mirror_and_export_boundaries_never_relabel_committed_completion(self):
        for label in ["mirror." + x for x in MIRROR] + ["export." + x for x in TX]:
            with self.subTest(label=label):
                self.fx = Fixture()
                budget = self.budget(label)
                session = self.stopped(budget)
                self.inject(budget, label)
                with self.assertRaises(Crash):
                    session.finalize()
                fresh = self.reopen(budget)
                report = fresh.inspect()
                self.assertEqual(report["attempts"][0]["state"], "synthetic_complete")
                self.assertEqual(report["consumed"], 1)
                self.retain(label + "-before-reconciliation", fresh)
                if report["issues"]:
                    with self.assertRaises(ReconciliationRequired):
                        self.reserve(fresh, self.other)
                    self.reconcile(fresh, "export_committed")
                receipt = fresh.committed_receipt(1)
                self.assertEqual(receipt, fresh.receipt_path(1).read_bytes())
                self.assertEqual(fresh.inspect()["issues"], [])
                self.retain(label + "-reconciled", fresh)

    def test_artifact_write_boundaries_retain_failed_consumed_attempt(self):
        for boundary in ("before_write", "after_write", "after_sync"):
            with self.subTest(boundary=boundary):
                self.fx = Fixture()
                budget = self.budget(boundary)
                session = self.session(budget)
                self.inject(budget, "artifact." + boundary, OSError)
                with self.assertRaises(OSError):
                    self.fx.epoch(session, 1)
                fresh = self.reopen(budget)
                self.assertEqual(fresh.inspect()["attempts"][0]["state"], "failed")
                self.assertEqual(fresh.inspect()["consumed"], 1)
                with self.assertRaises(AdmissionBlocked):
                    fresh.committed_receipt(1)

    def test_disk_fsync_failure_retains_partial_mirror_and_committed_failure(self):
        budget = self.budget()
        session = self.session(budget)
        with patch("consendxformer.performance_v2.durable_budget.os.fsync", side_effect=OSError("invented disk full")):
            with self.assertRaises(OSError):
                session.terminate("failed", reason="synthetic failure")
        fresh = self.reopen(budget)
        report = fresh.inspect()
        self.assertEqual(report["attempts"][0]["state"], "failed")
        self.assertTrue(report["pending_receipt_hashes"])
        pending_hashes = set(report["pending_receipt_hashes"].values())
        self.retain("disk-fsync-before-reconciliation", fresh)
        self.reconcile(fresh, "export_committed")
        self.assertEqual(set(fresh.inspect()["retained_evidence_hashes"].values()), pending_hashes)
        self.assertEqual(fresh.inspect()["consumed"], 1)

    def test_sql_write_failure_rolls_back_without_refund_of_prior_reservation(self):
        budget = self.budget()
        session = self.session(budget)
        self.inject(budget, "terminal.after_write", sqlite3.OperationalError)
        with self.assertRaises(sqlite3.OperationalError):
            session.terminate("failed", reason="invented SQL write error")
        fresh = self.reopen(budget)
        self.assertEqual(fresh.inspect()["attempts"][0]["state"], "reserved")
        self.assertEqual(fresh.inspect()["consumed"], 1)
        self.reconcile(fresh, "interrupt_reserved")

    def test_partial_and_direct_termination_cannot_forge_completion(self):
        budget = self.budget()
        session = self.session(budget)
        with self.assertRaisesRegex(ValueError, "partial"):
            session.finalize()
        with self.assertRaises(ValueError):
            session.ticket.terminate({"status": "synthetic_complete"})
        self.assertEqual(budget.inspect()["attempts"][0]["state"], "reserved")
        session.terminate("interrupted", reason="no experiment was run")

    def test_completion_artifact_drift_blocks_reuse_and_further_admission(self):
        budget = self.budget()
        session = self.stopped(budget)
        receipt, _ = session.finalize()
        self.assertEqual(json.loads(budget.committed_receipt(1)), receipt)
        (session.store.path / "execution-spec.json").write_bytes(b"invented drift")
        with self.assertRaises(ReconciliationRequired):
            budget.committed_receipt(1)
        with self.assertRaises(ReconciliationRequired):
            self.reserve(budget, self.other)
        with self.assertRaises(ReconciliationRequired):
            self.reconcile(budget, "export_committed")

    def test_conflicting_mirror_not_overwritten_and_no_completion_downgrade(self):
        budget = self.budget()
        session = self.stopped(budget)
        session.finalize()
        budget.receipt_path(1).write_bytes(b"conflicting invented receipt")
        with self.assertRaises(ReconciliationRequired):
            self.reconcile(budget, "export_committed")
        self.assertEqual(budget.receipt_path(1).read_bytes(), b"conflicting invented receipt")
        with self.assertRaises(ReconciliationRequired):
            self.reconcile(budget, "interrupt_reserved")

    def test_stale_snapshot_and_orphan_receipt_block_reconciliation(self):
        budget = self.budget()
        self.reserve(budget).close()
        snapshot = digest(canonical(budget.inspect()))
        budget.run_path(1).mkdir()
        (budget.run_path(1) / "invented.json").write_bytes(b"new evidence")
        with self.assertRaises(ReconciliationRequired):
            budget.reconcile(1, action="interrupt_reserved", expected_snapshot_sha256=snapshot,
                             decision_sha256=digest(b"review"))
        budget.receipt_path(1).write_bytes(b"unmanaged complete")
        with self.assertRaises(ReconciliationRequired):
            self.reconcile(budget, "interrupt_reserved")

    def test_plan_drift_expansion_and_configuration_change_rejected(self):
        budget = self.budget()
        changed = copy.deepcopy(self.fx.spec)
        changed["contract"]["seed"] = 99
        with self.assertRaises((ValueError, AdmissionBlocked)):
            self.reserve(budget, changed)
        plan = json.loads(self.plan)
        plan["limit"] = 4
        (budget.root / "plan.json").write_bytes(canonical(plan))
        with self.assertRaises(ValueError):
            self.reopen(budget)
        with self.assertRaises(ReconciliationRequired):
            self.reserve(budget)

    def test_budget_row_audit_contradiction_fails_closed(self):
        budget = self.budget()
        session = self.session(budget)
        session.terminate("failed", reason="invented failure")
        connection = sqlite3.connect(budget.root / "budget.sqlite3")
        try:
            connection.execute("UPDATE attempts SET exported=0")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ReconciliationRequired, "audit"):
            budget.inspect()

    def test_unexpected_journal_mode_rejected_without_conversion(self):
        budget = self.budget()
        connection = sqlite3.connect(budget.root / "budget.sqlite3")
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
        finally:
            connection.close()
        with self.assertRaisesRegex(ReconciliationRequired, "journal mode"):
            budget.inspect()
        with self.assertRaisesRegex(ReconciliationRequired, "journal mode"):
            self.reserve(budget)
        connection = sqlite3.connect(budget.root / "budget.sqlite3")
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        finally:
            connection.close()

    def test_restart_resume_uses_latest_boundary_and_new_budget_slot(self):
        budget = self.budget()
        session = self.session(budget)
        self.fx.epoch(session, 1)
        session.terminate("interrupted", reason="invented boundary interruption")
        raw = budget.receipt_path(1).read_bytes()
        terminal = json.loads(raw)
        recaptured = {}
        admission = review_resume(requested_spec=self.fx.spec, terminal_bytes=raw, terminal_sha256=digest(raw),
            checkpoint_bytes={k: (session.store.path / ("epoch-%03d.state.json" % int(k))).read_bytes()
                              for k in terminal["checkpoint_hashes"]}, checkpoint_hashes=terminal["checkpoint_hashes"],
            restore_components=lambda x: recaptured.update(x), capture_components=lambda: recaptured)
        fresh = self.reopen(budget)
        resumed = DurableRun(spec=self.fx.spec, inputs=self.fx.inputs, budget=fresh,
                             decision_sha256=digest(b"explicit resume"), resume=admission, parent_attempt=1)
        self.addCleanup(resumed.ticket.close)
        self.assertEqual(resumed.attempt, 2)
        self.assertEqual(resumed.progress, session.progress)
        resumed.terminate("interrupted", reason="synthetic resume evidence only")
        self.assertEqual(fresh.inspect()["consumed"], 2)
        with self.assertRaises(AdmissionBlocked):
            fresh.admit(self.fx.spec, decision_sha256=digest(b"stale parent"), parent_attempt=1,
                        resume_evidence=admission.evidence)
        # Third explicitly planned attempt uses the final slot; failures never refund.
        last = self.reserve(fresh, self.other, b"last slot")
        last.close()
        self.reconcile(fresh, "interrupt_reserved", 3)
        with self.assertRaises(AdmissionBlocked):
            fresh.admit(self.fx.spec, decision_sha256=digest(b"over cap"), parent_attempt=2,
                        resume_evidence=admission.evidence)

    def test_reconciled_crash_cannot_guess_resume_checkpoint(self):
        budget = self.budget()
        self.reserve(budget).close()
        self.reconcile(budget, "interrupt_reserved")
        with self.assertRaises(AdmissionBlocked):
            budget.admit(self.fx.spec, decision_sha256=digest(b"unsafe resume"), parent_attempt=1,
                         resume_evidence={"terminal_sha256": digest(budget.receipt_path(1).read_bytes()),
                                          "restored_components_sha256": digest(b"fabricated")})

    def test_constructor_failure_leaves_reserved_for_explicit_reconciliation(self):
        budget = self.budget()
        with patch("consendxformer.performance_v2.lifecycle.ArtifactStore.put", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.session(budget)
        self.assertEqual(budget.inspect()["attempts"][0]["state"], "reserved")
        self.reconcile(self.reopen(budget), "interrupt_reserved")

    def test_reconciliation_boundaries_preserve_original_terminal_bytes(self):
        for label in ["retention." + x for x in ("before_move", "after_move", "after_sync")] + ["reconcile." + x for x in TX]:
            with self.subTest(label=label):
                budget = self.budget(label)
                session = self.session(budget)
                self.inject(budget, "mirror.after_file_sync")
                with self.assertRaises(Crash):
                    session.terminate("interrupted", reason="invented interrupted export")
                fresh = self.reopen(budget)
                original_sha = fresh.inspect()["attempts"][0]["receipt_sha"]
                self.inject(fresh, label)
                with self.assertRaises(Crash):
                    self.reconcile(fresh, "export_committed")
                recovered = self.reopen(budget)
                self.assertEqual(recovered.inspect()["attempts"][0]["receipt_sha"], original_sha)
                self.reconcile(recovered, "export_committed")
                self.assertEqual(digest(recovered.receipt_path(1).read_bytes()), original_sha)
                self.assertEqual(recovered.inspect()["consumed"], 1)
                self.assertEqual(recovered.inspect()["issues"], [])

    def test_initialization_boundaries_never_overwrite_partial_root(self):
        for boundary in TX:
            with self.subTest(boundary=boundary):
                def fault(label):
                    if label == "initialize." + boundary:
                        raise Crash(label)
                with self.assertRaises(Crash):
                    PersistentBudget.create(self.root / boundary, plan_bytes=self.plan,
                        expected_plan_sha256=digest(self.plan), synthetic_directory_sync=lambda p: None, fault=fault)
                with self.assertRaises(FileExistsError):
                    self.budget(boundary)
                reopened = PersistentBudget(self.root / boundary, expected_plan_sha256=digest(self.plan),
                                             synthetic_directory_sync=lambda p: None)
                if boundary in ("after_commit", "after_sync"):
                    self.assertEqual(reopened.inspect()["consumed"], 0)
                else:
                    with self.assertRaises(ReconciliationRequired):
                        reopened.inspect()

    def test_duplicate_decision_and_missing_member_rejected(self):
        budget = self.budget()
        session = self.session(budget)
        session.terminate("failed", reason="invented failure")
        with self.assertRaises(AdmissionBlocked):
            self.reserve(budget, self.other)  # same decision, distinct allowed run
        (budget.root / "lease.lock").rename(budget.root / "retained-lock")
        with self.assertRaises(ReconciliationRequired):
            self.reserve(budget, self.other, b"new decision")

    def test_completion_proof_cannot_skip_a_checkpoint_or_selection(self):
        for mutation in ("missing_checkpoint", "selection"):
            with self.subTest(mutation=mutation):
                self.fx = Fixture()
                budget = self.budget(mutation)
                session = self.stopped(budget)
                # Corrupt the proposed in-memory proof before publication.
                if mutation == "missing_checkpoint":
                    del session.checkpoints[1]
                else:
                    session.history = session.history[:1]
                with self.assertRaises(ValueError):
                    session.finalize()
                self.assertEqual(budget.inspect()["attempts"][0]["state"], "reserved")

    def test_artifact_low_level_file_sync_link_and_cleanup_failures(self):
        for operation in ("fsync", "link", "cleanup"):
            with self.subTest(operation=operation):
                self.fx = Fixture()
                budget = self.budget(operation)
                session = self.session(budget)
                original = Path.unlink
                def fail_pending(path, *args, **kwargs):
                    if path.parent == session.store.path and path.name.endswith(".pending"):
                        raise OSError("invented cleanup failure")
                    return original(path, *args, **kwargs)
                target = ("consendxformer.performance_v2.lifecycle.os." + operation) if operation != "cleanup" else "pathlib.Path.unlink"
                replacement = fail_pending if operation == "cleanup" else lambda *a, **k: (_ for _ in ()).throw(OSError("invented disk error"))
                with patch(target, replacement):
                    if operation == "cleanup":
                        self.fx.epoch(session, 1)
                    else:
                        with self.assertRaises(OSError):
                            self.fx.epoch(session, 1)
                session.ticket.close()
                fresh = self.reopen(budget)
                self.assertEqual(fresh.inspect()["consumed"], 1)
                with self.assertRaises(AdmissionBlocked):
                    fresh.committed_receipt(1)
                self.assertTrue(any(p.name.endswith(".pending") for p in session.store.path.iterdir()))


if __name__ == "__main__":
    unittest.main()
