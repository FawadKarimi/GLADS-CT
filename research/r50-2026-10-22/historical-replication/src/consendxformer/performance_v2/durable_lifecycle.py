"""Persistent adapter for reviewed recording interfaces; never executes training.

Only the database commits terminal status and receipt together. Receipt files
are immutable exports; use PersistentBudget.committed_receipt for admission to
any subsequent reuse review. There is no execution authorization here.
"""
import copy

from .checkpoint import selection_state, state_sha256
from .lifecycle import RunLifecycle, ResumeAdmission
from .reuse import canonical, digest


class _Bridge:
    def __init__(self, ticket):
        self.ticket, self.claimed = ticket, False

    def reserve(self, run_id, *, kind, decision_sha256):
        ticket = self.ticket
        if (self.claimed or run_id != ticket.run_id or decision_sha256 != ticket.decision
                or kind != ("resume" if ticket.parent is not None else "start")):
            raise ValueError("constructor must use the single committed reservation")
        ticket.check()
        self.claimed = True
        return ticket.number

    def snapshot(self):
        return self.ticket.budget.inspect()

    def finish(self, attempt, status):
        # Constructor failure has no safely established epoch boundary. Leave
        # the consumed reservation intact for explicit crash reconciliation.
        self.ticket.close()


class _SynchronizedStore:
    def __init__(self, store, ticket, spec=None):
        self._store, self._ticket = store, ticket
        self.path = store.path
        from .auxiliary_caps import active
        self._auxiliary_capped = active(spec or {})

    def put(self, name, raw):
        self._ticket.check()
        budget = self._ticket.budget
        if self._auxiliary_capped:
            from .auxiliary_caps import admit_store
            admit_store(self.path,name,raw)
        budget._fault("artifact.before_write")
        result = self._store.put(name, raw)
        budget._fault("artifact.after_write")
        budget._sync(self.path)
        budget._fault("artifact.after_sync")
        return result

    def verify(self):
        self._ticket.check()
        return self._store.verify()


class DurableRun(RunLifecycle):
    """Record invented/externally observed events after exclusive reservation.

    The inherited input, inventory, optimizer, RNG and selection validators are
    preserved. Initialization failure or an uncertain commit closes the lease
    and fences this object. A new object cannot silently adopt its reservation.
    """
    def __init__(self, *, spec, inputs, budget, decision_sha256, resume=None, parent_attempt=None):
        if (spec.get("pilot_execution_authorized") is not False
                or spec.get("input_lock_sha256") != inputs.sha256 or spec.get("scope") != inputs.scope):
            raise ValueError("frozen input/authorization binding mismatch")
        if (resume is None) != (parent_attempt is None):
            raise ValueError("resume requires explicit parent and reviewed admission")
        if resume is not None and (type(resume) is not ResumeAdmission or resume.spec != spec
                                  or state_sha256(resume.payload) != resume._payload_sha256):
            raise ValueError("sealed resume recapture required")
        self.ticket = budget.admit(spec, decision_sha256=decision_sha256, parent_attempt=parent_attempt,
                                   resume_evidence=resume.evidence if resume else None)
        try:
            super().__init__(self.ticket.directory, spec=spec, inputs=inputs, budget=_Bridge(self.ticket),
                             decision_sha256=decision_sha256, resume=resume)
            budget._sync(self.store.path)
            budget._sync(self.store.path.parent)
            self.store = _SynchronizedStore(self.store, self.ticket, spec=spec)
        except BaseException:
            self.status = "reconciliation_required"
            self.ticket.close()
            raise

    def _running(self):
        super()._running()
        self.ticket.check()

    def terminate(self, status, *, reason):
        self._running()
        if status not in ("failed", "interrupted") or not isinstance(reason, str) or not 1 <= len(reason) <= 256:
            raise ValueError("bounded failure/interruption reason required")
        receipt = {"schema": "performance_v2_terminal_v1", "status": status, "reason": reason,
                   "run_id": self.spec["contract"]["run_id"], "spec_sha256": self._spec_sha256,
                   "progress": copy.deepcopy(self.progress), "epoch_boundary": self._boundary,
                   "last_checkpoint": self.last_checkpoint,
                   "checkpoint_hashes": {str(e): sha for e, (_, sha) in self.checkpoints.items()},
                   "pilot_execution_authorized": False}
        if self.retention is not None:receipt['retention'] = self.retention.snapshot()
        try:
            result = self.ticket.terminate(receipt)
        except BaseException:
            self.status = "reconciliation_required"
            self.ticket.close()
            raise
        self.status = status
        return result

    def finalize(self):
        self._running()
        if not self._boundary or not self.history or not selection_state(self.history)["stopped"]:
            raise ValueError("partial run cannot finalize")
        selection = selection_state(self.history)
        if self.retention is None:
            if set(self.checkpoints) != set(range(1,self.progress['epoch']+1)):raise ValueError('all validation checkpoint candidates required')
        else:
            from .retention import validate_snapshot
            validate_snapshot(self.spec,self.retention.snapshot(),{str(e):h for e,(_,h) in self.checkpoints.items()},complete=True)
        try:
            files = self.store.verify()
            manifest = {"schema": "performance_v2_outputs_v1", "files": files,
                        "spec_sha256": self._spec_sha256, "selection": selection,
                        "best": self.checkpoints[selection["best_epoch"]][0], "last": self.last_checkpoint,
                        "scope": self.spec["scope"], "attempt": self.attempt,
                        "progress": copy.deepcopy(self.progress), "budget_before_commit": self.budget.snapshot(),
                        "pilot_execution_authorized": False}
            if self.retention is not None:manifest['retention']=self.retention.snapshot()
            raw = canonical(manifest)
            self.store.put("output-manifest.json", raw)
            artifacts = self.store.verify()
            status = "synthetic_complete" if self.spec["scope"] == "synthetic" else "complete"
            receipt = {"schema": "performance_v2_completion_v1", "status": status,
                       "run_id": self.spec["contract"]["run_id"],
                       "contract_sha256": digest(canonical(self.spec["contract"])),
                       "spec_sha256": self._spec_sha256, "output_manifest_sha256": digest(raw),
                       "scope": self.spec["scope"], "pilot_execution_authorized": False}
            receipt = self.ticket.budget._commit_terminal(self.ticket, receipt, artifacts)
        except BaseException:
            # Never relabel a possibly committed completion or retry this object.
            self.status = "reconciliation_required"
            self.ticket.close()
            raise
        self.status = status
        return receipt, manifest
