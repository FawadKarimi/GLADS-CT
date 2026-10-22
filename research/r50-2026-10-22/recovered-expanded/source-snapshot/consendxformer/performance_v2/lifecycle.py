"""Review interfaces for exclusive artifacts, budgets and epoch-boundary resume.

This module never calls a model, optimizer, data reader or launcher. A future
authorized adapter must authenticate inputs and supply actual observed events.
Synthetic completion is deliberately incompatible with production run reuse.
"""
import copy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import threading
import uuid

from .checkpoint import (MAX_BYTES, encode_state, state_sha256, validate_checkpoint,
                         validate_components, selection_state, _validate_execution_inventory)
from .metrics import DiagnosticAccumulator
from .reuse import _hash, _json, canonical, digest, require_exact_reuse
from .streams import TrainingStreams


class ArtifactStore:
    """Create a new, exclusive directory; publish files with no-overwrite links.

    A private owner-only, non-hostile output root is a deployment prerequisite.
    Hard-link/fsync and crash-durability semantics still need native Linux tests.
    Failed writes retain bounded .pending evidence, never a complete filename.
    """
    def __init__(self, directory):
        path = Path(directory).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("symlink output paths rejected")
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.path = path
        self.files = {}

    def put(self, name, raw):
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
                or name.endswith(".pending") or type(raw) is not bytes or len(raw) > MAX_BYTES):
            raise ValueError("bounded flat artifact name/bytes required")
        if self.path.is_symlink() or name in self.files or (self.path / name).exists():
            raise ValueError("artifact overwrite or output-root substitution rejected")
        pending = self.path / (uuid.uuid4().hex + ".pending")
        with pending.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # A successful link is the publication point; an existing name fails.
        os.link(pending, self.path / name)
        self.files[name] = digest(raw)
        try:
            pending.unlink()
        except OSError:
            pass  # retain the bounded copy; verification blocks finalization
        return self.files[name]

    def verify(self):
        actual = {}
        for path in self.path.iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
                raise ValueError("unclassified output object")
            actual[path.name] = digest(path.read_bytes())
        if actual != self.files:
            raise ValueError("output missing, changed or incomplete; retain evidence")
        return dict(actual)


class BudgetLedger:
    """Single-owner accounting. Every start/resume reservation consumes a slot.

    Failed construction does not refund it. No retry/replacement/expansion API.
    Durable multi-process admission/authorization remains an integration gate.
    """
    def __init__(self, *, limit, plan_sha256):
        if type(limit) is not int or not 1 <= limit <= 13 or not _hash(plan_sha256):
            raise ValueError("reviewed bounded run plan required")
        self.limit, self.plan_sha256 = limit, plan_sha256
        self._entries = []
        self._lock = threading.RLock()

    def reserve(self, run_id, *, kind, decision_sha256):
        with self._lock:
            if (not isinstance(run_id, str) or not run_id or kind not in ("start", "resume")
                    or not _hash(decision_sha256) or len(self._entries) >= self.limit):
                raise ValueError("budget exhausted or explicit start/resume decision missing")
            number = len(self._entries) + 1
            self._entries.append({"attempt": number, "run_id": run_id, "kind": kind,
                                  "decision_sha256": decision_sha256, "status": "reserved"})
            return number

    def finish(self, attempt, status):
        with self._lock:
            if (type(attempt) is not int or not 1 <= attempt <= len(self._entries)
                    or status not in ("failed", "interrupted", "complete", "synthetic_complete")
                    or self._entries[attempt - 1]["status"] != "reserved"):
                raise ValueError("invalid/duplicate terminal budget event")
            self._entries[attempt - 1]["status"] = status

    def snapshot(self):
        return {"schema": "performance_v2_budget_v1", "limit": self.limit,
                "plan_sha256": self.plan_sha256, "entries": copy.deepcopy(self._entries),
                "pilot_execution_authorized": False}

    @classmethod
    def restore(cls, raw, *, expected_sha256):
        if type(raw) is not bytes or not _hash(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError("trusted budget ledger digest required")
        obj = _json(raw)
        if (type(obj) is not dict or set(obj) != {"schema", "limit", "plan_sha256", "entries", "pilot_execution_authorized"}
                or obj["schema"] != "performance_v2_budget_v1" or obj["pilot_execution_authorized"] is not False
                or type(obj["entries"]) is not list):
            raise ValueError("invalid budget ledger")
        restored = cls(limit=obj["limit"], plan_sha256=obj["plan_sha256"])
        for expected, entry in enumerate(obj["entries"], 1):
            if (type(entry) is not dict or set(entry) != {"attempt", "run_id", "kind", "decision_sha256", "status"}
                    or entry["attempt"] != expected):
                raise ValueError("nonsequential budget attempts")
            number = restored.reserve(entry["run_id"], kind=entry["kind"], decision_sha256=entry["decision_sha256"])
            if entry["status"] != "reserved":
                restored.finish(number, entry["status"])
        return restored


class RunLifecycle:
    pilot_execution_authorized = False

    def __init__(self, directory, *, spec, inputs, budget, decision_sha256, resume=None):
        self.spec = copy.deepcopy(spec)
        self._spec_sha256 = digest(canonical(spec))
        if (spec.get("pilot_execution_authorized") is not False
                or spec.get("input_lock_sha256") != inputs.sha256 or spec.get("scope") != inputs.scope):
            raise ValueError("lifecycle input/authorization binding mismatch")
        self.inputs, self.budget = inputs, budget
        self.history, self.checkpoints = [], {}
        self.progress = {"epoch": 0, "updates": 0, "elapsed_seconds": 0.0}
        self._plan, self._batch = None, 0
        self._boundary = True
        self.status = "running"
        self.last_checkpoint = None
        self._streams = TrainingStreams(spec["contract"]["seed"])
        if resume is not None:
            if (type(resume) is not ResumeAdmission or resume.spec != spec
                    or state_sha256(resume.payload) != resume._payload_sha256):
                raise ValueError("verified resume admission required")
            self.progress = copy.deepcopy(resume.payload["progress"])
            self.history = copy.deepcopy(resume.payload["selection"]["history"])
            self._streams.restore(resume.payload["components"]["rng"]["data_streams"])
        self.attempt = budget.reserve(spec["contract"]["run_id"], kind="resume" if resume else "start",
                                      decision_sha256=decision_sha256)
        try:
            self.store = ArtifactStore(directory)
            self.store.put("execution-spec.json", canonical(spec))
            self.store.put("budget-admission.json", canonical(budget.snapshot()))
            from .retention import enabled, SparseRetention
            self.retention = SparseRetention(self,resume) if enabled(spec) else None
            if resume:
                # Preserve all prior epoch states required by the global selection rule.
                for epoch, (raw, sha) in resume.checkpoints.items():
                    self._retain_checkpoint(epoch, raw, sha)
                self.last_checkpoint = "epoch-%03d.state.json" % self.progress["epoch"]
                if self.retention is not None:
                    for name,raw in resume.retention_metadata.items():
                        self.store.put(name,raw)
                self.store.put("resume-admission.json", canonical(resume.evidence))
        except BaseException:
            self.status = "failed"
            budget.finish(self.attempt, "failed")
            raise

    def _running(self):
        if self.status != "running" or digest(canonical(self.spec)) != self._spec_sha256:
            raise ValueError("terminal run cannot continue or finalize")

    @contextmanager
    def guard(self):
        """Retain caught failure/cancellation; normal early exit is interrupted.

        A process kill/power loss cannot be caught. Its reserved budget slot and
        absence of a committed completion remain partial evidence, never reuse.
        """
        self._running()
        try:
            yield self
        except BaseException as error:
            self._retain_original_error(error)
            raise
        else:
            if self.status == "running":
                self.terminate("interrupted", reason="scope exited without completed finalization")

    def _retain_original_error(self, error):
        if self.status != 'running':
            return
        try:
            self.terminate('interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                           reason=type(error).__name__)
        except BaseException as finalization_error:
            if hasattr(error, 'add_note'):
                error.add_note('terminal evidence unavailable: ' + type(finalization_error).__name__)
            if hasattr(self, 'ticket'):
                self.status = 'reconciliation_required'
                try:
                    self.ticket.close()
                except BaseException:
                    pass  # original failure remains authoritative; no retry

    def begin_epoch(self, plan):
        self._running()
        if not self._boundary or self.history and selection_state(self.history)["stopped"]:
            raise ValueError("unfinished or stopped run")
        ids = self.inputs.ids("train")
        order = plan.get("order", [])
        if (set(plan) != {"epoch", "order", "augmentation", "batches", "batch_size", "drop_last", "workers"}
                or type(plan["epoch"]) is not int or plan["epoch"] != self.progress["epoch"] + 1
                or len(order) != len(ids) or set(order) != set(ids)
                or plan["batches"] != [order[i:i + 16] for i in range(0, len(order), 16)]
                or [x.get("id") for x in plan["augmentation"]] != order
                or plan["batch_size"] != 16 or plan["drop_last"] is not False or plan["workers"] != 0):
            raise ValueError("exact frozen batch plan required")
        staged_streams = TrainingStreams(self.spec["contract"]["seed"])
        staged_streams.restore(self._streams.snapshot())
        if plan != staged_streams.epoch_plan(ids, epoch=plan["epoch"]):
            raise ValueError("shuffle/augmentation plan does not match isolated RNG streams")
        try:
            self.store.put("epoch-%03d.plan.json" % plan["epoch"], canonical(plan))
        except BaseException as error:
            self._retain_original_error(error)
            raise
        self._streams = staged_streams
        self._plan, self._batch, self._boundary = copy.deepcopy(plan), 0, False

    def record_batch(self, ids, *, observed_optimizer_updates):
        """Record an externally observed step count; never perform an update."""
        self._running()
        if (self._plan is None or self._batch >= len(self._plan["batches"])
                or list(ids) != self._plan["batches"][self._batch]
                or type(observed_optimizer_updates) is not int
                or observed_optimizer_updates != self.progress["updates"] + 1):
            raise ValueError("batch order/partial batch/update accounting mismatch")
        try:
            self.store.put("epoch-%03d.batch-%04d.json" % (self._plan["epoch"], self._batch + 1),
                           canonical({"ids": list(ids), "optimizer_updates": observed_optimizer_updates}))
        except BaseException as error:
            self._retain_original_error(error)
            raise
        self.progress["updates"] = observed_optimizer_updates
        self._batch += 1

    def _retain_checkpoint(self, epoch, raw, sha):
        validate_checkpoint(raw, expected_sha256=sha, requested_spec=self.spec)
        name = "epoch-%03d.state.json" % epoch
        self.store.put(name, raw)
        self.checkpoints[epoch] = (name, sha)

    def save_epoch(self, metrics, components, *, elapsed_seconds):
        self._running()
        if (self._plan is None or self._batch != len(self._plan["batches"])
                or type(metrics) is not DiagnosticAccumulator or metrics.inputs.sha256 != self.inputs.sha256
                or type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds)
                or elapsed_seconds < self.progress["elapsed_seconds"]):
            raise ValueError("complete epoch, bound validation and monotone elapsed time required")
        epoch = self._plan["epoch"]
        record = metrics.selection_record(epoch)
        history = self.history + [dict(record.__dict__)]
        progress = dict(self.progress, epoch=epoch, elapsed_seconds=elapsed_seconds)
        selection = selection_state(history)
        validate_components(self.spec, components, epoch=epoch, updates=progress["updates"])
        if components["rng"]["data_streams"] != self._streams.snapshot():
            raise ValueError("saved shuffle/augmentation RNG state differs from epoch plan")
        payload = {"schema": "performance_v2_checkpoint_v1", "spec": self.spec,
                   "components": components, "progress": progress, "selection": selection}
        raw = encode_state(payload)
        try:
            self._retain_checkpoint(epoch, raw, digest(raw))
            self.store.put("epoch-%03d.metrics.json" % epoch, canonical(metrics.summary()))
        except BaseException as error:
            self._retain_original_error(error)
            raise
        self.history, self.progress = history, progress
        self.last_checkpoint = self.checkpoints[epoch][0]
        self._plan, self._batch, self._boundary = None, 0, True
        if self.retention is not None:
            try:self.retention.complete_epoch()
            except BaseException as error:
                self._retain_original_error(error)
                raise
        return copy.deepcopy(selection)

    def terminate(self, status, *, reason):
        self._running()
        if status not in ("failed", "interrupted") or not isinstance(reason, str) or not 1 <= len(reason) <= 256:
            raise ValueError("bounded terminal reason required")
        self.status = status
        self.budget.finish(self.attempt, status)
        terminal = {"schema": "performance_v2_terminal_v1", "status": status, "reason": reason,
                    "run_id": self.spec["contract"]["run_id"], "spec_sha256": digest(canonical(self.spec)),
                    "progress": copy.deepcopy(self.progress), "epoch_boundary": self._boundary,
                    "last_checkpoint": self.last_checkpoint,
                    "checkpoint_hashes": {str(e): sha for e, (_, sha) in self.checkpoints.items()},
                    "budget": self.budget.snapshot(), "pilot_execution_authorized": False}
        # A full/unwritable disk may prevent this record. Previously written bytes
        # stay in the exclusive directory; no completion record is attempted.
        self.failure_retention_error = None
        try:
            self.store.put("terminal.json", canonical(terminal))
        except OSError as error:
            self.failure_retention_error = type(error).__name__
        return terminal

    def finalize(self):
        self._running()
        if not self._boundary or not self.history or not selection_state(self.history)["stopped"]:
            raise ValueError("partial run cannot finalize")
        selection = selection_state(self.history)
        if set(self.checkpoints) != set(range(1, self.progress["epoch"] + 1)):
            raise ValueError("missing checkpoint evidence for global selection")
        receipt_raw = None
        try:
            files = self.store.verify()
            manifest = {"schema": "performance_v2_outputs_v1", "files": files,
                        "spec_sha256": digest(canonical(self.spec)), "selection": selection,
                        "best": self.checkpoints[selection["best_epoch"]][0], "last": self.last_checkpoint,
                        "scope": self.spec["scope"], "attempt": self.attempt,
                        "progress": copy.deepcopy(self.progress), "budget_before_commit": self.budget.snapshot(),
                        "pilot_execution_authorized": False}
            manifest_raw = canonical(manifest)
            self.store.put("output-manifest.json", manifest_raw)
            self.store.verify()
            status = "synthetic_complete" if self.spec["scope"] == "synthetic" else "complete"
            receipt = {"schema": "performance_v2_completion_v1", "status": status,
                       "run_id": self.spec["contract"]["run_id"],
                       "contract_sha256": digest(canonical(self.spec["contract"])),
                       "spec_sha256": digest(canonical(self.spec)),
                       "output_manifest_sha256": digest(manifest_raw), "scope": self.spec["scope"],
                       "pilot_execution_authorized": False}
            receipt_raw = canonical(receipt)
            self.store.put("completion.json", receipt_raw)  # last publication
        except BaseException as error:
            # The atomic receipt link is the commit point. An interruption after
            # that link must not relabel a committed, fully verified result as a
            # failed run with a contradictory complete receipt.
            committed = False
            path = self.store.path / "completion.json"
            if receipt_raw is not None and path.is_file() and not path.is_symlink():
                try:
                    committed = path.read_bytes() == receipt_raw
                except OSError:
                    pass
            if not committed:
                self.terminate("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", reason=type(error).__name__)
                raise
            self.post_commit_interruption = type(error).__name__
        self.status = status
        self.budget.finish(self.attempt, status)
        return receipt, manifest


class ResumeAdmission:
    """Construct only via review_resume; validate live recapture before admission."""
    def __init__(self, spec, payload, checkpoints, evidence, *, _token=None):
        if _token is not _RESUME_TOKEN:
            raise ValueError("use review_resume to validate and recapture all state")
        self.spec, self.payload, self.checkpoints, self.evidence = spec, payload, checkpoints, evidence
        self._payload_sha256 = state_sha256(payload)


_RESUME_TOKEN = object()


class _CheckpointReferences:
    """Lazy immutable digest references; reacquire/revalidate before child copy.

    The caller supplies a bounded no-follow reader, not an eager body mapping.
    No reread is trusted merely because admission read that path earlier.
    """
    def __init__(self, source, hashes, spec):
        self._source = source
        self._hashes = copy.deepcopy(hashes)
        self._spec = copy.deepcopy(spec)

    def items(self):
        # Already ordered; sorted(items()) would eagerly acquire every body.
        for key in sorted(self._hashes, key=int):
            raw = self._source[key]
            payload = validate_checkpoint(raw, expected_sha256=self._hashes[key], requested_spec=self._spec)
            if payload["progress"]["epoch"] != int(key):
                raise ValueError("retained checkpoint epoch identity drift")
            del payload
            yield int(key), (raw, self._hashes[key])
            del raw


def review_resume(*, requested_spec, terminal_bytes, terminal_sha256, checkpoint_bytes,
                  checkpoint_hashes, restore_components, capture_components, retention_metadata_bytes=None):
    """Hash and fully validate all evidence before restoration, one body at a time.

    callbacks must target an isolated candidate. A failed restore/recapture must
    discard that candidate; no live run is admitted or budget auto-reserved.
    """
    _validate_execution_inventory(requested_spec)
    if (type(terminal_bytes) is not bytes or not _hash(terminal_sha256)
            or digest(terminal_bytes) != terminal_sha256):
        raise ValueError("trusted terminal evidence hash required")
    terminal = _json(terminal_bytes)
    if (type(terminal) is not dict or terminal.get("schema") != "performance_v2_terminal_v1"
            or terminal.get("status") != "interrupted" or terminal.get("epoch_boundary") is not True
            or terminal.get("run_id") != requested_spec["contract"]["run_id"]
            or terminal.get("spec_sha256") != digest(canonical(requested_spec))
            or terminal.get("pilot_execution_authorized") is not False):
        raise ValueError("only an exactly bound interrupted epoch boundary is resumable")
    hashes = terminal["checkpoint_hashes"]
    if hashes != checkpoint_hashes or set(checkpoint_bytes) != set(hashes) or not hashes:
        raise ValueError("all retained selected/last candidate bytes and trusted digests required")
    # Validate every historical state one at a time. Only selection metadata and
    # the final decoded state survive; no list/dict of checkpoint bodies.
    epochs = sorted(int(key) for key in hashes)
    from .retention import enabled, validate_snapshot, MetadataReferences
    sparse = enabled(requested_spec)
    metadata = None
    if sparse:
        validate_snapshot(requested_spec,terminal.get('retention',{}),hashes)
        metadata = MetadataReferences(retention_metadata_bytes,terminal['retention']['metadata_hashes'])
        metadata.validate(terminal['retention']['selection'])
    if any(key != str(int(key)) for key in hashes) or (not sparse and epochs != list(range(1, max(epochs) + 1))):
        raise ValueError('noncanonical or incomplete epoch history')
    history_by_epoch, last = {}, None
    for epoch in epochs:
        key = str(epoch)
        raw = checkpoint_bytes[key]
        payload = validate_checkpoint(raw, expected_sha256=hashes[key], requested_spec=requested_spec)
        if payload["progress"]["epoch"] != epoch:
            raise ValueError("checkpoint epoch identity mismatch")
        history_by_epoch[epoch] = copy.deepcopy(payload["selection"]["history"])
        if epoch == epochs[-1]:
            last = payload
        del raw, payload
    if (last["progress"] != terminal["progress"]
            or terminal["last_checkpoint"] != "epoch-%03d.state.json" % epochs[-1]
            or last["selection"]["stopped"]):
        raise ValueError("stale, partial or already stopped resume state")
    for epoch, history in history_by_epoch.items():
        if history != last["selection"]["history"][:epoch]:
            raise ValueError("history changed between retained checkpoints")
    retained = _CheckpointReferences(checkpoint_bytes, hashes, requested_spec)
    if sparse and terminal['retention']['selection'] != last['selection']:raise ValueError('sparse terminal/latest selection mismatch')
    wanted = state_sha256(last["components"])
    restore_components(copy.deepcopy(last["components"]))
    if state_sha256(capture_components()) != wanted:
        raise ValueError("resume recapture differs: model/optimizer/scheduler/RNG not restored exactly")
    evidence = {"terminal_sha256": terminal_sha256, "last_checkpoint_sha256": hashes[str(epochs[-1])],
                "restored_components_sha256": wanted, "pilot_execution_authorized": False}
    admission = ResumeAdmission(copy.deepcopy(requested_spec), last, retained, evidence, _token=_RESUME_TOKEN)
    if sparse:
        evidence['retention'] = copy.deepcopy(terminal['retention'])
        admission.retention_metadata = metadata
    return admission


def require_completed_reuse(spec, *, completion_bytes, completion_sha256, manifest_bytes, verified_output_hashes):
    _validate_execution_inventory(spec)
    if (type(completion_bytes) is not bytes or not _hash(completion_sha256)
            or digest(completion_bytes) != completion_sha256):
        raise ValueError("completion digest mismatch before parsing")
    receipt = _json(completion_bytes)
    if type(receipt) is not dict or receipt.get("output_manifest_sha256") != digest(manifest_bytes):
        raise ValueError("manifest digest mismatch before parsing")
    manifest = _json(manifest_bytes)
    if (type(manifest) is not dict or spec.get("scope") != "production" or receipt.get("scope") != "production"
            or receipt.get("schema") != "performance_v2_completion_v1"
            or receipt.get("spec_sha256") != digest(canonical(spec))
            or manifest.get("spec_sha256") != digest(canonical(spec))
            or manifest.get("scope") != "production"
            or selection_state(manifest["selection"]["history"]) != manifest["selection"]
            or manifest["selection"]["stopped"] is not True):
        raise ValueError("production complete, validation-selected evidence required")
    return require_exact_reuse(spec["contract"], spec["contract"], completion_bytes=completion_bytes,
                              completion_sha256=completion_sha256, manifest_bytes=manifest_bytes,
                              verified_output_hashes=verified_output_hashes)
