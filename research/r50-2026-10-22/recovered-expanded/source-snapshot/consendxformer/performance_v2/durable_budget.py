"""Persistent single-active-attempt admission; SQLite owns terminal receipts.

No runner, retry, plan expansion, checkpoint loading or PID/age lock stealing.
Requires a private local filesystem and cooperating clients. Native filesystem
and power-loss guarantees must be validated before production acceptance.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sqlite3
import uuid

from .reuse import canonical, digest, _hash, _json

TERMINAL = ("failed", "interrupted", "complete", "synthetic_complete")
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


class AdmissionBlocked(RuntimeError):
    pass


class ReconciliationRequired(AdmissionBlocked):
    pass


def sync_directory(path):
    """No silent Windows/network-filesystem fallback for directory durability."""
    if os.name != "posix":
        raise ReconciliationRequired("native directory fsync unavailable; synthetic callback required on this host")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Lease:
    """Nonblocking OS lifetime lock. The lock file is never deleted or replaced."""
    def __init__(self, path):
        self.stream = None
        if path.is_symlink() or path.stat().st_size != 1:
            raise ReconciliationRequired("invalid persistent lock file")
        stream = path.open("r+b", buffering=0)
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise AdmissionBlocked("another owner holds the budget lease; never steal by PID or age") from error
        try:
            stream.seek(0)
            if stream.read(1) != b"0" or (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino) != (path.stat().st_dev, path.stat().st_ino):
                raise ReconciliationRequired("lock file identity changed")
        except BaseException:
            stream.close()
            raise
        self.stream = stream

    def close(self):
        if self.stream is not None:
            # Closing the handle releases the kernel lock, including on process death.
            self.stream.close()
            self.stream = None


def _plan(raw, expected):
    if type(raw) is not bytes or not _hash(expected) or digest(raw) != expected:
        raise ValueError("trusted immutable plan digest required")
    value = _json(raw)
    if (type(value) is not dict or set(value) != {"schema", "scope", "limit", "runs", "pilot_execution_authorized"}
            or value["schema"] != "performance_v2_durable_plan_v1" or value["scope"] not in ("synthetic", "production")
            or type(value["limit"]) is not int or not 1 <= value["limit"] <= 13
            or type(value["runs"]) is not dict or not value["runs"] or len(value["runs"]) > value["limit"]
            or not all(isinstance(k, str) and 0 < len(k) <= 128 and _hash(v) for k, v in value["runs"].items())
            or value["pilot_execution_authorized"] is not False):
        raise ValueError("bounded exact run plan required; no execution authorization")
    return value


class PersistentBudget:
    """No cached budget counter: every admission reloads the immutable plan/DB.

    Tests may explicitly supply synthetic_directory_sync for synthetic plans.
    Production plans always require the real directory barrier, never that hook.
    fault labels are deterministic injection points; no retries are performed.
    """
    def __init__(self, root, *, expected_plan_sha256, synthetic_directory_sync=None, fault=None):
        self.root = Path(root).absolute()
        self.expected_plan_sha256 = expected_plan_sha256
        self._safe_root()
        self.plan_bytes = (self.root / "plan.json").read_bytes()
        self.plan = _plan(self.plan_bytes, expected_plan_sha256)
        if synthetic_directory_sync is not None and self.plan["scope"] != "synthetic":
            raise ValueError("synthetic directory barrier cannot admit production")
        self._sync = synthetic_directory_sync or sync_directory
        self.storage_profile = "synthetic_directory_barrier" if synthetic_directory_sync else "native_directory_fsync"
        self._fault = fault or (lambda label: None)

    @classmethod
    def create(cls, root, *, plan_bytes, expected_plan_sha256, synthetic_directory_sync=None, fault=None):
        plan = _plan(plan_bytes, expected_plan_sha256)
        if synthetic_directory_sync is not None and plan["scope"] != "synthetic":
            raise ValueError("synthetic barrier requires a synthetic plan")
        path = Path(root).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("symlink root rejected")
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        sync = synthetic_directory_sync or sync_directory
        for name, raw in (("plan.json", plan_bytes), ("lease.lock", b"0"), ("budget.sqlite3", b"")):
            with (path / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        for name in ("runs", "receipts", "retained"):
            (path / name).mkdir(mode=0o700)
            sync(path / name)
        sync(path)
        sync(path.parent)
        result = cls(path, expected_plan_sha256=expected_plan_sha256,
                     synthetic_directory_sync=synthetic_directory_sync, fault=fault)
        lease = result._lease()
        try:
            def initialize(connection):
                connection.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL, plan BLOB NOT NULL, profile TEXT NOT NULL)")
                connection.execute("INSERT INTO meta VALUES(1,?,?,?)", (uuid.uuid4().hex, plan_bytes, result.storage_profile))
                connection.execute("""CREATE TABLE attempts (
                    number INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('start','resume')),
                    decision TEXT UNIQUE NOT NULL, parent INTEGER UNIQUE REFERENCES attempts(number), token TEXT UNIQUE NOT NULL,
                    spec BLOB NOT NULL, spec_sha TEXT NOT NULL, state TEXT NOT NULL
                        CHECK(state IN ('reserved','interrupted','failed','complete','synthetic_complete')),
                    receipt BLOB, receipt_sha TEXT, artifacts BLOB, exported INTEGER NOT NULL DEFAULT 0 CHECK(exported IN (0,1)),
                    CHECK((state='reserved' AND receipt IS NULL AND receipt_sha IS NULL AND artifacts IS NULL AND exported=0)
                       OR (state!='reserved' AND receipt IS NOT NULL AND receipt_sha IS NOT NULL AND artifacts IS NOT NULL)),
                    CHECK((kind='start' AND parent IS NULL) OR (kind='resume' AND parent IS NOT NULL)))""")
                connection.execute("CREATE UNIQUE INDEX one_reserved_attempt ON attempts((1)) WHERE state='reserved'")
                connection.execute("CREATE UNIQUE INDEX one_initial_start ON attempts(run_id) WHERE kind='start'")
                connection.execute("CREATE TABLE events (number INTEGER PRIMARY KEY, body BLOB NOT NULL, sha TEXT NOT NULL)")
            result._transaction("initialize", initialize)
        finally:
            lease.close()
        return result

    def _safe_root(self):
        if not self.root.is_dir() or any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ReconciliationRequired("private existing nonsymlink budget root required")
        for name in ("plan.json", "budget.sqlite3", "lease.lock", "runs", "receipts", "retained"):
            if (self.root / name).is_symlink():
                raise ReconciliationRequired("budget path substituted by symlink")
            expected_directory = name in ("runs", "receipts", "retained")
            if not ((self.root / name).is_dir() if expected_directory else (self.root / name).is_file()):
                raise ReconciliationRequired("budget member missing or type changed")

    def _lease(self):
        self._safe_root()
        return _Lease(self.root / "lease.lock")

    @contextmanager
    def _connection(self, *, write=False):
        self._safe_root()
        uri = (self.root / "budget.sqlite3").as_uri() + ("?mode=rw" if write else "?mode=ro")
        connection = sqlite3.connect(uri, uri=True, timeout=0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=0")
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ReconciliationRequired("unexpected SQLite journal mode; no automatic conversion")
            if write:
                connection.execute("PRAGMA synchronous=EXTRA")
                if connection.execute("PRAGMA synchronous").fetchone()[0] != 3:
                    raise ReconciliationRequired("SQLite EXTRA synchronization required")
            else:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")  # one consistent snapshot across rows, audit and receipt checks
            yield connection
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()

    def _transaction(self, label, action):
        with self._connection(write=True) as connection:
            self._fault(label + ".before_begin")
            connection.execute("BEGIN IMMEDIATE")
            self._fault(label + ".after_begin")
            result = action(connection)
            self._fault(label + ".after_write")
            self._fault(label + ".before_commit")
            connection.commit()
            self._fault(label + ".after_commit")
            self._sync(self.root)
            self._fault(label + ".after_sync")
            return result

    @staticmethod
    def _event(connection, kind, attempt, detail):
        last = connection.execute("SELECT number,sha FROM events ORDER BY number DESC LIMIT 1").fetchone()
        number, previous = (last["number"] + 1, last["sha"]) if last else (1, "0" * 64)
        raw = canonical({"number": number, "previous_sha256": previous, "kind": kind, "attempt": attempt, "detail": detail})
        connection.execute("INSERT INTO events VALUES(?,?,?)", (number, raw, digest(raw)))

    def run_path(self, number):
        if type(number) is not int or not 1 <= number <= 13:
            raise ValueError("invalid attempt number")
        return self.root / "runs" / ("attempt-%04d" % number)

    def receipt_path(self, number):
        self.run_path(number)
        return self.root / "receipts" / ("attempt-%04d.json" % number)

    def _artifacts(self, number):
        path = self.run_path(number)
        if not path.exists():
            return {}
        if not path.is_dir() or path.is_symlink():
            raise ReconciliationRequired("unclassified run directory")
        result = {}
        for member in path.iterdir():
            if (member.is_symlink() or not member.is_file() or member.stat().st_size > MAX_ARTIFACT_BYTES
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", member.name) is None):
                raise ReconciliationRequired("unclassified run artifact")
            # Stream digesting never deserializes or loads model/checkpoint tensors.
            import hashlib
            hasher = hashlib.sha256()
            with member.open("rb") as source:
                for block in iter(lambda: source.read(1048576), b""):
                    hasher.update(block)
            result[member.name] = hasher.hexdigest()
        return result

    def _load(self, connection):
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ReconciliationRequired("SQLite integrity failure")
        meta = connection.execute("SELECT * FROM meta WHERE id=1").fetchone()
        if (meta is None or meta["plan"] != self.plan_bytes or (self.root / "plan.json").read_bytes() != self.plan_bytes
                or meta["profile"] != self.storage_profile):
            raise ReconciliationRequired("plan or storage profile drift")
        rows = [dict(row) for row in connection.execute("SELECT * FROM attempts ORDER BY number")]
        if len(rows) > self.plan["limit"]:
            raise ReconciliationRequired("budget exceeded")
        for number, row in enumerate(rows, 1):
            if (row["number"] != number or row["run_id"] not in self.plan["runs"]
                    or digest(row["spec"]) != row["spec_sha"] or row["spec_sha"] != self.plan["runs"][row["run_id"]]):
                raise ReconciliationRequired("attempt identity or specification drift")
            if row["state"] != "reserved":
                if digest(row["receipt"]) != row["receipt_sha"]:
                    raise ReconciliationRequired("canonical receipt corrupt")
                receipt = _json(row["receipt"])
                if (type(receipt) is not dict or receipt.get("status") != row["state"] or receipt.get("run_id") != row["run_id"]
                        or receipt.get("spec_sha256") != row["spec_sha"]
                        or receipt.get("pilot_execution_authorized") is not False
                        or receipt.get("persistent_budget") != self._binding(meta["identity"], row)):
                    raise ReconciliationRequired("budget and canonical receipt contradict")
                artifacts = _json(row["artifacts"])
                if type(artifacts) is not dict or not all(_hash(v) for v in artifacts.values()):
                    raise ReconciliationRequired("invalid stored artifact bindings")
        previous = "0" * 64
        events = []
        for number, row in enumerate(connection.execute("SELECT * FROM events ORDER BY number"), 1):
            body = _json(row["body"])
            if (type(body) is not dict or row["number"] != number or row["sha"] != digest(row["body"])
                    or body.get("number") != number or body.get("previous_sha256") != previous):
                raise ReconciliationRequired("audit chain corrupt")
            previous = row["sha"]
            events.append(body)
        # Replay terminal/export transitions, independently of the current rows.
        replay = {}
        for event in events:
            n, kind, detail = event["attempt"], event["kind"], event["detail"]
            if kind == "reserve":
                if n in replay or n != len(replay) + 1:
                    raise ReconciliationRequired("duplicate or missing reservation audit")
                replay[n] = ["reserved", None, 0]
            elif kind == "terminal":
                if n not in replay or replay[n][0] != "reserved" or detail["state"] not in TERMINAL:
                    raise ReconciliationRequired("invalid terminal audit transition")
                replay[n] = [detail["state"], detail["receipt_sha256"], 0]
            elif kind == "export":
                if n not in replay or replay[n][0] == "reserved" or replay[n][1] != detail["receipt_sha256"]:
                    raise ReconciliationRequired("invalid export audit transition")
                replay[n][2] = 1
            elif kind != "reconcile_export" or n not in replay or replay[n][0] == "reserved":
                raise ReconciliationRequired("unknown audit transition")
        if len(replay) != len(rows) or any(replay.get(r["number"]) != [r["state"], r["receipt_sha"], r["exported"]] for r in rows):
            raise ReconciliationRequired("audit and budget contradict")
        return meta["identity"], rows, previous, events

    def _binding(self, identity, row):
        return {"identity": identity, "plan_sha256": self.expected_plan_sha256,
                "attempt": row["number"], "token": row["token"]}

    def _report(self, connection):
        identity, rows, tail, events = self._load(connection)
        issues, attempts = [], []
        expected_dirs = {self.run_path(row["number"]).name for row in rows}
        if any(p.name not in expected_dirs or p.is_symlink() for p in (self.root / "runs").iterdir()):
            issues.append("orphan_run_directory")
        receipt_names = {self.receipt_path(row["number"]).name for row in rows if row["state"] != "reserved"}
        pending = {}
        for path in (self.root / "receipts").iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1048576:
                issues.append("unsafe_receipt_object:" + path.name)
            elif path.name.endswith(".pending"):
                pending[path.name] = digest(path.read_bytes())
                issues.append("pending_receipt:" + path.name)
            elif path.name not in receipt_names:
                issues.append("orphan_receipt:" + path.name)
        for row in rows:
            number = row["number"]
            observed = self._artifacts(number)
            if "completion.json" in observed or "terminal.json" in observed:
                issues.append("unmanaged_local_receipt:%d" % number)
            if row["state"] == "reserved":
                issues.append("reserved_attempt:%d" % number)
            else:
                if observed != _json(row["artifacts"]):
                    issues.append("artifact_drift:%d" % number)
                path = self.receipt_path(number)
                if not path.exists():
                    issues.append("missing_receipt_export:%d" % number)
                elif path.is_symlink() or digest(path.read_bytes()) != row["receipt_sha"]:
                    issues.append("conflicting_receipt_export:%d" % number)
                if not row["exported"]:
                    issues.append("unacknowledged_export:%d" % number)
            attempts.append({k: row[k] for k in ("number", "run_id", "kind", "decision", "parent", "state", "exported", "receipt_sha")}
                            | {"artifact_hashes": observed})
        retained = {}
        for path in (self.root / "retained").iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1048576:
                raise ReconciliationRequired("unsafe retained evidence")
            retained[path.name] = digest(path.read_bytes())
        return {"schema": "performance_v2_persistent_budget_snapshot_v1", "identity": identity,
                "plan_sha256": self.expected_plan_sha256, "limit": self.plan["limit"], "consumed": len(rows),
                "attempts": attempts, "issues": sorted(issues), "pending_receipt_hashes": pending,
                "retained_evidence_hashes": retained,
                "audit_tail_sha256": tail, "event_count": len(events), "storage_profile": self.storage_profile,
                "pilot_execution_authorized": False}

    def inspect(self):
        """Read-only observation, not permission to admit, resume or steal a lock."""
        try:
            with self._connection() as connection:
                return self._report(connection)
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
            raise ReconciliationRequired("budget state unreadable or ambiguous") from error

    def admit(self, spec, *, decision_sha256, parent_attempt=None, resume_evidence=None):
        from .checkpoint import _validate_execution_inventory
        _validate_execution_inventory(spec)
        raw, run_id = canonical(spec), spec["contract"]["run_id"]
        if (not _hash(decision_sha256) or spec["scope"] != self.plan["scope"]
                or self.plan["runs"].get(run_id) != digest(raw) or spec["pilot_execution_authorized"] is not False):
            raise AdmissionBlocked("run or decision not bound by immutable plan")
        lease = self._lease()
        try:
            def reserve(connection):
                report = self._report(connection)
                if report["issues"]:
                    raise ReconciliationRequired("reconcile unresolved state before any further admission")
                _, rows, _, _ = self._load(connection)
                if len(rows) >= self.plan["limit"] or any(row["decision"] == decision_sha256 for row in rows):
                    raise AdmissionBlocked("budget exhausted or duplicate decision")
                existing = [row for row in rows if row["run_id"] == run_id]
                kind = "resume" if parent_attempt is not None else "start"
                if kind == "start":
                    if existing:
                        raise AdmissionBlocked("duplicate initial admission; no retry/replacement")
                else:
                    if (type(parent_attempt) is not int or not existing or existing[-1]["number"] != parent_attempt
                            or existing[-1]["state"] != "interrupted" or type(resume_evidence) is not dict
                            or _json(existing[-1]["receipt"]).get("epoch_boundary") is not True
                            or resume_evidence.get("terminal_sha256") != existing[-1]["receipt_sha"]
                            or not _hash(resume_evidence.get("restored_components_sha256"))):
                        raise AdmissionBlocked("explicit latest interrupted resume evidence required")
                from .retention import enabled
                if kind=='resume' and enabled(spec) and resume_evidence.get('retention')!=_json(existing[-1]['receipt']).get('retention'):
                    raise AdmissionBlocked('sparse recovery evidence differs from authoritative parent receipt')
                number, token = len(rows) + 1, uuid.uuid4().hex
                connection.execute("INSERT INTO attempts(number,run_id,kind,decision,parent,token,spec,spec_sha,state) VALUES(?,?,?,?,?,?,?,?, 'reserved')",
                                   (number, run_id, kind, decision_sha256, parent_attempt, token, raw, digest(raw)))
                self._event(connection, "reserve", number, {"decision_sha256": decision_sha256, "kind": kind})
                return number, token
            number, token = self._transaction("admission", reserve)
            return Admission(self, lease, number, token, spec, decision_sha256, parent_attempt, _token=_ADMISSION_TOKEN)
        except BaseException:
            lease.close()
            raise

    def _owned(self, connection, ticket):
        if not ticket.live:
            raise AdmissionBlocked("closed/poisoned admission cannot execute")
        identity, rows, _, _ = self._load(connection)
        row = next((row for row in rows if row["number"] == ticket.number), None)
        if row is None or row["token"] != ticket.token or row["state"] != "reserved":
            raise AdmissionBlocked("attempt is no longer reserved by this owner")
        return identity, row

    def _commit_terminal(self, ticket, receipt, artifacts):
        status = receipt["status"]
        if status not in TERMINAL:
            raise ValueError("invalid terminal state")
        def commit(connection):
            identity, row = self._owned(connection, ticket)
            raw = canonical(dict(receipt, persistent_budget=self._binding(identity, row)))
            if (receipt["run_id"] != row["run_id"] or receipt["spec_sha256"] != row["spec_sha"]
                    or receipt.get("pilot_execution_authorized") is not False or self._artifacts(row["number"]) != artifacts):
                raise ReconciliationRequired("terminal identity/artifacts changed before commit")
            if status in ("complete", "synthetic_complete"):
                self._validate_completion(row, receipt, artifacts)
            connection.execute("UPDATE attempts SET state=?,receipt=?,receipt_sha=?,artifacts=? WHERE number=? AND state='reserved'",
                               (status, raw, digest(raw), canonical(artifacts), row["number"]))
            self._event(connection, "terminal", row["number"], {"state": status, "receipt_sha256": digest(raw)})
            return raw
        try:
            raw = self._transaction("completion" if status in ("complete", "synthetic_complete") else "terminal", commit)
            self._export(ticket.number, raw)
            ticket.close()
            return _json(raw)
        except BaseException:
            # No compensating failure write: COMMIT may already have succeeded.
            ticket.close()
            raise

    def _validate_completion(self, row, receipt, artifacts):
        """Recheck retained typed evidence before issuing any complete receipt."""
        from .checkpoint import selection_state, validate_checkpoint
        spec = _json(row["spec"])
        wanted = "synthetic_complete" if spec["scope"] == "synthetic" else "complete"
        raw = (self.run_path(row["number"]) / "output-manifest.json").read_bytes()
        manifest = _json(raw)
        if (receipt.get("schema") != "performance_v2_completion_v1" or receipt["status"] != wanted
                or receipt.get("scope") != spec["scope"] or receipt.get("output_manifest_sha256") != digest(raw)
                or receipt.get("contract_sha256") != digest(canonical(spec["contract"]))
                or manifest.get("schema") != "performance_v2_outputs_v1"
                or manifest.get("spec_sha256") != row["spec_sha"] or manifest.get("scope") != spec["scope"]
                or manifest.get("attempt") != row["number"] or manifest.get("pilot_execution_authorized") is not False
                or artifacts != dict(manifest["files"], **{"output-manifest.json": digest(raw)})
                or any(name.endswith(".pending") or name in ("completion.json", "terminal.json") for name in artifacts)):
            raise ReconciliationRequired("incomplete or contradictory completion proof")
        selection = selection_state(manifest["selection"]["history"])
        epoch = manifest["progress"]["epoch"]
        if (selection != manifest["selection"] or not selection["stopped"] or epoch != len(selection["history"])
                or manifest["best"] != "epoch-%03d.state.json" % selection["best_epoch"]
                or manifest["last"] != "epoch-%03d.state.json" % epoch):
            raise ReconciliationRequired("partial or invalid validation selection")
        from .retention import enabled,validate_snapshot,MetadataReferences
        if enabled(spec):
            retention=manifest.get('retention',{})
            hashes={str(int(name[6:9])):h for name,h in artifacts.items() if __import__('re').fullmatch(r'epoch-\d{3}\.state\.json',name)}
            validate_snapshot(spec,retention,hashes,complete=True)
            MetadataReferences({n:(self.run_path(row['number'])/n).read_bytes() for n in retention['metadata_hashes']},retention['metadata_hashes']).validate(retention['selection'])
            if any(artifacts.get(k)!=h for k,h in retention['metadata_hashes'].items()):raise ReconciliationRequired('retention metadata contradicts output manifest')
            epochs=sorted(map(int,hashes))
        else:epochs=range(1,epoch+1)
        for n in epochs:
            name = "epoch-%03d.state.json" % n
            payload = validate_checkpoint((self.run_path(row["number"]) / name).read_bytes(),
                                          expected_sha256=artifacts[name], requested_spec=spec)
            if (payload["progress"]["epoch"] != n or payload["selection"]["history"] != selection["history"][:n]
                    or n == epoch and payload["progress"] != manifest["progress"]):
                raise ReconciliationRequired("checkpoint history/progress contradicts manifest")

    def _export(self, number, raw):
        target = self.receipt_path(number)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != raw:
                raise ReconciliationRequired("conflicting receipt mirror; never overwrite")
        else:
            temporary = target.parent / ("attempt-%04d.%s.pending" % (number, uuid.uuid4().hex))
            self._fault("mirror.before_write")
            with temporary.open("xb") as stream:
                stream.write(raw)
                self._fault("mirror.after_write")
                stream.flush()
                os.fsync(stream.fileno())
                self._fault("mirror.after_file_sync")
            os.link(temporary, target)
            self._fault("mirror.after_link")
            self._sync(target.parent)
            self._fault("mirror.after_directory_sync")
            temporary.unlink()
            self._sync(target.parent)
            self._fault("mirror.after_cleanup")
        self._sync(target.parent)
        def acknowledge(connection):
            _, rows, _, _ = self._load(connection)
            row = rows[number - 1]
            if row["state"] == "reserved" or row["receipt"] != raw or target.read_bytes() != raw:
                raise ReconciliationRequired("export not equal to authoritative terminal transaction")
            connection.execute("UPDATE attempts SET exported=1 WHERE number=?", (number,))
            self._event(connection, "export", number, {"receipt_sha256": digest(raw)})
        self._transaction("export", acknowledge)

    def reconcile(self, number, *, action, expected_snapshot_sha256, decision_sha256):
        """Explicit CAS reconciliation with no lock expiry, retry or execution.

        interrupt_reserved consumes the existing slot and never guesses a safe
        checkpoint boundary. export_committed mirrors the already committed
        receipt only, preserving any interrupted mirror bytes under retained/.
        """
        if not _hash(expected_snapshot_sha256) or not _hash(decision_sha256):
            raise ValueError("fresh snapshot and explicit reconciliation decision required")
        lease = self._lease()
        ticket = None
        try:
            with self._connection(write=True) as connection:
                report = self._report(connection)
                identity, rows, _, _ = self._load(connection)
            if digest(canonical(report)) != expected_snapshot_sha256:
                raise ReconciliationRequired("stale reconciliation snapshot")
            if type(number) is not int or not 1 <= number <= len(rows):
                raise ValueError("known attempt required")
            row = rows[number - 1]
            if action == "interrupt_reserved":
                if row["state"] != "reserved" or report["issues"] != ["reserved_attempt:%d" % number]:
                    raise ReconciliationRequired("ambiguous receipt/artifact state; do not infer interruption")
                spec = _json(row["spec"])
                ticket = Admission(self, lease, number, row["token"], spec, row["decision"], row["parent"], _token=_ADMISSION_TOKEN)
                receipt = {"schema": "performance_v2_terminal_v1", "status": "interrupted", "run_id": row["run_id"],
                           "spec_sha256": row["spec_sha"], "reason": "explicit reconciliation of unowned reservation",
                           "progress": None, "epoch_boundary": False, "last_checkpoint": None, "checkpoint_hashes": {},
                           "reconciliation_decision_sha256": decision_sha256, "pilot_execution_authorized": False}
                return self._commit_terminal(ticket, receipt, self._artifacts(number))
            if action != "export_committed" or row["state"] == "reserved":
                raise ReconciliationRequired("action does not match committed state")
            allowed = {"missing_receipt_export:%d" % number, "unacknowledged_export:%d" % number}
            allowed |= {"pending_receipt:" + name for name in report["pending_receipt_hashes"]
                        if re.fullmatch(r"attempt-%04d\.[a-f0-9]{32}\.pending" % number, name)}
            if not set(report["issues"]) <= allowed:
                raise ReconciliationRequired("conflicting/corrupt evidence requires independent review")
            for name, sha in report["pending_receipt_hashes"].items():
                source = self.root / "receipts" / name
                if digest(source.read_bytes()) != sha:
                    raise ReconciliationRequired("pending evidence changed")
                target = self.root / "retained" / (uuid.uuid4().hex + "." + name)
                self._fault("retention.before_move")
                os.rename(source, target)  # both fixed directories inside this verified root
                self._fault("retention.after_move")
                self._sync(target.parent)
                self._sync(source.parent)
                self._fault("retention.after_sync")
            def audit(connection):
                self._event(connection, "reconcile_export", number,
                            {"decision_sha256": decision_sha256, "snapshot_sha256": expected_snapshot_sha256,
                             "retained_pending_hashes": report["pending_receipt_hashes"]})
            self._transaction("reconcile", audit)
            self._export(number, row["receipt"])
            return _json(row["receipt"])
        finally:
            if ticket:
                ticket.close()
            else:
                lease.close()

    def committed_receipt(self, number):
        """Fresh budget AND filesystem check; mirrors alone cannot prove completion."""
        lease = self._lease()
        try:
            with self._connection() as connection:
                report = self._report(connection)
                _, rows, _, _ = self._load(connection)
            if report["issues"] or type(number) is not int or not 1 <= number <= len(rows):
                raise ReconciliationRequired("unreconciled budget/evidence")
            if rows[number - 1]["state"] not in ("complete", "synthetic_complete"):
                raise AdmissionBlocked("failed/interrupted/reserved is not complete")
            return rows[number - 1]["receipt"]
        finally:
            lease.close()


_ADMISSION_TOKEN = object()


class Admission:
    def __init__(self, budget, lease, number, token, spec, decision, parent, *, _token=None):
        if _token is not _ADMISSION_TOKEN:
            raise ValueError("atomic budget admission required")
        self.budget, self.lease, self.number, self.token = budget, lease, number, token
        self.spec_sha256, self.run_id = digest(canonical(spec)), spec["contract"]["run_id"]
        self.decision, self.parent = decision, parent

    @property
    def live(self):
        return self.lease.stream is not None

    @property
    def directory(self):
        return self.budget.run_path(self.number)

    def check(self):
        with self.budget._connection() as connection:
            self.budget._owned(connection, self)

    def close(self):
        self.lease.close()

    def terminate(self, receipt):
        if receipt.get("status") not in ("failed", "interrupted"):
            raise ValueError("termination cannot issue completion")
        self.check()
        return self.budget._commit_terminal(self, receipt, self.budget._artifacts(self.number))
