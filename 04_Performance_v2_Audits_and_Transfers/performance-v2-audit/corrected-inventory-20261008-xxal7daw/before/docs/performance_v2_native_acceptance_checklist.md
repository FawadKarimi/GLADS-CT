# Consolidated native acceptance checklist (review candidate)

2026-10-07. `pilot_execution_authorized=false`. Stage B and GPU pilots remain
paused. No item below authorizes package installation, weight acquisition,
research evaluation, training, publication or remote dispatch.

## Frozen package and one offline entry point

`configs/performance_v2/native_acceptance_source_lock.json` binds the current
Performance-v2 modules, all Performance-v2 tests, both historical/current pilot
configurations, provenance and protocol documents, diagnostic support scripts,
and accepted Stage A closure documents. The source-lock file is excluded from
its own member ledger; its exact SHA-256 is supplied separately in the release
review. Verify the ZIP and lock against that independent review, not a digest
computed from an untrusted replacement package.

The single entry point is `scripts/performance_v2_native_acceptance.py`.
`inventory` uses standard-library metadata inspection only. It does not import
Torch/OpenCV, probe a GPU, connect to a service or create output files; shell
redirection may separately retain its stdout. `test` verifies the lock before
imports, rejects missing dependencies without installation, creates a new
exclusive output directory and uses synthetic temporary directories.

The default `--suite full` runs the entire frozen Performance-v2 synthetic test
inventory: implementation/contract/training-mode/checkpoint/lifecycle tests,
real subprocess tests, plus historical diagnostic, transform and input-audit
pytest tests. It requires the already installed Torch, torchvision, NumPy,
pytest, OpenCV and Git runtime. No dependency versions become installation pins
by appearing in a test report. Optional LIDC and Phase-5 environments remain
separate. `--suite implementation` is explicit partial evidence and can never
meet the complete-suite/native-evidence gate. Missing pytest is a blocker, not a
reason to silently skip the older tests or substitute another test framework.

Tests deny network operations and unrelated subprocesses. Allowed subprocesses
are exact child-test invocations of this entry point and, for the legacy tests,
bounded Git fixture commands inside the temporary root. Git global/system
configuration and hooks/templates are isolated. Children verify the same lock
and apply their own network/weight/optimizer guards. Actual AdamW/SGD updates,
pretrained loading and research-file access remain forbidden. These application
guards are defense in depth, not a replacement for host network isolation.

## Genuine process-kill evidence and its limits

The new tests start distinct interpreter PIDs. Two children compete for one
reservation; the winner keeps its lease while the other is rejected. The parent
also kills a child while reserved, before and after failed-terminal COMMIT,
before and after complete COMMIT, and during receipt export after hard-link
publication. The parent performs a fresh inspection and explicit reconciliation,
checks immutable receipt bytes, rejects duplicate starts and proves the original
slot remains consumed when the next distinct planned attempt uses slot two.

Linux uses SIGKILL and the existing real directory-fsync implementation. Windows
uses TerminateProcess with an explicitly mocked directory barrier. Workers stop
at signaled barriers so the parent knows which commit boundary was reached;
there is no exception-based unwind in the killed child. PIDs, exit codes,
boundary signals, worker logs, budgets, receipts and partial evidence are kept.
Any actual hot-journal/restart failure is a test failure, never assumed repaired.
The prior exception-injection suite remains intact and is separately identified.

SIGKILL leaves the kernel and storage stack running. It does **not** simulate
power loss, controller cache loss, torn sectors or dishonest fsync. None of those
guarantees is certified by this package. There is no automatic database repair,
checkpoint rollback, attempt retry or budget expansion.

## Private filesystem and runtime requirements

- Use a dedicated, non-root service UID and an owned mode-0700 scratch/budget
  parent, `umask 077`, directories 0700 and state/receipt files 0600. Review ACLs
  as well as mode bits; no other writer or default ACL may grant access. Do not
  use symlinks, junctions, writable shared mounts or a synchronized folder.
- Proposed native evidence scope is a private local ext4 or XFS filesystem.
  The command checks the mount type and ownership/mode. Verify actual mount,
  namespace, device and volume identity independently; an ext4 label alone is
  not evidence of controller durability. NFS/SMB/FUSE/overlay and different
  filesystem behavior require separate review. All budget members, journal,
  pending exports and retained files must stay on that same filesystem.
- Require working advisory file locks, exclusive hard links, file fsync and
  directory fsync. The permanent lease file is never removed or stolen based on
  PID/age. All cooperating admissions must use the same authoritative root.
- SQLite uses DELETE journal mode, `synchronous=EXTRA` (3), foreign keys ON,
  zero busy timeout, explicit `BEGIN IMMEDIATE` for writes and consistent
  read transactions. Journal-mode drift is rejected without conversion. Record
  the SQLite library version/compile options and interpreter hash. Do not alter
  SQLite or package versions to make acceptance pass.
- Use a pre-established offline host/network namespace with no external egress,
  read-only reviewed source and a writable private scratch volume. Keep research
  storage and pretrained caches unmounted. Inventory does not create the
  namespace or change host security policy. Verify that policy separately.

## Required external anti-rollback record (not implemented)

Local source hashes, SQLite transaction checks and audit hashes cannot detect a
coherent restoration of an older database plus matching files. Before any
production execution, provision a separately administered append-only authority
outside the snapshot/restore domain of the budget volume. It must provide a
monotone compare-and-swap sequence and an authenticated acknowledgement; a second
JSON file on the same disk or in the same backup is insufficient.

Each record must bind authority/stream ID, sequence, prior record digest, database
identity UUID, plan digest, source-lock digest, deployment/volume identity,
attempt number, run/spec/decision digest, attempt state, consumed-slot high-water
mark, local audit event count/tail digest, canonical terminal receipt digest when
present, and a fencing/ownership generation. Retain an authenticated authority
acknowledgement and operator/reconciliation decision identity. Reject replay,
missing records, unavailable authority and any sequence/high-water regression.

A reviewed future adapter must reserve an external admission intent with CAS,
commit the local reservation, then acknowledge the exact local state externally
before allowing execution. An unacknowledged intent blocks later admission and
remains consumed pending explicit reconciliation. Terminal receipt commitment
must likewise be acknowledged externally before production completion/reuse can
be accepted. Cross-system partial failures remain blocked: this is not claimed
to be a distributed atomic transaction. Current `PersistentBudget` has **no**
external-authority integration; no production anti-rollback claim is made.

## One release checklist

| Gate | Reusable evidence and exact remaining requirement | Current disposition |
| --- | --- | --- |
| Stage A / frozen B1 | Reuse dated R3/R4/R5 bounded acceptances in the frozen documents. Historical pending wording and execution-time false closure markers remain unchanged. Existing comparable B1 remains the reference; no automatic retraining. Superiority and unconditional readiness remain uncertified. | Preserved; do not reopen or repeat acceptance. |
| Source and native runtime | Verify independently supplied ZIP/lock hashes, fresh read-only inventory, exact target image/distro/glibc/architecture and package identities, private filesystem/ACL/egress controls, then the full offline suite and process-kill evidence. Review nonzero failures/skips and all warnings. | Native Linux full-suite acceptance pending. Local partial evidence is not promotion. |
| Production adapters and runner | Bind genuine frozen input/supervision identities, train-only statistics, three-class metrics, validation-only selection, actual optimizer/scheduler/RNG/update reports, permissions and authoritative budget/external record to the reviewed interfaces. Review typed checkpoint import/export and completion consumers. Test explicit same-state resume without rollback and all distributed acknowledgement failures. | Adapters, external anti-rollback integration and execution authorization remain blockers. |
| Dependencies / image closure | Reuse the successful 14-request discovery record (run 37272190686, user-established history) and seven-sidecar proposal as scoped discovery evidence. The sidecar compatibility run 37305200076 is archived transport/parser synthetic evidence; its bounded acceptance document is a draft, not blanket acceptance of this model runtime. Obtain reviewed recursive metadata closure, root constraints, runtime/test/build scopes, setuptools/pkg_resources resolution, exact image/glibc/wheel tags and security/support disposition. 394 nominees/eight references are alternatives, not a lock. | Recursive closure and full Docker acceptance pending. No indexes, sidecars, binaries or packages fetched here. |
| Pretrained compatibility / provenance | The implemented R50 hybrid has its own topology; it is not the old custom CNN or a renamed plain R50 control. Reuse current synthetic topology/initialization tests. Require exact artifact digest, publisher/release/terms, semantic key/shape/dtype/layout mapping, preprocessing and leakage/overlap review for each proposed initialization before acquiring or loading it. | No pretrained bytes acquired/loaded. ImageNet and especially RadImageNet gates remain unresolved where evidence is missing. |
| GPU feasibility / scientific release | After separate permission, verify target GPU/runtime/memory, device RNG and deterministic resume, numerical behavior, full-batch/partial-batch feasibility and actual budget accounting. Preserve macro-F1 primary, accuracy/micro-F1 secondary; separate complete-method gains from component attribution and historical-test exploration from unused independent confirmation. | GPU pilots and Stage B paused. No GPU-hour estimate or scientific gain inferred. |

The supplied release review contains exact commands with the actual lock digest.
Successful tests are a review input; all authorization records remain inactive.
