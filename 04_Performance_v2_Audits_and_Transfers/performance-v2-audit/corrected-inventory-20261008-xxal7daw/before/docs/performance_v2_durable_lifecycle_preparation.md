# Durable lifecycle preparation (review only)

2026-10-07. `pilot_execution_authorized=false`. Stage A and frozen B1 are
preserved. Stage B and GPU pilots remain paused. This additive interface neither
executes a model nor changes the protocol, optimizer, checkpoint inventory,
input bindings, selection rules or existing lifecycle evidence.

## Canonical state and admission

`PersistentBudget` binds an immutable, externally reviewed plan digest to exact
execution-specification digests and a fixed cap of at most 13 attempts. A fresh
private root is created exclusively; an existing root is never initialized over.
No API expands the cap or refunds a slot. Every start and explicit resume counts.
The corrected execution inventory validator remains on the admission path.

A nonblocking OS lock on a permanent one-byte file is held for the entire
attempt, including its terminal commit and receipt export. There is one active
attempt per budget. SQLite also enforces a unique reserved attempt, unique
initial run and unique decision. Admission commits before `DurableRun` creates
its run artifacts or accepts observations. A competing owner receives rejection;
there is no polling, timeout-based lock stealing or automatic retry.

SQLite DELETE journaling, `synchronous=EXTRA`, zero busy timeout and explicit
transactions hold reservations, terminal receipt bytes, artifact hashes and a
hash-chained transition audit. Terminal state and receipt are one transaction.
The `receipts/` files are verified exports of canonical database bytes. They
cannot independently establish successful completion. Consumers must obtain a
fresh `committed_receipt()` result before applying the existing reuse checks;
synthetic completion never qualifies for production reuse.

`DurableRun` preserves the reviewed lifecycle validation and adds directory
barriers. Completion requires stopped validation-only selection, every candidate
checkpoint, exact inventory/optimizer moments/RNG bindings and consistent
progress. The backend rechecks these retained typed records before committing.
It does not load pretrained or existing research checkpoints. There is no local
`completion.json` or `terminal.json` in an attempt directory: those names block
admission as unmanaged receipts.

## Recovery decisions

| Observed state | Permitted next review action |
| --- | --- |
| Lease still held | Reject competing admission and reconciliation, regardless of lock-file age. |
| Unowned reserved attempt | Block admission. Explicit snapshot-bound `interrupt_reserved` preserves the consumed slot; it does not guess progress or a safe checkpoint boundary. |
| Committed terminal, missing/partial/unacknowledged export | Block admission. Explicit snapshot-bound `export_committed` exports the same canonical bytes; partial mirror bytes move into `retained/`. |
| Committed complete with matching acknowledged export and artifacts | Completion may be read for subsequent review. This is not execution permission. |
| Conflicting receipt, changed artifact, audit/row contradiction, corrupt database, orphan output or changed plan | Reject admission and routine reconciliation; retain evidence for independent investigation. |

Each reconciliation supplies a fresh canonical snapshot hash and a documented
decision hash. No action converts an uncertain reservation to complete, rolls a
checkpoint back, refunds budget, retries a run or downgrades committed completion
to failed. Failure after a commit may mean the commit succeeded: the in-memory
object is fenced and a fresh inspection determines the outcome.

An ordinary recorded interruption at a verified epoch boundary can use the
existing sealed `review_resume` recapture. A fresh explicit decision and latest
parent receipt are required, and a resume consumes another slot. Reconciled
crashes deliberately have `epoch_boundary=false`; the interface cannot infer a
safe resume point from leftover checkpoint filenames.

## Verification boundaries and remaining gates

Synthetic tests use temporary roots, real local SQLite transactions/file fsync
and Windows byte-range locks, invented tensor states and mocked update counters.
They inject interruptions at transaction, mirror and artifact boundaries. They
do not kill a process or cut power. The explicitly supplied synthetic directory
barrier is a mock; it cannot be supplied for production plans. The default
directory-fsync implementation rejects this Windows host.

Before production acceptance, validate the exact native Linux runtime and local
filesystem: cross-process locks, actual process death and hot-journal recovery,
directory/file fsync, exclusive hard links, disk-full/I/O failures and restart
behavior. Read-only inspection of an unrecovered hot journal may reject until
an independently reviewed recovery procedure is performed; there is no automatic
repair command here. Network filesystems, hostile owners and filesystem rollback
are outside this prepared guarantee.

Deployment must establish one authoritative budget root, owner/ACL/mount
controls, an externally anchored plan/database identity and protection against
restoring or copying an older whole budget. A local hash chain alone cannot
detect rollback of the entire database and its files. A future authorized runner
must also bind truthful input/event reporting and actual execution authorization.
No runner, launcher, retry service or production authorization is added.
