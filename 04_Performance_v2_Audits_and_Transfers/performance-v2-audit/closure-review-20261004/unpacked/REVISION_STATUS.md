# Supervisor persistence correction and Linux lifecycle source revision

Date: 2026-09-28. Preparation/mock testing only. All execution approvals remain
false. Stage A untouched; Stage B paused. This is **not a deployment-ready or
complete Linux backend integration**. Item 1 is corrected; item 2 has substantial
reviewable source and mocked coverage but the integration work below remains.

## Inputs and preservation

Parent ZIP SHA-256:
`28872953c4abbadaeaef4c6c40ebcc713817c266d735136a27da725ab59b591b`.
All 57 parent ledger entries are verified before packaging. The original
15 regression tests, historical workflow, production dependency pins, embedded
inspector, proposal and metadata resolver remain byte-identical. Prior review
and validation documents are historical; this document and REVISION_VALIDATION.json
describe this revision. The parent package generator is historical, not the
generator for this delta. No research checkout was changed.

## Persistence correction

Both Supervisor.run and independent_cleanup use the same corrected finalizer.
The operation/cleanup outcome is separate from evidence finalization. On-disk
reports and successful stdout output say EVIDENCE_PENDING, never VERIFIED_FAKE.
A completion receipt, written last, binds the final ledger SHA-256 and outcome.
The return value is finalized only after that succeeds. Consumers must use
validate_complete: a report, stdout line, or ledger alone is insufficient.

Report, seal, stdout or completion-write failures mark FAILED before subsequent
report/fallback attempts. A failure record and ledger refresh are attempted
independently of the original operational error. No completion is written when
an earlier persistence step fails. Primary and persistence errors remain separate.
Old provisional reports cannot become success evidence after seal failure.

If every persistence channel fails, only the in-process return may survive.
There is no guarantee of durable diagnostics after total disk/stdout failure,
SIGKILL or host loss. A receipt write/fsync failure followed by total inability
to persist failure can leave uncertain bytes: do not interpret file presence as
proof of crash durability. No success is printed on stdout. The protocol does
not promise impossible atomic delivery across disk, stdout and caller memory.

## Linux source added, not Linux acceptance

- linux_host.py: review-only native filesystem/proc identity and bounded
  subprocess streaming primitives. Constructor refuses live use. Mocked tests
  invoke selected methods without constructing a live host.
- linux_lifecycle.py: durable intent-before-mutation/observed-after-mutation
  ownership, cgroup-v2 controls/read-back, recursive census, TERM then cgroup.kill,
  populated/process/listener read-back, fixed backing/loop/mount observations,
  reverse partial cleanup, private daemon argv and executable hash checks.
- IndependentWatchdog: owner-loss/expiry/acknowledgement algorithm performs fresh
  termination checks rather than trusting the owner. It is not yet installed
  as a separately bounded Linux service.
- evidence_export.py: new-directory, bounded local copy with independent
  destination hash read-back. Acknowledgement is required before evidence-volume
  detach. This is local export, not remote artifact upload. Backing bytes are
  deliberately retained, never automatically deleted.

No live entry point is enabled. No Docker, privileged host operation, binary
download, package installation, weight/research access, publication or dispatch
was performed. All new host-effect tests use fakes/mocks; report export tests
use invented bytes in local temporary directories.

## Exact outstanding implementation and acceptance work

1. A complete Backend.request/read adapter and provisioning coordinator are
   still needed: root/runtime directory and bounded tmpfs setup, private
   containerd/dockerd transient services, small separately bounded supervisor
   and watchdog services, admission barrier, and operation helper placement.
   The new lifecycle routines are not wired into the old fake-only supervisor.
2. Incremental recovery needs a persisted-journal loader and safe reconciliation
   of intent-without-observation crash windows. Current source retains observed
   resources and fails closed; it does not claim to recover every partially
   created resource. All task-owned helpers and allocation/formatting operations
   must be enclosed in the verified resource subtree by that coordinator.
3. Complete fresh snapshots need global escaped-process/shim census, socket
   inode ownership, storage-path/no-spill attestation, inode/free-capacity checks,
   and boot/PID-start/executable identity binding to the supervisor schema.
   A local subtree census alone does not establish absence of escaped siblings.
4. Watchdog service arming before any daemon activity, independent lifetime and
   resources, restart-free behavior and crash/cancellation testing remain.
   LinuxHost's native blocking filesystem calls are not proof of hard deadlines
   under uninterruptible kernel I/O; an external watchdog/VM boundary is required.
5. Local evidence copy/acknowledgement is implemented. Remote upload receipt,
   fresh zero-paid storage gate, export of the final teardown record, runtime
   tmpfs/cgroup removal, and backing-file disposal remain deliberately absent.
   Do not claim complete host teardown. No destructive cleanup of unknown paths.
6. Actual Linux enforcement, quota exhaustion and lifecycle acceptance are
   unexecuted. Mock tests do not prove syscall, systemd, Docker, namespace, loop,
   mount or native filesystem behavior. Namespace-relative cgroup views are
   explicitly refused rather than guessed by this host-side implementation.
7. Metadata resolution, candidate artifact bytes/full hashes, immutable lock,
   native/Python/Actions security dispositions and compatibility clearance are
   outstanding. Candidates are unapproved; production pins are unchanged.

The next code-review step is the persistence fix and this explicitly partial
Linux source. Complete items 1-5 using mocked boundaries before requesting any
live Linux acceptance. Do not publish the historical workflow or dispatch.
