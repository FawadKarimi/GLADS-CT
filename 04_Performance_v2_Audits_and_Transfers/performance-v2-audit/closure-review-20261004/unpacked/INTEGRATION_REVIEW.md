# Supervisor integration and termination/durability corrections

2026-09-28. Preparation and mocked tests only. All execution approvals false;
Stage B paused; Stage A and production pins unchanged. No public live entry
point and no workflow publication. This document supersedes earlier package
status descriptions only for the source changes described here.

## Verified parent

`performance-v2-supervisor-linux-revision-20260928.zip`, SHA-256
`a579bc9ba7eaff586f95a21e1cbc45c72b34fca2c63f4133896605c969279129`.
All 69 parent ledger entries independently verified. Historical reports,
proposal, metadata resolver, embedded inspector, original 15 regression tests
and dependency/security evidence remain unchanged. Their dates/statuses are
historical, not newly granted approvals.

## 1. Teardown authorization correction

One termination contract (`pv2_termination_v1`) binds exactly:
schema, literal `termination_confirmed=true`, empty errors list, run token,
boot/cgroup path/inode identity, observation time, and a random receipt nonce.
It must equal the latest receipt issued by that lifecycle instance and be no
more than 10 monotonic seconds old. A recovered process must issue a new receipt;
it cannot import another instance's cached authority. Future timestamps,
superseded results, altered fields, integer 1 and truthy failed dictionaries
are rejected. Errors are conservative blockers even if later empty read-back
succeeds. A fresh termination attempt may resolve a transient error.

partial_cleanup, detach_volume and export_and_ack enforce the same contract.
Before unmount and loop detach, they re-read boot/cgroup identity, populated
state, process census, global token-bearing processes and private listeners.
Volume identity and loop association are independently checked. Repopulation
after the receipt, or during volume inspection, prevents the destructive call.
After export, a new receipt is obtained rather than extending an old timestamp.

## 2. Integrated source paths and mocked coverage

- Coordinator: controller-resource prerequisite; incremental ownership; fixed
  volumes/runtime tmpfs; private services/configuration; admission only after
  the independently observed watchdog handshake and a complete snapshot.
- Services: private systemd service/slice arguments, resource/lifetime settings,
  explicit private containerd/dockerd paths and configurations, inventory hashes,
  service PID/cgroup read-back. The worker algorithm publishes its own identity.
- Backend.request/read: accepted supervisor state machine, bounded streaming,
  operation service below the workload slice, exact record binding, termination
  receipt generation, fresh mount/loop absence checks. CLI exit alone never
  proves daemon termination.
- Snapshots: resource controls, /proc process identities, global runtime census,
  baseline external-process identity protection, writable-FD paths, exact daemon
  argv, configuration bytes/inodes, socket inode ownership, volume/loop identities,
  bounded runtime tmpfs, watchdog identity/controls, free bytes/inodes and listeners.
- Recovery: verifies journal hashes and complete member coverage, boot/token and
  ordered intent/observation pairing. Observed resources feed independent cleanup.
  Intent without an observation fails closed; it is not guessed into ownership.
- Finalization: fresh termination, stop/read-back of the external watchdog,
  local evidence export/independent byte read-back/acknowledgment, evidence and
  runtime detach, recorded backing inode deletion, nonrecursive empty-cgroup and
  directory removal. Failures preserve diagnostics; no remote upload occurs.
  An external final sink uses the accepted provisional-report/final-receipt
  mechanism and binds the export receipt/manifest hash. It is not placed in a
  filesystem being removed.

The integrated tests run these orchestration paths against an invented in-memory
Linux OS interpreter. They do not replace the coordinator with canned success.
The fake intercepts every service/subprocess, proc, cgroup, loop, mount and file
operation. Local report/export tests use temporary invented bytes only.

## 3. Linux journal durability

FileEvidence now requires directory synchronization. On Linux, sync_directory
opens the directory with O_DIRECTORY/O_NOFOLLOW, fsyncs and closes it. Creating
the journal syncs its parent and itself. An ownership append returns only after
file flush/fsync AND journal-directory fsync. Atomic ledger replacement uses
temporary-file fsync, directory fsync, replace, then directory fsync. Completion
also fsyncs its directory; any directory-sync failure poisons completion.

Local evidence export additionally syncs the destination parent and destination
directory before acknowledgment. Unsupported directory fsync fails closed; no
Windows emulation is silently treated as Linux durability. Windows tests inject
explicit fake directory synchronizers and mock native calls for error/order
tests. A file already visible after a failed fsync is not a durable-success claim.

Host loss, SIGKILL, hardware/storage faults and uninterruptible kernel I/O remain
limitations. No userspace mechanism promises guaranteed physical teardown or
atomic delivery across disk/stdout/caller memory. Missing/incomplete records are
inconclusive. Controller operations, export and teardown receipts require review
as one linked evidence set; an intermediate report is not final authorization.

## 4. Controller versus inspection Python

Controller source requires Python >=3.10, independently of the frozen inspection
environment. annotate() uses add_note where available and a __notes__ fallback on
3.10, preserving the original exception. Python-3.10 syntax parsing is tested;
an actual Python-3.10 controller runtime was NOT executed. Actual tests use the
existing Windows Python recorded in INTEGRATION_VALIDATION.json. No interpreter,
torch/torchvision or inspection dependency was installed, upgraded or approved.

## 5. Exact remaining blockers and limits

1. Independent source review and unexecuted Linux acceptance remain required.
   Mock results do not prove systemd handling of manually established slices,
   cgroup delegation, native filesystem durability, socket/proc races, kernel
   resource enforcement, quota exhaustion, unmount or process teardown.
2. Public live construction remains disabled. Binding the internal worker to an
   executable service launcher and approved controller/source inventory is not
   enabled by this package. The caller-supplied service/operation argv used by
   tests are invented, not an execution recipe. No historical workflow is wired
   to this coordinator. A reviewed live bootstrap must place the controller in
   its separately bounded cgroup before construction; the coordinator refuses
   unbounded control processes rather than moving itself.
3. Recovery intentionally blocks ambiguous crash windows (durable intent but no
   observed resource identity), unknown/new external runtime processes and
   unreadable/racing proc observations. They require inspection, not guessed
   deletion or automatic retries. This is fail-closed recovery, not guaranteed
   reclamation after every possible interruption. TERM errors remain blockers.
4. Snapshot confinement is observational, not proof of all future native writes
   or absence of every possible host escape. Privileged setup/control operations
   run in the separately bounded controller; allocation/format helpers and
   research-free operations use the workload slice. Actual privilege, parent
   hierarchy and aggregate-storage behavior must be demonstrated on disposable
   Linux. No security clearance follows from these mocks.
5. Local export/read-back is implemented. Remote artifact transport, a fresh
   zero-paid usage/storage gate and server-side receipt verification remain
   outside this disabled source integration. No public availability is claimed.
6. Metadata resolution, actual binary verification, approved immutable lock,
   native/Python/Actions advisory dispositions, topology/compatibility clearance
   and real Linux testing remain outstanding. Dependency candidates remain
   unapproved and production pins unchanged.

The next step is independent review of this source/patch and its mocked tests,
including the intentionally disabled bootstrap boundary. Do not dispatch,
install, load weights, access research data or launch any pilot.
