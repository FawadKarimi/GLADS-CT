# Activation bridge source review — 2026-09-29

Base ZIP: `8a0365021c70f5fe0110fee2ef5ef8e209e52c232e81cd5b06aa78b1e8b68988`.
Source preparation and mocked tests only. No execution, native acceptance,
security approval, dependency approval or pilot permission is granted here.
Stage A and production pins are unchanged; Stage B remains paused.

This document supersedes only the implementation-gap statements in historical
LAUNCHER_REVIEW.md. Historical documents/reports remain unchanged, not evidence
that the new code has been run on Linux.

## Connected implementation

`native_launcher.py` now dispatches real bootstrap/controller/watchdog/helper
handlers, not a placeholder backend.launch_role. Native LinuxHost construction
requires a capability produced by source, inventory, environment and authorization
verification. The CLI cannot select an injected/fake backend. is_fake=true alone
does not activate LinuxHost. Historical injected test backends remain available
to unit tests; they are not CLI configuration or an authorization bypass.

The controller uses existing Coordinator/Supervisor provisioning, snapshots,
operation budgets and cleanup. It publishes an exclusive pre-watchdog capsule
after durable storage/branch ownership, before starting either daemon. A new
watchdog interpreter reads this capsule, reobserves boot, slice paths/inodes and
its own service identity, and arms through the existing worker. No controller
boolean substitutes for that arm. A complete capsule is separately persisted
before operation admission. Helpers use the fixed synthetic payloads; timeout
waits 65 seconds against the existing 60-second operation cap. Helper units now
also have explicit 256 MiB/zero-swap/.25-CPU/16-task limits.

The bootstrap is an independently bounded systemd service supplied by the future
owner launch, outside work/guard branches. It starts the controller with 1080s
RuntimeMaxSec, polls for at most 1100s, requests identity-bound stop if necessary,
and always attempts collection. This wrapper does not claim a hard guarantee
against uninterruptible kernel operations. Watchdog and workload keep their
existing 1200s bounds. TERM/KILL/read-back retain their independent 10s budgets.

Controller loss is not operation success. The watchdog records independent
termination and remains available for a bounded wait until the collector stops
it. The collector checks controller absence, reconstructs the complete capsule,
compares the durable journal, issues fresh termination receipts, detaches data
if still mounted, stops the watchdog writer, builds the export manifest, performs
read-back acknowledgment and calls the existing final teardown. A failed/missing
controller result stays failed even when teardown succeeds. Partial provisioning
without a complete capsule is retained for review; any available early capsule
permits independently verified termination, not speculative resource deletion.
Unpaired mutation intents remain fail-closed as previously agreed.

The bootstrap is excluded from workload escape checks only by its exact saved
PID/start-time/executable/cgroup identity. It is not granted a broad system.slice
exemption. Manager-reported work/guard paths and filesystem-bound teardown remain.

## Authorization, identity and trust

The authorization binds exact scenario (`success`, `output_overflow` or `timeout`),
the ordered role chain `[bootstrap, controller, watchdog, helper]`, source-manifest
canonical SHA-256, inventory canonical SHA-256, token, boot and a <=3600s validity
window. Old scenario-agnostic authorizations now fail. Every interpreter validates
authorization on entry. Changing scenario/role does not reuse an existing grant.
No sample file with granted approvals is included.

An exclusive, directory-fsynced token directory under `/var/lib/pv2-acceptance`
and exclusive per-role claims enforce one bootstrap/controller/watchdog/helper
per token. Failed reservation durability consumes the token; it is never removed
for retry. Children require matching predecessor records and current parent
process identity; helper admission additionally requires the watchdog claim.
Authority expires for new role admission, not for already admitted safe cleanup.

The stdlib-only launcher bootstrap verifies root ownership, no group/other write
and no symlink components for authorization/source paths before adjacent imports.
It verifies source bytes against the owner-authorized canonical manifest hash.
Python uses `-I -S -B`: no cwd, user/site package startup or bytecode writes.
The interpreter/stdlib running this initial check is a separately trusted host
prerequisite; code cannot authenticate the interpreter before it starts.

The native validator rechecks the closed source set and sizes; exact executable
and declared runtime-closure hashes; root-owned host/security evidence documents;
boot, architecture, kernel release/build, OS-release bytes, Python version and
running interpreter; canonical executable aliases; and protected path modes.
Declared versions are metadata bound to byte hashes, not inferred by running an
untrusted version command. Runtime closure completeness still requires review;
the program hashes the declared closure, not an invented automatic proof that
all potential native dependencies were found. Daemon canonical-path differences
require an explicit reviewed path amendment, never PATH fallback.

Native command dispatch resolves inventoried commands/embedded helper executables
to the approved absolute paths. Manifest and inventory hash conventions are sorted
compact UTF-8 JSON; the package ledger separately records original file bytes.
Root is trusted. This is not protection against a malicious concurrent root or
kernel: root-owned immutability and the verified interpreter are trust anchors.

## Proposed prerequisites and invocation — NOT EXECUTED

A dedicated root-owned `/var/lib/pv2-acceptance` filesystem must already exist,
at most 1 GiB with >=768 MiB available; this bounds control journals and exported
evidence outside the disposed workload filesystems. At least 12 GiB available
on /var/tmp is also required. This is a newly explicit host-layout prerequisite,
not a change to inspection dependency pins or an instruction to mount anything
during this task. Reuse/exhaustion fails closed. Native host loss may still prevent
all report channels; no software receipt proves termination after host loss.

The owner must first supply a reviewed fresh Linux x86-64/systemd/cgroup-v2
inventory and root-owned authorization. Fields beyond the previous schema include
`kernel_version`, `os_release_sha256`, `runtime_files` (absolute canonical path to
SHA-256 map), and `runtime_closure_reviewed`. The last boolean is NOT set by this
package; it represents a later independent review decision. Required fixed
security/host documents are `/run/pv2-authorization/security_review.json` and
`host_evidence.json`. Actual bytes/hashes, current inventory and security clearance
remain missing. No fabricated values are supplied.

Future bounded bootstrap command (NOT EXECUTED; authorization and readiness must
be independently reviewed first; substitute a newly approved 32-hex token in
both places and in its authorization):

```sh
systemd-run --unit=pv2-0123456789abcdef0123456789abcdef-bootstrap.service --service-type=exec --property=Restart=no --property=KillMode=control-group --property=RuntimeMaxSec=1320 --property=TimeoutStopSec=10 --property=MemoryMax=268435456 --property=MemorySwapMax=0 --property=CPUQuota=25% --property=TasksMax=16 -- /usr/bin/python3 -I -S -B /opt/pv2-native-acceptance/native_launcher.py --role bootstrap --token 0123456789abcdef0123456789abcdef --scenario success --source-manifest /opt/pv2-native-acceptance/launcher-sources.json --inventory /run/pv2-authorization/inventory.json --authorization /run/pv2-authorization/authorization.json
```

Expected retained tree: role claims/reservation/start intent, early/full capsules,
controller/watchdog results, journal/operation/watchdog/collector/final evidence,
local read-back-verified export, activation result/manifest/receipt. Missing or
false completion is failure, not acceptance. Do not delete retained evidence on
failure. Remote transfer remains an owner step; local copy acknowledgment is not
GitHub artifact delivery. No historical workflow is published or enabled.

## Remaining implementation versus evidence prerequisites

Connected role handlers, bounded controller creation/polling, serialized watchdog
reconstruction, synthetic helper dispatch, one-use chain, controller-loss collection
and export/finalization are implemented as source. Ambiguous partial provisioning
intentionally retains resources; automatic cleanup without established ownership
is not an unfinished bypass to add. There is no claimed native acceptance.

Native tests still must establish actual independent interpreter startup/races,
systemd MainPID and service state behavior, daemon startup/socket readiness,
controller-loss/cancellation timing, quota enforcement, fd confinement, durability,
storage capacity and teardown on the chosen host. Mock reconstruction is not a
real interpreter launch. Stop and retain evidence if any native assumption fails.
Those observations may reveal further implementation defects requiring review.

A also still requires exact current host/native/runtime inventory, declared
runtime closure review, dated advisory applicability and residual-risk decisions,
trusted deployment/authorization and a fresh bounded execution approval. None is
granted here. B still separately requires reviewed dependency artifacts/security,
compatibility, inspector Linux acceptance and loading authorization. Passing A
would not clear B, change pins, permit weight loading or approve pilots.

## Validation

See ACTIVATION_VALIDATION.json and activation-tests.log for actual commands,
runtime and results. Tests use invented process/clock/filesystem observations
and mocked OS boundaries, not Docker, cgroups, native services or research data.
One historical launcher fixture gained explicit scenario/role-chain fields to
match the now-mandatory contract. Prior rejection tests are preserved. Initial
development failures (error-message compatibility and Windows path fixture
semantics) were corrected without weakening production guards. The historical
acceptance_driver.py Python 3.14 return-in-finally warning remains unchanged.
