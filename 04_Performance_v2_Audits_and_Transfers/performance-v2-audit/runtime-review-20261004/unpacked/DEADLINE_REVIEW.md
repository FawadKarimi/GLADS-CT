# Aggregate kill deadline correction

Date: 2026-09-28. Preparation and mocked testing only. Execution approvals false;
production pins and Stage A unchanged; Stage B paused.

Parent archive SHA-256:
`194e103548a0fe4167e377a85e4c7a9d2af045fe8038429054fb34fced8b1d99`.
The generator independently verifies all 87 parent ledger entries and restricts
edits to the three timeout/caller source files. Existing tests, hierarchy,
mount lifetime, persistence/finalization code and all historical artifacts are
preserved. New tests and this review describe the present change.

## Correction and regression evidence

kill_members starts its absolute monotonic deadline before the initial slice
identity query. That same deadline covers inventory, every unit show (including
unrelated units), the pre-signal identity recheck, and the signal command.
command_until checks positive remaining time before invoking the backend, passes
only that budget, and checks for overrun after return. An overrun accompanying
a backend exception raises a TimeoutError with the original exception as cause.
Stage/operation and deadline/time details remain in timeout diagnostics.

No unrelated-unit branch bypasses the budget. Exhaustion prevents additional
queries/signals. Exactly-on-deadline completion is permitted only if no further
operation is needed; the next blocking call is refused. There are no retries.

The reported 20-unrelated-unit reproduction now fails at 10 fake seconds after
10 commands, with decreasing budgets 10 through 1. It cannot return success at
22 seconds. Other fixtures cover exhausted budget before reaching the workload,
slow inventories/rechecks, overruns at each command boundary, original-error
preservation, inherited shorter deadlines, and successful boundary completion.

## Immediate caller audit and correction

The backend previously called full terminate() from both kill_subtree and
wait_empty even though each request advertised 10 seconds. That could nest
three separate cleanup phases inside one caller budget. It now maps:

- term_service: one TERM phase, including fresh ownership verification;
- kill_subtree: one KILL phase, including fresh ownership verification;
- wait_empty: read-back/authorization/snapshot only, all sharing its deadline.

Relevant identity, service and volume commands within the snapshot inherit the
remaining caller budget; no query gets a fresh ten seconds inside wait_empty.
Direct standalone calls retain their existing per-command behavior where no
aggregate deadline is supplied. Timeout plumbing does not change hierarchy,
file identities, mount disposal or persistence success criteria.

Standalone lifecycle terminate(), including the watchdog caller, intentionally
retains three independent cleanup phases: TERM <=10s, KILL <=10s, read-back
<=10s. A TERM failure does not cancel the independent safe KILL/read-back attempt.
Its errors remain recorded even if a later read-back is empty. The watchdog
explicitly reports termination_confirmed=false when cleanup errors remain.
The existing teardown authorization already rejects any receipt with errors.

These are cooperative backend budgets with post-return overrun detection, not
an ability to preempt a blocked syscall or undo elapsed time. Native file/proc
reads and diagnostic persistence still carry the previously documented host-loss
and uninterruptible-I/O limitations. This focused correction is not a new claim
that every operation in the entire launcher has a hard aggregate wall-clock cap.

## Actual validation

See DEADLINE_VALIDATION.json and deadline-tests.log for the exact command,
Windows interpreter, elapsed time and full result. Twelve new tests use invented
monotonic time and fake command results only; no real sleeps, subprocess host
commands, Docker, cgroups, mounts, downloads or research data were used.
The unchanged original 15 regression tests remain included in the full suite.
The retained acceptance_driver.py emits its historical return-in-finally
SyntaxWarning under the existing Python 3.14 runtime; it is not altered here.

## What still prevents live launcher binding

The public live constructor/entry point remains disabled. A reviewed executable
controller/watchdog launcher, source/inventory binding, correct privileged host
bootstrap and final execution authorization are not supplied by mocked success.
Dependency candidates remain unapproved; immutable metadata, actual binary hash
verification, native/Python/Actions security and compatibility clearance remain
outstanding. Remote evidence transport and fresh zero-paid-usage gates remain
outside this correction. No historical workflow is published or dispatched.

## What still prevents Linux acceptance

No native Linux execution was authorized or performed. Actual systemd, kernel,
Docker scope placement, process races, backend timeout enforcement, quota behavior,
filesystem/directory durability and teardown must be evaluated in a separately
authorized disposable Linux environment after source/security review. Ambiguous
recovery and host-loss limitations remain; passing mocks do not close these gates.
