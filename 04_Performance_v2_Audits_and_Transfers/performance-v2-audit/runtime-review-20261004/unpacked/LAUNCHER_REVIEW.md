# Native launcher and inventory binding preparation — 2026-09-28

Status: source preparation only. All execution approvals remain false. Stage A
is unchanged, Stage B paused, pilots unauthorized. Nothing in this package is
authorization to activate native primitives. No historical workflow is enabled.

Base ZIP SHA-256:
`737f7a8c814592378c27f7240aea5be2ab20affb9b4310dfe52ebe615f9c0df7`.

## Implemented interfaces and limits

`launcher-sources.json` binds all eleven controller/watchdog/helper and imported
runtime source files to exact proposed `/opt/pv2-native-acceptance/` paths, byte
sizes and SHA-256 values. It is generated separately; its own hash is not inside
itself. Authorization binds the SHA-256 of canonical JSON (UTF-8, sorted keys,
compact separators), both for this manifest and the complete inventory. The
file ledger additionally binds raw JSON bytes. Python is explicitly
`/usr/bin/python3 -I -B`; controller minimum remains Python 3.10, separate from
the frozen inspection interpreter. `command()` constructs three real parser
interfaces, not arbitrary caller-provided shell strings. The new integration
test uses these interfaces in the existing coordinator instead of invented argv.
Historical tests remain byte-for-byte unchanged.

The dispatcher validates scope, source/inventory hashes, boot/run identity,
short authorization validity, literal approval booleans and explicit denial of
B/pilots. Its injected fake adapter is inaccessible through the command line.
The CLI rejects live activation even with valid authorization, prints a failed
JSON receipt, and exits 2. Existing native constructors remain disabled.
This is intentional: it must not misrepresent a parser and mocked dispatch as
a completed privileged bootstrap. The stdout rejection is not a durable sealed
execution report; an external collector must retain it if the CLI is examined.

The accepted hierarchy, deadlines, watchdog algorithm, storage lifetime,
termination authorization and evidence-finalization code are unchanged. The
finite helper defines success/overflow payloads; timeout is simulated, never a
real sleep or fork. No test runs a native command.

## Exact remaining activation work (not hidden by the mock result)

An authenticated owner authorization file must be delivered through a trusted,
root-owned channel; plain JSON hashes are integrity bindings, not signatures.
Before activation, implement/review the bootstrap which verifies root ownership,
non-writability and absence of symlink components for the whole source tree,
manifest, inventory and authorization; resolves and rehashes executable bytes;
records interpreter/stdlib/shared-library dependencies; reserves a token without
reuse; and rechecks boot identity. Current source verification is local file
comparison, not TOCTOU-proof code loading. The launcher imports its small binding
module before validation; this therefore requires a trusted immutable bootstrap,
not execution of an untrusted downloaded directory.

The live bridge still needs to create the bounded controller service through the
manager; reconstruct watchdog lifecycle state in a separate interpreter using a
durable bootstrap record before daemons start; publish/observe arm and export
acknowledgment; connect controller completion to actual export/final teardown;
and collect partial evidence after controller loss. The current fake kernel
manufactures the watchdog arm record: it does NOT exercise separate-process
startup. Existing recovery/teardown code is available but not silently enabled.
Native acceptance cannot be dispatched until this bridge is reviewed. This
package supplies concrete interfaces and gate semantics, not proof of that
unimplemented native bootstrap. No hash-bound authorization is supplied here.

## A — proposed bounded native lifecycle acceptance (NOT EXECUTED)

Only a fresh disposable Linux x86-64, systemd, unified cgroup-v2 host is in this
proposal. Cgroup-v1 inspector work is historical, not substituted here. No
container/image build, package installation, registry pull, weight or research
input is allowed. Private Docker/containerd services use already provisioned,
separately approved host binaries. The synthetic helper runs as a native unit.

One reviewed scenario per unique token, no retries: success, helper-output
overflow, helper timeout, owner disappearance, partial provisioning, and
interrupted teardown/recovery. Each scenario needs a distinct authorization and
output directory. Never run scenarios concurrently. Preserve the policy:
workload 6 GiB, swap 0, 1.5 CPU, 256 tasks; controller/guard independently bounded
256 MiB, swap 0, .25 CPU, 16 tasks; data 8 GiB, evidence 512 MiB, runtime tmpfs
64 MiB. Workload/watchdog lifetime 1200 seconds, operation 60 seconds, output
1 MiB, TERM/KILL/read-back separate 10-second budgets, export 60 seconds/512 MiB.
Systemd-reported ControlGroup paths, not inferred slice paths, remain authoritative.
Reserve at least 12 GiB dedicated free host storage plus an independently bounded
external evidence destination before provisioning; reject other host consumers
where these bounds cannot be assured. No claim of hard preemption of blocked
kernel/filesystem operations is made.

Acceptance requires original ownership/journal/ledger records, snapshots before
and after operations, independent watchdog identity/arm, timeout/error details,
fresh termination receipts, export manifest/read-back acknowledgment and final
completion receipt validated by `validate_complete`. Failure must retain the
primary error separately from persistence/cleanup errors. Never discard backing
storage without fresh bound termination authorization and export acknowledgment.
Unconfirmed cleanup means failure and retained resources, not fabricated success.
External watchdog/collector must survive controller loss; host loss can prevent
all persistence and remains an explicit limit. Transfer completed evidence only
after offline ledger verification; receipt of copied bytes is not Linux acceptance.

Proposed role command after future bridge approval, NOT EXECUTED and presently
guaranteed to reject (the enclosing controller unit bootstrap is still missing):

```sh
/usr/bin/python3 -I -B /opt/pv2-native-acceptance/native_launcher.py --role controller --token 0123456789abcdef0123456789abcdef --scenario success --source-manifest /opt/pv2-native-acceptance/launcher-sources.json --inventory /run/pv2-authorization/inventory.json --authorization /run/pv2-authorization/authorization.json
```

Do not submit this as a GitHub workflow yet. No runnable workflow or activation
authorization is included. Concrete watchdog/helper argv use the same interface
with `--role watchdog` / `--role helper`, emitted by `command()`.

## Outstanding host/runtime and security evidence for A

No actual current host inventory is asserted. Required evidence:

- Exact OS release/image identifier, kernel/build/architecture/boot ID, Python
  interpreter version and bytes, stdlib and native linkage inventory, systemd
  version and manager properties. Resolve every path in `EXECUTABLES` to actual
  canonical bytes, publisher/package identity, version and hash. Distribution
  aliases such as `/usr/bin` versus `/usr/sbin` need an explicit reviewed path
  amendment, not fallback search through PATH.
- cgroup mount/controller availability, systemd delegation/privilege policy,
  observed quotas, swap controls, branch independence, controller placement,
  loop capacity, mount/namespace/security profile and filesystem fsync support.
  Establish free bytes/inodes and exclusive bounded storage, export destination
  capacity, process visibility and authority to stop owned units only.
- Dated applicability review for exact kernel/systemd, Docker/containerd/runc,
  Python/stdlib/shared libraries, util-linux, e2fsprogs and filesystem/security
  tooling versions; package-origin/checksum evidence and any unresolved CVEs.
  Document privileged-native attack surface, daemon network/socket exposure,
  uninterruptible I/O and host-loss residual risk. A hash alone is not clearance.
- If Actions is later chosen: fresh standard-runner image/inventory and action
  commit/security records, private repository and zero-paid-usage evidence,
  independent cancellation collector and authorization. Prior host metadata is
  historical and cannot replace a fresh boot-bound inventory.

Inventory validation implemented here checks closed executable coverage and
binding fields; it does not fetch or authenticate this missing evidence, assess
advisories, or freshly inspect executable bytes on Linux.

## B — dependency installation and weight inspection remains separate

Even successful A would clear only exercised host lifecycle scenarios. Frozen
production dependency pins and embedded inspector bytes are unchanged. Candidate
upgrades remain unapproved; immutable source and binary identities, full dependency
closure, publisher-vs-independent-byte hashes, advisory/native-security clearance,
compatibility/topology tests and Linux synthetic inspector acceptance remain open.
Restricted loading, tensor storage/layout checks, classifier and optimizer checks,
weight terms/overlap and RadImageNet blockers remain applicable. B needs a new
authorization; neither A completion nor its JSON authorization grants installation,
binary acquisition or weight deserialization. No Stage A result is reopened.
