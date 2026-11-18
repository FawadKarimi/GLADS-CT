# Linux synthetic acceptance workflow — proposal, not deployment approval

## Latest revision

Read REVISION_REVIEW.md for subsequent independent code corrections, validation
and the precise unresolved execution blockers. Approval remains false.

## Correction disposition — 2026-09-28

Read SECURITY_REVIEW.md and build_lifecycle.py for the current amendment. Historical legacy-builder paragraphs below are superseded: named offline installation-container compute now has inspected memory/swap/CPU/PID controls and bounded exact-resource cleanup. Docker daemon pull/copy/snapshot I/O termination and hard layer storage quotas remain unresolved, explicitly reported rather than inferred from a CLI timeout. Fresh security metadata identifies affected pins; full native and Actions-transitive coverage remains pending. All approval flags remain false. Inspector package bytes and dependency pins are unchanged.

validate_review.py now requires --root --evidence --package --ledger and is strictly read-only, including on failure. generate_review.py is a separate, explicit artifact writer. Windows mocked tests do not establish Linux runtime acceptance. Complete report generation and ledger verification are separate operations.

## Evidence review (2026-09-28)

Supplied ZIP SHA-256: `041f038ccb61a51ca8af5aca03e648e7b8bbda73a0dc0726647620b8a46dd86b`.
Run: https://github.com/FawadKarimi/performance-v2-host-readiness/actions/runs/36412911139
Commit: `12bb66279168c73b65a9b06f3112c6fa7a70950b`, attempt 1, archived conclusion success.
Internal SHA256SUMS, embedded report and log equality are checked by validate_review.py.
This is independent local consistency checking of supplied records, not a fresh GitHub API authentication of the run or billing.

Archived host: Ubuntu24.04.5 x64, image20260920.314.1, kernel6.17.0-1022-azure, Docker28.0.4, cgroup2/systemd, AppArmor/seccomp/cgroupns; five required resource capability flags true, no Docker warnings; 14,348,578,816 free disk bytes, two logical CPUs, 8,128,876 KiB total RAM, 3 GiB host swap. Host swap is not a container swap allowance. All acceptance/security/pilot flags remain false. The VM is ephemeral and is not reused.

## Transport and trust

Proposed isolated PRIVATE repository remains FawadKarimi/performance-v2-host-readiness. No publication occurred here. Put the unchanged reviewed ZIP at reviewed/inspector.zip, SHA-256 `2f97fdaf4dd7e3dd5a04255aa62a7dd7b603fcda0fd82ffcb026b3957407d45e`. The ZIP is only 236,982 bytes and contains no weight binaries. Commit binary bytes without filtering. Publish the reviewed driver, an explicitly approved synthetic-approval.json and workflow (.github/workflows/linux-synthetic-acceptance.yml) only after separate authorization. The driver checks the full archive before materializing only named build/launcher inputs and validates their internal hashes. No original inspector member is edited.

Checkout v4.2.2 commit `11bd71901bbe5b1630ceea73d27597364c9af683` and upload-artifact v4.6.2 commit `ea165f8d65b6e75b540449e92b4886f43607fa02` were resolved through official repository tag APIs during preparation. This verifies immutable identities, not exhaustive action security review. Approval of these actions and refresh of their advisories is pending. Only contents:read; no persistent checkout credentials, extra secrets, caches, research data or weights. Standard literal ubuntu-24.04 only. The platform token is not forwarded into containers.

## Acquisition/build/runtime proposal

Exactly 13 wheel artifacts, totaling 224,391,547 bytes, are acquired from the lock's official HTTPS URLs with size/SHA checks, bounded downloads and no retry; redirects must remain on allowed publisher hosts. No pip resolution, DEFAULT weights or apt update. Base is the unchanged Python3.10.17 Debian12 amd64 digest from environment-lock.json; Docker verifies registry layer digests and the driver verifies the pinned image config identity. Downloading/building happens only in a future explicitly authorized job.

The unchanged Dockerfile installs those wheels offline. Legacy Docker build is explicitly selected to enforce its memory/swap/CPU flags; network=none, no pull during build, no build cache reuse. If that builder is unavailable, fail; do not silently switch builders. Build needs separate review: legacy build does not offer the inspector's per-container PID/seccomp profile, and CLI timeout alone does NOT guarantee daemon-side build termination. This is a concrete deployment blocker, not a waived guard. The current draft is intentionally gated by native_build_execution_reviewed=false. A reviewed bounded build isolation/lifecycle solution is required before enabling it; this must not modify inspector runtime safeguards or dependency pins. No claim is made that a restricted research-loading receipt has been approved.

Runtime uses unchanged launch_synthetic.py and launch.py: no weight mounts, no network, read-only root, uid65534, capabilities dropped, reviewed seccomp/no-new-privileges, 4GiB memory with no extra swap, one CPU, 16 tasks, tmpfs bounds, 120-second per-scenario limit, 64MiB capture limit, independent bounded cleanup and termination evidence. It runs pipeline, deny-network, deny-fork, deny-root-write, forced-failure, overflow, timeout once each. Failure scenarios must match their intended cause, not merely exit nonzero. Unexpected failure stops later scenarios; no retries or repairs. Cgroup resolution runs inside the synthetic inspector on the fresh host. Complete observations still require independent review and never set Linux acceptance/security clearance true automatically.

## Resources, billing and failure retention

Runtime is UNMEASURED. Planning allowance 10–20 minutes, hard job cap25 minutes (21-minute work step) is not a benchmark or promise. Seven unchanged runtime caps total at most14 minutes before cleanup; acquisition/build add bounded time. Global timeout may interrupt work earlier. Require at least30 included Linux minutes remaining immediately before dispatch, allowing rounding/overhead, and an enforced account-wide zero-paid-usage block. Billing remains owner-provided and must be refreshed; no paid fallback.

Require >=5GiB available host RAM and >=10GiB free on both output and Docker filesystems; these are conservative proposed operational floors, not exact measured peak needs. Wheel downloads are ~214MiB; expanded torch, image layers, build copies and stopped containers need additional space. Do not delete runner preinstalled software to make room. Readiness archive's ~13.36GiB free does not guarantee the next runner meets this floor.

Failure reports are uploaded with always(), unique run/attempt artifact name and one-day retention. Need: ephemeral VM would otherwise lose launch reports and bounded container logs. No package, wheels, image layers or research records are uploaded. Worst-case seven64MiB scenario logs plus two16MiB build/pull logs ~480MiB before small JSON files; reserve at least490MiB available included artifact storage and include concurrent usage. Default expected logs are much smaller; overflow data compresses well, but do not depend on compression to guarantee zero cost. Existing host workflow has no artifacts. If storage allowance cannot cover this, block execution and review a smaller retention/export policy instead of silently truncating evidence. Artifact upload uses GitHub's job runtime token, no added secret. Cancellation, host loss or upload failure can prevent retention; missing reports are never a pass. Manual download/preservation before one-day expiry is required.

## Security and next action

Existing JIT affected-version disposition/monkeypatch remains a guard, not proof of non-reachability. Native Debian/Python/build-tool advisory review is still pending. Review must cover executing the pinned image even with invented tensors, not just real deserialization. RadImageNet remains blocked. Stage A unchanged; Stage B paused; pilots unauthorized. Actual weights and research data never enter this workflow.

Next action: review this draft and resolve bounded build isolation/termination and native-package/advisory disposition, then authorize preparation of a hash-bound approval receipt. Current false template must NOT simply be toggled to bypass these blockers. Do not publish/dispatch this draft until those issues and zero-cost artifact/minute gates are resolved. A later approved receipt must bind driver SHA, package SHA and an identifiable decision. Production loading approval remains false. No change to reviewed inspector or dependency pins is proposed.
