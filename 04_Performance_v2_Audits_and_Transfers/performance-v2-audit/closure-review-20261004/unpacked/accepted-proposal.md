# Performance-v2: daemon/storage containment and inspection pin-refresh proposal

Date: 2026-09-28. Status: proposed, NOT execution-approved. Stage A unchanged; Stage B paused; GPU pilots unauthorized. No implementation, dependency upgrade, Docker execution or package installation is performed by this proposal.

## 1. Verified input and preserved decisions

Canonical revised input: `performance-v2-linux-acceptance-revised-20260928.zip`, SHA-256 `c374d9c61c29f2ee08fbdb95ab8ca1277429112b67f8dfe78046a9eef4b95caf`.
Read REVISION_REVIEW.md and SECURITY_REVIEW.md first. All45 candidate ledger records matched hashes/sizes with complete file coverage. Embedded inspector remains `2f97fdaf4dd7e3dd5a04255aa62a7dd7b603fcda0fd82ffcb026b3957407d45e`. All three approval booleans remain false. Independent reported15-test results are historical attribution, not a rerun here.

Retain the revised independent inspect/kill/remove/absence attempts; earlier cleanup errors; primary-error preservation on report-write failure; stdout fallback; nonzero cleanup failures; fresh absence checks; post-cleanup ledger refresh; and portable, read-only verifier. Missing evidence remains inconclusive. Retain named offline installation and synthetic-only scenarios. Do not regress to the older corrected package or modify frozen Stage A/university environments.

## 2. Recommended build/transport amendment

### Dedicated daemon and complete ownership boundary

Use a **dedicated rootful Docker daemon on the disposable standard ubuntu-24.04 x64 runner**, not the runner's shared `/var/run/docker.sock`. Keep the named installation-container method inside that daemon. This is a proposed host orchestration change requiring later authorization for privileged transient setup; do not assume sudo, mount or cgroup delegation is available.

Create a unique run/attempt ID, transient systemd slice and service, private Unix socket, PID file, exec-root, data-root, and a dedicated containerd root/state/socket. Never connect to or stop the pre-existing Docker/containerd services. Disable bridge creation, IP forwarding and iptables manipulation in the dedicated daemon; build and inspector containers remain network=none. Host-side registry acquisition is restricted to pinned publisher identities. No daemon socket is mounted into a build or inspection container.

Require cgroup v2 for this **GitHub host supervisor**, without deleting or weakening the inspector's existing v1 support. Place dockerd, its private containerd, all shims, helper clients/downloaders, and all build/runtime container cgroups beneath the same unique parent. Configure container cgroup-parent explicitly: process ancestry alone is insufficient because systemd/runtimes can create sibling scopes. Verify actual `/proc/PID/cgroup` membership and recursively enumerate the subtree before accepting ownership. Reject any escaped child/sibling or attachment to shared containerd. Use immutable executable identities for dockerd/containerd/runc from the fresh runner inventory; an unsupported version or missing control blocks execution, not an automatic install.

Proposed enclosing budgets: memory.max6GiB, memory.swap.max0, cpu.max150000/100000, pids.max256. These provide daemon overhead and do not increase the inspector's existing4GiB/no-extra-swap/one-CPU/16-task limits. Verify the nested inspector limits independently. Fail on unavailable controllers. Parent budgets are hypotheses requiring synthetic resource tests, not measured peak requirements.

### Termination protocol

An independent host supervisor lives outside the workload subtree and has its own small bounded resources. Before starting the daemon it durably records resource identity, cgroup path, filesystem/loop identities and PID start times. Arm a separate watchdog before any pull/copy/commit; watchdog expiry does not depend on the Actions step surviving. Proposed workload lifetime20minutes, job25minutes, leaving cleanup/upload headroom. Preserve current per-operation deadlines (control10s, copy/snapshot60s, installation300s, runtime120s).

On success, failure, cancellation or watchdog expiry: stop admitting operations; preserve available diagnostics; attempt graceful service termination for10s; kill the **entire verified workload subtree** via cgroup.kill; wait at most10s for cgroup.events populated=0; verify private daemon/containerd PIDs and Unix listener absent, including PID start-time checks. Runtime processes are included even if their Docker CLI has exited. The independent workflow cleanup step repeats read-back rather than trusting the earlier claim. Failure of one check must not skip the others. Retain original and cleanup errors separately.

After processes disappear, attempt ordinary unmount within10s, verify mount absent, detach only the recorded loop device and verify absence. Do not use lazy unmount as proof of I/O completion. Uninterruptible kernel I/O, host loss or SIGKILL can defeat timely teardown; if populated=0 or unmount cannot be proved, report `daemon_io_termination_verified=false`, fail and do not reuse resources or retry. This offers bounded attempts plus fail-closed verification, NOT an impossible promise that Linux can always kill stuck kernel I/O. If the owner requires guaranteed physical teardown even in that case, standard GitHub-hosted Actions is insufficient: choose a separately reviewed disposable VM provider with observable destroy completion. Do not silently weaken that requirement.

### Aggregate storage, not individual-file limits

Propose an8GiB **fully allocated**, fixed-size loop-backed ext4 filesystem for ALL task-owned dependency downloads, OCI content, Docker layers/writable layers, containerd content/snapshots, builder scratch, daemon logs and temporary files. A separate512MiB fully allocated evidence filesystem survives workload teardown. Use a64MiB bounded runtime tmpfs for sockets/state if needed, accounted against the enclosing memory limit. Record actual byte and inode capacities, backing allocation, mount source and free blocks. Reject sparse/auto-expanding storage and spill paths outside these mounts; reject symlinks/path escapes. Configure Docker AND containerd explicitly: data-root alone does not relocate every image-store implementation.

Reserve >=10GiB actually available host disk before allocating8.5GiB. Never reclaim preinstalled runner software automatically. OCI decompression, copy, commit, temporary image duplication and logs all consume the same8GiB capacity and fail ENOSPC without growth/fallback. Original per-container tmpfs bounds remain nested inside memory accounting. Disk capacity for this refreshed environment is unmeasured; exceeding it triggers review of a new explicit capacity, not automatic expansion. No aggregate claim covers unrelated runner software or provider journaling; redirect task logs to the bounded evidence store and state this scope clearly.

Replace post-hoc `capture_output` checks with streaming drains: 64KiB chunks, a1MiB aggregate in-memory control-response limit, and bounded on-disk diagnostic tails. On overflow, record observed bytes/truncation, terminate the verified owned operation/subtree as appropriate and fail. Preserve existing runtime64MiB capture semantics; do not silently lower or truncate inspector outputs. Detect output volume exhaustion separately from the original error. Redundant small stdout evidence is best-effort only. Budget artifact upload against actual compressed size and fresh included storage;512MiB on disk is NOT automatically free under a0.5GB account allowance. If upload cannot fit, block/review export policy rather than charging or discarding evidence.

Official supporting interfaces: [Docker daemon storage configuration](https://docs.docker.com/engine/daemon/), [dockerd private socket/exec-root/containerd controls](https://docs.docker.com/reference/cli/dockerd/), [Linux cgroup v2 hierarchy, cgroup.kill and populated accounting](https://docs.kernel.org/admin-guide/cgroup-v2.html). These document mechanisms, not proof the future hosted runner permits this setup. Review implementation against its actual systemd and kernel versions.

## 3. Proposed inspection-only pin refresh

These are **candidate versions**, not a resolved/hash-locked environment or permission to upgrade. No proposed version should be called globally vulnerability-free. Preserve old lock/package as historical evidence and create a new versioned lock after review.

| Component | Current | Proposed target / decision |
|---|---|---|
| CPython |3.10.17|3.10.21, same minor for reduced compatibility drift; near3.10 end-of-support in October2026, so reject for execution after support expiry unless an explicit maintained alternative is reviewed.|
| Official base |old Python3.10.17 Bookworm digest|Candidate official Python3.10.21 slim-bookworm linux/amd64, **if published and current native inventory acceptable**; resolve manifest/config/layer digests before lock approval. No fabricated digest or mutable tag at execution.|
| torch CPU |2.10.0+cpu|2.13.0+cpu, selected to exceed the documented JIT fix floor.|
| torchvision CPU |0.25.0+cpu|0.28.0+cpu, official2.13 pairing; resolve tag to immutable source commit and exact compatible CPU wheel.|
| pip |23.0.1|26.2.1; above the seven listed pip advisory fix floors.|
| wheel |0.45.1|0.46.2, the recorded wheel-unpack fix floor.|
| setuptools |65.5.1 embedded,84.0.0 locked|84.0.0 throughout the executed installer environment; remove exposure to old embedded copy, not merely replace it later.|
| NumPy/Pillow and other locked packages |existing13-package lock|Keep current versions as initial compatibility candidates only; resolve refreshed torch metadata, Python markers and fresh advisories. Report every required transitive change before approval.|
| Native OpenSSL/glibc |3.0.16-1~deb12u1 /2.36-9+deb12u10|Require at least3.0.17-1~deb12u3 /2.36-9+deb12u11 for the cited CVEs **and** all later applicable fixes. Exact selected versions follow candidate image inventory/backport review, not those old minimum floors alone.|

Official evidence consulted2026-09-28: [Python3.10.21 release](https://www.python.org/downloads/release/python-31021/); [PyTorch2.13/torchvision0.28 announcement](https://dev-discuss.pytorch.org/t/pytorch-2-13-0-general-availability/3412); [torchvision compatibility matrix](https://github.com/pytorch/vision); [pip26.2.1](https://pypi.org/project/pip/26.2.1/); [wheel0.46.2](https://pypi.org/project/wheel/0.46.2/); [JIT affected <=2.12.1/fixed2.13](https://github.com/advisories/GHSA-rrmf-rvhw-rf47); [Debian OpenSSL fix](https://security-tracker.debian.org/tracker/CVE-2025-9230); [Debian glibc fix](https://security-tracker.debian.org/tracker/CVE-2025-4802). Availability of these releases does not establish CPU wheel size/hash or base-image contents; those remain exact metadata-resolution gates.

### Installer bootstrap and vulnerability coverage

Do not use old pip23.0.1 to install its own replacement. Propose validating the new pip wheel's full ZIP structure and RECORD, then executing its supported standalone wheel entry point with refreshed Python and isolated paths, or selecting a freshly inventoried base already containing the approved installer. Confirm the supported bootstrap invocation from pip documentation before implementation. Block if old bundled setuptools/pip could enter sys.path unexpectedly. No sdists, setup.py, VCS dependencies, online index resolution or unreviewed plugin execution.

For every wheel: official source/redirect receipt; size/full SHA; wheel tag/Requires-Python/Requires-Dist; safe normalized member paths, no absolute/parent/drive/backslash traversal, no duplicate/case-colliding paths or unsupported symlink entries, expansion/member-count limits, RECORD hash verification and entrypoint/script destination validation. Review the reachable wheel installer path despite publisher hashes; hashes establish identity, not safe metadata.

For the exact candidate base: obtain every native binary package's source name AND source version, backport revision, relevant bundled/static libraries, interpreter and installer inventory. Use scoped official Debian tracker records with all pagination covered; maintain an applicability row for each package and CVE. Review runtime imports, native codecs, SSL/crypto and extraction paths. Do not infer safety from an empty query or from the previous105-row inventory. Include future runner kernel, Docker/containerd/runc versions in dated host security assessment.

Actions: retain existing immutable checkout/upload commits as reference candidates, not approved unchanged dependencies. Obtain their package-lock and bundled dist provenance, enumerate actual shipped npm transitive versions, review current advisories and runner Node runtime. Choose a newer official action commit only if required by that review and bind its immutable SHA before publication. No invented replacement SHA. Manual trusted branch, contents:read, no stored checkout credentials, no secrets into containers, one-day reports-only uploads remain. Complete transitive review or propose a separately reviewed first-party transport; do not delete the requirement. [GitHub secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).

## 4. Compatibility and acceptance gates

The refresh changes the **disposable inspection environment only**, not research models, trained checkpoints, transforms or approved pilot recipes. New torch/torchvision may require guard-source hash changes; never just update an expected hash to obtain a pass.

Before any actual weights: compare torchvision0.28 ResNet50 source against both the frozen0.18 reference and previously reviewed0.25 definitions; review each AST/numerical attribute difference. Require identical intended topology, exact state keys/shapes/dtypes and expected storage layout rules. Test actual import order, restricted-load behavior, retained JIT guard, no-forward/no-gradient behavior, CPU-only operation, seeded classifier replay/distinction, encoder byte preservation and all four optimizer groups. Preserve deterministic tensor serialization; document any changed seeded bytes rather than claiming cross-version equivalence. Synthetic-only fixtures cover contiguous/stride/offset/extent/aliasing, malformed serialized input and report round trips. No real checkpoint is needed.

Before authorizing Linux acceptance, mocked lifecycle tests must cover daemon subtree escape, shared socket/containerd refusal, PID reuse, cancellation before resource record, CLI timeout with live daemon worker, failed TERM/KILL, independent cleanup after first failure, stale absence claims, process nonempty after kill, loop/mount mismatch, ENOSPC/inode exhaustion, mount still busy, streamed output overflow, evidence-write failure and primary-error preservation. Keep revised15 regression cases and portable verifier success/failure immutability checks.

Later, separately authorized disposable Linux synthetic acceptance must prove actual resource hierarchy, quota exhaustion and process/daemon cleanup; it is the next acceptance milestone, not a prerequisite to authorizing that very test. Missing controls stop without fallback. Security review can approve bounded synthetic experimentation only after documented residual-risk decisions; production loading approval remains false regardless.

## 5. Exact next implementation step for approval

Authorize a **preparation-and-mocked-tests-only v2 supervisor branch/candidate**, based on the revised ZIP above. Implement (a) the unique daemon/cgroup/filesystem ownership manifest and pure state-machine interfaces; (b) streaming control-output bounds; (c) kill-all/verify-empty/unmount read-back and independent cleanup while retaining all revised reporting fixes; and (d) inert metadata-only pin-resolution and coverage tooling for the proposed versions. Do not start a daemon, create a loop mount, install packages or load weights in that step. Return a focused diff and mocked tests for review.

Separately authorize any future official wheel/base bytes acquisition needed to establish the new exact lock. Until immutable base/CPU-wheel identities, full security dispositions, Actions transitive review and privileged hosted-runner feasibility are resolved, **all approval flags remain false**. Do not dispatch merely because this proposal was accepted. If a strict guaranteed teardown rather than fail-closed verified teardown is required, decide on an externally controllable disposable VM before implementing the GitHub daemon supervisor.
