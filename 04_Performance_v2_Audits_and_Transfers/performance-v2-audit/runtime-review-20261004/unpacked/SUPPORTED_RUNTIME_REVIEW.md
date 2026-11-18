# Supported-runtime candidate review — 2026-10-04

Review only. This document supersedes historical runtime-selection recommendations, not Stage A scientific files. Stage B remains paused. All approvals remain false. The public native launcher now rejects before trust bootstrap or host operations, even with a historical authorization file. Mocked role-handler tests are not execution authorization.

## Proposed replacement, not a validated lock

`runtime-candidate.json` replaces the future Python 3.10 selection with Python 3.12.15 / CPython cp312, proposed Linux amd64 `python:3.12.15-slim-bookworm`. Python 3.12 receives security support through October 2028. Python 3.10 is no longer an acceptable new base. Exact OCI digest resolution failed; this tag is a proposal, not an acquired or approved image.

Propose torch 2.13.0+cpu with torchvision 0.28.0+cpu, pip 26.2.1, wheel 0.46.2 and setuptools 84.0.0. The official PyTorch previous-versions page documents the 2.13.0/0.28.0 CPU pairing. The resolver now selects cp312 Linux wheels and rejects cp310 and non-CPU substitutes. No package or image binaries were downloaded, installed or loaded.

The old `reviewed/environment-lock.json`, `reviewed/base-inventory.json` and embedded inspector archive remain byte-identical historical evidence. Their cp310 hashes, native inventory and old runtime must NOT be used as the new candidate's lock. No production executable dependency lock has been approved. A complete cp312 transitive closure, exact artifact URLs/sizes/hashes and bootstrap method are still required. Publisher hash assertions must remain distinct from independently hashed artifact bytes.

## Official evidence and bounded security assessment

Sources consulted on 2026-10-04 (metadata only):

- https://www.python.org/downloads/release/python-31215/ — supported security-release line and October 2028 horizon.
- https://pytorch.org/get-started/previous-versions/ — official torch 2.13.0 / torchvision 0.28.0 CPU pairing.
- https://pypi.org/project/pip/26.2.1/ and https://pypi.org/project/wheel/0.46.2/ — proposed package releases, not byte verification.
- https://github.com/advisories/GHSA-rrmf-rvhw-rf47 — CVE-2025-3000 JIT issue: affected through 2.12.1, fixed 2.13.0. This fixes that documented version range only, not all native security risks. The JIT guard remains defense in depth, not proof of non-reachability.
- https://github.com/advisories/GHSA-qwm4-qh6w-59xr — pip CVE-2026-13346 fixed at 26.2.0; 26.2.1 meets that floor.
- https://github.com/advisories/GHSA-8rrh-rw8j-w5fx — wheel CVE-2026-24049 fixed at 0.46.2; the proposed version meets that floor.
- https://security-tracker.debian.org/tracker/CVE-2025-9230 — bookworm OpenSSL fixed floor 3.0.17-1~deb12u3; historical 3.0.16-1~deb12u1 is below it.
- https://security-tracker.debian.org/tracker/CVE-2025-4802 — bookworm glibc fixed floor 2.36-9+deb12u11; historical 2.36-9+deb12u10 is below it.

The exact-version metadata resolver was attempted once and returned partial/exit 1: all HTTP requests were refused by local socket permissions. Its original failure receipt is retained in `metadata-refresh-20261004/metadata-resolution.json`. No resolved artifact identities or negative advisory findings can be inferred from this failure. The web checks above are selected advisory checks, NOT exhaustive current clearance. Recheck serializer CVE-2025-32434 / CVE-2026-24747, JIT and native dependencies against the eventual complete lock, alongside pip/wheel/setuptools, torchvision, every transitive wheel and pinned Actions' bundled JavaScript/runtime. Do not bootstrap with the historical vulnerable pip merely to install a newer pip.

## Debian inventory: explicitly incomplete refresh

`native-inventory-review.json` carries the 105 historical binary rows as historical-only comparison records, each marked pending replacement-image verification. It is NOT an inventory of the proposed image. Available Debian fixed versions do not establish installed versions. The new exact OCI manifest/config/layer identities, package database or hash-bound SBOM, binary-to-source mappings, Debian backport dispositions, architecture, repository origin and support coverage remain unavailable. They cannot be supplied by relabeling the old inventory. Package/image downloads are prohibited in this task; no new installed inventory is claimed. Current selected advisory findings above refresh the known risks while full inventory/advisory closure remains a blocker.

## Compatibility review still required

Resolve the candidate OCI amd64 digest and immutable torchvision 0.28.0 commit; verify each artifact and complete dependency closure. Check Python ABI and wheel platform/glibc requirements. Compare new ResNet50 topology, keys/shapes/dtypes/storage layout and forward-independent loaded-state equality against the frozen reference; verify seeded head replacement and exhaustive four optimizer groups using authorized synthetic tests first. Preserve restricted CPU loading, tensor/security checks, no forwards and no fallback. Official pairing does not validate the inspector or acquired weight compatibility. No weights were accessed. RadImageNet remains blocked.

## Daemon-I/O and aggregate storage: retained blockers

CLI timeout/kill is not proof that Docker daemon pull/copy/commit I/O stopped. The reviewed private-daemon ownership, separate controller/watchdog hierarchy, one aggregate deadline, independent cleanup, fresh termination read-back and export-before-disposal remain unchanged. Linux acceptance must prove daemon/containerd/helper containment and cancellation, including controller loss. Uninterruptible kernel I/O may prevent timely termination: report unconfirmed termination, retain resources/evidence, and require externally verified host disposal rather than claiming success or detaching live storage.

The proposed bounded allocations remain 8 GiB data, 512 MiB evidence, 64 MiB runtime tmpfs and a separately bounded state filesystem. Initial /var/tmp admission is 12 GiB; child admission uses ownership-verified allocations and the reviewed remaining-space rule, not a repeated initial threshold. Those are source/mocked guarantees only. Native tests must prove all Docker/containerd data, downloads, temporary files, logs and evidence stay on the intended bounded filesystems; verify backing allocation and no spill, ENOSPC, inode exhaustion, cancellation and teardown. Account separately for host journal and other external writes; neither a loop-volume capacity nor free-space check alone proves an aggregate host-write quota. No such native proof exists here. Do not activate the historical build method or workflow as a workaround.

## Exact next step and gates

Review this supported-version/ABI change and disabled entry point. Next preparation may resolve metadata through an authorized channel, then propose a complete immutable lock and native inventory acquisition procedure for separate approval. Binary acquisition, dependency installation and native acceptance are NOT authorized by this package. Host/runtime inventory, security decision, aggregate storage/I/O acceptance and hash-bound execution authorization remain prerequisites. Passing the Windows mocks does not clear any of them.
