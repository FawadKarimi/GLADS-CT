# Supervisor v2 preparation — 2026-09-28

Status: preparation and mocked testing only. This is not a Linux execution implementation or an approval to deploy. Stage A unchanged; Stage B paused; all execution, security, production-loading and pilot approvals remain false.

## Verified authority/base

Owner-provided PROPOSAL.md SHA-256: d850279dbf6e13e5f774d174e6a3a6455e4073877cd3c6ed899fbef66b995d74.
Revised base ZIP SHA-256: c374d9c61c29f2ee08fbdb95ab8ca1277429112b67f8dfe78046a9eef4b95caf.
Both verified before work. The base's45 ledger entries and46 ZIP members were checked before copying into this external candidate. No research repository was edited. The revised15 regression tests and their implementation files are byte-preserved; old reports remain historical. Original base ledger is retained under history/revised-base-file-hashes.json; the root ledger covers the complete extended candidate.

## New implementation boundary

supervisor_v2.py implements validated ownership records and a deterministic state machine: NEW -> RECORDED -> ACTIVE -> CLEANUP -> VERIFIED_FAKE/FAILED. No subprocess, daemon, socket, mount or cgroup writer is implemented. A Backend protocol must provide bounded observations/operations. This candidate accepts only explicitly fake backends and contains no live backend, CLI bypass or workflow wiring to enable one. The legacy workflow is preserved as historical, unapproved code, not the supervisor's deployment entry point. Never publish it as the new implementation.

Ownership records bind run/attempt/token, boot and cgroup inode, process PID/start time/executable hashes, private socket/PID/exec/data/containerd/download/scratch/log paths, private service/network settings, watchdog identity, external-supervisor membership, fixed filesystem capacities, allocation, inode capacity, UUID/device/mount IDs and exact loop/backing identities. Shared sockets/containerd, path escapes, PID reuse, resource drift, sparse backing and observed spill paths fail closed. Process census must include escaped owned processes, not merely list descendants; the backend contract explicitly requires complete independent read-back.

Admission requires a durably persisted record, matched current snapshot and independently armed watchdog. The state machine admits one opaque bounded fake operation at a time; it does not implement acquisition/build orchestration. The future backend is responsible for actual bounded provisioning and Linux observation, cgroup resolution and independently scheduled watchdog enforcement. Parent6GiB/zero-swap/1.5CPU/256tasks,8GiB work storage,512MiB retained evidence and64MiB runtime tmpfs remain proposed limits; inner inspector4GiB/zero-extra-swap/1CPU/16tasks and existing seccomp guards are unchanged.

Streaming responses merge stdout/stderr through the backend with <=64KiB reads and a1MiB aggregate retained cap. Timeout/overflow fails into subtree cleanup; no post-hoc unlimited capture is used by the new supervisor. Real backend reads MUST be interruptible/bounded; checking elapsed time after return cannot stop an uncooperative real implementation. Oversized backend chunks are rejected, but the fake interface is not a proof of a live transport's allocation bounds.

Cleanup independently attempts close-admission, TERM, subtree KILL and fresh empty/process/listener read-back. Each destructive attempt requires current ownership identity. An identity mismatch prevents mutation of potentially unrelated resources; earlier control failures do not skip later safe observations. Work-volume unmount and loop detach require termination and exact current identities; busy/mismatched resources are retained with failure, not lazily unmounted. Evidence volume deliberately remains attached for later export; no claim of total host cleanup is made. A future export/acknowledged evidence-volume teardown phase is still required. Independent cleanup ignores prior success claims. Report/ledger/stdout failures remain separate from primary errors.

supervisor_evidence.py writes only a caller-selected NEW local report directory. Exclusive writes, flush/fsync, final report hashes, atomic ledger replacement and bounded stdout fallback support report preservation. No report can be guaranteed after disk/stdout failure or host loss. A nonempty/unreadable subtree, uncertain daemon I/O, or missing evidence is not a pass.

## Metadata-only tooling

metadata_resolution.py has an explicit --fetch-metadata opt-in and a separate injected transport used by tests. This task did NOT invoke its HTTP mode. It can query exact-version PyPI JSON, the official PyTorch CPU indexes, OCI manifest/index JSON, the read-only official Docker registry token endpoint, torchvision tag metadata, GitHub advisory pagination and hash-pinned Actions package-lock JSON. It refuses wheel/archive/OCI blob/layer endpoints, non-HTTPS or credential-bearing URLs, unapproved redirects and oversized responses. No installer, model import, package resolver execution or binary acquisition exists.

Candidate targets remain Python3.10.21, torch2.13.0+cpu, torchvision0.28.0+cpu, pip26.2.1, wheel0.46.2 and setuptools84.0.0. These are proposal inputs, not a new lock. The tool records each public metadata response's bytes/hash/UTC/URL; registry tokens are neither persisted nor reported. Publisher wheel/config/layer digests carry artifact_bytes_verified=false. Only downloaded metadata itself, including child manifest bytes matched against index descriptors, can be independently hash-verified. Actions lockfile inventories are candidates for dist review, not proof of shipped dependency equivalence. No empty advisory response grants clearance. OCI platform ambiguity, missingCPU wheel/index hash, malformed identities, paging limits and API failures remain explicit errors.

Read-only live metadata resolution, if subsequently requested, can use:

```text
python -B metadata_resolution.py --fetch-metadata --output NEW_EXTERNAL_METADATA_DIRECTORY
```

This is NOT an execution command for Linux acceptance. No live metadata results are claimed in this review; all new resolver results in tests are invented fixtures.

## Validation and preserved regressions

Run the existing15 revised tests and all added tests with bytecode disabled/CUDA hidden in the existing local Python runtime. New cases cover all listed proposal section4 lifecycle classes: shared endpoints, escaped processes, PID reuse, failure before record, live daemon after client timeout, failedTERM/KILL, independent cleanup/stale claims, nonempty after kill, mount/loop mismatches, byte/inode exhaustion, busy mount, streaming overflow, report/ledger failure and primary-error retention. Tests use fake clocks, invented observations and temporary report directories. Metadata tests cover corrupt/ambiguous references, binary URL refusal, pagination, no security approval from empty results and publisher-vs-byte hash distinctions. See PREPARATION_VALIDATION.json and preparation-tests.log for actual results.

## Exact unresolved items / next implementation step

1. A separately reviewed **real Linux backend and provisioning phase** is not supplied: implement systemd/cgroup hierarchy and independent watchdog creation, streaming OS transport, proc/mountinfo census, fixed-volume allocation and private daemon/containerd config. Do not replace fake observations with echoes of requested settings. Needs later privileged-host authorization and Linux acceptance; no settings changed here.
2. Verify aggregate path confinement, fixed storage behavior, ENOSPC and real cgroup/seccomp enforcement on a disposable host. Uninterruptible I/O/host-loss uncertainty remains; no provider-level destroy guarantee exists. Evidence filesystem cleanup follows confirmed export, not workload cleanup.
3. Obtain live metadata identities, then separately authorize official binary-byte acquisition for independent wheel/base hash and native inventory checks. No new package pins are resolved/applied by this task. Python3.10 support expiry and installer bootstrap remain review gates.
4. Complete native binary-to-source/backport coverage, wheel member/RECORD/entrypoint validation, Actions dist/npm/Node review and dated advisory applicability. Old affected pins retain negative clearance. No compatibility claims for proposed torch/vision until source/topology/seed/state synthetic checks are reviewed.
5. Later bind the reviewed new backend and pin refresh into a new, separately approved workflow/build receipt. Keep current approval flags false. Fresh zero-paid-usage/minutes/storage checks are required only before eventual authorized dispatch.

This package is ready for source/mocked-test review, not publication or execution. No Docker, cgroup mutation, mounts, package/image binaries, installations, real weights, research data, training or university-server work occurred.
