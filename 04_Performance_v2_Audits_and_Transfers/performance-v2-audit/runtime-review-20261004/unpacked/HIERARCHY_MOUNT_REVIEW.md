# Focused hierarchy and mount-lifetime correction

2026-09-28. Preparation and mocked testing only. Approvals remain false;
Stage B paused, Stage A and production dependency pins unchanged.

Parent ZIP SHA-256:
`c56f961ce6103877607f5608ddb6535bba800dd34d972b3e20c885b298ef0610`.
All 79 parent ledger entries are verified by the package generator. Historical
reports remain historical; this document describes this correction. The accepted
persistence finalizer, directory-fsync implementation and Python compatibility
fallback are unchanged. No live Linux operation or dependency acquisition ran.

## Systemd ownership correction

The previous hyphenated control slice was a descendant of the workload slice.
The replacement uses separate manager-owned transient units:
`pv2work<TOKEN>.slice` and `pv2guard<TOKEN>.slice`. The already bounded controller
must be in its own systemd service outside the workload branch. Source code
obtains paths from systemd ControlGroup and verifies them against /proc and
recorded inode/boot identities. It does not derive authority from a unit-name
string or a fake assumed cgroup path.

systemd exclusively creates and removes these slices through StartTransientUnit
and stop operations. Memory/swap/CPU/task properties are passed to the manager
and independently read back. No mkdir, resource-control writes, subtree-control
writes, cgroup.kill write or rmdir is performed on systemd-owned cgroups.
Termination enumerates services/scopes, selects only manager-reported descendants,
rechecks each path, and signals those units through systemd. It retains deadline,
empty-census/listener checks, fresh termination receipts and error separation.
Final slice stop requires the original inode and verifies manager inactivity and
native path absence. The external watchdog is not selected by workload kills.

The infrastructure-only daemon setting changes from cgroupfs to systemd; Docker
receives the workload slice unit as cgroup-parent. Runtime scopes therefore use
systemd's supported management model, not direct manipulation of the slice tree.
No delegation is requested on a slice. The inspection scientific protocol and
dependency pins are unchanged; future container commands must retain this parent.

Official interface basis reviewed during this task:

- [systemd cgroup ownership, hierarchy and delegation](https://systemd.io/CGROUP_DELEGATION/)
- [systemd transient slice support](https://github.com/systemd/systemd/blob/main/docs/TRANSIENT-SETTINGS.md)
- [Docker systemd cgroup-parent rules](https://docs.docker.com/reference/cli/dockerd/)

These documents justify the interface choice, not compatibility or deployment
clearance for any particular installed host version. Two freedesktop manual-page
fetches returned HTTP 403; no content was inferred from those failed fetches.

The integrated ownership record is now pv2_ownership_v3 and binds the workload,
guard and controller unit/path identities. Historical v2 fake records remain
usable by their retained unit tests; the integrated coordinator emits v3 only.
Old journal records are not silently migrated into new slice ownership.

## Filesystem lifetime correction

Directory, mountpoint, backing and configuration identities now record the
containing mount (ID, device, mount/root path, filesystem type and source).
They explicitly distinguish host entries from entries internal to the data,
evidence or runtime filesystem. Mountpoints are recorded before mounting, so
their identities are the underlying host directory identities.

After verified unmount and export-before-disposal, internal data/tmp and
data/downloads entries are retired with their disposed filesystem. They are
not statted or removed using an old inode against the uncovered host path.
Host mountpoints and backing files still require their original inode AND
containing-filesystem binding before nonrecursive removal. Replaced mountpoints,
unknown filesystems and nonempty host directories fail closed. Backing removal
still requires loop absence and a fresh termination receipt.

## Mock improvements and regression scope

The fake expands foo-bar-baz.slice into its documented hierarchy. It demonstrates
the previous nesting error, and proves that workload signalling leaves the
independently bounded watchdog and external controller alive. It rejects manual
writes/removal in manager-owned cgroups. Unit stop and cgroup absence are modeled.

The fake preserves covered host mountpoints, exposes distinct mounted roots,
hides mounted contents on unmount, and raises for missing paths/nonempty
directories. Tests cover complete finalization, forbidden lookups of hidden
children, replacement inode/path refusal, interruptions before evidence unmount
and before backing disposal, and preservation of unrelated host children.
No successful test response is native Linux acceptance.

## Remaining implementation/acceptance gaps

- Public live entry and controller/watchdog executable launcher binding remain
  disabled. No historical workflow is published or connected to this candidate.
- Actual systemd transient-property support, Docker scope parent propagation,
  systemd unit garbage collection/stop semantics, cgroup quotas, process races,
  mount/unmount behavior and directory durability require separately authorized
  disposable Linux acceptance. Unexpected unit disappearance/errors fail closed.
- Interrupted teardown preserves a failed partial report and remaining resources;
  this does not add automatic resume after every irreversible deletion. Recovery
  still refuses ambiguous intent-without-observation and missing owned slices.
  No guessed deletion or automatic retry is introduced.
- Linux must independently establish that all operation/container units remain
  below the workload branch. An enumerated snapshot is not a security proof of
  all future privileged behavior or guaranteed termination of kernel-stuck I/O.
- Metadata/binary verification, approved immutable dependency/security inventory,
  compatibility review, remote evidence transport and fresh zero-paid-usage gates
  remain outstanding. There is no loading, pilot or deployment authorization.

See HIERARCHY_VALIDATION.json and hierarchy-tests.log for actual local runtime,
commands and results. Review artifacts are outside the research repository.
