# Metadata binding correction and home inspection

Review-only candidate, 2026-10-08. All changes remain unstaged/unpublished.
Stage A and frozen B1 preserved; Stage B/GPU pilots paused;
pilot_execution_authorized=false.

## Findings

The observed home `/home/chaidir` is mode 0755, uid/gid 1009:1009;
its `/` and `/home` ancestors are root-owned 0755, with no symlinks or additional
ACL grants. It is a safer creation parent than the group-writable audit area.
No dedicated private pair was identified in the bounded immediate-child review.
Proposed, not created: `.performance-v2-scratch` and `.performance-v2-reports`
directly under that home, owned by uid 1009, mode 0700. See HOME-INSPECTION.md.

Prior preflight remains preserved in references: Python 3.10.17, Torch
2.3.0+cu121, torchvision 0.18.0+cu121, NumPy 2.2.6, OpenCV 5.0.0.93,
pytest 8.4.2, packaging 26.3 in the target environment. Base packaging 20.3
was also observed. The old empty hash for that base metadata is not validation.
No claim of import/ABI compatibility follows from distribution versions alone.

## Exact repository changes

1. scripts/performance_v2_native_acceptance.py
2. tests/test_performance_v2_acceptance_entry.py
3. docs/performance_v2_native_acceptance_checklist.md
4. configs/performance_v2/native_acceptance_source_lock.json

The inventory uses the invocation path and hash-bound pyvenv.cfg to identify a
virtual environment even under Python 3.10 -S. Explicit virtual/base scopes avoid
mistaking the base prefix for the target environment. Selected dependencies are
recorded as lists rather than overwritten by name. Canonicalized duplicate names
retain all records and their locations. Cross-scope duplicates are reported;
duplicates in the active set reject the inventory. Base dependencies cannot fill
missing virtual dependencies unless include-system-site-packages is explicitly
true. Invalid/missing configuration does not silently authorize base fallback.

METADATA and standard egg-info PKG-INFO are hashed as exact bytes with paths and
lengths. Missing/empty/malformed metadata is rejected, including malformed
descriptors outside the selected dependency names because their identity cannot
be trusted. Valid partial records and errors remain available in inventory JSON
with exit 2. No empty digest fallback remains. Invalid inventory blocks test
admission before scientific imports and retains a blocked report.

The selected-name scope remains torch, torchvision, numpy, pytest, opencv-python,
opencv-python-headless and packaging. This is not complete dependency closure.
Editable/.pth/custom-import-hook behavior is not emulated or certified; no site
initialization or package import occurs during this metadata inspection. I/O
failures outside individual descriptor parsing may still terminate with stderr;
retain stderr rather than assuming a complete inventory.

## Validation

13 focused invented-record tests passed, 0 failures/errors/skips, under local
CPython 3.14.6 Windows, C:\Python314\python.exe, SHA-256
03168c01b7b7491423350e82c26fee71f35b43694d1319d3c668bda6903a0c38.
Run command: `python -I -B -S run_focused_tests.py` from this review directory.
Network, shell and subprocess calls were denied with an audit hook. The blocked
admission test mocks inventory and verifies early rejection; it does not execute
the acceptance suite. Tests use temporary invented metadata only. Exact logs and
runtime are in FOCUSED-TESTS.json/log. Python 3.10 behavior is simulated, not
claimed as a new native runtime result. The expected complete unittest count is
143 (historical 130 + new 13), plus the three historical pytest modules.

The old complete review ZIP is embedded unchanged for historical validation and
failure evidence. Its 130-test evidence is tied to its old source lock, not
relabelled as validation of these changed bytes. FOCUSED.diff is relative to the
pre-edit working tree; COMPLETE-WORKTREE.diff includes all current untracked
preparation files relative to HEAD. Package verification records exact byte
reconstruction, unchanged HEAD/index and preservation of unrelated files.

## Smallest remaining requirements for one CPU-only Linux synthetic run

1. Review this correction and its frozen hashes. Separately authorize creation
   of the two proposed private directories, or supply an existing suitable pair.
   Verify fresh ownership/ACL/ancestor/mount evidence and umask 077. Resolve the
   account's same-UID sharing assumptions; no new account/infrastructure is
   automatically proposed.
2. Separately authorize/upload the verified corrected package and extract into
   a fresh source directory. Run its corrected read-only inventory with the
   specified existing interpreter. Bind actual pyvenv.cfg, environment/base
   metadata, Git, SQLite, distro/glibc and host/deployment identity. No Docker
   image digest is inferred for this native host. Resolve errors/duplicates;
   do not install or upgrade to hide failures.
3. Verify the existing environment meets the reviewed offline and research/weight
   exclusion boundary. Host egress/isolation and exclusion are not established
   by home-directory stat/ACL evidence. Reuse suitable existing controls; changes
   require their own review. Empty CUDA_VISIBLE_DEVICES plus existing GPU guards
   constrain the future CPU-only invocation, not host isolation by themselves.
4. Separately authorize exactly one full CPU-only synthetic acceptance attempt.
   It must run all 143 unittest cases and the three frozen legacy pytest modules,
   retain warnings/failures and all SIGKILL recovery records, without automatic
   retries. Native import/ABI compatibility (including Torch/NumPy interaction)
   is unresolved until bounded native validation. Pytest is already installed;
   no package installation is an observed requirement.

No full/native acceptance, training, research-data access, pretrained loading,
GPU activity, Docker, publication, remote dispatch or package change occurred.
Power-loss guarantees and production anti-rollback integration remain separate
production blockers; neither demands extra infrastructure for temporary tests.
