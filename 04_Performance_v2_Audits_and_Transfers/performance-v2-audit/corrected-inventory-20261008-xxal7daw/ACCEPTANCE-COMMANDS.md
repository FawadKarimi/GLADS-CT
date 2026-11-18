# Exact proposed inventory and acceptance commands

These are prepared commands, not execution authorization. Do not install or
upgrade dependencies to satisfy them. Run no research, weight or GPU task.
Verify the review ZIP against its independently supplied SHA-256 before trusting
any extracted executable. Work from the extracted review package directory.

Exact source-lock SHA-256:

`939856a467b20da0e1bb9f5e9f4ff54c08fffc896dbf39a70e6c30b1bece2a5e`

## Required operator inputs

Set `PY` to the absolute path of the already installed, reviewed target Python.
Set `PV2_PRIVATE_SCRATCH` and `PV2_PRIVATE_REPORTS` to existing dedicated private
directories on the reviewed local filesystem, owned by the non-root operator,
mode 0700 with reviewed ACLs. Neither path may overlap `candidate/`. Establish
offline networking and ensure research storage/weight caches are not mounted.
Do not infer an image digest from the Python inventory: supply the actual immutable
image/deployment identity as independent runtime evidence.

## Read-only target inventory

This command writes JSON to stdout only. `-S` avoids site initialization and
package imports; installed distribution metadata is inspected explicitly.
No SQLite file, subprocess, GPU probe or package installation is performed.

```sh
"${PY:?Set the reviewed installed interpreter}" -I -B -S \
  candidate/scripts/performance_v2_native_acceptance.py inventory \
  --scratch-parent "${PV2_PRIVATE_SCRATCH:?Set the private scratch parent}" \
  --expected-lock-sha256 939856a467b20da0e1bb9f5e9f4ff54c08fffc896dbf39a70e6c30b1bece2a5e
```

Retain this stdout separately if desired. Review Python/executable hash, SQLite
library/extension/compile settings, distro/glibc, architecture, distribution
metadata, Git binary identity, UID and scratch mount/mode. Null or absent image,
ACL, package-artifact or network-isolation evidence remains unresolved.

## Full offline native acceptance evidence

Only after separate native-test authorization, with the required runtime already
present, execute exactly once in the pre-established offline environment:

```sh
umask 077
export CUDA_VISIBLE_DEVICES=""
"${PY:?Set the reviewed installed interpreter}" -I -B \
  candidate/scripts/performance_v2_native_acceptance.py test \
  --suite full --require-linux \
  --scratch-parent "${PV2_PRIVATE_SCRATCH:?Set the private scratch parent}" \
  --output "${PV2_PRIVATE_REPORTS:?Set the private reports parent}/native-acceptance-001" \
  --expected-lock-sha256 939856a467b20da0e1bb9f5e9f4ff54c08fffc896dbf39a70e6c30b1bece2a5e
```

The output directory must not exist. Do not delete/overwrite it or automatically
retry after failure. A later authorized attempt needs a new output name, retaining
the first evidence unchanged. Source-lock failures precede test/output creation;
retain their stderr and do not proceed with those bytes.

Full success requires all 143 current unittest cases and all three frozen legacy
pytest modules to execute without failures/skips/xfails, seven real process-kill
records, eight child interpreter invocations including the rejected competitor,
fresh source-lock verification and `native_acceptance_evidence_passed=true`.
Review every warning and retained boundary/receipt/budget record. The separate
`native_acceptance` and `pilot_execution_authorized` fields stay false: an offline
pass is evidence for review, not production release or permission to train.

Exit 0 means the requested test scope passed; 1 means test failure; 2 means
preflight or fatal failure inside the evidence scope. Missing dependencies stop
without installation or substitution. A hash/parser failure before output setup
may terminate directly with stderr and no result file.

Linux child termination is SIGKILL, not power-loss simulation. The new process
tests use real directory fsync on Linux; older mocked failure-path tests retain
their original scope. Neither result proves storage-controller power-loss safety.


## Inventory correction

See REVIEW.md. Invalid/missing metadata yields JSON with errors and exit 2. Virtual/base records and duplicates are separate. No full acceptance command was run for this revision. The directory proposal is not creation or execution authorization.
