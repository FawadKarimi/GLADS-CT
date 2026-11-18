# Next bounded metadata round — review only, 2026-10-04

Baseline ZIP SHA-256: `292bf529f83b8aa0d2ba2839aa85b42f2f3909db84f589bf88fef25b472a3024`; 156 ledger entries verified. Existing source, historical reports and false approval flags remain unchanged. This is a staged metadata-closure plan, NOT a complete dependency lock or installation authorization.

## Newly recovered starting evidence

The previously supplied CPU-index HTML already advertises hash-bound PEP 658/714 sidecars for both exact cp312 wheels. No new network access is necessary to discover these links:

- torch metadata SHA-256: `b3c7f4b3f77d06f86a42aab7d0facef7b661ca3c22967bfcc396917d935dd335`.
- torchvision metadata SHA-256: `a656cfc3611928c2004484b8d7fcb56616eb771b9a204afd56158fec58075d7b`.

The exact four method/URL pairs are embedded in closure_metadata.py: HEAD the two already-selected download-r2.pytorch.org wheel URLs, GET only their `.whl.metadata` sidecars. These are metadata bytes, not wheel contents. No image URL is permitted. No redirects are presently approved: every 3xx is recorded and rejected, including same-host redirects. A later review must approve an exact redirect pair in source; no host wildcard or automatically learned URL is allowed. No fallback wheel GET, Range GET, pip, Docker, dependency import or installation exists.

Reference: https://packaging.python.org/en/latest/specifications/simple-repository-api/ (PEP 658/714 core-metadata attribute and sidecar convention), checked 2026-10-04. A matching sidecar hash verifies the archived publisher assertion about metadata, not downloaded wheel bytes or publisher authenticity.

## Round behavior and limits

Standard library only, controller Python >=3.10, invoked with -I -S -B; no package prerequisites or installation. Exact source-embedded allowlist; environment proxies disabled. TLS verification retained. HEAD body is never read; a positive decimal Content-Length <=2 GiB is required. GET metadata capped at 1 MiB per response. No retries. 10-second socket timeout, 90-second worker aggregate deadline with before/after checks and 120-second parent wall timeout (also bounds DNS/headers). Parent kills/waits the worker on subprocess timeout. No worker descendants are created. Retain started.json, each completed request receipt, bounded sidecar bytes including hash failures, final report and parent finished.json outside any repository in a new directory. Completed prior receipts survive worker termination; an in-flight receipt may be absent. Host loss/all-channel failure cannot guarantee a final report. stdout/stderr are discarded, not unbounded logs; sanitized errors are recorded. Cancellation/host-loss leaves the started marker explicitly incomplete. No sidecar/report completion authorizes execution.

## Recursive closure, without silently broadening network scope

This first round collects the two exact CPU Requires-Dist lists, validates name/version and publisher sidecar hashes, and records raw requirements and Requires-Python. The graph recursively visits known exact nodes, terminates cycles and records missing children. It intentionally does NOT implement a partial PEP 440/508 evaluator or guess active markers/extras. A requirement's parsed name is a review hint; direct URLs and markers do not authorize a request. Full closure remains false even if all currently known nodes are visited.

After review of these sidecars, select exact versions for each active transitive dependency and enumerate exact metadata-only URLs and their reviewed redirects in a new bounded source plan. Review marker evaluation for CPython 3.12.15 / Linux / x86_64 / CPU and no extras, including requirements pulled by transitive extras. Record every edge, original constraint/marker, selected version, applicability decision and its evidence. Add pip 26.2.1, wheel 0.46.2, setuptools 84.0.0 and wheel's packaging>=24.0 requirement from the existing archive. Do not fetch test/doc/core extras merely because they appear in metadata. Version conflicts or unsupported metadata stop closure, never trigger automatic upgrades.

Repeat review-bounded rounds until all active edges lead to exact compatible wheel identities and sidecar metadata. Each new node needs filename, tag/ABI/platform, publisher size/hash, Requires-Python, yanked status, metadata URL/hash and source receipt. Verify marker/specifier decisions using a separately reviewed complete standards implementation in a later approved environment; no dependency installation is authorized here. The current script cannot complete arbitrary recursive network resolution in one run because those exact future URLs/versions have not yet been reviewed. This is an explicit remaining implementation/planning boundary, not a claim of full closure.

If a chosen publisher exposes no hash-bound wheel-specific core metadata (404, missing sidecar/hash, inconsistent identity or sidecar content), mark that node unresolved. Release-level PyPI JSON can guide review but cannot prove the CPU wheel's exact Requires-Dist or tag-specific metadata. Do not inspect wheel bytes to bridge the gap; identify the exact wheel and missing field for separate authorization. Unknown HEAD size also stays unresolved; no GET fallback.

## Proposed command — NOT EXECUTED

Upload the reviewed standalone script to `$HOME/pv2-metadata-review/closure_metadata.py`; create that upload directory manually outside research repositories. Verify its SHA-256 against this package's ledger before execution. After separate approval, run exactly:

```sh
/usr/bin/python3 -I -S -B "$HOME/pv2-metadata-review/closure_metadata.py" --authorize-metadata-only --output "$HOME/pv2-metadata-review/cpu-closure-round-01"
```

The output directory must not exist. Exit 0 means the four requests succeeded, NOT closure/security approval; exit 1/2 or absent finished.json requires review of partial evidence. Return the entire output directory, script bytes/hash and immediate exit status. No workflow is prepared or dispatched.

## Remaining blockers

Execution of this proposed round has not occurred. Both actual CPU sizes, sidecar contents, recursive exact versions, marker evaluation and full dependency closure remain open. Artifact bytes, Debian inventory, advisory coverage, setuptools/pkg_resources compatibility, Python alternative decision, native lifecycle/I/O/quota acceptance and hash-bound authorization remain blocked. Native supervisor stays disabled, all approvals false, Stage A frozen and Stage B paused.
