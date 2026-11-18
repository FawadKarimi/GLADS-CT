# Metadata reconciliation and next bounded plan — 2026-10-04

Review only. All approvals remain false; native supervisor remains disabled; Stage B paused. No Stage A files changed. No packages/images downloaded or installed, no Docker, weights, research or workflow execution. The historical local socket-failure report remains unchanged: this later owner-supplied archive resolves selected metadata gaps, not artifact-byte or execution gates.

## Evidence and verification

Supported-runtime ZIP SHA-256: `808c6df3c515802f3c4b237010cd9f67abf1b21ad54a97539dac7659a3b0aadc`; all 135 ledger entries verified. Supplied metadata ZIP SHA-256: `b9c3397f798464b85560d0f8faf2763c2b428144e3d677304985b115720a8edb` (computed locally, not compared with a separately supplied expected digest).

All 15 receipts bind to 11 unique metadata byte streams with matching SHA-256 and size. Retrieval times in the supplied receipts span 2026-10-04 10:10:30–10:10:37 UTC; these are attributed to the supplied run, not locally witnessed network responses. Replayed parsing offline against the preserved resolver reproduces PyPI wheel selections, CPU-index selections, torchvision commit and both Actions lockfile candidate lists. The OCI index selects exactly one Linux amd64 manifest, whose actual archived metadata bytes match its descriptor digest and size. No authentication signature, live origin or original execution host is independently attested by this archive. A metadata checksum is not a package checksum verification.

OCI tag at retrieval: `python:3.12.15-slim-bookworm`.

- Index metadata SHA-256: `54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3`.
- Resolved amd64 manifest: `sha256:9901e0a8d75037d8242ed43155cbcb2d1f61be1356383d8054afb59fd50e39c4` (1,752 archived bytes verified).
- Config descriptor: `sha256:3abf66fa95bf9285d35eac802319797d29d01fbcfef9c0cd560f73d97f4289e8`, 5,684 bytes; bytes NOT acquired.
- torchvision v0.28.0 commit: `8fb87713a24951e639c494b0f2a8a81b5f8e33a6`; tag-reference metadata only, not source-tree/topology verification.

The complete URL/descriptor/size plan is `reconciled.json`, including each layer and its expected compressed size. Nothing in that file is an acquisition or installation authorization.

| Proposed wheel | Publisher-reported SHA-256 | Expected bytes from supplied metadata |
|---|---|---:|
| torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl | `4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2` | Unknown — stop before acquisition |
| torchvision-0.28.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl | `b545d46f4d2f9d30381281cf22874bfe1d32a8a7b0ee8396fccde89f30c6a9d9` | Unknown — stop before acquisition |
| pip-26.2.1-py3-none-any.whl | `71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e` | 1,816,632 |
| wheel-0.46.2-py3-none-any.whl | `33ae60725d69eaa249bc1982e739943c23b34b58d51f1cb6253453773aca6e65` | 29,971 |
| setuptools-84.0.0-py3-none-any.whl | `51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670` | 818,216 |

All five wheel-byte verification flags remain false. No CPU wheel METADATA or dependency closure is present.

## Actions and advisory limits

Checkout commit `11bd71901bbe5b1630ceea73d27597364c9af683`: lockfile metadata hash `8db9b3d9236032153dbec34a878b7d6697bab2f09ce4428485252b9a3055d028`, 264,924 bytes; 32 non-dev lockfile candidate records reproduced.

Upload-artifact commit `ea165f8d65b6e75b540449e92b4886f43607fa02`: hash `fddf66cd1d48dde834ddf318b0b0d6107174bc82298aef3e1f8e6e34b1adfe31`, 531,417 bytes; 166 non-dev candidate records reproduced. These are dependency-path counts, not distinct audited packages. Original lists, versions and publisher integrity strings remain in metadata-resolution.json. No npm artifacts, bundled dist equivalence, action.yml runtime, runner Node version or npm advisory closure have been verified. SHA-1 integrity entries in the historical lock are not upgraded to SHA-256 claims.

Five GitHub pip-ecosystem `affects` queries (torch@2.13.0, torchvision@0.28.0, pip@26.2.1, wheel@0.46.2, setuptools@84.0.0) returned archived `[]` bodies. Each body is two bytes, SHA-256 `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`. Exact URLs/time/hash receipts are retained. Original resolver reports pagination_complete=true; response headers were not exported, so that assertion cannot be independently reconstructed. Empty exact-version results are not exhaustive security clearance: local +cpu variants, vendored/native libraries, transitives, OS packages, unpublished issues and later advisories remain outside this observation.

## Compatibility choices still open

Keep Python 3.12.15 as the candidate, not an approved deployment choice. Python 3.12 is supported through October 2028; 3.13/3.14 offer longer horizons (2029/2030). A newer supported interpreter may be preferable for a long-lived environment, but entails different exact OCI identity, cp313/cp314 wheel selection, native ABI and full dependency checks. The current evidence selects only cp312; it does not validate alternatives or free-threaded ABI variants. Revisit the choice before acquisition approval; do not silently switch. This disposable inspection environment is separate from the university Python and controller interpreter. Source: [Python lifecycle](https://devguide.python.org/versions/), checked 2026-10-04.

Setuptools 84 inherits removal of pkg_resources introduced in 82.0.0, not a removal first occurring in 84. Static review of the exact inspector, frozen-reference import paths and every selected wheel must search ordinary/dynamic imports, namespace declarations and resource/entry-point usage. Absence of text matches is not a runtime proof. Later authorized isolated CPU import/loading tests must exercise these paths without model forwards. If a dependency needs pkg_resources, STOP and propose an explicit reviewed migration or alternative pin with a new security assessment; no monkeypatch, hidden fallback or automatic downgrade. Source: [setuptools history](https://setuptools.pypa.io/en/latest/history.html#v82-0-0), checked 2026-10-04.

Full dependency closure is unresolved. wheel requires packaging>=24.0; the five selected wheels are not a complete lock. From verified wheel METADATA (or hash-bound publisher core metadata first), evaluate Requires-Python, Requires-Dist and markers for CPython 3.12 Linux x86_64 with no optional test/doc/core extras unless explicitly needed. Recursively choose exact wheels and verify constraints, CPU variant, manylinux/glibc compatibility, hashes, sizes and yanked status. Do not reuse cp310 closure or generic PyPI torch GPU requirements as evidence for the CPU wheel. Record all transitive decisions and vendored native components. No sdists, setup.py, build hooks, pip resolver execution, install or import in the byte-only phase.

## Proposed next phases — NOT EXECUTED or authorized here

1. Metadata completion first: obtain body-free, bounded HEAD metadata for the two exact CPU URLs (record redirect chain, final approved host, Content-Length and encoding) or an authoritative size-bearing index. Resolve missing transitive metadata and exact versions. Reject missing/conflicting size, unexpected redirect, architecture or version. HEAD establishes a publisher size assertion, not byte verification. Freeze a reviewed URL/hash/size allowlist before any GET. No automatic tag refresh or alternate version selection.
2. Separately authorize byte-only acquisition to a fresh external no-overwrite directory. Use a small reviewed streaming downloader, not pip/Docker. Proposed limits: one file at a time, 15-second connection/read timeout, 30-minute aggregate deadline, no retries/resume, exact expected size cap plus one-byte overflow detection, TLS verification, no automatic redirects beyond reviewed exact destinations. Record request/final URL, redirects, headers, UTC, received length and independent SHA-256. Compare with frozen expected size/hash before atomic finalization. On mismatch preserve bounded failure receipt/partial bytes in quarantine, never label verified or continue to installation. Never log bearer tokens.
3. OCI byte verification is a separately permitted subset of step 2: fetch only the manifest-pinned config and four blobs in reconciled.json, not Docker pull or a mutable tag. Check compressed byte sizes/digests before any later bounded offline archive inspection. No extraction or decompression in the download step. Later reviewed inventory extraction must honor OCI layer ordering/whiteouts, reject path escapes and special files, enforce expanded-byte/member limits, and recover dpkg/source mappings and interpreter/bootstrap identities without executing image code.
4. Disk admission for compressed byte-only staging: known config+four-layer subtotal is 45,903,889 bytes; adding pip/wheel/setuptools gives 48,568,708 bytes. Total B = 48,568,708 + exact torch size + exact torchvision size + all new transitive-wheel sizes (deduplicated). B is UNKNOWN today. Require free disk >= 2*B + 1 GiB and an enforced staging quota at that same bound, allowing one retained acquisition plus a review copy and bounded receipts. Refuse acquisition until B and the quota implementation are reviewed; do not guess from the old image or claim 12 GiB is enough. Decompression/installation/native acceptance needs a separate expanded-size budget; this compressed-byte budget does not resolve aggregate daemon storage.
5. Retain immutable acquisition receipt, prior metadata bytes, expected/observed hash ledger, original manifest/index and configuration, HTTP metadata excluding credentials, failure/partial records and a complete closure manifest. Later source inspection may read only bounded METADATA/WHEEL/RECORD and source text as data, never import or execute it. Artifact integrity is distinct from trust, compatibility and execution permission.

## Remaining blockers and next concrete action

Review the metadata conclusions and authorize, separately, metadata-only completion of missing CPU sizes and closure. Then review an exact byte-acquisition manifest before permitting downloads. New Debian inventory/backport review remains absent; config and layer descriptors alone cannot resolve it. Bootstrap installer security, pkg_resources compatibility, topology/loaded-state/storage contracts and synthetic Linux acceptance remain open. Pinned Actions' bundled code and Node/native dependencies need review. Daemon-I/O termination and aggregate no-spill storage remain blocked exactly as described in SUPPORTED_RUNTIME_REVIEW.md. No metadata result clears them. All execution/security/pilot approvals remain false.
