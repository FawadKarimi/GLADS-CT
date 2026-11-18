# Independent correction review — 2026-09-28

Decision: candidate improved; execution remains blocked. This is not Linux
container acceptance, native security clearance, or permission to train.

## Input identities and scope

- Supplied corrected ZIP: b26fe767f345f654494de257463779d990f9b02705e21f7c92c0bf5ea9240bb7.
- The embedded inspector ZIP remains unchanged at
  2f97fdaf4dd7e3dd5a04255aa62a7dd7b603fcda0fd82ffcb026b3957407d45e.
- Dependency versions, image digest, runtime inspector files and safeguards
  are unchanged. The supplied named installation-container method is retained.
- All local tests use fake Docker responses. No Docker command, package
  acquisition/installation, model loading or forward pass was performed.

## Corrections in this revision

1. Cleanup now attempts state read-back, removal and absence verification
   independently, even if the initial inspection or kill command fails.
   Earlier errors remain in the report and prevent a clean acceptance result.
2. Build report-write failure no longer masks the primary build failure.
   The structured report is also printed to the job log as a fallback.
3. Independent cleanup now returns nonzero on failure. Prior removal claims
   receive a fresh exact-name absence query. Missing resource records are
   explicitly inconclusive and never certify termination.
4. The report ledger is refreshed after independent cleanup, binding
   final-cleanup.json and other retained reports. A ledger-write failure
   also fails that step. Self-hashing of the ledger is deliberately excluded.
5. Orchestration attempts report and ledger persistence separately; write
   failures retain the primary diagnostic in stdout and yield failure.
6. The portable verifier now checks the exact historical evidence ZIP hash,
   archive/path safety, full candidate ledger coverage, false approval flags
   and approval-to-implementation bindings. It still writes nothing.
7. Additional mocked tests cover failed-kill follow-up, nonzero fallback
   cleanup, final evidence hashing, stale cleanup claims, primary-error
   preservation on disk failure and portable path rejection.

## Security disposition

The supplied SECURITY_REVIEW.md and per-package coverage remain an explicit
negative clearance decision, not an exhaustive successful vulnerability audit.
An independent metadata-only read confirmed these reference points on
2026-09-28:

- https://github.com/advisories/GHSA-rrmf-rvhw-rf47 lists torch <=2.12.1 as
  affected and 2.13.0 as patched. Its current GitHub rating is Low (1.9).
  The pinned 2.10.0 lies in that range. The guard is a mitigation for the
  intended entry point, not a proof that every native path is unaffected.
- https://www.python.org/downloads/release/python-31018/ documents tarfile,
  unicode-escape and ipaddress security fixes after pinned Python 3.10.17.
- https://security-tracker.debian.org/tracker/CVE-2025-9230 gives the
  Bookworm OpenSSL fixed source version as 3.0.17-1~deb12u3.
- https://security-tracker.debian.org/tracker/CVE-2025-4802 gives the
  Bookworm glibc fixed source version as 2.36-9+deb12u11.

These four checks corroborate selected findings only. They do not newly
verify all supplied advisory responses, all 105 native packages, every wheel
member, or Actions' bundled JavaScript dependencies. Absence of a database
match is not a declaration of safety. Exposure depends on exercised paths;
no exploitation or compromised package has been established by this review.

## Exact remaining execution blockers

- A hard bound on daemon pull/copy/commit I/O and image-layer storage has not
  been established. Container compute termination is separately checked;
  stopping a CLI does not prove its daemon operation stopped. Control-command
  capture also checks size after capture, rather than enforcing a streaming
  allocation bound. These limits prevent full lifecycle clearance.
- Exact wheel installation-path assessment, native source-package/backport
  mapping and coverage, and Actions transitive/runner runtime review remain
  incomplete. Existing known affected pins have no accepted disposition.
- Linux execution of the proposed build and synthetic scenarios remains
  untested. Mocked results cannot establish enforced cgroup/seccomp behavior.
- Fresh included-minute/storage availability belongs to eventual dispatch;
  no additional host-readiness rerun is needed merely for this correction.

SIGKILL, host loss and failure of stdout/storage can prevent report retention.
No application can guarantee a report after its host disappears. Such missing
evidence is inconclusive, never a pass. This does not waive the separate
daemon lifecycle blocker.

## Next action

Do not toggle approval flags or dispatch this candidate. Prepare a separately
reviewed inspection-environment pin-refresh and build-transport proposal,
targeting the documented affected pins and daemon lifecycle requirements.
Keeping old pins immutable indefinitely is not a security fix. This revision
does not select or install replacements; any change must remain isolated from
the frozen research environment and receive compatibility review. If the exact
pins must be retained, their actual reachable exposure needs an explicit
bounded disposition with supporting evidence; no blanket waiver is provided.

Stage A remains intact; Stage B paused; GPU pilots unauthorized.
