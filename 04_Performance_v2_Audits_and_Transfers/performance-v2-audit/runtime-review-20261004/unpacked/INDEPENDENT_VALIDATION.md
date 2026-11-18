# Validation — 2026-09-28

- All 40 input ledger entries matched their supplied hashes and byte lengths.
- Original supplied nine mocked tests passed independently before revision.
- Revised 15 mocked tests passed in 0.025 seconds on the local Linux Python
  runtime recorded in validation-report.json. Docker was mocked, not run.
- The portable read-only verifier passed with explicit candidate/evidence/
  package/ledger paths. Candidate hashes and mtimes were unchanged.
- A modified temporary candidate was rejected; its hashes and mtimes were
  likewise unchanged by verification. No original input was modified.
- Focused patch checked, applied, matched authored text, and reverse-checked
  against an LF-normalized copy of the supplied corrected candidate.
  Original CRLF byte application is not claimed. The complete ZIP is canonical.
- Inspector archive and its internal ledger, candidate ledger, Python
  compilation, YAML parsing and whitespace checks passed.
- Historical validation and patch are retained under history/.

No full-suite claim, Docker/container run, environment installation, dependency
download, actual weight loading, research analysis, publication or dispatch.
Source-document browsing retrieved security metadata only. See
REVISION_REVIEW.md and SECURITY_REVIEW.md for the negative clearance decision.

The final package contains the refreshed focused patch, full hash ledger and
false approval template. Hash identity is integrity evidence, not security
clearance. The outstanding Linux synthetic run is the next runtime-validation
milestone after the existing lifecycle/security decisions; it is not a demand
to run that test before authorizing the same test.
