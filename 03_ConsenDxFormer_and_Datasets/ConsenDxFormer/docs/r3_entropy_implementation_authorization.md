# R3 archived-entropy implementation authorization

The user approved the archive-scoped validation contract and descriptive policy
in `r3_entropy_resolution_proposal.md`, raw SHA-256
`e767983361beada298f76fbc6be12c940f21da1d02be35ae3d34af0dbb125c29`,
on the implementation task following its 2026-09-27 evidence review.
This is bounded authorization for implementation and invented synthetic CPU
tests only. The unchanged proposal retains its historical documentation-only
status. No research summary execution, server access or R3 closure is authorized.

## Version 1 saved-record contract

`r3_entropy_source.py` is a separate archive adapter. Existing live loaders and
arithmetic are unchanged. Exact ZIP, manifest, provenance, per-seed CSV, payload
digest and six experiment pins cannot be overridden by CLI or loader arguments.
The original archive is embedded byte-for-byte in a later output. Source paths
inside provenance are strings only; no original file is opened. Missing or
inconsistent records fail before summaries/export. Stored entropy alone is used.

The archived cohort ledger schema is `r3_archived_cohort_v1`: a JSON object with
`schema`, `fields`, and `rows`. Fields in order are `patient_id`,
`series_instance_uid`, `nodule_id`, `split`, `true_class`. Project exact CSV strings
from full seed 42, sort lexically by this five-field tuple, and retain one object
per nodule. Reject duplicates rather than deduplicate. Hash UTF-8 JSON with sorted
keys, compact separators, `ensure_ascii=False`, and no newline, using the existing
`hashing.sha256_json`. The reviewed archive ledger SHA-256 is
`8b836f06e8289000073ce0df9a82d6542c8b11fed4faa9e4d02fc6a5b1162f04`.
This is an archived cohort authority, not independent supervision verification.
Membership records reference zero-based CSV data-record positions excluding the
header, preserve stored entropy text and retain both original three-class labels.

The versioned `PROTOCOL` object in `r3_entropy.py` is exported and canonically
hashed into run identity with archive and both adapter source hashes. It fixes
24 group rows, 12 directional mean contrasts, and 2,196 membership rows. Each
model/seed remains separate. Quartiles use linear interpolation; sample SD uses
ddof=1. Empty statistics/contrasts are null (empty CSV cells with explicit status
or reason); singleton SD is null. Finite/range failures stop for source review,
not clipping, imputation or recomputation. Patient intersection IDs/counts and
union counts are exported for each partition; no patient diagnosis is invented.

`artifacts/phase5/r3_entropy/r3-entropy-<identity>/` is a separate protected
namespace. The CLI requires a clean `phase4-consendxformer` checkout, exact proposal
hash and explicit execution flag; benchmark or other subset completion cannot
authorize it. Existing output, including partial output, cannot be overwritten.
The source archive/proposal are rehashed before completion; exports are read back.
Interruptions preserve partial evidence without a valid completion marker.

Exports: protocol, unchanged proposal, exact source ZIP, archived-cohort ledger,
provenance with original archived bindings, limitations, patient overlap, validation
counts, three CSV tables, payload manifest and final scope-only completion marker.
Every completion explicitly disclaims original prediction/supervision/checkpoint
byte verification and live-source verification. No model forwards, entropy
recomputation, training, formal tests, R3 review closure, R4/R5 closure, full
supplement completion or Stage A certification is claimed. Descriptive post-hoc
limitations and primary macro-F1 are retained.

## Future command — not authorized or executed by implementation

After code review, publication and separate research-execution authorization:

```powershell
python -B scripts/52_run_r3_entropy.py --archive "<unchanged-reviewed-archive.zip>" --authorize-execution
```

Do not run this command during implementation. Tests use invented records and
test-local monkeypatching of trust constants to exercise the same archive checks;
there is no production alternate-pin option. Unmodified production pins reject
synthetic archives. The only permitted actual-archive operation now is read-only
`load_archive` integrity validation, which does not construct groups or summaries.
