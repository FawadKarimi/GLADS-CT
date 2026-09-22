# Phase 5 Candidate C: saved-prediction integration for review

Prepared 2026-09-22 against `ca6c018ffbedae12988074fb66052592212a3c2f`,
branch `phase4-consendxformer`. This is an implementation/synthetic-test
milestone, not a study-statistics execution, independent integrated-code
approval, or research-readiness certificate. All 15 study hypotheses remain
`unresolved_not_computed`; the full supplement and Stage A remain incomplete.

## Chronology and unchanged decisions

The resolution proposal, amendment decision and implementation authorization
are retained byte-equivalent under their documented LF hash convention. Their
earlier pending language describes earlier milestones. The September 22 handoff
records the later acceptance of Candidate C's conditional independent,
possibly heterogeneous patient-cluster mean-null interpretation and the later
server-reported R reference pass. Neither this document nor the runner replaces
that history with a backdated approval. The historical core validation report
and its `NOT_RUN` R status are unchanged.

The supplied arithmetic core is unchanged. Exact patient scores are
`U_p=(b_p-c_p)/n_p`, `S=sum(U_p)`, with logical bounds `[-1,1]` and
`p=min(1,2 exp(-S*S/(2P)))`. All 99 patients, including zero and cancelled
scores, remain in each same-seed contrast. This explicitly substitutes for the
failed Durkalski/chi-square approach; it does not repair that approach. There is
one two-sided 15-slot Holm family at `1/20`, with the original hypothesis IDs.
No seeds are pooled as independent observations.

The estimand is secondary, equal-patient mean paired correctness under the
accepted conditional model. It is not patient diagnosis, primary macro-F1 or
the existing nodule-weighted accuracy. Independence is an assumption, not a
consequence of matching identities, synthetic tests, the hash split or R
formula agreement. All inference remains exploratory post-hoc. Non-rejection
does not establish equivalence; this subset does not establish general
classification superiority, clinical utility or SOTA. The existing 33
descriptive, unadjusted intervals and all completed subsets remain unchanged.

## Trust barrier and arithmetic

`configs/phase5_inference_v1.json` is checked against the exact versioned
specification. The separate `scripts/51_run_phase5_inference.py` entry point
accepts only explicitly supplied source directories:

- Three complete-model final sources, loaded by the unchanged
  `load_phase4_final_prediction_source` contract through
  `load_saved_prediction_inputs`.
- Three B1 sources through that aggregate loader's distinct
  `load_b1_prediction_source` contract.
- Twelve OFF sources through the separate `load_ablation_source` contract:
  four fixed variants at seeds 42/43/44, with exact same-seed complete-model
  reference records.

The existing source guards retain checkpoint selection, manifest, supervision,
split, label, input-lock, prediction and historical source-commit validation.
The integration adds a complete 18-source barrier, exact expected identities
from trusted supervision, and fixed 183-nodule/99-patient checks. It does not
accept summary CSVs or discover replacement sources. Source directories and
their members must not traverse symlinks. Per-source prediction/checkpoint
hashes, relative source ledgers and the combined input snapshot are bound into
the new evaluation identity. No model is constructed or forwarded. Historical
loaders may deserialize checkpoints and recompute their existing saved-metric
consistency checks; those guards are not bypassed by preflight.

Only after every source, cohort and evidence check passes may the explicitly
authorized statistics branch call the core. Rational scores, exact patient
denominators, Decimal log/probability enclosures, precision schedule
80/160/320 and whole-family failure are preserved. All-zero valid input yields
a defined bound of one; an invalid source or numerical failure never produces
placeholder p-values or a partially completed family.

Independent read-back rebuilds integer paired-correctness cells from the
exported original hard predictions, recomputes exact score sums and evaluates
the tail with mpmath at 60 digits beyond the reported Decimal precision. It
checks rank-free Holm adjustment independently of the core's sorted cumulative
maximum, including bound containment and decision resolution. This is an
implementation check, not a new method, calibration campaign or sampling proof.

## Evidence binding and remaining gates

The reviewed ZIP is external evidence, not repository content. Its SHA-256 is
`6c6c6c0bb308813fdaea2217c131cd34465cbb46e1fae1c408d62ac48c2e1e47`.
Its exact member manifest and every archive member are checked without
extracting/rerunning a campaign. Small pinned run02 status/reference fixtures
are copied into the new evidence record; the large simulation ledgers are not
copied into the repository or new evaluation bundle.

The saved 79-test pass, 52 settings plus six exact fixtures, 600,000 prescribed
simulation rows, and independent numerical read-back are **prior supplied
evidence**, not fresh campaign results. Source compatibility is checked against
run02's source hashes. Only `tests/test_phase5_bounded_mean.py` is adapted:
governing documents come from repository `docs/`, and integrity-corruption
tests use small invented temporary artifacts rather than an optional run01
directory. The core, campaign generators and numerical references are unchanged.
The adapted tests must pass anew; their historical test-file hash is not
misrepresented as the current hash.

Actual R outputs have not been supplied to this Windows integration task. The
server-reported `R_REFERENCE_COMPARISONS=PASS; RESEARCH_READY=False` is attributed
to the handoff, not independently re-executed or certified locally. Before a
new preflight/results bundle can be written, the runner requires:

1. A directory containing exactly `STATUS.txt`, `reference_identity.txt`,
   `legacy_results.csv`, `holm_results.csv` and `binomial_results.csv` from the
   completed R reference run.
2. A separate UTF-8 JSON object mapping exactly those five filenames to their
   supplied lowercase SHA-256 values, plus the externally expected SHA-256 of
   that JSON file. It is a reviewed receipt, not a runner-generated substitute
   for missing evidence.
3. Verified status/runtime/reference identity (R 3.6.3, clust.bin.pair 0.1.2,
   pinned source commit/function blob), exact output schemas/counts/IDs, and
   agreement of the 26 legacy, 90 Holm and nine binomial output rows with the
   pinned synthetic fixtures at the bridge's `5e-13*max(1,abs(expected))`
   tolerance. The reference comparison validates legacy formula agreement,
   not its failed calibration or Candidate C's substantive assumptions.

R is neither installed nor invoked by the integration. Verified later output
bytes/hashes are recorded additively; no old status file is edited. Hashes bind
supplied evidence; they do not independently attest who ran R or certify an
unavailable server. Independent integrated-code review, receipt/verification of
the actual R evidence and a separate explicit research-execution release remain
gates. Synthetic tests use clearly labelled fake evidence/trust anchors only;
their success cannot satisfy these real-evidence gates.

## Modes, outputs and failure semantics

The opt-in modes are mutually exclusive. `--authorize-preflight` validates
sources and evidence but never calls Candidate C or computes research test
statistics; its 15 slots remain unresolved. `--authorize-research-statistics`
is a separate future execution option, not authorized by this task or by a
successful preflight. No automatic research execution follows either mode.

Outputs must be beneath `artifacts/phase5/paired_correctness/`, in a fresh
content-addressed attempt directory. Source/protected-directory overlaps,
symlinks, nesting within completed bundles and existing attempts are rejected.
No prior evidence is overwritten. The identity binds the exact protocol,
governing documents, current implementation/Git/runtime identity, source
ledgers, original predictions and validated reference evidence. All input
snapshots and the clean repository identity are rechecked before/after export
and completion read-back.

Deterministic exports comprise protocol, evaluation/provenance, evidence and
hard-prediction JSON, exact inference JSON, a 15-row hypothesis CSV, a summary,
and a checksummed artifact manifest. The statistics mode also exports all
1,485 patient/contrast count records; its probabilities and adjusted values
are numerical enclosures, not falsely exact rounded scalar p-values.
Preflight writes `PREFLIGHT.json`, never `COMPLETED.json`. Statistics completion
certifies only `scope=paired_correctness_only`, never Stage A/full supplement
or general research readiness. Failure/interruption after reservation writes a
best-effort `FAILED.json` and independently removes success markers without
masking the original exception. Offline validation refuses failed, partial,
extra-file or inconsistent attempts and independently checks numeric exports.

## Proposed future zero-forward preflight (not executed)

After integrated-code review/publication and reconciliation of the server
checkout and actual reference receipt, run from a clean repository root with
its normal dependencies plus `requirements-phase5-inference.txt`. Every path
below is an explicit operator-supplied placeholder, not a discovery pattern.
This command does not authorize statistics, R reruns, inference or training:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
python -B scripts/51_run_phase5_inference.py --authorize-preflight \
  --full-source "$FULL_42" --full-source "$FULL_43" --full-source "$FULL_44" \
  --b1-source "$B1_42" --b1-source "$B1_43" --b1-source "$B1_44" \
  --ablation-source "$ATTENTION_OFF_42" --ablation-source "$ATTENTION_OFF_43" --ablation-source "$ATTENTION_OFF_44" \
  --ablation-source "$DISAGREEMENT_OFF_42" --ablation-source "$DISAGREEMENT_OFF_43" --ablation-source "$DISAGREEMENT_OFF_44" \
  --ablation-source "$ORDINAL_OFF_42" --ablation-source "$ORDINAL_OFF_43" --ablation-source "$ORDINAL_OFF_44" \
  --ablation-source "$FUSION_OFF_42" --ablation-source "$FUSION_OFF_43" --ablation-source "$FUSION_OFF_44" \
  --core-evidence-zip "$REVIEWED_CORE_ZIP" \
  --r-reference-dir "$EXISTING_R_RESULTS" \
  --r-reference-hashes "$REVIEWED_R_HASH_RECEIPT" \
  --r-reference-hashes-sha256 "$EXPECTED_R_RECEIPT_SHA256"
```

The no-overwrite rule intentionally rejects an identical rerun into the same
container. A separately reviewed attempt container under the same namespace
may be supplied with `--output-root`; this never permits overwriting or
implicitly upgrading a preflight to a statistics execution.
