# Phase 5 Candidate C: core implementation and synthetic validation

**Date:** 2026-09-18 (Asia/Jakarta)
**Disposition:** PASS for the implemented arithmetic/pairing core and the finite
synthetic null campaign. **NOT READY for research execution.** Actual R runtime
cross-checks, historical source-loader integration and independent code review
remain outstanding. These are not waived by this report.

## Work completed

The new standard-library core implements the accepted equal-patient mean test,
with exact rational count/score arithmetic, directed Decimal probability/log-tail
enclosures, threshold refinement or failure, fixed 15-slot Holm bookkeeping,
and exact same-seed nodule pairing. It imports no model, checkpoint loader, old
bootstrap, pointing, or perturbation runner, and performs no file I/O.

Numerical p-value endpoints enclose the specified mathematical value; they are
not data-selected/tightened statistical bounds. Holm decisions must be resolved
by the entire adjusted enclosure. Missing or malformed records cause errors;
zeros and within-patient cancellations remain in the patient denominator.
The core supports generic cohort sizes for synthetic unit fixtures. A future
research runner must enforce the frozen 99 patients/183 nodules and historical
source contracts before calling it. This deliverable intentionally contains
**no research execution entry point**.

The user acceptance and bounded CPU scope are recorded separately in
`phase5_inference_implementation_authorization.md`. The three governing original
Markdown files were hash-checked before and after validation and remain unchanged.
The earlier decision document is not overwritten or backdated.

## Evidence summary

| Check | Observed result |
|---|---|
| Focused synthetic unit tests | 79 passed, zero failed, zero skipped in the supplied final log |
| Frozen null register | All 52 settings retained in specified order |
| Exact frozen settings | 46, with rational support masses and total mass exactly one |
| Monte Carlo settings | Six, 100,000 replicates each, 600,000 unique prescribed replicates total |
| Additional aggregate-null fixtures | Six exact fixtures; none expands the simulation budget |
| Exact support rows retained and independently reconstructed | 15,254 |
| Candidate / historical / historical-gated tail cells | 2,610 = 58 settings/fixtures x 15 thresholds x three rules |
| Candidate tail cells | 870: 780 exact not above nominal; 90 Monte Carlo upper bounds below nominal |
| Candidate adverse or inconclusive cells | Zero in this finite suite |
| Independent read-back | Passed all saved support and synthetic-ledger rows, all tail counts and intervals |
| Deterministic engineering replay | Nine numerical/register payload files byte-identical |
| Model forwards / research-data reads | Zero / zero |
| Actual R reference execution | NOT RUN: Rscript not available in the sandbox |
| Historical research source-loader integration | Not implemented or validated in this arithmetic-core tranche |
| Full repository regression suite / GPU environment validation | Not run |

The unit tests cover manual arithmetic, all-zero and constant scores, cancellation,
model swaps, patient/row permutations, within-patient replication invariance,
invalid counts/IDs/classes/seeds, exact pairing, empty/missing/duplicate family
members, undefined-slot bookkeeping, Holm ties, near-threshold numerical blocking,
Decimal context isolation, exact RNG traversal, document-hash corruption,
no-overwrite output reservation, symlink rejection, deterministic NPZ output,
bundle corruption and forbidden core imports.

## Preserved statistical plan

The six symmetric settings use the frozen PCG64/SeedSequence schedule
`[20260916, scenario_index]`. Traversal is replicate, patient, nodule, including
unused uniform draws. The batch implementation was checked against a separate
literal scalar generator and for batch-size invariance. Synthetic cluster sizes
are the proposal's sizes, not copied from observed research errors: the symmetric
settings have 99/246, while the other frozen settings have 99/183.

The 36 asymmetric, four heterogeneous, and six zero-inflated settings use exact
binomial/product/multinomial support enumeration. All 15 alpha thresholds
`0.05/j` are retained. Monte Carlo confidence allocation uses the approved
three-rule extension `1 - 0.01/(52*15*3)`, with two-sided Clopper-Pearson intervals.
Additional heterogeneous aggregate-null fixtures are exact, so the simulation
allocation is not silently expanded. Intervals were computed with SciPy and
checked by a separate read-back calculation; actual R `binom.test` agreement
remains pending rather than claimed.

The independent mpmath oracle does not call the production Decimal helpers.
It checks the candidate tail and threshold decisions for every realized distinct
score sum in the campaign. Historical reference tails are checked by high-
precision erfc for exact supports and a fixed deterministic set of 100 rows per
simulated setting, plus any close-to-threshold rows. This is not substituted for
the required R runtime reference.

## Known counterexample retained

The original 20-informative-patient counterexample is reproduced exactly. The
historical ungated and 20-score-gated rules both have rejection probability:

- `7549007091249/95367431640625` at alpha 0.05;
- `1100957668913/95367431640625` at alpha 0.05/15.

Candidate C has zero rejections on that exact support at every registered
threshold. These are marginal synthetic null probabilities, not research p-values,
not a measured joint study family-wise error rate, and not evidence of superiority.
The failed historical method stays failed; its results are not suppressed.

## What the passes do not establish

A finite suite does not prove the universal bound, establish real patient
independence, establish representativeness, make a post-hoc study confirmatory,
show useful power for small effects, or imply model superiority. The mathematical
justification remains conditional on the accepted bounded independent-cluster
model. No model/method choice was made from research outcomes.

The original 183/99 test membership, selected checkpoints, three seeds, primary
macro-F1, nodule-weighted accuracy, existing intervals and all completed evidence
are unchanged. All 15 research hypotheses remain `unresolved_not_computed`.
Full supplemental evaluation and Stage A remain incomplete.

## Provenance and replay

`run01` is the initial synthetic-core campaign; `run02` binds the finalized
numerical guards, added integrity tests, R bridge and authorization file. They
use **the same prescribed stream and budgets**; they are an engineering replay,
not 200,000 replicates per setting and not an adaptive extension after an adverse
result. All nine numerical/register payloads match exactly. Both records and
both source snapshots are retained. The candidate formula, null, thresholds and
scenario definitions did not change between them.

`run02/VALIDATION_STATUS.json` binds ten source/authorization/dependency files,
the original document hashes, software versions, all scenario details and
explicit pending gates. Each run's artifact manifest binds its payload hashes.
The original uploaded documents and the unmodified proposal copies were
rechecked. No actual Windows working tree or university filesystem was inspected.
No stage, commit, push, training or research execution was performed.

Validation environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, mpmath 1.3.0,
pytest 9.0.2. This is a sandbox CPU environment, not the frozen GPU environment.

## Remaining gates and next concrete action

1. Check the intended validation machine's `Rscript --version`, then pin the
   actual runtime before executing the supplied R reference bridge. It requires
   `clust.bin.pair` 0.1.2 and checks the body of the pinned Durkalski implementation.
   It tests 26 synthetic legacy cases, six 15-slot Holm families and nine binomial
   confidence-interval fixtures. Input bytes are pinned; outputs must use a new
   directory. No R success is claimed before execution and review.
2. Implement/review a separate integration layer that reuses the existing full,
   B1 and four OFF-source loaders without weakening their identities, source
   hashes, checkpoints, pairing, cohort and no-overwrite contracts. Test its
   corruption, source-preservation and failure behavior with synthetic fixtures.
   The pure pairing helper here is not a substitute for these loaders.
3. Independently review the implementation, numerical policy, complete validation
   evidence and actual checkout compatibility. Only then consider a separately
   authorized saved-prediction research run. No research command is supplied here.

## Primary/reference sources consulted

- Accepted `phase5_inference_amendment_decision.md`, especially sections 4–6.
- Unchanged `phase5_inference_resolution_proposal.md`, sections 4–6.
- Frozen revised `phase5_supplemental_evaluation_proposal.md`, sections 4–5.
- Hoeffding (1963), DOI `10.1080/01621459.1963.10500830`, Theorem 2, as identified
  in the accepted decision. No new inequality is substituted.
- Python Decimal documentation: correctly rounded ln/exp and adjacent values,
  `https://docs.python.org/3/library/decimal.html`.
- R stats p.adjust documentation: fixed-family Holm,
  `https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html`.
- Pinned CRAN source commit `e464ac7e5d094bffd25b7c4cbba67820b60f1cc1`, file
  `R/durkalski.R`, Git blob `f6523d982dca310c1a76954c9965e36aaa2f9cf1`.
