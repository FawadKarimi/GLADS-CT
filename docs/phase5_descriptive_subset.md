# Phase 5 pointing and accuracy subset

Implementation scope is `pointing_and_accuracy_only`, under sections 2 and 6 of
the unchanged [reviewed proposal](phase5_supplemental_evaluation_proposal.md).
All results are exploratory post-hoc. The historical implementation milestone
and the subsequently completed university execution and independent artifact
audit are distinguished below. Completion applies only to this descriptive
subset, not the full supplement or Stage A. Macro-F1 remains the primary endpoint.

## Historical implementation milestone

Implementation commit `75b5d7ce895338e4b21747c05b039850eae2b936` was validated
using synthetic CPU tests, without university data or research execution at that
milestone. The following implementation contracts remain unchanged; they are not
a statement that the later execution is still pending or authorization to rerun it.

### Inputs and authorization

The separate opt-in entry point is `scripts/48_run_phase5_descriptive.py`.
It requires a clean `phase4-consendxformer` implementation checkout, the exact
versioned `configs/phase5_descriptive_v1.json`, and the reviewed proposal content
(LF-normalized SHA-256 `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`).
Do not change historical configurations or allowlists to admit new sources.

At implementation review, a later, separately authorized saved-artifact execution
was required to supply explicit paths through these options; no directory
discovery or fallback is performed:

- `--full-source DIR` three times, one completed complete model per seed.
- `--b1-source DIR` three times, one completed B1 model per seed.
- `--ablation-source DIR` twelve times: each of the four frozen OFF variants
  at seeds 42/43/44, with the exact same-seed complete-model references.
- `--attention-source DIR` once, the completed matched native-map/prior bundle
  `phase4-evaluation-22cb656eabd040c2`, with consistent historical replay.
- `--authorize-analysis` explicitly. Without it, no source scoring is allowed.
- Optional `--output-root DIR`, strictly below
  `artifacts/phase5/pointing_and_accuracy/`.

The implementation handoff did not run or authorize a research command.
There is no model/device/perturbation/training/test-statistic CLI option. Existing
source loaders may deserialize trusted persisted checkpoints and check saved
source metrics as part of their unchanged validity contracts; this is not a
model forward or a rerun of an old evaluation suite. No dataset, model or prior
is reconstructed. All 18 sources, exact identities and labels, artifact/array
hashes, source supervision, archive source descriptors/checkpoint/prediction
hashes and frozen 807-training-nodule prior provenance must pass before scoring.
Both archived B1 and complete-model logits/probabilities are compared to the
bound saved prediction tables without inference. Runtime inference replay is
not claimed. A missing or invalid source is an error, never an inner-join filter.

### Endpoint and export contracts

Pointing uses the saved 8x8 maps, the unnormalized native soft-mask key at
`>=0.5`, exact maximum ties resolved by first row-major index, and the existing
prior once. All 183 nodules/99 patients remain. Empty targets are misses with
explicit reasons, not evidence of failure against an available positive target.
The known flagged nodule must remain present and empty at that binary threshold;
contradiction stops for review. Tie/constant-map flags and raw map/mask hashes
are retained. No old alignment metrics or mask thresholds are recalculated.

Accuracy alone uses all 18 saved model/seed predictions. One PCG64 ledger with
seed 20260914 draws 99 patients with replacement in each of 10,000 replicates.
Integer patient multiplicities retain every nodule per occurrence; ratios stay
nodule weighted, not equally patient weighted. The same draws serve all models,
seeds and 15 complete-minus-comparator differences. Intervals use linear
2.5/97.5 percentiles. The 18 individual plus 15 paired intervals are descriptive,
unadjusted and conditional on model fits, not simultaneous coverage or adjusted
findings. No averaging of interval endpoints or invocation of the old
multi-metric bootstrap suite occurs.

New content-addressed sub-bundles use
`artifacts/phase5/pointing_and_accuracy/phase5-descriptive-<hash>/`. Identity binds
the scope, protocol, implementation commit, runtime versions and source hashes.
Existing completed or incomplete paths, protected-source overlaps, escaped
namespace paths and completed ancestors are rejected. Failed partial exports
are retained without completion. Old result bundles are never overwritten.

Outputs include:

- `tables/pointing_per_nodule.csv`: 1,281 records (six model/seed maps plus
  one prior per nodule), exact peak/tie coordinates, mask foreground and reason.
- `pointing_per_seed.csv` (seven rows including the one prior),
  `pointing_aggregate.csv`, `pointing_paired_descriptive.csv` (nine contrasts),
  `pointing_paired_per_nodule.csv`, and `flagged_case.csv` (seven records), all
  under `tables/`. Aggregate seed mean/sample SD is descriptive only.
- `tables/accuracy_intervals.csv` (33 rows) and `accuracy_per_nodule.csv`.
- `accuracy_bootstrap.npz`: sorted patient IDs, common draw indices and their
  hash, nodule denominators, ordered endpoint IDs and all 10,000-by-33 replicate
  estimates. Archives reject pickle/object arrays and verify byte-level array
  hashes on read-back; CSVs verify numeric/text round trips.
- `tables/hypothesis_register.csv`: all 15 original IDs remain
  `unresolved_not_computed`. There are no Q, raw/adjusted p-values, placeholder
  p=1 values or patient-weighted correctness results.
- `protocol.json`, `evaluation.json`, `summary.md`,
  `provenance/provenance.json`, a SHA-256 artifact manifest and final
  `COMPLETED.json`. Input snapshots are rechecked before/after completion.

The marker states `subset_completed=true`, `scope=pointing_and_accuracy_only`,
`full_supplemental_completion=false`, `formal_tests_computed=false` and
`model_forwards=0`. It cannot certify Stage A, the full supplement, McNemar or
multiplicity. Insertion/deletion and calibration remain not executed; formal
testing remains unresolved. No perturbation timing or R/calibration validation
evidence is fabricated. The original proposal remains intact for those later
decisions.

### Synthetic verification

`tests/test_phase5_descriptive.py` tests pure metrics, unequal patient clusters,
common draws and estimators. `tests/test_phase5_descriptive_runner.py` constructs
synthetic saved bundles with 183/99 identities and 807 prior-training identities.
Only expensive historical source loaders/Git cleanliness are replaced by test
fixtures; new production cohort/protocol/count/hash guards are not weakened.
Existing source-contract tests remain in the full suite. Tests intercept model
calls and old-suite entry points, check corruption and immutable sources,
deterministic exports, no overwrite and subset-only completion.

The implementation verification instructions required CUDA-hidden,
bytecode-disabled tests with a fresh temporary `--basetemp` and
`-p no:cacheprovider`. Research execution and the subsequent artifact audit are
separate events, recorded below; this documentation update performs neither.

## Completed university execution and independent artifact audit

Evidence source: the supplied
`Phase5_Descriptive_Execution_Audit_04d6daaccc698431.md`. The report states that
the supplied bundle passed the independent integrity and numerical checks
performed there. No implementation correction or analysis rerun is indicated.
This section records that supplied audit, rather than claiming a new local
verification of the research artifacts or access to the university filesystem.

### Execution and provenance

- Evaluation: `phase5-descriptive-04d6daaccc698431`.
- Execution/implementation commit:
  `75b5d7ce895338e4b21747c05b039850eae2b936`, branch `phase4-consendxformer`.
- Server bundle:
  `/experiment-storage/chaidir/fawad-dsai/ConsenDxFormer/artifacts/phase5/pointing_and_accuracy/phase5-descriptive-04d6daaccc698431`.
- Uploaded ZIP SHA-256:
  `faf1799e7f722ade581720febca5bb089068f89ca6b18655f63a5ec4cd49ad51`.
- Source snapshot SHA-256:
  `cae4cc716a90d52d95574e638f536d6937ce54780435b92136d7d48111348738`.
  It matches the preceding successful server preflight and binds 1,572
  source-file entries.
- Common patient-draw array SHA-256:
  `cbc62fd6e30d49feda921b2cd217bd842979845a6185c1ae663f84d557c710d4`.
- Reviewed proposal LF-normalized SHA-256:
  `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`.
- Original matched-attention evidence: `phase4-evaluation-22cb656eabd040c2`,
  recovered from `phase4-evaluation-22cb656eabd040c2(1).zip`, with ZIP SHA-256
  `e182e2a39e6c0881c6ef021fac9535d3d45fa49cd9320a4ae0be538dacb5e7b2`.

The completed scope is `pointing_and_accuracy_only`. Exported scope flags record
zero model forwards, no training, no formal tests and
`full_supplemental_completion=false`. The 18 distinct model/seed sources comprise
the complete model, B1 and four OFF variants at seeds 42/43/44. All use the same
183 test nodules from 99 patients: 183 rows per group, 3,294 prediction rows in
total. These repeated model/seed evaluations are not independent patients.

### Independent audit coverage and limits

The supplied artifact audit verified:

- ZIP membership and CRC, all 14 payload hashes, manifest bundle digest,
  completion-marker link, content-addressed evaluation ID and matching
  evaluation/provenance identities. The ZIP contains 16 files including the
  manifest and marker.
- Equality of the exported protocol to the reviewed versioned JSON, the
  unchanged proposal hash, and the execution commit and source-snapshot digest
  against the supplied preflight.
- Uniqueness and distinct directories of all 18 model/seed source records,
  agreement of checkpoint/file hashes with the merged snapshot, and exact
  same-seed complete-model reference bindings for all 12 OFF runs.
- Identical nodule identities, labels and patient membership across all 18
  prediction groups and the original matched-attention test archive. All
  correctness flags agree with saved true/predicted labels; all six complete/B1
  groups' predicted classes match the original archived probability argmax.
- Original attention payload/NPZ array hashes and exact equality of its
  provenance to the evidence provenance embedded in Phase 5. The fixed prior
  binds 807 training-nodule identities, with patient IDs disjoint from the test
  IDs; no prior was refitted.
- All 1,281 pointing rows against original saved native maps and masks, not
  merely exported hit flags: identities, map/mask hashes, peak coordinates and
  values, exact maxima/ties, thresholded foreground, empty targets, hits and
  reasons. Native masks exactly equal unnormalized area averages of the archived
  64x64 input masks. All per-seed, paired and aggregate pointing tables agree
  with the per-nodule records.
- Exact reproduction of sorted 99-patient IDs and all 10,000-by-99 PCG64 draw
  indices from seed 20260914, with the same ledger referenced by every endpoint.
- Exact agreement of all 330,000 saved bootstrap estimates, all nodule
  denominators, 33 point estimates and 66 linear-percentile interval endpoints.
  The independent oracle directly materialized every nodule row for each sampled
  patient occurrence; it neither called the production bootstrap function nor
  used its multiplicity-matrix implementation.
- All 15 hypothesis entries remaining `unresolved_not_computed` and the
  subset-only completion flags.

The audit used only existing uploaded predictions, original maps and the saved
bootstrap ledger. It did not execute the research runner, create a new result
bundle, train or run a model, alter data or thresholds, regenerate a prior,
compute new hypothesis tests, or rerun boundary/uncertainty analyses. Bootstrap
estimates matched exactly with audit NumPy 2.3.5 / pandas 2.2.3 versus server
NumPy 2.2.6 / pandas 2.3.3.

The audit did not independently access the live university filesystem or rehash
all live source files/checkpoints. Live immutability was reported by the
successful server diagnostic; the audit checked uploaded records and snapshot
bindings. The original 12 OFF prediction files and checkpoints are not embedded
in this ZIP. Their mathematical verification uses complete exported per-nodule
predictions; original-source validity is supported by the guarded server loaders.
The audit does not revalidate model training or independently instrument the
completed server process.

### Verified pointing-game results

From the audited per-seed, aggregate and paired pointing tables, in seed order
42/43/44:

- ConsenDxFormer: 180/183 = 98.36%; 181/183 = 98.91%; 179/183 = 97.81%.
  Arithmetic mean across seeds: 98.36%; sample SD: 0.55 percentage points.
- B1: 0/183 = 0.00% at each seed.
- Fixed training-only prior, evaluated once: 181/183 = 98.91%. It is not a
  third seed ensemble. Complete-minus-prior differences are -0.55, 0.00 and
  -1.09 percentage points for seeds 42/43/44.

The prior's unique maximum is the same zero-based native-grid coordinate `(4, 4)`
for all 183 nodules. The fixed spatial prior matches or exceeds every complete
model's pointing score. This supports a strong spatial-centering advantage in
this ROI/grid/target setting and limits this endpoint's ability to demonstrate
image-specific localization benefit. It does not prove all complete-model
attention is uninformative or invalidate other prespecified spatial metrics.

All 1,281 maps have unique maxima; none is constant. B1's zero hits are not a
tie-breaking or constant-map artifact: its peaks vary in location outside these
thresholded targets. This measures native attention-peak correspondence, not
zero classification accuracy, a malfunctioning classifier or proof that B1 uses
no nodule information elsewhere in its network.

Exactly one unique empty native binary target,
`LIDC-IDRI-0052__nodule-49c6132bfb77f3f1`, remains in all six model-seed groups and
the prior group, giving seven explicit miss rows for one nodule, not seven empty
nodules or exclusions. Every score retains denominator 183. No alternate
threshold, mask normalization, post-hoc exclusion or nonempty-only headline score
was introduced.

The complete model has strong peak correspondence relative to B1 under the
frozen definition, but no pointing advantage over the fixed prior. Pointing
does not establish causal faithfulness for class predictions, full-scan detection
performance or clinical usefulness.

### Verified descriptive accuracy intervals

Macro-F1 remains primary; this accuracy supplement does not replace it. The
following 18 individual results are accuracy percentages with unadjusted 95%
patient-cluster percentile intervals, transcribed from the supplied audit in
seed order 42/43/44. The estimand is nodule-weighted accuracy, conditional on the
fitted models. Each sampled patient contributes all its nodules at every
occurrence; bootstrap nodule denominators range from 142 to 246.

- ConsenDxFormer: 64.48 [55.61, 73.53]; 62.30 [53.02, 71.69];
  71.58 [64.77, 78.21]. Arithmetic mean: 66.12%.
- B1: 67.76 [60.11, 75.42]; 66.12 [58.56, 73.46];
  66.67 [58.29, 74.86]. Arithmetic mean: 66.85%.
- Attention alignment OFF: 62.84 [54.40, 71.20]; 58.47 [49.04, 67.78];
  57.92 [48.04, 67.63]. Arithmetic mean: 59.74%.
- Disagreement weighting OFF: 72.13 [65.06, 79.01]; 68.85 [61.50, 76.39];
  69.40 [62.28, 76.44]. Arithmetic mean: 70.13%.
- Ordinal auxiliary OFF: 60.11 [51.39, 68.75]; 57.92 [49.13, 66.85];
  64.48 [56.68, 72.37]. Arithmetic mean: 60.84%.
- Hierarchical fusion OFF: 69.40 [62.23, 76.53]; 66.67 [58.54, 75.00];
  68.85 [61.62, 76.00]. Arithmetic mean: 68.31%.

These arithmetic means summarize three fitted models; they are not ensemble
accuracies, pooled independent-sample estimates or averages of interval bounds.
ConsenDxFormer's mean is 66.12% versus B1's 66.85%, a difference of -0.73
percentage points, with seed-specific directions varying.

All 15 paired complete-minus-comparator differences follow, in percentage
points with unadjusted 95% descriptive intervals, again in seed order 42/43/44:

- B1: -3.28 [-8.74, +2.17]; -3.83 [-11.24, +3.23];
  +4.92 [-0.63, +10.55].
- Attention alignment OFF: +1.64 [-4.06, +7.22]; +3.83 [-0.58, +8.56];
  +13.66 [+4.00, +22.60].
- Disagreement weighting OFF: -7.65 [-15.27, +0.52]; -6.56 [-11.76, -1.18];
  +2.19 [-1.11, +5.62].
- Ordinal auxiliary OFF: +4.37 [-1.54, +10.00]; +4.37 [+1.01, +8.21];
  +7.10 [+1.14, +12.95].
- Hierarchical fusion OFF: -4.92 [-12.30, +2.59]; -4.37 [-9.77, +1.10];
  +2.73 [-1.15, +6.50].

All three complete-minus-B1 intervals contain zero. Classification superiority
over B1, equivalence and non-inferiority are not established. Repeated seeds
use the same patients and are not independent clinical replications. These
intervals are descriptive and unadjusted, not simultaneous coverage or
multiplicity-adjusted findings; do not infer significant results or p-values.

The intervals entirely on one side of zero (attention alignment OFF at seed 44,
ordinal auxiliary OFF at seeds 43/44, and disagreement weighting OFF at seed 43
in the opposite direction) are descriptive interval locations only. They do not
close any formal hypothesis. Mean accuracy ordering is mixed: removing attention
alignment or ordinal supervision lowers observed mean accuracy, whereas removing
disagreement weighting or hierarchical fusion raises it. This does not justify
selecting an OFF variant after test exposure or claiming every component helps
every outcome. Preserve the complete model, all seeds, endpoint hierarchy and
all ablation evidence.

### Remaining evidence status

- Native-map pointing-game and all 33 descriptive accuracy intervals: completed
  and independently audited for `pointing_and_accuracy_only`.
- All 15 formal hypotheses: `unresolved_not_computed`. The cluster-aware
  McNemar/formal paired-correctness method remains unresolved because the
  previously proposed method failed its specified null calibration counterexample.
  No replacement is validated by this audit.
- Multiplicity correction / registered 15-test family: unresolved. Descriptive
  intervals do not supply valid replacement p-values.
- Insertion/deletion: unexecuted by this subset; separately reviewed scope and
  later execution authorization are required. No calibration campaign is
  completed by this subset or audit.
- Full supplement / Stage A: incomplete. No original requirement is waived.
- Earlier completed boundary and uncertainty evidence, source-validity accounting,
  proposal, protocol and result bundles remain unchanged. This audit is not a
  reason to rerun those analyses or the completed descriptive subset.
