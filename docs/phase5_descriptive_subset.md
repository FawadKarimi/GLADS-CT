# Phase 5 pointing and accuracy subset

Implementation scope is `pointing_and_accuracy_only`, under sections 2 and 6 of
the unchanged [reviewed proposal](phase5_supplemental_evaluation_proposal.md).
All results are exploratory post-hoc. This infrastructure milestone uses only
synthetic CPU tests, not university data or research execution.

## Inputs and authorization

The separate opt-in entry point is `scripts/48_run_phase5_descriptive.py`.
It requires a clean `phase4-consendxformer` implementation checkout, the exact
versioned `configs/phase5_descriptive_v1.json`, and the reviewed proposal content
(LF-normalized SHA-256 `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`).
Do not change historical configurations or allowlists to admit new sources.

A later, separately authorized saved-artifact execution must supply explicit
paths through these options; no directory discovery or fallback is performed:

- `--full-source DIR` three times, one completed complete model per seed.
- `--b1-source DIR` three times, one completed B1 model per seed.
- `--ablation-source DIR` twelve times: each of the four frozen OFF variants
  at seeds 42/43/44, with the exact same-seed complete-model references.
- `--attention-source DIR` once, the completed matched native-map/prior bundle
  `phase4-evaluation-22cb656eabd040c2`, with consistent historical replay.
- `--authorize-analysis` explicitly. Without it, no source scoring is allowed.
- Optional `--output-root DIR`, strictly below
  `artifacts/phase5/pointing_and_accuracy/`.

No research command is run or authorized by this implementation handoff.
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

## Endpoint and export contracts

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

## Synthetic verification

`tests/test_phase5_descriptive.py` tests pure metrics, unequal patient clusters,
common draws and estimators. `tests/test_phase5_descriptive_runner.py` constructs
synthetic saved bundles with 183/99 identities and 807 prior-training identities.
Only expensive historical source loaders/Git cleanliness are replaced by test
fixtures; new production cohort/protocol/count/hash guards are not weakened.
Existing source-contract tests remain in the full suite. Tests intercept model
calls and old-suite entry points, check corruption and immutable sources,
deterministic exports, no overwrite and subset-only completion.

Use CUDA-hidden, bytecode-disabled tests with a fresh temporary `--basetemp`
and `-p no:cacheprovider`. No server work, research scoring, inference, training,
calibration campaign, commit or push belongs to this milestone.
