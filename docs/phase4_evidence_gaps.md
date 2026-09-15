# Phase 4 evidence-gap analysis infrastructure

This milestone implements analysis infrastructure only. It does not run the
university analyses or establish any new research result. Do not retrain, tune,
change thresholds/lambdas, or modify completed source bundles.

## Future execution

Run from the clean repository root on `phase4-consendxformer` after review.
Use the existing Python environment and frozen rendered inputs. Both commands
are CPU compatible. Set `FULL42`, `FULL43`, `FULL44` to the exact completed
full-model experiment directories, and `OFF42`, `OFF43`, `OFF44` to the exact
completed `disagreement_weighting`-OFF experiment directories. These are
directories containing `COMPLETED.json`, not checkpoint file paths. The OFF
sources must reference those same full-model source directories and hashes.
The actual university directory names are deliberately not guessed here.

Uncertainty comparison (Bash; saved predictions only, no model inference):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python scripts/47_run_phase4_evidence_gaps.py uncertainty \
  --full-source "${FULL42:?Set completed full seed 42 directory}" \
  --full-source "${FULL43:?Set completed full seed 43 directory}" \
  --full-source "${FULL44:?Set completed full seed 44 directory}" \
  --off-source "${OFF42:?Set completed disagreement-OFF seed 42 directory}" \
  --off-source "${OFF43:?Set completed disagreement-OFF seed 43 directory}" \
  --off-source "${OFF44:?Set completed disagreement-OFF seed 44 directory}"
```

Boundary stress (Bash; frozen persisted-best checkpoint inference only):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python scripts/47_run_phase4_evidence_gaps.py boundary \
  --full-source "${FULL42:?Set completed full seed 42 directory}" \
  --full-source "${FULL43:?Set completed full seed 43 directory}" \
  --full-source "${FULL44:?Set completed full seed 44 directory}"
```

Outputs are reserved once beneath
`artifacts/phase4/evidence_gaps/{uncertainty,boundary}/phase4-evaluation-<identity-hash>/`.
An optional `--output-root` must remain under `artifacts/phase4/evidence_gaps`.
Existing output directories, completed ancestors, symlinks and source-directory
overlap are rejected. Failed partial bundles have no valid completion marker;
they are not silently overwritten. No source guard, historical bundle, frozen
configuration, model, training procedure or previous result layout is changed.

Each new bundle contains `evaluation.json`, readable limitations in `README.md`,
raw CSV tables, `provenance/provenance.json`, `artifact_manifest.json` with SHA-256
file and bundle hashes, and a final `COMPLETED.json` binding the manifest hash.
Provenance records exact source descriptors/ablation identities, checkpoints,
saved test prediction hashes, input snapshots, evaluator Git commit, evaluation
configuration and seed set. Sources and clean Git state are checked again before
completion. Uncertainty also exports `curves/risk_coverage.json`.

## Uncertainty definitions and limits

The completed-source loaders are unchanged: full models retain the historical
full-source contract and ablations retain their separate validated ablation
contract. Full/OFF reference, supervision hashes and all six test cohorts must
match exactly. Saved rows may differ in order, but cannot differ in identities
or labels. Numeric outputs are replayed from saved probabilities, with no new
primary test forward passes.

Predictive entropy uses natural logarithms. Radiologist disagreement is the
existing frozen `disagreement_entropy`, not a recomputed reader statistic. The
fixed grouping uses the median disagreement in frozen primary training
supervision: low is strictly below it, high includes ties. This reuses the
existing evaluator helpers, not a new threshold or subgroup definition.

Spearman is reported separately per seed and for one **mean predictive entropy
across seeds per nodule** against its single frozen disagreement value. This
matches the existing full-model pooling definition; it is not entropy of mean
probabilities. With the expected cohort, pooling has 183 nodules, not 549
independent observations. Nodules themselves can share patients. No
independence-based significance test or confidence interval is supplied.

Low/high summaries are provided per seed and on seed-mean nodule entropies.
Full-minus-OFF entropy, confidence and class-probability differences pair the
same seed and exact nodule identity; tables also collapse these differences to
one seed-mean difference per nodule. Paired AURC differences are descriptive
within-seed comparisons. These are not randomized causal effects or evidence
of causal superiority.

Both entropy-ranked and confidence-ranked risk–coverage curves reuse the
existing deterministic patient/series/nodule tie-break and discrete mean-risk
AURC. Curves remain per seed, with descriptive mean/sample-SD AURC summaries
across seeds. No concatenated 549-row curve or new ensemble risk definition is
introduced. This is exploratory post-hoc evidence, not model selection.

## Boundary definitions and limits

The existing validated Phase 2 boundary loader remains authoritative, including
completed source-chain, frozen readiness, image/mask hashes, split and primary
cohort binding. The existing inference function freezes the validated persisted
`best.pt`, uses the unaugmented boundary dataset and `no_grad`, and checks exact
ordered prediction identities. It does not run attention-map analysis or
primary test inference. The new orchestrator uses deterministic CPU inference.

The lower boundary is `[2.5, 3.0)`; the upper is `(3.0, 3.5]`. Exactly `3.0`
remains primary indeterminate, never boundary. Raw rows retain mean malignancy,
rating sum/count, region, original split, paths, class probabilities, predicted
class, entropy, confidence, seed and checkpoint identity. Region summaries give
exact nodule/patient counts, predicted-class counts, mean class probabilities,
and mean/median confidence and entropy. Aggregates first reduce to one
seed-mean row per nodule; they do not relabel ensemble probabilities as another
model or combine repeated seeds as independent samples.

Patient overlap is an explicit set intersection with each frozen **primary**
train/validation/test patient set. Counts and patient IDs are reported, along
with the number of boundary nodules belonging to those patients. Boundary
patients absent from the primary cohort are listed separately, and original
boundary split nodule counts remain available. Primary nodule non-overlap does
not imply patient non-overlap.

Boundary results are **descriptive stress testing only**, not an independent
held-out test set. No primary class target is invented and no accuracy, AUC,
calibration-to-hard-label score or primary headline metric is computed. The
results must not drive model selection, filtering or tuning. Actual boundary
counts and patient overlaps are unknown until the reviewed future execution;
this implementation makes no claims about them.

### Source-validity accounting correction (inference remains paused)

Boundary membership is determined by the same Phase 2 rule before source
validity is considered. Identity, rating/marker, split and primary-nodule
overlap checks still cover the entire boundary membership. Only afterward are
`excluded_malformed_geometry` rows recorded as pre-existing source-invalid
exclusions. The renderable allowlist remains exactly `valid` and
`valid_with_missing_ratings`; other or missing boundary statuses fail closed.
Exclusions never depend on missing rendered rows or files. Every eligible row
must still pass the existing rendering and evaluation checks. An empty eligible
subset fails before inference.

New runtime boundary provenance is schema 3. Existing `sample_count`,
`split_counts`, ordered inference identities and cohort hashes describe the
source-valid subset. The added `source_validity_accounting` object records the
full boundary membership and split counts, source-invalid counts by reason and
split, eligible counts by split, and exclusion records. Exclusion records retain
the canonical mean-malignancy value without rounding, alongside patient/nodule
IDs, split, original source-validity status and explicit reason. Full-membership
and exclusion hashes supplement the existing frozen source snapshots.

The provenance validator reconstructs the partition from the canonical ledger
bound to the frozen checksum fingerprint, checks all accounting values/hashes,
and reconciles the evaluated identities to the eligible subset. Schema 2
records are not silently upgraded. Existing result bundles are left untouched;
any future authorized run must construct new schema 3 provenance.

Both boundary reporting entrypoints describe the evaluated population as the
**source-valid boundary subset**, preserve full accounting in JSON, and export
`boundary_source_invalid_exclusions.csv` separately from prediction tables.
This does not turn boundary stress testing into primary held-out performance.

The supplied server diagnostic reports 729 boundary members: 726 with `valid`
status and three pre-existing malformed-geometry exclusions. These are supplied
evidence, not locally verified data or production selection constants. Rendering
integrity and evaluation eligibility of the 726 cases have not been established
by that diagnostic or by this patch. Boundary inference remains paused pending
patch review.
