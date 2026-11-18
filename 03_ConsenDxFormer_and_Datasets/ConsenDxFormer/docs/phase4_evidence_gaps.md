# Phase 4 evidence gaps and boundary execution status

## Completed boundary execution (supplied audit)

The user-supplied boundary audit records the following completed execution.
These findings are transcribed from the supplied audit summary; this
documentation update does not rerun or independently re-audit the server data.

- Evaluation: `phase4-evaluation-febe8402535570e1`.
- Execution commit: `b2a91901c022d73e94d3f683933dd74aced9eda0`.
- Full boundary membership: 729 nodules; 3 pre-existing malformed-geometry
  exclusions; source-valid evaluated subset: 726 nodules from 450 patients.
- All three frozen seeds, 42/43/44, were evaluated. Class predictions varied
  substantially across seeds; completion does not imply seed-stable behavior.
- Primary-training patient overlap: 198 of 450 evaluated patients (44.0%).

This is completed **descriptive stress analysis only**, not independent
held-out performance. Exclusion of boundary nodules from the primary cohort
does not remove the documented patient overlap. No superiority or causal
superiority claim is supported by this execution, and its results must not be
used for model selection, threshold changes or tuning.

The boundary-execution evidence gap is now closed at this descriptive scope.
The historical implementation and paused-execution notes below predate this
audit; they are retained for provenance, not as the current execution status.
The failed server attempt and all existing result bundles remain preserved.

## Remaining evidence status

- Verified limitation of the completed boundary evidence: it does not provide
  independent held-out boundary performance or establish superiority. The
  44.0% primary-training patient overlap and substantial seed variation must
  remain explicit in interpretation.
- Previously reported completed: the full-model versus
  `disagreement_weighting`-OFF uncertainty comparison. Its completion and
  audit evidence are not independently reverified by this boundary-only
  documentation update. Preserve the existing uncertainty bundle; this
  verification limitation is not a reason to rerun the comparison.
  No uncertainty-result conclusions are established by the boundary audit.
- Other primary, attention-comparison and ablation-analysis statuses are not
  reassessed or marked complete by this update. Boundary completion is not
  evidence of completion of those separate analyses.

## Historical infrastructure milestone

The original milestone implemented analysis infrastructure only; it did not run
the university analyses or establish a new research result. The subsequent
completed boundary execution is recorded above. The agreed methodology remains
unchanged: do not retrain, tune, change thresholds/lambdas, or modify completed
source bundles.

## Execution commands (reference only)

These commands are retained for reproducibility, not as a request to rerun the
completed boundary evaluation. Any new execution requires separate authorization.

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

Boundary stress (historical command; frozen persisted-best checkpoint inference only):

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
results must not drive model selection, filtering or tuning. At the original
implementation stage, actual boundary counts and patient overlaps were not yet
established. The completed execution's audited counts and overlap are now
recorded above; the methodological limits in this section still apply.

### Historical source-validity accounting correction (pre-execution)

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

At the time of this correction, the supplied server diagnostic reported 729
boundary members: 726 with `valid` status and three pre-existing malformed-
geometry exclusions. Those were supplied evidence, not locally verified data
or production selection constants. That diagnostic and the patch alone had
not established rendering integrity or evaluation eligibility of the 726 cases;
boundary inference was paused pending patch review. This historical limitation
of the diagnostic must not be confused with the later completed execution
recorded above.

### Historical boundary entropy correction and retained numerical policy

This correction was published in execution commit
`b2a91901c022d73e94d3f683933dd74aced9eda0`, before the completed evaluation above.
Its entropy-only preparation policy remains part of the execution provenance.

Boundary inference accepts finite class probabilities in `[0,1]` whose row
sums satisfy `rtol=0, atol=1e-6`. Float32 probabilities promoted to float64 can
meet that contract but fail the shared entropy metric's unchanged `1e-8`
row-sum guard. The boundary summary now prepares entropy inputs only: it copies
probabilities into a private float64 `[N,3]` array in the existing class order,
rejects empty/malformed/nonfinite/out-of-range or excessive-mass-error inputs,
and only then divides a separate copy by its row sums. The existing
`predictive_entropy` function validates that copy with its original strict
guard and computes natural-log entropy. Nothing is clipped or dropped.

Raw `prob_*` columns, predicted classes, confidence, row identities/order and
probability summaries remain unchanged. Only entropy and summaries derived
from entropy use the normalized copy. No normalization is added to primary or
uncertainty evaluations, the inference guard, or source-validity accounting.

The boundary evaluation identity/provenance and `evaluation.json` record
`boundary_entropy_preparation_policy`. Raw prediction rows include the raw
row sum, signed and absolute mass deviations, normalization factor, normalized
sum and maximum absolute probability adjustment. Per-seed diagnostics report
row counts, nonunit-mass counts and maximum corrections. These diagnostics and
the policy are also exported to `provenance/boundary_entropy_preparation.json`,
which binds the raw evidence table by SHA-256; the existing artifact manifest
and completion marker cover both files.

This is a numerical compatibility correction, not a new probability acceptance
threshold or a change to the cohort. The supplied server population remains
729 full boundary members, 3 source-invalid exclusions and 726 eligible members;
the patch does not locally reverify these data or modify their accounting.
The failed server attempt and all existing outputs must be retained. No real
analyses or inference were performed as part of that implementation milestone;
the later audited boundary execution is distinct from the implementation work.
