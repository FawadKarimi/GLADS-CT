# Phase 5 supplemental evaluation proposal

## Status, chronology and authorization

**Proposed protocol only; not approved, implemented or executed.** Prepared
against `e3c1a24d87d5e21662984a9d1a4e77d6e5b0f03a` on
`phase4-consendxformer`. The coding checkout matched that commit and was clean
before this document was added. University inputs and execution remain on the
university server; no local verification of unavailable research bundles is
claimed.

The original Phase 5 brief required pointing-game, insertion/deletion, McNemar
and multiplicity correction. The supplied read-only source audit found these
absent at this commit. No explicit agreement removing them was recovered.
Existing bootstrap intervals were intentionally descriptive, not substitutes
for the missing tests. This proposal supplements that record; it neither
backdates agreement nor repeats the completed boundary or other analysis audits.

**All analyses specified here are exploratory post-hoc.** Their details are
being proposed after test results were seen. Approval now would fix a future
implementation protocol, not make it preregistered or confirmatory. Even
adjusted p-values would not establish confirmatory superiority, causality or
clinical benefit. The clustered-test estimand amendment below requires explicit
review; the original McNemar requirement is not silently declared satisfied.

The sequence is protocol review, then separate implementation and synthetic
validation/code review, then separately authorized supplemental server
execution. This document authorizes none of those subsequent steps. No training,
inference, real-data scoring, tuning, CSV edits or old-evidence replacement is
part of its preparation. The September 16, 2026 review of the initial proposal
(SHA-256 `2eac673f8cf1db8983f5c48f955d3ec172f059471000a241f07a80f9485583e3`)
identified a finite-null calibration failure. This revision independently
verified that hypothetical counterexample by exact binomial enumeration only;
it did not run a Monte Carlo campaign or evaluate research data. Formal testing
is now explicitly **blocked pending a justified, reviewed method or scope
amendment**. The original McNemar/inferential requirement remains unresolved.

## 1. Frozen decisions and common input contract

Already frozen, not proposed changes:

- The existing 2D architecture and actual pinned B0/B1 definitions; completed
  complete-model and four single-component ablation definitions; labels,
  cohorts, source-validity rules and patient-disjoint primary splits.
- Exactly seeds `42, 43, 44`; each source's validation-selected persisted
  `best.pt`, effective configuration and source/checkpoint identities. The
  complete-model source commit remains
  `e1b174490245c616e355ef7c0c6f7c3c239d3acd`; ordinal/attention lambdas remain
  `0.5`/`0.05`. Nothing here selects another checkpoint or changes a loss.
- Primary test membership: the same 183 nodules from 99 patients, not boundary
  cases or a newly filtered cohort. Training-only prior membership: the same
  807 training nodules. These are existing contract counts, not new observations
  from server data in this documentation task.
- Class order: `benign, indeterminate, malignant`. Existing probabilities,
  hard predictions, image preprocessing, attention definitions and result
  bundles remain unchanged.
- Native attention and prior evidence, mask semantics and replay definitions
  in [the attention evidence contract](phase4_attention_evidence.md) and
  `configs/phase4_attention_evidence_v1.json`; existing statistical definitions
  in [the evaluation contract](phase4_evaluation.md). The historical
  [boundary/uncertainty status](phase4_evidence_gaps.md) is not reopened.

Proposed supplemental preflight, reusing rather than weakening existing guards:

1. Require one completed complete-model and B1 source per seed. For paired
   correctness also require one completed source for each of
   `attention_alignment` OFF, `disagreement_weighting` OFF,
   `ordinal_auxiliary` OFF and `hierarchical_fusion` OFF per seed. Use
   `load_phase4_final_prediction_source`, `load_b1_prediction_source` and
   `load_ablation_source` through their existing, distinct trust contracts.
   Each ablation must reference the same-seed complete-model identity and
   hashes. Never broaden a historical source allowlist to admit a supplement.
2. Resolve the existing completed matched-attention evidence bundle by explicit
   path and manifest hash, not newest-directory discovery. Require its source
   descriptors/checkpoint hashes to bind those same six B1/complete-model
   sources, its original replay to be consistent, and its comparison
   interpretation flag to permit use. A diagnostic-only export with replay
   discrepancies is not sufficient. Do not regenerate attention or refit a
   prior to repair an unsuitable source; stop for review.
3. Read `test_inputs.npz` with `allow_pickle=False`: identity arrays
   `patient_id, nodule_id, series_instance_uid, split, true_class`,
   `ct_crops`, `soft_masks_input`, `soft_masks_native_grid` and
   `target_distributions`. Read `b1_attention`, `cdx_attention`, corresponding
   logits/probabilities from each `seed_<seed>_model_arrays.npz`, and `prior`
   from `location_prior.npz`. Validate file and dtype/shape/array hashes through
   the existing archive and manifest machinery, including prior construction
   provenance. No object arrays, display-normalized overlays or rounded tables
   may replace the raw evidence.
4. Preserve archive row order. Match saved prediction rows by exact identity,
   checking every identity field and hard label before alignment. Reject
   missing/extra/duplicate nodules, patient/series changes, wrong splits,
   duplicate/missing seeds, different checkpoints and nonfinite data. Never
   inner-join away a mismatch. Check all 183 rows and all 99 patient identities
   before any metric; agreement of counts alone is insufficient.
5. The frozen 64-by-64, single-channel image models have three factor-two
   encoder reductions: expected native DSA shape is `[183,1,8,8]`, the native
   soft-mask shape is `[183,8,8]`, and prior shape is `[8,8]`. This follows
   from source/configuration, not a new server-array inspection. Validate these
   shapes against the archived metadata; stop on disagreement, without grid
   adaptation. Inputs must be the original unaugmented float32 `[183,1,64,64]`
   CT tensors, finite and within the existing `[0,1]` input range. Native masks
   are finite in `[0,1]`; maps/prior are finite, nonnegative and unit spatial
   mass within the existing absolute `1e-6` guard. Do not modify source arrays.
6. Preserve the frozen mask's area-resize convention. For a binary target use
   the **unnormalized area-resized soft mask** at `>= 0.5`, never the unit-mass
   target distribution. Load the existing training-only prior: it was made by
   area-resizing and separately unit-normalizing each training mask, then
   averaging and normalizing numerical drift once. No recomputation, test-mask
   fitting, new threshold or alternative prior selection is proposed.
7. Seal source/input hashes before reading, and recheck them before completion.
   Reuse persisted-best validation and pinned external-source checks for any
   later perturbed-input execution. A source or lock mismatch fails closed,
   without filtering, fallback models or a scientific completion marker.

The following sections are **new recommended decisions for review**, not claims
that these choices were frozen in the original brief. All denominators retain
the complete primary cohort. No method treats 549 seed-nodule evaluations as
549 independent observations or 297 seed-patient appearances as 297 patients.

## 2. Native-grid pointing-game adaptation

### Method, estimand and exact computation

The original pointing game asks whether an attention maximum falls within an
annotated object. Here the proposed endpoint is native-grid nodule localization
agreement, not diagnosis accuracy or a causal-faithfulness measurement.
Unlike the original class-cued maps, the repository's DSA is not a separate
class-specific explanation. Its use here, the coarse-grid target and tie rule
are explicit adaptations. We do not import the original paper's interpolation,
15-pixel tolerance margin or category averaging.
[Zhang et al., ECCV 2016, section 4.1](https://arxiv.org/html/1608.00507#S4.SS1).

For each nodule `i`, model `m` in `{B1, complete}` and seed `s`:

1. Use the saved native DSA array `A[i,m,s,r,c]`, without smoothing, upsampling,
   a top-percentile threshold or a new forward. Use the identical archived
   native grid for the fixed prior, evaluated once rather than once per seed.
2. Define `B[i,r,c] = 1{soft_masks_native_grid[i,r,c] >= 0.5}`. This is native
   cell membership; it does not assert that every pixel in that input tile is
   lesion tissue. Do not dilate the mask or grant a distance tolerance.
3. Among cells exactly equal to the maximum stored map value, choose the
   smallest row-major flat index `j = 8*r + c`. No epsilon grouping, random
   tie breaking or favorable tied-cell selection. Record all maximizing
   indices, tie count and chosen `(r,c)`. A constant unit-mass map is valid,
   has 64 tied maxima and chooses `(0,0)`; flag its degeneracy explicitly.
4. `hit[i,m,s] = B[i,r_max,c_max]`. Report
   `PG[m,s] = sum_i hit[i,m,s] / 183`, with numerator, denominator and patient
   count. Report the prior's analogous score once. Paired descriptive
   differences are `complete - B1`, `complete - prior`, `B1 - prior`, each
   computed from the same nodule records separately for seeds 42/43/44.
5. The known empty binary target
   `LIDC-IDRI-0052__nodule-49c6132bfb77f3f1` stays in all relevant rows and in
   denominator 183. An empty target produces `hit=0` with
   `reason=empty_native_binary_target`, not a silent exclusion or repaired
   foreground. Apply the same rule to any empty target, without an ID-specific
   selection branch. Record the flagged case's observed foreground count;
   contradiction of its supplied empty-target evidence requires review before
   interpreting the supplement. Empty-target misses are target-resolution
   limitations, not proof that the model looked outside a visible lesion.

Report target availability as nonempty/empty target counts over all 183, but do
not replace the primary denominator by a nonempty-only denominator. Give each
seed's full score and paired differences; an arithmetic mean and sample SD
(`ddof=1`) of the three seed scores is descriptive only. Prior repetition is not
three prior fits. There is no binomial nodule-independence interval, p-value or
significance star for this endpoint. Patient IDs remain attached to every row.

### Outputs and synthetic validation plan

Export all `183 * (2*3 + 1) = 1,281` map/nodule records, including 7 records for
the flagged nodule, plus per-seed summaries and 9 paired descriptive summaries.
Each record includes all identity fields, model/seed (null seed for the one
prior), map and target array hashes, peak/tie indices and values, native target
foreground count, hit, empty/constant flags and explicit reason. No existing
Dice/IoU/KL/correlation/overlap result is recomputed or replaced.

Later CPU-only synthetic tests must cover unique peaks inside/outside, ties on
opposite sides of a mask, all-cell ties, exactly `0.5` mask values, empty/full
targets, and a tiny positive soft mask that becomes empty after the frozen
area-resize/binarization. Demonstrate that using a normalized distribution
instead of a soft mask would be rejected. Check all-seed pairing, flagged-case
retention, denominators, raw-array immutability, bad shapes/nonfinite values,
identity/hash failures and deterministic export round trips. Zero model
forwards are needed for pointing-game, including later real-data execution.

## 3. Fixed-order insertion/deletion response analysis

### Scope, baseline, ranking and spatial mapping

Recommend an insertion/deletion adaptation of RISE's probability-versus-
perturbation curves and trapezoidal AUC. This measures model response to a
specified artificial input intervention. It is baseline-dependent and can
create out-of-distribution images; it is not clinical causality, proof of
attention-loss contribution or a comparison of diagnostic performance.
[Petsiuk, Das and Saenko, BMVC 2018](https://arxiv.org/abs/1806.07421).

Use only the same complete/B1 models, three seeds and 183 nodules. No ablation
forwards, gradients, saliency generation or tuning is proposed. The following
choices are fixed recommendations to be approved, not result-selected options:

- Baseline image `b` is a single-channel float32 all-zero `[1,64,64]` tensor
  in the model's existing normalized input coordinates, used for **both**
  insertion and deletion. Zero is the lower input bound, not a claim of a
  physically exact air image. There is no Gaussian blur, sigma, padding,
  inpainting, image-dependent baseline or parameter search. Using the same
  baseline fixes identical endpoints and isolates ordering differences. This
  differs from RISE's blurred insertion example and is a proposed adaptation
  requiring review, with acknowledged black-tile/distribution-shift artifacts.
- Each native cell `(r,c)` maps to exactly the input indices
  `[8*r,8*r+8) x [8*c,8*c+8)`. These 64 nonoverlapping tiles cover all 4,096
  input pixels. This is a deterministic evaluation tessellation, not a claim
  that a DSA cell's CNN receptive field is restricted to that tile. Do not
  interpolate maps or rank individual pixels within a tile.
- Define three full permutations of native cells: the model's saved native
  DSA, the single frozen training-only spatial prior, and one reproducible
  non-saliency ordering. For DSA/prior sort by descending raw map value, then
  ascending flat index. Exact ties only; include every cell, including zeros.
- For the non-saliency control, compute SHA-256 for the UTF-8 encoding of the
  JSON array `["phase5-supplement-v1-order", patient_id, nodule_id,
  series_instance_uid, j]` for each integer `j` from 0 through 63. Serialize
  with `ensure_ascii=True`, separators `(',', ':')`, no whitespace/newline.
  Sort ascending by the 32 unsigned digest bytes, then `j` if digests tie.
  This is a fixed pseudo-random ordering, not an estimated random-order mean.
  Export every digest/permutation. It is the same for a nodule across models
  and seeds; no label, image value, mask, model output or saliency enters it.
- Rankings are fixed from the unperturbed saved evidence, separately for each
  model/seed's DSA. Never re-rank after perturbation. The prior and ordering
  control are applied to **each** model, not treated as stand-alone classifiers.
  Test masks enter neither ranking nor any forward. Perturbation indicator
  arrays described below are generated from tile order only, not lesion masks.

The author's reference implementation fixes the output class at the original
prediction and integrates a fixed-order score curve. Its RGB/CUDA example is
not a drop-in implementation for this grayscale CPU-compatible protocol.
[RISE authors' evaluation code](https://github.com/eclique/RISE/blob/master/evaluation.py).

### Algorithm, endpoint and aggregation

Use eval mode, frozen parameters, no gradients/autocast and existing
deterministic source/runtime settings. Later execution must first replay one
unperturbed input per model/seed/nodule against its saved logits/probabilities,
using the existing per-class rule
`abs(actual-saved) <= 1e-5 + 1e-4*abs(saved)`. Also require the saved predicted
class and replayed argmax to agree; a near-tie disagreement stops evaluation,
not a switch of target. Do not modify saved predictions to achieve agreement.

This is a **global barrier**: all 1,098 unperturbed replays and argmax checks
across both models, all three seeds and every nodule must pass before the first
baseline or interior perturbed forward. Cache their validated raw outputs and
fixed targets, seal that cache to the source/checkpoint/input hashes, and reuse
it as the unperturbed endpoints. A failure anywhere stops the whole perturbation
stage; do not complete one model/seed's curves while other replays remain
unchecked. The barrier reorders the scheduled calls, adding no replay calls.

For model `m`, seed `s`, nodule `i`, fix target `c*` to its validated saved
unperturbed predicted class (the existing argmax/class-order rule). Keep that
class for all orders, both directions and all steps, including wrong original
predictions. Do not select the true class or new maximum at each step. The
endpoint is the model's primary-head softmax probability of `c*`, not the
ordinal output, attention value, logit or probability renormalized by endpoints.
For every fresh unperturbed, baseline and interior output, require a nonempty
float32 probability array of shape `[N,3]` in the fixed class order, with finite
entries in `[0,1]`. On a private float64 copy for checking, accumulate each
row sum in float64 and require `abs(sum-1) <= 1e-6` (`rtol=0`). Reject invalid
shapes, values or mass before any downstream calculation; no clipping, dropped
rows or normalization. Preserve the raw float32 values, predictions and fixed
targets. Float64 promotion for AUC arithmetic is permitted, but the probabilities
themselves are not renormalized. Do not route fresh outputs through the shared
metric helper's stricter `1e-8` mass check. Existing saved-prediction and replay
validators remain unchanged. Export each raw row sum/deviation and the checking
policy alongside the raw values; nonfinite logits also fail closed.

Let `M_k` be the binary input-tile indicator for the first `k` cells of the
fixed permutation, for every `k = 0,...,64`. Then:

```text
x_ins(k) = M_k * x + (1 - M_k) * b
x_del(k) = (1 - M_k) * x + M_k * b
t_k = k/64
y_direction(k) = p_m,s(c* | x_direction(k))
AUC_direction = sum[k=1..64] (t_k-t_(k-1)) * (y(k-1)+y(k))/2
```

One complete tile changes per step; there are 65 points per curve, including
both endpoints. Do not truncate low/high-response curves, assume monotonicity,
subtract the baseline score, divide by an endpoint difference or change steps
after results. High insertion AUC / low deletion AUC describes the conventional
direction of rapid recovery / rapid loss of this fixed target's response only.

For every model/seed/order/direction, report the arithmetic nodule mean of AUC
over all 183, median and linear 25th/75th percentiles, with 99 patient IDs
accounted for. Paired within-model descriptive differences are DSA-minus-prior
and DSA-minus-ordering-control for both AUCs. Also report each mean curve at the
65 fixed fractions. A constant input or baseline-identical input stays in the
denominator, with its flat/equal-curve flag. Nonmonotonic curves remain intact.

There are 36 model/seed/order/direction summaries and 24 within-model paired
AUC differences. Show every seed, optionally summarized by the mean and sample
SD of its three seed means, never by a pooled independent-observation analysis.
Complete/B1 target classes may differ for a nodule: their raw AUCs are not a
same-class causal contrast. Do not add a cross-model significance test or
claim superiority from their AUC difference. All perturbation results are
descriptive; no extra hypothesis family is implied.

### Exact proposed computational budget and outputs

For each of `183 * 2 * 3 = 1,098` nodule/model/seed units, the three orderings
and two directions require `6 * 65 = 390` curve positions. Cache only the
unperturbed endpoint and baseline endpoint within that unit, sharing them
between all six curves. Evaluate every interior position without additional
cross-order or cross-nodule deduplication:

```text
per unit forward evaluations = 2 + 2*3*(64-1) = 380
per trained seed             = 183*2*380 = 139,080
all three seeds              = 417,240 forward evaluations
  unperturbed replay         =   1,098
  baseline endpoint          =   1,098
  interior perturbed inputs  = 415,044
new non-original inputs      = 416,142
curves                       = 6,588
exported curve positions     = 428,220
```

Recommend batch size one, making these also the exact forward-call counts for
one successful run, with no automatic retries. No extra model calls are needed
for pointing, correctness tests or intervals. Even identical input/baseline
values do not change this scheduled count. Later execution logs actual calls,
cache references and failed attempts. **Runtime is unmeasured**: these scheduled
call counts do not establish wall-clock feasibility or distinct tensor counts.
Batch size one is an engineering choice, not an endpoint definition. Before a
future full run, a separately authorized synthetic timing exercise on the
intended device must report model/input shapes, warm-up, measured calls,
extrapolation and limits of representativeness. None is run in this revision.
Any batching/caching optimization requires declared numerical validation and
review, without changing cases, tiles, steps, directions, targets or controls.
This task makes **zero** model calls and introduces no baseline search.

Export per-cell orders/digests, each unit's fixed target, baseline specification
and tensor hash, input/map/source/checkpoint hashes, step/fraction/tile index,
three logits/probabilities and target response, endpoint cache identity, and
per-nodule AUCs and differences. Retain native maps and originals by immutable
source reference, not overwritten copies. Record replay discrepancies and fail
before perturbation if replay fails; invalid/nonfinite outputs or a missing
step invalidate the analysis, not just that nodule. Completion requires all
curves and denominators, not a partial best-looking subset.

### Synthetic validation plan

Later CPU fixtures will use a tiny image-only deterministic model with known
tile contributions and a fixed three-class softmax. Verify mapping coverage,
stable ties, hash-order golden vectors, identical controls across seeds/models,
65-point endpoints and hand-integrated constant/linear curves. Test a case
whose predicted class changes after perturbation to prove `c*` stays fixed.
Changing evaluation masks must change neither order nor model-call tensors;
the forward signature must accept images only. Changing perturbed attention
must not change rankings. Check original-array immutability, all-zero images,
nonmonotonic curves, per-seed pairing, call-count/cache accounting, forbidden
source mutation, replay failure and incomplete-curve failure. Golden synthetic
exports must replay AUCs at float64 `atol=1e-12, rtol=0`, without model calls.
Add a late failing unperturbed replay to prove that **zero** baseline/interior
calls occur until the entire replay barrier passes. Include float32 probability
vectors whose float64 row-mass errors lie between `1e-8` and `1e-6`: they must
be accepted unchanged, with AUC computed from those raw values. Reject excessive
mass errors, zero-mass rows, wrong/empty shapes, negative/out-of-range entries,
NaN and infinity. Verify diagnostics and the absence of clipping, normalization
or calls to the strict shared probability helper.

## 4. Secondary patient-weighted correctness: formal testing blocked

### Scope revision and verified reference

The initial proposal recommended the **Durkalski method-of-moments clustered
McNemar test**, separately per seed, for complete versus B1 and each OFF model.
Its use for formal inference is now **blocked by a known calibration failure**.
The formula below remains the correctly transcribed diagnostic reference, not
an approved test. No replacement is claimed validated in this revision.
Independent nodule-level McNemar is inappropriate when a patient contributes
multiple paired outcomes. The primary paper proposes a cluster variance
adjustment accommodating unequal cluster sizes and within-cluster dependence.
[Durkalski, Palesch, Lipsitz and Rust, Statistics in Medicine 2003,
22:2417-2428](https://doi.org/10.1002/sim.1438).

Reference verification for this proposal is **read-only source inspection, not
an executed numerical validation**. CRAN `clust.bin.pair` version `0.1.2` was
identified from its package metadata; the source mirror is pinned to commit
`e464ac7e5d094bffd25b7c4cbba67820b60f1cc1`. Its
[Durkalski implementation](https://github.com/cran/clust.bin.pair/blob/e464ac7e5d094bffd25b7c4cbba67820b60f1cc1/R/durkalski.R)
and [public dispatch/tail calculation](https://github.com/cran/clust.bin.pair/blob/e464ac7e5d094bffd25b7c4cbba67820b60f1cc1/R/clust.bin.pair.R)
were checked against the formula below. Explicitly select
`method="durkalski"`; the package default is `yang`, and its `mcnemar` option
is not cluster-adjusted. Version and count-cell meanings are also documented
in the [CRAN reference manual](https://stat.ethz.ch/CRAN/web/packages/clust.bin.pair/clust.bin.pair.pdf).
Runtime comparison to this pinned reference remains a required synthetic test
of transcription, not proof of finite-sample validity.

**Secondary estimand amendment requiring approval:** equal patient weighting
asks about the mean within-patient nodule-correctness difference when a patient,
rather than a nodule, receives equal weight. This complementary question avoids
letting patients with more annotated nodules dominate the summary. That is a
scientific rationale for a secondary endpoint, not a reason to replace primary
macro-F1 or existing nodule-weighted accuracy. Keep their definitions and
reporting priority unchanged. No patient diagnosis, majority vote or new label
is created. Report both weightings transparently; do not choose whichever looks
more favorable or use this secondary endpoint to rescue a primary-endpoint
claim. The weighting amendment and a valid inferential method are separate
decisions, both still requiring review.

### Exact input, estimand and algorithm

Use only the validated saved primary test predictions from section 1. For each
model `m`, seed `s`, patient `p`, nodule `j`, define
`C[m,s,p,j] = 1{saved_predicted_class == frozen_true_class}`. Multiclass
classification is reduced to paired correctness, not to a new disease label.
All nodules, including indeterminate labels and concordantly wrong outcomes,
remain. Each contrast is paired by the same sample identities and seed.

For complete model `A` versus comparator `B`, count within each patient:
`a_p = both correct`, `b_p = A only correct`, `c_p = B only correct`,
`d_p = both wrong`, and `n_p = a_p+b_p+c_p+d_p > 0`. Then:

```text
u_p = (b_p - c_p) / n_p
delta_patient = (1/P) * sum_p u_p                 [P = 99]
H0: E_patient[u_p] = 0; H1: E_patient[u_p] != 0
Q = (sum_p u_p)^2 / sum_p (u_p^2)
p_reference = Pr[ChiSquare(df=1) >= Q]            [diagnostic; not approved]
delta_nodule = sum_p (b_p - c_p) / sum_p n_p      [N = 183; descriptive]
```

For mathematical/synthetic diagnosis use the reference statistic without a
continuity correction, finite-sample factor, sign-selected alternative or
macro-F1 substitution. Compute the
chi-square survival function directly to avoid subtractive cancellation in
`1-CDF`; compare ordinary-range p-values and all statistics to the reference.
No logit/probability threshold is changed. A future valid test of this null
would concern secondary patient-weighted correctness only, not macro-F1 or
proof of the benefit of a particular loss mechanism. The current reference
tail probabilities are not authorized as research p-values.

Patients are the independent sampling clusters; within-patient dependence is
allowed. Model fits and the three training seeds are conditioned on, not sampled
as independent patients. This is an asymptotic test, not an exact test or a
guarantee of external-population representativeness. Report all cluster sizes,
total discordant nodules, patients with discordance and patients with nonzero
`u_p`; unequal nodule burden can change the two estimands' signs.

### Undefined cases and the failed feasibility safeguard

Reject malformed/missing counts, unequal identity sets, empty patient IDs and
noninteger/negative cells as integrity errors. Never use complete-case deletion.
If `sum u_p^2 == 0`, return null statistic/p-value with
`reason=zero_cluster_contrast_second_moment`, even if discordances cancel
within patients. Retain the zero effect and every patient; do not manufacture
`p=0`, invoke naive McNemar, or count only discordant nodules as the sample.
If `sum u_p=0` but the second moment is positive, `Q=0, p_reference=1` is
mathematically defined; this does not remove the calibration block.

The original proposal's gate of at least 20 nonzero patient scores is not a
sufficient calibration safeguard: the counterexample below passes it in every
realization. It is retained only as a historical diagnostic comparison in the
stress plan, not a readiness rule. Do not raise it until one adverse case is
excluded. All 99 patients and 183 nodules remain in the proposed estimands.

### Independently verified finite-null counterexample

Use hypothetical independent patients, not the frozen cohort's actual cluster
sizes or model errors. Twenty patients have four nodules each: independently,
with probability `1/5`, counts `(a,b,c,d)=(0,4,0,0)` give `u=1`; otherwise
counts `(3,0,1,0)` give `u=-1/4`. Another 24 patients have two concordantly
correct nodules each, and 55 have one, all with `u=0`. Thus `P=99`, `N=183`,
and each patient's expected score is zero. Exactly 20 scores are nonzero.
This is an allowed asymmetric mean-zero null; symmetry was never assumed.

For `K ~ Binomial(20,1/5)`, the reference formula reduces to
`Q(K)=(5*K-20)^2/(20+15*K)`. Before editing this document, the revision
independently enumerated all 21 values with standard-library Python: derive
`Q` from the 99 rational scores, assert the formula agrees, compute
`p_reference=erfc(sqrt(Q/2))`, and sum the exact rational masses
`comb(20,K)*4^(20-K)/5^20` for each rejected value. The masses sum exactly to
one. Only the chi-square tail comparison uses floating-point arithmetic; no
tail value is near either threshold. Results:

- At `alpha=0.05`, reject `K in {0,1,9,...,20}`. Exact probability is
  `7549007091249/95367431640625 = 0.0791570765971351`, or **7.9157%**.
- At `alpha=0.05/15=1/300`, reject `K in {0,13,...,20}`. Exact probability
  is `1100957668913/95367431640625 = 0.0115443778863812`, or **1.1544%**,
  compared with the nominal **0.3333%** first-Holm threshold.
- `K=0` alone has probability `(4/5)^20=0.0115292150460685`, gives `Q=20`
  and `p_reference=7.74421643104408e-6`. The largest rejected / smallest
  nonrejected p-values are respectively `0.04463816 / 0.09096895` at 0.05
  and `0.002147949 / 0.004677735` at 1/300, separating numerical rounding
  from the substantive failure.

These are **marginal** rejection probabilities under one hypothetical null,
not the joint 15-test family-wise error rate. The latter is unknown here and
requires a joint test distribution; neither multiplication by 15 nor an
independence formula supplies it. The counterexample establishes a known
finite-sample calibration failure of the proposed rule, not that the published
asymptotic method is universally invalid or that real experiment p-values are
known. This exact mathematical check is the only executed statistical check
in this documentation revision.

### Recommended scope decision and failure consequences

Keep the broad patient-weighted mean-zero question and all 15 registered slots,
but **do not implement or execute formal research tests under the current
proposal**. The original McNemar/inferential-completion requirement remains
open. This is a proposed deferral of inferential scope for review, not removal
of the requirement or satisfaction by descriptive results. Noninferential
endpoints may proceed only under an explicitly approved partial-scope task;
the full supplemental protocol is not ready for implementation/execution.

Do not substitute naive McNemar, select a method from actual model errors,
or relabel a sign-flip test as exact for this broader mean-zero null: symmetry
or exchangeability would be additional assumptions and an explicit scope
amendment, not consequences of a zero mean. No such narrowing is proposed here.
Before unblocking, require a method/assumption/estimand justification from
primary sources, a pinned and numerically checked reference, reviewed coverage
of the stress cases below, and a new explicit approval. A finite stress suite
cannot prove validity for every permitted null. Merely matching reference code,
raising a score-count cutoff or failing to detect inflation cannot approve it.

### Outputs and synthetic validation plan

For all 15 contrast/seed combinations export paired nodule correctness and
`(a,b,c,d,n,u)` for all patients, signed patient- and nodule-weighted effects,
and method/status metadata if a descriptive partial task is later approved.
Research raw/adjusted p-values must remain null with
`reason=method_calibration_unresolved`; do not compute research `Q`/reference
tails merely to attach a caveat. Keep reference statistics/tails confined to
the hypothetical validation records until a method is approved. The common
hash/provenance contract applies; no model forward is needed for correctness.

Future unit tests must compare method-core calculations with the pinned R
implementation on synthetic unequal-size clusters, including within-patient
discordance cancellation and all-concordant patients. One hand-check is the
four `(a,b,c,d)` rows `(0,1,0,0), (0,0,1,0), (1,1,0,0), (3,0,0,0)`:
`u=(1,-1,1/2,0)`, `delta_patient=1/8`, `delta_nodule=1/7`, `Q=1/9`.
Repeating these four synthetic cluster types eight times with distinct patient
IDs gives 32 patients, 24 nonzero scores and `Q=8/9`, checking arithmetic rather
than certifying a feasibility gate. Swapping models negates the effect but
preserves `Q` and the reference tail; row permutations preserve results.
Uniformly duplicating each patient's
nodules leaves this patient-weighted statistic unchanged, unlike naive
nodule-level McNemar. Test null reasons, invalid cells/identities, correct
inclusion of concordant nodules in `n_p`, and no across-seed pooling.

### Prespecified null stress cases: future validation only

The following 52 settings are fixed for review before observing actual errors.
All patients are independent, all expected patient scores are zero, all nodules
are retained, and mirrored settings swap `b` and `c`. They test a candidate,
not provide a menu for selecting a favorable method. No campaign is run now.

1. **Six original symmetric settings retained.** Use 99 patients with sizes
   `n_p=1+(p mod 4)` for zero-based `p`. Cross discordance probabilities
   `{0.1,0.3,0.6}` with independent nodule discordance/sign or a single shared
   discordance/sign per patient. Signs are fair; concordant pairs are both
   correct/both wrong with equal probability. These hypothetical sizes total
   246 nodules, not the actual cohort; they never replace research membership.
2. **36 asymmetric settings.** Choose `(d,M)` from `d=2` with
   `M in {1,2,5,10,20,40,84}`, `d=4` with `M in {1,2,5,10,20,28}`, and
   `d=8` with `M in {1,2,5,10,12}`, each with both sign orientations. The
   first `M` patients have `d` nodules: `(0,d,0,0)` with probability
   `1/(d+1)`, otherwise `(d-1,0,1,0)`. Thus scores are `+1` and `-1/d`
   with mean zero. Among the remaining `99-M` concordantly correct patients,
   the first `84-M*(d-1)` have two nodules, the rest one. Each setting has
   exactly 99 patients/183 nodules. It spans sparse informative clusters,
   unequal score magnitudes and multiple sizes; `d=4,M=20` is the verified
   counterexample, not a new unexamined case.
3. **Four heterogeneous settings.** H1 and its mirror have 12 patients of
   size 2, 12 of size 4 and 4 of size 8, each following its size's asymmetric
   law in item 2. Complete with 8 concordant size-2 and 63 concordant size-1
   patients. H2 and its mirror have one size-4 patient following that
   asymmetric law; another 27 size-4 patients independently have scores
   `+1/4,-1/4,0` with probabilities `1/200,1/200,99/100`, realized by counts
   `(3,1,0,0)`, `(3,0,1,0)`, `(4,0,0,0)`. Another 71 size-1 patients are
   concordantly correct. Both designs total 99/183. H2 concentrates most
   score variance in one patient without violating the mean-zero null.
4. **Six zero-inflated asymmetric settings.** Start with the counterexample's
   sizes. Independently activate each of its 20 size-4 patients with
   probability `rho in {0.05,0.25,0.5}`; active patients use the `d=4`
   asymmetric law, otherwise counts are `(4,0,0,0)`. Use both sign orientations.
   This retains 99/183 and mean zero while varying the number of nonzero scores
   and the chance of a zero second moment. Do not drop sparse realizations.

Require exact finite enumeration for items 2-4, aggregating counts of identical
patient types: binomial counts for item 2, products of binomial counts for H1,
and multinomial activation/sign counts for H2 and item 4. Use rational masses,
include every support point and verify total mass one. For item 1, plan
100,000 independent replicates per setting (an explicit increase from the
initial 10,000 for tail precision). Fix scenario indices 0-51 in the listed
order: item 1 by increasing discordance then independent/shared; item 2 by
increasing `d`, then `M`, then original/mirror; item 3 H1 original/mirror then
H2 original/mirror; item 4 increasing `rho`, then original/mirror. Any simulation
uses `Generator(PCG64(SeedSequence([20260916, scenario_index])))`. Exact settings
use no RNG. For simulated settings traverse replicate, patient, then nodule
in ascending order. Independent mode draws three uniforms per nodule for
discordance, sign and concordant correctness, including unused draws. Shared
mode draws discordance/sign once per patient, then one concordant-correctness
uniform for every nodule, including unused draws. Use `<` against the stated
probabilities; record this traversal and no outcome-dependent budget extension.

### Holm-tail checks, uncertainty and explicit stop rules

For **every** setting, check all 15 thresholds `alpha_j=0.05/j`, `j=1,...,15`,
including the first-Holm `1/300` and unadjusted `0.05` levels. Report marginal
`Pr(p_reference <= alpha_j)`, its difference from `alpha_j`, and zero-second-
moment/withheld frequencies. Evaluate both the unfiltered reference rule and
the historical 20-score gate, with withheld results counted as nonrejections
and their counts separately visible. Never use only selected passing
realizations as the denominator or infer joint research FWER from this grid.

For exact settings report exact masses, no Monte Carlo interval. For each
simulated count `R` out of `B=100000`, report `R/B` and a two-sided
Clopper-Pearson interval at confidence `1-0.01/(52*15*2)`. The fixed denominator
includes both diagnostic rules, giving a conservative simultaneous 99%
Monte Carlo uncertainty envelope over the planned grid. Set the lower limit
to 0 when `R=0`, and the upper limit to 1 when `R=B`; obtain the other limits
through the exact binomial reference procedure. Independent simulation
replicates justify these
binomial limits, not independence of real nodules or of the 15 research tests.
[R's documented exact binomial interval](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html)
cites [Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404).

An exact rejection probability above its nominal threshold is a known failure;
a simulated lower bound above it is evidence of failure. A simulated interval
extending above the threshold is inconclusive, **not a pass**. An upper bound
below the threshold supports only that generator/threshold, not the entire
broad null. Missing settings, numerical failures or absent tail checks also
block readiness. Retain all adverse results; no post-result cutoff changes,
method shopping or narrowing of assumptions. The independently verified case
already activates the failure rule at both stated thresholds for both
diagnostic rules. Do not wait for a campaign to acknowledge the block, or claim
that later favorable settings undo it. Any alternative remains a new reviewed
proposal with its assumptions and scope changes explicit.

## 5. Multiplicity: one enumerated paired-correctness family

The registered question for currently blocked, secondary supplemental testing
is whether the complete model's patient-weighted paired correctness differs
from its fair baseline or any of its four single-component OFF comparators,
conditional on each trained
seed. These are related opportunities to claim a complete-model advantage.
Recommend **one family `F_correctness`, size 15**, not separate families for
seeds, favorable endpoints, baseline versus ablations, or directions of effect.

Every member below has endpoint `delta_patient`, null zero, two-sided
alternative nonzero and the full 183-nodule/99-patient test cohort. Their
inferential method is **unresolved** following section 4's failure; preserve
the slots rather than running the failed reference test. The ID suffix is the
trained seed:

- `C_B1_42`, `C_B1_43`, `C_B1_44`: complete minus B1, respectively seeds
  42, 43, 44.
- `C_attention_off_42`, `C_attention_off_43`, `C_attention_off_44`: complete
  minus `attention_alignment` OFF, respectively seeds 42, 43, 44.
- `C_disagreement_off_42`, `C_disagreement_off_43`,
  `C_disagreement_off_44`: complete minus `disagreement_weighting` OFF,
  respectively seeds 42, 43, 44.
- `C_ordinal_off_42`, `C_ordinal_off_43`, `C_ordinal_off_44`: complete minus
  `ordinal_auxiliary` OFF, respectively seeds 42, 43, 44.
- `C_fusion_off_42`, `C_fusion_off_43`, `C_fusion_off_44`: complete minus
  `hierarchical_fusion` OFF, respectively seeds 42, 43, 44.

This is the entire proposed hypothesis register. B0 retains its original
reproduction/baseline role, but is not an additional fair-comparison test here;
no B0/B1 definition changes. No all-toggles-OFF model is substituted for B1.
There are no formal hypotheses for macro-F1, per-class metrics, pointing hits,
insertion/deletion AUCs, uncertainty or boundary results, and no extra pooled
seed test. Pointing contrasts, the 24 within-model AUC contrasts and accuracy
intervals below are explicitly descriptive and cannot carry significance
claims. This testing scope is proposed for review, not evidence that an
unspecified original hypothesis was previously waived. Adding a hypothesis
requires a reviewed family revision, not post-result selective testing.

Once a justified test is separately approved, the proposed adjustment remains
Holm step-down at two-sided family-wise alpha `0.05`. Sort
the 15 raw p-values ascending (ties by registered ID); for sorted position
`i` define
`p_adj(i) = min(1, max_{j<=i} [(15-j+1)*p_raw(j)])`, then restore ID order.
For a defined test report rejection only when its adjusted p is `<= 0.05`.
Holm accommodates dependence among valid p-values, so reused patients/seeds
do not require an independence fiction. Its nominal protection does not
repair asymptotic miscalibration or retrospective selection of this protocol.
[Holm, Scandinavian Journal of Statistics 1979, 6:65-70](https://www.jstor.org/stable/4615733).
In particular, it does not repair the known counterexample. The 15-slot family
is preserved, but has no approved research p-values or established joint FWER
under the failed proposal. No adjusted inference extends to primary macro-F1,
nodule-weighted accuracy intervals or localization endpoints by association.

For a prespecified statistical undefined/withheld result, keep its slot in
the family. Substitute `1` **only inside adjustment bookkeeping**, retain
null raw/adjusted p-values and `reject=false` in its public result, and export
the placeholder and reason separately. Do not shrink the family or pass only
defined tests to an adjustment routine. Invalid p-values outside `[0,1]`, a
missing hypothesis record or a source/integrity failure is an execution error,
not a statistical null; it prevents a valid completed testing bundle.
The current method-level calibration block is not ordinary per-test degeneracy:
retain all slots with `method_calibration_unresolved`, null raw/adjusted p-values
and no inferential claim. Bookkeeping placeholders cannot turn the unresolved
McNemar requirement into a completed testing family.

Export the complete hypothesis register, fixed family size, alpha, raw and
adjusted values, ranks, placeholder flags, defined/withheld counts and source
hashes. Synthetic tests must check hand examples, ties, shuffled input,
all-ones/zero p-values, negative/nonfinite rejection, all-undefined and one-
undefined families, missing/duplicate IDs and no family-size reduction.
Compare to `stats::p.adjust(..., method="holm", n=15)` after explicit
placeholder construction, recording the R version. The
[official R adjustment documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
provides the reference behavior. No p-value may be reconstructed from an old
confidence interval, standard deviation, rounded point estimate or seed mean.

## 6. Accuracy intervals and limits on thesis claims

The existing `phase4_bootstrap.py` metric list is macro-F1, benign F1,
indeterminate F1/recall and malignant F1; it does **not** supply accuracy
intervals. A thesis claim reporting accuracy uncertainty would need the
corresponding nodule-accuracy interval; a paired comparative accuracy claim
would also need the accuracy-difference interval, not subtraction of two
separate CI endpoints. A patient-weighted correctness p-value cannot stand in
for inference about the nodule-weighted difference. This identifies missing
evidence, not a claim that any particular uninspected thesis sentence is valid.

Recommend one separate, saved-prediction-only accuracy supplement: 18 per-model
intervals (complete, B1, four OFF models, each at 42/43/44) and 15 paired
complete-minus-comparator intervals with the exact contrasts/seeds above.
No rerun of the existing bootstrap endpoint suite, no editing its metric list
and no replacement of published intervals. These 33 new intervals are also
**exploratory, descriptive, unadjusted 95% intervals**, not extra hypothesis
tests, simultaneous coverage or a backdoor superiority decision.
They are conditional on the trained fits and do not jointly have simultaneous
95% coverage. Exclusion of zero from one interval is not a Holm-adjusted finding.
Do not use disagreement with a secondary patient-weighted result to select the
preferred weighting, endpoint or conclusion.

Use the existing patient-cluster sampling convention: sort the 99 unique
patient IDs lexicographically; with NumPy `Generator(PCG64(20260914))`, draw
99 patient indices with replacement for each of 10,000 replicates, retaining
every nodule for each occurrence. Use identical draws for every model, seed
and contrast; export the ordered patient list and draw matrix/hash. Reuse a
validated identical draw ledger if available, otherwise create only this
supplement's ledger. No outcome-based stratification, seed resampling or
discarding a replicate because a class is absent.

For replicate patient multiplicities `w_p`, use
`accuracy* = sum_p w_p * correct_count_p / sum_p w_p * n_p`; the paired
difference is the two accuracies on that **same** draw. Take NumPy linear
quantiles at `0.025` and `0.975`. This ratio remains nodule weighted; do not
replace it by the mean of patient accuracies. Denominators must be positive;
all-correct/all-wrong or constant differences produce valid degenerate
intervals. Missing predictions are integrity errors, not imputed failures.

The source for this choice is the already-frozen
[patient-bootstrap contract](phase4_evaluation.md#patient-bootstrap-and-ablations)
and `src/consendxformer/phase4_bootstrap.py`, not a newly optimized resampling
method. Export all 10,000 estimates, point/interval/denominator definitions and
pairing hashes in the new bundle. Later synthetic tests must demonstrate
whole-cluster duplication, unequal patient sizes, common paired draws, fixed
RNG/quantiles, class-absent replicates, deterministic replay and the absence of
old-suite calls. Do not average CI endpoints across trained seeds.

## 7. Proposed output and provenance contract

Reserve a distinct new namespace, proposed as
`artifacts/phase5/supplemental_evaluation/phase5-supplement-<identity_hash>/`.
This is a future layout, not a created directory or an implemented command.
Reject an existing destination, including a failed partial bundle, and any
symlink/source/input/completed-bundle overlap. Preserve every earlier success
or failure. No new output is written within a Phase 4 result bundle.

The future bundle should contain:

- `protocol.json` and this approved document's content/hash, with
  `analysis_status=exploratory_post_hoc`, `preregistered=false`, proposal and
  approval dates/revisions, original requirement chronology, approved estimand
  amendment, all numerical policies, hypothesis register and computational
  budget. Record the review, verified finite-null failure, blocked formal-test
  status and any explicitly approved partial scope. Changes after approval
  require a new version and explanation.
- `provenance/provenance.json`: separate proposal/implementation/execution
  commits, clean branch/HEAD, exact 18 classification-source descriptors,
  checkpoint/config/prediction/manifest/completion hashes, matched-attention
  source hashes and raw-array metadata, prior training provenance, frozen
  data/split/supervision/input-lock snapshots, ordered test identity hash,
  pinned external baseline commit and unchanged-tree evidence. Record runtime
  versions/device/determinism, pinned R reference and later verification log,
  all replay checks and the global barrier/cache binding, raw float32
  probability validation/deviations, actual call counts, timing evidence (null
  while unmeasured), and source snapshots before/after. Later calibration
  evidence must bind the full scenario/threshold register, exact probabilities
  or simulated counts/uncertainty, RNG definitions and all failed checks.
- `tables/pointing_per_nodule.csv`, `pointing_per_seed.csv`,
  `pointing_paired_descriptive.csv`, and a flagged-case extract with references
  to the same underlying rows, not a new selected cohort.
- `orders.npz`, `tables/perturbation_curves.csv`,
  `perturbation_auc_per_nodule.csv`, `perturbation_per_seed.csv` and
  `perturbation_paired_descriptive.csv`, with the fields/counts in section 3.
- `tables/correctness_per_nodule.csv`, `correctness_per_patient.csv`,
  `hypothesis_register.csv`, `clustered_tests.csv` and `multiplicity.json`;
  the 15 planned tests stay visible even if some are statistically undefined.
- `accuracy_bootstrap.npz` for the draw ledger and replicate estimates,
  `tables/accuracy_intervals.csv` for all 33 planned intervals, and
  `evaluation.json` plus `summary.md` with all denominators, limitations,
  undefined reasons, per-seed variation and no confirmatory-superiority claim.
- `artifact_manifest.json` with file SHA-256 and canonical bundle hash, and
  `COMPLETED.json` written last only after required read-back, source/input
  immutability, all-row/all-seed checks and manifest validation. A valid
  statistical degeneracy may be `complete_with_undefined_tests`, with counts
  and interpretation limits; a source mismatch or incomplete inference cannot.
  That status presupposes an approved, justified inferential method: the known
  method-level calibration failure cannot qualify. Do not emit completion for
  the formal family or full supplement while it is unresolved. Any separately
  authorized descriptive-only sub-bundle must explicitly identify its partial
  scope and the unfulfilled McNemar requirement, never full completion.

NPZs must forbid pickle/object arrays and use the existing deterministic array
hash convention. CSVs must retain exact identities and round-trip numeric
precision; nullable statistical results require reason fields, not silent NaN
substitution. Completion binds the hypothesis register and every result hash,
not merely the presence of files. Package/reference source hashes are recorded
before later reference execution; do not fetch moving `master` at runtime.
Provenance validation must reject a changed policy, source, seed set or family.

Later integration tests use only synthetic CPU fixtures and isolated temporary
outputs. They must prove full-cohort pairing and all-three-seed export, reject
source/checkpoint/array/lock mutations, and cover read-back consistency,
undefined-test completion, no-overwrite, escaped/symlink paths and completion-
marker ordering. Run targeted and full CPU suites only in that later
implementation task. Add checks that a method-level calibration block cannot
produce inferential/full-protocol completion. No evaluator tests, R runtime
comparison or Monte Carlo validation are claimed completed here; only the
hypothetical exact enumeration reported in section 4 was executed.

## 8. Decisions requiring protocol review before implementation

1. Approve the native-grid pointing adaptation: exact peak ties resolved
   row-major, unchanged `>=0.5` target, all-183 denominator with empty targets
   explicitly scored as misses, including the known flagged case. This is
   spatial correspondence, not class-specific or causal explanation.
2. Approve zero-baseline insertion **and** deletion, whole 8-by-8 input tiles,
   64 fixed steps, saved predicted-class targets, prior and hash-order controls,
   descriptive within-model AUC comparisons and the 417,240-forward later-run
   budget. Baseline/OOD and cross-model target differences remain limitations.
3. Review the explicitly secondary patient-weighted estimand and the proposed
   deferral of formal testing. The Durkalski reference rule has a known
   asymmetric-null failure, not just untested calibration; the 20-score gate
   does not fix it. Review the expanded null/tail validation plan and require
   a justified method or explicit assumption/scope amendment before unblocking.
   The method remains unresolved; no replacement or narrowed null is approved.
4. Approve the single 15-test Holm family, all seeds/comparators, two-sided
   hypotheses, fixed treatment of undefined tests and the absence of formal
   localization/perturbation/B0 tests. Approval must not be represented as a
   recovered historical waiver or preregistration, and family approval alone
   cannot override the unresolved single-test calibration issue.
5. Approve the separate 33 descriptive accuracy intervals if accuracy uncertainty
   or comparative accuracy is to be reported, without rerunning the existing
   bootstrap suite. Resolve thesis wording to the correct estimand; neither
   these intervals nor retrospective multiplicity adjustment licenses a
   confirmatory superiority claim.
6. At later execution preflight, bind exact server source directories and hashes
   and verify archived shapes/replay permissions through the existing loaders.
   Those unavailable inputs are an execution gate, not a reason for new
   inference, source repair or another completed-analysis audit now.

Stop at proposal review. Implementation/synthetic testing and supplemental
real-data execution require their separate approvals; no commit, push or
server work accompanies this proposal.
