# Phase 5 paired-correctness inference: decision proposal

## Status, chronology and decision requested

Methods-only proposal, dated 2026-09-17, against
`ca6c018ffbedae12988074fb66052592212a3c2f` on `phase4-consendxformer`.
**Not an approved amendment, implementation, calibration result or execution
authorization. All 15 formal hypotheses remain unresolved.**

Read this alongside the unchanged
[supplemental proposal](phase5_supplemental_evaluation_proposal.md), especially
sections 4-6, and the completed
[pointing/accuracy](phase5_descriptive_subset.md) and
[insertion/deletion](phase5_insertion_deletion_subset.md) subset records.
The frozen proposal's LF-normalized SHA-256 remains
`4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`.
Those records distinguish historical proposals from subsequently audited
execution. Pointing/accuracy and insertion/deletion are completed subsets;
neither resolved formal inference. The full supplement and Stage A remain
incomplete. No other evidence-gap status is changed here.

Recommendation for review: consider a **two-sided, bounded independent-patient
mean test**, with a Hoeffding tail bound and the existing 15-slot Holm family.
Its attraction is a finite-sample justification for the broad mean-zero null,
including asymmetry, rather than an improved-looking simulation result. Its
cost is substantial conservatism. This would explicitly replace the proposed
Durkalski chi-square calibration, not repair or rename it as McNemar. Approval
of the secondary patient-weighted question, the methodological amendment and
the sampling assumptions are separate decisions. Until those decisions and
later validation are complete, no research statistics or p-values are authorized.

The recommendation uses mathematical properties and the previously documented
hypothetical counterexample, not observed model contrasts, signs, intervals,
discordance counts or a search for significance. Reading the completed subset
documentation supplies chronology and preservation requirements, not a method
selection criterion. No research bundles, error vectors or checkpoints were
opened for this decision; no new enumeration or calibration campaign was run.

All new specifications are **exploratory post-hoc**, being proposed after test
exposure. A valid fixed-procedure probability bound would not retroactively
make the study preregistered or its conclusions confirmatory.

## 1. What must remain the same, and which question is under review

Macro-F1 remains the primary classification endpoint. Existing nodule-weighted
accuracy, its 18 individual and 15 paired descriptive intervals, and all other
completed evidence remain unchanged. The intervals are unadjusted, conditional
on fitted models, and are not tests or simultaneous coverage statements. Do not
derive p-values from their endpoints, seed SDs or published point estimates.

Keep every one of the 183 frozen test nodules and 99 patients, all models,
labels, validity rules, splits, configurations, selected best checkpoints,
source identities and seeds 42/43/44. No retraining, new forwards, new patient
diagnosis labels, majority vote, cohort filtering, seed pooling or selected
comparator is proposed. The same patients observed at three seeds are not
297 independent patients; 549 seed-nodule observations are not independent.

For one fixed seed and complete-minus-comparator contrast, let
`C_A,pj` and `C_B,pj` be correctness indicators from the unchanged saved hard
predictions and true nodule label. For patient `p`, retain all `n_p > 0` nodules:

```text
b_p = number correct only under A
c_p = number correct only under B
D_p = b_p - c_p = sum_j (C_A,pj - C_B,pj)
U_p = D_p / n_p                         in [-1, 1]
delta_patient_hat = sum_p U_p / P       P = 99
delta_nodule_hat  = sum_p D_p / N       N = sum_p n_p = 183
```

The blocked proposal concerns **secondary, equal-patient-weighted mean nodule
correctness difference**, not patient diagnosis accuracy. It is equivalent to
choosing a patient uniformly and then a nodule uniformly within that patient.
All nodules contribute, but their weights differ from nodule-weighted accuracy.
The two effects can differ when nodule burden and comparative correctness are
associated. Changing weights to obtain a preferred direction is prohibited.

For independent, identically sampled eligible patient clusters, the proposed
estimand is `delta_patient = E[U_p]`, and the two-sided null is
`H0: delta_patient = 0`. Within-patient dependence is unrestricted. Models and
their selection histories are held fixed; this is not inference over future
training seeds, refits, institutions or clinical deployment populations.

For an independent but non-identically distributed patient design, a precise
extension is `delta_design = (1/P) sum_p E[U_p]`, with null `delta_design = 0`.
Individual patient means need not each be zero. This is a finite-design mean,
not automatically a superpopulation effect. Review must specify which sampling
interpretation is justified; this document does not silently exchange them.

Patient-disjoint splits establish non-overlap, not a probability-sampling
design or independence theorem. A future methods review must assess whether
independent patient clusters, conditional on fixed fits and the specified
sampling frame, are defensible. In particular, conditioning on outcome-based
selection or fixed stratum totals can induce dependence. Counting 99 unique
IDs alone does not justify a nominal error guarantee. No numerical check of
research outcomes can supply the missing design assumption.

## 2. The established calibration failure is not a transcription error

The historical statistic and reference tail are:

```text
Q_D = (sum_p U_p)^2 / sum_p U_p^2
p_D = Pr(chi-square_1 >= Q_D)
```

The pinned `clust.bin.pair` implementation agrees with this formula. The
published clustered matched-pair adjustment is an asymptotic method, not a
distribution-free finite-sample guarantee for every allowed asymmetric null.
Agreement with software establishes transcription, not calibration.
[Durkalski et al. (2003)](https://doi.org/10.1002/sim.1438);
[pinned reference formula](https://github.com/cran/clust.bin.pair/blob/e464ac7e5d094bffd25b7c4cbba67820b60f1cc1/R/durkalski.R).

The following is the frozen proposal's previously exactly enumerated
**hypothetical null**, not a description of actual errors:

- Twenty independent size-four patients have `(a,b,c,d)=(0,4,0,0)` with
  probability `1/5`, giving `U=1`; otherwise `(3,0,1,0)`, giving `U=-1/4`.
- Twenty-four size-two and 55 size-one patients are concordantly correct,
  giving `U=0`. There are 99 patients, 183 nodules, and exactly 20 nonzero
  patient scores in every realization. Every patient's expected score is zero.
- For `K ~ Binomial(20,1/5)`,
  `Q_D(K)=(5K-20)^2/(20+15K)`. Exact support masses are
  `choose(20,K)*4^(20-K)/5^20` for `K=0,...,20`.

At nominal `0.05`, rejection occurs for `K in {0,1,9,...,20}`; the recorded
exact probability is `7549007091249/95367431640625`, approximately
**0.0791570765971351**. At `0.05/15 = 1/300`, rejection occurs for
`K in {0,13,...,20}`; probability `1100957668913/95367431640625`, approximately
**0.0115443778863812**. The latter exceeds the first Holm threshold's nominal
0.0033333333333333. These values are retained from section 4 of the frozen
proposal, not recomputed in this documentation task.

These are **single-test marginal threshold probabilities**, not the unknown
joint 15-test family-wise error rate. The dependence among the 15 null
statistics was not specified by this example. Neither multiplication by 15
nor an independence formula gives the study's family-wise error rate.

The former safeguard of at least 20 nonzero scores always passes here. On the
quite probable `K=0` event, many modest negative scores appear without the
rarer large positive score needed for mean balance; a chi-square reference is
too optimistic. Raising the score cutoff does not prove tail validity and can
only suppress this particular example by excluding its realizations. More
bootstrap draws, software agreement, a continuity correction or Holm cannot
establish validity of the defective reference tail. This is a specific
finite-null failure, not a claim that the published method is universally
invalid. Keep it blocked under the broad proposed null.

## 3. Methodological options and their consequences

### A. Cluster-score t / estimating-equation inference

Use the 99 patient scores, not 183 supposedly independent differences. An
intercept-only mean model estimates `delta_patient_hat`; a simple statistic is
`T=sqrt(P)*mean(U)/sd(U)`, with sample SD denominator `P-1`. This preserves the
equal-patient estimand and mean-zero null. Normal/Student reference inference
requires justified distributional or large-independent-cluster approximations;
99 nominal clusters do not guarantee many informative clusters. Heterogeneous
means require extra care in interpreting the variance estimator.

GEE/sandwich methods allow within-cluster dependence but rely on asymptotic
conditions. Bias-reduced cluster-robust variance with Satterthwaite degrees of
freedom improves some small-sample approximations; it is not a finite guarantee
for all bounded asymmetric, sparse nulls. Relevant primary foundations are
[Liang and Zeger (1986)](https://doi.org/10.1093/biomet/73.1.13) and
[Pustejovsky and Tipton (2018), with corrected manuscript](https://arxiv.org/abs/1601.01981v2).

**Decision:** potentially defensible with a separately specified estimating
equation, weights, assumptions and validated calibration domain; not an approved
replacement. A normality assumption would be an explicit strengthening and
cannot literally hold for nondegenerate finite discrete `U`. A CLT approximation
is different from an exact normal claim. Sparse variance, skewness and a dominant
cluster can defeat small-sample accuracy. Zero estimated variance needs an
explicit rule, not automatic significance or silent removal. A default logistic
GEE coefficient is an odds-scale question, not automatically this mean risk
difference; default nodule weighting would also change the target.

Patient-cluster bootstrap-t or null-imposed wild-cluster bootstrap are related
alternatives for this same mean/null when their score construction and weights
are fixed appropriately. Their justification is generally asymptotic, not
distribution-free exactness for sparse asymmetric patient scores. Resampling
whole patients needs a suitable sampling/exchangeability model; a few influential
clusters can remain problematic. These would require a reviewed null generator,
studentization and independent tail validation, not conversion of the completed
percentile intervals into tests. Published bootstrap refinements motivate an
option, not proof for this specific design.
[Cameron, Gelbach and Miller (2008)](https://doi.org/10.1162/rest.90.3.414).
They are not selected or authorized here.

### B. Whole-patient sign-flip / model-label randomization

One could compare `abs(sum U)` with its distribution under independent sign
changes of entire patient scores, retaining all within-patient observations.
Finite randomization validity requires the joint null law to be invariant under
those transformations, for example independent symmetric patient scores. Model
labels were not randomly assigned treatments, so an experimental randomization
argument is unavailable. Mean zero alone does not imply sign symmetry. The
counterexample in section 2 is explicitly not symmetric. Approximate symmetry
can support asymptotic randomization, but needs its own limiting argument.
[Canay, Romano and Shaikh (2017)](https://doi.org/10.3982/ECTA13081),
[authors' paper](https://home.uchicago.edu/amshaikh/webfiles/weakrand.pdf).

**Estimand/null:** mean difference is still a useful reported effect, but an
exact claim tests a stronger invariance model, not unrestricted mean equality.
**Decision:** acceptable only as an explicitly reviewed assumption amendment,
not the recommended resolution. Studentization or random sign generation does
not by itself remove that distinction. Exhaustive signs may be computationally
large; sampled signs need conservative Monte Carlo p-value handling and ties.
Those computational fixes cannot cure the null mismatch. A patient sign test
instead asks about sign probabilities/median behavior, not the mean magnitude;
it too changes the question. Never flip individual nodules as independent units.

### C. Bounded independent-patient mean inference (recommended candidate)

**Estimand/null:** the equal-patient mean and two-sided mean-zero null of
section 1, including its explicitly reviewed independent heterogeneous-design
interpretation. No symmetry, positive variance, minimum nonzero count or
common distribution is needed for that design interpretation. The required
independence is between patient clusters, not within them.

Hoeffding's bounded independent-sum inequality gives, for independent
`X_p in [a_p,b_p]`,
`Pr(abs(sum(X_p-E X_p)) >= t) <= 2*exp(-2*t^2/sum(b_p-a_p)^2)`.
It applies without equal distributions. Using the fixed logical range
`[-1,1]` for each `U_p` yields the candidate below.
[Hoeffding (1963), Theorem 2](https://doi.org/10.1080/01621459.1963.10500830),
[original paper copy](https://www.csee.umbc.edu/~lomonaco/f08/643/hwk643/Hoeffding.pdf).

```text
S = sum_p U_p
p_bound = min(1, 2 * exp(-S^2 / (2*P)))
equivalently: log(p_bound) = min(0, log(2) - S^2/(2*P))
```

Here is the proposed application, not a fitted calibration: under
`sum E[U_p]=0`, for any `0 < alpha < 1`, the event `p_bound <= alpha` implies
`abs(S) >= sqrt(2*P*log(2/alpha))`. Applying the inequality bounds that event's
probability by `alpha`. This supplies super-uniform marginal p-values for the
specified fixed procedure. It does not claim uniformly distributed p-values,
exact size equal to alpha, optimal power or validated software.

**Finite-sample limitations:** potentially severe conservatism because the
bound ignores variance and discordance sparsity. At `P=99`, the absolute
patient-mean threshold is `sqrt(2*log(2/alpha)/99)`: approximately 0.273 at
alpha 0.05 and 0.359 at alpha 0.05/15. These are algebraic thresholds, not power
calculations or comparisons to research effects. Non-rejection cannot establish
equivalence, no useful effect, or nodule-weighted/macro-F1 equality. The formula
does not address dependent patients or model/method selection based on test data.

**Explicit method amendment:** retain the proposed secondary mean question,
but replace the Durkalski/chi-square rule with a bounded-mean procedure. This is
not conventional McNemar or the previously specified clustered McNemar test.
The original literal McNemar requirement would need an explicit approved
substitution, with its chronology preserved. Merely adding this document does
not supply that approval or close any hypothesis.

### D. A nodule-weighted clustered mean test is a different target

If reviewers want formal inference specifically about existing nodule-weighted
accuracy, use cluster sums, not naive McNemar. Under independent sampled
patients, the population ratio target is `E[D_p]/E[n_p]` (finite positive
`E[n_p]`), distinct from `E[D_p/n_p]`. Its zero null concerns `E[D_p]=0`.
Ratio-score or cluster-robust inference needs appropriate moment and asymptotic
conditions. It is not licensed by the existing descriptive percentile intervals.

A finite-design alternative, conditional on externally fixed cluster sizes and
independent clusters, uses `D_p in [-n_p,n_p]` and null `sum E[D_p]=0`:
`p_nodule_bound=min(1,2*exp(-(sum D_p)^2/(2*sum n_p^2)))`. This is another
application of the inequality in C, not an additional method to run alongside
it. Unequal large clusters weaken the bound. Conditioning on observed sizes is
not automatically valid for the unconditional random-burden population ratio;
the design and conditional null must be justified explicitly.

**Decision:** not recommended for these slots without an estimand amendment.
It preserves the existing accuracy weighting but changes the frozen proposal's
secondary hypothesis question. Keep existing nodule-weighted estimates and
intervals regardless of that decision. Do not pick C or D by observed effects,
run both and report the smaller p-value, or quietly add a second family.

### E. Explicit descriptive-only scope amendment

There is no mathematical necessity to abandon inference merely because the
Durkalski rule fails: C supplies a conservative candidate under stated
independence assumptions. But if reviewers cannot justify a sampling model,
will not approve the method/estimand substitution, or decide the conservative
question does not serve the thesis, descriptive-only is an honest option.

That decision would explicitly leave the original inferential requirement
unfulfilled, not call it satisfied by intervals. Retain all 15 IDs with null
statistical fields and a documented scope-amendment reason; do not erase them,
report p=1 as though tested, or declare full supplement/Stage A completion.
There is no new null or error-rate guarantee under this option. No fresh
patient-weighted results or old-bootstrap rerun is implicitly authorized.

## 4. Exact proposed candidate policy, subject to approval

If C is approved, calculate one score per patient for each frozen contrast,
using integer correctness counts and all 99 patients. Always use the logical
range `[-1,1]`, even if observed scores have a smaller range, are mostly zero,
or cancel within patients. Do not replace `P` with the number of discordant or
nonzero-score patients. Do not estimate a tighter bound from the actual errors.
The candidate has no tunable cutoff, continuity correction or tail direction.

All-zero scores give `S=0` and `p_bound=1`, a defined non-rejection under this
candidate. Identical nonzero scores also have a defined bound; no sample-SD
division occurs. This is an **explicit candidate-specific change** from the
old zero-second-moment undefined rule. Historical Durkalski fields remain
historical/blocked, never populated with the new number. Review must approve
this degeneracy policy as part of the method amendment. Within-patient
cancellation remains in the denominator and receives its diagnostic flag.

For later implementation, sum the rational `D_p/n_p` deterministically with a
high-precision independent reference; compare log tails without underflow or
rounding p-values to zero. A decision too close to a numerical threshold needs
precision refinement or an explicit numerical-failure block, not a relaxed
significance boundary. Missing/duplicate IDs, inconsistent labels or splits,
unpaired predictions, invalid counts, wrong seeds or source/hash mismatch stop
the run. They are not statistical degeneracies to delete from the denominator.

The future source contract would reuse the existing completed complete-model,
B1 and four OFF-source loaders and their identity/hash/provenance guards,
without allowing new sources, regenerating predictions or weakening historical
checks. Exact same-seed pairing must precede correctness counting. No model
forwards are needed for this proposed question. There is no execution command
in this document.

## 5. Preserve the family, endpoint hierarchy and unresolved status

Retain one family `F_correctness`, two-sided, family alpha 0.05, consisting
of exactly these complete-minus-comparator hypotheses. Each is a separate
fixed-fit contrast over the same full cohort and the secondary estimand:

- B1: `C_B1_42`, `C_B1_43`, `C_B1_44`.
- Attention alignment OFF: `C_attention_off_42`, `C_attention_off_43`,
  `C_attention_off_44`.
- Disagreement weighting OFF: `C_disagreement_off_42`,
  `C_disagreement_off_43`, `C_disagreement_off_44`.
- Ordinal auxiliary OFF: `C_ordinal_off_42`, `C_ordinal_off_43`,
  `C_ordinal_off_44`.
- Hierarchical fusion OFF: `C_fusion_off_42`, `C_fusion_off_43`,
  `C_fusion_off_44`.

The common question is evidence of a correctness difference from the complete
model across the prespecified comparators and seeds; splitting into seed- or
comparator-specific families would understate the selection opportunities.
Add no B0, pooled-seed, per-class, macro-F1, pointing, perturbation, boundary or
uncertainty tests. Correctness hypotheses cannot test primary macro-F1.

After method approval and validation only, sort the 15 raw p-values ascending,
breaking exact ties by hypothesis ID. For rank `i`, Holm adjusted p is
`min(1, max_{j<=i} (16-j)*p_(j))`; reject at adjusted p `<=0.05`. Valid
marginal p-values permit strong family-wise control despite dependence among
seeds/comparators; invalid marginals do not. This is a fixed-family guarantee,
not protection against undisclosed post-hoc selection of methods or questions.
[Holm (1979), Theorem 1](https://www.jstor.org/stable/4615733).

Retain the proposed undefined-slot convention: family size never shrinks;
internal adjustment may conservatively substitute 1 for an individually
undefined slot, but public raw/adjusted values stay null with reason and
`reject=false`. Source failure or method-level block prevents completion and
must not be disguised as 15 tested p=1 values. Candidate C has no statistical
undefined value for a valid nonempty cohort; a numerical/integrity failure is
not permission to publish a partially successful family.

At this document's milestone all 15 remain `unresolved_not_computed`, with no
research statistic, raw p, adjusted p or rejection. The 33 completed accuracy
intervals remain descriptive and unadjusted, with the same 10,000 common
patient-cluster draws, seed 20260914 and linear 0.025/0.975 quantiles. They are
not rerun or reinterpreted as intervals for the secondary patient-weighted
effect. No new interval set is proposed here.

## 6. Future synthetic validation plan (not executed)

Validation would test implementation against the bound's proof and expose
failures; simulation would not establish a universal theorem. Approval of this
plan would still require a separate implementation/synthetic-testing task,
followed by code review and separate authorization for any research execution.

### Mathematical and independent-reference gates

1. Have a statistical reviewer check the bounded-score derivation, the exact
   sampling/null interpretation, weight choice and all degeneracy/failure rules.
   Marginal validity must hold across the full proposed null, not just each
   patient having zero mean. No claim of exact finite equality to nominal size.
2. A later independently written high-precision oracle must use exact rational
   score sums and the stated log-tail expression, without calling production
   functions. Record its version/hash, numeric precision and threshold checks.
   There is no verified candidate implementation yet. Keep the pinned Durkalski
   code only as a reference for reproducing its known failure, not certification
   of the replacement. Reference agreement alone cannot establish validity.
3. Test hand-derived scores, model-swap sign reversal with unchanged two-sided
   p, row/patient permutation, within-patient replication invariance of `U`,
   all-zero/cancelled scores, extreme +/-1, one nonzero patient, unequal cluster
   sizes, tied values, invalid/missing identities and rational cancellation.
   All 99 patients remain; never condition evaluation on a nonzero-count gate.

### Null stress register and Holm-relevant tails

Retain, without editing, all 52 synthetic null settings already specified in
section 4 of the frozen proposal: six symmetric dependence settings, 36
asymmetric/oriented settings, four heterogeneous settings and six sparse
zero-inflated settings. Retain the exact counterexample as a mandatory oracle.
Use the existing synthetic generator definitions, scenario order and proposed
PCG64/SeedSequence schedule; do not estimate generators from actual errors.

For the counterexample and finite-support settings designated for exact
enumeration, compute rational probability masses over the entire support and
verify total mass one. Retain every zero/degenerate outcome. Evaluate the
candidate at **every** `alpha_j=0.05/j`, `j=1,...,15`, not only 0.05. Reproduce
the two historical Durkalski rejection probabilities separately to demonstrate
that the test harness can detect the known failure; do not compute that method
on research data. The old 20-score gate may be reported as a historical
diagnostic, never as the new candidate's filter or a conditioning denominator.

For settings scheduled for simulation, retain the frozen 100,000-replicate
budget. Report three rules: candidate C, historical Durkalski without its gate,
and historical Durkalski with its 20-score gate (withheld outcomes count as
non-rejections). For this proposed validation extension, use two-sided
Clopper-Pearson confidence level `1 - 0.01/(52*15*3)` per simulated cell for a
conservative 99% simultaneous Monte Carlo envelope. The change from the frozen
two-rule allocation explicitly accounts for the additional candidate; it does
not edit or retroactively execute the old plan. Report unconditional rejection
rates, exact/Monte Carlo status, denominators, numerical failures and all
threshold results. No adaptive stopping or extra repetitions after looking at
an adverse tail. An interval straddling nominal size is inconclusive, not
evidence of calibration. A pass at selected generators does not prove the
broad null; the candidate's mathematical argument remains essential.

Add these **proposed supplemental exact fixtures** to check that the candidate
does not accidentally assume zero mean in every patient:

- Fixed hypothetical sizes: 20 patients with four nodules, 24 with two and
  55 with one (99/183). In each size category pair patients deterministically.
  One patient of a pair has `U in {0,1}`, the other `U in {0,-1}`, independently
  with nonzero probability `r`, for `r=1/20` and `r=1/2`. Unpaired patients
  have `U=0`. Individual means differ, but their sum is zero. Counts can be
  realized by all-correct versus all-discordant nodules without inventing labels.
- Deterministic opposite-sign pairs with `U=+1` and `U=-1`, plus zero patients,
  verify a heterogeneous aggregate-null/zero-variance case. Independence permits
  degenerate variables. Include the sign-reversed ordering in both fixture types.

Enumerate these added fixtures exactly, not in a new Monte Carlo search. Thus
the proposed three-rule simulation confidence allocation is not silently diluted
by adding simulation cells. If later validation expands the simulated register
or compares more methods, review its complete register and simultaneous
uncertainty budget before execution; publish all attempts, not the winning method.

### Family, source and reporting tests

Use synthetic jointly generated correctness indicators with a shared complete
model to preserve feasibility of the 15 contrasts. Include perfectly dependent
seed/contrast cases, heterogeneous joint dependence and mixtures of true/false
nulls. Check Holm arithmetic independently against a pinned R `stats::p.adjust`
reference with `method="holm", n=15`, including ties and undefined placeholders.
Do not infer joint error from one marginal rate. A full joint simulation campaign
would require its own prereviewed generator/budget; it is not required to prove
Holm once super-uniform marginals are established and is not authorized here.

Future synthetic integration tests must reject source/pairing/hash corruption,
preserve every source and existing bundle, verify deterministic exports/read-back,
and prove zero forwards and no invocation of the old multi-metric bootstrap,
pointing or perturbation runners. Require no-overwrite guards and an explicitly
new inference-only namespace. Verify that blocked or failed validation cannot
produce inferential, full-supplement or Stage A completion.

### Failure consequences and power limitations

An exact null rejection probability above its nominal threshold contradicts the
candidate's claimed guarantee under its stated assumptions: block publication
and investigate derivation, generator and implementation. An implementation/oracle
mismatch, missing stress case, unresolved numerical comparison or synthetic
provenance failure also blocks progression. A simulated lower confidence bound
above nominal is an investigation/block, not a reason to drop the scenario.
No increase in cutoffs, assumption narrowing or method substitution occurs without
a new explicit review. Keep unsuccessful validation evidence.

Conservatism is not a calibration failure and should not be concealed. Any
power study must use effects fixed independently of research results and have
a separately reviewed synthetic register. Do not weaken the test because it
fails to reject actual contrasts, call non-rejection equivalence, or choose a
test from the existing accuracy intervals. This document asserts a mathematical
candidate, not validated implementation, useful achieved power or completed
calibration.

## 7. Approval and provenance required before any next step

Review must explicitly decide:

1. Whether the secondary equal-patient estimand is approved, distinct from
   unchanged primary macro-F1 and nodule-weighted accuracy. Choosing the
   nodule-weighted inferential alternative is a separately recorded amendment.
2. Whether independent patient sampling and the chosen population or
   finite-design null are defensible, without claiming representativeness of
   all patients or clinical settings from these test cases.
3. Whether to approve C as a **substitution for the original McNemar requirement**,
   accepting its power limitations and candidate-specific all-zero rule; or
   explicitly approve descriptive-only scope while recording the unmet
   inferential requirement. No narrowed symmetry/normality null is approved here.
4. Whether the future synthetic/reference plan and unchanged 15-slot Holm family
   are adequate. Implementation verification and any research use remain
   separate gates, not consequences of accepting a document.

Any later approved protocol must bind this decision's content hash, the unchanged
proposal hash, amendment decision/date, implementation commit, reference versions,
full validation register/results and all existing source/checkpoint/prediction
identities. Preserve all 15 hypothesis IDs, estimand/formula identifiers, cohort
counts, per-patient counts/scores, null/assumption declarations, numerical
diagnostics and explicit unresolved/failure reasons in a new protected bundle.
Report historical Durkalski failure separately from any future candidate result.
Do not alter historical registers or completion markers. Multiplicity claims
apply only to the approved fixed family and model, not to macro-F1, prior
descriptive intervals or all decisions made during the research program.

This proposal adds no protocol/configuration/allowlist, changes no completed
result, and authorizes no execution. All 15 hypotheses and full-supplement/
Stage A completion remain unresolved. Stop for methodological review.
