# R3 entropy resolution proposal

Prepared 2026-09-27 on `phase4-consendxformer`, reviewed published HEAD
`e94cd06d4cf35ff7ac4488d282e9ff5e787f00e1`. Branch/HEAD matched and the
working tree was clean before this file was added. No applicable ancestor,
repository-root or docs `AGENTS.md` was found.

**Proposal only: R3 remains unresolved.** R1/R2 remain resolved within their
documented evidence scope; R4/R5 remain unresolved and Stage A is not certified.
This task authorizes documentation, not implementation, research calculations,
server access or execution. The later evidence review below inspected archived
CSV integrity/identities only; no proposed entropy summaries or entropy
recomputation were performed.

## 1. Requirement and approval search

The original brief's uncertainty-quality requirement explicitly names
correct/incorrect and indeterminate/definitive entropy separation. Source:
the original `pasted-text.txt` identified in
[Stage A reconciliation](stage_a_evidence_reconciliation.md), SHA-256
`eb90ef9f60920198f208553ec6bc3d7b130f648c56b984c19c2210a81964ea1f`,
under “Reality-check constraints on result framing.”

The later [Phase 4 evaluation contract](phase4_evaluation.md) and
[uncertainty comparison contract](phase4_evidence_gaps.md#uncertainty-definitions-and-limits)
define reader-disagreement association, training-median low/high disagreement
groups and entropy/confidence risk–coverage. These are completed analyses, but
neither correctness nor true-class grouping is interchangeable with their
reader-disagreement groups. The later [supplemental proposal](phase5_supplemental_evaluation_proposal.md)
preserves completed uncertainty evidence and adds no formal uncertainty tests;
the [Candidate C authorization](phase5_inference_implementation_authorization.md)
does not waive these entropy subquestions. The published reconciliation's R3 row
and the supplied `Stage_A_Existing_Evidence_Audit_2c3c485e05d0.md` leave R3 open.

**No existing explicit approval resolving R3 was found in the inspected records.**
This is not proof that no inaccessible decision or artifact exists. The supplied
server text search reportedly found the two comparisons only in the reconciliation;
that is limited search evidence, not an exhaustive artifact audit. Before any
new summary is authorized, accept an existing exact report or dated explicit
scope decision if one is recovered and reviewed. Do not infer approval from
completion of the other uncertainty analyses or from this proposal request.

## 2. Recommended bounded scope and source

Recommend an additive, **exploratory post-hoc descriptive** summary of the
already stored `predictive_entropy_nats`, separately for the complete model
and disagreement-weighting-OFF at each seed 42, 43 and 44. The complete model
addresses the original subquestions; the existing OFF source supplies context,
not a new superiority endpoint. Inclusion of both models is a proposed decision
for review, not a claim that the original brief specified this exact six-group
summary. Do not expand to B0, B1 or other OFF variants without separate review.

Use the existing primary test cohort: all 183 eligible nodules from 99 patients,
with the frozen `paper_aligned_multi_reader_2_to_4` identities/labels/splits.
No boundary, validation or training rows enter the summaries. Keep the frozen
2D architecture, configurations, loss weights, checkpoint selection, source
validity and preprocessing unchanged. No new thresholds or case exclusions.

The user reported, and the supplied archive now contains, the source at this
repository-relative server path:

```text
artifacts/phase4/evidence_gaps/uncertainty/
  phase4-evaluation-8c5876a39d46aaa7/
    tables/uncertainty_per_nodule_per_seed.csv
```

The reported columns include `patient_id`, `nodule_id`, `series_instance_uid`,
`split`, `true_class`, `predicted_class`, the three `prob_*` columns, `seed`,
`experiment_id`, `model`, `predictive_entropy_nats`, `disagreement_entropy` and
`disagreement_group`. Current export code in
[`phase4_evidence_gaps.py`](../src/consendxformer/phase4_evidence_gaps.py),
`uncertainty_comparison`, uses model values `full` and
`disagreement_weighting_off`. It retains prediction rows plus stored natural-log
entropy. This code inspection corroborates the reported schema, not the actual
live server file bytes. The subsequently supplied archive/file hashes and local
integrity findings are recorded in section 6; live source files were not accessed.

Use that per-seed CSV, **not** `uncertainty_per_nodule_seed_mean.csv`, paired
difference exports, rounded report tables or recomputed entropy of probabilities.
Use the full stored numerical precision; do not clip, normalize, rescale by
log(3), recompute entropy, regenerate predictions or change hard labels. Stored
probabilities and disagreement fields remain evidence, not inputs to new entropy
calculations or substitute grouping variables.

### Required provenance and eligibility validation before later summarization

1. Require the exact completed uncertainty bundle ID above, its original
   `COMPLETED.json`, `artifact_manifest.json`, `evaluation.json`, `README.md`
   and `provenance/provenance.json`. Validate original manifest/completion
   binding, payload hashes, analysis identity and recorded protocol using the
   existing canonical hash conventions. Reject failed/partial, duplicate,
   unexpected or path-escaping members; preserve the archive unchanged.
2. Bind the six original source descriptors and recorded prediction/checkpoint
   hashes from provenance. Preserve the distinct historical
   `load_phase4_final_prediction_source` and `load_ablation_source` contracts:
   full seeds 42/43/44, fixed full training source
   `e1b174490245c616e355ef7c0c6f7c3c239d3acd`, ordinal/attention weights
   0.5/0.05, OFF variant exactly `disagreement_weighting`, and OFF's exact
   same-seed full-model reference. Do not broaden source allowlists or substitute
   checkpoints. Archived provenance is not fresh checkpoint-byte validation.
3. Require one group for every `(model, seed)` and exactly one row per frozen
   `(patient_id, series_instance_uid, nodule_id, split)` within each group.
   All six groups must match the same trusted test identities and true labels,
   not merely the same row count. Expected cohort sizes are contract checks,
   not newly observed results. No inner join, deduplication or missing-row drop.
4. Check exact seed/model/experiment bindings and frozen supervision, input-lock,
   manifest and source snapshot identities. Compare exported labels/predicted
   classes with their hash-bound saved source records or reviewed archived
   validation evidence. Keep original row positions for traceability; any
   deterministic sorting is on a separate view. No argmax-based replacement.
5. Before implementation, review whether the existing archived evidence suffices
   for a saved-record-only adapter. If existing loader reuse requires unavailable
   checkpoints, do not bypass its guards or claim it ran: request the requisite
   evidence or review a separately specified archive-validation contract.
   This proposal does not authorize checkpoint/model access or a loader rewrite.
   Recheck all actually consumed files' hashes before/after a later export.

## 3. Group definitions and summaries proposed for review

For each model/seed independently, let `H_i` be the stored
`predictive_entropy_nats` of nodule i. Two separate partitions retain every
eligible row; they are not crossed into additional subgroups:

| Axis | Group | Exact membership |
|---|---|---|
| Prediction correctness | correct | `predicted_class == true_class` |
| Prediction correctness | incorrect | `predicted_class != true_class` |
| True class | indeterminate | `true_class == "indeterminate"` |
| True class | definitive | `true_class` in `{"benign", "malignant"}` |

Use the canonical textual class order benign/indeterminate/malignant and reject
unknown or missing class values. “True” denotes the frozen study label, not
pathology-proven cancer truth. Definitive is a reporting union, not a relabeling
of the underlying three-class task. Predicted-indeterminate membership is not
the requested true-indeterminate group. Correctness membership can change across
models/seeds; true-class membership must remain identical.

Proposed output: four group rows per model/seed (24 rows for six combinations),
including empty groups. Each reports model, seed, experiment ID, axis, group,
eligible total, group nodule count, group distinct-patient count, finite-entropy
count, missing/invalid count, mean, sample SD, median, Q1, Q3, minimum and maximum
entropy in nats. Use nodule weighting within each group:

```text
mean = sum(H_i) / n
sample_SD = sqrt(sum((H_i - mean)^2) / (n - 1)), for n >= 2
Q1, median, Q3 = linear quantiles at 0.25, 0.50, 0.75
```

Keep each partition's group nodule counts summing to 183. Patient group counts
need not sum to 99 because one patient can have nodules in both groups; report
that overlap explicitly and never infer independent patient diagnoses. A
single nodule appears once per partition in each model/seed, not twice in a
combined independent sample.

For an explicit descriptive contrast, propose two mean differences per
model/seed: `mean_incorrect - mean_correct` and
`mean_true_indeterminate - mean_true_definitive`. Export both group denominators
with each difference (12 contrast rows planned). Positive means higher stored
entropy in the first group; it is not a test or a prespecified success threshold.
There is no paired-nodule interpretation of differences between disjoint groups.
No full-minus-OFF contrast, ensemble, seed-pooled entropy, pooled correctness
rule, across-seed mean/SD, confidence interval, bootstrap, p-value or effect-size
test is proposed. Show the six model/seed summaries side by side, not as 549 or
1,098 independent observations. Counts above describe the planned output schema,
not calculated research results.

### Missing, invalid and empty-group policy

- Missing identity, label, model/seed/experiment binding, duplicate/missing cohort
  member or source mismatch is an integrity failure. Stop before research summary
  export; do not recode missing comparisons as incorrect or definitive.
- Require every stored entropy to parse as a finite number. Missing/NaN/infinite
  or malformed values fail closed, with offending source row/identity and reason
  in diagnostic evidence. Do not silently use `nanmean`, complete-case filtering,
  imputation or a probability-derived replacement. Preserve the saved numeric
  values and historical numerical policy; an apparent entropy-range anomaly
  requires source review, not a newly chosen clipping tolerance.
- A valid empty group has count zero, patient count zero, `status=empty_group`,
  and all entropy summaries null (not zero). Its contrast is null with explicit
  reason `empty_comparison_group`. It is retained in the fixed output schema.
- For one member, mean/median/quartiles/min/max equal that stored value;
  sample SD is null with `insufficient_n_for_sample_sd`. Constant nonempty
  groups are valid; sample SD is zero when n >= 2. Do not discard them.
- Successful complete input has no missing/invalid entropy values. Legitimate
  empty-group summaries are allowed and are distinct from invalid input or a
  failed source contract. No outcome-dependent selection of a different summary.

## 4. Evidence needed from the server, without a new analysis

The initial evidence request was an unchanged copy of the **existing** bundle
`artifacts/phase4/evidence_gaps/uncertainty/phase4-evaluation-8c5876a39d46aaa7/`
with all original manifest-listed members plus manifest and completion marker.
That archive and the S7 review have now been supplied and checked as recorded in
section 6. Do not request their transfer again. The remaining original-file
evidence distinction and proposed archive-only contract are specified there.
This includes the per-seed CSV above and the identity/provenance/evaluation files
listed in section 2. A whole-bundle transfer allows integrity checks without
manufacturing a reduced replacement bundle. Provide its externally recorded
archive SHA-256, original manifest SHA-256 and the per-seed CSV SHA-256; a transfer
manifest should map archive members to original paths/lengths/hashes. No new
entropy summaries, GPU probe or research execution is requested now.

Also supply the existing `ConsenDxFormer_Disagreement_Uncertainty_Review.md`
identified as S7 in the historical checklist, any exact two-axis summary already
available, and any dated explicit decision claimed to resolve these original
subquestions. Record the scope of the reported server search; a negative text
search alone cannot certify absence.

If the uncertainty provenance does not include enough archived identity/label
validation evidence, supply unchanged referenced records for all six original
sources: `predictions/test.csv`, full-model/OFF identity descriptors and
completion/manifest records, plus hash-bound primary supervision/test-identity
evidence. Resolve their exact directories from provenance rather than guessed
paths or newest-run discovery. Which additional members are necessary must be
settled at source-contract review; do not use incomplete evidence to relax
historical loader requirements. Do not request raw images, rerun models, refit
the prior or regenerate an uncertainty bundle to fill an evidence-transfer gap.

## 5. Later export, review and R3 closure criteria

If this protocol is approved and no equivalent completed summary is recovered,
separate implementation/synthetic validation and execution authorizations are
still required. Propose a distinct no-overwrite namespace such as
`artifacts/phase5/r3_entropy/`, outside every existing result bundle. Export a
versioned protocol/proposal hash, source archive/manifest/CSV identities,
identity-preserving group membership ledger with stored entropy and source-row
references, the 24 summary rows and 12 descriptive contrasts, validation counts,
limitations, manifest and scope-only completion marker. Record source validation
coverage and omissions honestly; do not rewrite the old uncertainty outputs.
Proposed scope is `r3_entropy_descriptive_only`, with no model forwards, training
or formal tests and no Stage A/full-supplement certification.

Before any later research execution, synthetic tests should cover all six
groups, exact partitions, changed correctness across seeds, identical true-class
groups, unequal patient cluster sizes/overlap, empty and singleton groups,
constant entropies, null/invalid data, duplicate/missing identities, source/hash
corruption, deterministic quantiles/exports and no-overwrite. Confirm that the
summary uses stored entropy only and cannot invoke models or shared entropy
recomputation. These tests are planned, not implemented or run here.

R3 can close only after either (a) a reviewed existing artifact demonstrates
both exact requested summaries with valid source/identity provenance, or (b)
this bounded summary protocol is approved, separately implemented/validated,
authorized for execution, and its completed output independently reviewed.
An explicit accepted scope amendment replacing the subquestions would be a
different closure route and must be recorded as such, not as completed summaries.
Success does not require a favorable difference: valid zero, reversed or
empty-group findings are reported without tuning. The closure decision must
identify the evidence and limitations; accepting this proposal is not closure.

Entropy summarizes predictive uncertainty, not epistemic uncertainty alone,
calibration, discrimination, causal effects or clinical usefulness. Conditioning
on correctness is outcome-defined; models can have different error sets, so
between-model group summaries do not isolate disagreement weighting's effect.
True definitive pools benign and malignant cases with potentially different
entropy distributions. Retain that aggregation limitation without selecting new
subgroups after seeing results. Nodule-weighted summaries favor patients with
more nodules; no patient/seed independence or population inference is claimed.
All choices are exploratory post-hoc, made after earlier test results were seen.
Macro-F1 stays primary; the completed 15-test Candidate C/Holm family and all
earlier results remain unchanged. R4/R5 require their own decisions even if R3
is later resolved. Stop at protocol review; no current research execution or
manuscript-writing authorization follows.

## 6. Supplied uncertainty evidence review — 2026-09-27

This addendum supersedes the initial unverified-transfer status, not the frozen
summary definitions above. Repository HEAD remains
`e94cd06d4cf35ff7ac4488d282e9ff5e787f00e1`; the only existing working-tree change
was this unstaged proposal, which has been preserved and revised. S7 was read
completely. It reports reader-disagreement correlations, low/high entropy and
risk–coverage results, with exploratory limitations; it supplies neither the two
R3 partitions nor an explicit waiver. Its additional descriptive check concerns
full-minus-OFF entropy by disagreement group, not correct/incorrect or true-class
entropy separation. No existing approval closing R3 was recovered.

### Independently checked archive evidence

| Object | SHA-256 |
|---|---|
| Supplied `phase4-evaluation-8c5876a39d46aaa7 (1).zip` | `6530307c59cf9f8a58d41982b64eed8aaeff8b68c39e1ff24a55dde1543282bd` |
| `artifact_manifest.json` | `d2eaec89a15dda91ddbd7efddb977c90fc5dacf5e6b36b0d844b2d48fbe0c02a` |
| Canonical payload-hash map | `a7bc955d56fcd8e5a55f2fdebcda24fcf3c0f3118d69418fd792cd88c35997b8` |
| `provenance/provenance.json` | `79a90082dc31db738a56dcfbb1775a2f603bebf2e2590952567064cc48e1ac26` |
| `tables/uncertainty_per_nodule_per_seed.csv` | `7c2b8790a27342e8b03a40f2297b15991b1d539553f408798ccbb5be5c2235b8` |

Local read-only checks passed ZIP CRC, ten unique file members under the expected
root, safe paths/no symlinks, exact eight-payload manifest membership, every
payload SHA-256, canonical manifest digest and completion-to-manifest linkage.
Canonical hashing of the archived provenance reproduced evaluation ID
`phase4-evaluation-8c5876a39d46aaa7`. Recorded evaluator commit is
`bb63d1a7301ad5ebaeb15e4ae1a9bc252c79e3a0`, with clean branch recorded; that
historical identity is not replaced by today's documentation HEAD.

The six source records contain 24 snapshot entries per full source and 20 per
OFF source. Their 132-entry union exactly equals the archived global input
snapshot. Each source's recorded best-checkpoint and test-prediction hash agrees
with its snapshot entry. Full descriptors bind the recorded artifact-manifest
hash; OFF records reference the exact same-seed full descriptor, source directory
and best hash. Full training commit, selected lambdas, OFF variant/toggles,
input-lock and cohort/supervision/fingerprint fields agree across those references.
These are internal archived-reference checks, not original-byte revalidation.

| Model | Seed | Archived experiment ID |
|---|---:|---|
| full | 42 | `phase4-final-s42-c1e650d77585ed9c` |
| full | 43 | `phase4-final-s43-946220804796df93` |
| full | 44 | `phase4-final-s44-8c74c3dd1e666bb5` |
| disagreement_weighting_off | 42 | `phase4-ablation-disagreement_weighting-s42-e1a00dcc25479c12` |
| disagreement_weighting_off | 43 | `phase4-ablation-disagreement_weighting-s43-d21e9a6a3beaa01e` |
| disagreement_weighting_off | 44 | `phase4-ablation-disagreement_weighting-s44-d89020abb1db5cfe` |

The raw table has 1,098 rows: exactly six model/seed groups, each with 183 unique
test identities and 99 patients. All groups have the same identity set and true
labels; seed/experiment bindings match their descriptors. True/predicted classes
are valid canonical classes, and all stored entropy values parse as finite.
Frozen disagreement values/groups/thresholds agree across the six records for
each nodule. The existing paired table has 549 unique seed/nodule rows; its
labels, predictions, experiment IDs and stored entropies match the corresponding
raw rows exactly. No new correctness groups, group counts, entropy means,
quantiles, contrasts, correlations or AURCs were computed. S7's numerical
interpretations were read, not independently recalculated in this review.

### What the provenance does and does not satisfy

This bundle meets the archive-integrity, internal cohort consistency and
archived binding portions of section 2. It contains enough saved fields to
implement the proposed summaries without models. It does **not**, alone, meet
a requirement to independently match the table against original prediction
bytes or authoritative supervision bytes: those files are only hash-referenced.
Cross-seed agreement is not an independent cohort oracle. Nor do the recorded
`source_artifacts_unchanged`/clean flags constitute live-server observation.
Checkpoint tensors, original source manifests/completion records and raw data
were not transferred here. No historical source loader or model was invoked.

Recommend the following **explicit archive-scoped validation contract for
review**, rather than silently claiming the live-loader requirements passed:

1. Accept only this externally pinned ZIP/manifest/provenance/raw-CSV identity,
   not a newest-directory search or self-generated hash receipt. Preserve the
   exact ten-member inventory and verify CRC, paths, member hashes, completion
   binding, canonical payload digest and content-addressed evaluation identity
   before opening the table for later summary work.
2. Validate the six exact model/seed/experiment bindings above and all internal
   source-reference equalities just checked, including the complete 132-entry
   snapshot union, selected best/test hashes, cohort/input-lock/supervision
   fields and OFF-to-full references. Treat source paths as provenance strings,
   not permission to dereference server paths. Never modify original loaders,
   trust allowlists or historical bundles to make this archive pass.
3. Pin the reviewed archive's canonical identity/true-label ledger as the
   **archived cohort**, preserving row provenance; require exact six-way identity
   and label agreement, 183 nodules/99 patients per group, canonical classes,
   fixed seed set and experiment mapping. Validate paired/raw agreement without
   computing new entropy statistics. Do not call that ledger independently
   verified original supervision. Before production implementation, record its
   deterministic schema/hash convention in the new versioned adapter protocol.
4. Apply section 3's finite/missing-value and empty-group rules to the stored
   entropy only. Do not invoke `predictive_entropy`, re-normalize probabilities,
   replace classes by argmax or rerun any earlier analysis. Export exactly the
   source bytes/hash references plus new membership/summary records separately;
   source snapshots must remain unchanged and output paths must be fresh.
5. Explicitly export validation scope `archived_saved_records`, with
   `original_prediction_bytes_reverified=false`,
   `original_supervision_bytes_reverified=false`,
   `checkpoint_bytes_reverified=false`, `live_sources_reverified=false`, and
   zero model forwards. Retain S7 as attributed historical audit evidence, not
   a substitute for those unperformed checks. Completion covers the new bounded
   R3 subset only and cannot close R4/R5 or certify Stage A.

Accepting this contract is an explicit source-validation scope decision, not
already granted approval. It preserves the archived execution chain while
avoiding a misleading claim of full historical loader replay. No essential
field is missing for a calculation **conditional on these reviewed saved
records**. The proposal's stronger original-byte verification remains unmet.

### Exact additional files if original-byte validation is required

For each of the six experiment IDs above, use the exact `source_dir` in the
archived provenance and supply its unchanged `predictions/test.csv`,
`artifact_manifest.json` and `COMPLETED.json`. The prediction bytes establish
the recorded SHA-256 and allow identity/label/prediction equality checks against
this derived CSV; source manifests/markers establish their original completion
bindings. Do not recreate a CSV from the derived table: formatting differences
cannot verify the original file hash.

Supply at least the full seed-42 source's unchanged
`config/input_supervision.csv` and its manifest binding to establish authoritative
test identities and labels independently; if recorded supervision hashes are not
identical across the six sources, provide each distinct hash-bound file and stop
on inconsistency. For OFF identity reconstruction rather than archived-reference
checking, also provide each OFF `config/run_identity.json` and its manifest
binding. Each requested path is relative to that exact recorded source directory,
not an invented replacement location. Verify expected entries before transfer.
Checkpoint `best.pt`/`last.pt` and other loader-required inputs would be necessary
only for a separate claim of complete historical source-loader replay; they are
not needed for the proposed archive-only summary and are not requested to run a
model. Original entropy derivation remains recorded provenance in either path;
recomputing entropy is excluded from this task and proposed adapter.

**Implementation recommendation:** after explicit approval of the archive-scoped
contract and summary policy, authorize only a separate pure saved-record adapter
and synthetic CPU tests for the checks above, corruption/mismatch failures,
group semantics, determinism and no-overwrite/no-inference behavior. Use invented
fixtures for tests; do not execute the research summaries during implementation.
If the reviewer requires original-byte validation instead, obtain the exact
additional records before finalizing that production trust contract. Either
route requires later separate execution/review authorization. R3 remains open;
this integrity review neither produces its requested entropy comparisons nor
certifies Stage A.
