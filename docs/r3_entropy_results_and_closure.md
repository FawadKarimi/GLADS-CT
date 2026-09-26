# R3 stored-entropy results and bounded closure decision — for review

## Accepted bounded R3 decision — 2026-09-27

The owner accepts the final documentation review and closes R3
only for the completed correct/incorrect and true
indeterminate/definitive stored-entropy descriptive comparisons
in r3-entropy-83023569d06acd93, conditional on the accepted
archived_saved_records validation contract.

This decision supersedes the pending-acceptance wording below.
It does not establish model superiority or independently verify
original prediction, supervision or checkpoint bytes. R4/R5
remain unresolved; Stage A remains uncertified. Original result
bundles and execution-time completion markers remain unchanged.

Prepared 2026-09-27 against `7908b8d69a8064240ac7733d683c28489b0077a2`
on `phase4-consendxformer`; branch/HEAD matched and the starting working tree
was clean. This is a documentation-only integrity/read-back review of the supplied
completed archive. No entropy summaries, contrasts or other research statistics
were recalculated; no model, checkpoint, server job or test suite was run.

**Recommendation, not adopted closure:** accept R3's two descriptive comparisons
as evidenced under the approved `archived_saved_records` contract, subject to
final documentation review. The separate execution authorization is now recorded
below from the owner-provided conversation record. This is not independent original-file verification. R4/R5 remain
unresolved; Stage A is not certified. Macro-F1 remains primary.

## 1. Authorization and chronology

The user explicitly accepted the archive-scoped contract and summary policy in
the [R3 proposal](r3_entropy_resolution_proposal.md), SHA-256
`e767983361beada298f76fbc6be12c940f21da1d02be35ae3d34af0dbb125c29`.
The separate [implementation authorization](r3_entropy_implementation_authorization.md)
records implementation and synthetic CPU testing only. Publication of that
reviewed candidate at the HEAD above did not authorize research execution or
close R3. Neither historical document is rewritten here.

The user supplies `r3-entropy-83023569d06acd93.zip`, its expected ZIP hash,
and a manifest hash described as matching the server receipt. The archive
demonstrates a completed saved-record export whose two adapter hashes match the
published implementation. The stated manifest digest was independently matched
to the transferred bytes during the preceding integrity review.

**Execution authorization — owner-provided ChatGPT conversation record:**
at **2026-09-27 04:10:29, UTC+07:00**, the owner stated exactly:

> “I authorize the R3 archive-scoped descriptive run:
> 24 summaries and 12 contrasts.”

The supplied conversation record reports subsequent server completion at
**2026-09-27 04:11:39, UTC+07:00** for `r3-entropy-83023569d06acd93`.
These authorization and completion times are attributed to that record, not
independently verified server telemetry. **No separate receipt file was created**;
no independently verified server authorization receipt is claimed.

This resolves the draft's missing-authorization-record caveat without broadening
the earlier implementation-only authorization. Bounded closure remains pending
final documentation review. The archive itself does not contain the dated approval,
invocation transcript, execution timestamp or execution-checkout Git receipt.
Its code binding establishes reviewed adapter bytes, not independent observation
of a clean server checkout at a particular commit. The CLI flag/completion marker
is not substituted for the owner-provided authorization record. No checks or
analyses were rerun to record this clarification.

## 2. Exact evidence identities and verification

Bundle: `r3-entropy-83023569d06acd93`, scope `r3_entropy_descriptive_only`.
Expected output namespace: `artifacts/phase5/r3_entropy/`; the ZIP does not itself
prove its former physical server location. All hashes below are SHA-256.

| Evidence | Hash |
|---|---|
| Supplied result ZIP | `840407afc53e0bc12d2525a6b124aeea789f862cc38526995506c99a4be00d40` |
| Result `artifact_manifest.json` (user-supplied server-receipt digest matched) | `30c8a36003663c5a4521e8dfcbe08248ca473b97ef700d03f6cd0ef74b34beda` |
| Canonical result payload-hash map | `733816425d651dfcf99513e1af9e058ce66629c8d7c35be98cb90ec9dd3f0390` |
| Original result `COMPLETED.json` | `50cf23d700355696a54d6bd579f2f0ba35627452bf29c59a8d59571e4afe894b` |
| `protocol.json` file bytes | `41f71f1bd2045c46699f7bbb4132e7db6add7bf7ac8e0008a222086837db70f6` |
| Canonical protocol JSON, used by run identity | `3b48cb215724fe225f3cba1194acda8206495f2c65d7c27821bfef5b2839b25d` |
| `provenance.json` | `49dc479e2a0b6556f5fb1a891046966192342b8fe6f5d1b628a060272c510dda` |
| `archived_cohort.json` file bytes | `404e74d318b8be5ea34cc82cc064a4991313259041e2f72443643d0f2276d52c` |
| Canonical archived-cohort ledger | `8b836f06e8289000073ce0df9a82d6542c8b11fed4faa9e4d02fc6a5b1162f04` |
| `tables/group_summaries.csv` | `73fa9f0e5e087229fe62c05e979f857313bd77fd9fdda64c5c76767532a10169` |
| `tables/descriptive_contrasts.csv` | `69dd988e6f7ce77605bd612e85cdaeff8da948779588e40379d12e9e30a41bc4` |
| `tables/membership.csv` | `f0371404d36b1e5d5e1c8d6b2bdefd72205035fcdfd00c1760ebcc02c496b8d4` |
| `patient_overlap.json` | `06f34ccb2099b4224f7bf53be63b4132e723966c9eaae20c0ebbad2fe3f4e9eb` |
| `validation.json` | `d812a670bf2d9f7988fa774d13af598c49d0d9491ed456cf5a8cc9099c65fb0e` |
| `limitations.json` | `4b8ccce45d599d3379b6c52a4764f845d59d0c5256df2765fb6851aab35e9e4f` |
| Embedded unchanged `proposal.md` | `e767983361beada298f76fbc6be12c940f21da1d02be35ae3d34af0dbb125c29` |
| Embedded `source/archive.zip` | `6530307c59cf9f8a58d41982b64eed8aaeff8b68c39e1ff24a55dde1543282bd` |
| Bound summary/export adapter | `6256eae0b219b4b7404728f0e956d8c776c40135e026e1f2e125d600da21b9cd` |
| Bound source adapter | `803a37bcc785578151870b62c136d2b77210b018d122517c890a1652ca4780ff` |

The embedded source is the reviewed uncertainty bundle
`phase4-evaluation-8c5876a39d46aaa7`. Its exact manifest, provenance and raw per-seed
CSV hashes are respectively
`d2eaec89a15dda91ddbd7efddb977c90fc5dacf5e6b36b0d844b2d48fbe0c02a`,
`79a90082dc31db738a56dcfbb1775a2f603bebf2e2590952567064cc48e1ac26`, and
`7c2b8790a27342e8b03a40f2297b15991b1d539553f408798ccbb5be5c2235b8`.
Its canonical payload digest remains
`a7bc955d56fcd8e5a55f2fdebcda24fcf3c0f3118d69418fd792cd88c35997b8`.
Its historical evaluator commit is `bb63d1a7301ad5ebaeb15e4ae1a9bc252c79e3a0`;
full-model training source is `e1b174490245c616e355ef7c0c6f7c3c239d3acd`.
Those earlier identities are not replaced with the R3 implementation HEAD.

Read-only local checks passed:

- ZIP hash, CRC, exact 13 unique safe regular members; exact 11 manifest payloads,
  every payload hash, canonical bundle digest and completion-to-manifest linkage.
- Canonical provenance identity reproduces `r3-entropy-83023569d06acd93`;
  exported protocol equals the published version-1 protocol, proposal bytes are
  unchanged, and both recorded adapter hashes match current published bytes.
- Embedded source ZIP/ten-member inventory, CRC, original manifest/completion,
  all payload hashes and content-addressed uncertainty identity; the embedded
  archived provenance equals the result's source-provenance copy.
- Six exact source bindings, fixed seeds/variant/lambdas, OFF-to-same-seed-full
  references, recorded best/prediction hashes and the complete 132-entry source
  snapshot union. Paths are checked as provenance strings, never dereferenced.
- 1,098 saved source rows, six identical 183-nodule/99-patient test cohorts with
  canonical true labels, frozen disagreement fields and paired/raw agreement;
  the exact archived-cohort ledger matches its pinned canonical hash.
- All 2,196 membership rows preserve exact source-row indices, identities,
  labels and entropy text, with exactly one occurrence per source row per axis.
  Group definitions, 24 summary keys, 12 contrast keys, source experiment IDs,
  group/contrast denominators and all 12 patient-overlap records agree.

These are integrity and stored-record consistency checks, not a rerun of summary
arithmetic. Means, SDs, quantiles, extrema and differences below were read from
existing exports; they were not independently recomputed in this review.

## 3. All six model/seed results

All summaries are nodule-weighted stored predictive entropy in nats, separately
for each model and seed. Correct means predicted class equals the frozen true
class; incorrect means it differs. True indeterminate is the frozen indeterminate
label; true definitive combines benign and malignant without relabeling them.
The two partitions are separate, not crossed. “True” is a study label, not a
pathology-proven patient diagnosis. Every model/seed retains 183 nodules/99 patients.

| Model | Seed | Source experiment |
|---|---:|---|
| Full ConsenDxFormer | 42 | `phase4-final-s42-c1e650d77585ed9c` |
| Full ConsenDxFormer | 43 | `phase4-final-s43-946220804796df93` |
| Full ConsenDxFormer | 44 | `phase4-final-s44-8c74c3dd1e666bb5` |
| Disagreement-weighting OFF | 42 | `phase4-ablation-disagreement_weighting-s42-e1a00dcc25479c12` |
| Disagreement-weighting OFF | 43 | `phase4-ablation-disagreement_weighting-s43-d21e9a6a3beaa01e` |
| Disagreement-weighting OFF | 44 | `phase4-ablation-disagreement_weighting-s44-d89020abb1db5cfe` |

### Correct versus incorrect

Values are transcribed at exported precision. N/P means nodules/distinct patients;
the last column is the exported patient intersection between the two groups.

| Model | Seed | Correct N/P | Correct mean | Incorrect N/P | Incorrect mean | Incorrect minus correct | Patient overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 42 | 118/77 | 0.6639254979509941 | 65/45 | 0.9123646365099394 | 0.24843913855894528 | 23 |
| Full | 43 | 114/75 | 0.7529408860913405 | 69/47 | 0.9393573940524612 | 0.18641650796112075 | 23 |
| Full | 44 | 131/79 | 0.6812267362272273 | 52/42 | 0.9129827717081399 | 0.23175603548091261 | 22 |
| OFF | 42 | 132/80 | 0.6363736036278356 | 51/40 | 0.882532606317808 | 0.24615900268997248 | 21 |
| OFF | 43 | 126/82 | 0.5582366717193886 | 57/42 | 0.8138318347736154 | 0.2555951630542268 | 25 |
| OFF | 44 | 127/75 | 0.5700796178278094 | 56/45 | 0.8436351146488462 | 0.27355549682103686 | 21 |

### True indeterminate versus true definitive

In every row, indeterminate N/P is **41/36**, definitive N/P is **142/84**,
patient union is 99 and patient intersection is **21**. Those overlap IDs are
preserved in `patient_overlap.json`, not treated as separate independent patients.

| Model | Seed | Indeterminate mean | Definitive mean | Indeterminate minus definitive |
|---|---:|---:|---:|---:|
| Full | 42 | 0.9264554610401562 | 0.701846734005049 | 0.22460872703510715 |
| Full | 43 | 0.9612293837056052 | 0.7833839188176255 | 0.1778454648879797 |
| Full | 44 | 0.9013360771318304 | 0.7025424465646831 | 0.19879363056714727 |
| OFF | 42 | 0.8754282593842772 | 0.6557599997628673 | 0.2196682596214099 |
| OFF | 43 | 0.8310130618065628 | 0.582075349892042 | 0.24893771191452074 |
| OFF | 44 | 0.8368929091812942 | 0.6009230183664375 | 0.23596989081485675 |

All 24 saved groups have `status=ok`, zero missing/invalid entropy count and
finite count equal to group nodule count. There are no empty/singleton groups
or undefined contrasts in this execution. The unchanged summary CSV additionally
retains every sample SD, median, linear Q1/Q3, minimum and maximum at original
precision; its exact hash above binds those dispersion results. Empty-group and
singleton rules remain in the frozen protocol, not altered because unused here.

## 4. Interpretation and limits

In each of the six model/seed exports, mean stored entropy is higher for incorrect
than correct predictions, and higher for true indeterminate than true definitive
nodules. All twelve saved mean differences are positive. This documents the two
requested **descriptive mean comparisons**, not perfect distributional separation,
a decision threshold, significance, calibration, epistemic uncertainty alone,
clinical utility, causal benefit or generalization to a new population.

No across-seed pooled sample, average contrast, full-minus-OFF statistic, confidence
interval, hypothesis test or superiority claim is introduced. A repeated nodule
across seeds is not an independent observation. Patient groups overlap within an
axis, and nodules within a patient are dependent. Correctness is outcome-defined
and different models/seeds have different error sets. Nodule weighting favors
patients with more nodules; definitive pools two potentially different classes.
The OFF rows provide descriptive context, not a causal estimate of disagreement
weighting. Macro-F1 and all completed classification/uncertainty evidence remain
unchanged. The analysis details were fixed after prior test results were seen:
this is exploratory post-hoc evidence, not preregistered confirmatory testing.

The result marker explicitly records zero model forwards, no training, no entropy
recomputation and no formal tests. Those are archived execution declarations,
consistent with the reviewed saved-record-only adapter, not independent live
process monitoring. The embedded source archive bytes were freshly checked here;
the original prediction, supervision and checkpoint files were not. Their hashes
are archived references only. `original_prediction_bytes_reverified`,
`original_supervision_bytes_reverified`, `checkpoint_bytes_reverified` and
`live_sources_reverified` correctly remain false. No original GPU/environment or
server filesystem preservation is freshly asserted.

Implementation validation remains historical: final targeted tests reported
47 passed; the earlier full-suite attempt reported 1,061 passed and four failures
(three Windows symlink-privilege failures and one existing PyTorch autocast API
incompatibility). That attempt preceded final R3 hardening; no green final full
suite is claimed and no tests were rerun for this documentation review.

## 5. Proposed later decision, separate from execution markers

On acceptance of the final documentation review, with the execution release
attributed to the owner-provided conversation record in section 1,
record: **R3 closed only for the completed correct/incorrect and true
indeterminate/definitive stored-entropy descriptive comparisons, conditional on
the accepted `archived_saved_records` source-validation contract.** This supplies
the previously missing exact comparisons rather than redefining them as the
earlier reader-disagreement/risk–coverage analyses. No favorable-result criterion
is being imposed; the closure basis is the specified evidence and its limits.

This proposed decision is not yet owner sign-off. Preserve the original
`COMPLETED.json`, provenance and limitations unchanged: `r3_review_closed=false`
records execution-time review status. A later accepted decision belongs in a
separate dated review record, not in a rewritten completion marker. Likewise,
`r4_closed=false`, `r5_closed=false`, `stage_a_certified=false` and
`full_supplement_complete=false` remain valid scope limits.

R4 still requires acceptance of distributed packaging and bounded claims. R5
still requires separate final sign-off after the remaining decisions. This record
does not certify Stage A, start manuscript work, authorize new calculations or
request a rerun of existing evidence.
