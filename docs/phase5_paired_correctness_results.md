# Phase 5 paired correctness: completed and audited results

## Status and attribution — 2026-09-22

The supplied independent offline audit,
`Phase5_Research_Results_Audit_79f6f8de6cab19b6.md` (2026-09-22), reports **PASS
within the paired-correctness subset**. Results are
`phase5-inference-results-79f6f8de6cab19b6`, executed at
`4141ae78b8e0a28cbc5779cc4190f83bf7cdae8d` on `phase4-consendxformer`.
All 15 registered comparisons are computed, not unresolved placeholders.
All raw and Holm-adjusted probability enclosures are exactly `[1,1]` and
all rejection flags are false (0/15 rejections).

This record transcribes the supplied audit and archived results; it does not
perform new research statistics. Local checks verified the supplied ZIP hash,
manifest hash/completion binding and all nine manifest payload hashes, and
read the exported hypothesis table. The independent arithmetic audit described
below is attributed to the supplied report, not claimed as a new local audit.
No live server, research checkpoint or GPU was accessed; no test-suite rerun,
model forward, training, R invocation or calibration campaign occurred here.

Macro-F1 remains primary. This secondary endpoint is the equal-patient mean
paired correctness difference, **not** patient diagnosis accuracy, primary
macro-F1 or nodule-weighted accuracy. All analyses remain exploratory post-hoc,
not preregistered or confirmatory. No classification-superiority, equivalence,
clinical-utility or SOTA claim follows.

## Provenance and bounded completion

The 18 fixed sources are complete ConsenDxFormer, B1 and the four single-component
OFF variants (attention alignment, disagreement weighting, ordinal auxiliary,
hierarchical fusion), each at seeds 42/43/44. Each comparison retains the same
183 nodules from 99 patients. The archive contains 3,294 saved-prediction records,
1,485 patient/contrast count records and 15 hypothesis records. Exact source,
checkpoint and prediction identities remain in `evaluation.json` and
`provenance.json`; their hashes below bind those descriptors without replacing
the original sources. The server runtime recorded is Python 3.10.17,
NumPy 2.2.6, pandas 2.3.3 and mpmath 1.3.0.

| Object | SHA-256 |
|---|---|
| Supplied results ZIP (local filename has ` (1)` suffix) | `4e72b918ef4caf22f7bdd6a6454ba09b2b2b74fdbd0a504521fcaed1dcc4f20c` |
| `artifact_manifest.json` | `4c82d7141e0a09f58bdd58793f3af9c2ddb39607f3e3090dc01487884a656200` |
| Canonical payload-hash map | `3e44a53234c32b0945e3d88b87a2e42b25aba3b0e8da92cf7c797bcac2192c0e` |
| `evaluation.json` | `4cf0b8f2e138e341c437bcaf16d7a5ed7b54fda35f059cf5f10ab8a5d763c62d` |
| `evidence.json` | `85c427bf4590392bdd6aaef8c2afc145b8fd9d308c432c6b41f2169786eccd57` |
| `inference.json` | `3e99cc18b3308b92847411626b2e8918415f9b166ed0fd02c3c8896378c6f213` |
| `protocol.json` | `f474fc4cc146f86d5d4fe2d04fe6b1c7f0202d339bb0d471e8eb621f31894c9b` |
| `provenance.json` | `da7ba50b9f24d15b4dc20f79235bae12f7b56cb5e82f45ccc0a101bc36a20405` |
| `saved_predictions.json` | `08120e3cb3c3c023e78af0330552c4ec31572ea0332f89856f068c9de635d352` |
| `summary.md` | `7645835d4f15f58d729a3961a43e476eb597117a23c6502f3e2aff4e49474f97` |
| `tables/hypotheses.csv` | `149abe05d7a9c8726d00365e6aeed6a001ce99dd4d041906de2bfab116422bdc` |
| `tables/patient_counts.csv` | `2adb164382450e38ef73e84148a9fb76d36c6034b4cdcf361e806514b596b3d6` |
| Shared preflight/results source snapshot, canonical JSON | `577aab318fa865494f21fba37af7ac0ad9bac6cc2c0354bf48e2d4d4ea271ec8` |

ZIP, manifest-file and canonical payload-map hashes identify different objects.
Reviewed preflight `phase5-inference-preflight-5b67b8e275de62f5` has archive hash
`5c45baf46e319b02feba9c31a66f75c4be1ff09e09712bef57f6a4e1f550bc9d`
and manifest hash `a2605cd1decd219638e655fc36fb25186ca55c304688726f188afbff509cc866`,
as reported by the audit; the preflight archive was not rechecked in this task.
The historical failed preflight `phase5-inference-preflight-f817aa9732d04827`
is a distinct attempt and remains preserved, not overwritten by either success.

The results completion marker records `scope=paired_correctness_only`,
`formal_tests_computed=true`, `subset_completed=true`, `preflight_only=false`,
`model_forwards=0`, `training_performed=false`, and
`analysis_status=exploratory_post_hoc`. It explicitly retains
`full_supplemental_completion=false`, `stage_a_complete=false` and
`research_readiness_certified=false`. Recorded clean Git status, zero forwards
and training status are provenance assertions supported by the reviewed path,
not independently observed live-server telemetry.

## Complete comparison ledger

Direction is complete ConsenDxFormer minus comparator, same trained seed.
Fractions are exact equal-patient differences; pp values are rounded percentage
points transcribed from the audit. Every row has 183 nodules, 99 patients,
status `computed`, precision 80, raw/adjusted enclosures `[1,1]`, log enclosure
`[0,0]`, and `reject=false`. All 15 belong to one two-sided Holm family at
alpha 0.05. Tied probability ranks are by hypothesis ID, not effect size.

| Hypothesis | Exact difference | Difference (pp) | Holm rank |
|---|---:|---:|---:|
| `C_B1_42` | -73/1980 | -3.686869 | 1 |
| `C_B1_43` | -23/1980 | -1.161616 | 2 |
| `C_B1_44` | 5/216 | +2.314815 | 3 |
| `C_attention_off_42` | 13/1485 | +0.875421 | 4 |
| `C_attention_off_43` | 257/5940 | +4.326599 | 5 |
| `C_attention_off_44` | 1267/11880 | +10.664983 | 6 |
| `C_disagreement_off_42` | -697/11880 | -5.867003 | 7 |
| `C_disagreement_off_43` | -17/270 | -6.296296 | 8 |
| `C_disagreement_off_44` | 25/792 | +3.156566 | 9 |
| `C_ordinal_off_42` | 133/3960 | +3.358586 | 13 |
| `C_ordinal_off_43` | 7/132 | +5.303030 | 14 |
| `C_ordinal_off_44` | 563/11880 | +4.739057 | 15 |
| `C_fusion_off_42` | -19/990 | -1.919192 | 10 |
| `C_fusion_off_43` | -247/5940 | -4.158249 | 11 |
| `C_fusion_off_44` | 41/2376 | +1.725589 | 12 |

## Why all probability bounds equal one

The unchanged, approved Candidate C calculation includes all patients:

```text
U_p = (b_p - c_p) / n_p
S = sum_p U_p
delta_hat = S / 99
p_bound = min(1, 2 * exp(-S*S / (2*99)))
```

Here b counts nodules correct only for the complete model, c those correct only
for its comparator, and n all nodules of that patient. Zero scores and
within-patient cancellations are retained. The audit reports saturation at
one for absolute differences up to 11.833416 pp. The largest observed absolute
difference, 10.664983 pp for attention OFF at seed 44, gives an uncapped bound
approximately 1.138971; it therefore correctly caps to one, as do all others.
The reported unadjusted 0.05 and first-Holm thresholds are approximately
27.298868 and 35.948700 pp. These are algebraic thresholds, not estimated power.
No formula or research statistics were recomputed for this documentation task.

The all-one outcome is the accepted conservative method's result, not a software
failure or justification for choosing another test after seeing outcomes.
Non-rejection neither proves equivalence nor absence of a difference. Effects
vary by seed; seed 44 is positive descriptively for all comparators, but must
not be selected or pooled with repeated patients as independent observations.

Candidate C explicitly substitutes for the failed Durkalski/chi-square method,
under the accepted conditional independent, possibly heterogeneous patient-cluster
mean-null model. Its execution does not repair legacy calibration. Identity
matching, formula agreement and synthetic checks do not establish substantive
patient independence, representative clinical sampling or generalizability.
Do not average p-values, drop zero-score patients, subtract interval endpoints,
or convert these tests into macro-F1 or nodule-accuracy conclusions. The existing
33 descriptive, unadjusted accuracy intervals retain their original meaning.

## Independent audit coverage and limits

The supplied report checked 11 unique regular ZIP members, safe paths, completion,
manifest/payload integrity, content-addressed identity, and the transition from
reviewed preflight (only `identity.preflight_only` changed). It reports byte-identical
protocol/reference/predictions, all 397 source snapshot entries, 18 descriptors
and 369 per-source ledger entries matching earlier descriptive provenance after
prefix normalization, and unchanged bound implementation/arithmetic/governing hashes.
All five embedded R output files matched their supplied originals without invoking R.

It reconstructed all 1,485 integer patient-count rows and exact rational scores
from archived original hard predictions, checked all identities/labels and family
slots, and independently matched enclosures, Holm ranks/decisions and CSV/JSON
records using 160-digit arithmetic. Reviewed pure validators were isolated from
unavailable historical checkpoint-loader dependencies with function bodies unchanged;
all 23 reviewed candidate file hashes were checked. This was not a full historical
integration-suite rerun or live checkpoint/server validation. Earlier pointing,
accuracy, perturbation and timing results were not numerically re-audited by this
paired-correctness audit.

## Evidence-to-requirement reconciliation and remaining closure

Read the frozen [supplemental proposal](phase5_supplemental_evaluation_proposal.md)
as chronology alongside the [amendment decision](phase5_inference_amendment_decision.md)
and [acceptance/implementation release](phase5_inference_implementation_authorization.md).
Their original pending statements remain unchanged. The later actual-R verification,
review/publication, corrected preflight and separate explicit research release are
recorded by the supplied audit and [integration addendum](phase5_inference_integration.md).
No acceptance or execution authorization is backdated.

| Requirement / stream | Evidence and current bounded status | Remaining requirement or retained limitation |
|---|---|---|
| Native pointing-game | [Descriptive subset](phase5_descriptive_subset.md), `phase5-descriptive-04d6daaccc698431`: previously completed/audited | Fixed prior matches or exceeds model pointing scores; spatial correspondence, not causal faithfulness. No rerun required by this audit. |
| 18 individual + 15 paired accuracy intervals | Same completed descriptive bundle; all 33 intervals | Descriptive, unadjusted nodule-weighted intervals, not simultaneous tests; classification superiority over B1 is not established. |
| Insertion and deletion with controls | [Completed perturbation subset](phase5_insertion_deletion_subset.md), `phase5-perturbation-5e31065629b5137c` | Descriptive within-model results only; zero-baseline/OOD, fixed targets, nonmonotonic curves and patient/seed dependence remain. No cross-model faithfulness claim. |
| Timing evidence | Perturbation record embeds approved `phase5-timing-030925dee6228ebb`; [historical timing contract](phase5_synthetic_timing.md) | Synthetic adapter measurements/estimates are not end-to-end research runtime; already completed evidence, not a new timing job. |
| Original McNemar / cluster-aware paired correctness | Explicit Candidate C substitution and this completed, independently audited 15-comparison bundle | Requirement addressed at the accepted amended scope, not by repairing legacy calibration or performing naive McNemar; conditional independence remains an assumption. |
| Multiplicity | All 15 fixed same-seed tests in one two-sided Holm family, computed `[1,1]` adjusted enclosures | 0/15 rejections; not unresolved placeholders or superiority/equivalence evidence. No additional outcome-driven tests. |
| Full supplemental evidence packaging / sign-off | Proposal section 7 described a combined bundle; actual reviewed executions certify separate subsets only | Review and explicitly accept the mapping of subset bundles to the combined proposal's deliverables/provenance, or specify a documentation-only closure scope. No combined completion certificate is supplied; preserve existing markers. |
| Stage A / general research readiness | Every current completion certificate remains false at this broader scope | An authoritative Stage A acceptance checklist and explicit evidence-backed closure decision are not established by the inspected records. Obtain/reconcile that scope before certifying; do not invent a missing experiment. |
| Earlier Phase 4 evidence | [Boundary/evidence-gap record](phase4_evidence_gaps.md) | Boundary descriptive execution completed; uncertainty comparison previously reported complete but not independently reverified there or here. Other primary/attention/ablation statuses are not newly certified by this audit. Missing local artifacts do not justify reruns. |

Thus the original named supplemental endpoints now have bounded completed evidence,
including the explicitly amended inferential method; this does **not** automatically
close the full supplement or Stage A. Verified outstanding work is review/publication
of the accurate current record and broader evidence-to-requirement/packaging sign-off.
Any further Stage A requirement is of unknown status until its authoritative criteria
and existing evidence are supplied and reconciled. No new experiment is identified
or authorized by this reconciliation. Thesis claims must remain consistent with
the primary macro-F1 endpoint, descriptive intervals, non-rejections and all retained
limitations. Preserve original bundles, failed attempts and historical records.
