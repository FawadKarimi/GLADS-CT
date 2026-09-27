# R4 claims-and-packaging acceptance proposal

## Accepted bounded R4 decision — 2026-09-27

The owner accepts R4 only for the documented distributed packaging, schema
equivalents, hash-bound external calibration evidence arrangement and bounded
claim limits. Retain access to the exact external calibration archive identified
below and in the R4 proposal; no result repackaging is authorized or required.
This is acceptance of separate existing bundles, not a claim that one combined
supplement already exists.

This dated decision supersedes the pending-R4 wording preserved below as
historical proposal and review context. It does not claim permanent public
availability, fresh original-checkpoint verification, model superiority or
Stage A certification. All original bundles, results, historical completion
markers and verification limitations remain unchanged. R3 remains accepted
within its existing bounds. R5 remains unresolved; Stage A remains uncertified.

## Status — 2026-09-27; owner acceptance pending

Prepared against `b5804990c7c550e5fc74088dcbaa7089f7661925` on
`phase4-consendxformer`; branch and HEAD matched and the initial working tree
was clean. This is a proposed R4 decision, not an adopted closure, a combined
result bundle, a new analysis or Stage A certification. R5 remains unresolved.
The [accepted bounded R3 decision](r3_entropy_results_and_closure.md) remains
accepted and is not reopened here.

Recommend accepting the existing **distributed, separately scoped bundles**
and the schema equivalents below for the supplemental packaging requirement,
subject to explicit acceptance of their claim limits and external-evidence
custody. Do not manufacture an omnibus completion marker, replace tables,
rename old outputs or rewrite historical partial-completion flags.

## 1. Governing records and chronology

- [Frozen supplemental proposal](phase5_supplemental_evaluation_proposal.md),
  especially sections 2–7: LF-normalized SHA-256
  `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`.
- [Inference-resolution proposal](phase5_inference_resolution_proposal.md),
  [amendment decision](phase5_inference_amendment_decision.md) and
  [implementation authorization](phase5_inference_implementation_authorization.md)
  distinguish the failed legacy calibration, approval in principle, and later
  acceptance of Candidate C's independent, possibly heterogeneous patient-cluster
  sampling/null interpretation. The frozen proposal is historical, not rewritten
  to imply that Candidate C was its original method.
- [Inference integration](phase5_inference_integration.md) and
  [paired-correctness results](phase5_paired_correctness_results.md) document the
  later R-reference integration, preflight and separately authorized execution.
- [Stage A reconciliation](stage_a_evidence_reconciliation.md), section 3 and R4,
  defines the packaging/claims decision addressed here. R1/R2 resolution is bounded
  by its existing evidence; accepted R3 is a separate later decision.

The synthetic validation and implementation approvals did not themselves authorize
research execution. All supplemental analyses remain exploratory post-hoc: their
details were fixed after test results had been seen. Neither this cross-reference
nor the later approvals establish preregistration or confirmatory superiority.
The original McNemar requirement was explicitly amended to Candidate C, not
silently omitted or relabeled as naive nodule-level McNemar. The known asymmetric
null counterexample and failure of the former 20-score safeguard remain part of
the methods history, not evidence that the legacy test became calibrated.

## 2. Cross-bundle index and verification boundary

Paths below are repository-relative **recorded output locations**, not claims of
access to the university filesystem. D/I/T/C are aliases used only in this document.
Each bundle retains its own `artifact_manifest.json` and `COMPLETED.json`.

| Alias and exact identity | Recorded root | Governing protocol and execution record |
|---|---|---|
| D: `phase5-descriptive-04d6daaccc698431` | `artifacts/phase5/pointing_and_accuracy/phase5-descriptive-04d6daaccc698431/` | `configs/phase5_descriptive_v1.json`; proposal sections 2/6; [descriptive subset](phase5_descriptive_subset.md); execution `75b5d7ce895338e4b21747c05b039850eae2b936` |
| I: `phase5-perturbation-5e31065629b5137c` | `artifacts/phase5/insertion_deletion/phase5-perturbation-5e31065629b5137c/` | `configs/phase5_insertion_deletion_v1.json`; proposal section 3; [insertion/deletion subset](phase5_insertion_deletion_subset.md); execution `422c8de64cf0e6d9f942a7a556f5473133cc7d8d` |
| T: `phase5-timing-030925dee6228ebb` | `artifacts/phase5/synthetic_timing/phase5-timing-030925dee6228ebb/`; consumed copy at I `provenance/timing/` | `configs/phase5_synthetic_timing_v1.json`; timing evidence described in the insertion/deletion subset record; synthetic scope, not research authorization |
| C: `phase5-inference-results-79f6f8de6cab19b6` | `artifacts/phase5/paired_correctness/phase5-inference-results-79f6f8de6cab19b6/` | `configs/phase5_inference_v1.json`; governing amendments above; [paired-correctness results](phase5_paired_correctness_results.md); execution `4141ae78b8e0a28cbc5779cc4190f83bf7cdae8d` |

All hashes in this table are SHA-256 of the **raw ZIP or manifest file bytes**,
not a canonical payload-map digest. They were freshly checked from local copies
in `D:/Downloads/` for this documentation task. The I-embedded and standalone T
manifest bytes yield the same manifest hash.

| Alias / local ZIP filename | ZIP SHA-256 | `artifact_manifest.json` SHA-256 |
|---|---|---|
| D / `phase5-descriptive-04d6daaccc698431.zip` | `faf1799e7f722ade581720febca5bb089068f89ca6b18655f63a5ec4cd49ad51` | `d9e27c377368d6c67e76de8709868f77ca82826a356a2a52860989dd9c3c494d` |
| I / `phase5-perturbation-5e31065629b5137c.zip` | `c15135ec3576dfaca00f80fdd43b8de15c53a4443b42d792333a392515e8c320` | `00f559e83a92ea8400e0afd5dd875166ee46e509d0074cccbc156a46a65839b9` |
| T / `phase5-timing-030925dee6228ebb.zip` | `72c9e7deb243bb601d23d094346cf2ff2dbe6fdbf4528a5f9c39591e028a7c29` | `688ab84bbc7ea90c4154954daa5dcbae6d40f9e2e70c63820455fe151693aa6d` |
| C / `phase5-inference-results-79f6f8de6cab19b6 (1).zip` | `4e72b918ef4caf22f7bdd6a6454ba09b2b2b74fdbd0a504521fcaed1dcc4f20c` | `4c82d7141e0a09f58bdd58793f3af9c2ddb39607f3e3090dc01487884a656200` |

The corresponding supplied independent audit reports, also available locally,
are `Phase5_Descriptive_Execution_Audit_04d6daaccc698431.md`,
`Phase5_Insertion_Deletion_Execution_Audit_5e31065629b5137c.md` and
`Phase5_Research_Results_Audit_79f6f8de6cab19b6.md`. Their arithmetic, full payload
integrity and source-preservation findings are **historical audit attribution**.
This task checked archive identities, manifest bytes/inventories, existing
governing-document/configuration identities and selected evidence metadata. It
did not repeat those numerical audits, verify every payload, rerun tests, inspect
live original checkpoints or GPUs, or independently observe server preservation.
An archived source hash reference is not a fresh hash of the original file bytes.

Dependencies retained by these records include matched native attention/prior
evidence `phase4-evaluation-22cb656eabd040c2`: the descriptive subset records ZIP
SHA-256 `e182e2a39e6c0881c6ef021fac9535d3d45fa49cd9320a4ae0be538dacb5e7b2`
and manifest SHA-256
`daaedff0f11f642318d04ad23536e522e7d50f9a277fe50087c8cc7bc750b07b`.
Those two dependency hashes are quoted from the existing record, not freshly
verified here. Its native maps, masks, input arrays and training-only prior remain
in the original evidence source, not newly copied into an omnibus supplement.
Exact original source paths/identities remain in D/I provenance.

C's successful preflight is `phase5-inference-preflight-5b67b8e275de62f5`;
the results record attributes ZIP hash
`5c45baf46e319b02feba9c31a66f75c4be1ff09e09712bef57f6a4e1f550bc9d`
and manifest hash
`a2605cd1decd219638e655fc36fb25186ca55c304688726f188afbff509cc866`
to the audit. These preflight hashes were not rechecked here. Failed preflight
`phase5-inference-preflight-f817aa9732d04827` remains a distinct preserved attempt.

## 3. Original deliverables mapped to existing exports

This maps the frozen proposal's section 7, not a promise that its literal directory
tree was implemented unchanged. Paths in this table are relative to D/I/T/C above.
No equivalent means that two different estimands have become interchangeable.

| Original deliverable or contract | Existing files / documented equivalent | Proposed disposition and literal difference |
|---|---|---|
| One `artifacts/phase5/supplemental_evaluation/phase5-supplement-<identity_hash>/` namespace | Separate D/I/T/C roots | Accept distributed packaging explicitly; the proposed combined bundle does not exist. |
| Versioned protocol, proposal identity, chronology, approvals, family and failure rules | D/I/C `protocol.json`; I `provenance/reviewed_proposal.md`; governing documents/configurations in section 1 | Accept per-subset protocol plus immutable references; there is no universal `protocol.json`. Later inference amendment governs C. |
| Source/configuration/checkpoint/prediction hashes, locks, cohort identities, external pin and source snapshots | D/I `provenance/provenance.json`; C `provenance.json`, `evaluation.json`, `saved_predictions.json`; T `source_snapshot.json`, `models.json` | Accept separate schemas, not one merged provenance file. Source descriptors do not embed or independently reverify every original source byte. |
| Native maps, fixed binary targets, training-only prior and flagged empty target | Original matched-attention source; D `tables/pointing_per_nodule.csv`, `pointing_per_seed.csv`, `pointing_aggregate.csv`, `flagged_case.csv` | Direct metric exports; original arrays stay in their validated dependency. Retain all 183 nodules and explicit empty-target misses, including `LIDC-IDRI-0052__nodule-49c6132bfb77f3f1`. |
| Paired pointing descriptions and traceability | D `tables/pointing_paired_descriptive.csv`, `pointing_paired_per_nodule.csv` | Direct descriptive equivalent; no causal faithfulness interpretation. |
| Fixed perturbation orders and raw curves | I `orders.npz`, `curves.npz`, `tables/perturbation_curves.csv`, `perturbation_mean_curves.csv`, `units.csv` | Direct exports, with 65 positions and fixed targets/orders; no replacement curves needed. |
| AUC per nodule/seed and within-model ordering comparisons | I `tables/perturbation_auc_per_nodule.csv`, `perturbation_per_seed.csv`, `perturbation_paired_descriptive.csv`, `perturbation_paired_per_nodule.csv` | Direct equivalents; preserve direction: higher insertion and lower deletion are favorable. |
| Global replay barrier, raw-float32 policy, endpoint reuse and forward budget | I `replay_barrier.json`, `replay_checks.json`, `replay_cache.npz`, `baseline_cache.npz`, `call_ledger.json`, protocol/provenance | Audited 1,098 successful unperturbed replays before perturbation; 417,240 attempted/accepted calls, 6,588 curves and 428,220 positions. Endpoint reuse is not extra forwards. |
| Timing evidence, environment/device, actual/estimated runtime distinction | I `provenance/timing/` embeds T; T `bootstrap.json`, `identity.json`, `plan.json`, `specification.json`, `measurements.json`, `models.json`, `source_snapshot.json`, synthetic NPZs and `summary.md`; I evaluation/provenance | Later measured evidence supplements the frozen protocol's historical null field; it does not rewrite that field or authorize research. See runtime limits below. |
| Literal `correctness_per_nodule.csv` | D `tables/accuracy_per_nodule.csv`; C `saved_predictions.json` with identities, labels and predictions | Sufficient saved per-nodule correctness inputs are exported; **no dedicated 15-contrast per-nodule correctness CSV** exists. Accept this schema difference rather than generate a new table. |
| Literal `correctness_per_patient.csv` | C `tables/patient_counts.csv`, `inference.json` | Accept patient/contrast counts and exact fractions as the approved Candidate C representation, not patient diagnosis labels. |
| Full hypothesis register | C `tables/hypotheses.csv`, `inference.json`, `protocol.json`; historical D/I `tables/hypothesis_register.csv` | All 15 slots are computed in C. D/I's unresolved registers remain execution-time history and must not be rewritten. |
| Literal `clustered_tests.csv` and `multiplicity.json` | C `tables/hypotheses.csv`, `inference.json`, `protocol.json` | Literal files absent; equivalent records provide method, endpoint, counts, probability enclosures, Holm family/order and decisions. Candidate C replaces the failed legacy proposal by explicit amendment. |
| 18 individual plus 15 paired accuracy intervals and common patient draws | D `accuracy_bootstrap.npz`, `tables/accuracy_intervals.csv`, protocol/evaluation | Direct 33-interval export; 10,000 replicates, seed 20260914, common cluster draws and prescribed quantiles. No rerun of the earlier multi-metric bootstrap suite. |
| Calibration register, generators, exact supports, tail checks, ledgers, reference outputs and failures | External core package/run02 plus selected embedded C `evidence.json` records (section 4) | Accept hash-bound external evidence explicitly; full campaign ledgers are **not embedded** in C. Preserve earlier failures and historical R-not-run status alongside later R evidence. |
| Evaluation/summary, denominators, undefined outcomes, seed variation and limitations | Separate D/I/C `evaluation.json`, `summary.md`, protocols and results documents; T `summary.md` | No omnibus evaluation or summary exists. This proposed index is a documentary cross-reference, not an analysis product or combined completion certificate. |
| Manifest, completion linkage, no-overwrite, failure/partial scope, deterministic exports | Each bundle's manifest/marker and existing implementation/audit records | Accept separate subset completions. Preserve all historical false full-supplement/Stage A flags; this document changes no marker. Numeric read-back and synthetic guard coverage remain attributed to existing audits/tests, not rerun here. |
| NPZ no-pickle policy, deterministic array hashes, exact CSV identities/precision, null reasons and fail-closed policies | Subset protocols, implementations and documented read-back/synthetic validation in the D/I/C subset and integration records | These remain applicable contracts, not waived by accepting different filenames. This metadata-only review does not independently recertify every serialization or failure-path test. |
| Synthetic integration tests for pairing, three seeds, mutations, path escape, no-overwrite and completion ordering | Existing subset test/validation records and the Candidate C integration history | Existing evidence is referenced, not rerun. The historical blocked-method rule is not bypassed: C uses the explicitly approved replacement method; partial D/I completion never certified the full formal family. |

The substantive exports are present within their approved subset contracts. The
literal omissions above are real packaging differences, not missing model runs
or a reason to recompute results. A universal completion record and the original
combined directory/schema have not been produced; accepting their replacement
by distributed references requires the owner's R4 decision, not an inference
from successful execution. Final Stage A disposition still requires R5.

## 4. External calibration and independent R-reference evidence

The available local archive `D:/Downloads/phase5_candidate_c_core_review (1).zip`
has freshly checked SHA-256
`6c6c6c0bb308813fdaea2217c131cd34465cbb46e1fae1c408d62ac48c2e1e47`.
Its root is `phase5_candidate_c_review/`, including `PACKAGE_SHA256.json` and
`validation/phase5_candidate_c/runs/run02/`. That run contains
`scenario_register.json`, `exact_support.csv`, `tail_checks.csv`,
`S00_ledger.npz` through `S05_ledger.npz`, `VALIDATION_STATUS.json` and
`artifact_manifest.json`. The latter's raw hash was freshly checked as
`8b86629329664dd7915525d6e7cc207fd69f93b27bd608599accc41822dbf5a9`.
Run01's separate manifest hash is
`269817cc714077879411759907d65eeb8e7b3e7d4a132c242f128456f4ac285e`.
Run01/run02 are not two independent campaigns to double the simulation budget.

C `evidence.json` binds this review ZIP and package manifest, selected run02
status/manifest records, `evidence/independent-readback-run02.json`,
`evidence/unit-tests-final.txt`, and fixtures under
`validation/phase5_candidate_c/runs/r_reference_inputs/`:
`binomial_cases.csv`, `fixture_manifest.json`, `holm_cases.csv`,
`legacy_counts.csv`, `legacy_expected.csv`. The full large calibration ledgers
remain in the external package. The historical synthetic evidence covers exact,
asymmetric/heterogeneous/sparse and Holm-tail cases; it is not a proof of every
possible distribution or evidence that the research sampling assumptions hold.

The actual five R outputs are preserved as text in C `evidence.json` under
`r_outputs`, with the receipt under `r_hashes_text`. These are the recorded raw
file SHA-256 identities, not a newly executed R comparison:

| R reference file | Recorded SHA-256 |
|---|---|
| `STATUS.txt` | `52d58973b78d29046251a02203ec8e81f185976be9ba4300729e6decc98c37a9` |
| `reference_identity.txt` | `fba8027cbdd173d78358d80c7c851ca8149c6a847f5713887c5feec5e1d9c9e8` |
| `legacy_results.csv` | `6ffd83e07f29c195d5bbec97a6847c07b5cfe44b7c6eed1b0deeb08b61468c8f` |
| `holm_results.csv` | `50cf456bb48830cf611e671f67f5fb4bdd35a5b9d7932974fe3e94aaf1ea2fbf` |
| `binomial_results.csv` | `e93b7d715159c3bb05954c7d870b2e811a273ffe07a1a3c229067b62abab624d` |

The recorded receipt hash is
`15b164ca6c12c0f85a782a1989501135b64baf3c11e3f20d99a54d0f0bb84ff2`.
The reference identifies R 3.6.3 and `clust.bin.pair` 0.1.2, source commit
`e464ac7e5d094bffd25b7c4cbba67820b60f1cc1` and Durkalski source blob
`f6523d982dca310c1a76954c9965e36aaa2f9cf1`. Historical validation reports 26
legacy, 90 Holm and nine binomial fixture comparisons. R agreement verifies the
reference computations; it does not repair the legacy calibration failure,
prove Candidate C's sampling assumptions, or authorize new research work.

No essential reference for this **bounded packaging proposal** was found
unavailable: the external core archive, embedded R evidence and supplied audit
reports are locally available. Their durable custody/distribution remains a
packaging decision: retain the exact hash-bound external core archive with the
bundle index, rather than present C alone as containing its full campaign.
No permanent public archive URL is established by these local paths. Original
research source/checkpoint bytes are not independently reverified here; this
known verification boundary must not be disguised as fresh source validation.

## 5. Required claim limits and adverse findings

1. **Macro-F1 remains primary.** Existing F1 and accuracy intervals are descriptive,
   unadjusted intervals. D's nodule-weighted accuracy intervals do not become
   multiplicity-adjusted or formal inference by being indexed beside C.
2. Candidate C concerns the secondary equal-patient mean paired correctness
   difference, using all nodules within each patient. It is not a significance
   test of macro-F1, nodule-weighted accuracy or a created patient diagnosis.
   Its independent, possibly heterogeneous cluster null is an explicit approved
   assumption, not established by hash splitting or source validation.
3. All 15 comparisons (complete versus B1 and four OFF variants at seeds 42/43/44)
   were computed in one two-sided Holm family. All raw and adjusted enclosures
   are `[1,1]`, with no rejections. These non-rejections do not establish
   equivalence, non-inferiority or superiority; conservative bounds are not
   evidence of equal performance. Repeated patients across seeds are not new
   independent patients.
4. D records that the fixed spatial prior matches or exceeds complete-model
   pointing scores. Empty targets are misses, not silently removed. Pointing is
   localization agreement, not causal faithfulness. Classification superiority
   over B1 is not established; adverse ablation findings are not erased or used
   for retrospective model selection.
5. I records small mixed complete-DSA versus prior differences. Its insertion
   advantage over hash-order accompanies less favorable deletion. B1 DSA has
   less favorable mean AUC than its own prior in both directions at all seeds.
   These are within-model descriptions, not cross-model faithfulness,
   classification-superiority, clinical-utility or SOTA claims. Preserve zero
   baseline/OOD, fixed predicted-target (including incorrect predictions),
   nonmonotonic curves, patient/seed dependence and post-hoc limitations.
6. T measured 2,496 synthetic calls (192 warm-up, 2,304 measured, 18 blocks),
   using synthetic weights. The historical extrapolation of
   1,902.490166971094 seconds is an estimate, not research runtime. I's observed
   2,147.845612185076-second curve-execution scope excludes preparation/final
   export and is not end-to-end wall time. Different scopes, contention,
   startup/I/O and synthetic weights preclude treating their difference as a
   calibrated runtime error. The frozen historical null and later evidence coexist.

These findings are transcribed from existing results/audits, not newly calculated.
No endpoint, threshold, cohort, seed, model, target, baseline or completed result
is changed by the proposed acceptance.

## 6. Separate accepted R3 reference

[R3 results and closure](r3_entropy_results_and_closure.md) records the owner's
2026-09-27 acceptance for correct/incorrect and true indeterminate/definitive
stored-entropy descriptions under `archived_saved_records` only. Its result is
`r3-entropy-83023569d06acd93`, at
`artifacts/phase5/r3_entropy/r3-entropy-83023569d06acd93/`.
The accepted record gives archive SHA-256
`840407afc53e0bc12d2525a6b124aeea789f862cc38526995506c99a4be00d40`
and manifest SHA-256
`30c8a36003663c5a4521e8dfcbe08248ca473b97ef700d03f6cd0ef74b34beda`.
These are references to the accepted record, not a new R3 archive audit here.
Its 24 summaries and 12 contrasts are separate from the original supplement's
33 accuracy intervals and 15 hypotheses. Its execution-time
`r3_review_closed=false` marker remains historical; the later owner decision
is documentary. Original prediction/supervision/checkpoint byte verification
is not implied. R3 acceptance neither adopts R4 nor certifies Stage A.

## 7. Specific pending acceptance decisions

Propose that the owner explicitly accept:

- The section 3 equivalents and distributed D/I/T/C package in place of the
  original single namespace, universal summary/provenance and literal CSV/JSON
  filenames, including the absence of a dedicated paired per-nodule correctness
  CSV. This accepts existing evidence, not a newly combined supplement.
- The hash-bound external calibration package plus C's embedded reference
  records as the reference-evidence arrangement, with retention/access to the
  external archive. A durable distribution locator can be supplied if required;
  no rerun or replacement tables are needed to decide this.
- The distinct endpoints, approved inference amendment, exploratory chronology,
  adverse findings and verification/claim limitations in this proposal.

Until that acceptance is recorded, **R4 remains pending**. R5 still requires the
separate final bounded Stage A decision described in the reconciliation; it is
not discharged by this proposal. Stage A remains uncertified. There is no
authorization here for additional experiments, result repackaging, manuscript
claims beyond these limits or changes to original completion markers.
