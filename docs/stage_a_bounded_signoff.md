# R5 bounded Stage A sign-off proposal

## Accepted bounded R5 decision — 2026-09-27

The owner accepts the bounded Stage A closure scope in section 4, including the
limitations in sections 5–6. R5 is accepted and closed for that bounded scope.
Stage A implementation/evidence work is complete within that scope.

This is not unconditional certification, proof of superiority, clinical validation
or a fully passing final test suite. All recorded limitations and the accepted
R3/R4 boundaries remain applicable. Stage B remains a separate, unstarted task.

This dated owner acceptance supersedes historical pending wording below. The
historical proposal, evidence, original result bundles and execution-time
completion markers remain unchanged. No new analyses, tests, training, server
jobs or result modifications are authorized by this acceptance.

## Recommendation — 2026-09-27; owner acceptance pending

**Ready for bounded acceptance of Stage A's implemented and evidenced Phase 0–5
scope, with the approved amendments and limitations below.** This is a proposed
decision, not owner acceptance, certification, a new execution receipt or
permission to begin Stage B. R5 remains pending. No specific missing execution
artifact is identified as blocking this bounded recommendation in the reviewed
records; a stronger claim of unrestricted reproducibility or scientific success
would not be supported.

Reviewed checkout: `phase4-consendxformer`, HEAD
`38a41f00fa2ac62b77d160d86e13c1dde537e7ac`, with a clean initial working tree.
No applicable ancestor/root/docs `AGENTS.md` was found. This review read the
original brief, current reconciliation, accepted decisions and existing evidence
records. It performed no research calculation, test-suite rerun, archive arithmetic
audit, model forward, server access or original-checkpoint verification.

## 1. Authority and evidence references

The original brief is the supplied attachment
`f5d1f11c-c671-49fd-b179-1b720a18e633/pasted-text.txt`, recorded in the
[reconciliation](stage_a_evidence_reconciliation.md) as SHA-256
`eb90ef9f60920198f208553ec6bc3d7b130f648c56b984c19c2210a81964ea1f`.
It requires implementation before writing, actual rather than assumed cohort
counts, B0/B1, a 2D proposed model, ablations and the named evaluation suite.
Its locked novelty decision is not reopened. Later approved cohort, endpoint and
method decisions govern where they differ from initial approximate descriptions.

The reconciliation's detailed Phase 0–5 tables and evidence locator remain the
requirement-level record. Its E reference is the supplied
`Stage_A_Existing_Evidence_Audit_2c3c485e05d0.md` and
`stage-a-existing-evidence-2c3c485e05d0.zip` (recorded ZIP SHA-256
`7046201b4e88ab87509028bdace0351c84b83e10284a587eca063b5ea9d53a40`).
Its H reference, `ConsenDxFormer_Phase4_Evidence_Checklist.md`, consolidates
historical S1–S8 reviews; it is not a fresh audit of every original artifact.

The [accepted R4 record](r4_claims_packaging_acceptance.md), sections 2–4,
supplies exact distributed paths, archive/manifest hashes, governing protocols
and audit filenames for the supplemental bundles and external reference package.
The [accepted R3 record](r3_entropy_results_and_closure.md) supplies the separate
stored-entropy result identities, authorization attribution and validation limits.
This document references those identities rather than constructing another
combined result bundle. Hashes quoted here are existing recorded identities,
not a claim to have rehashed all archives or original sources in this task.

## 2. Concise Phase 0–5 completion checklist

“Evidenced” means completed within the cited implementation/execution/audit scope,
not that every historical source byte or every test has been independently checked
again. Counts and findings below are transcribed, not newly calculated.

| Phase / original requirement | Completion evidence and bounded disposition |
|---|---|
| **0 — environment/configuration/source locks, seeds and exact patient splits** | [Phase 0 README](../README_PHASE0.md), committed `artifacts/phase0/manifests/`, and E's environment/source-lock receipts. Evidenced: 1,010 patients / 1,018 series; 707/152/151 patient split and exact IDs. Pinned upstream is `0e5bcadcadfce5ea99df8484d23705d7bd4d778a`. Early receipt Git fields remain null; DOI-to-GitHub byte equivalence is not established. |
| **1 — B0/B1 training/evaluation infrastructure** | [Phase 1 README](../README_PHASE1.md), `configs/b0.yaml`, `configs/b1.yaml`, scripts 10–12 and E receipts. Both definitions retained: released-source B0 and paper-aligned B1 on the baseline architecture, not a claim of exact reproduction of every paper detail. |
| **2 — data pipeline, actual classes, geometry and cohort accounting** | [Phase 2 README](../README_PHASE2.md), E readiness/canonical/rendered/metadata records, reconciliation's count tables. Evidenced primary cohort: 1,133 nodules, class counts 511/303/319 and train/validation/test counts 807/143/183 under frozen 2–4-reader and source-validity rules. Do not force the paper's 4,095 count. Boundary remains separate: 729 members, three geometry exclusions, 726 eligible. |
| **3 — real multiseed baselines and saved logits/probabilities with patient identities** | [Phase 3 README](../README_PHASE3.md), E's six real B0/B1 runs at seeds 42/43/44 and 12 validation/test prediction files; execution `21d25fde6a268222cc124e194da4d0f502b25d73`. Separate seed-42 receipts complement matrix `phase3-matrix-8df46600bc0c22d1` (43/44). Synthetic runs are not substituted for real evidence. Checkpoint tensor bytes were omitted from E. |
| **4 — frozen 2D model and independently toggleable components** | Reconciliation Phase 4 table, [ablation contract](phase4_ablation_execution.md), H/S1–S6 and later source descriptors. Three complete-model runs and 12 single-component OFF runs are evidenced. Multi-rater diagnostic/ordinal learning, DSA alignment and supporting cross-scale fusion remain frozen. Ordinal OFF disables the auxiliary loss, not necessarily the head; attention-alignment OFF does not remove DSA. |
| **5 — classification, calibration, uncertainty and boundary stress** | [Evaluation contract](phase4_evaluation.md), [evidence-gap record](phase4_evidence_gaps.md), H/S1/S7/S8; primary `phase4-evaluation-38c18ff5e30b0773`, uncertainty `phase4-evaluation-8c5876a39d46aaa7`, boundary `phase4-evaluation-febe8402535570e1`. Classification/class-specific metrics, ECE/Brier/NLL, reader-disagreement association and risk–coverage are evidenced. Accepted R3 adds the exact correctness and true-class entropy partitions, not a substitution of disagreement groups for those partitions. Boundary is descriptive stress testing only. |
| **5 — explanation correspondence, pointing and perturbation** | [Attention contract](phase4_attention_evidence.md), matched source `phase4-evaluation-22cb656eabd040c2`; [descriptive subset](phase5_descriptive_subset.md), `phase5-descriptive-04d6daaccc698431`; [insertion/deletion subset](phase5_insertion_deletion_subset.md), `phase5-perturbation-5e31065629b5137c`. Dice/IoU, native pointing with explicit empty-target misses, and fixed-target insertion/deletion are evidenced. I records 1,098 replay checks, 417,240 attempted/accepted forwards and endpoint reuse. |
| **5 — intervals, paired correctness and multiplicity** | Existing Phase 4 patient-clustered F1/recall intervals; D's 33 descriptive accuracy intervals (18 individual / 15 paired); [inference amendment acceptance](phase5_inference_implementation_authorization.md), [integration](phase5_inference_integration.md), [paired-correctness results](phase5_paired_correctness_results.md), `phase5-inference-results-79f6f8de6cab19b6`. Candidate C explicitly replaces the failed legacy McNemar/Durkalski proposal; all 15 Holm-family tests are computed non-rejections. These are different endpoints, not interchangeable tests. |
| **Later feasibility and packaging dependencies** | I embeds approved synthetic timing `phase5-timing-030925dee6228ebb`; [timing contract](phase5_synthetic_timing.md) and I audit distinguish synthetic estimates from observed curve-execution time. Accepted R4 permits distributed bundles, documented schema equivalents and hash-bound external calibration/R-reference evidence, without a new universal bundle or rewritten markers. |

## 3. Current R1–R4 dispositions

| Item | Current disposition and retained limit |
|---|---|
| **R1** | Missing B0 execution evidence resolved by E's real seed-42/43/44 records. No waiver, synthetic substitution or new training required. Sidecars/recorded checkpoint hashes do not independently verify omitted tensor bytes. |
| **R2** | Early receipts and cohort-count evidence recovered/verified within E's transferred-record scope. Counts are the supplied audit's read-back of saved metadata, not a newly discovered historical standalone report. Null early Git fields, raw-source/header limits and incomplete original approval history remain explicit. |
| **R3** | Owner accepted bounded closure on 2026-09-27 for the two stored-entropy comparisons in `r3-entropy-83023569d06acd93`, under `archived_saved_records`. It contains 24 summaries / 12 contrasts for full and disagreement-OFF seeds 42/43/44. No independent original prediction/supervision/checkpoint-byte verification is implied. |
| **R4** | Owner accepted on 2026-09-27; published in current HEAD. Acceptance covers distributed packaging/schema equivalents, retained access to the exact hash-bound external calibration archive, and bounded claims. It does not promise permanent public availability, a combined supplement, fresh checkpoint verification, superiority or Stage A certification. |

Do not request repeat approval of R3/R4 or infer they are still unresolved from
older paragraphs. The dated acceptance blocks supersede historical pending
wording. Earlier D/I unresolved hypothesis registers are superseded as current
status by C's computed results, not edited retrospectively. Old R `NOT_RUN`
records and failed preflights/calibration remain preserved chronology. Original
full-supplement/Stage A false flags remain valid execution-time scope declarations.

## 4. Exact proposed closure scope

For owner review, propose the following decision, **not adopted by this document**:

> Accept Stage A as complete only for the documented implementation and existing
> execution/audit evidence of the frozen 2D ConsenDxFormer Phase 0–5 program,
> with the later approved cohort/method amendments, accepted R3 saved-record
> validation contract and accepted R4 distributed evidence arrangement. This is
> bounded administrative/evidence sign-off, not certification of universal
> reproducibility, scientific superiority, clinical validity or a fully passing
> test suite. Preserve every source/result bundle and execution-time marker.
> Any later writing or new scientific work requires a separate task.

This covers fixed B0/B1 definitions, full/ablation seeds 42/43/44, frozen
checkpoints, patient-disjoint primary splits, source-invalid exclusions and
the documented evaluation definitions. Primary test support remains 183 nodules
from 99 patients. It accepts actual cohort counts rather than paper targets,
and the amended secondary inference scope rather than claiming a formal test of
every accuracy/F1 delta. It does not reopen novelty selection or relabel boundary
cases, substitute a favorable ablation, authorize tuning, or certify the original
pre-implementation design discussion as fully recovered.

## 5. Scientific, validation and evidence-access limits that survive acceptance

- **Endpoint/claim separation:** macro-F1 remains primary. F1/accuracy intervals
  are descriptive and unadjusted, not simultaneous intervals or superiority
  tests. Candidate C tests secondary equal-patient mean paired correctness,
  not macro-F1, nodule-weighted accuracy or an invented patient diagnosis.
  All 15 raw and Holm-adjusted probability enclosures are `[1,1]`, with zero
  rejections. These computed conservative bounds establish neither equivalence,
  non-inferiority nor superiority. Conditional independent heterogeneous-cluster
  assumptions are not proved by source hashes, outcome-blind splitting or R checks.
- **Mixed/adverse evidence:** overall classification superiority over B1 and
  improved average calibration are not established. Ablations do not demonstrate
  uniform benefit from disagreement weighting/fusion. The fixed spatial prior
  matches or exceeds model pointing scores. Complete-model DSA/prior perturbation
  differences are small and mixed; insertion advantage over hash-order accompanies
  less favorable deletion. B1 DSA is less favorable than its own prior in both
  directions at all seeds. These are completed findings, not unfinished work.
- **Scope of explanation and uncertainty:** annotation-informed, centered 2D
  crops are not full-scan detection, 3D segmentation or clinical validation.
  Spatial agreement is not causal faithfulness. Perturbations use a possibly OOD
  zero baseline and fixed saved predicted-class targets; nonmonotonic curves and
  differing model targets preclude a cross-model faithfulness claim. R3 mean
  entropy differences are descriptive, not perfect separation or significance.
- **Dependence and chronology:** nodules cluster within patients; the same
  patients recur across seeds. Supplemental details were fixed after test results
  were seen: exploratory post-hoc, not preregistered/confirmatory. Boundary's
  evaluated 726 nodules/450 patients include 198 training-overlap patients (44.0%);
  this is not independent held-out performance. No clinical utility, external
  validity or SOTA conclusion follows.
- **Validation is not a green final suite:** the R3 results record reports final
  targeted tests of 47 passed, but an earlier full-suite attempt had 1,061 passed
  and four failures (three Windows symlink-privilege failures and an existing
  PyTorch autocast API incompatibility). That attempt preceded final R3 hardening.
  There is no claimed green final full suite; no tests were rerun here. Bounded
  evidence acceptance is not remediation or concealment of those failures.
- **Source and runtime verification:** historical audits/receipts have declared
  scopes. Original DICOM/XML cryptographic equivalence, omitted image/mask and
  checkpoint bytes, live GPU identity and live server preservation are not freshly
  verified here. R3 execution authorization is attributed to the owner-provided
  conversation record, not an independently verified receipt file. Timing estimates
  from synthetic weights and observed curve-execution duration have distinct
  scopes; neither is an end-to-end runtime guarantee.
- **Evidence custody:** retain the external calibration archive
  `D:/Downloads/phase5_candidate_c_core_review (1).zip`, SHA-256
  `6c6c6c0bb308813fdaea2217c131cd34465cbb46e1fae1c408d62ac48c2e1e47`,
  as required by accepted R4. Local access is not permanent public availability.
  C embeds selected calibration/reference records and actual R outputs, not the
  complete external campaign ledgers. R4's index provides the exact bundle/hash
  references; no new result packaging is necessary or authorized here.

## 6. Genuinely pending items versus accepted limitations

**The remaining decision is R5 owner review/acceptance of the scope above.** It
has not occurred. Stage A remains uncertified at this proposal stage. No missing
named evaluation or essential reference was identified as preventing this bounded
recommendation from the accessible records. In particular, no new macro-F1 test,
calibration campaign, archive merger or repeated R3/R4 approval is required.

There are residual traceability limits, not claims of recovered evidence:

- H does not supply exact S4–S6 ablation-analysis bundle IDs. Their existing
  review/manifest IDs would be needed for an exhaustive original-analysis archive
  index. H reports their completed reviews, and later D/C source descriptors
  corroborate completed OFF sources, but are not replacements for those IDs.
- The complete original gap-analysis/thesis-alignment/2–4-option discussion and
  phase-by-phase confirmation transcript are not reconstructed. Original dated
  design/approval records would be needed to certify that full historical process;
  current code and commit order cannot supply them. The locked decision and later
  accepted amendments are preserved without demanding repeat approval.
- A fresh all-green final software-validation claim would require an actual
  applicable final validation report; the available historical suite report does
  not support it. Such a claim is excluded from this proposed sign-off, not silently
  assumed or made into an unauthorized testing task.

These limitations remain visible in the proposed bounded acceptance. If a stronger
scope is desired, identify its specific evidence requirement separately; do not
invent evidence, prescribe repeat research execution, or broaden this proposal.
Stage B/C manuscript deliverables are subsequent work, not missing Stage A result
bundles, and are not started or authorized here.
