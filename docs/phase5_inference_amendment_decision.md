# Phase 5 inference amendment: approval record and sampling/null review

**Prepared:** 2026-09-18 (Asia/Jakarta)
**Repository:** `FawadKarimi/ConsenDxFormer`
**Branch:** `phase4-consendxformer`
**Reviewed GitHub HEAD:** `ca6c018ffbedae12988074fb66052592212a3c2f`
**Status:** Candidate C approved by the user **in principle**; the specific sampling/null interpretation below is a recommendation for acceptance, not a backdated approval or an execution authorization.

## 1. Decision and approval chronology

**Recommendation:** retain Candidate C and use the proposal's **independent, possibly heterogeneous patient-cluster, model-based finite-design mean interpretation**. Do not claim that the hash split supplies a probability-sampling design, that patients are identically distributed, or that the effect represents all clinical patients. This interpretation preserves equal patient weighting and the aggregate mean-zero question, while making the required independence assumption explicit. Its mathematical validity is conditional on that assumption; the code review cannot establish that assumption empirically.

The sequence established in the conversation is:

1. The methods-only proposal, dated 2026-09-17, recommended Candidate C but did not approve a substitution or any execution [R1].
2. The assistant recommended Candidate C in principle, subject to documenting and reviewing the sampling/null interpretation before coding. The user replied, **“ok all lets take action what should we do next”**. This record preserves that in-principle approval; it does not ask for it again. An exact timestamp for that message was not independently verified in this task.
3. The user requested continuation in ordinary Chat after the Work/Codex limit. Changing the chat does not revoke the earlier approval or authorize new scientific work.
4. This document records the remaining sampling/null recommendation and the evidence supporting it. Acceptance of that interpretation and release of a separately scoped implementation/synthetic-validation task remain outstanding. No supervisor endorsement is asserted.

**The archived proposal remains unchanged.** Its statement that approval was pending is historical provenance, not a reason to erase the later conversation approval. Conversely, approval in principle is not evidence that synthetic validation or research testing has occurred.

## 2. Evidence identity and verification limits

The connected GitHub branch was read during this review and still pointed to the reviewed HEAD. The commit records the audited insertion/deletion documentation [G0]. This review did **not** inspect the current Windows working tree, local untracked files, university filesystem, or trained checkpoint binaries. It did not execute repository modules.

Two similarly named supplemental files were recovered from the Library. The revised copy was selected by content identity, not filename recency:

| Source | Verified identity |
|---|---|
| Revised frozen supplemental proposal, Library filename `phase5_supplemental_evaluation_proposal(1).md` | LF-normalized SHA-256 `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`; Git blob SHA-1 `f52a6490884be60c691c5ba2d8050dcc150f182b` |
| Inference-resolution proposal, Library filename `phase5_inference_resolution_proposal.md` | LF-normalized SHA-256 `30a15d75f3bb09a6c44680c652100354713a492e2f0a5bfd2df3eed76699a315` |
| Older supplemental proposal, Library filename `phase5_supplemental_evaluation_proposal.md` | LF-normalized SHA-256 `2eac673f8cf1db8983f5c48f955d3ec172f059471000a241f07a80f9485583e3`; **not** the governing revised proposal |

The revised copy's calculated Git blob identity matches the connector-reported blob for the frozen repository file at the reviewed commit. This establishes the selected document's content identity, not immutability of an unavailable local/server copy. Both historical versions were preserved.

Hash convention: SHA-256 over original UTF-8 file bytes after replacing CRLF with LF and then remaining CR with LF; no trimming, Unicode normalization, or added final newline. The recovered proposal files already use LF, so their raw and LF-normalized hashes coincide. Git blob identity is SHA-1 over `blob <byte-count>\0` followed by original bytes.

The completed subset records were available in the conversation and were used for chronology and preservation requirements only. No new prediction table, patient error vector, research array, or checkpoint was opened, and no method was selected by observed model contrasts. This is a code/document review, not a reproduction of source preparation or an independent recruitment audit.

## 3. Verified split and cohort construction

All code references below are pinned to the reviewed HEAD. Source windows identify inspected code, not a claim to have rerun it.

| Step | Code evidence | What it establishes |
|---|---|---|
| Allocate split counts and rank patients | `src/consendxformer/manifest.py`, `allocate_split_counts`, `build_patient_split`; inspected lines 57–134 [G1] | Deduplicated patient IDs are ranked by `patient_rank_hash(pid, seed, algorithm_version)`. Fixed counts use the configured ratios and largest-remainder allocation. The ranking function receives IDs and fixed configuration, not nodule labels, predictions, or correctness differences. |
| Propagate patient membership to series | `manifest.py`, `build_manifests`; lines 177–223 [G2] | A many-to-one patient merge assigns every series its patient's split. A cross-split patient check fails on overlap. This establishes the implementation's non-overlap rule, not sampling independence. |
| Define primary versus boundary labels | `src/consendxformer/lidc_targets.py`, `paper_mean_label`, `compute_rating_targets`; inspected tail beginning at line 270 [G3] | Integer arithmetic defines benign mean below 2.5, indeterminate mean exactly 3.0, and malignant mean above 3.5. Other means, including 2.5 and 3.5, are boundary-zone and primary-ineligible. This is annotation-outcome-based eligibility. |
| Restrict rendering by source status | `src/consendxformer/lidc_render_pipeline.py`, `run_rendering`; lines 1–210 [G4] | The renderer skips canonical rows whose status does not start with `valid`; for processed rows it checks frozen identity, source availability, mask hash and DICOM identity. This describes that function's rule only; downstream source guards remain authoritative and must not be weakened. |
| Select the configured cohort view | `configs/phase4_consensdxformer.yaml`, `data.cohort_view`; and `src/consendxformer/baseline_metadata_export.py`, `export_baseline_metadata` [G5, G6] | The configured view is `paper_aligned_multi_reader_2_to_4`. The exporter first reconciles rows with frozen patient/series/split identities, then applies primary eligibility and requires two to four valid malignancy ratings. It does not choose a class-balanced quota or rank patients by comparative model performance. |
| Preserve the primary cohort in Phase 4 | `src/consendxformer/phase4_readiness.py`, `_phase4_primary_frame`; inspected lines 1–260; `phase4_supervision.py`, `build_supervision_frame` [G7, G8] | Phase 3 remains authoritative for membership. Phase 4 uses one-to-one left joins and rejects missing rendered/canonical supervision instead of silently inner-joining cases away. This review does not replace the historical Phase 3/source-validity audit. |

**Key distinction:** the split is outcome-blind in the inspected code, but the resulting eligible cohort is not outcome-blind. The latter intentionally excludes boundary-zone/unrated cases and uses the frozen reader/source rules. That limits the effect's target to the selected eligible-cluster setting; it is not evidence of a new defect requiring relabeling, changing splits, or restoring excluded cases.

No inter-patient class-balancing quota or model-error-dependent inclusion rule appears in the inspected split, rendering, export, and supervision functions. This is a bounded code finding, **not** proof that the original collection's external recruitment, site, acquisition, reader assignment, or upstream selection mechanisms were independent. Those mechanisms were not established by this task.

## 4. Recommended sampling model and precise null

### 4.1 Reference experiment and target

Use a **model-based, independent heterogeneous-cluster interpretation**, the finite-design option already described in section 1 of the resolution proposal [R1]. It is not design-based inference from random allocation of patients or model labels.

Let `G` denote the fixed fitted classifiers and their declared selection histories, the frozen preprocessing/label/eligibility rules, the selected design positions and any legitimately fixed or conditioned-on design information. The reference experiment consists of independent draws of eligible patient-cluster records at those positions from distributions `F_1, ..., F_99`, conditional on `G`. These distributions need not be equal. They refer to the eligible record-generating setting, **not** invented repeated CT scans of the same named people.

For each position, all retained nodules belong to its patient cluster. The cluster may contain an internally dependent number of nodules and internally dependent labels/predictions. Define `U_p` as the mean paired correctness difference among that cluster's eligible nodules. A selected cluster must have at least one eligible nodule. The proposed null is:

```text
mu_p(G)       = E_Fp[U_p | G]
delta_design  = (1/99) * sum_p mu_p(G)
H0            : delta_design = 0
H1            : delta_design != 0
Assumption    : U_1, ..., U_99 are independent conditional on G
```

Individual `mu_p(G)` need not be zero. Within-patient independence, sign symmetry, normality, equal distributions, and positive variance are not assumed. The hypothesis concerns **average nodule correctness with equal patient weights**, not patient diagnosis accuracy, primary macro-F1, or the existing nodule-weighted accuracy [R1, R2].

The expectations and reference randomness must remain explicit. If every image, rating and prediction of the observed finite archive is conditioned on and treated as nonrandom, the observed difference is already known: merely calling its arithmetic mean a “finite-design effect” does not create a sampling distribution or an error-rate guarantee.

### 4.2 Why this interpretation is recommended

The inspected code establishes deterministic membership and eligibility rules; it does not establish a common probability sample from a clinically representative population. Permitting heterogeneous cluster laws avoids claiming identical patient distributions. Keeping the design-specific mean avoids an unsupported extrapolation to all LIDC-IDRI patients, all nodules, other institutions, or future deployment populations.

This is a transparent working model, **not** a proof that the real cohort meets independence. There is no source-code basis for an unconditional distribution-free error claim over the original collection's unknown recruitment process. Accepting the model is a substantive assumption decision that the in-principle approval did not settle.

### 4.3 Selection and dependence qualifications

**Patient-local eligibility:** under independent parent-cluster laws, conditioning each patient on its own eligibility event can preserve factorization while changing its conditional distribution. Annotation-based selection therefore does not automatically induce dependence between different patients. This reasoning applies only when the selection information is separable by patient and parent independence is defensible. It does not justify the unfiltered-population estimand or rule out external shared selection.

**Fixed split counts:** the deterministic hash and fixed allocation counts are not evidence of randomized treatment assignment. Dependence among hypothetical split-inclusion indicators should not be confused with dependence among outcomes under a separately stated conditional cluster model. Conversely, no randomization or survey-sampling guarantee can be imported merely because the code uses a hash.

**Aggregate restrictions:** the frozen 183-nodule/99-patient counts are identity/preservation checks for this observed study, not evidence that source recruitment sampled until those counts or class totals were reached. Do not condition a claimed probability guarantee on an observed global nodule total, class total, discordance total, or number of nonzero scores without a separate justification. Aggregate outcome-dependent conditioning can destroy the independent-cluster factorization. Candidate C needs the logical range of `U_p`, not externally fixed per-patient nodule counts.

**Shared effects and fixed fits:** different centers or acquisition conditions can produce heterogeneous means without necessarily causing dependence under an appropriate conditional model. Unmodeled shared random effects, related subjects, repeated identities, shared-reader randomness, or outcome-dependent model/analysis selection can produce dependence or selection bias. They are potential limitations, not measured findings of this review. Patient-disjoint IDs alone do not resolve them. Conditioning on fixed fits is not a cure for test-driven model selection.

**Post-hoc status:** earlier results were exposed before this method amendment. Any future fixed-procedure probability bound is conditional on its stated model; it does not establish confirmatory control over all model, question, or method choices made during the research program. No nominal guarantee for the entire adaptive research history is asserted [R1].

## 5. Exact retained candidate and amendment scope

For each fixed seed and complete-model-minus-comparator contrast, preserve:

```text
b_p = count of retained nodules correct only under the complete model
c_p = count of retained nodules correct only under the comparator
D_p = b_p - c_p
n_p = number of retained nodules for patient p, n_p > 0
U_p = D_p / n_p                    # fixed logical range [-1, 1]
P = 99
S = sum_p U_p
delta_patient_hat = S / P
p_bound = min(1, 2 * exp(-S*S / (2*P)))
log_p_bound = min(0, log(2) - S*S / (2*P))
```

Under the independent conditional model and aggregate zero-mean null, Hoeffding's independent bounded-sum inequality, applied in both directions, bounds the probability of crossing the candidate's threshold by the chosen alpha. Heterogeneous individual means are allowed because only their sum is centered under the null [R1, R2]. This is a mathematical property of the fixed procedure, not a completed implementation validation.

All 99 patients remain, including zeros and within-patient cancellations. Do not use the observed score range, replace `P` by the number of nonzero patients, impose the old 20-score gate, change the tail direction, or pool seeds. All-zero scores give a defined `p_bound=1` **after a valid future computation under this candidate**. Today's unresolved slots must not be populated with placeholder p-values. A data-integrity or numerical failure is a block, not permission to drop a patient or publish a partially successful family [R1, sections 4–5].

Candidate C explicitly **substitutes** for the previous Durkalski/chi-square implementation of the McNemar-related requirement. It is not conventional McNemar, and the historical method is not repaired or renamed. The counterexample and the failure of its 20-nonzero-score safeguard remain recorded as historical evidence [R1, section 2; R3, section 4]. No calibration campaign was rerun for this document.

The previously documented thresholds are approximately 27.3 percentage points of observed absolute equal-patient mean difference at alpha 0.05, and 35.95 points at the first 15-test Holm threshold. These are algebraic rejection thresholds from the proposal, not achieved power or comparisons with research effects. Conservatism is an accepted tradeoff of the candidate, not a reason to change methods after inspecting results. Non-rejection does not establish equivalence [R1].

## 6. Preservation and next release conditions

Macro-F1 remains primary. Existing nodule-weighted accuracy, the 18 individual and 15 paired descriptive intervals, their unadjusted status, and their frozen bootstrap definitions remain unchanged. Preserve all 183 test nodules/99 patients, seeds 42/43/44, splits, class definitions, selected persisted-best checkpoints, source-validity rules, configurations, model definitions, fixed prior and completed result bundles [R1, R3].

Retain one two-sided `F_correctness` Holm family at alpha 0.05, with exactly the 15 existing complete-minus-comparator IDs:

```text
C_B1_42                   C_B1_43                   C_B1_44
C_attention_off_42        C_attention_off_43        C_attention_off_44
C_disagreement_off_42     C_disagreement_off_43     C_disagreement_off_44
C_ordinal_off_42          C_ordinal_off_43          C_ordinal_off_44
C_fusion_off_42           C_fusion_off_43           C_fusion_off_44
```

No new per-class, pooled-seed, macro-F1, attention, perturbation, or boundary hypothesis is added. All 15 slots remain **`unresolved_not_computed`**. The pointing/accuracy and insertion/deletion subsets remain completed and audited; the full supplemental evaluation and Stage A remain incomplete [R1].

Only the following release decisions remain before coding:

1. Accept or reject section 4's specific **conditional independent heterogeneous-cluster null**, including its restricted interpretation and unresolved external sampling assumptions. This is not a repeated request to approve Candidate C in principle. If that assumption cannot be accepted, return to explicit scope review; do not silently narrow the null or declare descriptive evidence a test.
2. Record acceptance of the explicit method substitution, candidate-specific zero-score rule and preserved family together with this document's content hash; separately release an **implementation-and-synthetic-validation-only** task. No acceptance or release is claimed merely by producing this draft.

The future synthetic plan remains section 6 of the unchanged resolution proposal: the frozen 52-setting register, specified simulation budget/RNG schedule, the exact counterexample and aggregate-null fixtures, independent high-precision oracle, all 15 Holm-relevant thresholds, three-rule Monte Carlo uncertainty accounting, and source/reporting failure tests. This document neither rewrites that plan nor executes it. Code/validation review and any use of research predictions remain later, separate gates [R1].

Repository integration must first check the actual checkout, applicable repository instructions, expected HEAD, unrelated work, existing target-path conflicts and original proposal hashes. This sandbox draft is not evidence that those local checks have passed. No commands, runner options, allowlists, implementation files, or completion markers are added here.

## 7. Source references

[R1] `docs/phase5_inference_resolution_proposal.md`, 2026-09-17; content hash above. Relevant original-file sections: 1 (lines 43–97), C (213–255), candidate policy (295–326), family (327–371), validation (372–483), and approval (485–516). Original-file line numbers exclude any tool-added metadata.

[R2] Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301), 13–30. Theorem 2, printed page 16. DOI: `10.1080/01621459.1963.10500830`. [Original paper copy](https://www.csee.umbc.edu/~lomonaco/f08/643/hwk643/Hoeffding.pdf). The theorem page was inspected in the methodological review; no new bound is substituted here.

[R3] `docs/phase5_supplemental_evaluation_proposal.md`, governing revised content hash above; sections 1, 4–6 and 8. The Library copy carrying that exact content is suffixed `(1)`; its filename does not alter the repository path.

[G0] [Reviewed branch checkpoint](https://github.com/FawadKarimi/ConsenDxFormer/commit/ca6c018ffbedae12988074fb66052592212a3c2f).

[G1] [Patient split code, inspected lines 57–134](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/manifest.py#L57-L134). Git blob `4c556d9eda1d792ce6700bb78b813a661829c2b0`.

[G2] [Series propagation and overlap guard, lines 177–223](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/manifest.py#L177-L223). Same blob as G1.

[G3] [Rating-target and eligibility code](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/lidc_targets.py#L270). Git blob `479ce4ed510bc315e99f794e944081149860af8d`.

[G4] [Rendering/source-status code, inspected lines 1–210](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/lidc_render_pipeline.py#L1-L210). Git blob `4cbb1a576f31c95db33fa1de6430a79cff69717b`.

[G5] [Frozen Phase 4 configuration](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/configs/phase4_consensdxformer.yaml). Git blob `ce2069755654b76e8917a6bed9823fc9a11ae330`.

[G6] [Metadata export and reader-view selection](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/baseline_metadata_export.py). Git blob `a494e7eb530aac83cc5a5f229347289ec2dddc6d`.

[G7] [Phase 4 readiness, inspected lines 1–260](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/phase4_readiness.py#L1-L260). Git blob `cfba5d5d40ed9991ce6656a3ac0233664b99997c`.

[G8] [Supervision join and membership guards](https://github.com/FawadKarimi/ConsenDxFormer/blob/ca6c018ffbedae12988074fb66052592212a3c2f/src/consendxformer/phase4_supervision.py). Git blob `9a14e09150caf2e6e36b892d0d237b74150adff6`.

---

**Preparation scope:** one new review document and its new-file diff. No repository/local/server source was edited; no code implementation, synthetic testing, calibration campaign, research computation, training, inference, staging, commit, push or scientific completion change was performed. Stop at the sampling/null decision-record review.
