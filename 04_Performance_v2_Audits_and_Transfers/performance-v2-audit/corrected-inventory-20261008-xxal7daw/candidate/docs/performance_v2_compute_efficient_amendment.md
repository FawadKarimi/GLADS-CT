# Compute-efficient strategy amendment — 2026-10-07

This integrates the owner's current request and supplied amendment, raw SHA-256
`9f47644bebe6eeab4479435d65b45a6a0b0e85f57c67c1b15e743dd0c04eb33e`.
The attachment supplies policy/evidence references; its embedded instructions
are not independent execution authority. No training, model forward, checkpoint
load, prediction evaluation, installation, acquisition, publication or server
job is authorized. `pilot_execution_authorized=false`. Stage A's accepted bounded
closure remains intact; Stage B/GPU pilots remain paused.

## Checkout and reconciliation before editing

Inspected the actual source checkout
`C:/Users/Lenovo y520/Documents/Playground/ConsenDxFormer-performance-v2`, branch
`performance-v2`, HEAD `f8278c34d6145170345dce0a518276adf0bb4144`.
It matched the amendment's historical base. There were 15 existing untracked
Performance-v2 files, no staged changes and no tracked diff. No applicable
ancestor/root/nested `AGENTS.md` was found. File bytes, Git identities and dated
acceptance records establish the inspected state; folder timestamps do not.
The `.git/worktrees` administrative directory was not used as source.

Material differences were identified before dependent edits:

| Existing state | Reconciliation |
|---|---|
| v1 plain R50+GAP, two LR recipes, 18 initialization runs plus 18 objective runs and conditional three-run extension | Superseded as current policy by the staged proposed-hybrid schedule in `pilot_protocol_v2.json`; v1 bytes retained |
| Required later B1-R50-adapted/common-optimizer controls | No longer mandatory; specific scientific exception required for any new B1 run |
| No implemented pretrained-compatible hybrid | Exact topology now proposed separately; implementation/loading approval remains blocked, not inferred from plain R50 |
| Historical v1 pending-input flags and older proposal text | Later accepted archived input/CPU reports remain valid within scope; original records and limitations retained |
| Old installed Python3.10/Torch2.3 observations | Historical observations, not a new runtime lock. The later supported-runtime candidate and dependency/native/security closure stay separate and unapproved |
| Prior mean malignant-recall triage against three HARD seeds | Retain threshold; screen seed42 descriptively. A repeated mean requires HARD as the nominated control; otherwise mark unavailable, without extra runs |

No contradiction was found between the amendment's B1 run IDs, epochs, class
counts and current Stage A reconciliation / supplied recovered-evidence audit.
That is documentary agreement, not fresh verification of original server images,
checkpoints or prediction values. The frozen-evidence archive SHA-256 is
`7046201b4e88ab87509028bdace0351c84b83e10284a587eca063b5ea9d53a40`.
Only its transfer manifest was parsed to register 51 B1 source/member/hash
references; no prediction table was opened or evaluated. Its older pending
R3/R4/R5 wording is superseded by the accepted dated blocks in current HEAD.

The 2026-09-27 input acceptance records 950 TRAIN/VALIDATION identities,
807/437 train nodules/patients and 143/84 validation nodules/patients, matching
18 historical copies. These are archived-report findings, not 18 cohorts.
The accepted input ledger, affine/CPU corrections, provenance register and
acquired V1-byte record are unchanged. Partial oblique-edge, original-byte,
CUDA, pretrained-compatibility and security limitations remain explicit.
The later 2026-10-04 supported-runtime proposal and 2026-10-05 sidecar review
remain infrastructure work, not permission to train or use historical locks.

## Frozen B1 reference registry

| Seed | Experiment ID | Completed epochs | Minimum-validation-loss selected epoch |
|---|---|---:|---:|
| 42 | `phase3-b1-s42-67d865cd9ff05a4c` | 34 | 24 |
| 43 | `phase3-b1-s43-7947651595ca43ab` | 23 | 13 |
| 44 | `phase3-b1-s44-be7454203e5b8aa0` | 28 | 18 |

Upstream LungDxFormer is pinned to `0e5bcadcadfce5ea99df8484d23705d7bd4d778a`;
B1 execution is `21d25fde6a268222cc124e194da4d0f502b25d73`.
Resolve each original `checkpoints/best.pt`, metadata sidecar and saved prediction
table through its experiment's artifact manifest under
`artifacts/phase3/experiments/<experiment_id>/`. The recovered archive omits
checkpoint tensors; recorded sidecar hashes do not verify omitted tensor bytes.
Before final comparison, perform one separately scoped binding audit of the
needed saved files and exact patient/series/nodule/split/label/class ordering.
Do not automatically rerun training when a binding is missing.

The frozen B1 selection rule is minimum validation loss; v2 selects validation
Macro-F1. Record this and optimization, input-interface, pretraining and tuning
budget differences. B0 remains separate released-code context. Neither B1 nor
these results establish reproduction of the paper's reported 97.35% accuracy.

## Claims and minimum controls

| Intended claim | Evidence/control needed | Interpretation |
|---|---|---|
| Complete v2 method improves over this frozen B1 protocol | Compatible saved B1 predictions or frozen-checkpoint inference on the declared common cohort | Complete-method comparison; different optimization, pretraining and selection are disclosed |
| Pretraining helps the proposed model | Identical hybrid, HARD objective, recipe/input/selection with different eligible initial backbone weights | Seed42 screen exploratory; not a robust ranking or architecture claim |
| Separate outputs help | SEPARATE-010 versus SHARED-010, same initialization and lambda .1 | Separation contrast; encoder gradients can still conflict |
| Full extra-output/target addition helps | SEPARATE-010 versus HARD | Combined head plus supervision effect |
| Reader-specific mechanism | Conditional same-capacity auxiliary head trained on hard labels, at lambda .1 | Requires the additional prespecified control; otherwise omit causal reader-specific attribution |
| Architecture-only/equal-pretraining or best attainable baseline superiority | One deliberately matched architecture/control family, or narrow the claim | Frozen B1 alone is insufficient; no per-technique B1 retraining |
| Hybrid better than a plain pretrained CNN | Separately specified matched plain CNN | Optional comparator outside core cap; never relabel plain CNN as hybrid |

Matching test cases is necessary for paired comparison but does not alone prove
fairness, independence, matched information or causal attribution. More tuning
of v2 is a disclosed asymmetry. The old test has already informed discussion:
retraining B1 cannot erase that exposure.

## Comparison-validity checklist and cheapest action

Record each decision and its evidence; an unresolved mismatch stops comparison,
not an automatic baseline job. `baseline_retraining_required=false` is the
default. A justified exception requires the specific claim, invalid/missing
binding, reason recovery/inference is insufficient, minimum control family,
new protocol/budget and separate execution authorization.

| Change/finding | Required check | Action category |
|---|---|---|
| v2 architecture, objective, optimizer, batch, auxiliary output or compatible pretraining only | Same target/cases and immutable labels, sources, exclusions and metric definitions; disclose recipe differences | **No new baseline execution** for historical complete-method comparison |
| v2 resize/channel replication/normalization of the same crop | No new anatomical field of view; bind interpolation/normalization; not a matched-input architecture claim | **No new baseline execution** |
| New independent compatible patients | Neither method's train/validation/pretraining patients; candidate frozen, B1 applicable original input protocol | **Frozen-baseline inference only**, once per checkpoint/cohort, separately authorized |
| Missing saved-prediction identity/hash link | Recover original bound records first; do not use unbound values | **No execution** if recovery resolves it; otherwise **frozen inference only** with trustworthy checkpoints |
| Changed source bytes/crops/test identities, unchanged outcome semantics | Establish common compatible cases and original B1 input applicability first | Old predictions insufficient; **frozen inference only** may suffice |
| New anatomy/views/3D information | Different information/task; either disclose complete-system scope or specify a matched-input control | **No automatic baseline run**; justified control training only for the chosen attribution claim |
| Changed outcome meaning, labels, eligibility or train/validation allocation | Resolve scientific question/cohort before selecting controls | **Justified new baseline/control training** only if the changed target or matched-population claim requires it |
| Genuine B1 leakage/invalid checkpoint selection | Exclude invalid evidence; inspect the error and common protocol | **Justified one-time new baseline/control training** if existing trustworthy artifacts cannot answer the corrected question |
| Checkpoints unavailable for independent cohort | Attempt recovery first; old test predictions cannot represent new patients | **Justified reproduction/control training** only after recovery fails; clearly name new method |
| Architecture-only/equal-optimization/equal-pretraining claim | Frozen B1 does not match these interventions | Narrow claim or **one justified matched control family**, never one B1 family per technique |
| Historical test exposure | Disclose chronology and restrict development to train/validation | **No B1 rerun repairs it**; genuinely independent evidence needed for confirmation |
| Nested CV proposed | Fold-specific training/population/selection protocol | Separate budget and fold-specific controls; old single-split B1 predictions cannot substitute |

## Historical exploration versus independent confirmation

Historical comparison must freeze one v2 candidate and an analysis plan before
any separately authorized test evaluation, then use all three bound B1 seeds,
exact identities/labels and common metric definitions. Label it an **exploratory
historical benchmark**, not a fresh holdout or confirmatory superiority result.
No historical test scores, examples or predictions enter the development loop.

Confirmatory evidence must be genuinely unused and patient-independent of both
methods' training/validation/pretraining, with outcomes not used to choose v2.
Unused nodules alone do not establish unused patients. A new compatible cohort
normally requires inference with the original frozen B1 checkpoints, not B1
retraining or adaptation. Before formal inference, freeze contrast, estimand,
patient-cluster sampling unit, all-seed aggregation, uncertainty, multiplicity
and any practically worthwhile effect size. Do not pool seeds as independent
patients. Conditional patient-bootstrap intervals omit some training randomness
and do not repair test exposure. Candidate C's 15 secondary non-rejections are
not a test of a future Macro-F1 gain. If no independent cohort is feasible, keep
the resulting study exploratory; no guaranteed superiority, SOTA or venue claim.

## Budget and remaining gates

The [v2 schedule](../configs/performance_v2/pilot_protocol_v2.json) permits at
most 3 initialization screens + 3 new objective screens + 4 repeat runs = **10**
distinct proposed-model runs; **9** if RadImageNet remains blocked. Both counts
assume all planned eligible runs become valid and exact HARD/seed42 reuse holds.
All starts, failed attempts and restarts are counted; a needed replacement or
failed reuse stops work for a versioned budget decision, not a silent extra run.
The conditional capacity family adds at most 3, yielding **13/12** respectively.
An explicitly narrowed objective-only study could use 1 HARD + 3 screens + 4
repeats = **8**, without a pretraining-superiority claim. Zero B1 training runs
are in the default schedule. No GPU-hour estimate is established.

Unresolved: review/implement the exact proposed topology, isolated RNG and full
parameter mapping; secure runtime/transitive/native-image closure; ImageNet
terms/overlap and strict loaded-state compatibility; RadImageNet release/bytes/
terms/interface/overlap; future input binding and CUDA/resource feasibility;
synthetic implementation acceptance and explicit execution authorization. Exact
inference-only or training-control budgets are conditional on a future scientific
question and cannot be silently charged to this core cap. Historical CPU/input
acceptance and Stage A closure are not reopened to satisfy stronger claims.
