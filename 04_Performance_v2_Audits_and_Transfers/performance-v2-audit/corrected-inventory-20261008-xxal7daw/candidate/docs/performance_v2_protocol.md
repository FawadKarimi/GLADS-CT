# Performance-v2 protocol v2 — compute-efficient specification, not training release

Originally recorded 2026-09-27; amended 2026-10-07 by the current owner request
and [compute-efficient amendment](performance_v2_compute_efficient_amendment.md).
Original brief SHA-256:
`05cd1ae1bb19dc2c2991e66b8a26b54984b024fc8c993db4a91e9d2676ce059b`.
Current amendment SHA-256:
`9f47644bebe6eeab4479435d65b45a6a0b0e85f57c67c1b15e743dd0c04eb33e`.
Development remains in the actual `performance-v2` source checkout at Stage A
base `f8278c34d6145170345dce0a518276adf0bb4144`. Stage A stays closed within
accepted R1–R5 bounds; historical code, configs, results and markers stay intact.
Stage B/GPU pilots remain paused; `pilot_execution_authorized=false`.

The current non-launching specification is
[pilot_protocol_v2.json](../configs/performance_v2/pilot_protocol_v2.json).
[pilot_protocol_v1.json](../configs/performance_v2/pilot_protocol_v1.json) is
unchanged historical evidence; its plain-CNN matrix is superseded as current
policy. The [provenance register](../configs/performance_v2/pretraining_provenance_v1.json)
and later [input/acquisition acceptance](performance_v2_input_acceptance_and_acquisition.md)
are preserved. This amendment authorizes no repeated audit, model/checkpoint
load, forward, training, test evaluation, statistical calculation or acquisition.
No server launcher is supplied.

## Questions and frozen implementation findings

- H1: separate hard-diagnostic and reader-distribution outputs may reduce target
  conflict. The shared encoder can still receive competing gradients; a hard
  label term does not prove improvement.
- H2: pretraining may help this small cohort. Compare eligible initializations
  within the exact same proposed hybrid, not RANDOM on the old custom CNN
  against a different pretrained architecture.
- H3: validation total-loss selection differs from diagnostic Macro-F1 selection.
  Historical results reflect the complete training/selection procedure, including
  auxiliary scales, rather than isolating one component's gradients.
- H4: frozen `phase4_execution.py` passes `model.dropout` into attention/MLP/
  classifier paths in `phase4_model.py`; its separate `attention_dropout` setting
  is not passed by that constructor path. This remains a static finding, not a
  causal explanation. The new topology declares zero dropout explicitly.

The frozen `diagnostic_disagreement_loss` uses
`t_ic=(1-u_i)*1[c=y_i]+u_i*q_ic` and

```text
L_legacy = -sum_i,c w_c*t_ic*log_softmax(logits_d)_ic
           / max(sum_i,c w_c*t_ic, 1e-8).
L_h = sum_i w[y_i]*(-log_softmax(logits_d)[i,y_i]) / sum_i w[y_i]
L_r = mean_i[-sum_c q[i,c]*log_softmax(logits_r)[i,c]]
```

LEGACY is a batch numerator divided by batch target mass, not a mean of
individually normalized per-sample losses. Without weights w=1. Keep that
normalization exactly. Hard weights are train-only `N_train/(3*n_train,c)`;
reader CE is unweighted, with stable log-softmax. Validate finite q in [0,1],
sum tolerance atol1e-6/rtol0, u in [0,1]; no reconstruction from predictions,
silent renormalization or undocumented probability clipping. The [1,3,5]
reader-score example has an indeterminate hard label and uniform distribution;
it illustrates possible target conflict, not its prevalence or measured gradients.

## Frozen data, metrics and reference

Retain 807 train nodules/437 patients and 143 validation nodules/84 patients.
These counts are accepted within the later archived input-report scope, not a
new image audit. Retain exact patient/series/nodule/split identities, labels,
eligibility, exclusions, source bytes and three-class order benign/indeterminate/
malignant. Investigate discrepancies without filtering. Reader targets and masks
are supervision only, never forward inputs. No new folds or TEST loader.

Macro-F1 remains primary. Prediction is argmax in fixed class order, first class
on exact ties. Average all three class F1 values including absent classes; zero
denominators yield zero. Accuracy equals Micro-F1 for complete single-label
three-class predictions and is one secondary endpoint. Preserve per-class recall/
F1 and malignant trade-offs. The metric definitions in `baseline_evaluation.py`
and `phase4_evaluation_metrics.py` remain unchanged.

The 183-nodule/99-patient historical test set was already inspected. No test
predictions, scores or examples enter development/debugging/selection. Freeze
one v2 candidate and an analysis plan before any later separately authorized
test evaluation; label results an exploratory historical benchmark. Repeated
validation selection is also a source of optimism.

Existing LungDxFormer/B1 seeds42/43/44 are the frozen reference; exact experiments
and selection epochs are registered in the [amendment](performance_v2_compute_efficient_amendment.md).
B0 remains separate context. No per-technique or automatic final B1 training is
required: `baseline_retraining_required=false`. Missing/mismatched evidence
triggers a documented scientific exception, not a job. B1 selected minimum
validation loss; v2 selects validation Macro-F1. Disclose this and pretraining,
optimization, input-interface and tuning-budget differences. A complete-method
comparison does not isolate architecture advantage, best attainable B1 performance
or reproduction of the paper's reported accuracy. Matching cases alone is not
proof of fairness, equal information or causal attribution.

## Exact proposed model and common recipe

The [topology specification](performance_v2_model_topology_v2.md) defines
`performance_v2_r50_hybrid_v2`: R50 V1.5 layer2/3/4 features at strides8/16/32,
projections into a 128-dimensional 49-token Transformer (two blocks/four heads),
native spatial attention and coarse-to-fine gated fusion, then pooled 128-dim
features and linear 128-to-3 outputs. The topology is proposed, not implemented
or approved. The existing custom CNN cannot accept arbitrary ResNet50 weights.
A plain R50+GAP control is optional and must never be renamed as this hybrid.

| Setting | Preserved/proposed fixed policy |
|---|---|
| Input | Same frozen rendered-64 grayscale ROI bytes; no new window/crop/view/FOV. Decode once using float32 and divide by255 only if max>1; CPU bilinear resize to224, align_corners=false, antialias=true |
| Train transform | Independent h/v flips p=.5; uniform angle [-10,10], independent translations [-.08,.08] of each axis; tensor bilinear affine, center[112,112], scale1, shear[0,0], clockwise angle, post-rotation [horizontal,vertical] translation; int(round(fraction*axis_length)), ties-to-even on float product; fill0 before normalization; clip[0,1], replicate, normalize |
| Validation transform | Decode/resize/replicate/normalize only; no TTA. Antialias applies to resize, not affine |
| Normalization | mean[.485,.456,.406], std[.229,.224,.225]; common proposed interface, not validated for RadImageNet |
| Exclusions | No jitter/noise, random crop/scale, mixup/cutmix, new anatomy or re-rendering; 64→224 interpolation adds no CT information |
| Loader / BN | batch16, train shuffle, drop_last=false, workers0, no weighted sampler/accumulation; all BN trainable and update during train, eval during validation |
| Optimizer | Corrected four AdamW groups; betas(.9,.999), eps1e-8; recipe A encoder LR1e-4/head LR1e-3; Conv/Linear weight decay1e-4, biases/norm affine decay0; recipe B deferred |
| Schedule | Before first update each epoch: e1–5 `.1+.9*(e-1)/4`; e6–100 `.01+.99*(1+cos(pi*(e-6)/94))/2`; no plateau scheduler |
| Numerics | float32, AMP off, clip norm1.0, smoothing0; no EMA/ensembles; validation diagnostic CE unweighted |
| Stop | max100 epochs, no stop before20, patience15 without raw validation Macro-F1 improvement >1e-4; counter separate from checkpoint ranking |
| Checkpoint | Global maximum diagnostic validation Macro-F1 candidate set within1e-12; minimum unweighted diagnostic validation CE, then earliest epoch; never total auxiliary loss |

Keep four encoder/head × weights/bias-normalization groups, complete and disjoint
by parameter identity. All shared hybrid modules use encoder LR. The topology
explicitly assigns MHA projection weights and the positional-table zero-decay
exception; reject unknown parameters rather than falling back. There is no new
optimizer implementation in this amendment. Preserve the v1 RNG derivation and
encoder/diagnostic/auxiliary/shuffle/augmentation streams; add only the isolated
shared-fusion stream for new modules. Arm/recipe names stay out of common streams.
Synthetic hashes must prove an extra head cannot change common initialization,
sample order or augmentations. Log effective configs, inputs, groups, state/RNG
hashes, actual LR, loss numerators/denominators, class metrics and actual time.

The later CPU imports/bridge and hash-bound transform reports are accepted only
within their recorded synthetic scope. Archived input acceptance covers 950
decoded TRAIN/VALIDATION arrays, not fresh original-byte verification here.
Partially supported oblique edges, CUDA determinism and pretrained compatibility
remain unresolved. Old installed-version observations are not new runtime pins:
later supported-runtime/security/dependency work remains separate and unapproved.

## Phase 1 — initialization screening

Run HARD-only at seed42 and recipe A on this same hybrid for RANDOM, explicit
IMAGENET1K_V1 (never DEFAULT) and eligible official RADIMAGENET. Maximum three
screens, two if RadImageNet remains blocked. No automatic second LR recipe.
Set lambda_o=lambda_a=0; Transformer/DSA/fusion remain present. A provenance-blocked
arm is ineligible, not a zero score. A failed/incomplete eligible arm cannot be
ranked or promoted from partial results.

For complete eligible screens, form the global maximum selected-checkpoint
validation Macro-F1 candidate set within1e-4, then minimum unweighted diagnostic
CE, then fixed order RANDOM, IMAGENET, RADIMAGENET. Record the raw maximum and
tolerance-selected winner separately, not a non-transitive pairwise sort. This
seed42 screen is exploratory, not a robust pretraining ranking. Reinitialize later
runs from original source weights, never the winning trained checkpoint.
If all eligible arms are invalid/underconverged, stop and document a common
versioned amendment. Do not extend one arm alone or silently retry.

## Phase 2 — objective and separate-output screening

After a separate release, lock the selected initialization and recipe A. Reuse
the selected Phase1 HARD seed42 run only if its complete execution/input contract
matches. Add exactly these three seed42 screens:

| Recipe | Objective | Contrast |
|---|---|---|
| LEGACY | Exact frozen batch-mass diagnostic loss | Context for target conflict versus HARD |
| SHARED-010 | L_h + .1*unweighted CE(q,p_d) | Soft-supervision effect versus HARD |
| SEPARATE-010 | L_h + .1*L_r on separate output | Separation versus SHARED-010; combined head/target addition versus HARD |

The supporting output is an independent linear128→3 head, LR1e-3 with the same
schedule/decay/initialization distribution; discard it at inference. No reader
labels/masks enter diagnosis. The shared encoder can still receive conflicting
gradients. The .3 arms are deferred. lambda_o=lambda_a=0 throughout the core;
the long-term objective remains L_h+lambda_c*L_reader+lambda_o*L_ordinal+
lambda_a*L_attention, with restoration separately versioned.

Exact reuse requires all source/effective-config/topology/input/label/order/
weight/loaded-state/head/optimizer/schedule/RNG/transform/runtime/precision/
determinism/stopping/selection bindings and a valid completion manifest. The
v2 JSON lists every required equality and only administrative role aliases may
differ. A matching seed/name alone is insufficient. v1 plain-CNN runs cannot be
reused; failed reuse stops for a new budget decision, not an extra automatic fit.

## Phase 3 — limited repetition

Nominate one completed candidate and one distinct closest internal control using
validation only; declare the primary claim before repeats. Separate-output
attribution uses SHARED-010; the whole extra-output addition uses HARD. Other
questions require an explicit relevant control before repeats. Add seeds43/44
for both: four new runs; reuse each seed42 execution only with exact contracts.

Freeze final selection across these two complete three-seed families: arithmetic
mean selected-checkpoint validation Macro-F1; global maximum candidate set
within1e-4; minimum mean unweighted diagnostic CE; fixed order HARD, LEGACY,
SHARED-010, SEPARATE-010. Require all three valid seeds for both; no survivor,
best-seed or accuracy selection. Report both, every seed-wise sign, variability
and actual time. A reversal or uncertainty is not a reason to expand until success.
Retain the >5 percentage-point mean validation malignant-recall decrease flag
versus matched HARD and all failed/nonfinite runs; it is an engineering flag,
not a clinical margin or automatic replacement of the nominated winner. At
screening use seed42 descriptively. The three-seed mean flag is estimable only
if HARD is the nominated repeated control; otherwise record it as unavailable
under this budget. Do not add HARD runs merely to fill that flag.

Core cap: **3+3+4=10**, or **2+3+4=9** with RadImageNet blocked, assuming exact
reuse and successful eligible runs. Count every candidate/start/failure/restart;
no automatic replacement, retry or expansion. A reader-specific mechanism claim
conditionally adds at most three CAPACITY-010 runs with the same auxiliary head
trained on hard labels using unweighted CE at lambda .1: totals13/12. This is
optional and prespecified, not mandatory upon SEPARATE nomination or independent
confirmation. Without it, limit interpretation to the full objective addition.
An explicitly narrowed objective-only study can use eight runs and makes no
tested pretraining-superiority claim. New matched controls, ordinal/attention
restoration, URL/MOW-Net and other modules lie outside the core cap. No verified
GPU-hour estimate exists. One GPU requires separate release; batch16 OOM stops
the common protocol without per-arm AMP/batch/CPU fallback.

## Pretraining, comparison validity and release

Preserve official-source/release/terms, raw weight and source hashes, acquisition
date, strict semantic topology/key/shape/dtype/layout mapping, loaded-state
hashes, transforms and all-split contamination checks. ImageNet V1 acquisition
is already recorded; it is neither repeated nor equated with loading approval.
ImageNet is general-image pretraining. RadImageNet remains blocked for its own
bytes/terms/member mapping, topology, overlap and numerical interface. Its saved
example normalization differs; do not silently convert, use a mirror, or call a
checkpoint-plus-preprocessing comparison initialization-only. Unknown patient
membership is not proof of no leakage. Inductive local SSL must exclude all
validation/test patients; transductive work requires a separately named design.
BiomedCLIP, local SSL, URL/MOW-Net remain deferred.

Use the amendment's [comparison-validity checklist](performance_v2_compute_efficient_amendment.md)
to distinguish **no baseline execution**, **frozen-baseline inference only** and
**justified new baseline/control training**. A new compatible independent cohort
normally needs original frozen B1 checkpoint inference under B1's input protocol,
not retraining. Recover missing bindings/weights first. Architecture-only or
equal-pretraining/optimization claims need a deliberately matched control or
narrower wording; no upgraded-B1 family is mandatory by default.

Historical exploration and independent confirmation are distinct. Confirmatory
evidence must be genuinely unused and patient-independent of both methods'
training/validation/pretraining, with outcomes not used to design v2. Unused
nodules alone do not establish unused patients; retraining B1 cannot erase test
exposure. Before formal inference, freeze contrast, estimand, patient-cluster
sampling unit, all-seed aggregation, uncertainty/multiplicity and any claimed
practically worthwhile effect. Do not pool seed predictions as independent cases.
Patient-bootstrap intervals conditional on fitted models omit some training
randomness and do not repair exposure. Candidate C's secondary non-rejections
are not a test of future Macro-F1 improvement. No SOTA/venue/superiority promise.

Next: review the topology/config/proposal, then implement and synthetically
verify the hybrid, exact RNG and optimizer mapping. Resolve secure runtime/
transitive/native-image closure, provenance/loading/overlap, future input binding
and resource/determinism gates before any new explicit GPU release. Historical
CPU/input acceptance and Stage A closure are not reopened. Independent evaluation
and Stage B remain separate, unreleased work.
