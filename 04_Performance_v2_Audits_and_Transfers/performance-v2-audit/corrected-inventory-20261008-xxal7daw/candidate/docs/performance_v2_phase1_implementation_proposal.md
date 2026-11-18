# Performance-v2 implementation proposal — compute-efficient v2

Current strategy recorded 2026-10-07, superseding the earlier plain-CNN matrix
and compulsory upgraded-B1 controls. This is an implementation proposal, not an
implementation or launch approval. Use [pilot_protocol_v2.json](../configs/performance_v2/pilot_protocol_v2.json),
[the current protocol](performance_v2_protocol.md),
[the exact proposed topology](performance_v2_model_topology_v2.md) and
[the comparison-validity amendment](performance_v2_compute_efficient_amendment.md).
Historical v1 configuration and provenance/input/security records stay intact.
Stage A is preserved; Stage B/GPU paused; `pilot_execution_authorized=false`.

## Proposed component boundary

Implementation must be additive in a new Performance-v2 namespace. Do not change
Stage A modules, result bundles or checkpoints to implement this hybrid. The
existing custom CNN cannot accept arbitrary ResNet50 weights. Plain ResNet50
results, if ever produced, remain separate comparator/surrogate evidence.

1. **Source adapter:** frozen eligible TRAIN/VALIDATION identities only. Reuse
   existing manifest/identity/split/label/hash validation; no TEST loader/path or
   test-derived diagnostics. Export a content-addressed cohort ledger, including
   exact order, labels and source-image/supervision bindings before construction.
   Preserve 807/143 nodules, 437/84 patients and fixed class/eligibility rules;
   discrepancies stop review without filtering or regenerating splits.
2. **Input adapter:** decode once using the frozen float32 grayscale rule,
   finite/rank/shape/range checks; same 64 ROI→224 CPU bilinear interface and
   explicit affine center/rounding/order from v1 and accepted later reports.
   Clip, replicate and normalize; validation has no augmentation. Never add
   anatomy, channels from annotations, masks or reader targets as forward inputs.
3. **Model adapter:** implement `performance_v2_r50_hybrid_v2` exactly: R50 V1.5
   layer2/3/4 outputs, audited 128-dim projections, 49-token two-block/four-head
   Transformer, spatial attention, coarse-to-fine gated fusion, 128-dim pool and
   linear128→3 diagnostic head. Only SEPARATE/CAPACITY add the independent
   supporting output. No ordinal head, ordinal/alignment loss or dropout in the
   core. All initialization comparisons use this topology. Source weights
   require the existing restricted-load/security process, full semantic mapping,
   exact source-state coverage and state hashes; no missing-key bypass.
4. **Training/validation core:** consume only a reviewed immutable effective
   configuration and validated adapters. Weighted hard TRAIN CE, exact LEGACY
   batch target-mass reduction, and unweighted reader CE remain unchanged.
   Validate q/u; no silent normalization. Select diagnostic validation Macro-F1
   with the existing checkpoint/CE/earliest rules. Float32, AMP off, clip1.0,
   deterministic settings, trainable BN and the common batch policy remain.

## Corrected optimizer and initialization contract

| Group | LR under recipe A | Weight decay |
|---|---:|---:|
| Encoder Conv/Linear weights, including MHA projection matrices | 1e-4 | 1e-4 |
| Encoder biases/norm affine; explicit named positional-table exception | 1e-4 | 0 |
| Diagnostic/reader head weights | 1e-3 | 1e-4 |
| Diagnostic/reader head biases/norm affine | 1e-3 | 0 |

The first two groups include backbone, Transformer, token/position, attention
and fusion. Recipe B is deferred; do not reinstate it as an automatic fallback.
Retain AdamW betas(.9,.999), eps1e-8 and the exact epoch schedule. Every trainable
parameter must appear exactly once by identity and name; reject omissions,
overlap or unknown tensors. Export names/shapes/LR/decay. MHA `in_proj_weight`
has an explicit decayed mapping; `position.position` has an explicit zero-decay
mapping. No broad default or changed numerical optimizer policy is implied.

Retain the v1 SHA-256 stream derivation for encoder, diagnostic head, auxiliary
head, shuffle and augmentation. Add isolated `shared_fusion_initialization`
for new shared modules in the topology's declared order. Common streams exclude
arm/recipe names. The diagnostic/reader heads are always new and independently
seeded; no 1000-class source head is reused. Exact common initialization,
sample-order and augmentation hashes must match where required; extra-head
construction cannot consume those RNG streams. Initializations differ only in
approved backbone values, never topology or a hidden alternative interface.
Every later fit starts from original weights, not a trained screen checkpoint.

## Bounded non-launching schedule and reuse

- Phase1: HARD at seed42, recipe A, eligible RANDOM/IMAGENET/RADIMAGENET; at
  most three runs, two when RadImageNet remains blocked. A failed eligible arm
  prevents promotion; provenance-blocked arms remain ineligible. No automatic
  alternative recipe. If all eligible arms are underconverged/invalid, stop for
  a common documented amendment.
- Phase2: selected initialization/A at seed42; reuse the exact Phase1 HARD
  contract, then add LEGACY, SHARED-010 and SEPARATE-010 (three new runs).
  Lambda .1 was nominated before training; .3 is deferred. Keep lambda_o/a zero.
- Phase3: nominate one candidate and distinct closest internal control through
  validation only; declare claim before repeats. SHARED-010 controls separation,
  HARD controls the whole head/target addition. Add43/44 for both (four runs),
  reusing seed42 only after exact binding validation. Other claims need an
  explicit closest control first.
- Optional CAPACITY-010: same supporting head trained with unweighted hard-label
  CE at .1, at most three seeds; required only for a reader-specific mechanism
  claim, not automatically because SEPARATE was nominated.

Core10/9, conditional maximum13/12, zero default new B1 training. All starts,
failures and restarts count. No retry, replacement or expansion is implicit.
If exact reuse fails, stop and version the budget. A narrowed objective-only
study could use eight runs after recording its changed hypothesis, without a
pretraining-superiority claim. No GPU-hour estimate exists.

Reuse compares all source/effective-config/model/initialization/input/label/order/
transform/optimizer/loss/RNG/runtime/precision/determinism/stopping/selection
bindings listed in v2 JSON, plus valid completion/output manifest. Exclude only
the declared administrative phase/review/role aliases. One physical run may have
two analysis roles, never two independent-seed counts. Same name/seed is
insufficient; no v1 plain-CNN reuse. No scheduler/runner enforcing these policies
exists yet; the configuration is a specification, not enforcement evidence.

## Selection, outputs and future synthetic acceptance

Preserve max100 epochs, min-stop20, patience15 with raw Macro-F1 improvement
strictly >1e-4. Checkpoint candidates lie within1e-12 of the global maximum;
choose minimum unweighted diagnostic validation CE then earliest epoch. Patience
state stays separate. Seed42 screening uses a global maximum candidate set
within1e-4, CE tie, fixed arm/objective order. Final nomination uses complete
three-seed means for both repeat families, tolerance1e-4, mean CE and fixed order
HARD, LEGACY, SHARED-010, SEPARATE-010. Record maximum and selected winner.
No test/accuracy/best-seed selection, unrankable incomplete family or favorable
restart. Keep every seed sign, class metric, malignant-recall flag and actual time.

Every run requires an exclusive directory and immutable identity including
protocol/model/initialization/objective/recipe/seed and input bindings. Export
effective config/environment/source/cohort/RNG snapshots, source-to-loaded-state
mapping, exact optimizer groups and LR, per-epoch components and validation
metrics, actual epochs/time, best/last identity/state hashes, and completion only
after complete manifest validation. Future authorized reload verification is a
release requirement, not permission to load a checkpoint during this amendment.

Once implementation exists and synthetic execution is separately scoped, verify:

1. Finite phased schedule, blocked arms, exact reuse, no automatic B1 launch,
   no retry/budget growth, required scientific exception before any new control.
2. Source/identity/label/order/hash validation, patient disjointness, TEST denial,
   no-overwrite and partial-evidence retention through corruption/interruption.
3. Exact synthetic transform oracles, including preserved affine/rounding bounds;
   retain the accepted partial-oblique-edge limitation until explicitly resolved.
4. Exact topology/feature dimensions, strict source mappings and rejection of
   merely shape-compatible/extra/missing/transposed/unsafe-layout keys; no real
   weights needed for invented fixtures. Do not claim fakes validate real bytes.
5. All RNG streams and common-state/order/augmentation invariance across heads;
   complete optimizer partition including MHA/position and exact schedule.
6. Exact hard/reader/LEGACY reductions, metric class/zero-denominator conventions,
   checkpoint/patience/aggregate tie cases and unrankable incomplete families.
7. No hidden AMP, accumulation, sampling, alternate runtime or model fallback;
   bounded retention and trustworthy finalization under failures.

This task changes only documents and non-launching JSON. It performs static
configuration/reuse arithmetic checks, not any of those model-forward tests.

## Release gates and current evidence

Accepted archived TRAIN/VALIDATION input audit, CPU bridge and exercised
transform evidence remain accepted within scope. They do not certify original
server bytes, partial oblique edges, CUDA, topology or pretrained integration.
ImageNet V1 bytes were acquired under the earlier authorization; no new download
or load is needed for this plan change. Terms, all-split contamination, exact
secure runtime and semantic mapping/loaded-state evidence remain gates.
RadImageNet stays blocked on its own release/terms/bytes/interface/overlap.

The external supported-runtime review dated2026-10-04 supersedes historical
runtime-selection recommendations. It proposes a new disposable inspection
environment, not a complete approved lock. Preserve its restricted CPU/no-forward
inspection, storage-layout/no-alias checks, JIT/security controls, containment,
aggregate deadlines/storage, teardown and evidence rules. Do not reinstate the
old vulnerable runtime or infer full Docker acceptance from metadata-collector
compatibility. Dependency/metadata work remains separate from research execution.

Before pilots: independent topology/implementation review, exact runtime and
native dependency/security closure, terms/overlap/loading/transform compatibility,
future source bindings, GPU/determinism/resource feasibility, scoped synthetic
acceptance and explicit execution authorization. No B1 run is default: use the
comparison checklist and document a specific scientific exception if necessary.
Matching test cases alone does not prove fairness; old-test results remain
exploratory, and retraining B1 cannot remove exposure. Independent compatible
confirmation may use frozen B1 inference rather than new training.

## Preserved historical evidence chronology

The following 2026-09-27 observations, plans and dated acceptance blocks are
retained verbatim. Their older "next action"/pending statements are historical;
the current gates above and later input/security records govern now. This
amendment does not order those historical commands to be rerun.

## Returned installed-version metadata

On 2026-09-27 the owner supplied the following output from the previously
provided `importlib.metadata` command on the university server. It is an
owner-provided conversation record, without a separately supplied receipt or
independently observed execution. Its exact reported content is:

```json
{
"packages": {
"Pillow": "12.3.0",
"PyYAML": "6.0.3",
"numpy": "2.2.6",
"opencv-python": "5.0.0.93",
"torch": "2.3.0+cu121",
"torchvision": "0.18.0+cu121"
},
"python": "3.10.17"
}
```

The installed-version metadata gap is resolved within that attribution. No
successful imports, binary compatibility, CUDA determinism, transform equivalence
or pretrained-weight compatibility follows from this output. The versions are
observations, not new approved runtime pins. The provenance register records the
same values and retains all initialization releases as false. The command below
is retained for historical provenance; it need not be repeated.

```bash
cd /experiment-storage/chaidir/fawad-dsai/ConsenDxFormer && CUDA_VISIBLE_DEVICES="" /experiment-storage/chaidir/env/env-310-fawad/bin/python -B -c 'import importlib.metadata as m, json, platform; names=("torch","torchvision","numpy","Pillow","opencv-python","PyYAML"); installed={d.metadata["Name"].lower():d.version for d in m.distributions() if d.metadata["Name"]}; print(json.dumps({"python":platform.python_version(),"packages":{n:installed.get(n.lower()) for n in names}},sort_keys=True,indent=2))'
```

## Historical proposed CPU compatibility check

The following plan predates the returned bridge evidence below. Its import and
array-conversion portions have since been reported as passed by the owner.
Its affine API candidate is refined by the explicit contract below.

Review this check before a later server step. Use the same Python executable,
`-B`, `CUDA_VISIBLE_DEVICES=""`, synthetic arrays only and CPU devices explicitly.
Import no project training runner, instantiate no model and request no weights.
Record each step's outcome, traceback/warnings, reported runtime versions and
the checking script's hash. A failure stops the check; retain its partial report.
No automatic repair, package install/downgrade, retry, alternate implementation
or fallback is permitted.

1. Import NumPy, torch, torchvision, PIL, cv2 and yaml in the requested
   environment; compare exposed version strings with package metadata (allowing
   OpenCV's runtime string to omit its wheel packaging suffix, recording both).
   Record missing shared-library/operator errors and ABI warnings. Imports alone
   are not a compatibility result for subsequent steps.
2. Create a contiguous float32 64x64 NumPy ramp and zero/one/checkerboard arrays;
   verify `torch.from_numpy` on CPU and `.numpy()` round trips preserve shape,
   dtype and values. Explicitly test shared-storage mutation on a disposable
   array and copy isolation on a separate copy. Fail on conversion warnings,
   unexpected copies where aliasing is required, or ABI/runtime errors. NumPy
   2.2.6 with the reported torch build must be tested rather than assumed valid.
3. Exercise synthetic uint8 and float32 grayscale conversion and the documented
   divide-by-255-if-max>1 rule without image files. For CPU float32 NCHW arrays,
   call `torch.nn.functional.interpolate` to 224x224 with mode bilinear,
   `align_corners=False`, `antialias=True`. Check shape, dtype, device, finite
   range, constant preservation (atol 1e-6, rtol 0), and deterministic repeat.
4. Check exact horizontal/vertical flips. As a proposed API candidate only,
   exercise torchvision tensor `functional.affine` with bilinear interpolation,
   scale 1, shear 0, fill 0 and explicit synthetic angles/translations; check
   identity and integer-translation behavior on an asymmetric impulse/grid,
   zero-fill edges, and deterministic repeats. Pin center and fractional-to-pixel
   translation rounding in a reviewed fixture before adopting this candidate.
   Successful operation does not resolve that remaining semantic decision.
5. Clip to [0,1], replicate three equal channels and normalize with the accepted
   mean/std; compare elementwise to independent NumPy arithmetic at atol 1e-6,
   rtol 0. Require finite CPU float32 [1,3,224,224]. Check validation has no random
   geometry and train zero fill occurs before normalization.

Passing establishes only the exercised synthetic CPU operations in that
environment. CUDA determinism, research image decoding, exact transform
equivalence, model topology and pretrained compatibility remain separate gates.
The next action is review of this bounded check and the unresolved affine
semantics; no server compatibility execution occurred during this correction.

## Returned CPU imports and NumPy/Torch bridge evidence

On 2026-09-27 the owner supplied server JSON with scope
`cpu_imports_and_numpy_torch_bridge_only`, status `passed`, no warnings,
`model_forwards=0`, `training_performed=false`, `transforms_checked=false` and
`pilot_execution_authorized=false`. It records all six imports (numpy, torch,
torchvision, PIL, cv2, yaml), round trips for zeros/ones/ramp/checkerboard, and
shared-storage/copy-isolation checks as passed. Python is 3.10.17; runtime
versions are numpy 2.2.6, torch 2.3.0+cu121, torchvision 0.18.0+cu121,
PIL 12.3.0, cv2 5.0.0 and yaml 6.0.3. OpenCV's runtime version and previously
reported wheel version 5.0.0.93 are recorded distinctly.

The complete JSON values are retained in `cpu_bridge_evidence.reported_output`
in the provenance register and in the external review's
`owner-cpu-bridge-output.json`. This is owner-provided conversation evidence;
no separate receipt or checking-script hash was supplied and execution was not
independently observed. It resolves the reported imports and exercised CPU
array-bridge checks only. It establishes neither transforms nor general ABI,
CUDA/determinism, pretrained compatibility or Stage A equivalence. Prior pending
statements remain historical within this document.

## Explicit proposed v2 affine contract and synthetic probe

The proposed contract uses `torchvision.transforms.functional.affine` on CPU
float32 tensors with explicit BILINEAR interpolation, scale=1, shear=[0,0],
fill=0 and center=[112,112] for 224x224 images. Coordinates originate at the
upper-left image edge; pixel centers are (column+.5,row+.5). Positive angles
rotate clockwise. Translation is [horizontal,vertical] and is applied after
rotation, as specified by the
[torchvision 0.18 affine API](https://docs.pytorch.org/vision/0.18/generated/torchvision.transforms.functional.affine.html).
Each sampled fraction becomes `int(round(fraction * axis_length))`: Python
rounding is ties-to-even on the floating-point product. For example .5, 1.5,
2.5 pixel products become 0,2,2 (with corresponding negative values 0,-2,-2).
The accepted angle/fraction distributions remain [-10,10] degrees and [-.08,.08]
per axis. A 90-degree synthetic fixture tests direction/center and is not a new
training augmentation. Antialias=True applies to bilinear resize only; affine
does not expose that option. These are proposed v2 semantics, not verified
equivalence to Stage A.

The standalone [CPU transform probe](../scripts/54_probe_performance_v2_transforms.py)
accepts only a new output directory and constructs invented arrays internally.
It imports no repository runtime or model. The server command hides CUDA before
startup, uses the explicit Python executable and verifies the script hash.
`report.json` is flushed after each step using atomic replacement within a newly
created directory; existing output directories are rejected. Failures/warnings
stop processing with a nonzero exit and preserve prior passed steps, versions,
script hash, warning records and traceback. An uncatchable termination leaves
the last persisted report as running, never passed. No repairs or fallback exist.

Independent fixture expectations are:

- uint8/float32 grayscale scaling from explicit values, with invalid shapes,
  NaN/Inf and out-of-range inputs rejected; no research image reads;
- pixel-center bilinear interpolation with clamped edges computed in NumPy for
  64-to-224 upsampling, plus constant preservation; no downsampling claim;
- exact array-index reversals for flips;
- explicit forward integer-coordinate maps for identity, positive/negative
  shifts, clockwise/counterclockwise quarter-turns and rotation followed by
  translation, including lost border pixels and zero fill;
- for +/-10 degrees, an independently inverted pixel-center matrix applied to
  an analytical linear field, tested on fully supported interior pixels and
  fully outside zero-fill pixels. Partially supported oblique boundary pixels
  are not numerically certified by this minimal fixture;
- independent clipping, channel replication and NumPy normalization, including
  a composed training fixture that verifies zero fill precedes normalization.
  The validation fixture omits geometric augmentation.

All comparisons use rtol=0. Atol is 1e-6 for scaling/constants/normalization,
1e-5 for general resize, 5e-5 for affine, zero for flips and exact repetition.
Composed normalized validation/train tolerances are 5e-5/3e-4 respectively,
allowing interpolation error magnification by normalization; they are declared
before execution and recorded in the report. Repetition checks supplement,
not replace, independent geometric oracles. Outputs must be finite CPU float32
with the expected shape, finally [1,3,224,224].

The external validation report distinguishes local oracle/reporting tests and
any local probe failure from the still-pending university transform run. The next
action is owner execution of the supplied standalone command and return of its
report, including failure output. Passing releases only these synthetic CPU
operations; pilot authorization stays false and all weight/provenance gates remain.

## Accepted bounded synthetic CPU transform evidence — 2026-09-27

The owner supplied the university `report.json`, SHA-256
`3e79f956d99d39e79cd81dd218397f66f8bc4faa71ce07ccb62220ed3a69c894`,
bound to unchanged probe SHA-256
`3203c0d99072e2bc03574b67e1ad47a8ea61bf5dbef3ee8572594274897efab6`.
Read-only verification confirmed the exact contract, all 24 ordered passed steps,
17 numerical checks at their declared tolerances, zero warnings, null failure
fields, CPU float32 outputs and the scope flags. Python 3.10.17, NumPy 2.2.6,
torch 2.3.0+cu121 and torchvision 0.18.0+cu121 match the reported environment.
The probe was not rerun and execution was not independently observed.

Bounded acceptance covers the exercised synthetic CPU operations. Maximum affine
error is 4.57763671875e-5 against atol5e-5; checkerboard resize error is
4.359654020902326e-6 against atol1e-5; composed normalized train-chain error is
8.98526335250116e-5 against atol3e-4. For each oblique fixture, 45,643 interior
and 3,671 exterior pixels were checked. Partially supported oblique edges remain
uncertified. Output digests are report-bound references, not independently
regenerated array bytes. This acceptance establishes no Stage A equivalence,
research-input validity, CUDA determinism or pretrained compatibility.

Earlier pending statements and the desktop missing-torchvision report remain
historical. Next steps are review of the separate
[TRAIN/VALIDATION input audit](performance_v2_input_audit_plan.md) and
[weight acquisition/compatibility plan](performance_v2_pretraining_acquisition_plan.md).
Neither plan authorizes execution or weights acquisition. Stage B remains paused
and `pilot_execution_authorized=false`.
