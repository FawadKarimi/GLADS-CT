# Phase 5 insertion/deletion subset: completed and audited execution

## Current status and evidence attribution

The `insertion_deletion_only` subset is **completed and audited**, according to
the supplied `Phase5_Insertion_Deletion_Execution_Audit_5e31065629b5137c.md`, dated
2026-09-17. Its decision is PASS for bundle integrity, frozen-input consistency,
numerical read-back and subset completion. Execution took place on the university
server at commit `422c8de64cf0e6d9f942a7a556f5473133cc7d8d` on
`phase4-consendxformer`. This documentation update records that audit; it does
not independently reopen research checkpoints, revalidate a GPU, rerun the test
suite or repeat inference or numerical analyses locally.

Successful execution does not establish an overall DSA faithfulness advantage,
classification superiority or SOTA performance. All results are exploratory
post-hoc, with details fixed after test exposure, not preregistered or
confirmatory. Macro-F1 remains the primary classification endpoint.

Only this subset's status is advanced here. All 15 formal hypotheses remain
`unresolved_not_computed`; no p-values, new intervals or formal significance
claims are supplied by this execution. A validated clustered inferential
method/calibration remains unresolved. The full supplement and Stage A remain
incomplete. Other evidence-gap statuses and completed result bundles are
unchanged. This record authorizes no further execution, tuning or method change.

## Audited bundle identity and provenance

The following identities and checks are reported by the supplied independent
audit, not newly verified against local research bundles in this update:

- Research bundle: `phase5-perturbation-5e31065629b5137c`.
- Execution/implementation commit: `422c8de64cf0e6d9f942a7a556f5473133cc7d8d`.
- Uploaded ZIP SHA-256: `c15135ec3576dfaca00f80fdd43b8de15c53a4443b42d792333a392515e8c320`.
- Research artifact manifest SHA-256: `00f559e83a92ea8400e0afd5dd875166ee46e509d0074cccbc156a46a65839b9`.
- Completion schema: `phase5_insertion_deletion_completion_v2`.
- Hash-bound payloads: 32; with manifest and completion marker, 34 files total.
- Matched-attention source: `phase4-evaluation-22cb656eabd040c2`.
- Matched-attention manifest SHA-256: `daaedff0f11f642318d04ad23536e522e7d50f9a277fe50087c8cc7bc750b07b`.
- Approved embedded timing bundle: `phase5-timing-030925dee6228ebb`.
- Approved timing manifest SHA-256: `688ab84bbc7ea90c4154954daa5dcbae6d40f9e2e70c63820455fe151693aa6d`.
- Frozen proposal LF SHA-256: `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`.

The audit confirmed both research hashes against their pre-upload values, the
exact expected archive membership, all payload/NPZ-array hashes, and absence of
failure markers, duplicate members or unexpected payloads. It also checked the
original matched-attention archive's pinned manifest and every listed member.
The six source records, including selected checkpoint hashes, match the earlier
`phase5-descriptive-04d6daaccc698431` records. All 1,331 shared source-snapshot
entries match; the two scopes have different complete snapshot sets, so this
does not assert equal whole-snapshot hashes.

## Completed execution and population accounting

All 183 test nodules from 99 patients were retained for the complete
ConsenDxFormer and B1, each at seeds 42/43/44: six frozen model instances and
1,098 model-seed-nodule units, not 1,098 independent nodules or patients.

The ledger reports identical attempted and accepted counts at every stage:

- Original-input replays: 1,098 attempted / 1,098 accepted.
- Zero-baseline calls: 1,098 attempted / 1,098 accepted.
- Interior perturbations: 415,044 attempted / 415,044 accepted.
- Total: **417,240 attempted / 417,240 accepted**, zero automatic retries and
  no failure.

The passed global replay barrier binds all original outputs to sources, units
and orders. It records zero baseline/interior calls at release: all 1,098
original replays preceded perturbation calls. The final recorded call is B1
seed 44, unit 1097, hash-order deletion step 63, call 417,240.

There are 6,588 complete curves, each with 65 positions, giving 428,220 exported
positions. Their difference from the forward count is exactly endpoint reuse
within each unit. Original/baseline endpoints reuse logits, probabilities, input
hashes and call references exactly; every unique saved call index from 1 through
417,240 is present. No interior position was skipped or additionally
deduplicated. Zero-baseline outputs are identical across nodules within each
frozen model/seed, as expected for the same input.

The eight exported table row counts are:

- `hypothesis_register.csv`: 15.
- `perturbation_auc_per_nodule.csv`: 6,588.
- `perturbation_curves.csv`: 428,220.
- `perturbation_mean_curves.csv`: 2,340.
- `perturbation_paired_descriptive.csv`: 24.
- `perturbation_paired_per_nodule.csv`: 4,392.
- `perturbation_per_seed.csv`: 36.
- `units.csv`: 1,098.

## Descriptive contrasts and interpretation

These AUCs integrate the **fixed original predicted-class probability** over the
fraction of changed tiles. They are not ROC-AUC, accuracy or macro-F1; incorrect
original predictions still supply their saved predicted-class targets. Higher
insertion AUC and lower deletion AUC are favorable under this protocol.

Every contrast below is the mean paired **DSA minus control** difference, within
one model/seed, with arithmetic nodule weighting across all 183 nodules. For
insertion, positive favors DSA; for deletion, negative favors DSA. Values are
copied from the audit at nine decimal places; original CSVs retain their exported
precision. The source medians and linear quartiles were also audited, not
recomputed for this documentation update.

- **ConsenDxFormer, seed 42:** DSA-prior insertion **+0.000021347**, deletion
  **-0.005353607**; DSA-hash insertion **+0.164445566**, deletion **+0.042653050**.
- **ConsenDxFormer, seed 43:** DSA-prior insertion **+0.007041130**, deletion
  **+0.000451330**; DSA-hash insertion **+0.131802280**, deletion **+0.040973323**.
- **ConsenDxFormer, seed 44:** DSA-prior insertion **+0.000359901**, deletion
  **-0.005008835**; DSA-hash insertion **+0.182032078**, deletion **+0.056663770**.
- **B1, seed 42:** DSA-prior insertion **-0.146782393**, deletion
  **+0.094566542**; DSA-hash insertion **+0.047532486**, deletion **+0.172804288**.
- **B1, seed 43:** DSA-prior insertion **-0.137887204**, deletion
  **+0.094917030**; DSA-hash insertion **+0.029754838**, deletion **+0.155025362**.
- **B1, seed 44:** DSA-prior insertion **-0.140339399**, deletion
  **+0.126334200**; DSA-hash insertion **+0.053535279**, deletion **+0.205542943**.

ConsenDxFormer DSA is close to its fixed location prior, with small, mixed
differences: insertion is slightly higher at all three seeds, while deletion
favors DSA at seeds 42/44 and slightly favors the prior at seed 43. This does not
establish superiority, equivalence or an absence of all useful information in
DSA. Its insertion advantage over hash-order is accompanied by higher, **less
favorable deletion** AUC at every seed; there is no consistent advantage across
both directions. Reporting insertion alone would conceal this limitation.

B1's DSA is less favorable than its own fixed prior on mean AUC in **both
directions at all three seeds**. Against hash-order, B1 likewise has higher
insertion and higher, less favorable deletion AUC. These are within-model
observations, not a formal model comparison.

No cross-model faithfulness, classification-superiority or SOTA conclusion is
supported. The audit reports that the two models' fixed targets differ on
27/32/30 of the 183 nodules at seeds 42/43/44, respectively, and their zero-input
probability distributions differ. There is no authorized cross-model AUC
endpoint or hypothesis test. This experiment does not isolate the contribution
of attention-alignment loss, establish clinical localization validity or explain
why the perturbation directions disagree.

All 6,588 curves are flagged nonmonotonic by the exact rule requiring at least
one positive and one negative adjacent change, with no magnitude threshold.
None is flat; none of the 1,098 unit inputs is constant or identical to the zero
baseline. All curves remain included without smoothing, truncation or forced
monotonicity. These flags are not significance findings.

Zero-valued tiles in normalized coordinates are not physically defined air and
may be out of distribution. Controls disrupt content in different spatial ways;
these interventions do not establish causal faithfulness. The seeds reuse the
same 99 patients: seeds, nodules within patients and curve positions are not
independent replications. Quartiles describe distributions, not confidence
intervals. Findings remain descriptive, within-model and exploratory post-hoc.
Do not use these exposed test results to tune the frozen method, baseline,
ordering, threshold, seed or model. Macro-F1 remains primary.

## Independent audit coverage and limits

The supplied audit executed the exact production
`validate_completed_perturbation` and embedded timing-validator bodies offline,
using NumPy/pandas with heavy model-runtime imports omitted through AST
extraction. All 18 source files supplying imported modules or extracted helpers
matched their evidence-bound LF-normalized hashes. No validator condition or
numerical tolerance was relaxed.

Its production read-back checked all eight CSVs against NPZ evidence, with exact
raw curve values and `atol=1e-12, rtol=0` for derived tables. An independent
weighted-sum calculation checked every AUC and paired difference, then every
per-seed mean, median and linear quartile. Against the original matched-attention
archive, the audit checked all test identities, original image/DSA/prior hashes,
fixed targets, DSA descending-value/ascending-index tie orders, the unchanged
training-only prior ranking, and hash-control permutations/unsigned digest bytes.
All 428,220 perturbed-input hashes matched independent reconstruction from the
original float32 images, orders, direction and step, with zero model calls.

Replay-cache outputs independently agreed with the original archived outputs
under unchanged tolerances, preserving all targets. Maximum absolute archive
differences were `2.250075340270996e-06` for logits and
`7.152557373046875e-07` for probabilities. Separately, the stored replay reports
record maxima against the original saved prediction CSVs of
`2.2649765014648438e-06` and `7.889162242369352e-07`, respectively. Those CSVs and
checkpoint binaries were not supplied locally to the audit: the latter comparison
is recorded evidence, not the independently reproduced archive comparison.
All raw curve outputs remained finite float32; maximum absolute probability
row-sum deviation was `1.4419492799788713e-07`, below the frozen `1e-6` limit,
without clipping or normalization.

The approved embedded timing evidence, source bindings, recorded GPU UUID/PCI,
runtime compatibility, T1 launch-mode binding and T2 nanosecond accounting passed
offline validation. That is not a fresh GPU observation. Live server-source
preservation and frozen-weight checks were reported PASS by the supplied server
runner/verification; neither the independent audit nor this documentation update
independently rereads that live filesystem or reopens trained checkpoint binaries.

The independent audit performed zero model forwards, no training, inference
rerun, new hypothesis tests, intervals or calibration campaign. Its numerical
recomputation verified existing exports only. It did not rerun the full repository
test suite or modify original uploads, frozen files or result bundles. This
documentation-only update does not reproduce those audit calculations or rerun
tests. Integrity and recorded provenance do not constitute independent observation
of GPU execution.

## Observed runtime and approved timing evidence

Reported curve-execution wall time was **2,147.845612185076 seconds =
35.797426870 minutes**, approximately 35 minutes 48 seconds. It includes original
replays, input construction, model calls, validation and replay-cache barrier I/O.
It excludes preparation/loading and final result export/read-back: it is **not
end-to-end runtime**.

The approved embedded synthetic adapter-only estimate was
1,902.490166971094 seconds (31.708169450 minutes). The reported broader-scope
observed interval is 245.355445213982 seconds longer. That difference is not a
directly measured overhead breakdown or a like-for-like timing error: scopes,
inputs and weights differ. Earlier synthetic memory evidence is not a full
research-memory measurement. The frozen protocol's historical timing-null field
remains unchanged; the completed evidence above supersedes neither that historical
record nor the original methodology.

## Historical implementation and timing-proposal record

The following records the earlier implementation milestone and its execution
contract. Statements about authorization or unmeasured timing describe that
milestone, not the completed execution above. Operational guards remain applicable
to any separately authorized future work; this record is not an instruction to
rerun the completed subset.

Scope is `insertion_deletion_only`, implementing section 3 and applicable source
contracts of the unchanged `phase5_supplemental_evaluation_proposal.md` (LF
SHA-256 `4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620`).
That milestone authorized implementation and synthetic CPU tests only, not
research execution, server work or publication. The separate
versioned specification is `configs/phase5_insertion_deletion_v1.json`; existing
protocols, architecture, selected checkpoints and completed bundles are unchanged.

The proposed analyses were exploratory post-hoc, fixed after test exposure,
not preregistered or confirmatory. Macro-F1 remains primary. This subset computes
no pointing, accuracy intervals, boundary, uncertainty, patient-weighted
correctness, calibration or hypothesis tests. All 15 formal hypothesis IDs remain
`unresolved_not_computed`, with no p-values, placeholders or significance labels.
The full supplement and Stage A remain incomplete.

### Historical execution gate and protected-source contract

`scripts/49_run_phase5_insertion_deletion.py` is a separate opt-in entry point.
It rejects missing `--authorize-execution` before source loading/model preparation.
The reviewed execution gate requires a clean committed `phase4-consendxformer` checkout,
the exact new protocol and unchanged reviewed proposal, normal dependencies and
the actual pinned external B1 source. At that milestone, code review and later
timing review were separate prerequisites, not execution authorization. The
completed execution and its approved timing evidence are recorded above.

Historical opt-in options (contract reference, not executed by this update):

- `--full-source DIR` three times, exactly one complete source at each seed.
- `--b1-source DIR` three times, exactly one B1 source at each seed.
- `--attention-source DIR` for `phase4-evaluation-22cb656eabd040c2` and
  `--attention-manifest-sha256 HASH` for its explicitly approved manifest.
- `--authorize-execution`, after separate authorization.
- `--timing-bundle DIR` and `--timing-manifest-sha256 HASH` for separately reviewed
  synthetic timing evidence; completion, counts, integrity and architecture/runtime/
  device compatibility are checked before research source/data/checkpoint loading.
- `--device cpu` (default) or explicitly authorized `cuda`. Deterministic CUDA
  support now requires a full `--gpu-uuid`, fresh idle-device verification and
  exact-UUID isolation before importing PyTorch, with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` before interpreter startup. Optional
  `--torch-threads N` must match the timing environment; unsupported
  deterministic operations fail, without tolerance changes or fallbacks.
- Optional `--output-root DIR`, contained in the new namespace below.

There is no discovery, alternate checkpoint, source allowlist extension, reduced
cohort, ablation model or research retry option. The existing distinct complete/B1
loaders validate all six saved sources and exact frozen identities. They may
check saved metrics as part of their unchanged source-validity contracts; no
earlier evaluation routine or result bundle is rerun. The matched
archive's completion, checksums, array metadata, interpretation permission and
all six source bindings must agree, including original saved prediction replay.
The existing prior is checked against the same 807 training identities, disjoint
from test patients; it is never refitted. All 183 nodules from 99 patients remain.

Only persisted validation-selected best checkpoints are used, through the existing
complete-model loader and actual pinned B1 constructor/checkpoint loader. Before
any forward, existing input-lock/readiness guards are revalidated and unaugmented
live CT/mask/identity batches are compared exactly with the archived inputs.
Masks are validation evidence only; neither the curve engine nor any predictor
accepts lesion masks. Source/input hashes and in-memory frozen model state are
checked again before completion. Test stubs do not weaken these production guards.

### Preserved numerical and scheduling contract

Images are unchanged float32 `[183,1,64,64]` values in `[0,1]`; maps are the saved
native `[183,1,8,8]` DSA and the one `[8,8]` training-only prior. Both directions
use the float32 zero baseline in normalized input coordinates, with no blur,
baseline fitting, saliency generation, map interpolation or target-mask changes.

Each native cell controls its entire nonoverlapping 8-by-8 input tile. Every
ordering includes all 64 cells. DSA/prior orders sort raw values descending,
then flat index ascending for exact ties. The non-saliency order sorts SHA-256
digest bytes of the exact proposal JSON, then flat index. The stored digests
are indexed by original flat cell index; permutations give the removal/addition
order. Controls are identical across models/seeds for a nodule. Rankings stay
fixed from unperturbed saved evidence, regardless of any perturbed attention.

All 1,098 unperturbed batch-one replays across both models and seeds 42/43/44
must satisfy the existing per-class `1e-5 + 1e-4*abs(saved)` logit/probability
tolerance and saved-class argmax agreement. A late failure prohibits every
baseline/interior forward. The global endpoint cache is sealed to the complete
source/checkpoint/input identity, units, orders and raw output hashes and read
back before the barrier is granted. The target remains the saved predicted class,
including originally wrong predictions, not the true class or a changing argmax.

Every fresh probability output must be nonempty float32 `[N,3]`, finite, in
`[0,1]`. A private float64 copy is summed with `atol=1e-6, rtol=0`. The raw values
are not clipped, normalized or passed through the stricter shared metric guard.
Logits must be finite float32 as emitted by the frozen no-autocast runtime.
Raw probabilities, row sums and signed/absolute deviations are exported. Float64
promotion for CSV precision and trapezoidal AUC arithmetic changes no raw values.

There are 65 positions at `k/64` per curve. No curve is truncated, made monotone
or normalized by its endpoints. Cache only each unit's original and baseline
outputs, shared by its six curves. Every interior position is evaluated, even
for zero images or identical orderings; no cross-unit/order deduplication occurs.

- 1,098 original replays, 1,098 baseline calls and 415,044 interior calls.
- Exactly 417,240 batch-one forwards, 139,080 per trained seed, with no retries.
- 6,588 curves and 428,220 exported positions, including cached endpoints.
- 36 model/seed/order/direction summaries and 24 paired within-model summaries.

The implementation schedules model, seed, then archive row for the global replay;
after the barrier, unit, ordering, direction, then interior step. All six frozen
models are retained for this schedule. Preparation/loading makes no extra forward.
Actual attempted/accepted counts and failed attempts are recorded, not inferred
solely from the planned budget.

Per-seed summaries report arithmetic nodule means, median and linear quartiles
of AUC across all 183 nodules, and the 65 mean-curve positions. Paired differences
are DSA-minus-prior and DSA-minus-hash-order within each model/seed/direction.
Constant/baseline-identical images, flat/equal-order and nonmonotonic curves are
retained and flagged. There is no cross-model AUC hypothesis or patient/seed
independence claim.

### Exports, verification and failure semantics

New outputs use
`artifacts/phase5/insertion_deletion/phase5-perturbation-<identity hash>/`.
Identity distinguishes the proposal's base and publication commits from the
clean implementation/execution checkout commit (the latter two are equal for
this runner). It binds protocol/proposal, exact sources and
snapshot hashes, attention manifest, runtime/device, approved timing evidence and
frozen-loading provenance.
Reject existing complete or partial paths, source/data overlaps, escaped or
symlink paths and completed ancestors. Timing uses its own separate protected
namespace; neither namespace changes earlier result bundles.

The bundle includes:

- `protocol.json`, `provenance/reviewed_proposal.md`, `evaluation.json`,
  `summary.md` and `provenance/provenance.json`.
- `tables/units.csv`: exact identity/order, fixed target, original image, DSA,
  prior, baseline, source-record and best-checkpoint hashes; constant/zero flags.
  Curve rows reference this table's unique `unit_id` for their immutable source
  and map provenance rather than copying or overwriting original maps/CT tensors.
- `orders.npz`: all permutations and all unsigned hash-control digest bytes.
- `replay_cache.npz`, `replay_barrier.json`, `replay_checks.json`:
  globally validated original outputs, source/cache binding and all discrepancies.
- `baseline_cache.npz`, `curves.npz`: exact float32 logits/probabilities,
  scheduled call indices and hashes of actual forward inputs. Only endpoints
  reuse call indices; interior calls are individually accounted for.
- `tables/perturbation_curves.csv`: identities, fixed target, order, direction,
  step/fraction/changed tile, raw three-class logits/probabilities, current
  predicted class, fixed-target response, mass diagnostics and endpoint-cache
  references. Step zero uses changed-tile index `-1` (no change yet).
- `tables/perturbation_auc_per_nodule.csv`, `perturbation_per_seed.csv`,
  `perturbation_paired_per_nodule.csv`, `perturbation_paired_descriptive.csv`,
  `perturbation_mean_curves.csv`, and the unresolved `hypothesis_register.csv`.
- `call_ledger.json`, an artifact manifest and final `COMPLETED.json`.
- For newly executed v2 bundles, the entire verified timing bundle under
  `provenance/timing/`, its original source snapshot and identity/hash binding.
  Offline validation checks this copy without live-source/model access. Historical
  v1 completion remains read-only; it is not used for new executions.

NPZs use the existing deterministic ZIP/array hash machinery and forbid object
arrays/pickle. CSV numbers round-trip from promoted raw values. Before completion,
inference-free read-back checks every array hash, all steps, fixed targets,
endpoint reuse, call indices, cohorts, controls and exported tables. AUC/summary
replay uses float64 `atol=1e-12, rtol=0`; raw curve values are exact. Completion
binds every payload and certifies only `insertion_deletion_only`. It cannot close
the formal hypothesis register, the full supplement or Stage A.

Failures, including catchable Python interruptions (`KeyboardInterrupt`,
`SystemExit` and `GeneratorExit`), retain the new partial directory and attempt
to persist `FAILED.json` with actual attempted/accepted counts, current
stage/unit/step context and replay diagnostics, then re-raise the original
exception. An interrupted forward is not accepted as a validated output; no
further forwards or automatic retries/resume occur. Only the directory created
by this attempt is eligible for failure recording and completion-marker removal;
existing attempts and previous research successes/failures are never edited.

Failure recording and removal of this attempt's completion marker are independent
best-effort actions, including when final validation is interrupted. Their errors
must not replace the original exception. A recorded failed attempt is rejected
by completion validation even if the filesystem prevents marker removal. No
overwriting or reuse of partial/failed output directories is allowed. SIGKILL,
power loss, repeated interruption during cleanup or an unwritable filesystem
cannot guarantee persisted failure evidence or marker removal; a leftover marker
under those conditions is not proof of successful execution and requires review.

### Historical synthetic validation and proposed timing procedure

Implementation tests used invented images/identities and tiny deterministic
image-only predictors, not research model checkpoints. They exercise hand AUCs, mapping,
exact ties, independently generated hash-order golden vectors, changing predicted
classes, raw float32 edge cases, all-zero/nonmonotonic inputs, fixed rankings,
global late-replay failures, source corruption/mutation, deterministic exports,
complete call/cache accounting and subset-only completion. A full synthetic
183-nodule schedule exercises the 417,240 calls; it is not a research result or
a timing measurement. Historical source-loader fixtures are explicitly stubbed;
the existing source-contract tests remain part of the full CPU suite.

At the historical implementation milestone, research runtime and feasibility were
unmeasured. The following original procedure motivated the separate implementation
now documented in `phase5_synthetic_timing.md`; neither milestone itself executed
an actual timing benchmark. New runtime evidence is distinguished from the frozen
protocol's unchanged historical null. After separately authorized execution, v2
provenance reports synthetic measurements/adapter-only estimates separately from
observed research curve-execution wall time (not end-to-end runtime). The historical
procedure below is retained, not a benchmark result or execution authorization:

1. On the intended device, record hardware, package/source hashes, shapes,
   thread settings, memory and deterministic/no-autocast policy. Use the actual
   frozen architectures/configurations but synthetic weights/inputs only, with
   no research checkpoint, prediction, label or mask access. Identify clearly
   that synthetic weights do not validate research replay or accuracy.
2. For each architecture and seed 42/43/44, initialize synthetic weights once
   with that seed before timing. Use batch-one float32 `[1,1,64,64]` synthetic
   zero, linear-ramp and `Generator(PCG64(20260917))` uniform random images in
   `[0,1]`, plus the same tile-construction path. Cycle zero/ramp/random in that
   order; use identical input sequences across the six synthetic models. Fix
   this sequence before timing. Propose 32 warm-up
   calls and three measured blocks of 128 calls per model/seed: 2,496 calls
   including warm-up across six synthetic models, separate from the research
   budget. Report all blocks, not a fastest selected block. Review these timing
   choices before using them; they do not modify the evaluation protocol.
3. Synchronize CUDA around timed blocks where applicable; report warm-up and
   measured-call counts separately, block seconds, time per call and peak
   memory. Measure input construction, model execution and export/read-back
   overhead separately rather than treating model-only time as end-to-end time.
4. Extrapolate per-model/seed measurements to its fixed 69,540 scheduled calls,
   then sum all six; report the observed block range and representativeness
   limits. Loading, replay validation, disk I/O, six simultaneously resident
   models and synthetic-versus-research differences limit that extrapolation.
   No wall-clock feasibility or performance claim exists until this is reviewed
   and measured. Any later batching/caching optimization requires separate
   numerical validation and review; never alter the fixed scientific schedule.

Zero baseline is an input-space intervention, not a physically exact air image.
Black tiles can be out of distribution. High insertion/low deletion AUC describes
response of a fixed predicted class only, not clinical causality, causal
faithfulness, attention-loss contribution or diagnostic superiority. Complete
and B1 can target different classes. Do not choose a replacement model, baseline,
ordering, threshold or seed from perturbation results.
